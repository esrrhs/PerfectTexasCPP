#pragma once

// 离线 CFR blueprint：表格 external-sampling MCCFR（Regret Matching+ + 线性加权平均）。
// 设计口径见 docs/CFR_BLUEPRINT_PLAN.md。
//
// 三块：
//   1) RegretTable：开放寻址 key->idx + 稠密 regret/strategy-sum 数组。
//   2) mcfrTraverse<Game>：游戏无关的 external-sampling MCCFR 核心（Kuhn 单测验证正确性）。
//   3) HUNLGame：把 HUState + 牌面/动作抽象包成 Game 接口。
//
// 抽象（计划书 §1）：
//   牌面：翻前 169 类全分辨率；翻后 StrengthModel::postflopBucket(B=8)。
//   动作：fold / check-call / raise 0.5pot / raise 1.0pot / all-in（每街加注上限 R=3）。
//   信息集 key：当前街桶 + 量化底池 + 当前街抽象动作序列（跨街 imperfect recall）。

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "texas/hu.hpp"
#include "texas/strength.hpp"

namespace texas {

// 抽象动作 id（固定顺序，regret 按此索引）。
enum AbstractId : uint8_t {
    AB_FOLD = 0,
    AB_CHECK_CALL = 1,
    AB_RAISE_05 = 2,
    AB_RAISE_10 = 3,
    AB_ALLIN = 4,
};
constexpr int kAbstractStride = 5;   // HUNL 抽象动作槽位数
constexpr int kMaxGameStride = 8;    // 核心模板支持的最大动作数（Kuhn=2）
constexpr int kDefaultBuckets = 8;   // 翻后每街桶数 B
constexpr int kDefaultRaiseCap = 3;  // 每街加注上限 R（含 all-in）
constexpr int kMaxStreetSeq = 8;     // 当前街抽象序列最大长度

struct AbstractMove {
    int id = AB_FOLD;
    Move mv;
};

// 枚举当前局面的合法抽象动作（只依赖下注字段，与牌无关）。返回条数。
int legalAbstract(const HUState& s, int raiseCap, AbstractMove* out);

// 牌面桶：翻前 169 类，翻后 postflopBucket。
int cardBucketOf(const HUState& s, int buckets);

// 量化底池桶：floor(log2(pot/150)*4)，clamp [0,31]。
int potBucketOf(int pot);

// 打包信息集 key（唯一，无碰撞）。seq 是当前街抽象动作序列。
//
// seat 必须入 key：座位固定但 SB 翻前先动、翻后后动，两个座位在同一
// (街, 底池桶, 牌桶, 街内序列) 下需要完全不同的策略。共享 key 会让两个座位的
// regret 交替写同一张表、互相抵消，实测策略停在四不像且 BR 不随迭代下降。
// 默认 -1 只用于早期单测的 key 唯一性检查；训练/查表/BR 必须传真实 toAct。
//
// hist 是「每街 2 bit 的压缩加注历史」（bit 2k..2k+1 = 第 k 街加注档 0..3），
// recall=1 时入 key，把跨街 imperfect recall 升到「压缩完美召回」。0 = 旧口径。
//
// slim=1 时把 potBucket 与 hist 都置零（R2 瘦 key 对照）：key 只剩
// street + 牌桶 + 当前街序列 + seat，用来验证「把底池/跨街历史移出表」的取舍。
uint64_t packInfoset(int street, int pot, int cardBucket, const uint8_t* seq, int seqLen,
                     int seat = -1, int hist = 0, int slim = 0);

// 从 HUState.hist 重建当前街抽象动作序列（蒸馏/部署查表用）。返回长度。
// histOut 非空时同时输出压缩加注历史（与训练侧 HUNLGame 同口径）。
int seqFromState(const HUState& s, int raiseCap, uint8_t* out, uint8_t* histOut = nullptr);

// ---------------------------------------------------------------- RegretTable

class RegretTable {
public:
    explicit RegretTable(int stride = kAbstractStride);

    int stride() const { return stride_; }
    uint32_t size() const { return ninfosets_; }

    // 返回 key 对应的稠密 idx（不存在则插入）。
    uint32_t intern(uint64_t key);
    // 只查不插；不存在返回 -1。
    int64_t find(uint64_t key) const;

    // RM+ 当前策略：正遗憾在合法动作上归一，全零则均匀。out 长度 stride。
    void strategy(uint32_t idx, const bool* legal, float* out) const;
    // 平均策略（stratSum 归一）。
    void averageStrategy(uint32_t idx, const bool* legal, float* out) const;

    void addRegretClamped(uint32_t idx, int a, double delta);
    void addStrategy(uint32_t idx, int a, double weight);

    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    void grow();
    uint64_t hashKey(uint64_t k) const;

    int stride_;
    uint32_t ninfosets_ = 0;
    uint32_t capacity_ = 0;       // 哈希槽数（2 的幂）
    uint32_t mask_ = 0;
    std::vector<uint64_t> keys_;  // UINT64_MAX = 空槽
    std::vector<uint32_t> vals_;
    std::vector<float> regret_;   // capacity_ * stride_
    std::vector<float> strat_;    // capacity_ * stride_
};

// ---------------------------------------------------------------- MCCFR 核心
//
// Game 接口：
//   void deal(Rng&)            采样一个新的初始机会 outcome
//   bool terminal() const
//   double payoff(int p) const 终局 p 的收益（BB）
//   int currentPlayer() const
//   int actionStride() const
//   void legalMask(bool* m) const   长度 stride
//   uint64_t infosetKey() const
//   void apply(int id)              执行抽象动作（内部用 rng 采样机会）
//   Game clone() const
//   Rng& rng()

template <class Game>
double mcfrTraverse(Game& g, int traverser, RegretTable& tab, Rng& rng, double weight) {
    if (g.terminal()) return g.payoff(traverser);
    const int p = g.currentPlayer();
    const int stride = g.actionStride();
    bool legal[kMaxGameStride] = {};
    g.legalMask(legal);
    const uint32_t idx = tab.intern(g.infosetKey());
    float sigma[kMaxGameStride] = {};
    tab.strategy(idx, legal, sigma);

    if (p == traverser) {
        double util[kMaxGameStride] = {};
        double node = 0.0;
        for (int a = 0; a < stride; ++a) {
            if (!legal[a]) continue;
            Game child = g.clone();
            child.apply(a);
            // 遍历者的 branch reach：π_i(h·a) = π_i(h) · σ(h,a)
            util[a] = mcfrTraverse(child, traverser, tab, rng,
                                   weight * static_cast<double>(sigma[a]));
            node += static_cast<double>(sigma[a]) * util[a];
        }
        for (int a = 0; a < stride; ++a) {
            if (!legal[a]) continue;
            tab.addRegretClamped(idx, a, util[a] - node);
            // 平均策略累加在遍历者节点，权重 t·π_i(h)·σ(h,a)
            tab.addStrategy(idx, a, weight * static_cast<double>(sigma[a]));
        }
        return node;
    }

    // 对手节点：只采样一个动作，不累加平均策略（对手 reach 由采样概率抵消）。
    const double u = std::generate_canonical<double, 32>(rng);
    double cum = 0.0;
    int chosen = -1;
    for (int a = 0; a < stride; ++a) {
        if (!legal[a]) continue;
        cum += static_cast<double>(sigma[a]);
        if (u < cum) { chosen = a; break; }
    }
    if (chosen < 0) {
        for (int a = stride - 1; a >= 0; --a) if (legal[a]) { chosen = a; break; }
    }
    Game child = g.clone();
    child.apply(chosen);
    return mcfrTraverse(child, traverser, tab, rng, weight);
}

// 训练 iters 次；每次重新 deal，traverser 交替。weight 用迭代号（线性加权）。
//
// weight 的语义是 t × π_i(h)：t 是 Pluribus 同款 linear weighting，π_i 是遍历者
// 沿采样路径到达 h 的自身策略概率乘积。外部采样下对手/机会动作被采样，其 reach
// 由采样概率自动加权，所以 regret 更新不需要显式乘 π_{-i}；但平均策略必须在
// **遍历者节点**按 t·π_i·σ 累加，否则深层 infoset 的策略会被严重过加权。
template <class Game>
void mcfrTrain(Game& root, RegretTable& tab, uint64_t iters, Rng& rng, uint64_t iterOffset = 0) {
    for (uint64_t t = 1; t <= iters; ++t) {
        root.deal(rng);
        const int traverser = static_cast<int>((t + iterOffset) & 1);
        mcfrTraverse(root, traverser, tab, rng, static_cast<double>(t + iterOffset));
    }
}

// ---------------------------------------------------------------- HUNL 适配器

struct HUNLGame {
    HUState state;
    Rng* rng = nullptr;
    int buckets = kDefaultBuckets;
    int raiseCap = kDefaultRaiseCap;
    int recall = 0;  // 1 = 压缩完美召回（每街 2 bit 加注历史入 key）
    int slim = 0;    // 1 = 瘦 key（底池桶与跨街历史都不入 key，R2 对照）
    StrengthModel* sm = nullptr;
    int street = ST_PREFLOP;
    int seqLen = 0;
    uint8_t seq[kMaxStreetSeq] = {};
    uint8_t histBits = 0;   // 已完成街道的加注档（2 bit/街）
    uint8_t streetRaises = 0;  // 当前街加注次数（含 all-in）

    // 一手牌里底牌/公共牌固定，牌力桶最多只有 2 座位 × 4 街 = 8 个不同取值。
    // postflopBucket 要跑摊牌评估 + 听牌出路 + 秩表二分，若每个遍历节点都算一遍
    // 会成为吞吐瓶颈，所以按 (座位, 街) 缓存在 Game 里，clone 时随状态一起带走。
    mutable int bucketCache[2][4] = {{-1, -1, -1, -1}, {-1, -1, -1, -1}};

    HUNLGame() = default;
    HUNLGame(Rng* r, int B, int cap, StrengthModel* m, int rec = 0, int sl = 0)
        : rng(r), buckets(B), raiseCap(cap), recall(rec), slim(sl), sm(m) {}

    void deal(Rng& r) {
        state = dealNewHand(r);
        street = state.street;
        seqLen = 0;
        histBits = 0;
        streetRaises = 0;
        for (int p = 0; p < 2; ++p)
            for (int st = 0; st < 4; ++st) bucketCache[p][st] = -1;
    }
    bool terminal() const { return state.terminal(); }
    double payoff(int p) const { return static_cast<double>(texas::payoff(state, p)) / 100.0; }
    int currentPlayer() const { return state.toAct; }
    int actionStride() const { return kAbstractStride; }
    Rng& rngRef() { return *rng; }

    void legalMask(bool* m) const {
        for (int i = 0; i < kAbstractStride; ++i) m[i] = false;
        AbstractMove am[kAbstractStride];
        const int n = legalAbstract(state, raiseCap, am);
        for (int i = 0; i < n; ++i) m[am[i].id] = true;
    }

    uint64_t infosetKey() const {
        const int p = state.toAct;
        int cb = bucketCache[p][state.street];
        if (cb < 0) {
            cb = cardBucketOf(state, buckets);
            bucketCache[p][state.street] = cb;
        }
        return packInfoset(state.street, state.pot(), cb, seq, seqLen, state.toAct,
                           recall ? histBits : 0, slim);
    }

    void apply(int id) {
        AbstractMove am[kAbstractStride];
        const int n = legalAbstract(state, raiseCap, am);
        const Move* mv = nullptr;
        for (int i = 0; i < n; ++i) if (am[i].id == id) { mv = &am[i].mv; break; }
        if (!mv) return;  // 非法动作不应被选中
        const int prevStreet = state.street;
        if (id >= AB_RAISE_05) ++streetRaises;
        applyMove(state, *mv, *rng);
        if (state.street != prevStreet) {
            // 街切换：把刚结束街道的加注档（0..3）压进 histBits，再清空本街计数。
            const int lvl = std::min<int>(streetRaises, 3);
            histBits = static_cast<uint8_t>(histBits | (lvl << (2 * prevStreet)));
            streetRaises = 0;
            street = state.street;
            seqLen = 0;
        } else if (seqLen < kMaxStreetSeq) {
            seq[seqLen++] = static_cast<uint8_t>(id);
        }
    }

    HUNLGame clone() const { return *this; }
};

// 查表：对一个独立 HUState 输出 blueprint 平均策略（抽象动作分布，长度 kAbstractStride）。
// 返回 false 表示该 infoset 未被训练过（输出均匀）。
bool queryBlueprint(const HUState& s, const RegretTable& tab, int buckets, int raiseCap,
                    float* out, int recall = 0, int slim = 0);

// 蒸馏槽映射（计划书 §4.2）：把 5 个抽象动作的概率质量投到 19 个 actor 槽。
// fold/check-call/all-in 直接对位；两个 pot 比例加注按其真实金额落入 legalMoves
// 的尺寸带（带内金额任意，槽只表示量级），找不到覆盖带时取最近的带。
// 输出 target[kNumActions]，非法槽为 0、总和为 1。
bool abstractToSlots(const HUState& s, const RegretTable& tab, int buckets, int raiseCap,
                     float* target, int recall = 0, int slim = 0);

// 部署/采样翻译（计划书 §4.3）：按平均策略采样抽象动作，返回对应的真实 Move。
Move sampleBlueprintMove(const HUState& s, const RegretTable& tab, int buckets, int raiseCap,
                         Rng& rng, float temp = 1.0f, int recall = 0, int slim = 0);

}  // namespace texas