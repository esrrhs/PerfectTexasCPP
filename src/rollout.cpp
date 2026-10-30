#include "texas/rollout.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <map>

#include "texas/ppo.hpp"
#include "texas/rivercfr.hpp"
#include "texas/slumbot_range_table.hpp"
#include "texas/strength.hpp"

namespace texas {

// 部署期可选的"经验对手模型"：rangeEq 的似然改用 Slumbot 日志拟合的
// （街 × 动作类型 × 手牌百分位）密度表，而不是我们自己的策略。训练路径
// 默认关闭（训练里对手是自身策略，经验模型只对 Slumbot 有意义）。
static bool gEmpiricalRange = false;

void setEmpiricalRangeModel(bool on) { gEmpiricalRange = on; }

namespace {

enum RoleType : uint8_t {
    ROLE_LATEST = 0,
    ROLE_SNAPSHOT = 1,
    ROLE_RULE = 2,
    ROLE_RANDOM = 3,
    ROLE_FIXED = 4,
    ROLE_PRO = 5,
    ROLE_SEARCH = 6,
    ROLE_SOLID = 8,
    ROLE_FOLDER = 9,
    // 尖自对局：对手座位用**当前策略**按部署温度（默认 0.1）提取动作。
    // 训练行为策略仍是采样（T=1），但对手是部署时的尖版本，逼策略学会
    // 防住"不留送分动作"的精确打法。
    ROLE_LATEST_SHARP = 7,
    // 范围搜索剥削者：座位动作由 policySearchMove 给出（河牌按对手最近动作
    // 的似然做后验加权），其余街仍用最新策略采样。
    ROLE_SEARCH_OPP = 14,
    // 紧全下机器人：只用顶范围推/接全下，教策略"全下=紧范围"。
    ROLE_TIGHT = 10,
    // 陷阱机器人：强牌只跟注/过牌，河牌才发力。
    ROLE_TRAPPER = 11,
    // 小注骚扰机器人：翻后高频 1/3 池下注，惩罚过度弃牌。
    ROLE_SPAMMER = 12,
    // 价值型强对手：翻前紧凶、翻后只用两对+ 建池，惩罚"中等牌建大池"。
    ROLE_SHARK = 13,
    // 过度下注机器人：翻后 1.5~3 倍池超大尺寸开火（含诈唬）、面对下注也
    // 大加注、合理价格宽跟。惩罚"把大加注当坚果"与"超池诈唬打宽跟对手"。
    ROLE_MANIAC = 15,
    // 读范围机器人：跟注/弃牌按"对手范围 = 当前牌面最强 keep% 组件"的条件胜率，
    // 下注用强牌 + 固定频率诈唬。README #107(d)：其余脚本全是范围盲的
    // equityVsRandom，策略因此学成"原始胜率 > 赔率就跟"，在 Slumbot 的价值
    // 范围上变成 22% 高牌摊牌。这个对手提供缺失的"范围压力"。
    ROLE_RANGER = 16,
    // 河牌 CFR 对手：其它街按当前策略采样，河牌用 riverCfrMove 解出的均衡回应
    // （双方范围都从各自的动作似然估计）。README #109(d)：手写范围脚本
    // （ranger 各档 air）当教师都失败，因为它们假设"我们的价值范围 = 顶 K%"，
    // 比真实对手紧；均衡回应不靠阈值，按决策树的 regret matching 定价。
    ROLE_CFR = 17,
};

struct GameCtx {
    HUState st;
    int index = 0;
    uint8_t role[2] = {ROLE_LATEST, ROLE_LATEST};
    uint8_t snapTier[2] = {0, 0};  // 0 = recent, 1 = archive
    int snapIdx[2] = {-1, -1};
    uint8_t extraIdx = 0;          // 本局抽中的额外固定对手编号
    int lastIdx[2] = {-1, -1};
    float phi[2] = {0.0f, 0.0f};  // 上次开天眼势能
    uint8_t eqStamp = 255;        // 胜率缓存对应的公共牌张数
    float eq0 = 0.5f;
    // 某个座位最近一次决策的"动作前状态 + 实际动作槽"。rangeEq 特征与
    // 范围条件化搜索都要用它重建对手视角的观测（换入采样底牌），算策略似然。
    RangeObs lastAct[2];
    bool active = true;
};

// 一条街内底池不会改变胜率（发牌只在街道切换时发生），所以开天眼胜率按
// boardN 缓存。seat 1 的胜率严格等于 1 - seat 0（平局双方各算半胜）。
float equityFor(GameCtx& g, int seat, int iters) {
    if (g.eqStamp != g.st.boardN) {
        g.eq0 = float(perfectEquity(g.st, 0, iters));
        g.eqStamp = g.st.boardN;
    }
    return (seat == 0) ? g.eq0 : 1.0f - g.eq0;
}

// 开天眼势能 Φ = (胜率 × 底池 − 自己已投入) / 起始筹码。
//
// 这是"从此刻起双方都不再下注时的期望净收益"，所以它是一个真正的势函数：
//   - 弃牌不改变底池也不改变投入 → ΔΦ = 0，不会凭空奖励弃牌
//   - 跟注 b：ΔΦ = b·(2·eq − 1)，胜率过半才有正收益
// 早先版本用的是 Φ = 底池 × (2·eq − 1)，少了"自己已投入"这一项，于是对手
// 往底池里下注就会把 Φ 推高 —— 弃一手 60% 胜率的牌也能拿到正奖励。实测的
// 后果是策略塌成"从不弃牌、摊牌率 44% → 80%"。归一化到起始筹码是为了让
// 奖励落在 O(1)：原始筹码量级（±200）会让价值网 MSE 上千，梯度全被它吃掉。
float phiOf(const HUState& s, int seat, float eq) {
    return (float(eq) * float(s.pot()) - float(s.total[seat])) / float(kStartingStack);
}

size_t totalOptions(const std::vector<Transition>& tmp) {
    size_t n = 0;
    for (const auto& t : tmp) n += t.actions.size();
    return n;
}

// 规则机器人的确定性版本：把 jitter 从随机数变成参数，供脚本教师的分支续演
// 复用。逻辑与 ruleAgentMove 完全一致（ruleAgentMove 就是随机抽一个 jitter）。
Move ruleAgentMoveWithJitter(const HUState& s, float jitter) {
    LegalMove mv[kNumActions];
    const int n = legalMoves(s, mv);
    if (n == 0) return Move{};

    const int me = s.toAct;
    Card hole[2] = {static_cast<Card>(s.hole[me][0]), static_cast<Card>(s.hole[me][1])};
    Card board[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < s.boardN && i < 5; ++i) board[i] = static_cast<Card>(s.board[i]);
    const float strength = handStrength(hole, board, s.boardN);
    const int need = toCall(s, me);
    const int pot = s.pot();

    auto checkCall = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotCheckCall) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };
    auto fold = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotFold) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };

    const int me0 = s.toAct;
    auto betPotFraction = [&](double frac) {
        const int inc = std::max<int>(kBigBlind, static_cast<int>(std::llround(frac * double(pot))));
        return int(s.contrib[me0]) + inc;
    };

    Move out{};
    if (need == 0) {
        const float betThreshold = 0.58f - 0.22f * jitter;
        if (strength > betThreshold && isRaisePossible(s)) {
            const double frac = (strength > 0.85f) ? 1.0 : (strength > 0.7f ? 0.75 : 0.5);
            out = makeRaise(s, betPotFraction(frac));
            if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
        }
        checkCall(out);
        return out;
    }

    const float odds = float(need) / float(pot + need);
    if (strength > 0.80f && jitter > 0.55f && isRaisePossible(s)) {
        out = makeRaise(s, strength > 0.93f ? maxRaiseTo(s) : betPotFraction(0.9));
        if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
    }
    if (strength > odds + 0.10f && jitter > 0.12f) {
        checkCall(out);
        return out;
    }
    if (strength < odds * 0.55f) {
        if (fold(out)) return out;
    }
    checkCall(out);
    return out;
}

// 翻前的"胜率机器人"版本：没有翻后抽牌，只用 169 类表，所以很便宜。
// 教师用它作为第 4 种续演风格，让 Q 见过"3BB 开场 / 按胜率防守"的节奏，
// 而不是只会对付规则机器人那种被压成 1.5BB 的最小加注。
Move proPreflopMoveWithJitter(const HUState& s, float jitter) {
    LegalMove lm[kNumActions];
    const int n = legalMoves(s, lm);
    if (n == 0 || s.street != ST_PREFLOP) return ruleAgentMoveWithJitter(s, jitter);
    const int me = s.toAct;
    const float eq = float(StrengthModel::instance().preflopEquity(
        static_cast<Card>(s.hole[me][0]), static_cast<Card>(s.hole[me][1])));
    const int need = toCall(s, me);
    const int pot = s.pot();

    auto checkCall = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (lm[i].slot == kSlotCheckCall) {
                out = lm[i].mv;
                return true;
            }
        }
        return false;
    };
    auto fold = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (lm[i].slot == kSlotFold) {
                out = lm[i].mv;
                return true;
            }
        }
        return false;
    };

    Move out{};
    if (need == 0) {
        if (eq > 0.52f && jitter > 0.15f && isRaisePossible(s)) {
            out = makeRaise(s, 3 * kBigBlind);
            if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
        }
        checkCall(out);
        return out;
    }
    const float odds = float(need) / float(pot + need);
    if (eq > 0.78f && jitter > 0.35f && isRaisePossible(s)) {
        out = makeRaise(s, 10 * kBigBlind);
        if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
    }
    if (eq > odds + 0.08f) {
        checkCall(out);
        return out;
    }
    if (eq < odds * 0.90f) {
        if (fold(out)) return out;
    }
    checkCall(out);
    return out;
}

// 教师用的续演：从分支后的状态开始，双方都按同一种脚本风格打到底。
// style 0/1/2 = 规则机器人的紧/标准/松三种 jitter；style 3 = 翻前用胜率机器人、
// 翻后用标准规则。公共牌在开局就发定（applyMove 不再抽牌），所以所有候选动作
// 看到的是同一副公共牌、同一个对手底牌 —— 分支差里没有发牌方差。
float continuationPayoff(const HUState& s, int seat, const Move& first, int style) {
    HUState st = s;
    static thread_local Rng dummy(0x51ed270bULL);
    static constexpr float kJitter[] = {0.20f, 0.50f, 0.80f, 0.50f};
    const float jitter = kJitter[style];
    applyMove(st, first, dummy);
    int guard = 0;
    while (!st.finished && guard++ < 128) {
        const Move next = (style == 3) ? proPreflopMoveWithJitter(st, jitter)
                                       : ruleAgentMoveWithJitter(st, jitter);
        applyMove(st, next, dummy);
    }
    return st.finished ? float(payoff(st, seat)) : 0.0f;
}

void writeTensors(const std::vector<Transition>& tmp, torch::Tensor& nodeT, torch::Tensor& seqT,
                  torch::Tensor& extraT, torch::Tensor& actFeatT, torch::Tensor& actSampleT,
                  torch::Tensor& actIdT, torch::Tensor& rangeEqT, bool withExtra) {
    const int64_t B = static_cast<int64_t>(tmp.size());
    const size_t L = totalOptions(tmp);
    nodeT = torch::empty({B, kNodeInput}, torch::kFloat32);
    seqT = torch::empty({B, kHistSteps, kHistDim}, torch::kFloat32);
    extraT = torch::empty({B, withExtra ? kExtraInput : 1}, torch::kFloat32);
    actFeatT = torch::empty({static_cast<int64_t>(L), kActionInput}, torch::kFloat32);
    actSampleT = torch::empty({static_cast<int64_t>(L)}, torch::kLong);
    actIdT = torch::empty({static_cast<int64_t>(L)}, torch::kLong);
    rangeEqT = torch::empty({B}, torch::kFloat32);

    float* pNode = nodeT.data_ptr<float>();
    float* pSeq = seqT.data_ptr<float>();
    float* pExtra = extraT.data_ptr<float>();
    float* pFeat = actFeatT.data_ptr<float>();
    int64_t* pSample = actSampleT.data_ptr<int64_t>();
    int64_t* pId = actIdT.data_ptr<int64_t>();
    float* pRng = rangeEqT.data_ptr<float>();

    int64_t off = 0;
    for (int64_t i = 0; i < B; ++i) {
        const Transition& t = tmp[static_cast<size_t>(i)];
        std::copy(t.node.begin(), t.node.end(), pNode + i * kNodeInput);
        std::copy(t.seq.begin(), t.seq.end(), pSeq + i * kSeqInput);
        if (withExtra) {
            std::copy(t.extra.begin(), t.extra.end(), pExtra + i * kExtraInput);
        }
        pRng[i] = t.rangeEq;
        for (const ActionOption& a : t.actions) {
            std::copy(a.feat.begin(), a.feat.end(), pFeat + off * kActionInput);
            pSample[off] = i;
            pId[off] = a.slot;
            ++off;
        }
    }
}

}  // namespace

const Snapshot* HistoricalPool::at(int tier, int index) const {
    if (tier != 0 && tier != 1) return nullptr;
    const std::vector<std::shared_ptr<Snapshot>>& v = (tier == 0) ? recent : archive;
    if (index < 0 || index >= static_cast<int>(v.size())) return nullptr;
    return v[static_cast<size_t>(index)].get();
}

void HistoricalPool::push(const std::shared_ptr<Snapshot>& snap, int recentCap, int archiveCap) {
    if (!snap) return;
    recent.push_back(snap);
    if (recentCap < 0) recentCap = 0;
    while (static_cast<int>(recent.size()) > recentCap) recent.erase(recent.begin());

    archive.push_back(snap);
    if (archiveCap < 1) archiveCap = 1;
    // 满了隔一个丢一个：保留的间隔翻倍，越老越稀疏但永不整体消失。
    while (static_cast<int>(archive.size()) > archiveCap) {
        std::vector<std::shared_ptr<Snapshot>> thinned;
        thinned.reserve(archive.size() / 2 + 1);
        for (size_t i = 0; i < archive.size(); i += 2) thinned.push_back(archive[i]);
        archive.swap(thinned);
    }
}

namespace {

// 从已经归一化（且带掩码）的动作概率里采样。sampleAction 与动作级集成共用
// 这一条路径，"单模型"和"多模型概率平均"的采样口径因此完全一致。
std::pair<int, float> sampleFromProbs(const float* probs, const std::vector<ActionOption>& acts,
                                      Rng& rng) {
    double sum = 0.0;
    for (const ActionOption& a : acts) sum += probs[a.slot];
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (!(sum > 0.0)) {
        const int k = static_cast<int>(rng() % acts.size());
        return {acts[static_cast<size_t>(k)].slot, -std::log(double(acts.size()))};
    }
    double r = u(rng) * sum;
    for (const ActionOption& a : acts) {
        const double p = probs[a.slot];
        if (r < p || &a == &acts.back()) {
            return {a.slot, float(std::log(p / sum))};
        }
        r -= p;
    }
    return {acts.back().slot, float(std::log(probs[acts.back().slot] / sum))};
}

// 带温度的 masked softmax：把 logits 除以 temp 再归一化。temp<1 让分布更尖。
void maskedSoftmaxTemp(const float* logits, int n, float temp, float* probs) {
    if (temp == 1.0f) {
        ppo::maskedSoftmax(logits, n, probs);
        return;
    }
    float buf[kNumActions];
    const float inv = 1.0f / std::max(temp, 1e-3f);
    for (int a = 0; a < n; ++a) buf[a] = logits[a] * inv;
    ppo::maskedSoftmax(buf, n, probs);
}

// 评估用贪心：在合法动作里取概率最大的那个（不再随机）。
std::pair<int, float> argmaxAction(const float* probs, const std::vector<ActionOption>& acts) {
    int best = acts[0].slot;
    float bp = probs[best];
    for (const ActionOption& a : acts) {
        if (probs[a.slot] > bp) {
            bp = probs[a.slot];
            best = a.slot;
        }
    }
    return {best, std::log(std::max(bp, 1e-12f))};
}

// 我方在当前公共牌 vs 一个确定对手底牌的摊牌胜率（平局算半胜）。
// 转牌/河牌精确枚举剩余公共牌，翻牌/翻前用蒙特卡洛（iters 次同花顺抽样）。
double showdownEquityVsHole(const HUState& s, int heroSeat, Card o0, Card o1, Rng& rng,
                            int iters) {
    bool used[kNumCards] = {};
    Card hero[2] = {static_cast<Card>(s.hole[heroSeat][0]),
                    static_cast<Card>(s.hole[heroSeat][1])};
    used[hero[0]] = used[hero[1]] = true;
    used[o0] = used[o1] = true;
    Card board[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < s.boardN && i < 5; ++i) {
        board[i] = static_cast<Card>(s.board[i]);
        used[board[i]] = true;
    }
    const int boardN = s.boardN;
    if (boardN >= 5) {
        Card h[7] = {hero[0], hero[1], board[0], board[1], board[2], board[3], board[4]};
        Card o[7] = {o0, o1, board[0], board[1], board[2], board[3], board[4]};
        const int hs = evaluate7(h);
        const int os = evaluate7(o);
        return (hs > os) ? 1.0 : (hs < os ? 0.0 : 0.5);
    }

    Card rest[kNumCards];
    int nr = 0;
    for (int c = 0; c < kNumCards; ++c) {
        if (!used[static_cast<Card>(c)]) rest[nr++] = static_cast<Card>(c);
    }
    const int need = 5 - boardN;
    if (need < 1 || nr < need) return 0.5;

    if (need == 1) {
        double acc = 0.0;
        for (int i = 0; i < nr; ++i) {
            Card h[7] = {hero[0], hero[1], board[0], board[1], board[2], board[3], rest[i]};
            Card o[7] = {o0, o1, board[0], board[1], board[2], board[3], rest[i]};
            const int hs = evaluate7(h);
            const int os = evaluate7(o);
            acc += (hs > os) ? 1.0 : (hs < os ? 0.0 : 0.5);
        }
        return acc / double(nr);
    }

    // MC：每次从剩余牌里抽 need 张不重复的做公共牌。
    std::uniform_int_distribution<int> pick(0, nr - 1);
    const int its = std::max(iters, 1);
    double acc = 0.0;
    Card pool[kNumCards];
    for (int it = 0; it < its; ++it) {
        for (int i = 0; i < nr; ++i) pool[i] = rest[i];
        for (int k = 0; k < need; ++k) {
            const int j = k + pick(rng) % (nr - k);
            std::swap(pool[k], pool[j]);
            board[boardN + k] = pool[k];
        }
        Card h[7] = {hero[0], hero[1], board[0], board[1], board[2], board[3], board[4]};
        Card o[7] = {o0, o1, board[0], board[1], board[2], board[3], board[4]};
        const int hs = evaluate7(h);
        const int os = evaluate7(o);
        acc += (hs > os) ? 1.0 : (hs < os ? 0.0 : 0.5);
    }
    return acc / double(its);
}

// 计算 rangeEq：从对手视角重建"它最近一次决策"的观测（换入采样底牌），
// 用当前策略在该动作上的概率作为似然，对采样组合做贝叶斯重加权，再按后验
// 权重算我方 vs 对手范围的摊牌胜率。似然前向固定喂中性 rangeEq=0.5，
// 避免"rangeEq 依赖自身"的递归（定义清楚、部署也能复现）。
float rangeEquityFor(const HUState& now, int heroSeat, const RangeObs& last,
                     const nn::Actor* oppActor, int samples, int iters, Rng& rng) {
    const int oppSeat = 1 - heroSeat;
    if (!last.valid || last.slot < 0 || !oppActor) return 0.5f;
    if (last.pre.toAct != oppSeat || last.pre.finished) return 0.5f;

    LegalMove lm[kNumActions];
    const int n = legalMoves(last.pre, lm);
    if (n <= 0) return 0.5f;
    bool legalSlot = false;
    for (int i = 0; i < n; ++i) {
        if (lm[i].slot == last.slot) legalSlot = true;
    }
    if (!legalSlot) return 0.5f;

    // 采样池：当前公共牌 + 我方底牌都不能出现在对手手里。
    bool used[kNumCards] = {};
    used[static_cast<Card>(now.hole[heroSeat][0])] = true;
    used[static_cast<Card>(now.hole[heroSeat][1])] = true;
    for (int i = 0; i < now.boardN && i < 5; ++i) used[static_cast<Card>(now.board[i])] = true;
    Card rest[kNumCards];
    int nr = 0;
    for (int c = 0; c < kNumCards; ++c) {
        if (!used[static_cast<Card>(c)]) rest[nr++] = static_cast<Card>(c);
    }
    const int M = std::min(std::max(samples, 1), nr / 2);
    if (M < 1) return 0.5f;

    // 对手在动作前状态下的合法动作（对所有采样组合相同，只编码一次）。
    std::vector<ActionOption> opts(static_cast<size_t>(n));
    for (int k = 0; k < n; ++k) {
        opts[static_cast<size_t>(k)].slot = lm[k].slot;
        opts[static_cast<size_t>(k)].mv = lm[k].mv;
        opts[static_cast<size_t>(k)].hi = lm[k].toHi;
        encodeActionFeature(last.pre, lm[k], opts[static_cast<size_t>(k)].feat);
    }

    // 无放回洗牌，按相邻两张配成对手底牌。
    std::uniform_int_distribution<int> pick(0, nr - 1);
    for (int i = 0; i < 2 * M; ++i) {
        const int j = i + pick(rng) % (nr - i);
        std::swap(rest[i], rest[j]);
    }

    std::vector<Transition> tmp(static_cast<size_t>(M));
    std::vector<Card> h0(static_cast<size_t>(M)), h1(static_cast<size_t>(M));
    for (int m = 0; m < M; ++m) {
        h0[static_cast<size_t>(m)] = rest[2 * m];
        h1[static_cast<size_t>(m)] = rest[2 * m + 1];
        HUState st = last.pre;
        st.hole[oppSeat][0] = h0[static_cast<size_t>(m)];
        st.hole[oppSeat][1] = h1[static_cast<size_t>(m)];
        Obs obs;
        encodeObservation(st, obs, 0.5f);
        Transition& t = tmp[static_cast<size_t>(m)];
        t.node = obs.node;
        t.seq = obs.seq;
        t.actions = opts;
        t.rangeEq = 0.5f;
    }

    torch::NoGradGuard ng;
    torch::Tensor nodeT, seqT, extraT, actFeatT, actSampleT, actIdT, rangeEqT;
    writeTensors(tmp, nodeT, seqT, extraT, actFeatT, actSampleT, actIdT, rangeEqT, false);
    std::vector<double> w(static_cast<size_t>(M), 0.0);
    double wsum = 0.0;
    if (gEmpiricalRange) {
        // 经验对手模型（部署用）：用 Slumbot 日志拟合的"动作类型 → 手牌百分位"
        // 密度当似然。抽样组合均匀，prior 近似均匀，所以密度本身即可当权重。
        const int st = int(last.pre.street);
        const int bN = (st == ST_PREFLOP) ? 0 : st + 2;
        const int kind = last.slot == kSlotFold ? 0 : (last.slot == kSlotCheckCall ? 1 : 2);
        for (int m = 0; m < M; ++m) {
            Card hh[2] = {h0[static_cast<size_t>(m)], h1[static_cast<size_t>(m)]};
            int hhI[2] = {int(hh[0]), int(hh[1])};
            int bdI[5] = {0, 0, 0, 0, 0};
            for (int b = 0; b < bN && b < 5; ++b) bdI[b] = int(last.pre.board[b]);
            const float pct = handStrength(hhI, bdI, bN);
            const int bin = std::min(9, std::max(0, int(pct * 10.0f)));
            const double p = kOppDensity[st][kind][bin];
            w[static_cast<size_t>(m)] = p;
            wsum += p;
        }
    } else {
        torch::Tensor logits = (*oppActor)->forward(nodeT, seqT, actFeatT, actSampleT, actIdT,
                                                    rangeEqT);
        const float* lp = logits.data_ptr<float>();
        for (int m = 0; m < M; ++m) {
            float probs[kNumActions];
            ppo::maskedSoftmax(lp + size_t(m) * kNumActions, kNumActions, probs);
            const double p = probs[last.slot];
            w[static_cast<size_t>(m)] = p;
            wsum += p;
        }
    }

    double acc = 0.0;
    if (wsum > 1e-12) {
        for (int m = 0; m < M; ++m) {
            acc += w[static_cast<size_t>(m)] *
                   showdownEquityVsHole(now, heroSeat, h0[static_cast<size_t>(m)],
                                        h1[static_cast<size_t>(m)], rng, iters);
        }
        acc /= wsum;
    } else {
        for (int m = 0; m < M; ++m) {
            acc += showdownEquityVsHole(now, heroSeat, h0[static_cast<size_t>(m)],
                                        h1[static_cast<size_t>(m)], rng, iters);
        }
        acc /= double(M);
    }
    return float(std::min(1.0, std::max(0.0, acc)));
}

}  // namespace

// 把实际打出的 Move 映射回抽象动作槽（rangeEq 需要知道对手动作在策略里
// 对应哪个槽才能算似然）。尺寸带内按区间包含匹配，再退到最近端点。
int slotOfMove(const HUState& s, const Move& mv) {
    LegalMove lm[kNumActions];
    const int n = legalMoves(s, lm);
    if (n <= 0) return -1;
    if (mv.kind == MK_FOLD) {
        for (int i = 0; i < n; ++i) {
            if (lm[i].slot == kSlotFold) return lm[i].slot;
        }
        return -1;
    }
    if (mv.kind == MK_CHECK || mv.kind == MK_CALL) {
        for (int i = 0; i < n; ++i) {
            if (lm[i].slot == kSlotCheckCall) return lm[i].slot;
        }
        return -1;
    }
    int best = -1;
    int bestDist = 1 << 30;
    for (int i = 0; i < n; ++i) {
        if (lm[i].slot < kSlotSize0) continue;
        if (int(mv.to) >= int(lm[i].mv.to) && int(mv.to) <= lm[i].toHi) return lm[i].slot;
        const int d = std::abs(int(mv.to) - int(lm[i].mv.to));
        if (d < bestDist) {
            bestDist = d;
            best = lm[i].slot;
        }
    }
    return best;
}

float rangeEquityAt(const HUState& now, int heroSeat, const RangeObs& last,
                    const nn::Actor* oppActor, int samples, int iters, Rng& rng) {
    return rangeEquityFor(now, heroSeat, last, oppActor, samples, iters, rng);
}

std::pair<int, float> sampleAction(const float* logits, const std::vector<ActionOption>& acts,
                                   Rng& rng) {
    float probs[kNumActions];
    ppo::maskedSoftmax(logits, kNumActions, probs);
    return sampleFromProbs(probs, acts, rng);
}

Move policyMove(const nn::Actor* sb, const nn::Actor* bb, const HUState& s, Rng& rng,
                float temp, float rangeEq) {
    LegalMove lm[kNumActions];
    const int n = legalMoves(s, lm);
    if (n <= 0) return Move{};
    const nn::Actor* actors[2] = {sb, bb};
    const int seat = s.toAct;
    torch::NoGradGuard ng;

    const float eq = float(perfectEquity(s, seat, 24));
    Obs obs;
    encodeObservation(s, obs, eq);

    std::vector<Transition> tmp(1);
    Transition& t = tmp[0];
    t.node = obs.node;
    t.seq = obs.seq;
    t.extra = obs.extra;
    t.rangeEq = rangeEq;
    t.actions.resize(static_cast<size_t>(n));
    for (int k = 0; k < n; ++k) {
        ActionOption& a = t.actions[static_cast<size_t>(k)];
        a.slot = lm[k].slot;
        a.mv = lm[k].mv;
        a.hi = lm[k].toHi;
        encodeActionFeature(s, lm[k], a.feat);
    }

    torch::Tensor nodeT, seqT, extraT, actFeatT, actSampleT, actIdT, rangeEqT;
    writeTensors(tmp, nodeT, seqT, extraT, actFeatT, actSampleT, actIdT, rangeEqT, false);
    torch::Tensor logits = (*actors[seat])->forward(nodeT, seqT, actFeatT, actSampleT, actIdT,
                                                    rangeEqT);
    float probs[kNumActions];
    maskedSoftmaxTemp(logits.data_ptr<float>(), kNumActions, temp, probs);
    auto [slot, logp] = sampleFromProbs(probs, t.actions, rng);
    (void)logp;

    const ActionOption* opt = &t.actions.back();
    for (const ActionOption& a : t.actions) {
        if (a.slot == slot) opt = &a;
    }
    Move played = opt->mv;
    if (played.kind == MK_RAISE && opt->hi > int(played.to)) {
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        played.to = static_cast<int16_t>(amountInBand(int(played.to), opt->hi, u01(rng)));
    }
    return played;
}

// 河牌范围：枚举公共牌之外的 1081 个组合，用**该座位**在自己最近一次决策处
// 的策略似然当权重。和 rangeEquity / 范围搜索是同一口径（似然前向固定喂
// rangeEq=0.5），区别只是把"抽几个样本"换成"全枚举"——CFR 求解需要整条
// 范围，16 个样本会让范围退化成 16 个点。
//
// targetSeat 是要估范围的座位；last 必须是它在最近一次决策时的快照（快照的
// 公共牌可以少于 5 张：那时它还没看到河牌，似然正是在它当时的观测上算的）。
// 返回向量与 riverCombos(now) 一一对应；未归一化（求解器自己归一）。
std::vector<float> riverRangeFor(const HUState& now, int targetSeat, const RangeObs& last,
                                 const nn::Actor* actor) {
    const std::vector<RiverCombo> combos = riverCombos(now);
    std::vector<float> w(combos.size(), 1.0f);
    if (actor == nullptr || now.boardN < 5) return w;
    if (!last.valid || last.slot < 0 || last.pre.finished || last.pre.toAct != targetSeat) return w;

    LegalMove olm[kNumActions];
    const int on = legalMoves(last.pre, olm);
    if (on <= 0) return w;
    int observed = -1;
    for (int k = 0; k < on; ++k) {
        if (olm[k].slot == last.slot) observed = k;
    }
    if (observed < 0) return w;

    std::vector<ActionOption> opts(static_cast<size_t>(on));
    for (int k = 0; k < on; ++k) {
        opts[static_cast<size_t>(k)].slot = olm[k].slot;
        opts[static_cast<size_t>(k)].mv = olm[k].mv;
        opts[static_cast<size_t>(k)].hi = olm[k].toHi;
        encodeActionFeature(last.pre, olm[k], opts[static_cast<size_t>(k)].feat);
    }

    const int M = static_cast<int>(combos.size());
    std::vector<Transition> tmp(static_cast<size_t>(M));
    for (int m = 0; m < M; ++m) {
        HUState st = last.pre;
        st.hole[targetSeat][0] = combos[static_cast<size_t>(m)].c0;
        st.hole[targetSeat][1] = combos[static_cast<size_t>(m)].c1;
        Obs obs;
        encodeObservation(st, obs, 0.5f);
        Transition& t = tmp[static_cast<size_t>(m)];
        t.node = obs.node;
        t.seq = obs.seq;
        t.actions = opts;
        t.rangeEq = 0.5f;
    }

    torch::NoGradGuard ng;
    torch::Tensor nodeT, seqT, extraT, actFeatT, actSampleT, actIdT, rangeEqT;
    writeTensors(tmp, nodeT, seqT, extraT, actFeatT, actSampleT, actIdT, rangeEqT, false);
    torch::Tensor logits = (*actor)->forward(nodeT, seqT, actFeatT, actSampleT, actIdT,
                                             rangeEqT);
    const float* lp = logits.data_ptr<float>();
    for (int m = 0; m < M; ++m) {
        float probs[kNumActions];
        ppo::maskedSoftmax(lp + static_cast<size_t>(m) * kNumActions, kNumActions, probs);
        // 下限：否则模型认为"绝不可能"的组合会被彻底抹掉，范围变成几个点后
        // 求解器的 BR 会把"对手只拿这几手牌"当真，反而放大可利用度。
        w[static_cast<size_t>(m)] = std::max(probs[observed], 1e-4f);
    }
    return w;
}

Move ruleAgentMove(const HUState& s, Rng& rng) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    return ruleAgentMoveWithJitter(s, u(rng));
}

namespace {

// 给定一副完整牌（对手底牌和五张公共牌都确定）的动作价值。
void teacherValuesSingle(const HUState& s, int seat, std::array<float, kNumActions>& q,
                         int styles) {
    q.fill(0.0f);
    LegalMove lm[kNumActions];
    const int n = legalMoves(s, lm);
    if (n <= 0) return;

    // 四种续演风格：紧弱 / 标准 / 松凶的规则机器人，加上翻前胜率机器人。
    // 候选动作在每条风格下各续演一次，取平均 —— 单一风格会把教师变成
    // "只克这一种节奏"。
    static constexpr int kStyles[] = {0, 1, 2, 3};
    if (styles < 1) styles = 1;
    if (styles > 4) styles = 4;

    for (int i = 0; i < n; ++i) {
        Move mv = lm[i].mv;
        // 尺寸带没有唯一金额：教师用对数中点作为代表金额（策略采样在带内
        // 对数均匀，所以中点是中位数）。
        if (mv.kind == MK_RAISE && lm[i].toHi > int(mv.to)) {
            mv.to = static_cast<int16_t>(amountInBand(int(mv.to), lm[i].toHi, 0.5));
        }
        double acc = 0.0;
        for (int k = 0; k < styles; ++k) {
            acc += double(continuationPayoff(s, seat, mv, kStyles[k]));
        }
        q[lm[i].slot] = static_cast<float>(acc / (double(styles) * double(kStartingStack)));
    }
}

// 从"我们自己的信息集"重新发一手隐藏牌：保留自己的底牌和已发公共牌，
// 重新抽对手底牌和未发公共牌。用于把 clairvoyant Q 变成信息集上的期望 Q。
void sampleHiddenCards(const HUState& s, int seat, Rng& rng, HUState& out) {
    out = s;
    const int opp = 1 - seat;
    bool used[kNumCards] = {};
    used[s.hole[seat][0]] = true;
    used[s.hole[seat][1]] = true;
    for (int i = 0; i < s.boardN; ++i) used[s.board[i]] = true;

    Card deck[kNumCards];
    int n = 0;
    for (int c = 0; c < kNumCards; ++c) {
        if (!used[c]) deck[n++] = static_cast<Card>(c);
    }
    const int boardNeed = 5 - int(s.boardN);
    const int need = 2 + boardNeed;
    for (int i = 0; i < need; ++i) {
        std::uniform_int_distribution<int> d(i, n - 1);
        std::swap(deck[i], deck[d(rng)]);
    }
    out.hole[opp][0] = deck[0];
    out.hole[opp][1] = deck[1];
    for (int j = 0; j < boardNeed; ++j) {
        out.board[int(s.boardN) + j] = deck[2 + j];
    }
}

}  // namespace

void teacherActionValues(const HUState& s, int seat, std::array<float, kNumActions>& q,
                         int samples, int styles) {
    q.fill(0.0f);
    if (samples <= 1) {
        teacherValuesSingle(s, seat, q, styles);
        return;
    }

    // 教师抽样用的 RNG 和自对局 RNG 分开，避免改变训练主流程的随机序列。
    static thread_local Rng rng(0x51ed270b9e3779b9ULL);
    HUState hidden;
    std::array<float, kNumActions> one;
    for (int k = 0; k < samples; ++k) {
        sampleHiddenCards(s, seat, rng, hidden);
        teacherValuesSingle(hidden, seat, one, styles);
        for (int i = 0; i < kNumActions; ++i) q[i] += one[i];
    }
    const float inv = 1.0f / float(samples);
    for (int i = 0; i < kNumActions; ++i) q[i] *= inv;
}

Move searchAgentMove(const HUState& s, Rng& rng, int samples, int styles) {
    LegalMove lm[kNumActions];
    const int n = legalMoves(s, lm);
    if (n <= 0) return Move{};
    (void)rng;

    std::array<float, kNumActions> q;
    teacherActionValues(s, s.toAct, q, samples, styles);

    int best = 0;
    float bestQ = -1e30f;
    for (int i = 0; i < n; ++i) {
        const float v = q[lm[i].slot];
        if (std::isfinite(v) && v > bestQ) {
            bestQ = v;
            best = i;
        }
    }
    Move mv = lm[best].mv;
    if (mv.kind == MK_RAISE && lm[best].toHi > int(mv.to)) {
        mv.to = static_cast<int16_t>(amountInBand(int(mv.to), lm[best].toHi, 0.5));
    }
    return mv;
}

namespace {

// 策略内搜索的单步续演：把一批"批量 rollout"状态按行动座位分组，各自前向
// 一次 actor，按策略采样动作推进一步。返回哪些状态已经终局（并累加回报）。
struct Leaf {
    HUState st;
    int branch = 0;
    // 样本的后验权重（范围条件化搜索）。均匀搜索恒为 1，行为与原来一致。
    double w = 1.0;
};

void advanceLeaves(std::vector<Leaf>& live, const nn::Actor* const* actors, int equityIters,
                   Rng& rng, std::vector<double>& q, int seat, std::vector<Leaf>& next,
                   int oppSeat, Move (*oppScript)(const HUState&, Rng&)) {
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (int act = 0; act < 2; ++act) {
        std::vector<size_t> idx;
        for (size_t i = 0; i < live.size(); ++i) {
            if (live[i].st.toAct == act) idx.push_back(i);
        }
        if (idx.empty()) continue;

        // 鲁棒搜索的脚本对手：直接按脚本策略出招，不消耗一次前向。脚本的
        // 动作是确定的（不含隐藏牌信息），所以这些叶子对"对手模型"的假设
        // 是"它只会这么打"，和 actor 模型的混合形成最坏情况比较。
        if (act == oppSeat && oppScript != nullptr) {
            for (size_t j = 0; j < idx.size(); ++j) {
                Leaf& L = live[idx[j]];
                const Move mv = oppScript(L.st, rng);
                applyMove(L.st, mv, rng);
                if (L.st.finished) {
                    q[static_cast<size_t>(L.branch)] += L.w * double(payoff(L.st, seat));
                } else {
                    next.push_back(std::move(L));
                }
            }
            continue;
        }

        std::vector<Transition> tmp(idx.size());
        for (size_t j = 0; j < idx.size(); ++j) {
            Leaf& L = live[idx[j]];
            Transition& t = tmp[j];
            const float eq = float(perfectEquity(L.st, act, equityIters));
            Obs obs;
            encodeObservation(L.st, obs, eq);
            t.node = obs.node;
            t.seq = obs.seq;
            t.extra = obs.extra;
            LegalMove lm[kNumActions];
            const int n = legalMoves(L.st, lm);
            t.actions.resize(static_cast<size_t>(n));
            for (int k = 0; k < n; ++k) {
                ActionOption& a = t.actions[static_cast<size_t>(k)];
                a.slot = lm[k].slot;
                a.mv = lm[k].mv;
                a.hi = lm[k].toHi;
                encodeActionFeature(L.st, lm[k], a.feat);
            }
            t.gameId = static_cast<int>(j);
        }

        torch::Tensor nodeT, seqT, extraT, actFeatT, actSampleT, actIdT, rangeEqT;
        writeTensors(tmp, nodeT, seqT, extraT, actFeatT, actSampleT, actIdT, rangeEqT, false);
        torch::Tensor logits = (*actors[act])->forward(nodeT, seqT, actFeatT, actSampleT, actIdT,
                                                       rangeEqT);
        const float* pLogits = logits.data_ptr<float>();

        for (size_t j = 0; j < idx.size(); ++j) {
            Leaf& L = live[idx[j]];
            Transition& t = tmp[j];
            auto [slot, logp] = sampleAction(pLogits + int64_t(j) * kNumActions, t.actions, rng);
            (void)logp;
            const ActionOption* opt = &t.actions.back();
            for (const ActionOption& a : t.actions) {
                if (a.slot == slot) opt = &a;
            }
            Move played = opt->mv;
            if (played.kind == MK_RAISE && opt->hi > int(played.to)) {
                played.to = static_cast<int16_t>(amountInBand(int(played.to), opt->hi, u01(rng)));
            }
            applyMove(L.st, played, rng);
            if (L.st.finished) {
                q[static_cast<size_t>(L.branch)] += L.w * double(payoff(L.st, seat));
            } else {
                next.push_back(L);
            }
        }
    }
}

}  // namespace

Move policySearchMove(const nn::Actor* sb, const nn::Actor* bb, const HUState& s, int seat,
                      int samples, Rng& rng, int equityIters, const RangeObs* oppLast,
                      const nn::Actor* oppActor, std::array<float, kNumActions>* qOut,
                      const std::vector<HUState>* givenHidden,
                      const std::vector<double>* givenW,
                      Move (*oppScript)(const HUState&, Rng&)) {
    const nn::Actor* actors[2] = {sb, bb};
    LegalMove lm[kNumActions];
    const int n = legalMoves(s, lm);
    if (qOut != nullptr) qOut->fill(0.0f);
    if (n <= 0) return Move{};
    if (n == 1 || samples <= 0) {
        Move mv = lm[0].mv;
        if (mv.kind == MK_RAISE && lm[0].toHi > int(mv.to)) {
            mv.to = static_cast<int16_t>(amountInBand(int(mv.to), lm[0].toHi, 0.5));
        }
        return mv;
    }
    torch::NoGradGuard noGrad;

    // 根动作：尺寸带取对数中点（和教师/搜索机器人同一口径）。
    std::vector<Move> branch(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        branch[static_cast<size_t>(i)] = lm[i].mv;
        Move& mv = branch[static_cast<size_t>(i)];
        if (mv.kind == MK_RAISE && lm[i].toHi > int(mv.to)) {
            mv.to = static_cast<int16_t>(amountInBand(int(mv.to), lm[i].toHi, 0.5));
        }
    }

    // 隐藏牌样本：河牌上没有未发公共牌，唯一未知量是**对手底牌**，所以范围
    // 条件化只是给对手底牌加权；翻前到转牌还要一起重抽未发公共牌，按对手
    // 底牌似然加权会引入发牌偏差，因此只在河牌启用（调用方也只在这条街调）。
    const int M = samples;
    const int oppSeat = 1 - seat;
    std::vector<HUState> hidden;
    if (givenHidden != nullptr && static_cast<int>(givenHidden->size()) >= M) {
        hidden.assign(givenHidden->begin(), givenHidden->begin() + M);
    } else {
        hidden.resize(static_cast<size_t>(M));
        for (int k = 0; k < M; ++k) sampleHiddenCards(s, seat, rng, hidden[static_cast<size_t>(k)]);
    }

    // 后验权重 = 对手在它最近决策处打出实际动作的策略似然。似然前向固定喂
    // 中性 rangeEq=0.5（避免"权重依赖自身"的递归），与 rangeEq 特征同口径。
    std::vector<double> w(static_cast<size_t>(M), 1.0);
    bool useRange = oppLast != nullptr && oppActor != nullptr && s.boardN >= 5 &&
                    oppLast->valid && oppLast->slot >= 0 &&
                    oppLast->pre.toAct == oppSeat && !oppLast->pre.finished;
    if (givenW != nullptr && static_cast<int>(givenW->size()) >= M) {
        // 调用方预先算好的权重（鲁棒搜索里的脚本模型用均匀权重）。
        useRange = false;
        w.assign(givenW->begin(), givenW->begin() + M);
    } else if (useRange) {
        LegalMove olm[kNumActions];
        const int on = legalMoves(oppLast->pre, olm);
        bool slotLegal = false;
        for (int k = 0; k < on; ++k) {
            if (olm[k].slot == oppLast->slot) slotLegal = true;
        }
        if (on > 0 && slotLegal) {
            std::vector<ActionOption> opts(static_cast<size_t>(on));
            for (int k = 0; k < on; ++k) {
                ActionOption& a = opts[static_cast<size_t>(k)];
                a.slot = olm[k].slot;
                a.mv = olm[k].mv;
                a.hi = olm[k].toHi;
                encodeActionFeature(oppLast->pre, olm[k], a.feat);
            }
            std::vector<Transition> tmp(static_cast<size_t>(M));
            for (int k = 0; k < M; ++k) {
                HUState st = oppLast->pre;
                st.hole[oppSeat][0] = hidden[static_cast<size_t>(k)].hole[oppSeat][0];
                st.hole[oppSeat][1] = hidden[static_cast<size_t>(k)].hole[oppSeat][1];
                Obs obs;
                encodeObservation(st, obs, 0.5f);
                Transition& t = tmp[static_cast<size_t>(k)];
                t.node = obs.node;
                t.seq = obs.seq;
                t.actions = opts;
                t.rangeEq = 0.5f;
            }
            torch::Tensor nodeT, seqT, extraT, actFeatT, actSampleT, actIdT, rangeEqT;
            writeTensors(tmp, nodeT, seqT, extraT, actFeatT, actSampleT, actIdT, rangeEqT, false);
            torch::Tensor logits =
                (*oppActor)->forward(nodeT, seqT, actFeatT, actSampleT, actIdT, rangeEqT);
            const float* lp = logits.data_ptr<float>();
            double wsum = 0.0;
            for (int k = 0; k < M; ++k) {
                float probs[kNumActions];
                ppo::maskedSoftmax(lp + size_t(k) * kNumActions, kNumActions, probs);
                w[static_cast<size_t>(k)] = double(probs[oppLast->slot]);
                wsum += w[static_cast<size_t>(k)];
            }
            if (wsum > 1e-9) {
                for (double& x : w) x /= wsum;
            } else {
                useRange = false;  // 似然全零（数值退化）=> 退回均匀搜索
            }
        } else {
            useRange = false;
        }
    }

    std::vector<double> q(static_cast<size_t>(n), 0.0);
    std::vector<Leaf> live;
    live.reserve(static_cast<size_t>(M) * static_cast<size_t>(n) * 3);

    // 每个隐藏牌样本对**所有**根动作各续演一次：分支之间共享发牌，动作间的
    // 差没有发牌方差，这一点和教师的分支续演一样。样本回报按后验权重累加。
    for (int k = 0; k < M; ++k) {
        for (int i = 0; i < n; ++i) {
            Leaf L;
            L.st = hidden[static_cast<size_t>(k)];
            L.branch = i;
            L.w = w[static_cast<size_t>(k)];
            applyMove(L.st, branch[static_cast<size_t>(i)], rng);
            if (L.st.finished) {
                q[static_cast<size_t>(i)] += L.w * double(payoff(L.st, seat));
            } else {
                live.push_back(std::move(L));
            }
        }
    }

    int guard = 0;
    while (!live.empty() && guard++ < 64) {
        std::vector<Leaf> next;
        next.reserve(live.size());
        advanceLeaves(live, actors, equityIters, rng, q, seat, next, oppSeat, oppScript);
        live.swap(next);
    }
    // 理论上一手牌一定会终局；万一 guard 用尽，把未完成的局按当前净投入计。
    for (const Leaf& L : live) {
        q[static_cast<size_t>(L.branch)] += L.w * double(payoff(L.st, seat));
    }

    int best = 0;
    double wsum = 0.0;
    for (double x : w) wsum += x;
    if (qOut != nullptr) {
        // 权重和：均匀搜索是 M，后验加权是 1。除以它把 q 变成期望回报，
        // 再按起始筹码归一（与脚本教师 teacherQ 的单位一致）。
        const double denom = (wsum > 1e-9 ? wsum : 1.0) * double(kStartingStack);
        for (int i = 0; i < n; ++i) {
            (*qOut)[lm[i].slot] = static_cast<float>(q[static_cast<size_t>(i)] / denom);
        }
    }
    for (int i = 1; i < n; ++i) {
        if (q[static_cast<size_t>(i)] > q[static_cast<size_t>(best)]) best = i;
    }
    return branch[static_cast<size_t>(best)];
}

Move policySearchRobustMove(const nn::Actor* sb, const nn::Actor* bb, const HUState& s, int seat,
                            int samples, Rng& rng, int equityIters, const RangeObs* oppLast,
                            const nn::Actor* oppActor, bool useMean) {
    LegalMove lm[kNumActions];
    const int n = legalMoves(s, lm);
    if (n <= 0) return Move{};
    if (n == 1 || samples <= 0) {
        Move mv = lm[0].mv;
        if (mv.kind == MK_RAISE && lm[0].toHi > int(mv.to)) {
            mv.to = static_cast<int16_t>(amountInBand(int(mv.to), lm[0].toHi, 0.5));
        }
        return mv;
    }
    // 所有模型共享同一批隐藏牌：否则"最小 EV"里混进了发牌方差，
    // 最坏情况会退化成噪声。
    std::vector<HUState> hidden(static_cast<size_t>(samples));
    for (int k = 0; k < samples; ++k) sampleHiddenCards(s, seat, rng, hidden[static_cast<size_t>(k)]);
    const std::vector<double> uniform(static_cast<size_t>(samples), 1.0);
    const Rng saved = rng;

    // 对手模型：0 = 自身 actor（带后验范围），1 = 跟注型 solid，2 = 价值型 shark。
    // 脚本模型的动作似然与底牌无关，所以后验权重退化为均匀。
    struct Model {
        Move (*script)(const HUState&, Rng&);
        bool ownRange;
    };
    const Model models[3] = {{nullptr, true}, {&solidAgentMove, false}, {&sharkAgentMove, false}};
    std::array<float, kNumActions> q[3];
    for (int m = 0; m < 3; ++m) {
        rng = saved;
        policySearchMove(sb, bb, s, seat, samples, rng, equityIters,
                         models[m].ownRange ? oppLast : nullptr,
                         models[m].ownRange ? oppActor : nullptr, &q[m], &hidden,
                         models[m].ownRange ? nullptr : &uniform, models[m].script);
    }

    // 最坏情况（默认）或等权平均：可选项只在聚合口径上不同，三份 q 的样本相同。
    int best = 0;
    float bestScore = -1e30f;
    for (int i = 0; i < n; ++i) {
        const int slot = lm[i].slot;
        float score = useMean ? 0.0f : q[0][static_cast<size_t>(slot)];
        for (int m = 0; m < 3; ++m) {
            if (useMean) score += q[m][static_cast<size_t>(slot)] / 3.0f;
            else if (m > 0) score = std::min(score, q[m][static_cast<size_t>(slot)]);
        }
        if (score > bestScore) {
            bestScore = score;
            best = i;
        }
    }
    Move mv = lm[best].mv;
    if (mv.kind == MK_RAISE && lm[best].toHi > int(mv.to)) {
        mv.to = static_cast<int16_t>(amountInBand(int(mv.to), lm[best].toHi, 0.5));
    }
    return mv;
}

namespace {

// 对随机手牌的胜率：翻前用 169 类表（精确），翻后用蒙特卡洛。
// 这是"胜率机器人"的核心——比只看当前五张牌百分位的规则机器人更接近
// 真实胜率，尤其不会把花听/顺听当成空气牌。
float equityVsRandom(const HUState& s, int seat, int iters) {
    if (s.boardN <= 0) {
        return float(StrengthModel::instance().preflopEquity(
            static_cast<Card>(s.hole[seat][0]), static_cast<Card>(s.hole[seat][1])));
    }
    static thread_local Rng rng(0x7f4a7c15ULL);
    HUState hidden;
    double acc = 0.0;
    for (int it = 0; it < iters; ++it) {
        sampleHiddenCards(s, seat, rng, hidden);
        Card a[7] = {static_cast<Card>(hidden.hole[seat][0]),
                     static_cast<Card>(hidden.hole[seat][1]),
                     static_cast<Card>(hidden.board[0]), static_cast<Card>(hidden.board[1]),
                     static_cast<Card>(hidden.board[2]), static_cast<Card>(hidden.board[3]),
                     static_cast<Card>(hidden.board[4])};
        Card b[7] = {static_cast<Card>(hidden.hole[1 - seat][0]),
                     static_cast<Card>(hidden.hole[1 - seat][1]),
                     static_cast<Card>(hidden.board[0]), static_cast<Card>(hidden.board[1]),
                     static_cast<Card>(hidden.board[2]), static_cast<Card>(hidden.board[3]),
                     static_cast<Card>(hidden.board[4])};
        const int s0 = evaluate7(a);
        const int s1 = evaluate7(b);
        acc += (s0 > s1) ? 1.0 : ((s0 < s1) ? 0.0 : 0.5);
    }
    return float(acc / double(iters));
}

// 极化范围的条件胜率：假设对手范围 = "最强 keep 比例的组件"（价值）+ 占比
// airShare 的**空气**（诈唬）。这是对"只取顶 K%"版本的修正：
// 只取顶 K% 等于假设我们从不诈唬，对手因此过度弃牌，训练出来的策略会学到
// "多诈唬"（README #108 的 rg1 就是这样退化的）。加入空气份后，对手的跟注
// 频率由赔率决定，诈唬不再自动赚钱——这才是均衡附近的压力。
//
// airShare 是**假设范围里空气占的比例**。均匀采样下空气的自然占比是
// 1-keep，所以每个空气组件的权重取 wAir = airShare*keep / ((1-airShare)*(1-keep))。
float equityVsPolarizedRange(const HUState& s, int seat, int iters, float keep,
                             float airShare) {
    if (s.boardN <= 0) return equityVsRandom(s, seat, iters);
    static thread_local Rng rng(0x51f3a2b7ULL);
    keep = std::min(std::max(keep, 0.02f), 0.98f);
    airShare = std::min(std::max(airShare, 0.0f), 0.9f);
    HUState hidden;

    const int probe = 96;
    int scores[probe];
    for (int it = 0; it < probe; ++it) {
        sampleHiddenCards(s, seat, rng, hidden);
        int c[7];
        int n = 0;
        c[n++] = int(hidden.hole[1 - seat][0]);
        c[n++] = int(hidden.hole[1 - seat][1]);
        for (int i = 0; i < s.boardN; ++i) c[n++] = int(s.board[i]);
        scores[it] = evaluateBest(c, n);
    }
    std::sort(scores, scores + probe);
    const int cut = std::min(probe - 1, std::max(0, int(double(probe) * (1.0 - double(keep)))));
    const int thr = scores[cut];

    const double pAir = 1.0 - double(keep);
    const double wAir = (pAir > 1e-6)
                            ? double(airShare) * double(keep) /
                                  ((1.0 - double(airShare)) * pAir)
                            : 0.0;
    double acc = 0.0;
    double wSum = 0.0;
    for (int it = 0; it < iters; ++it) {
        sampleHiddenCards(s, seat, rng, hidden);
        int c[7];
        int n = 0;
        c[n++] = int(hidden.hole[1 - seat][0]);
        c[n++] = int(hidden.hole[1 - seat][1]);
        for (int i = 0; i < s.boardN; ++i) c[n++] = int(s.board[i]);
        const bool value = evaluateBest(c, n) >= thr;
        if (!value && wAir <= 0.0) continue;
        const double w = value ? 1.0 : wAir;
        Card a[7] = {static_cast<Card>(s.hole[seat][0]), static_cast<Card>(s.hole[seat][1]),
                     static_cast<Card>(hidden.board[0]), static_cast<Card>(hidden.board[1]),
                     static_cast<Card>(hidden.board[2]), static_cast<Card>(hidden.board[3]),
                     static_cast<Card>(hidden.board[4])};
        Card b[7] = {static_cast<Card>(hidden.hole[1 - seat][0]),
                     static_cast<Card>(hidden.hole[1 - seat][1]),
                     static_cast<Card>(hidden.board[0]), static_cast<Card>(hidden.board[1]),
                     static_cast<Card>(hidden.board[2]), static_cast<Card>(hidden.board[3]),
                     static_cast<Card>(hidden.board[4])};
        const int s0 = evaluate7(a);
        const int s1 = evaluate7(b);
        const double win = (s0 > s1) ? 1.0 : ((s0 < s1) ? 0.0 : 0.5);
        acc += w * win;
        wSum += w;
    }
    if (wSum <= 0.0) return 0.0f;
    return float(acc / wSum);
}

}  // namespace

Move proAgentMove(const HUState& s, Rng& rng) {
    LegalMove mv[kNumActions];
    const int n = legalMoves(s, mv);
    if (n == 0) return Move{};

    const int me = s.toAct;
    const int need = toCall(s, me);
    const int pot = s.pot();
    const float eq = equityVsRandom(s, me, 32);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    const float jitter = u(rng);

    auto checkCall = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotCheckCall) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };
    auto fold = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotFold) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };
    auto betPotFraction = [&](double frac) {
        const int inc = std::max<int>(kBigBlind, static_cast<int>(std::llround(frac * double(pot))));
        return int(s.contrib[me]) + inc;
    };

    Move out{};
    if (s.street == ST_PREFLOP) {
        if (need == 0) {
            // 小盲开场：可玩牌直接开 3BB，不再用底池比例（翻前底池只有 1.5BB，
            // 比例算出来的"加注"会被压成最小加注）。
            if (eq > 0.52f && jitter > 0.15f && isRaisePossible(s)) {
                out = makeRaise(s, 3 * kBigBlind);
                if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
            }
            checkCall(out);
            return out;
        }
        const float odds = float(need) / float(pot + need);
        if (eq > 0.78f && jitter > 0.35f && isRaisePossible(s)) {
            out = makeRaise(s, 10 * kBigBlind);
            if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
        }
        if (eq > odds + 0.08f) {
            checkCall(out);
            return out;
        }
        if (eq < odds * 0.90f) {
            if (fold(out)) return out;
        }
        checkCall(out);
        return out;
    }

    if (need == 0) {
        if (eq > 0.72f && jitter > 0.15f) {
            out = makeRaise(s, betPotFraction(eq > 0.88f ? 1.0 : 0.66));
            if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
        } else if (eq > 0.52f && jitter > 0.45f) {
            out = makeRaise(s, betPotFraction(0.5));
            if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
        } else if (eq < 0.30f && jitter < 0.12f) {
            out = makeRaise(s, betPotFraction(0.5));
            if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
        }
        checkCall(out);
        return out;
    }

    const float odds = float(need) / float(pot + need);
    if (eq > 0.82f && jitter > 0.35f) {
        out = makeRaise(s, betPotFraction(0.8));
        if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
    }
    if (eq > odds + 0.12f) {
        checkCall(out);
        return out;
    }
    if (eq < odds * 0.85f) {
        if (fold(out)) return out;
    }
    checkCall(out);
    return out;
}

// 跟注型"稳健"机器人。外部日志（169 手明细）显示 Slumbot 作为 BB 有 70%
// 的决策是 check/call、面对开牌 65% 跟注，而我们联赛里的脚本对手大多
// "被加注就弃"。缺了"被跟注"这层压力，BB 会学成 3-bet/弃牌 + 翻牌全下注。
// 这个机器人的目标不是强，而是**提供正确的对抗方向**：宽跟注、少诈唬、
// 只用强牌加注。
Move solidAgentMove(const HUState& s, Rng& rng) {
    LegalMove mv[kNumActions];
    const int n = legalMoves(s, mv);
    if (n == 0) return Move{};

    const int me = s.toAct;
    const int need = toCall(s, me);
    const int pot = s.pot();
    const float eq = equityVsRandom(s, me, 24);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    const float j = (u(rng) - 0.5f) * 0.05f;  // ±2.5% 抖动，避免完全确定

    auto checkCall = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotCheckCall) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };
    auto fold = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotFold) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };

    Move out{};
    if (s.street == ST_PREFLOP) {
        if (need == 0) {
            // 大盲平进：只有很强的牌才主动加注。
            if (eq > 0.72f + j && isRaisePossible(s)) {
                out = makeRaise(s, 3 * kBigBlind);
                if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
            }
            checkCall(out);
            return out;
        }
        if (s.numRaises <= 1) {
            // 面对开牌：弃 14% / 跟 65% / 3-bet 20% 是 Slumbot 的实测频率。
            if (eq > 0.60f + j && isRaisePossible(s)) {
                out = makeRaise(s, int(s.lastRaiseTo) * 3);
                if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
            }
            if (eq > 0.44f + j) {
                checkCall(out);
                return out;
            }
            if (fold(out)) return out;
            checkCall(out);
            return out;
        }
        // 面对 3-bet 及以后：46.8% 弃 / 44.7% 跟 / 8.5% 4-bet。
        if (eq > 0.72f + j && isRaisePossible(s)) {
            out = makeRaise(s, int(s.lastRaiseTo) * 2);
            if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
        }
        if (eq > 0.55f + j) {
            checkCall(out);
            return out;
        }
        if (fold(out)) return out;
        checkCall(out);
        return out;
    }

    // 翻后：过牌为主，只有强牌下注；面对下注时用赔率 + 宽跟注。
    if (need == 0) {
        if (eq > 0.78f + j && isRaisePossible(s)) {
            out = makeRaise(s, int(s.contrib[me]) + std::max<int>(kBigBlind, int(0.6 * pot)));
            if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
        }
        checkCall(out);
        return out;
    }
    const float odds = float(need) / float(pot + need);
    if (eq > 0.88f + j && isRaisePossible(s)) {
        out = makeRaise(s, int(s.contrib[me]) + std::max<int>(kBigBlind, int(0.9 * pot)));
        if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
    }
    if (eq > odds * 1.02f) {
        checkCall(out);
        return out;
    }
    if (fold(out)) return out;
    checkCall(out);
    return out;
}

// 「读范围」机器人。和 solid 的唯一结构性区别：面对下注时用条件胜率
// `equityVsPolarizedRange`（假设我们的范围 = 当前牌面最强的 keep%，外加
// airShare 份空气）而不是原始胜率；keep 随下注尺寸收紧。它的作用不是"更强"，
// 而是提供 README #107(d) 缺的那层压力——我们的薄价值下注与宽跟注会被它按
// 范围惩罚，而其余脚本（包括 solid 的宽跟、shark 的两对+建池）都只在某个
// 方向施压。
Move rangerAgentMove(const HUState& s, Rng& rng, float airShare) {
    LegalMove mv[kNumActions];
    const int n = legalMoves(s, mv);
    if (n == 0) return Move{};

    const int me = s.toAct;
    const int need = toCall(s, me);
    const int pot = s.pot();
    const float eqRaw = equityVsRandom(s, me, 16);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    const float j = (u(rng) - 0.5f) * 0.04f;  // ±2% 抖动，避免完全确定

    auto checkCall = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotCheckCall) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };
    auto fold = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotFold) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };

    Move out{};
    if (s.street == ST_PREFLOP) {
        if (need == 0) {
            if (eqRaw > 0.80f + j && isRaisePossible(s)) {
                out = makeRaise(s, 3 * kBigBlind);
                if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
            }
            checkCall(out);
            return out;
        }
        if (s.numRaises <= 1) {
            if (eqRaw > 0.68f + j && isRaisePossible(s)) {
                out = makeRaise(s, int(s.lastRaiseTo) * 3);
                if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
            }
            if (eqRaw > 0.50f + j) {
                checkCall(out);
                return out;
            }
            if (fold(out)) return out;
            checkCall(out);
            return out;
        }
        if (eqRaw > 0.80f + j && isRaisePossible(s)) {
            out = makeRaise(s, int(s.lastRaiseTo) * 2);
            if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
        }
        if (eqRaw > 0.62f + j) {
            checkCall(out);
            return out;
        }
        if (fold(out)) return out;
        checkCall(out);
        return out;
    }

    if (need == 0) {
        // 无人下注：对"跟注范围 = 最强 45% + 28% 空气"的条件胜率决定价值下注，
        // 另加 28% 纯诈唬（原始胜率很低时才诈唬），尺寸 0.6 池。
        // 诈唬份额必须与假设范围里的空气比例同量级：只拿强牌开火会让对手
        // （学习座位）过度弃牌，学不到"防守要有抓诈唬"（README #108 的 rg1）。
        const float eqValue = equityVsPolarizedRange(s, me, 24, 0.45f, airShare);
        if (eqValue > 0.72f + j && isRaisePossible(s)) {
            out = makeRaise(s, int(s.contrib[me]) + std::max<int>(kBigBlind, int(0.6 * pot)));
            if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
        }
        if (eqRaw < 0.30f && u(rng) < 0.28f && isRaisePossible(s)) {
            out = makeRaise(s, int(s.contrib[me]) + std::max<int>(kBigBlind, int(0.6 * pot)));
            if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
        }
        checkCall(out);
        return out;
    }

    // 面对下注：尺寸越大，假设我们的范围越强（keep 越小）；范围里始终留
    // 28% 的空气，否则跟注频率会过低、诈唬自动赚钱。
    const float odds = float(need) / float(pot + need);
    const float keep = std::min(0.60f, std::max(0.10f, 0.72f - 1.4f * odds));
    const float eq = equityVsPolarizedRange(s, me, 32, keep, airShare);
    if (eq > 0.85f + j && isRaisePossible(s)) {
        out = makeRaise(s, int(s.contrib[me]) + std::max<int>(kBigBlind, int(0.9 * pot)));
        if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
    }
    if (eq > odds * 1.05f) {
        checkCall(out);
        return out;
    }
    if (fold(out)) return out;
    checkCall(out);
    return out;
}

// 「紧全下」机器人。外部明细（1360 手，allin_adjusted 口径）显示我们的
// 赤字全部来自全下决策：翻前用 K3s/A7o/A8s/K9o 接 200BB 全下、转牌用
// 中等两对推、河牌用一对跟大注；非全下手已经打平。联赛里爱全下诈唬的
// 脚本对手把策略教成了"宽接"，这里提供反向压力：全下几乎总是顶范围。
// 目标不是强，而是让策略学到"面对全下要收紧"。
Move tightAgentMove(const HUState& s, Rng& rng) {
    LegalMove mv[kNumActions];
    const int n = legalMoves(s, mv);
    if (n == 0) return Move{};

    const int me = s.toAct;
    const int need = toCall(s, me);
    const int pot = s.pot();
    const float eq = equityVsRandom(s, me, 32);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    const float j = (u(rng) - 0.5f) * 0.03f;  // ±1.5% 抖动

    auto checkCall = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotCheckCall) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };
    auto fold = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotFold) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };
    auto raiseTo = [&](int to, Move& out) {
        if (!isRaisePossible(s)) return false;
        out = makeRaise(s, to);
        return out.kind == MK_RAISE || out.kind == MK_ALLIN;
    };

    // 面对全下：跟注会打光剩余筹码，或对手已经全下。
    const bool facingAllin = need > 0 && (s.allin[1 - me] != 0 || need >= s.stack[me]);

    Move out{};
    if (s.street == ST_PREFLOP) {
        if (facingAllin) {
            // 只有顶范围（对随机手牌胜率 > 0.68）才接。
            if (eq > 0.68f + j) {
                checkCall(out);
                return out;
            }
            if (fold(out)) return out;
            checkCall(out);
            return out;
        }
        if (need == 0) {
            if (eq > 0.80f + j && raiseTo(3 * kBigBlind, out)) return out;
            checkCall(out);
            return out;
        }
        if (eq > 0.80f + j && raiseTo(int(s.lastRaiseTo) * 3, out)) return out;
        if (eq > 0.45f + j) {
            checkCall(out);
            return out;
        }
        if (fold(out)) return out;
        checkCall(out);
        return out;
    }

    // 翻后：面对全下要求成牌够强；主动下注只用顶范围。
    if (facingAllin) {
        if (eq > 0.72f + j) {
            checkCall(out);
            return out;
        }
        if (fold(out)) return out;
        checkCall(out);
        return out;
    }
    if (need == 0) {
        if (eq > 0.85f + j &&
            raiseTo(int(s.contrib[me]) + std::max<int>(kBigBlind, int(0.75 * pot)), out)) {
            return out;
        }
        checkCall(out);
        return out;
    }
    const float odds = float(need) / float(pot + need);
    if (eq > 0.90f + j &&
        raiseTo(int(s.contrib[me]) + std::max<int>(kBigBlind, int(0.9 * pot)), out)) {
        return out;
    }
    if (eq > std::max(0.55f, odds + 0.18f) + j) {
        checkCall(out);
        return out;
    }
    if (fold(out)) return out;
    checkCall(out);
    return out;
}

// 「陷阱」机器人：强牌在前几条街只跟注/过牌，河牌才下注；从不弃顶范围。
// 外部路径分析（tight1 的 20 个翻后全下）显示：我们的 4-bet 常用 pct
// 0.43~0.66 的牌把底池建到 200BB，再被顶范围接住；这个对手专门"接住"
// 这些牌，逼策略学会控池。它不强，但提供正确的对抗方向。
Move trapperAgentMove(const HUState& s, Rng& rng) {
    LegalMove mv[kNumActions];
    const int n = legalMoves(s, mv);
    if (n == 0) return Move{};

    const int me = s.toAct;
    const int need = toCall(s, me);
    const int pot = s.pot();
    const float eq = equityVsRandom(s, me, 32);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    const float j = (u(rng) - 0.5f) * 0.03f;

    auto checkCall = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotCheckCall) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };
    auto fold = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotFold) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };
    auto raiseTo = [&](int to, Move& out) {
        if (!isRaisePossible(s)) return false;
        out = makeRaise(s, to);
        return out.kind == MK_RAISE || out.kind == MK_ALLIN;
    };

    Move out{};
    if (s.street == ST_PREFLOP) {
        if (need == 0) {
            checkCall(out);  // 从不主动加注：纯陷阱
            return out;
        }
        if (eq > 0.80f + j) {
            checkCall(out);  // AA/KK 也只跟，不 3-bet
            return out;
        }
        const bool facingAllin = s.allin[1 - me] != 0 || need >= s.stack[me];
        if (facingAllin) {
            if (eq > 0.62f + j) {
                checkCall(out);
                return out;
            }
            if (fold(out)) return out;
            checkCall(out);
            return out;
        }
        if (eq > 0.42f + j) {
            checkCall(out);
            return out;
        }
        if (fold(out)) return out;
        checkCall(out);
        return out;
    }

    // 翻后：强牌先过牌/跟注（陷阱），河牌才主动下注。
    if (need == 0) {
        if (s.street == ST_RIVER && eq > 0.85f + j &&
            raiseTo(int(s.contrib[me]) + int(1.2 * pot), out)) {
            return out;
        }
        checkCall(out);
        return out;
    }
    if (s.street == ST_RIVER && eq > 0.93f + j &&
        raiseTo(int(s.contrib[me]) + int(0.9 * pot), out)) {
        return out;
    }
    if (eq > 0.72f + j) {
        checkCall(out);
        return out;
    }
    const float odds = float(need) / float(pot + need);
    if (eq > std::max(0.35f, odds + 0.12f) + j) {
        checkCall(out);
        return out;
    }
    if (fold(out)) return out;
    checkCall(out);
    return out;
}

// 「价值型强对手」机器人：翻前紧凶（只用强牌 3-bet/4-bet，从不 5-bet 诈唬），
// 翻后只用两对及以上的成牌建池/加注，弱牌面对大注直接弃。外部 14.3k 手复盘
// （trap1x）：59 个全下手贡献 −0.32 BB/手，非全下只有 −0.01；逐手明细显示
// 都是我们自己用中等牌把底池建到 200BB（A8o 5-bet、J4s check-raise 全下、
// 一对 10 打光）。现有脚本对手都不会"用强牌惩罚加注"：solid 宽跟但不下注、
// trapper 只在河牌发力、spammer 被加注就弃。这个对手提供这个方向。
Move sharkAgentMove(const HUState& s, Rng& rng) {
    LegalMove mv[kNumActions];
    const int n = legalMoves(s, mv);
    if (n == 0) return Move{};

    const int me = s.toAct;
    const int need = toCall(s, me);
    const int pot = s.pot();
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    const float j = (u(rng) - 0.5f) * 0.04f;  // ±2% 抖动，避免完全确定

    auto checkCall = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotCheckCall) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };
    auto fold = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotFold) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };
    auto raiseTo = [&](int to, Move& out) {
        if (!isRaisePossible(s)) return false;
        out = makeRaise(s, to);
        return out.kind == MK_RAISE || out.kind == MK_ALLIN;
    };

    Move out{};
    if (s.street == ST_PREFLOP) {
        const float eq = float(StrengthModel::instance().preflopEquity(
            static_cast<Card>(s.hole[me][0]), static_cast<Card>(s.hole[me][1])));
        if (need == 0) {
            if (eq > 0.62f + j && raiseTo(3 * kBigBlind, out)) return out;
            checkCall(out);
            return out;
        }
        if (s.numRaises <= 1) {
            // 面对开牌：只用强牌 3-bet；中等牌跟，弱牌弃。
            if (eq > 0.80f + j && raiseTo(int(s.lastRaiseTo) * 3, out)) return out;
            if (eq > 0.44f + j) {
                checkCall(out);
                return out;
            }
            if (fold(out)) return out;
            checkCall(out);
            return out;
        }
        if (s.numRaises == 2) {
            // 面对 3-bet：只有 AA/KK/AKs 级别才 4-bet；TT+/AQ+ 跟，其余弃。
            if (eq > 0.90f + j && raiseTo(int(s.lastRaiseTo) * 5 / 2, out)) return out;
            if (eq > 0.74f + j) {
                checkCall(out);
                return out;
            }
            if (fold(out)) return out;
            checkCall(out);
            return out;
        }
        // 面对 5-bet 及以上：只用顶范围跟，绝不 6-bet 诈唬。
        if (eq > 0.84f + j) {
            checkCall(out);
            return out;
        }
        if (fold(out)) return out;
        checkCall(out);
        return out;
    }

    Card hole[2] = {static_cast<Card>(s.hole[me][0]), static_cast<Card>(s.hole[me][1])};
    Card board[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < s.boardN && i < 5; ++i) board[i] = static_cast<Card>(s.board[i]);
    Card all[7];
    int nAll = 0;
    all[nAll++] = hole[0];
    all[nAll++] = hole[1];
    for (int i = 0; i < s.boardN && i < 5; ++i) all[nAll++] = board[i];
    const int cat = handCategoryOf(bestHandScore(all, nAll));
    const double pct = StrengthModel::instance().postflopPercentile(hole, board, s.boardN);
    const DrawOuts draws = StrengthModel::computeOutsWithCat(hole, board, s.boardN, cat);

    // 无人下注：只用成牌价值下注，强听牌偶尔半诈唬，其余过牌（不空气开火）。
    if (need == 0) {
        if (cat >= 2 && pct > 0.80 &&
            raiseTo(int(s.contrib[me]) + static_cast<int>(0.7 * pot), out)) {
            return out;
        }
        if (cat == 1 && pct > 0.92 &&
            raiseTo(int(s.contrib[me]) + static_cast<int>(0.6 * pot), out)) {
            return out;
        }
        if (draws.total >= 9 && u(rng) < 0.30f &&
            raiseTo(int(s.contrib[me]) + static_cast<int>(0.5 * pot), out)) {
            return out;
        }
        checkCall(out);
        return out;
    }

    // 面对下注：两对+ 建池/接住，弱牌按赔率+牌力门限决定，不裸跟大注。
    const float odds = float(need) / float(pot + need);
    const bool monster = (cat >= 3) || (cat == 2 && pct > 0.90);
    if (monster) {
        if (u(rng) < 0.60f &&
            raiseTo(int(s.contrib[me]) + static_cast<int>(0.9 * pot), out)) {
            return out;
        }
        checkCall(out);  // 永不弃顶范围
        return out;
    }
    if (cat == 2) {
        if (pct > 0.80 || odds <= 0.30f) {
            checkCall(out);
            return out;
        }
        if (fold(out)) return out;
        checkCall(out);
        return out;
    }
    if (cat == 1) {
        if (pct > 0.86 || odds <= 0.20f) {
            checkCall(out);
            return out;
        }
        if (fold(out)) return out;
        checkCall(out);
        return out;
    }
    if (draws.total >= 8 && odds <= 0.28f) {
        checkCall(out);
        return out;
    }
    if (fold(out)) return out;
    checkCall(out);
    return out;
}

// 「小注骚扰」机器人：翻后只要没人下注就高频打 1/3 池（不看牌力），
// 面对加注就弃。目的是给策略"必须防守小注"的压力：外部分析显示 BB 在
// 300-600 池面对下注弃了 151 次，弃牌时胜率中位 0.414、赔率 0.250。
Move spammerAgentMove(const HUState& s, Rng& rng) {
    LegalMove mv[kNumActions];
    const int n = legalMoves(s, mv);
    if (n == 0) return Move{};

    const int me = s.toAct;
    const int need = toCall(s, me);
    const int pot = s.pot();
    const float eq = equityVsRandom(s, me, 24);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);

    auto checkCall = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotCheckCall) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };
    auto fold = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotFold) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };

    Move out{};
    if (s.street == ST_PREFLOP) {
        if (need == 0) {
            if (u(rng) < 0.30 && isRaisePossible(s)) {
                out = makeRaise(s, 3 * kBigBlind);
                if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
            }
            checkCall(out);
            return out;
        }
        const float odds = float(need) / float(pot + need);
        if (eq > std::max(0.40f, odds + 0.02f)) {
            checkCall(out);
            return out;
        }
        if (fold(out)) return out;
        checkCall(out);
        return out;
    }

    if (need == 0) {
        // 65% 的频率打 1/3 池，和牌力无关。
        if (u(rng) < 0.65 && isRaisePossible(s)) {
            out = makeRaise(s, int(s.contrib[me]) + std::max<int>(kBigBlind, int(0.33 * pot)));
            if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
        }
        checkCall(out);
        return out;
    }
    // 面对下注：宽跟注（赔率 + 2% 就够），只有很弱的牌才弃。
    const float odds = float(need) / float(pot + need);
    if (eq > std::max(0.28f, odds + 0.02f)) {
        checkCall(out);
        return out;
    }
    if (fold(out)) return out;
    checkCall(out);
    return out;
}

// 「按尺寸弃牌」机器人：把外部测到的 Slumbot 行为（面对下注的弃牌率随
// 下注额/底池上升：0.3x≈32%、0.8x≈52%、1.3x≈72%）变成联赛压力。
// 它翻牌后从不主动下注——把"要不要用大尺寸诈唬"完全留给被测策略；
// 自己只做两件事：按尺寸弃牌 / 强牌偶尔加注。这样策略必须学会用尺寸
// 剥削弃牌率，而不是靠部署时乘一个系数。
Move folderAgentMove(const HUState& s, Rng& rng) {
    LegalMove mv[kNumActions];
    const int n = legalMoves(s, mv);
    if (n == 0) return Move{};

    const int me = s.toAct;
    const int need = toCall(s, me);
    const int pot = s.pot();
    const float eq = equityVsRandom(s, me, 24);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);

    auto checkCall = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotCheckCall) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };
    auto fold = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotFold) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };

    Move out{};
    if (s.street == ST_PREFLOP) {
        if (need == 0) {
            if (eq > 0.78f && isRaisePossible(s) && u(rng) < 0.25f) {
                out = makeRaise(s, 3 * kBigBlind);
                if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
            }
            checkCall(out);
            return out;
        }
        const float ratio = float(need) / float(std::max(1, pot - need));
        if (eq < 0.55f - 0.20f * ratio && u(rng) < 0.5f) {
            if (fold(out)) return out;
        }
        if (eq > 0.70f && isRaisePossible(s) && u(rng) < 0.15f) {
            out = makeRaise(s, int(s.lastRaiseTo) * 2);
            if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
        }
        checkCall(out);
        return out;
    }

    if (need == 0) {
        checkCall(out);  // 翻牌后从不主动下注
        return out;
    }

    // 面对下注：弃牌率随尺寸上升，强牌显著少弃。
    const float b = float(need) / float(std::max(1, pot - need));
    float pFold = std::min(0.80f, 0.20f + 0.40f * b);
    pFold *= (1.0f - 2.0f * std::max(0.0f, eq - 0.55f));
    pFold = std::max(0.03f, pFold);
    if (u(rng) < pFold) {
        if (fold(out)) return out;
    }
    if (eq > 0.85f && isRaisePossible(s) && u(rng) < 0.10f) {
        out = makeRaise(s, int(s.contrib[me]) + std::max<int>(kBigBlind, int(0.8 * pot)));
        if (out.kind == MK_RAISE || out.kind == MK_ALLIN) return out;
    }
    checkCall(out);
    return out;
}

// 「过度下注」机器人：翻后不看牌力、用 1.5~3 倍池的超大尺寸开火（含诈唬），
// 面对下注也会用超大加注施压；合理价格下宽跟，不弃强牌。
//
// 加它的原因是外部逐决策复盘（rng1g 2153 手，`--dump-big calloff|fold`）暴露了
// 两类灾难，方向正好相反：
//   - 河牌把 333QQ（葫芦）弃给一次 4 倍加注：eq 0.977 vs 赔率 0.364；
//   - 河牌用 4.5% 胜率的牌打 8.6 倍池超池诈唬，被跟注后打光。
// 前者说明网络把"大加注"先验当成坚果（没见过大尺寸诈唬），后者说明它不
// 相信对手会宽跟。现有脚本对手都不提供这两个信号：spammer 只打 1/3 池、
// 被加注就弃；solid 宽跟但从不主动开火；folder 从不主动下注。
Move maniacAgentMove(const HUState& s, Rng& rng) {
    LegalMove mv[kNumActions];
    const int n = legalMoves(s, mv);
    if (n == 0) return Move{};

    const int me = s.toAct;
    const int need = toCall(s, me);
    const int pot = s.pot();
    const float eq = equityVsRandom(s, me, 24);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);

    auto checkCall = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotCheckCall) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };
    auto fold = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotFold) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };
    auto raiseTo = [&](int to, Move& out) {
        if (!isRaisePossible(s)) return false;
        out = makeRaise(s, to);
        return out.kind == MK_RAISE || out.kind == MK_ALLIN;
    };
    // 超过剩余筹码时 makeRaise 自动钳成全下。
    const int myContrib = int(s.contrib[me]);

    Move out{};
    if (s.street == ST_PREFLOP) {
        if (need == 0) {
            if (u(rng) < 0.50f && raiseTo(4 * kBigBlind, out)) return out;
            checkCall(out);
            return out;
        }
        // 面对加注：45% 反加（3 倍上次加注），10% 直接全下，其余宽跟。
        if (u(rng) < 0.45f && raiseTo(int(s.lastRaiseTo) * 3, out)) return out;
        if (u(rng) < 0.10f && raiseTo(myContrib + s.stack[me], out)) return out;
        const float odds = float(need) / float(pot + need);
        if (eq > odds - 0.10f) {
            checkCall(out);
            return out;
        }
        if (fold(out)) return out;
        checkCall(out);
        return out;
    }

    if (need == 0) {
        // 两极化：只用强牌（eq>0.62）或空气（eq<0.32）打 1.5~3 倍池，中等牌
        // 过牌。刻意**不**让"大尺寸"单向地等于弱牌，否则策略会学成"见大注
        // 就宽跟"，对 Slumbot 这种大注=价值的对手反而更糟。7% 直接全下。
        const bool polar = (eq > 0.62f) || (eq < 0.32f);
        if (polar && u(rng) < 0.07f && raiseTo(myContrib + s.stack[me], out)) return out;
        if (polar && u(rng) < 0.66f) {
            const float mult = 1.5f + 1.5f * u(rng);
            const int to = myContrib + std::max<int>(kBigBlind, int(mult * pot));
            if (raiseTo(to, out)) return out;
        }
        checkCall(out);
        return out;
    }

    // 面对下注：40% 大加注（强牌价值 / 空气诈唬各半），其余按赔率宽跟。
    if ((eq > 0.72f || eq < 0.25f) && u(rng) < 0.40f) {
        const int to = myContrib + std::max<int>(kBigBlind, int(2.0f * pot));
        if (raiseTo(to, out)) return out;
    }
    const float odds = float(need) / float(pot + need);
    if (eq > odds - 0.06f) {
        checkCall(out);
        return out;
    }
    if (fold(out)) return out;
    checkCall(out);
    return out;
}

void collectRollout(const ModelSet& latest, const HistoricalPool* pool, const RolloutConfig& cfg,
                    std::array<std::vector<Transition>, 2>& out, RolloutStats& stats) {
    torch::NoGradGuard noGrad;  // 采样只做前向，不需要建图
    Rng rng(cfg.seed);
    const int N = std::max(1, cfg.games);
    std::vector<GameCtx> gs(static_cast<size_t>(N));
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::uniform_real_distribution<double> uu(0.0, 1.0);
    // rangeEq 的采样/枚举用独立随机流：打开 rangeEq 不能改变牌局发牌与动作
    // 采样的随机流，否则和基线不可比。
    Rng rangeRng(cfg.seed ^ 0x9E3779B97F4A7C15ULL);

    // 记录"某座位最近一次决策"的动作前状态与动作槽。rangeEq 与范围条件化
    // 搜索都要从对手视角重建观测（换入采样底牌），所以必须留下动作前状态。
    auto recordAction = [&](GameCtx& g, int s, const Move& mv) {
        if (cfg.rangeEqSamples <= 0 && cfg.rangeSearchSamples <= 0 &&
            cfg.searchOppProb <= 0.0f && cfg.expertProb <= 0.0f && cfg.riverCfrIters <= 0) {
            return;
        }
        RangeObs& la = g.lastAct[s];
        la.pre = g.st;
        la.slot = slotOfMove(g.st, mv);
        la.valid = (la.slot >= 0);
    };
    auto play = [&](GameCtx& g, int s, const Move& mv) {
        recordAction(g, s, mv);
        applyMove(g.st, mv, rng);
    };

    out[0].clear();
    out[1].clear();

    // ---------------------------------------------------------------- 联赛
    //
    // 按**局**分配对手，而不是按座位各自独立抽。两个理由：
    //
    // 1. 独立抽会浪费算力。机器人 0.5 / 快照 0.3 时，约 (0.5)^2 + (0.3)^2 ≈ 34%
    //    的牌局两边都不是当前策略，整局跑完一条样本都采不到。
    // 2. 独立抽没法控制配比。要让"每局都有一个学习座位"，就只能按局指定。
    //
    // 所以每局先抽一类对手和一个座位，另一个座位固定是 ROLE_LATEST。
    // 座位左右各半，位置优势自然不会偏。
    const bool hasPool = pool != nullptr && !pool->empty();
    float wHist = hasPool ? std::max(0.0f, cfg.historicalProb) : 0.0f;
    const float wRule = std::max(0.0f, cfg.ruleProb);
    const float wRand = std::max(0.0f, cfg.randomProb);
    const float wPro = std::max(0.0f, cfg.proProb);
    const float wSearch = std::max(0.0f, cfg.searchProb);
    const float wSolid = std::max(0.0f, cfg.solidProb);
    const float wTight = std::max(0.0f, cfg.tightProb);
    const float wTrapper = std::max(0.0f, cfg.trapperProb);
    const float wShark = std::max(0.0f, cfg.sharkProb);
    const float wSpammer = std::max(0.0f, cfg.spammerProb);
    const float wManiac = std::max(0.0f, cfg.maniacProb);
    const float wRanger = std::max(0.0f, cfg.rangerProb);
    const float wCfr = (cfg.oppCfrIters > 0) ? std::max(0.0f, cfg.oppCfrProb) : 0.0f;
    const float wFolder = std::max(0.0f, cfg.folderProb);
    const float wSharp = std::max(0.0f, cfg.selfSharpProb);
    const float wSROpp = std::max(0.0f, cfg.searchOppProb);
    const bool hasExtra =
        cfg.mixedFrozen &&
        (cfg.extraOpponentCount > 0 || cfg.fixedOpponent[0] != nullptr ||
         cfg.fixedOpponent[1] != nullptr);
    const float wExtra = hasExtra ? std::max(0.0f, cfg.extraOpponentProb) : 0.0f;
    const float wSelf = std::max(
        0.0f, 1.0f - (wHist + wRule + wRand + wPro + wSearch + wSolid + wTight + wTrapper +
                      wShark + wSpammer + wManiac + wRanger + wCfr + wFolder + wSharp + wSROpp +
                      wExtra));
    const float wSum = wSelf + wHist + wRule + wRand + wPro + wSearch + wSolid + wTight +
                       wTrapper + wShark + wSpammer + wManiac + wRanger + wCfr + wFolder + wSharp +
                       wSROpp + wExtra;

    // 评估路径：强制某个座位是固定对手，另一个座位就是被测策略。
    int forcedSeat = -1;
    uint8_t forcedRole = ROLE_LATEST;
    if (cfg.forceActor != nullptr && cfg.forceActorSeat >= 0) {
        forcedSeat = cfg.forceActorSeat;
        forcedRole = ROLE_FIXED;
    } else if (cfg.forceRuleSeat >= 0) {
        forcedSeat = cfg.forceRuleSeat;
        forcedRole = ROLE_RULE;
    } else if (cfg.forceRandomSeat >= 0) {
        forcedSeat = cfg.forceRandomSeat;
        forcedRole = ROLE_RANDOM;
    } else if (cfg.forceProSeat >= 0) {
        forcedSeat = cfg.forceProSeat;
        forcedRole = ROLE_PRO;
    } else if (cfg.forceSearchSeat >= 0) {
        forcedSeat = cfg.forceSearchSeat;
        forcedRole = ROLE_SEARCH;
    } else if (cfg.forceSolidSeat >= 0) {
        forcedSeat = cfg.forceSolidSeat;
        forcedRole = ROLE_SOLID;
    } else if (cfg.forceRangerSeat >= 0) {
        forcedSeat = cfg.forceRangerSeat;
        forcedRole = ROLE_RANGER;
    } else if (cfg.forceCfrSeat >= 0) {
        forcedSeat = cfg.forceCfrSeat;
        forcedRole = ROLE_CFR;
    }

    for (int i = 0; i < N; ++i) {
        GameCtx& g = gs[static_cast<size_t>(i)];
        g.index = i;
        if (cfg.fixedDeals != nullptr && i < static_cast<int>(cfg.fixedDeals->size())) {
            g.st = (*cfg.fixedDeals)[static_cast<size_t>(i)];
        } else if (cfg.duplicateDeals && (i & 1) != 0 && i > 0) {
            // 同牌对调：上一局在 setup 阶段的初始状态还没被修改，复制过来
            // 只交换两人底牌，位置/盲注/筹码都不变。
            g.st = gs[static_cast<size_t>(i - 1)].st;
            for (int k = 0; k < 2; ++k) {
                const uint8_t tmp = g.st.hole[0][k];
                g.st.hole[0][k] = g.st.hole[1][k];
                g.st.hole[1][k] = tmp;
            }
        } else {
            g.st = dealNewHand(rng);
        }

        if (cfg.initialDeals != nullptr) cfg.initialDeals->push_back(g.st);

        int oppSeat = -1;
        uint8_t oppRole = ROLE_LATEST;
        int oppTier = 0;
        int oppIdx = -1;

        if ((cfg.fixedOpponent[0] != nullptr || cfg.fixedOpponent[1] != nullptr) &&
            !cfg.mixedFrozen) {
            // Best-response 训练：每局随机选一个座位放冻结对手。
            oppSeat = static_cast<int>(rng() & 1u);
            oppRole = ROLE_FIXED;
        } else if (forcedSeat >= 0) {
            oppSeat = forcedSeat;
            oppRole = forcedRole;
        } else if (wSum > 0.0f) {
            const float r = u(rng) * wSum;
            if (r < wHist) {
                oppRole = ROLE_SNAPSHOT;
                // 抽哪个快照：archive 占 archiveProb。recent 管近期战术，
                // archive 管远期记忆，两者都要有。
                const bool wantArchive =
                    !pool->archive.empty() &&
                    (pool->recent.empty() || u(rng) < cfg.archiveProb);
                const std::vector<std::shared_ptr<Snapshot>>& v =
                    wantArchive ? pool->archive : pool->recent;
                oppTier = wantArchive ? 1 : 0;
                oppIdx = static_cast<int>(rng() % v.size());
            } else if (r < wHist + wRule) {
                oppRole = ROLE_RULE;
            } else if (r < wHist + wRule + wRand) {
                oppRole = ROLE_RANDOM;
            } else if (r < wHist + wRule + wRand + wPro) {
                oppRole = ROLE_PRO;
            } else if (r < wHist + wRule + wRand + wPro + wSearch) {
                oppRole = ROLE_SEARCH;
            } else if (r < wHist + wRule + wRand + wPro + wSearch + wSolid) {
                oppRole = ROLE_SOLID;
            } else if (r < wHist + wRule + wRand + wPro + wSearch + wSolid + wTight) {
                oppRole = ROLE_TIGHT;
            } else if (r < wHist + wRule + wRand + wPro + wSearch + wSolid + wTight + wTrapper) {
                oppRole = ROLE_TRAPPER;
            } else if (r < wHist + wRule + wRand + wPro + wSearch + wSolid + wTight + wTrapper +
                           wShark) {
                oppRole = ROLE_SHARK;
            } else if (r < wHist + wRule + wRand + wPro + wSearch + wSolid + wTight + wTrapper +
                           wShark + wSpammer) {
                oppRole = ROLE_SPAMMER;
            } else if (r < wHist + wRule + wRand + wPro + wSearch + wSolid + wTight + wTrapper +
                           wShark + wSpammer + wManiac) {
                oppRole = ROLE_MANIAC;
            } else if (r < wHist + wRule + wRand + wPro + wSearch + wSolid + wTight +
                           wTrapper + wShark + wSpammer + wManiac + wRanger) {
                oppRole = ROLE_RANGER;
            } else if (r < wHist + wRule + wRand + wPro + wSearch + wSolid + wTight +
                           wTrapper + wShark + wSpammer + wManiac + wRanger + wCfr) {
                oppRole = ROLE_CFR;
            } else if (r < wHist + wRule + wRand + wPro + wSearch + wSolid + wTight +
                           wTrapper + wShark + wSpammer + wManiac + wRanger + wCfr + wFolder) {
                oppRole = ROLE_FOLDER;
            } else if (r < wHist + wRule + wRand + wPro + wSearch + wSolid + wTight +
                           wTrapper + wShark + wSpammer + wManiac + wRanger + wCfr + wFolder + wSharp) {
                oppRole = ROLE_LATEST_SHARP;
            } else if (r < wHist + wRule + wRand + wPro + wSearch + wSolid + wTight +
                           wTrapper + wShark + wSpammer + wManiac + wRanger + wCfr + wFolder + wSharp +
                           wSROpp) {
                oppRole = ROLE_SEARCH_OPP;
            } else if (r < wHist + wRule + wRand + wPro + wSearch + wSolid + wTight +
                           wTrapper + wShark + wSpammer + wManiac + wRanger + wCfr + wFolder + wSharp +
                           wSROpp + wExtra) {
                oppRole = ROLE_FIXED;
                if (cfg.extraOpponentCount > 1) {
                    g.extraIdx = static_cast<uint8_t>(rng() % cfg.extraOpponentCount);
                }
            }
            if (oppRole != ROLE_LATEST) {
                // 默认 0.5 时保持原来的 rng 流（逐位对齐旧实验）；偏置到 BB
                // 时才改用浮点抽样。
                if (cfg.learnerBBProb == 0.5f) {
                    oppSeat = static_cast<int>(rng() & 1u);
                } else {
                    oppSeat = (u(rng) < cfg.learnerBBProb) ? 0 : 1;
                }
            }
        }

        for (int seat = 0; seat < 2; ++seat) {
            if (seat == oppSeat && oppRole != ROLE_LATEST) {
                g.role[seat] = oppRole;
                g.snapTier[seat] = static_cast<uint8_t>(oppTier);
                g.snapIdx[seat] = oppIdx;
            } else {
                g.role[seat] = ROLE_LATEST;
                g.snapTier[seat] = 0;
                g.snapIdx[seat] = -1;
            }
        }

        if (forcedSeat >= 0) {
            if (forcedRole == ROLE_RULE) {
                ++stats.vsRule;
            } else if (forcedRole == ROLE_RANDOM) {
                ++stats.vsRandom;
            }
        } else if (oppRole == ROLE_SNAPSHOT) {
            if (oppTier == 1) ++stats.vsArchive; else ++stats.vsRecent;
        } else if (oppRole == ROLE_RULE) {
            ++stats.vsRule;
        } else if (oppRole == ROLE_RANDOM) {
            ++stats.vsRandom;
        } else if (oppRole == ROLE_PRO) {
            ++stats.vsPro;
        } else if (oppRole == ROLE_SEARCH) {
            ++stats.vsSearch;
        } else if (oppRole == ROLE_SOLID) {
            ++stats.vsSolid;
        } else if (oppRole == ROLE_TIGHT) {
            ++stats.vsTight;
        } else if (oppRole == ROLE_TRAPPER) {
            ++stats.vsTrapper;
        } else if (oppRole == ROLE_SHARK) {
            ++stats.vsShark;
        } else if (oppRole == ROLE_SPAMMER) {
            ++stats.vsSpammer;
        } else if (oppRole == ROLE_MANIAC) {
            ++stats.vsManiac;
        } else if (oppRole == ROLE_RANGER) {
            ++stats.vsRanger;
        } else if (oppRole == ROLE_CFR) {
            ++stats.vsCfr;
        } else if (oppRole == ROLE_FOLDER) {
            ++stats.vsFolder;
        } else if (oppRole == ROLE_LATEST_SHARP) {
            ++stats.vsSelfSharp;
        } else if (oppRole == ROLE_FIXED) {
            // 纯 BR 模式的冻结对手不计入联赛配比；混合模式要计数。
            if (cfg.mixedFrozen) ++stats.vsExtra;
        } else {
            ++stats.vsLatest;
        }

        for (int seat = 0; seat < 2; ++seat) {
            g.phi[seat] = phiOf(g.st, seat, equityFor(g, seat, cfg.equityIters));
        }
        g.active = true;
    }

    int remaining = N;
    auto finishGame = [&](GameCtx& g) {
        for (int p = 0; p < 2; ++p) {
            if (g.lastIdx[p] < 0) continue;
            auto& t = out[static_cast<size_t>(p)][static_cast<size_t>(g.lastIdx[p])];
            t.reward += float(payoff(g.st, p)) * cfg.rewardScale;
            t.terminal = true;
        }
        if (g.st.folded == 255) ++stats.showdowns;
        {
            const int w = (g.st.folded != 255) ? (1 - int(g.st.folded)) : showdownWinner(g.st);
            if (w == 0) ++stats.sbWins;
            else if (w == 1) ++stats.bbWins;
            else ++stats.splits;
        }
        stats.sbScore += payoff(g.st, 0);
        stats.bbScore += payoff(g.st, 1);
        if (cfg.handScores || cfg.handScoresByGame || cfg.handPotsByGame ||
            cfg.handSeatsByGame || cfg.handShowdownByGame) {
            int watched;
            if (cfg.watchedSeat >= 0) {
                watched = cfg.watchedSeat;
            } else if (cfg.watchLearningSeat) {
                watched = (g.role[0] == ROLE_LATEST) ? 0 : 1;
            } else {
                watched = (cfg.forceRuleSeat == 0 || cfg.forceRandomSeat == 0 ||
                           cfg.forceProSeat == 0 || cfg.forceSearchSeat == 0 ||
                           cfg.forceSolidSeat == 0 || cfg.forceRangerSeat == 0)
                              ? 1
                              : 0;
            }
            const int net = payoff(g.st, watched);
            if (cfg.handScores) cfg.handScores->push_back(net);
            if (cfg.handScoresByGame) (*cfg.handScoresByGame)[g.index] = net;
            if (cfg.handPotsByGame) (*cfg.handPotsByGame)[g.index] = g.st.pot();
            if (cfg.handSeatsByGame) (*cfg.handSeatsByGame)[g.index] = watched;
            if (cfg.handShowdownByGame) {
                const bool sd = g.st.folded == 255 && g.st.boardN == 5;
                (*cfg.handShowdownByGame)[g.index] = sd ? 1 : 0;
                int myScore = -1, oppScore = -1;
                if (sd) {
                    int c[7];
                    for (int p = 0; p < 2; ++p) {
                        int n = 0;
                        c[n++] = int(g.st.hole[p][0]);
                        c[n++] = int(g.st.hole[p][1]);
                        for (int i = 0; i < 5; ++i) c[n++] = int(g.st.board[i]);
                        const int sc = evaluateBest(c, n);
                        if (p == watched) myScore = sc;
                        else oppScore = sc;
                    }
                }
                if (cfg.handScoreByGame) (*cfg.handScoreByGame)[g.index] = myScore;
                if (cfg.handOppScoreByGame) (*cfg.handOppScoreByGame)[g.index] = oppScore;
            }
        }
        stats.games += 1;
        g.active = false;
        --remaining;
    };

    auto runBatch = [&](const nn::Actor& actor, const nn::Critic* critic, int seat,
                        const std::vector<int>& idx, bool withEnsemble = false,
                        float selectTemp = 1.0f, bool selectGreedy = false,
                        bool amountMid = false, float raiseMult = 1.0f,
                        float postMult = 1.0f) {
        if (idx.empty()) return;
        const int64_t B = static_cast<int64_t>(idx.size());
        std::vector<Transition> tmp(static_cast<size_t>(B));

        for (int64_t i = 0; i < B; ++i) {
            GameCtx& g = gs[static_cast<size_t>(idx[static_cast<size_t>(i)])];
            Transition& t = tmp[static_cast<size_t>(i)];
            const float eq = equityFor(g, seat, cfg.equityIters);
            Obs obs;
            encodeObservation(g.st, obs, eq);
            t.node = obs.node;
            t.seq = obs.seq;
            // extra 只喂 critic。关掉 criticOracleExtra 时留零，让价值网只能
            // 用 actor 也看得到的节点特征（信息集上的 V）。
            if (cfg.criticOracleExtra) t.extra = obs.extra;

            // 对手建模头标签：对手当前牌力百分位（与节点特征里的我方 handStrength
            // 同一口径，含听牌折算）。这里只做标注，不改变任何行为。
            {
                const int oppSeat = 1 - seat;
                Card oh[2] = {static_cast<Card>(g.st.hole[oppSeat][0]),
                              static_cast<Card>(g.st.hole[oppSeat][1])};
                Card bd[5] = {0, 0, 0, 0, 0};
                for (int i = 0; i < g.st.boardN && i < 5; ++i) {
                    bd[i] = static_cast<Card>(g.st.board[i]);
                }
                t.oppPct = handStrength(oh, bd, int(g.st.boardN));
            }

            // 范围胜率特征：组合级计算（采样对手底牌 + 对手动作似然重加权
            // + 我方摊牌胜率），网络从标量特征学不出来。似然前向喂中性 0.5。
            if (cfg.rangeEqSamples > 0) {
                const int oppSeat = 1 - seat;
                t.rangeEq = rangeEquityFor(g.st, seat, g.lastAct[oppSeat],
                                           &latest.actor[oppSeat], cfg.rangeEqSamples,
                                           cfg.rangeEqIters, rangeRng);
            }

            LegalMove lm[kNumActions];
            const int n = legalMoves(g.st, lm);
            t.actions.resize(static_cast<size_t>(n));
            for (int k = 0; k < n; ++k) {
                ActionOption& a = t.actions[static_cast<size_t>(k)];
                a.slot = lm[k].slot;
                a.mv = lm[k].mv;
                a.hi = lm[k].toHi;
                encodeActionFeature(g.st, lm[k], a.feat);
            }
            t.gameId = idx[static_cast<size_t>(i)];

            // 只有学习座位（带 critic 的那批）才做教师标注。快照 / 评估座位
            // 只负责推进牌局，标了也不会进训练集。
            //
            // 翻前永远标：翻前范围是当前策略最大的漏洞，而且翻前教师的分支
            // 续演已经把翻后的打法也评估进去了。翻后按 teacherProb 抽样，
            // 控制开销（教师要对隐藏牌平均，比一次前向贵一个量级）。
            const bool potOk =
                (cfg.teacherMinPot <= 0.0f) || (float(g.st.pot()) >= cfg.teacherMinPot);
            if (cfg.teacher && critic != nullptr && potOk &&
                (g.st.street == ST_PREFLOP || cfg.teacherProb >= 1.0f ||
                 u(rng) < cfg.teacherProb)) {
                teacherActionValues(g.st, seat, t.teacherQ, cfg.teacherSamples, cfg.teacherStyles);
                t.hasTeacher = true;
            }

            // 专家迭代：河牌用范围条件化搜索的 Q 当教师标签。搜索把当前策略
            // 当对手模型，所以只在对座位上**确实是当前策略**（自对局）时标注；
            // 对手是脚本/快照时模型错配，标签会把我们带向针对错误对手的打法。
            // 标签走同一条 softmax(Q/T) 蒸馏损失，只是来源不同。
            if (cfg.expertProb > 0.0f && critic != nullptr && g.st.boardN >= 5 &&
                g.role[1 - seat] == ROLE_LATEST &&
                (cfg.expertProb >= 1.0f || u(rng) < cfg.expertProb)) {
                policySearchMove(&latest.actor[0], &latest.actor[1], g.st, seat,
                                 cfg.expertSamples > 0 ? cfg.expertSamples : 8, rng,
                                 cfg.policySearchEquityIters, &g.lastAct[1 - seat],
                                 &latest.actor[1 - seat], &t.teacherQ);
                t.hasTeacher = true;
            }
        }

        torch::Tensor nodeT, seqT, extraT, actFeatT, actSampleT, actIdT, rangeEqT;
        writeTensors(tmp, nodeT, seqT, extraT, actFeatT, actSampleT, actIdT, rangeEqT,
                     critic != nullptr);

        torch::Tensor logits = actor->forward(nodeT, seqT, actFeatT, actSampleT, actIdT, rangeEqT);
        const float* pLogits = logits.data_ptr<float>();
        // 动作级集成：额外成员各跑一遍前向。它们看到的是同一个节点、同一批
        // 合法动作，所以概率可以在槽位上直接平均。
        std::vector<torch::Tensor> ensLogits;
        if (withEnsemble) {
            const int m = std::min(cfg.ensembleExtraCount[seat], kMaxEnsemble);
            for (int e = 0; e < m; ++e) {
                const nn::Actor* extra = cfg.ensembleExtra[seat][e];
                if (extra == nullptr) continue;
                ensLogits.push_back(
                    (*extra)->forward(nodeT, seqT, actFeatT, actSampleT, actIdT, rangeEqT));
            }
        }
        torch::Tensor values;
        const float* pValues = nullptr;
        if (critic) {
            values = (*critic)->forward(nodeT, seqT, extraT).squeeze(-1);
            pValues = values.data_ptr<float>();
        }

        for (int64_t i = 0; i < B; ++i) {
            const int gi = idx[static_cast<size_t>(i)];
            GameCtx& g = gs[static_cast<size_t>(gi)];
            Transition& t = tmp[static_cast<size_t>(i)];
            std::pair<int, float> picked;
            const bool applyTemp = (selectTemp != 1.0f || selectGreedy);
            if (!applyTemp && ensLogits.empty()) {
                picked = sampleAction(pLogits + i * kNumActions, t.actions, rng);
            } else {
                float probs[kNumActions];
                float part[kNumActions];
                maskedSoftmaxTemp(pLogits + i * kNumActions, kNumActions, selectTemp, probs);
                for (const torch::Tensor& el : ensLogits) {
                    maskedSoftmaxTemp(el.data_ptr<float>() + i * kNumActions, kNumActions,
                                      selectTemp, part);
                    for (int a = 0; a < kNumActions; ++a) probs[a] += part[a];
                }
                const float inv = 1.0f / float(1 + static_cast<int>(ensLogits.size()));
                for (int a = 0; a < kNumActions; ++a) probs[a] *= inv;
                picked = selectGreedy ? argmaxAction(probs, t.actions)
                                      : sampleFromProbs(probs, t.actions, rng);
            }
            const int slot = picked.first;
            const float logp = picked.second;

            // 规则教师：河牌大池、无人下注 -> 目标是最接近 1.5x 池的加注槽。
            if (cfg.ruleMinPot > 0.0f && critic != nullptr && g.st.street == ST_RIVER &&
                g.st.boardN == 5 && texas::toCall(g.st, seat) == 0 &&
                float(g.st.pot()) >= cfg.ruleMinPot) {
                int bestSlot = -1;
                double bestDiff = 1e18;
                for (const ActionOption& a : t.actions) {
                    if (a.mv.kind != MK_RAISE && a.mv.kind != MK_ALLIN) continue;
                    const double ratio = double(a.mv.to - g.st.contrib[seat]) /
                                         double(std::max(1, g.st.pot()));
                    const double d = std::abs(ratio - double(cfg.ruleTargetRatio));
                    if (d < bestDiff) {
                        bestDiff = d;
                        bestSlot = a.slot;
                    }
                }
                t.ruleSlot = bestSlot;
            }

            const ActionOption* opt = &t.actions.back();
            for (const ActionOption& a : t.actions) {
                if (a.slot == slot) opt = &a;
            }
            t.action = slot;
            t.logp = logp;
            if (pValues) {
                const float v = pValues[i];
                t.value = std::isfinite(v) ? v : 0.0f;
            }

            // 尺寸带 -> 具体金额：带内对数均匀落一个整数。
            // 槽位是抽象动作（"打多大这个量级"），金额是带内的任意值，
            // 所以引擎层面没有任何预设档位限制。评估时可以让金额固定在
            // 对数中点（amountMid），去掉这一层随机性。
            Move played = opt->mv;
            if (played.kind == MK_RAISE && opt->hi > int(played.to)) {
                played.to = static_cast<int16_t>(amountInBand(
                    int(played.to), opt->hi, amountMid ? 0.5 : uu(rng)));
            }
            // 部署专用尺寸放大：只动翻前加注（含全下），向上钳到筹码上限。
            // 只放大 3-bet（面对开牌时的再加注）：numRaises==1。开牌本身
            // （numRaises==0）不动——外部日志里大尺寸开牌的整手收益是负的。
            if (raiseMult > 1.0f && g.st.street == ST_PREFLOP && g.st.numRaises == 1 &&
                (played.kind == MK_RAISE || played.kind == MK_ALLIN)) {
                const int cap = int(g.st.contrib[seat]) + int(g.st.stack[seat]);
                int to = int(std::lround(double(played.to) * double(raiseMult)));
                if (to > cap) to = cap;
                if (to > int(played.to)) played.to = static_cast<int16_t>(to);
            }
            // 后翻牌尺寸放大：所有下注/加注按系数放大，向上钳到筹码上限。
            if (postMult > 1.0f && g.st.street > ST_PREFLOP &&
                (played.kind == MK_RAISE || played.kind == MK_ALLIN)) {
                const int cap = int(g.st.contrib[seat]) + int(g.st.stack[seat]);
                int to = int(std::lround(double(played.to) * double(postMult)));
                if (to > cap) to = cap;
                if (to > int(played.to)) played.to = static_cast<int16_t>(to);
            }
            play(g, seat, played);
            ++stats.decisions;
            if (played.kind == MK_ALLIN) ++stats.allinDecisions;
            if (played.kind == MK_FOLD) ++stats.folds;

            if (cfg.oracleReward) {
                const float eq = equityFor(g, seat, cfg.equityIters);
                const float phi = phiOf(g.st, seat, eq);
                t.reward += cfg.oracleCoef * (phi - g.phi[seat]);
                g.phi[seat] = phi;
            }

            const int index = static_cast<int>(out[static_cast<size_t>(seat)].size());
            out[static_cast<size_t>(seat)].push_back(std::move(t));
            g.lastIdx[seat] = index;
            if (g.st.terminal()) finishGame(g);
        }
    };

    auto runRandom = [&](int seat, const std::vector<int>& idx) {
        (void)seat;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            LegalMove lm[kNumActions];
            const int n = legalMoves(g.st, lm);
            if (n <= 0) {
                g.active = false;
                --remaining;
                continue;
            }
            const int pick = static_cast<int>(rng() % static_cast<size_t>(n));
            Move played = lm[pick].mv;
            if (played.kind == MK_RAISE && lm[pick].toHi > int(played.to)) {
                played.to = static_cast<int16_t>(
                    amountInBand(int(played.to), lm[pick].toHi, uu(rng)));
            }
            play(g, seat, played);
            if (lm[pick].mv.kind == MK_FOLD) ++stats.folds;
            if (g.st.terminal()) finishGame(g);
        }
    };

    auto runRule = [&](int seat, const std::vector<int>& idx) {
        (void)seat;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            const Move mv = ruleAgentMove(g.st, rng);
            play(g, seat, mv);
            if (mv.kind == MK_FOLD) ++stats.folds;
            if (mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (g.st.terminal()) finishGame(g);
        }
    };

    auto runPro = [&](int seat, const std::vector<int>& idx) {
        (void)seat;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            const Move mv = proAgentMove(g.st, rng);
            play(g, seat, mv);
            if (mv.kind == MK_FOLD) ++stats.folds;
            if (mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (g.st.terminal()) finishGame(g);
        }
    };

    auto runFolder = [&](int seat, const std::vector<int>& idx) {
        (void)seat;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            const Move mv = folderAgentMove(g.st, rng);
            play(g, seat, mv);
            if (mv.kind == MK_FOLD) ++stats.folds;
            if (mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (g.st.terminal()) finishGame(g);
        }
    };

    auto runSolid = [&](int seat, const std::vector<int>& idx) {
        (void)seat;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            const Move mv = solidAgentMove(g.st, rng);
            play(g, seat, mv);
            if (mv.kind == MK_FOLD) ++stats.folds;
            if (mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (g.st.terminal()) finishGame(g);
        }
    };

    auto runTight = [&](int seat, const std::vector<int>& idx) {
        (void)seat;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            const Move mv = tightAgentMove(g.st, rng);
            play(g, seat, mv);
            if (mv.kind == MK_FOLD) ++stats.folds;
            if (mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (g.st.terminal()) finishGame(g);
        }
    };

    auto runTrapper = [&](int seat, const std::vector<int>& idx) {
        (void)seat;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            const Move mv = trapperAgentMove(g.st, rng);
            play(g, seat, mv);
            if (mv.kind == MK_FOLD) ++stats.folds;
            if (mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (g.st.terminal()) finishGame(g);
        }
    };

    auto runSpammer = [&](int seat, const std::vector<int>& idx) {
        (void)seat;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            const Move mv = spammerAgentMove(g.st, rng);
            play(g, seat, mv);
            if (mv.kind == MK_FOLD) ++stats.folds;
            if (mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (g.st.terminal()) finishGame(g);
        }
    };

    auto runManiac = [&](int seat, const std::vector<int>& idx) {
        (void)seat;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            const Move mv = maniacAgentMove(g.st, rng);
            play(g, seat, mv);
            if (mv.kind == MK_FOLD) ++stats.folds;
            if (mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (g.st.terminal()) finishGame(g);
        }
    };

    auto runShark = [&](int seat, const std::vector<int>& idx) {
        (void)seat;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            const Move mv = sharkAgentMove(g.st, rng);
            play(g, seat, mv);
            if (mv.kind == MK_FOLD) ++stats.folds;
            if (mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (g.st.terminal()) finishGame(g);
        }
    };

    auto runRanger = [&](int seat, const std::vector<int>& idx) {
        (void)seat;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            const Move mv = rangerAgentMove(g.st, rng, cfg.rangerAir);
            play(g, seat, mv);
            if (mv.kind == MK_FOLD) ++stats.folds;
            if (mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (g.st.terminal()) finishGame(g);
        }
    };

    auto runSearch = [&](int seat, const std::vector<int>& idx) {
        (void)seat;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            const Move mv = searchAgentMove(g.st, rng, cfg.searchSamples, cfg.searchStyles);
            play(g, seat, mv);
            if (mv.kind == MK_FOLD) ++stats.folds;
            if (mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (g.st.terminal()) finishGame(g);
        }
    };

    // 策略内搜索：ROLE_LATEST 座位不直接按策略采样，而是对每个合法动作做
    // 一批"用自己策略续演"的 rollout，取平均回报最大的动作（见 policySearchMove）。
    const nn::Actor* psActors[2] = {cfg.policySearchActors[0], cfg.policySearchActors[1]};
    auto runPolicySearch = [&](int seat, const std::vector<int>& idx) {
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            const Move mv = policySearchMove(psActors[0], psActors[1], g.st, seat,
                                             cfg.policySearchSamples, rng,
                                             cfg.policySearchEquityIters);
            play(g, seat, mv);
            ++stats.decisions;
            if (mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (mv.kind == MK_FOLD) ++stats.folds;
            if (g.st.terminal()) finishGame(g);
        }
    };

    // 河牌范围条件化搜索：只在河牌决策上做"按对手最近动作的似然加权"的
    // 一步前瞻（其余街走 runBatch 的原策略）。对手模型优先用真实对手
    // （对打时的 forceActor），否则用同一条支路里的另一座位网络。
    auto runRangeSearchActs = [&](const nn::Actor* a0, const nn::Actor* a1, int seat,
                                  const std::vector<int>& idx) {
        const nn::Actor* acts[2] = {a0, a1};
        // 评估路径用 rangeSearchSamples；训练对手用 searchOppSamples（更便宜）。
        const int samples = cfg.rangeSearchSamples > 0 ? cfg.rangeSearchSamples
                                                       : cfg.searchOppSamples;
        if (samples <= 0 || acts[0] == nullptr || acts[1] == nullptr) return;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            const nn::Actor* oppActor =
                (cfg.forceActor != nullptr && cfg.forceActorSeat == 1 - seat)
                    ? cfg.forceActor
                    : acts[1 - seat];
            const Move mv =
                cfg.robustSearch
                    ? policySearchRobustMove(acts[0], acts[1], g.st, seat, samples, rng,
                                             cfg.policySearchEquityIters,
                                             cfg.rangeSearchUniform ? nullptr : &g.lastAct[1 - seat],
                                             oppActor, cfg.robustMean)
                    : policySearchMove(acts[0], acts[1], g.st, seat, samples, rng,
                                       cfg.policySearchEquityIters,
                                       cfg.rangeSearchUniform ? nullptr : &g.lastAct[1 - seat],
                                       oppActor);
            play(g, seat, mv);
            ++stats.decisions;
            if (mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (mv.kind == MK_FOLD) ++stats.folds;
            if (g.st.terminal()) finishGame(g);
        }
    };
    auto runRangeSearch = [&](int seat, const std::vector<int>& idx) {
        runRangeSearchActs(psActors[0], psActors[1], seat, idx);
    };

    // 河牌 CFR+ 子博弈：固定公共牌，把对手范围全枚举成 1081 个组合（权重 =
    // 它最近一次动作的策略似然），解一棵抽象下注树上的均衡，取自己这张牌的
    // 均衡动作。和一步前瞻搜索的区别：这里的对手按均衡回应，而不是按我们的
    // 网络续演，所以输出的是"不被针对"的打法（minimax），两者互补。
    auto runRiverCfr = [&](int seat, const std::vector<int>& idx) {
        if (idx.empty()) return;
        const nn::Actor* acts[2] = {psActors[0] != nullptr ? psActors[0] : &latest.actor[0],
                                    psActors[1] != nullptr ? psActors[1] : &latest.actor[1]};
        RiverCfrConfig rc;
        rc.iters = cfg.riverCfrIters;
        rc.maxRaises = cfg.riverCfrMaxRaises;
        rc.allin = cfg.riverCfrAllin;
        std::vector<int> fallback;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            const nn::Actor* oppActor =
                (cfg.forceActor != nullptr && cfg.forceActorSeat == 1 - seat)
                    ? cfg.forceActor
                    : acts[1 - seat];
            const std::vector<float> oppRange =
                riverRangeFor(g.st, 1 - seat, g.lastAct[1 - seat], oppActor);
            // 自己的范围同样要估：均衡策略取决于双方范围。用均匀范围会让求解器
            // 以为我们手里全是中等牌，从而系统性偏向小注/过牌（这正是"对脚本
            // 对手丢失剥削力"的来源之一）。
            const std::vector<float> ownRange =
                riverRangeFor(g.st, seat, g.lastAct[seat], acts[seat]);
            Move mv{};
            if (!riverCfrMove(g.st, rc, mv, oppRange, ownRange)) {
                fallback.push_back(gi);  // 求解失败不丢样本，退回网络策略
                continue;
            }
            play(g, seat, mv);
            ++stats.decisions;
            if (mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (mv.kind == MK_FOLD) ++stats.folds;
            if (g.st.terminal()) finishGame(g);
        }
        if (!fallback.empty()) {
            runBatch(latest.actor[seat], &latest.critic[seat], seat, fallback,
                     /*withEnsemble=*/true, cfg.policyTemp, cfg.policyGreedy, cfg.amountMid,
                     cfg.preflopRaiseMult, cfg.postflopBetMult);
        }
    };

    // 河牌 CFR 当**对手**（README #109(d)）。和学习座位上的求解器用的是同一套
    // riverCfrMove，只是座位对调：
    //   - 我们的范围：latest.actor[1-seat] 在它最近一次动作上的似然（这是求解器
    //     惩罚薄价值/宽跟注的依据）；
    //   - 它自己的范围：latest.actor[seat] 在同一口径上的似然。
    // 两条街的**打法**必须和范围估计自洽：其它街用当前策略采样（下面的 runBatch），
    // 河牌才用均衡回应。第一版让其它街走 solid，范围模型与实际打出来的牌错位，
    // 实测只比 solid 强 0.14BB/手；自洽版强 2.6BB/手（README #109(d)）。
    // 小池 / 求解失败仍退回当前策略（不是 solid），否则范围又错位。
    auto runOppCfr = [&](int seat, const std::vector<int>& idx) {
        if (idx.empty()) return;
        RiverCfrConfig rc;
        rc.iters = cfg.oppCfrIters;
        rc.maxRaises = cfg.riverCfrMaxRaises;
        rc.allin = cfg.riverCfrAllin;
        std::vector<int> rest;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            const bool bigEnough =
                cfg.oppCfrMinPot <= 0 || g.st.pot() >= cfg.oppCfrMinPot;
            if (g.st.street != ST_RIVER || g.st.boardN < 5 || !bigEnough) {
                rest.push_back(gi);
                continue;
            }
            const std::vector<float> oppRange =
                riverRangeFor(g.st, 1 - seat, g.lastAct[1 - seat], &latest.actor[1 - seat]);
            const std::vector<float> ownRange =
                riverRangeFor(g.st, seat, g.lastAct[seat], &latest.actor[seat]);
            Move mv{};
            if (!riverCfrMove(g.st, rc, mv, oppRange, ownRange)) {
                rest.push_back(gi);
                continue;
            }
            play(g, seat, mv);
            ++stats.decisions;
            if (mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (mv.kind == MK_FOLD) ++stats.folds;
            if (g.st.terminal()) finishGame(g);
        }
        // 非河牌：用当前策略采样（不是 solid），保证 `ownRange` 的似然代理自洽
        // ——它翻前/翻牌的范围就是它自己的策略打出来的。
        if (!rest.empty()) {
            runBatch(latest.actor[seat], nullptr, seat, rest, /*withEnsemble=*/false,
                     cfg.policyTemp, cfg.policyGreedy);
        }
    };

    while (remaining > 0) {
        for (int seat = 0; seat < 2; ++seat) {
            std::vector<int> latestIdx;
            std::map<std::pair<int, int>, std::vector<int>> snapIdx;  // (tier, index)
            std::vector<int> sharpIdx;
            std::vector<int> ruleIdx;
            std::vector<int> randomIdx;
            std::vector<int> fixedIdx;
            std::vector<int> proIdx;
            std::vector<int> searchIdx;
            std::vector<int> solidIdx;
            std::vector<int> tightIdx;
            std::vector<int> trapperIdx;
            std::vector<int> sharkIdx;
            std::vector<int> spammerIdx;
            std::vector<int> maniacIdx;
            std::vector<int> rangerIdx;
            std::vector<int> cfrIdx;
            std::vector<int> folderIdx;
            std::vector<int> bigIdx;
            std::vector<int> sOppIdx;
            const bool useBig =
                cfg.bigPotOpponent[seat] != nullptr && cfg.bigPotThreshold > 0.0f;

            for (int i = 0; i < N; ++i) {
                GameCtx& g = gs[static_cast<size_t>(i)];
                if (!g.active || g.st.toAct != seat) continue;
                // 大池切换只在"对手座位"生效，学习座位保持原策略。
                if (useBig && g.role[seat] != ROLE_LATEST &&
                    float(g.st.pot()) >= cfg.bigPotThreshold) {
                    bigIdx.push_back(i);
                    continue;
                }
                switch (g.role[seat]) {
                    case ROLE_LATEST: latestIdx.push_back(i); break;
                    case ROLE_SNAPSHOT:
                        snapIdx[{int(g.snapTier[seat]), g.snapIdx[seat]}].push_back(i);
                        break;
                    case ROLE_RANDOM: randomIdx.push_back(i); break;
                    case ROLE_LATEST_SHARP: sharpIdx.push_back(i); break;
                    case ROLE_SEARCH_OPP: sOppIdx.push_back(i); break;
                    case ROLE_FIXED: fixedIdx.push_back(i); break;
                    case ROLE_PRO: proIdx.push_back(i); break;
                    case ROLE_SEARCH: searchIdx.push_back(i); break;
                    case ROLE_SOLID: solidIdx.push_back(i); break;
                    case ROLE_TIGHT: tightIdx.push_back(i); break;
                    case ROLE_TRAPPER: trapperIdx.push_back(i); break;
                    case ROLE_SHARK: sharkIdx.push_back(i); break;
                    case ROLE_SPAMMER: spammerIdx.push_back(i); break;
                    case ROLE_MANIAC: maniacIdx.push_back(i); break;
                    case ROLE_RANGER: rangerIdx.push_back(i); break;
                    case ROLE_CFR: cfrIdx.push_back(i); break;
                    case ROLE_FOLDER: folderIdx.push_back(i); break;
                    default: ruleIdx.push_back(i); break;
                }
            }

            if (cfg.riverCfrIters > 0) {
                // 河牌：CFR+ 子博弈；其它街：原策略。
                std::vector<int> solvedIdx;
                std::vector<int> otherIdx;
                for (int gi : latestIdx) {
                    const GameCtx& g = gs[static_cast<size_t>(gi)];
                    const bool bigEnough =
                        cfg.riverCfrMinPot <= 0 || g.st.pot() >= cfg.riverCfrMinPot;
                    if (g.st.street == ST_RIVER && g.st.boardN >= 5 && bigEnough) {
                        solvedIdx.push_back(gi);
                    } else {
                        otherIdx.push_back(gi);
                    }
                }
                runRiverCfr(seat, solvedIdx);
                if (!otherIdx.empty()) {
                    runBatch(latest.actor[seat], &latest.critic[seat], seat, otherIdx,
                             /*withEnsemble=*/true, cfg.policyTemp, cfg.policyGreedy,
                             cfg.amountMid, cfg.preflopRaiseMult, cfg.postflopBetMult);
                }
            } else if (cfg.rangeSearchSamples > 0 && cfg.policySearchActors[0] != nullptr &&
                       cfg.policySearchActors[1] != nullptr) {
                // 河牌：范围条件化搜索；其它街：原策略（runBatch）。
                // 底池 < rangeSearchMinPot 的河牌决策也走原策略：搜索的模型
                // 错配成本在小池最不划算（外部赤字集中在大池摊牌）。
                std::vector<int> riverIdx;
                std::vector<int> otherIdx;
                for (int gi : latestIdx) {
                    const GameCtx& g = gs[static_cast<size_t>(gi)];
                    const bool bigEnough = cfg.rangeSearchMinPot <= 0 ||
                                           g.st.pot() >= cfg.rangeSearchMinPot;
                    const bool facingBet = !cfg.rangeSearchDefenseOnly ||
                                           toCall(g.st, seat) > 0;
                    if (g.st.boardN >= 5 && bigEnough && facingBet) {
                        riverIdx.push_back(gi);
                    } else {
                        otherIdx.push_back(gi);
                    }
                }
                runRangeSearch(seat, riverIdx);
                if (!otherIdx.empty()) {
                    runBatch(latest.actor[seat], &latest.critic[seat], seat, otherIdx,
                             /*withEnsemble=*/true, cfg.policyTemp, cfg.policyGreedy,
                             cfg.amountMid, cfg.preflopRaiseMult, cfg.postflopBetMult);
                }
            } else if (cfg.policySearchSamples > 0 && cfg.policySearchActors[0] != nullptr &&
                       cfg.policySearchActors[1] != nullptr) {
                runPolicySearch(seat, latestIdx);
            } else {
                runBatch(latest.actor[seat], &latest.critic[seat], seat, latestIdx,
                         /*withEnsemble=*/true, cfg.policyTemp, cfg.policyGreedy, cfg.amountMid,
                         cfg.preflopRaiseMult, cfg.postflopBetMult);
            }
            if (!sOppIdx.empty()) {
                // 范围搜索剥削者：河牌用最新策略做后验加权前瞻，其它街按
                // 最新策略采样。它和"尖自对局"一样，只在对手座位出现。
                std::vector<int> riverIdx;
                std::vector<int> otherIdx;
                for (int gi : sOppIdx) {
                    if (gs[static_cast<size_t>(gi)].st.boardN >= 5) {
                        riverIdx.push_back(gi);
                    } else {
                        otherIdx.push_back(gi);
                    }
                }
                runRangeSearchActs(&latest.actor[0], &latest.actor[1], seat, riverIdx);
                if (!otherIdx.empty()) {
                    runBatch(latest.actor[seat], nullptr, seat, otherIdx,
                             /*withEnsemble=*/false, cfg.policyTemp, cfg.policyGreedy,
                             cfg.amountMid, cfg.preflopRaiseMult, cfg.postflopBetMult);
                }
            }
            // 尖自对局：同一套权重、按部署温度提取动作；这些转移和快照对手
            // 一样只作为"对手视角"的样本来用。
            if (!sharpIdx.empty()) {
                runBatch(latest.actor[seat], nullptr, seat, sharpIdx, /*withEnsemble=*/false,
                         cfg.selfSharpTemp, /*selectGreedy=*/false);
            }
            if (!fixedIdx.empty() && cfg.extraOpponentCount > 0 &&
                cfg.forceActor == nullptr) {
                // 真双 oracle：按本局抽中的编号分桶，每个 BR 单独一批前向。
                std::array<std::vector<int>, RolloutConfig::kMaxExtraOpponents> buckets;
                for (int gi : fixedIdx) {
                    buckets[gs[static_cast<size_t>(gi)].extraIdx].push_back(gi);
                }
                for (int e = 0; e < cfg.extraOpponentCount; ++e) {
                    if (buckets[static_cast<size_t>(e)].empty()) continue;
                    const nn::Actor* ea = cfg.extraOpponents[e][seat];
                    if (ea == nullptr) ea = cfg.fixedOpponent[seat];
                    if (ea == nullptr) continue;
                    runBatch(*ea, nullptr, seat, buckets[static_cast<size_t>(e)],
                             /*withEnsemble=*/false, cfg.fixedOpponentTemp,
                             cfg.fixedOpponentGreedy);
                }
            } else if (!fixedIdx.empty()) {
                const nn::Actor* fa =
                    (cfg.forceActor != nullptr) ? cfg.forceActor : cfg.fixedOpponent[seat];
                if (fa != nullptr) {
                    // forceActor 是对打用的 B 侧，fixedOpponent 是训练 BR 用的
                    // 冻结对手；两者的动作提取方式分开配置。
                    if (cfg.forceActor != nullptr) {
                        runBatch(*fa, nullptr, seat, fixedIdx, /*withEnsemble=*/false,
                                 cfg.forceActorTemp, cfg.forceActorGreedy, /*amountMid=*/false,
                                 cfg.forceActorPreflopRaiseMult,
                                 cfg.forceActorPostflopBetMult);
                    } else {
                        runBatch(*fa, nullptr, seat, fixedIdx, /*withEnsemble=*/false,
                                 cfg.fixedOpponentTemp, cfg.fixedOpponentGreedy);
                    }
                } else {
                    // 防御：指针没配齐也不能让牌局卡死。
                    runBatch(latest.actor[seat], nullptr, seat, fixedIdx);
                }
            }
            for (auto& [key, idx] : snapIdx) {
                const Snapshot* snap = pool ? pool->at(key.first, key.second) : nullptr;
                if (snap == nullptr) {
                    // 池子被裁剪过，退化成本策略自己打，不要丢样本。
                    runBatch(latest.actor[seat], nullptr, seat, idx);
                    continue;
                }
                runBatch(snap->actor[seat], nullptr, seat, idx);
            }
            if (!bigIdx.empty()) {
                runBatch(*cfg.bigPotOpponent[seat], nullptr, seat, bigIdx, /*withEnsemble=*/false,
                         cfg.bigPotTemp);
            }
            runRule(seat, ruleIdx);
            runRandom(seat, randomIdx);
            runPro(seat, proIdx);
            runSearch(seat, searchIdx);
            runSolid(seat, solidIdx);
            runTight(seat, tightIdx);
            runTrapper(seat, trapperIdx);
            runShark(seat, sharkIdx);
            runSpammer(seat, spammerIdx);
            runManiac(seat, maniacIdx);
            runRanger(seat, rangerIdx);
            runOppCfr(seat, cfrIdx);
            runFolder(seat, folderIdx);
        }
    }
}

}  // namespace texas
