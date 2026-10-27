#include "texas/rollout.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <map>

#include "texas/ppo.hpp"
#include "texas/strength.hpp"

namespace texas {

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
    // 紧全下机器人：只用顶范围推/接全下，教策略"全下=紧范围"。
    ROLE_TIGHT = 10,
    // 陷阱机器人：强牌只跟注/过牌，河牌才发力。
    ROLE_TRAPPER = 11,
    // 小注骚扰机器人：翻后高频 1/3 池下注，惩罚过度弃牌。
    ROLE_SPAMMER = 12,
    // 价值型强对手：翻前紧凶、翻后只用两对+ 建池，惩罚"中等牌建大池"。
    ROLE_SHARK = 13,
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
                  torch::Tensor& actIdT, bool withExtra) {
    const int64_t B = static_cast<int64_t>(tmp.size());
    const size_t L = totalOptions(tmp);
    nodeT = torch::empty({B, kNodeInput}, torch::kFloat32);
    seqT = torch::empty({B, kHistSteps, kHistDim}, torch::kFloat32);
    extraT = torch::empty({B, withExtra ? kExtraInput : 1}, torch::kFloat32);
    actFeatT = torch::empty({static_cast<int64_t>(L), kActionInput}, torch::kFloat32);
    actSampleT = torch::empty({static_cast<int64_t>(L)}, torch::kLong);
    actIdT = torch::empty({static_cast<int64_t>(L)}, torch::kLong);

    float* pNode = nodeT.data_ptr<float>();
    float* pSeq = seqT.data_ptr<float>();
    float* pExtra = extraT.data_ptr<float>();
    float* pFeat = actFeatT.data_ptr<float>();
    int64_t* pSample = actSampleT.data_ptr<int64_t>();
    int64_t* pId = actIdT.data_ptr<int64_t>();

    int64_t off = 0;
    for (int64_t i = 0; i < B; ++i) {
        const Transition& t = tmp[static_cast<size_t>(i)];
        std::copy(t.node.begin(), t.node.end(), pNode + i * kNodeInput);
        std::copy(t.seq.begin(), t.seq.end(), pSeq + i * kSeqInput);
        if (withExtra) {
            std::copy(t.extra.begin(), t.extra.end(), pExtra + i * kExtraInput);
        }
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

}  // namespace

std::pair<int, float> sampleAction(const float* logits, const std::vector<ActionOption>& acts,
                                   Rng& rng) {
    float probs[kNumActions];
    ppo::maskedSoftmax(logits, kNumActions, probs);
    return sampleFromProbs(probs, acts, rng);
}

Move policyMove(const nn::Actor* sb, const nn::Actor* bb, const HUState& s, Rng& rng,
                float temp) {
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
    t.actions.resize(static_cast<size_t>(n));
    for (int k = 0; k < n; ++k) {
        ActionOption& a = t.actions[static_cast<size_t>(k)];
        a.slot = lm[k].slot;
        a.mv = lm[k].mv;
        a.hi = lm[k].toHi;
        encodeActionFeature(s, lm[k], a.feat);
    }

    torch::Tensor nodeT, seqT, extraT, actFeatT, actSampleT, actIdT;
    writeTensors(tmp, nodeT, seqT, extraT, actFeatT, actSampleT, actIdT, false);
    torch::Tensor logits = (*actors[seat])->forward(nodeT, seqT, actFeatT, actSampleT, actIdT);
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
};

void advanceLeaves(std::vector<Leaf>& live, const nn::Actor* const* actors, int equityIters,
                   Rng& rng, std::vector<double>& q, int seat, std::vector<Leaf>& next) {
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (int act = 0; act < 2; ++act) {
        std::vector<size_t> idx;
        for (size_t i = 0; i < live.size(); ++i) {
            if (live[i].st.toAct == act) idx.push_back(i);
        }
        if (idx.empty()) continue;

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

        torch::Tensor nodeT, seqT, extraT, actFeatT, actSampleT, actIdT;
        writeTensors(tmp, nodeT, seqT, extraT, actFeatT, actSampleT, actIdT, false);
        torch::Tensor logits = (*actors[act])->forward(nodeT, seqT, actFeatT, actSampleT, actIdT);
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
                q[static_cast<size_t>(L.branch)] += double(payoff(L.st, seat));
            } else {
                next.push_back(L);
            }
        }
    }
}

}  // namespace

Move policySearchMove(const nn::Actor* sb, const nn::Actor* bb, const HUState& s, int seat,
                      int samples, Rng& rng, int equityIters) {
    const nn::Actor* actors[2] = {sb, bb};
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

    std::vector<double> q(static_cast<size_t>(n), 0.0);
    std::vector<Leaf> live;
    live.reserve(static_cast<size_t>(samples * n * 3));

    // 每个隐藏牌样本对**所有**根动作各续演一次：分支之间共享发牌，动作间的
    // 差没有发牌方差，这一点和教师的分支续演一样。
    for (int k = 0; k < samples; ++k) {
        HUState hidden;
        sampleHiddenCards(s, seat, rng, hidden);
        for (int i = 0; i < n; ++i) {
            Leaf L;
            L.st = hidden;
            L.branch = i;
            applyMove(L.st, branch[static_cast<size_t>(i)], rng);
            if (L.st.finished) {
                q[static_cast<size_t>(i)] += double(payoff(L.st, seat));
            } else {
                live.push_back(std::move(L));
            }
        }
    }

    int guard = 0;
    while (!live.empty() && guard++ < 64) {
        std::vector<Leaf> next;
        next.reserve(live.size());
        advanceLeaves(live, actors, equityIters, rng, q, seat, next);
        live.swap(next);
    }
    // 理论上一手牌一定会终局；万一 guard 用尽，把未完成的局按当前净投入计。
    for (const Leaf& L : live) {
        q[static_cast<size_t>(L.branch)] += double(payoff(L.st, seat));
    }

    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (q[static_cast<size_t>(i)] > q[static_cast<size_t>(best)]) best = i;
    }
    return branch[static_cast<size_t>(best)];
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

void collectRollout(const ModelSet& latest, const HistoricalPool* pool, const RolloutConfig& cfg,
                    std::array<std::vector<Transition>, 2>& out, RolloutStats& stats) {
    torch::NoGradGuard noGrad;  // 采样只做前向，不需要建图
    Rng rng(cfg.seed);
    const int N = std::max(1, cfg.games);
    std::vector<GameCtx> gs(static_cast<size_t>(N));
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::uniform_real_distribution<double> uu(0.0, 1.0);

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
    const float wFolder = std::max(0.0f, cfg.folderProb);
    const float wSharp = std::max(0.0f, cfg.selfSharpProb);
    const bool hasExtra =
        cfg.mixedFrozen &&
        (cfg.extraOpponentCount > 0 || cfg.fixedOpponent[0] != nullptr ||
         cfg.fixedOpponent[1] != nullptr);
    const float wExtra = hasExtra ? std::max(0.0f, cfg.extraOpponentProb) : 0.0f;
    const float wSelf = std::max(
        0.0f, 1.0f - (wHist + wRule + wRand + wPro + wSearch + wSolid + wTight + wTrapper +
                      wShark + wSpammer + wFolder + wSharp + wExtra));
    const float wSum = wSelf + wHist + wRule + wRand + wPro + wSearch + wSolid + wTight +
                       wTrapper + wShark + wSpammer + wFolder + wSharp + wExtra;

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
            } else if (r < wHist + wRule + wRand + wPro + wSearch + wSolid + wTight +
                           wTrapper + wShark + wSpammer + wFolder) {
                oppRole = ROLE_FOLDER;
            } else if (r < wHist + wRule + wRand + wPro + wSearch + wSolid + wTight +
                           wTrapper + wShark + wSpammer + wFolder + wSharp) {
                oppRole = ROLE_LATEST_SHARP;
            } else if (r < wHist + wRule + wRand + wPro + wSearch + wSolid + wTight +
                           wTrapper + wShark + wSpammer + wFolder + wSharp + wExtra) {
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
            cfg.handSeatsByGame) {
            int watched;
            if (cfg.watchedSeat >= 0) {
                watched = cfg.watchedSeat;
            } else if (cfg.watchLearningSeat) {
                watched = (g.role[0] == ROLE_LATEST) ? 0 : 1;
            } else {
                watched = (cfg.forceRuleSeat == 0 || cfg.forceRandomSeat == 0 ||
                           cfg.forceProSeat == 0 || cfg.forceSearchSeat == 0 ||
                           cfg.forceSolidSeat == 0)
                              ? 1
                              : 0;
            }
            const int net = payoff(g.st, watched);
            if (cfg.handScores) cfg.handScores->push_back(net);
            if (cfg.handScoresByGame) (*cfg.handScoresByGame)[g.index] = net;
            if (cfg.handPotsByGame) (*cfg.handPotsByGame)[g.index] = g.st.pot();
            if (cfg.handSeatsByGame) (*cfg.handSeatsByGame)[g.index] = watched;
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
        }

        torch::Tensor nodeT, seqT, extraT, actFeatT, actSampleT, actIdT;
        writeTensors(tmp, nodeT, seqT, extraT, actFeatT, actSampleT, actIdT, critic != nullptr);

        torch::Tensor logits = actor->forward(nodeT, seqT, actFeatT, actSampleT, actIdT);
        const float* pLogits = logits.data_ptr<float>();
        // 动作级集成：额外成员各跑一遍前向。它们看到的是同一个节点、同一批
        // 合法动作，所以概率可以在槽位上直接平均。
        std::vector<torch::Tensor> ensLogits;
        if (withEnsemble) {
            const int m = std::min(cfg.ensembleExtraCount[seat], kMaxEnsemble);
            for (int e = 0; e < m; ++e) {
                const nn::Actor* extra = cfg.ensembleExtra[seat][e];
                if (extra == nullptr) continue;
                ensLogits.push_back((*extra)->forward(nodeT, seqT, actFeatT, actSampleT, actIdT));
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
            applyMove(g.st, played, rng);
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
            applyMove(g.st, played, rng);
            if (lm[pick].mv.kind == MK_FOLD) ++stats.folds;
            if (g.st.terminal()) finishGame(g);
        }
    };

    auto runRule = [&](int seat, const std::vector<int>& idx) {
        (void)seat;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            const Move mv = ruleAgentMove(g.st, rng);
            applyMove(g.st, mv, rng);
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
            applyMove(g.st, mv, rng);
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
            applyMove(g.st, mv, rng);
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
            applyMove(g.st, mv, rng);
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
            applyMove(g.st, mv, rng);
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
            applyMove(g.st, mv, rng);
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
            applyMove(g.st, mv, rng);
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
            applyMove(g.st, mv, rng);
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
            applyMove(g.st, mv, rng);
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
            applyMove(g.st, mv, rng);
            ++stats.decisions;
            if (mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (mv.kind == MK_FOLD) ++stats.folds;
            if (g.st.terminal()) finishGame(g);
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
            std::vector<int> folderIdx;
            std::vector<int> bigIdx;
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
                    case ROLE_FIXED: fixedIdx.push_back(i); break;
                    case ROLE_PRO: proIdx.push_back(i); break;
                    case ROLE_SEARCH: searchIdx.push_back(i); break;
                    case ROLE_SOLID: solidIdx.push_back(i); break;
                    case ROLE_TIGHT: tightIdx.push_back(i); break;
                    case ROLE_TRAPPER: trapperIdx.push_back(i); break;
                    case ROLE_SHARK: sharkIdx.push_back(i); break;
                    case ROLE_SPAMMER: spammerIdx.push_back(i); break;
                    case ROLE_FOLDER: folderIdx.push_back(i); break;
                    default: ruleIdx.push_back(i); break;
                }
            }

            if (cfg.policySearchSamples > 0 && cfg.policySearchActors[0] != nullptr &&
                cfg.policySearchActors[1] != nullptr) {
                runPolicySearch(seat, latestIdx);
            } else {
                runBatch(latest.actor[seat], &latest.critic[seat], seat, latestIdx,
                         /*withEnsemble=*/true, cfg.policyTemp, cfg.policyGreedy, cfg.amountMid,
                         cfg.preflopRaiseMult, cfg.postflopBetMult);
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
            runFolder(seat, folderIdx);
        }
    }
}

}  // namespace texas
