#include "texas/features.hpp"

#include <algorithm>
#include <cstring>

#include "texas/strength.hpp"

namespace texas {

namespace {

constexpr float kStackF = float(kStartingStack);
constexpr float kPotF = 2.0f * kStartingStack;
// 加注次数没有上限（ACPC 口径），归一化到 6 并对更大的值饱和。
// 6 次加注在 200BB 下已经接近"再加就没意义"的深度。
constexpr float kRaiseNorm = 6.0f;

float raiseNorm(int n) { return std::min(float(n), kRaiseNorm) / kRaiseNorm; }

void putCard(float* plane, Card c) {
    if (c >= 0 && c < kNumCards) plane[c] = 1.0f;
}

}  // namespace

float handStrength(const Card* hole, const Card* board, int boardN) {
    if (boardN <= 0) return float(StrengthModel::instance().preflopEquity(hole[0], hole[1]));
    return float(StrengthModel::instance().postflopPercentile(hole, board, boardN));
}

float oracleEquity(const HUState& s, int seat, int iters) {
    return float(perfectEquity(s, seat, iters));
}

void encodeObservation(const HUState& s, Obs& o, int equityIters) {
    encodeObservation(s, o, float(perfectEquity(s, s.toAct, equityIters)));
}

void encodeObservation(const HUState& s, Obs& o, float eqActing) {
    o.node.fill(0.0f);
    o.extra.fill(0.0f);
    o.seq.fill(0.0f);

    const int me = s.toAct;
    const int opp = 1 - me;
    const int pot = s.pot();
    const int need = toCall(s, me);

    // ---- 底牌 / 公共牌 ----
    for (int i = 0; i < 2; ++i) {
        putCard(o.node.data() + i * kNumCards, static_cast<Card>(s.hole[me][i]));
    }
    for (int i = 0; i < s.boardN && i < 5; ++i) {
        putCard(o.node.data() + kHoleDim + i * kNumCards, static_cast<Card>(s.board[i]));
    }

    // ---- 标量 ----
    Card hole[2] = {static_cast<Card>(s.hole[me][0]), static_cast<Card>(s.hole[me][1])};
    Card board[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < s.boardN && i < 5; ++i) board[i] = static_cast<Card>(s.board[i]);
    const float strength = handStrength(hole, board, s.boardN);
    const DrawOuts outs = StrengthModel::computeOuts(hole, board, s.boardN);

    float* x = o.node.data() + kHoleDim + kBoardDim;
    x[0] = float(pot) / kStackF;
    x[1] = float(need) / kStackF;
    x[2] = float(s.stack[me]) / kStackF;
    x[3] = float(s.stack[opp]) / kStackF;
    x[4] = (need > 0) ? float(need) / float(pot + need) : 0.0f;
    x[5] = float(s.street) / 3.0f;
    for (int i = 0; i < 4; ++i) x[6 + i] = (s.street == i) ? 1.0f : 0.0f;
    x[10] = (me == 0) ? 1.0f : 0.0f;                       // 小盲 / 按钮
    x[11] = (s.street > 0 && me == 0) ? 1.0f : 0.0f;       // 翻牌后有位置
    x[12] = (pot > 0) ? float(s.lastRaiseTo) / float(pot) : 0.0f;
    const int effStack = std::min<int>(s.stack[0], s.stack[1]);
    x[13] = (pot > 0) ? float(effStack) / float(pot) : 0.0f;
    x[14] = strength;
    x[15] = float(outs.total) / 15.0f;
    x[16] = raiseNorm(s.numRaises);
    x[17] = s.allin[me] ? 1.0f : 0.0f;
    x[18] = s.allin[opp] ? 1.0f : 0.0f;
    x[19] = (pot > 0) ? float(s.contrib[me]) / float(pot) : 0.0f;
    x[20] = (pot > 0) ? float(s.contrib[opp]) / float(pot) : 0.0f;
    x[21] = (s.street == ST_PREFLOP) ? float(StrengthModel::preflopClass(hole[0], hole[1])) / 168.0f : 0.0f;
    x[22] = float(s.boardN) / 5.0f;
    x[23] = 1.0f;

    // ---- 历史序列（最近 kHistSteps 步，前面补零）----
    const int total = std::min<int>(s.histLen, kHistSteps);
    for (int i = 0; i < total; ++i) {
        const HistEntry& e = s.hist[s.histLen - total + i];
        float* t = o.seq.data() + i * kHistDim;
        t[0] = (e.seat == 0) ? 1.0f : 0.0f;
        t[1] = float(e.street) / 3.0f;
        t[2] = float(e.to) / kStackF;
        t[3] = float(e.potAfter) / kPotF;
        t[4] = (e.kind == MK_FOLD) ? 1.0f : 0.0f;
        t[5] = (e.kind == MK_CHECK) ? 1.0f : 0.0f;
        t[6] = (e.kind == MK_CALL) ? 1.0f : 0.0f;
        t[7] = (e.kind == MK_RAISE) ? 1.0f : 0.0f;
        t[8] = (e.kind == MK_ALLIN) ? 1.0f : 0.0f;
        t[9] = 1.0f;
        t[10] = (e.potAfter > 0) ? float(e.to) / float(e.potAfter) : 0.0f;
        t[11] = float(e.street + 1) / 4.0f;
    }

    // ---- 开天眼：对手底牌 + 完美信息胜率 ----
    for (int i = 0; i < 2; ++i) {
        putCard(o.extra.data() + i * kNumCards, static_cast<Card>(s.hole[opp][i]));
    }
    const float eq = eqActing;
    o.equityPerfect = eq;
    float* ex = o.extra.data() + kHoleDim;
    ex[0] = eq;
    ex[1] = (need > 0) ? float(need) / float(pot + need) : 0.0f;  // 底池赔率
    ex[2] = eq - ex[1];                                          // 跟注的边际
    ex[3] = 1.0f;
}

void encodeActionFeature(const HUState& s, const LegalMove& lm,
                         std::array<float, kActionInput>& out) {
    out.fill(0.0f);
    const int me = s.toAct;
    const int opp = 1 - me;
    const int pot = s.pot();
    const int potF = std::max(pot, 1);
    const int to = lm.mv.to;
    const int delta = to - int(s.contrib[me]);
    const bool isRaise = (lm.mv.kind == MK_RAISE || lm.mv.kind == MK_ALLIN);
    const int potAfter = pot + delta;

    switch (lm.mv.kind) {
        case MK_FOLD: out[0] = 1.0f; break;
        case MK_CHECK:
        case MK_CALL: out[1] = 1.0f; break;
        case MK_RAISE: out[2] = 1.0f; break;
        case MK_ALLIN: out[3] = 1.0f; break;
        default: break;
    }

    out[4] = float(to) / float(potF);
    out[5] = float(to) / kStackF;
    out[6] = float(delta) / float(potF);
    out[7] = float(potAfter) / kPotF;
    out[8] = float(s.stack[me] - delta) / kStackF;
    out[9] = (lm.mv.kind == MK_ALLIN) ? 1.0f : 0.0f;
    out[10] = isRaise ? 1.0f : 0.0f;
    out[11] = (potAfter > 0) ? float(std::max(0, to - int(s.contrib[opp]))) / float(potAfter) : 0.0f;
    out[12] = raiseNorm(s.numRaises + (isRaise ? 1 : 0));
    out[13] = 1.0f;
    out[14] = (potF > 0) ? float(delta) / float(potF) : 0.0f;
    out[15] = (s.street == ST_PREFLOP) ? 1.0f : 0.0f;
}

}  // namespace texas
