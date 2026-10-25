#include "texas/hu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace texas {

namespace {

// 每街尺寸阶梯的目标投入额。
//   翻牌前：上次加注额的倍数，开局 lastRaiseTo = 2（大盲）→ 4/5/6/8/11
//   翻牌后：在上次加注额之上加底池比例，无人下注时就是纯底池比例下注
int ladderTo(const HUState& s, int i) {
    if (s.street == ST_PREFLOP) {
        static const double kMul[kMaxSizes] = {2.2, 2.5, 3.0, 4.0, 5.5};
        return static_cast<int>(std::llround(double(s.lastRaiseTo) * kMul[i]));
    }
    static const double kFrac[kMaxSizes] = {1.0 / 3, 0.5, 2.0 / 3, 1.0, 1.5};
    int inc = static_cast<int>(std::llround(double(s.pot()) * kFrac[i]));
    if (inc < kBigBlind) inc = kBigBlind;
    return int(s.lastRaiseTo) + inc;
}

void pushHist(HUState& s, int seat, int kind, int to) {
    if (s.histLen >= kMaxHist) return;
    HistEntry& e = s.hist[s.histLen++];
    e.seat = static_cast<uint8_t>(seat);
    e.kind = static_cast<uint8_t>(kind);
    e.street = s.street;
    e.to = static_cast<int16_t>(to);
    e.potAfter = static_cast<int16_t>(s.pot());
}

// 本街结束：发下一条街，或者直接跑到摊牌。
void closeStreet(HUState& s) {
    if (s.street == ST_RIVER) {
        s.finished = 1;
        return;
    }
    if (s.allin[0] || s.allin[1]) {
        s.boardN = 5;
        s.finished = 1;
        return;
    }
    ++s.street;
    s.boardN = (s.street == ST_FLOP) ? 3 : static_cast<uint8_t>(s.street + 2);
    s.contrib[0] = s.contrib[1] = 0;
    s.acted[0] = s.acted[1] = 0;
    s.lastRaiseTo = 0;
    s.lastRaiseSize = 0;
    s.numRaises = 0;
    s.lastAggressor = 255;
    s.toAct = 1;  // 翻牌后大盲先行动，小盲（按钮）在后
}

double equityFromScores(int s0, int s1) {
    return (s0 > s1) ? 1.0 : ((s0 < s1) ? 0.0 : 0.5);
}

}  // namespace

HUState makeState(const Card* hole0, const Card* hole1, const Card* board, int boardN) {
    HUState s;
    s.hole[0][0] = static_cast<uint8_t>(hole0[0]);
    s.hole[0][1] = static_cast<uint8_t>(hole0[1]);
    s.hole[1][0] = static_cast<uint8_t>(hole1[0]);
    s.hole[1][1] = static_cast<uint8_t>(hole1[1]);
    for (int i = 0; i < 5; ++i) s.board[i] = board ? static_cast<uint8_t>(board[i]) : 0;
    s.boardN = static_cast<uint8_t>(boardN);
    s.stack[0] = kStartingStack - kSmallBlind;
    s.stack[1] = kStartingStack - kBigBlind;
    s.contrib[0] = kSmallBlind;
    s.contrib[1] = kBigBlind;
    s.total[0] = kSmallBlind;
    s.total[1] = kBigBlind;
    s.street = ST_PREFLOP;
    s.toAct = 0;
    s.lastRaiseTo = kBigBlind;
    s.lastRaiseSize = kBigBlind;
    s.numRaises = 0;
    s.lastAggressor = 255;
    s.folded = 255;
    s.allin[0] = s.allin[1] = 0;
    s.acted[0] = s.acted[1] = 0;
    s.finished = 0;
    s.histLen = 0;
    return s;
}

HUState dealNewHand(Rng& rng) {
    Card deck[kNumCards];
    for (int i = 0; i < kNumCards; ++i) deck[i] = i;
    for (int i = kNumCards - 1; i > 0; --i) {
        std::uniform_int_distribution<int> d(0, i);
        std::swap(deck[i], deck[d(rng)]);
    }
    Card h0[2] = {deck[0], deck[1]};
    Card h1[2] = {deck[2], deck[3]};
    Card board[5] = {deck[4], deck[5], deck[6], deck[7], deck[8]};
    return makeState(h0, h1, board, 0);
}

int legalMoves(const HUState& s, LegalMove* out) {
    if (s.finished) return 0;
    const int p = s.toAct;
    const int o = 1 - p;
    const int need = toCall(s, p);
    const int myStack = s.stack[p];
    int n = 0;

    if (need > 0) {
        out[n].slot = kSlotFold;
        out[n].mv.kind = MK_FOLD;
        out[n].mv.to = s.contrib[p];
        ++n;
    }

    const int callTo = s.contrib[p] + std::min(need, myStack);
    out[n].slot = kSlotCheckCall;
    out[n].mv.kind = (need > 0) ? MK_CALL : MK_CHECK;
    out[n].mv.to = static_cast<int16_t>(callTo);
    ++n;

    const int maxTo = s.contrib[p] + myStack;
    const bool canRaise = (s.numRaises < kMaxRaises) && !s.allin[o] && myStack > need;
    if (canRaise) {
        int minTo = int(s.lastRaiseTo) + std::max<int>(s.lastRaiseSize, kBigBlind);
        if (minTo <= s.contrib[p]) minTo = s.contrib[p] + kBigBlind;
        int slot = 0;
        int prev = -1;
        for (int i = 0; i < kMaxSizes; ++i) {
            int to = ladderTo(s, i);
            if (to < minTo) to = minTo;
            if (to >= maxTo) break;  // 再大就是 all-in，不重复占槽
            if (to == prev) continue;
            prev = to;
            out[n].slot = slotOfSize(slot++);
            out[n].mv.kind = MK_RAISE;
            out[n].mv.to = static_cast<int16_t>(to);
            ++n;
        }
        if (maxTo > callTo) {
            out[n].slot = kSlotAllIn;
            out[n].mv.kind = MK_ALLIN;
            out[n].mv.to = static_cast<int16_t>(maxTo);
            ++n;
        }
    }
    return n;
}

void applyMove(HUState& s, const Move& m, Rng& rng) {
    (void)rng;  // 公共牌在开局时一次发定，机会节点只推进 boardN，不需要再抽
    if (s.finished) return;
    const int p = s.toAct;
    const int o = 1 - p;

    if (m.kind == MK_FOLD) {
        s.folded = static_cast<uint8_t>(p);
        s.finished = 1;
        pushHist(s, p, MK_FOLD, s.contrib[p]);
        return;
    }

    int to = m.to;
    if (to < s.contrib[p]) to = s.contrib[p];
    const int myMax = int(s.contrib[p]) + int(s.stack[p]);
    if (to > myMax) to = myMax;
    const int delta = to - int(s.contrib[p]);
    s.contrib[p] = static_cast<int16_t>(to);
    s.stack[p] = static_cast<int16_t>(s.stack[p] - delta);
    s.total[p] = static_cast<int16_t>(s.total[p] + delta);
    s.acted[p] = 1;

    if (m.kind == MK_RAISE || m.kind == MK_ALLIN) {
        s.lastRaiseSize = static_cast<int16_t>(to - int(s.lastRaiseTo));
        s.lastRaiseTo = static_cast<int16_t>(to);
        ++s.numRaises;
        s.lastAggressor = static_cast<uint8_t>(p);
        s.acted[o] = 0;
    }
    if (s.stack[p] == 0) s.allin[p] = 1;

    pushHist(s, p, m.kind, to);

    if (s.acted[0] && s.acted[1] && s.contrib[0] == s.contrib[1]) {
        closeStreet(s);
    } else if (s.allin[0] && s.allin[1]) {
        s.boardN = 5;
        s.finished = 1;
    } else {
        s.toAct = static_cast<uint8_t>(o);
    }
}

int showdownWinner(const HUState& s) {
    Card a[7];
    Card b[7];
    for (int i = 0; i < 2; ++i) {
        a[i] = s.hole[0][i];
        b[i] = s.hole[1][i];
    }
    for (int i = 0; i < 5; ++i) {
        a[2 + i] = s.board[i];
        b[2 + i] = s.board[i];
    }
    const int s0 = evaluate7(a);
    const int s1 = evaluate7(b);
    if (s0 > s1) return 0;
    if (s1 > s0) return 1;
    return 2;
}

int payoff(const HUState& s, int p) {
    if (!s.finished) return 0;
    const int pot = s.pot();
    if (s.folded != 255) {
        return (s.folded == p) ? -int(s.total[p]) : pot - int(s.total[p]);
    }
    const int w = showdownWinner(s);
    if (w == 2) {
        // 平分时奇数筹码给 0 号位，两人净收益之和仍为 0
        const int p0 = pot - pot / 2;
        return (p == 0) ? p0 - int(s.total[0]) : pot / 2 - int(s.total[1]);
    }
    return (w == p) ? pot - int(s.total[p]) : -int(s.total[p]);
}

double perfectEquity(const HUState& s, int p, int iters) {
    const int boardN = s.boardN;
    const Card h0[2] = {static_cast<Card>(s.hole[0][0]), static_cast<Card>(s.hole[0][1])};
    const Card h1[2] = {static_cast<Card>(s.hole[1][0]), static_cast<Card>(s.hole[1][1])};

    bool used[kNumCards] = {};
    used[h0[0]] = used[h0[1]] = used[h1[0]] = used[h1[1]] = true;
    Card rest[kNumCards];
    int nRest = 0;
    for (int c = 0; c < kNumCards; ++c) {
        bool onBoard = false;
        for (int i = 0; i < boardN; ++i) {
            if (int(s.board[i]) == c) onBoard = true;
        }
        if (used[c] || onBoard) continue;
        rest[nRest++] = c;
    }

    Card board[5];
    for (int i = 0; i < boardN; ++i) board[i] = static_cast<Card>(s.board[i]);

    // 转牌/河牌能精确枚举（44 / 1 种），翻牌前与翻牌用抽样。
    if (boardN >= 4) {
        double acc = 0.0;
        int cnt = 0;
        if (boardN == 5) {
            Card a[7] = {h0[0], h0[1], board[0], board[1], board[2], board[3], board[4]};
            Card b[7] = {h1[0], h1[1], board[0], board[1], board[2], board[3], board[4]};
            const double eq0 = equityFromScores(evaluate7(a), evaluate7(b));
            return (p == 0) ? eq0 : 1.0 - eq0;
        }
        for (int i = 0; i < nRest; ++i) {
            board[4] = rest[i];
            Card a[7] = {h0[0], h0[1], board[0], board[1], board[2], board[3], board[4]};
            Card b[7] = {h1[0], h1[1], board[0], board[1], board[2], board[3], board[4]};
            acc += equityFromScores(evaluate7(a), evaluate7(b));
            ++cnt;
        }
        const double eq0 = acc / cnt;
        return (p == 0) ? eq0 : 1.0 - eq0;
    }

    static thread_local Rng rng(0x9e3779b97f4a7c15ULL + 17);
    const int need = 5 - boardN;
    double acc = 0.0;
    Card pool[kNumCards];
    for (int it = 0; it < iters; ++it) {
        std::copy(rest, rest + nRest, pool);
        for (int i = 0; i < need; ++i) {
            std::uniform_int_distribution<int> d(i, nRest - 1);
            std::swap(pool[i], pool[d(rng)]);
            board[boardN + i] = pool[i];
        }
        Card a[7] = {h0[0], h0[1], board[0], board[1], board[2], board[3], board[4]};
        Card b[7] = {h1[0], h1[1], board[0], board[1], board[2], board[3], board[4]};
        acc += equityFromScores(evaluate7(a), evaluate7(b));
    }
    const double eq0 = acc / iters;
    return (p == 0) ? eq0 : 1.0 - eq0;
}

std::string moveToString(const HUState& s, const Move& m) {
    char buf[48];
    switch (m.kind) {
        case MK_FOLD:
            return "fold";
        case MK_CHECK:
            return "check";
        case MK_CALL:
            std::snprintf(buf, sizeof buf, "call %d", int(m.to) - int(s.contrib[s.toAct]));
            return buf;
        case MK_RAISE:
            std::snprintf(buf, sizeof buf, "raise %d", int(m.to));
            return buf;
        case MK_ALLIN:
            std::snprintf(buf, sizeof buf, "allin %d", int(m.to));
            return buf;
        default:
            return "?";
    }
}

std::string boardToString(const HUState& s) {
    std::string out;
    for (int i = 0; i < s.boardN; ++i) {
        if (!out.empty()) out += ' ';
        out += cardToString(s.board[i]);
    }
    return out.empty() ? "-" : out;
}

}  // namespace texas
