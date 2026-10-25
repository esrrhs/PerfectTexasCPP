#pragma once

// 无限注德州扑克（No-Limit Texas Hold'em）规则引擎。
//
// 覆盖完整规则：52 张牌、四条街（翻牌前/翻牌/转牌/河牌）、无限注下注、
// all-in、边池分配、摊牌比牌。
//
// 支持 2~6 人。下注顺序按标准：翻牌前从大盲左手第一位开始（单挑时按钮/小盲先），
// 翻牌后从小盲之后开始。
//
// 说明：无限注的下注额是连续实数，必须做动作抽象才能求均衡，
// 抽象由 abstraction.hpp 提供，本文件只负责"合法动作是什么"。

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "card.hpp"

namespace texas {

struct Move {
    enum Kind : std::uint8_t {
        Fold = 0,       // 弃牌
        CheckCall = 1,  // 过牌 / 跟注
        Raise = 2,      // 下注 / 加注到 amount（该玩家的总投入）
    };
    Kind kind = CheckCall;
    int amount = 0;
};

struct HoldemGame {
    static constexpr int kMaxPlayers = 6;
    static constexpr int kSmallBlind = 1;
    static constexpr int kBigBlind = 2;
    static constexpr int kBuyIn = 200;    // 100BB
    static constexpr int kMaxRaises = 2;  // 每条街最多 2 次加注
    static constexpr int kMaxActions = 6;

    struct State {
        int numPlayers = 2;
        Card hole[kMaxPlayers][2] = {};
        Card board[5] = {};
        int boardCount = 0;
        int street = 0;  // 0=preflop 1=flop 2=turn 3=river

        int contrib[kMaxPlayers] = {};
        int roundContrib[kMaxPlayers] = {};
        int raises = 0;
        int acts = 0;
        int player = 0;

        bool folded[kMaxPlayers] = {};
        bool allIn[kMaxPlayers] = {};

        int pendingDeal = 0;  // >0 表示待发公共牌张数（机会节点）
        bool finished = false;
        std::string hist;
    };

    static State init(int numPlayers) {
        State s;
        s.numPlayers = numPlayers;
        s.contrib[0] = kSmallBlind;
        s.roundContrib[0] = kSmallBlind;
        s.contrib[1] = kBigBlind;
        s.roundContrib[1] = kBigBlind;
        // 翻牌前：单挑由按钮（小盲）先行动；多人由枪口位（大盲左手）先行动
        s.player = (numPlayers == 2) ? 0 : 2;
        return s;
    }

    static int maxRoundContrib(const State& s) {
        int m = 0;
        for (int i = 0; i < s.numPlayers; ++i) {
            if (!s.folded[i]) m = std::max(m, s.roundContrib[i]);
        }
        return m;
    }

    static int pot(const State& s) {
        int p = 0;
        for (int i = 0; i < s.numPlayers; ++i) p += s.contrib[i];
        return p;
    }

    static int numActive(const State& s) {
        int n = 0;
        for (int i = 0; i < s.numPlayers; ++i) {
            if (!s.folded[i]) ++n;
        }
        return n;
    }

    static int numCanAct(const State& s) {
        int n = 0;
        for (int i = 0; i < s.numPlayers; ++i) {
            if (!s.folded[i] && !s.allIn[i]) ++n;
        }
        return n;
    }

    static bool anyAllIn(const State& s) {
        for (int i = 0; i < s.numPlayers; ++i) {
            if (!s.folded[i] && s.allIn[i]) return true;
        }
        return false;
    }

    static bool isTerminal(const State& s) {
        if (numActive(s) <= 1) return true;
        return s.finished;
    }

    static bool needsChance(const State& s) { return s.pendingDeal > 0 && !s.finished; }

    static bool streetDone(const State& s) {
        int canAct = 0;
        int maxRC = 0;
        for (int i = 0; i < s.numPlayers; ++i) {
            if (s.folded[i]) continue;
            maxRC = std::max(maxRC, s.roundContrib[i]);
            if (!s.allIn[i]) ++canAct;
        }
        if (canAct == 0) return true;  // 全部 all-in，本街直接结束
        // 还能行动的玩家必须都已投入到位，否则还欠一个跟注/弃牌的决定
        for (int i = 0; i < s.numPlayers; ++i) {
            if (s.folded[i] || s.allIn[i]) continue;
            if (s.roundContrib[i] != maxRC) return false;
        }
        return s.acts >= canAct;
    }

    // ---------- 合法动作 ----------

    static int actions(const State& s, Move* out, int maxOut) {
        if (isTerminal(s) || needsChance(s)) return 0;
        const int p = s.player;
        if (p < 0) return 0;
        const int maxRC = maxRoundContrib(s);
        const bool facing = s.roundContrib[p] < maxRC;
        const int stackLeft = kBuyIn - s.contrib[p];
        const int callNeed = maxRC - s.roundContrib[p];

        int n = 0;
        if (facing && n < maxOut) out[n++] = Move{Move::Fold, 0};
        if (n < maxOut) out[n++] = Move{Move::CheckCall, 0};

        if (s.raises < kMaxRaises && stackLeft > callNeed && n < maxOut) {
            const int potNow = pot(s);
            const double fracs[2] = {0.5, 1.0};  // 半个底池、一个底池
            int prevTarget = -1;
            for (int k = 0; k < 2 && n < maxOut; ++k) {
                int raiseBy = static_cast<int>(potNow * fracs[k]);
                if (raiseBy < kBigBlind) raiseBy = kBigBlind;
                const int targetTotal = std::min(kBuyIn, s.contrib[p] + callNeed + raiseBy);
                if (targetTotal <= s.contrib[p]) continue;
                if (targetTotal == prevTarget) continue;
                prevTarget = targetTotal;
                out[n++] = Move{Move::Raise, targetTotal};
            }
            // all-in 单独作为一个动作，避免被两个比例尺度吞掉
            if (n < maxOut && s.contrib[p] < kBuyIn && prevTarget != kBuyIn) {
                out[n++] = Move{Move::Raise, kBuyIn};
            }
        }
        return n;
    }

    // ---------- 状态转移 ----------

    static State next(const State& s, const Move& m) {
        State t = s;
        const int p = s.player;
        const int maxRC = maxRoundContrib(s);

        if (m.kind == Move::Fold) {
            t.folded[p] = true;
            t.hist.push_back('f');
            t.player = -1;
            return t;
        }

        if (m.kind == Move::CheckCall) {
            const int delta = maxRC - s.roundContrib[p];
            t.contrib[p] += delta;
            t.roundContrib[p] = maxRC;
            ++t.acts;
            t.hist.push_back(delta > 0 ? 'c' : 'k');
        } else {
            const int targetTotal = std::min(m.amount, kBuyIn);
            const int delta = targetTotal - s.contrib[p];
            t.contrib[p] = targetTotal;
            t.roundContrib[p] = s.roundContrib[p] + delta;
            ++t.raises;
            ++t.acts;
            t.hist.push_back('r');
        }

        for (int i = 0; i < t.numPlayers; ++i) {
            if (!t.folded[i] && t.contrib[i] >= kBuyIn) t.allIn[i] = true;
        }

        t.player = -1;
        if (numActive(t) <= 1) return t;

        if (!streetDone(t)) {
            t.player = nextPlayer(t, p);
            return t;
        }

        // 本街结束
        for (int i = 0; i < t.numPlayers; ++i) t.roundContrib[i] = 0;
        t.raises = 0;
        t.acts = 0;
        t.hist.push_back('/');

        if (t.street == 3) {
            t.finished = true;  // 河牌打完，摊牌
            t.pendingDeal = 0;
        } else if (anyAllIn(t)) {
            t.pendingDeal = 5 - t.boardCount;  // 一次发完剩余公共牌
            t.street = 3;
        } else {
            t.street += 1;
            t.pendingDeal = (t.street == 1) ? 3 : 1;
        }
        return t;
    }

    static int nextPlayer(const State& s, int cur) {
        for (int i = 1; i <= s.numPlayers; ++i) {
            const int p = (cur + i) % s.numPlayers;
            if (!s.folded[p] && !s.allIn[p]) return p;
        }
        return -1;
    }

    // ---------- 发牌（机会节点） ----------

    static std::vector<Card> remainingDeck(const State& s) {
        bool used[kNumCards] = {};
        for (int i = 0; i < s.numPlayers; ++i) {
            used[s.hole[i][0]] = true;
            used[s.hole[i][1]] = true;
        }
        for (int i = 0; i < s.boardCount; ++i) used[s.board[i]] = true;
        std::vector<Card> deck;
        deck.reserve(kNumCards);
        for (int c = 0; c < kNumCards; ++c) {
            if (!used[c]) deck.push_back(c);
        }
        return deck;
    }

    template <class Rng>
    static void deal(State& s, Rng& rng) {
        std::vector<Card> deck = remainingDeck(s);
        const int need = s.pendingDeal;
        for (int i = 0; i < need && i < static_cast<int>(deck.size()); ++i) {
            std::uniform_int_distribution<int> dist(i, static_cast<int>(deck.size()) - 1);
            std::swap(deck[i], deck[dist(rng)]);
            s.board[s.boardCount++] = deck[i];
        }
        s.pendingDeal = 0;
        if (numCanAct(s) <= 1) {
            s.finished = true;
            s.player = -1;
        } else {
            s.player = nextPlayer(s, 0);  // 翻牌后由小盲之后开始
        }
    }

    // ---------- 摊牌与边池 ----------

    static int handScore(const State& s, int p) {
        Card all[7];
        int n = 0;
        all[n++] = s.hole[p][0];
        all[n++] = s.hole[p][1];
        for (int i = 0; i < s.boardCount; ++i) all[n++] = s.board[i];
        if (n == 7) return evaluate7(all);
        if (n == 5) return evaluate5(all);
        if (n == 6) {
            int best = -1;
            for (int skip = 0; skip < 6; ++skip) {
                Card h5[5];
                int k = 0;
                for (int i = 0; i < 6; ++i) {
                    if (i != skip) h5[k++] = all[i];
                }
                best = std::max(best, evaluate5(h5));
            }
            return best;
        }
        return 0;
    }

    // 每人净收益（零和）。按投入额分层结算边池。
    static void payouts(const State& s, double* out) {
        for (int i = 0; i < s.numPlayers; ++i) out[i] = 0.0;

        if (numActive(s) == 1) {
            const int total = pot(s);
            for (int i = 0; i < s.numPlayers; ++i) {
                if (s.folded[i]) {
                    out[i] = -static_cast<double>(s.contrib[i]);
                } else {
                    out[i] = static_cast<double>(total - s.contrib[i]);
                }
            }
            return;
        }

        int levels[kMaxPlayers];
        int levelCount = 0;
        for (int i = 0; i < s.numPlayers; ++i) {
            const int v = s.contrib[i];
            bool dup = false;
            for (int j = 0; j < levelCount; ++j) {
                if (levels[j] == v) dup = true;
            }
            if (!dup && v > 0) levels[levelCount++] = v;
        }
        std::sort(levels, levels + levelCount);

        int scores[kMaxPlayers];
        for (int i = 0; i < s.numPlayers; ++i) {
            scores[i] = s.folded[i] ? -1 : handScore(s, i);
        }

        int prev = 0;
        for (int li = 0; li < levelCount; ++li) {
            const int lv = levels[li];
            const int slice = lv - prev;
            if (slice <= 0) continue;

            int potSlice = 0;
            for (int i = 0; i < s.numPlayers; ++i) {
                if (s.contrib[i] >= lv) potSlice += slice;
            }

            int bestScore = -1;
            for (int i = 0; i < s.numPlayers; ++i) {
                if (s.contrib[i] >= lv && !s.folded[i]) {
                    bestScore = std::max(bestScore, scores[i]);
                }
            }
            if (bestScore < 0) {
                prev = lv;
                continue;
            }
            int nw = 0;
            for (int i = 0; i < s.numPlayers; ++i) {
                if (s.contrib[i] >= lv && !s.folded[i] && scores[i] == bestScore) ++nw;
            }
            const double share = static_cast<double>(potSlice) / nw;
            for (int i = 0; i < s.numPlayers; ++i) {
                if (s.contrib[i] >= lv && !s.folded[i] && scores[i] == bestScore) out[i] += share;
            }
            prev = lv;
        }

        for (int i = 0; i < s.numPlayers; ++i) out[i] -= static_cast<double>(s.contrib[i]);
    }

    static double utilityFrom(const State& s, int player) {
        double pay[kMaxPlayers];
        payouts(s, pay);
        return pay[player];
    }
};

}  // namespace texas
