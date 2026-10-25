#pragma once

// Leduc Hold'em：6 张牌（J,J,Q,Q,K,K），两人各发 1 张，各放 1 注。
// 两轮下注，每轮 bet/raise 固定 2 注，每轮最多 2 次加注；
// 第二轮开始前翻 1 张公共牌。成对（手牌 rank == 公共牌 rank）胜过非成对，
// 都非成对时比 rank 大小。
// 这是不完全信息博弈的标准 benchmark，比 Kuhn 多了公共牌与多轮下注。

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace texas {

struct LeducGame {
    static constexpr int kDeckSize = 6;   // J,J,Q,Q,K,K
    static constexpr int kBetSize = 2;
    static constexpr int kSecondRoundBetSize = 4;  // 标准 Leduc：第二轮下注翻倍
    static constexpr int kMaxRaises = 2;  // 每轮最多 2 次加注（bet 记作第 1 次）
    static constexpr int kAnte = 1;

    struct State {
        int priv[2] = {-1, -1};
        int board = -1;
        int round = 0;
        int contrib[2] = {kAnte, kAnte};
        int roundContrib[2] = {0, 0};
        int raises = 0;
        int acts = 0;
        int player = 0;
        bool folded = false;
        int folder = -1;
        bool roundEnd = false;
        std::string hist;
    };

    static int rankOf(int card) { return card / 2; }

    static bool isTerminal(const State& s) {
        return s.folded || (s.roundEnd && s.round == 1);
    }

    static bool needsChance(const State& s) {
        return s.roundEnd && s.round == 0;
    }

    static int playerToAct(const State& s) { return s.player; }

    static int actions(const State& s, std::array<char, 3>& out) {
        const int p = s.player;
        const int maxC = std::max(s.roundContrib[0], s.roundContrib[1]);
        int n = 0;
        if (s.roundContrib[p] < maxC) out[n++] = 'f';
        out[n++] = 'c';
        if (s.raises < kMaxRaises) out[n++] = 'r';
        return n;
    }

    static State next(const State& s, char a) {
        State t = s;
        t.hist.push_back(a);
        const int p = s.player;

        if (a == 'f') {
            t.folded = true;
            t.folder = p;
            return t;
        }

        const int maxC = std::max(t.roundContrib[0], t.roundContrib[1]);
        const int oldC = t.roundContrib[p];

        if (a == 'r') {
            const int betSize = (t.round == 0) ? kBetSize : kSecondRoundBetSize;
            t.roundContrib[p] = maxC + betSize;
            t.contrib[p] += t.roundContrib[p] - oldC;
            ++t.raises;
            ++t.acts;
            t.player = 1 - p;
            return t;
        }

        // 'c' = check 或 call
        t.contrib[p] += maxC - oldC;
        t.roundContrib[p] = maxC;
        ++t.acts;

        const bool bothChecked = (t.raises == 0 && t.acts == 2);
        const bool calledBet = (t.raises > 0 && t.roundContrib[0] == t.roundContrib[1]);
        if (bothChecked || calledBet) {
            t.roundEnd = true;
            t.hist.push_back('/');
            t.player = 0;
        } else {
            t.player = 1 - p;
        }
        return t;
    }

    // 第二轮前的公共牌机会节点：剩余 4 张等概率
    static std::vector<State> chanceStates(const State& s) {
        std::vector<State> v;
        bool used[kDeckSize] = {};
        if (s.priv[0] >= 0) used[s.priv[0]] = true;
        if (s.priv[1] >= 0) used[s.priv[1]] = true;
        for (int b = 0; b < kDeckSize; ++b) {
            if (used[b]) continue;
            State t = s;
            t.board = b;
            t.round = 1;
            t.roundEnd = false;
            t.roundContrib[0] = 0;
            t.roundContrib[1] = 0;
            t.raises = 0;
            t.acts = 0;
            t.player = 0;
            v.push_back(t);
        }
        return v;
    }

    static std::uint64_t infosetId(const State& s) {
        const int p = s.player;
        std::uint64_t h = 0;
        for (char c : s.hist) {
            const std::uint64_t code =
                (c == 'f') ? 1u : (c == 'c') ? 2u : (c == 'r') ? 3u : 0u;
            h = h * 4 + code;
        }
        // 用 rank 而不是具体牌 id：同 rank 的两张牌在博弈上完全等价
        // （对手看不到花色，发牌分布相同），合并后信息集减半、收敛更快。
        const std::uint64_t myRank = static_cast<std::uint64_t>(rankOf(s.priv[p]));
        const std::uint64_t boardRank =
            (s.board >= 0) ? static_cast<std::uint64_t>(rankOf(s.board)) : 3u;
        return (static_cast<std::uint64_t>(p) << 56) | (myRank << 52) | (boardRank << 48) |
               (h + 1);
    }

    static std::string infosetName(const State& s) {
        static const char kRank[] = {'J', 'Q', 'K'};
        const int p = s.player;
        std::string name = "P";
        name += static_cast<char>('0' + p);
        name += " card=";
        name += (s.priv[p] >= 0) ? kRank[rankOf(s.priv[p])] : '?';
        name += " board=";
        name += (s.board >= 0) ? kRank[rankOf(s.board)] : '-';
        name += " hist=";
        name += s.hist;
        return name;
    }

    static double utilityP0(const State& s) {
        if (s.folded) {
            // 弃牌方输掉自己投入的部分，对手赢走弃牌方的投入
            return (s.folder == 1) ? static_cast<double>(s.contrib[1])
                                   : -static_cast<double>(s.contrib[0]);
        }
        const int rb = rankOf(s.board);
        const int r0 = rankOf(s.priv[0]);
        const int r1 = rankOf(s.priv[1]);
        const int s0 = (r0 == rb) ? 3 + r0 : r0;
        const int s1 = (r1 == rb) ? 3 + r1 : r1;
        if (s0 > s1) return static_cast<double>(s.contrib[0]);
        if (s0 < s1) return -static_cast<double>(s.contrib[0]);
        return 0.0;
    }

    static double utilityFrom(const State& s, int p) {
        const double u0 = utilityP0(s);
        return p == 0 ? u0 : -u0;
    }

    static std::vector<State> rootStates() {
        std::vector<State> v;
        v.reserve(kDeckSize * (kDeckSize - 1));
        for (int a = 0; a < kDeckSize; ++a) {
            for (int b = 0; b < kDeckSize; ++b) {
                if (a == b) continue;
                State s;
                s.priv[0] = a;
                s.priv[1] = b;
                v.push_back(s);
            }
        }
        return v;
    }

    static double rootProbability() {
        return 1.0 / static_cast<double>(kDeckSize * (kDeckSize - 1));
    }
};

}  // namespace texas
