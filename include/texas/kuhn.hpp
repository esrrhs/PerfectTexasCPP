#pragma once

// Kuhn Poker：3 张牌（J/Q/K），各发 1 张，各放 1 注，只进行一轮下注。
// 动作：'p' = pass（check 或 fold），'b' = bet 或 call。
// 理论性质：先手（player 0）的博弈值为 -1/18，共 12 个信息集。
// 用来验证 CFR 实现是否正确 —— 有解析解可以硬对照。

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace texas {

struct KuhnGame {
    static constexpr int kDeckSize = 3;

    struct State {
        int priv[2] = {0, 0};
        std::string hist;
    };

    static bool isTerminal(const State& s) {
        const std::string& h = s.hist;
        return h == "pp" || h == "bp" || h == "bb" || h == "pbp" || h == "pbb";
    }

    static bool needsChance(const State&) { return false; }

    // Kuhn 没有机会节点，但模板需要这个接口存在
    static std::vector<State> chanceStates(const State&) { return {}; }

    static int playerToAct(const State& s) {
        return static_cast<int>(s.hist.size() & 1u);
    }

    static int actions(const State&, std::array<char, 3>& out) {
        out[0] = 'p';
        out[1] = 'b';
        return 2;
    }

    static State next(const State& s, char a) {
        State t = s;
        t.hist.push_back(a);
        return t;
    }

    // 整数信息集 ID：player | 自己的牌 | 下注历史
    static std::uint64_t infosetId(const State& s) {
        const int p = playerToAct(s);
        std::uint64_t h = 0;
        for (char c : s.hist) h = h * 3 + (c == 'b' ? 1u : 2u);
        return (static_cast<std::uint64_t>(p) << 40) |
               (static_cast<std::uint64_t>(s.priv[p]) << 32) | (h + 1);
    }

    static std::string infosetName(const State& s) {
        static const char kCards[] = {'J', 'Q', 'K'};
        const int p = playerToAct(s);
        std::string name = "P";
        name += static_cast<char>('0' + p);
        name += " card=";
        name += kCards[s.priv[p]];
        name += " hist=";
        name += s.hist.empty() ? "\"\"" : s.hist;
        return name;
    }

    // 以 player 0 为视角的收益
    static double utilityP0(const State& s) {
        const std::string& h = s.hist;
        const bool p0Higher = s.priv[0] > s.priv[1];
        if (h == "pp") return p0Higher ? 1.0 : -1.0;
        if (h == "bp") return 1.0;    // P1 弃牌
        if (h == "pbp") return -1.0;  // P0 弃牌
        if (h == "bb" || h == "pbb") return p0Higher ? 2.0 : -2.0;
        return 0.0;
    }

    static double utilityFrom(const State& s, int p) {
        const double u0 = utilityP0(s);
        return p == 0 ? u0 : -u0;
    }

    static std::vector<State> rootStates() {
        std::vector<State> v;
        v.reserve(6);
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

    static double rootProbability() { return 1.0 / 6.0; }
};

}  // namespace texas
