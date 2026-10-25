#pragma once

// 最优反应（best response）与可利用度（exploitability）。
//
// 直观理解：exploitability 是"一个专门针对该策略训练出来的对手，
// 平均每手能从它身上多赢多少"。均衡策略的 exploitability 为 0。
//
// 计算方法：对 BR 玩家做逐层策略提升。
// 每一轮遍历整棵树，按"对手到达概率"加权累积每个信息集里各动作的反事实
// 价值（不乘自己的到达概率），然后对每个信息集取 argmax。
// 由于是完美回忆博弈，深层信息集先定型、再逐层上传，
// 迭代 O(树深) 轮即可收敛到精确最优反应。

#include <algorithm>
#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "cfr.hpp"

namespace texas {

template <class G>
class BestResponse {
public:
    using State = typename G::State;

    // 返回 brPlayer 对 oppSigma 的最优反应收益（brPlayer 视角）
    static double value(const Strategy& oppSigma, int brPlayer, int rounds = 48) {
        Strategy brPol;
        Acc acc;

        const std::vector<State> roots = G::rootStates();
        const double rp = G::rootProbability();

        for (int r = 0; r < rounds; ++r) {
            acc.clear();
            for (const auto& s : roots) walk(s, brPlayer, rp, brPol, oppSigma, acc);
            for (const auto& kv : acc) {
                const int na = kv.second.n;
                int best = 0;
                for (int i = 1; i < na; ++i) {
                    if (kv.second.v[i] > kv.second.v[best]) best = i;
                }
                std::vector<double>& pol = brPol[kv.first];
                pol.assign(static_cast<std::size_t>(na), 0.0);
                pol[static_cast<std::size_t>(best)] = 1.0;
            }
        }

        acc.clear();
        double v = 0.0;
        for (const auto& s : roots) v += rp * walk(s, brPlayer, rp, brPol, oppSigma, acc);
        return v;
    }

private:
    struct Slot {
        std::array<double, 3> v{};
        int n = 0;
    };
    using Acc = std::unordered_map<InfoKey, Slot>;

    static double walk(const State& s,
                       int brPlayer,
                       double reachOpp,
                       const Strategy& brPol,
                       const Strategy& oppSigma,
                       Acc& acc) {
        if (G::isTerminal(s)) return G::utilityFrom(s, brPlayer);

        if (G::needsChance(s)) {
            const auto outs = G::chanceStates(s);
            const double p = 1.0 / static_cast<double>(outs.size());
            double v = 0.0;
            for (const auto& t : outs) {
                v += p * walk(t, brPlayer, reachOpp, brPol, oppSigma, acc);
            }
            return v;
        }

        std::array<char, 3> acts;
        const int na = G::actions(s, acts);
        const InfoKey key = G::infosetId(s);
        const int p = G::playerToAct(s);

        std::array<double, 3> vals{};
        if (p == brPlayer) {
            for (int i = 0; i < na; ++i) {
                vals[i] = walk(G::next(s, acts[i]), brPlayer, reachOpp, brPol, oppSigma, acc);
            }
            Slot& slot = acc[key];
            slot.n = na;
            for (int i = 0; i < na; ++i) slot.v[i] += reachOpp * vals[i];

            const auto it = brPol.find(key);
            if (it == brPol.end()) {
                double v = 0.0;
                for (int i = 0; i < na; ++i) v += vals[i];
                return v / static_cast<double>(na);
            }
            double v = 0.0;
            for (int i = 0; i < na; ++i) v += it->second[static_cast<std::size_t>(i)] * vals[i];
            return v;
        }

        const auto it = oppSigma.find(key);
        double v = 0.0;
        for (int i = 0; i < na; ++i) {
            const double pr =
                (it == oppSigma.end()) ? 1.0 / na : it->second[static_cast<std::size_t>(i)];
            if (pr <= 0.0) continue;
            v += pr * walk(G::next(s, acts[i]), brPlayer, reachOpp * pr, brPol, oppSigma, acc);
        }
        return v;
    }
};

// 两人零和：NashConv = br0 + br1（各自视角），exploitability = NashConv / 2
template <class G>
double exploitability(const Strategy& sigma, int rounds = 48) {
    const double br0 = BestResponse<G>::value(sigma, 0, rounds);
    const double br1 = BestResponse<G>::value(sigma, 1, rounds);
    return (br0 + br1) / 2.0;
}

}  // namespace texas
