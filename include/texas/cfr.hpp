#pragma once

// 通用 CFR / CFR+ 求解器。
//
// 游戏通过模板参数 G 提供静态接口：
//   G::State                      状态类型
//   G::isTerminal(s)              是否终局
//   G::needsChance(s)             是否需要展开机会节点
//   G::chanceStates(s)            机会节点的所有后继
//   G::playerToAct(s)             当前行动玩家
//   G::actions(s, out)            合法动作字符，返回个数
//   G::next(s, a)                 执行动作后的状态
//   G::infosetId(s)               整数信息集键
//   G::infosetName(s)             可读信息集名（仅首次出现时调用）
//   G::utilityFrom(s, p)          终局收益（p 的视角）
//   G::rootStates() / rootProbability()
//
// 用整数键而不是字符串，是为了避免每次递归都构造 std::string。

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace texas {

using InfoKey = std::uint64_t;
using Strategy = std::unordered_map<InfoKey, std::vector<double>>;

struct InfoSet {
    std::array<double, 3> regret{};
    std::array<double, 3> avg{};
    std::array<char, 3> actChars{};
    int n = 0;
};

template <class G>
class CFRSolver {
public:
    using State = typename G::State;

    struct Options {
        bool plus = true;       // CFR+：把遗憾裁剪到非负
        bool linearAvg = true;  // 线性加权平均策略
        Options() = default;
        Options(bool plusFlag, bool linearAvgFlag)
            : plus(plusFlag), linearAvg(linearAvgFlag) {}
    };

    CFRSolver() = default;
    explicit CFRSolver(Options opt) : opt_(opt) {}

    void iterate(int n) {
        if (roots_.empty()) {
            roots_ = G::rootStates();
            rootProb_ = G::rootProbability();
            names_.reserve(1024);
        }
        for (int i = 0; i < n; ++i) {
            ++iter_;
            // 交替更新（CFR+ 的标配）：先让 0 号玩家当 traverser，再让 1 号
            for (int p = 0; p < 2; ++p) {
                for (const auto& s : roots_) {
                    cfr(s, p, rootProb_, rootProb_);
                }
            }
        }
    }

    // 平均策略（归一化后），这才是收敛到的解
    Strategy averageStrategy() const {
        Strategy out;
        out.reserve(isets_.size());
        for (const auto& kv : isets_) {
            const InfoSet& is = kv.second;
            std::vector<double> v(static_cast<std::size_t>(is.n), 0.0);
            double sum = 0.0;
            for (int i = 0; i < is.n; ++i) {
                v[i] = is.avg[i];
                sum += v[i];
            }
            if (sum > 0.0) {
                for (double& x : v) x /= sum;
            } else {
                for (double& x : v) x = 1.0 / static_cast<double>(is.n);
            }
            out.emplace(kv.first, std::move(v));
        }
        return out;
    }

    // 当前（最后一轮）的遗憾匹配策略，一般不用于实战
    Strategy currentStrategy() const {
        Strategy out;
        out.reserve(isets_.size());
        for (const auto& kv : isets_) {
            const InfoSet& is = kv.second;
            std::vector<double> v(static_cast<std::size_t>(is.n), 0.0);
            double sum = 0.0;
            for (int i = 0; i < is.n; ++i) {
                v[i] = std::max(0.0, is.regret[i]);
                sum += v[i];
            }
            if (sum > 0.0) {
                for (double& x : v) x /= sum;
            } else {
                for (double& x : v) x = 1.0 / static_cast<double>(is.n);
            }
            out.emplace(kv.first, std::move(v));
        }
        return out;
    }

    // 用给定策略遍历博弈树，返回 player 0 视角的期望收益
    double expectedValue(const Strategy& sigma) const {
        double v = 0.0;
        for (const auto& s : G::rootStates()) {
            v += G::rootProbability() * evalValue(s, sigma, 0);
        }
        return v;
    }

    int iterations() const { return iter_; }
    std::size_t numInfosets() const { return isets_.size(); }

    const std::unordered_map<InfoKey, std::string>& names() const { return names_; }

    const std::unordered_map<InfoKey, InfoSet>& infosets() const { return isets_; }

private:
    static double evalValue(const State& s, const Strategy& sigma, int view) {
        if (G::isTerminal(s)) return G::utilityFrom(s, view);
        if (G::needsChance(s)) {
            const auto outs = G::chanceStates(s);
            const double p = 1.0 / static_cast<double>(outs.size());
            double v = 0.0;
            for (const auto& t : outs) v += p * evalValue(t, sigma, view);
            return v;
        }
        std::array<char, 3> acts;
        const int na = G::actions(s, acts);
        const auto it = sigma.find(G::infosetId(s));
        double v = 0.0;
        for (int i = 0; i < na; ++i) {
            const double pr = (it == sigma.end()) ? 1.0 / na : it->second[i];
            if (pr <= 0.0) continue;
            v += pr * evalValue(G::next(s, acts[i]), sigma, view);
        }
        return v;
    }

    double cfr(const State& s, int traverser, double reachT, double reachO) {
        if (G::isTerminal(s)) return G::utilityFrom(s, traverser);

        if (G::needsChance(s)) {
            const auto outs = G::chanceStates(s);
            const double p = 1.0 / static_cast<double>(outs.size());
            double v = 0.0;
            for (const auto& t : outs) {
                v += p * cfr(t, traverser, reachT * p, reachO * p);
            }
            return v;
        }

        std::array<char, 3> acts;
        const int na = G::actions(s, acts);
        const InfoKey key = G::infosetId(s);
        InfoSet& is = isets_[key];
        if (is.n == 0) {
            is.n = na;
            for (int i = 0; i < na; ++i) is.actChars[i] = acts[i];
            names_.emplace(key, G::infosetName(s));
        }

        // 遗憾匹配：σ ∝ max(R, 0)
        std::array<double, 3> sigma{};
        double sum = 0.0;
        for (int i = 0; i < na; ++i) {
            sigma[i] = std::max(0.0, is.regret[i]);
            sum += sigma[i];
        }
        if (sum > 0.0) {
            for (int i = 0; i < na; ++i) sigma[i] /= sum;
        } else {
            const double u = 1.0 / static_cast<double>(na);
            for (int i = 0; i < na; ++i) sigma[i] = u;
        }

        const int p = G::playerToAct(s);
        std::array<double, 3> vals{};
        double nodeVal = 0.0;
        for (int i = 0; i < na; ++i) {
            const State t = G::next(s, acts[i]);
            if (p == traverser) {
                vals[i] = cfr(t, traverser, reachT * sigma[i], reachO);
            } else {
                vals[i] = cfr(t, traverser, reachT, reachO * sigma[i]);
            }
            nodeVal += sigma[i] * vals[i];
        }

        if (p == traverser) {
            // 反事实遗憾：只乘对手的到达概率，不乘自己的
            for (int i = 0; i < na; ++i) {
                const double r = vals[i] - nodeVal;
                if (opt_.plus) {
                    is.regret[i] = std::max(0.0, is.regret[i] + reachO * r);
                } else {
                    is.regret[i] += reachO * r;
                }
            }
            // 平均策略按自己的到达概率加权
            const double w = opt_.linearAvg ? static_cast<double>(iter_) : 1.0;
            for (int i = 0; i < na; ++i) {
                is.avg[i] += w * reachT * sigma[i];
            }
        }
        return nodeVal;
    }

    Options opt_{};
    std::unordered_map<InfoKey, InfoSet> isets_;
    std::unordered_map<InfoKey, std::string> names_;
    std::vector<State> roots_;
    double rootProb_ = 1.0;
    int iter_ = 0;
};

}  // namespace texas
