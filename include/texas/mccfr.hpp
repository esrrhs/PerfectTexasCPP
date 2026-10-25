#pragma once

// 抽象无限注德州的 CFR 求解器。
//
// 翻牌有 C(48,3)=17296 种组合，全树遍历不可能，所以改用 external sampling MCCFR：
// 轮到 traverser 时展开全部动作，对手和机会节点各采样一个。
//
// 信息集 = (街, 牌力桶, 下注历史)。牌力桶由 strength.hpp 提供，
// 这才让"具体手牌"塌缩成有限个状态。

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "holdem.hpp"
#include "strength.hpp"

namespace texas {

struct HoldemAbstraction {
    int buckets = 10;

    static std::uint64_t hashHist(const std::string& hist) {
        std::uint64_t h = 1469598103934665603ULL;
        for (char c : hist) {
            h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
            h *= 1099511628211ULL;
        }
        return h;
    }

    int bucketOf(const HoldemGame::State& s, int p) const {
        const StrengthModel& m = StrengthModel::instance();
        if (s.boardCount < 3) {
            return m.preflopBucket(s.hole[p][0], s.hole[p][1], buckets);
        }
        return m.postflopBucket(s.hole[p], s.board, s.boardCount, buckets);
    }

    std::uint64_t infosetId(const HoldemGame::State& s) const {
        const int p = s.player;
        std::uint64_t h = hashHist(s.hist);
        h ^= static_cast<std::uint64_t>(bucketOf(s, p) + 1) * 0x9E3779B97F4A7C15ULL;
        h ^= static_cast<std::uint64_t>(s.street + 1) * 0xC2B2AE3D27D4EB4FULL;
        h ^= static_cast<std::uint64_t>(p + 1) * 0x165667B19E3779F9ULL;
        return h;
    }
};

class HoldemSolver {
public:
    struct InfoSet {
        std::array<double, HoldemGame::kMaxActions> regret{};
        std::array<double, HoldemGame::kMaxActions> avg{};
        int n = 0;
    };

    explicit HoldemSolver(int buckets = 10, std::uint64_t seed = 20240518ULL)
        : rng_(seed) {
        abstraction_.buckets = buckets;
    }

    void iterate(int rounds) {
        for (int i = 0; i < rounds; ++i) {
            ++iter_;
            HoldemGame::State root = HoldemGame::init(2);
            dealHole(root);
            // 交替更新，每轮只让一个玩家做 traverser
            const int traverser = iter_ & 1;
            traverse(root, traverser, 1.0);
        }
    }

    // 每个信息集的平均策略
    std::vector<double> averageStrategy(const HoldemGame::State& s) const {
        const auto it = isets_.find(abstraction_.infosetId(s));
        const int na = legalCount(s);
        std::vector<double> v(static_cast<std::size_t>(na), 1.0 / na);
        if (it == isets_.end() || it->second.n != na) return v;
        double sum = 0.0;
        for (int i = 0; i < na; ++i) {
            v[static_cast<std::size_t>(i)] = std::max(0.0, it->second.avg[i]);
            sum += v[static_cast<std::size_t>(i)];
        }
        if (sum > 0.0) {
            for (double& x : v) x /= sum;
        } else {
            for (double& x : v) x = 1.0 / na;
        }
        return v;
    }

    // 按平均策略采样一个动作索引
    int sampleAction(const HoldemGame::State& s) {
        const std::vector<double> probs = averageStrategy(s);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        const double u = dist(rng_);
        double acc = 0.0;
        for (std::size_t i = 0; i < probs.size(); ++i) {
            acc += probs[i];
            if (u <= acc) return static_cast<int>(i);
        }
        return static_cast<int>(probs.size()) - 1;
    }

    std::size_t numInfosets() const { return isets_.size(); }
    int iterations() const { return iter_; }
    const HoldemAbstraction& abstraction() const { return abstraction_; }

private:
    static int legalCount(const HoldemGame::State& s) {
        Move moves[HoldemGame::kMaxActions];
        return HoldemGame::actions(s, moves, HoldemGame::kMaxActions);
    }

    void dealHole(HoldemGame::State& s) {
        std::vector<Card> deck = HoldemGame::remainingDeck(s);
        for (int i = 0; i < s.numPlayers * 2; ++i) {
            std::uniform_int_distribution<int> dist(i, static_cast<int>(deck.size()) - 1);
            std::swap(deck[i], deck[dist(rng_)]);
        }
        int k = 0;
        for (int p = 0; p < s.numPlayers; ++p) {
            s.hole[p][0] = deck[k++];
            s.hole[p][1] = deck[k++];
        }
    }

    double traverse(const HoldemGame::State& s, int traverser, double reachT) {
        if (HoldemGame::isTerminal(s)) {
            return HoldemGame::utilityFrom(s, traverser);
        }
        if (HoldemGame::needsChance(s)) {
            HoldemGame::State t = s;
            HoldemGame::deal(t, rng_);
            return traverse(t, traverser, reachT);
        }

        const int p = s.player;
        if (p < 0) return 0.0;

        Move moves[HoldemGame::kMaxActions];
        const int na = HoldemGame::actions(s, moves, HoldemGame::kMaxActions);
        if (na <= 0) return HoldemGame::utilityFrom(s, traverser);

        const std::uint64_t key = abstraction_.infosetId(s);
        InfoSet& is = isets_[key];
        if (is.n != na) {
            if (is.n == 0) {
                is.n = na;
            } else {
                // 动作数变化（理论上同一信息集不会发生），保守处理
                is.n = na;
            }
        }

        std::array<double, HoldemGame::kMaxActions> sigma{};
        double sum = 0.0;
        for (int i = 0; i < na; ++i) {
            sigma[i] = std::max(0.0, is.regret[i]);
            sum += sigma[i];
        }
        if (sum > 0.0) {
            for (int i = 0; i < na; ++i) sigma[i] /= sum;
        } else {
            for (int i = 0; i < na; ++i) sigma[i] = 1.0 / na;
        }

        if (p == traverser) {
            double vals[HoldemGame::kMaxActions] = {};
            double nodeVal = 0.0;
            for (int i = 0; i < na; ++i) {
                vals[i] = traverse(HoldemGame::next(s, moves[i]), traverser, reachT * sigma[i]);
                nodeVal += sigma[i] * vals[i];
            }
            // CFR+：遗憾裁到非负
            for (int i = 0; i < na; ++i) {
                is.regret[i] = std::max(0.0, is.regret[i] + (vals[i] - nodeVal));
            }
            // 线性加权平均
            const double w = static_cast<double>(iter_);
            for (int i = 0; i < na; ++i) is.avg[i] += w * reachT * sigma[i];
            return nodeVal;
        }

        // 对手节点：按当前策略采样一个动作
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        const double u = dist(rng_);
        int pick = na - 1;
        double acc = 0.0;
        for (int i = 0; i < na; ++i) {
            acc += sigma[i];
            if (u <= acc) {
                pick = i;
                break;
            }
        }
        return traverse(HoldemGame::next(s, moves[pick]), traverser, reachT);
    }

    HoldemAbstraction abstraction_;
    std::unordered_map<std::uint64_t, InfoSet> isets_;
    std::mt19937_64 rng_;
    int iter_ = 0;
};

}  // namespace texas
