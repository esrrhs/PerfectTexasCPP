#pragma once

// 牌力抽象（card abstraction），完全自包含、不依赖任何外部数据表。
//
// 无限注德州的状态空间太大，必须把"具体手牌"压成"牌力桶"才能让 CFR 收敛。
//
// 分两段处理：
//   翻牌前：169 类手牌（对子 / 同花 / 非同花），强度用蒙特卡洛对随机对手的胜率。
//   翻牌后：按张数统计 5/6/7 张牌"最佳五张"分数的真实分布，用分位点把分数
//           映射成百分位，再切桶。这等价于"该手牌在所有同张数组合中的排名"，
//           只是分布靠自己采样估计，不需要任何预生成的数据文件。
//
// 两张表都是首次使用时惰性构建（约 0.6 秒），之后 O(log N) 查询。

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "card.hpp"

namespace texas {

// 从 5~7 张里挑出最大的 5 张牌分数
inline int bestHandScore(const Card* c, int n) {
    if (n == 5) return evaluate5(c);
    if (n == 7) return evaluate7(c);
    if (n == 6) {
        int best = -1;
        for (int skip = 0; skip < 6; ++skip) {
            Card h5[5];
            int k = 0;
            for (int i = 0; i < 6; ++i) {
                if (i != skip) h5[k++] = c[i];
            }
            best = std::max(best, evaluate5(h5));
        }
        return best;
    }
    return 0;
}

class StrengthModel {
public:
    static StrengthModel& instance() {
        static StrengthModel model;
        return model;
    }

    int preflopBucket(Card a, Card b, int buckets) const {
        return toBucket(preflopTable_[preflopClass(a, b)], buckets);
    }

    // 翻牌后（board 3~5 张）
    int postflopBucket(const Card* hole, const Card* board, int boardCount, int buckets) const {
        return toBucket(postflopPercentile(hole, board, boardCount), buckets);
    }

    double preflopStrength(Card a, Card b) const { return preflopTable_[preflopClass(a, b)]; }

    // 翻牌后的百分位：0（最弱）~1（最强）
    double postflopPercentile(const Card* hole, const Card* board, int boardCount) const {
        Card all[7];
        int n = 0;
        all[n++] = hole[0];
        all[n++] = hole[1];
        for (int i = 0; i < boardCount; ++i) all[n++] = board[i];
        const int score = bestHandScore(all, n);
        const Quantiles& q = quantiles_[std::clamp(n, 5, 7) - 5];
        // 二分找第一个 >= score 的分位点
        const int idx = static_cast<int>(
            std::lower_bound(q.points.begin(), q.points.begin() + q.count, score) -
            q.points.begin());
        return static_cast<double>(idx) / static_cast<double>(q.count);
    }

    // 169 类编码：0-12 对子，13-90 同花，91-168 非同花
    static int preflopClass(Card a, Card b) {
        const int hi = std::max(cardRank(a), cardRank(b));
        const int lo = std::min(cardRank(a), cardRank(b));
        if (hi == lo) return hi;
        const bool suited = cardSuit(a) == cardSuit(b);
        int idx = 0;
        for (int h = 1; h < 13; ++h) {
            for (int l = 0; l < h; ++l) {
                if (h == hi && l == lo) return suited ? 13 + idx : 91 + idx;
                ++idx;
            }
        }
        return 0;
    }

private:
    static constexpr int kQuantiles = 2048;
    static constexpr int kSamples = 400000;

    struct Quantiles {
        std::array<int, kQuantiles> points{};
        int count = 0;
    };

    StrengthModel() {
        buildPreflopTable();
        for (int n = 5; n <= 7; ++n) buildQuantiles(n);
    }

    static int toBucket(double s, int buckets) {
        if (buckets <= 1) return 0;
        int b = static_cast<int>(s * buckets);
        if (b < 0) b = 0;
        if (b >= buckets) b = buckets - 1;
        return b;
    }

    void buildPreflopTable();
    void buildQuantiles(int numCards);

    std::array<double, 169> preflopTable_{};
    std::array<Quantiles, 3> quantiles_{};  // 5/6/7 张
};

// ---------------- 实现 ----------------

// 单次蒙特卡洛：2 张手牌 vs 随机对手，发完 5 张公共牌
inline double monteCarloWinOnce(const Card* hole, std::mt19937_64& rng) {
    bool used[kNumCards] = {};
    used[hole[0]] = true;
    used[hole[1]] = true;
    Card deck[50];
    int n = 0;
    for (int c = 0; c < kNumCards; ++c) {
        if (!used[c]) deck[n++] = c;
    }
    for (int i = 0; i < 7; ++i) {
        const int j = i + static_cast<int>(rng() % static_cast<std::uint64_t>(n - i));
        std::swap(deck[i], deck[j]);
    }
    Card board[5] = {deck[2], deck[3], deck[4], deck[5], deck[6]};
    const Card h1[7] = {hole[0], hole[1], board[0], board[1], board[2], board[3], board[4]};
    const Card h2[7] = {deck[0], deck[1], board[0], board[1], board[2], board[3], board[4]};
    const int s1 = evaluate7(h1);
    const int s2 = evaluate7(h2);
    return (s1 > s2) ? 1.0 : ((s1 < s2) ? 0.0 : 0.5);
}

inline void StrengthModel::buildPreflopTable() {
    // 169 类各取一个代表组合，估对随机对手的胜率。
    constexpr int kIters = 800;
    std::mt19937_64 rng(20240518ULL);

    auto fill = [&](Card a, Card b) {
        const Card hole[2] = {a, b};
        double win = 0.0;
        for (int i = 0; i < kIters; ++i) win += monteCarloWinOnce(hole, rng);
        preflopTable_[preflopClass(a, b)] = win / kIters;
    };

    for (int r = 0; r < 13; ++r) fill(r * 4 + 2, r * 4 + 3);  // 对子
    for (int hi = 1; hi < 13; ++hi) {
        for (int lo = 0; lo < hi; ++lo) {
            fill(hi * 4 + 2, lo * 4 + 2);  // 同花
            fill(hi * 4 + 2, lo * 4 + 3);  // 非同花
        }
    }
}

inline void StrengthModel::buildQuantiles(int numCards) {
    Quantiles& q = quantiles_[numCards - 5];
    std::mt19937_64 rng(0x5EED0000ULL + static_cast<std::uint64_t>(numCards));

    std::vector<int> scores;
    scores.reserve(kSamples);

    Card deck[kNumCards];
    for (int c = 0; c < kNumCards; ++c) deck[c] = c;

    for (int s = 0; s < kSamples; ++s) {
        for (int i = 0; i < numCards; ++i) {
            const int j = i + static_cast<int>(rng() % static_cast<std::uint64_t>(kNumCards - i));
            std::swap(deck[i], deck[j]);
        }
        scores.push_back(bestHandScore(deck, numCards));
    }
    std::sort(scores.begin(), scores.end());

    q.count = kQuantiles;
    for (int i = 0; i < kQuantiles; ++i) {
        const std::size_t at =
            static_cast<std::size_t>(static_cast<double>(i) * kSamples / kQuantiles);
        q.points[i] = scores[std::min(at, scores.size() - 1)];
    }
}

}  // namespace texas
