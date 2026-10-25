#pragma once

// 牌力抽象（card abstraction）。
//
// 无限注德州的状态空间太大，必须把"具体手牌"压成"牌力桶"才能让 CFR 收敛。
//
// 分两段处理：
//   翻牌前：169 类手牌（对子 / 同花 / 非同花），强度用蒙特卡洛对随机对手的胜率，
//           首次使用时惰性计算并缓存。
//   翻牌后：优先调用 texas_algorithm 的查表排名（GetWinProbability，约 280ns/次，
//           给出该手牌在所有同张数组合中的精确百分位）；未启用该库时回退到
//           evaluate7 的牌型 + 踢脚启发式。
//
// 启用查表后端：CMake 加 -DTEXAS_CFR_WITH_ALGORITHM=ON，并在运行时 load 数据表。

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "card.hpp"

#if defined(TEXAS_CFR_WITH_ALGORITHM)
#include "texas_algorithm_util.h"
#endif

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

    static bool tableBackendAvailable() {
#if defined(TEXAS_CFR_WITH_ALGORITHM)
        return texas_algorithm::IsLoaded();
#else
        return false;
#endif
    }

    int preflopBucket(Card a, Card b, int buckets) const {
        return toBucket(preflopTable_[preflopClass(a, b)], buckets);
    }

    int postflopBucket(const Card* hole, const Card* board, int boardCount, int buckets) const {
        return toBucket(postflopStrength(hole, board, boardCount), buckets);
    }

    double preflopStrength(Card a, Card b) const { return preflopTable_[preflopClass(a, b)]; }

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
    StrengthModel() { buildPreflopTable(); }

    static int toBucket(double s, int buckets) {
        if (buckets <= 1) return 0;
        int b = static_cast<int>(s * buckets);
        if (b < 0) b = 0;
        if (b >= buckets) b = buckets - 1;
        return b;
    }

    void buildPreflopTable();
    double postflopStrength(const Card* hole, const Card* board, int boardCount) const;

    std::array<double, 169> preflopTable_{};
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
    // 采样数不必很高：同类内部差异远小于采样噪声。
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

inline double StrengthModel::postflopStrength(const Card* hole, const Card* board,
                                              int boardCount) const {
#if defined(TEXAS_CFR_WITH_ALGORITHM)
    if (texas_algorithm::IsLoaded() && boardCount >= 3 && boardCount <= 5) {
        static const int kSuitToColor[4] = {1, 0, 2, 3};  // c,d,h,s -> 梅,方,红,黑
        std::vector<int> pokes;
        pokes.reserve(static_cast<std::size_t>(boardCount) + 2);
        for (int i = 0; i < 2; ++i) {
            const int value = cardRank(hole[i]) + 2;
            const int color = kSuitToColor[cardSuit(hole[i])];
            pokes.push_back((color << 4) | value);
        }
        for (int i = 0; i < boardCount; ++i) {
            const int value = cardRank(board[i]) + 2;
            const int color = kSuitToColor[cardSuit(board[i])];
            pokes.push_back((color << 4) | value);
        }
        const double p = texas_algorithm::GetWinProbability(pokes);
        return std::clamp(p, 0.0, 1.0);
    }
#endif
    // 回退：牌型类别占主要权重，踢脚做微调
    Card all[7];
    int n = 0;
    all[n++] = hole[0];
    all[n++] = hole[1];
    for (int i = 0; i < boardCount; ++i) all[n++] = board[i];
    const int score = bestHandScore(all, n);
    const int cat = handCategoryOf(score);
    const int tie = score % kScoreBase;
    const double base = static_cast<double>(cat) / 8.0 * 0.85;
    const double kick = static_cast<double>(tie) / static_cast<double>(kScoreBase) * 0.15;
    return base + kick;
}

}  // namespace texas
