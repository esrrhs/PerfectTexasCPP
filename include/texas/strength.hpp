#pragma once

// 牌力抽象（card abstraction），完全自包含、不依赖任何外部数据表。
//
// 无限注德州的状态空间太大，必须把"具体手牌"压成"牌力桶"才能让 CFR 收敛。
// 抽象分三段，精度损失逐段控制：
//
//  1) 牌力评估：card.hpp 的 evaluateBest，O(n) 快路径，37ns/次。
//
//  2) 翻牌前：169 类手牌（对子 / 同花 / 非同花）**全分辨率使用，不再分桶**。
//     这一点很关键——翻牌前本来就只有 169 个不同信息，再压成 10 个桶会把
//     AA 和 KK、AKs 和 AKo 合并成同一个信息集，纯属自毁信息。
//
//  3) 翻牌后：不能只看当前牌力排名。花听、顺听在"当前最佳五张"的排名里
//     只是高牌（百分位 0.2 左右），真实价值却接近中等成牌。所以用听牌出路
//     （outs）按超几何分布算出命中概率，对百分位做期望修正：
//
//        eff = (1 - p_hit) * 当前百分位 + p_hit * 命中后的百分位
//
//     p_hit 用精确的 outs / 未见牌数 计算（两张牌到来时用 1 - C(a,2)/C(b,2)），
//     命中后的百分位按出路构成加权（同花 ≈ 0.94，顺子 ≈ 0.89）。
//
// 百分位本身来自"该张数下最佳五张分数的真实分布"：5 张用 C(52,5) 精确枚举，
// 6/7 张用大样本蒙特卡洛估计，最后都压缩成"分数 -> 严格小于它的组合数"的精确
// 秩表（最佳五张的取值不到 7500 种，表只有几十 KB），查询时二分定位。

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "card.hpp"

namespace texas {

// 从 5~7 张里挑出最大的 5 张牌分数
inline int bestHandScore(const Card* c, int n) { return evaluateBest(c, n); }

// 听牌出路统计结果
struct DrawOuts {
    int flush = 0;     // 补成同花的牌数
    int straight = 0;  // 补成顺子的牌数
    int total = 0;     // 两者并集（按具体牌去重），即真正的 outs
};

class StrengthModel {
public:
    static constexpr int kPreflopClasses = 169;

    static StrengthModel& instance() {
        static StrengthModel model;
        return model;
    }

    // ---- 翻牌前：169 类，直接用类号当"桶"，不做任何合并 ----
    // 0-12 对子，13-90 同花，91-168 非同花
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

    // 对单个随机对手的胜率（平局算半胜）
    double preflopEquity(Card a, Card b) const { return preflopTable_[preflopClass(a, b)]; }

    // ---- 翻牌后 ----

    // 当前最佳五张在"同张数所有组合"中的排名，0（最弱）~1（最强）
    //
    // 这里返回的是**精确**百分位，不是分位点近似。7 张牌的最佳五张结果只有不到
    // 7500 种不同取值，所以"分数 -> 严格小于它的组合数"可以完整存下来（几十 KB）。
    // 早先用 4096 个分位点近似时，量化步长 1/4096 ≈ 0.0244% 比"四条"整类
    // （0.024%）还宽，四条和同花顺会被压成同一个百分位，进而同桶。
    double postflopPercentile(const Card* hole, const Card* board, int boardCount) const {
        Card all[7];
        int n = 0;
        all[n++] = hole[0];
        all[n++] = hole[1];
        for (int i = 0; i < boardCount; ++i) all[n++] = board[i];
        return percentileOfScore(bestHandScore(all, n), n);
    }

    // 听牌出路：只看"补成同花 / 补成顺子"这两类，它们是当前百分位最严重的低估项
    static DrawOuts computeOuts(const Card* hole, const Card* board, int boardCount) {
        int sc[4] = {};
        int rc[13] = {};
        std::uint64_t used = 0;
        const int n = 2 + boardCount;
        for (int i = 0; i < 2; ++i) {
            ++rc[cardRank(hole[i])];
            ++sc[cardSuit(hole[i])];
            used |= 1ULL << hole[i];
        }
        for (int i = 0; i < boardCount; ++i) {
            ++rc[cardRank(board[i])];
            ++sc[cardSuit(board[i])];
            used |= 1ULL << board[i];
        }

        Card all[7];
        int m = 0;
        all[m++] = hole[0];
        all[m++] = hole[1];
        for (int i = 0; i < boardCount; ++i) all[m++] = board[i];
        const int cat = handCategoryOf(bestHandScore(all, n));

        std::uint64_t flushCards = 0;
        std::uint64_t straightCards = 0;

        // 同花听牌：某个花色正好 4 张，且手里有该花色（否则是公牌的花，不算我们的）
        if (cat < static_cast<int>(HandCategory::Flush)) {
            for (int s = 0; s < 4; ++s) {
                if (sc[s] != 4) continue;
                if (cardSuit(hole[0]) != s && cardSuit(hole[1]) != s) continue;
                for (int r = 0; r < 13; ++r) {
                    const Card c = r * 4 + s;
                    if ((used >> c & 1) == 0) flushCards |= 1ULL << c;
                }
            }
        }

        // 顺子听牌：5 张连续 rank 的窗口里恰好缺 1 个 rank
        // （缺 1 个的两头张会同时命中两个窗口，自然算成 8 个 outs）
        if (cat < static_cast<int>(HandCategory::Straight)) {
            int rmask = 0;
            for (int r = 0; r < 13; ++r) {
                if (rc[r] > 0) rmask |= 1 << r;
            }
            auto addRankCards = [&](int r) {
                for (int s = 0; s < 4; ++s) {
                    const Card c = r * 4 + s;
                    if ((used >> c & 1) == 0) straightCards |= 1ULL << c;
                }
            };
            for (int high = 4; high <= 12; ++high) {
                const int run = 0x1F << (high - 4);
                const int miss = run & ~rmask;
                if (miss != 0 && (miss & (miss - 1)) == 0) {
                    addRankCards(static_cast<int>(__builtin_ctz(static_cast<unsigned>(miss))));
                }
            }
            const int wheel = 0x100F;  // A-2-3-4-5
            const int missW = wheel & ~rmask;
            if (missW != 0 && (missW & (missW - 1)) == 0) {
                addRankCards(static_cast<int>(__builtin_ctz(static_cast<unsigned>(missW))));
            }
        }

        DrawOuts out;
        out.flush = __builtin_popcountll(flushCards);
        out.straight = __builtin_popcountll(straightCards);
        out.total = __builtin_popcountll(flushCards | straightCards);
        return out;
    }

    // t 张牌到来时至少命中一次的概率（超几何，outs 视为不变）
    static double hitProbability(int outs, int unseen, int toCome) {
        if (outs <= 0 || unseen <= 0 || toCome <= 0) return 0.0;
        if (toCome >= 2) {
            const double a = static_cast<double>(unseen - outs);
            const double none = (a < 2.0) ? 0.0 : (a / unseen) * ((a - 1.0) / (unseen - 1.0));
            return 1.0 - none;
        }
        return std::min(1.0, static_cast<double>(outs) / static_cast<double>(unseen));
    }

    // 潜力修正后的牌力，0~1。河牌没有后续牌，直接退化成百分位。
    double postflopStrength(const Card* hole, const Card* board, int boardCount) const {
        const double pct = postflopPercentile(hole, board, boardCount);
        const int toCome = 5 - boardCount;
        if (toCome <= 0) return pct;

        const DrawOuts outs = computeOuts(hole, board, boardCount);
        if (outs.total <= 0) return pct;

        const int n = std::clamp(2 + boardCount, 5, 7);
        const int unseen = kNumCards - (2 + boardCount);
        const double pAny = hitProbability(outs.total, unseen, toCome);
        // 命中之后，牌力落在"该牌型的区间中段"（区间边界来自真实分布，不是拍脑袋）
        const double flushHit = catMid(n, HandCategory::Flush);
        const double straightHit = catMid(n, HandCategory::Straight);
        const double denom = static_cast<double>(outs.flush + outs.straight);
        const double hitValue =
            (denom > 0.0) ? (outs.flush * flushHit + outs.straight * straightHit) / denom : flushHit;
        const double eff = (1.0 - pAny) * pct + pAny * hitValue;
        return std::clamp(eff, 0.0, 1.0);
    }

    // 翻牌后分桶。
    //
    // 这里**不能**用等质量分位切桶。5 张牌里"两对及以上"只占 8.2%，20 个等质量
    // 桶下两对、三条、顺子、同花、葫芦、四条全部挤进最后一个桶——模型分不清
    // "超对"和"暗三条"，翻牌后策略直接废掉。而且这不是加迭代数能救的。
    //
    // 改为按牌型分配桶位：九个牌型各自拿到一段百分位区间，区间内部再等分成若干
    // 子桶。牌型边界不是写死的常数，而是构建分位表时从真实分数分布里取出来的
    // （见 buildScoreRank），所以 5/6/7 张各自不同。
    //
    // 因为作用在 postflopStrength 的返回值上，听牌被潜力修正抬起来之后可以跨过
    // 牌型边界往上走（花听会从中段一对的位置再高一些），不会被人为压在底层。
    int postflopBucket(const Card* hole, const Card* board, int boardCount, int buckets) const {
        const int n = std::clamp(2 + boardCount, 5, 7);
        return bucketOfStrength(postflopStrength(hole, board, boardCount), n, buckets);
    }

    // 把 [0,1] 的牌力映射到桶号（暴露出来方便测试与调试）
    int bucketOfStrength(double s, int numCards, int buckets) const {
        const int nc = std::clamp(numCards, 5, 7);
        ensureAllocation(buckets);
        const CatBounds& cb = catBounds_[nc - 5];
        const double x = std::clamp(s, 0.0, 1.0);
        for (int c = 0; c < kNumCategories; ++c) {
            const double lo = cb.lo[c];
            const double hi = cb.hi[c];
            if (x <= hi || c == kNumCategories - 1) {
                const double span = hi - lo;
                double within = (span > 0.0) ? (x - lo) / span : 1.0;
                within = std::clamp(within, 0.0, 1.0);
                int k = static_cast<int>(within * sub_[c]);
                if (k >= sub_[c]) k = sub_[c] - 1;
                if (k < 0) k = 0;
                return offset_[c] + k;
            }
        }
        return buckets - 1;
    }

    int bucketCount() const { return totalBuckets_; }

    // 某张数下某牌型的百分位区间（调试/测试用）
    double categoryLo(int numCards, HandCategory cat) const {
        return catBounds_[std::clamp(numCards, 5, 7) - 5].lo[static_cast<int>(cat)];
    }

    // 某个张数下"不同牌力取值"的个数（调试用）
    int distinctScores(int numCards) const {
        return static_cast<int>(ranks_[std::clamp(numCards, 5, 7) - 5].score.size());
    }

private:
    static constexpr int kPreflopSamples = 30000;
    static constexpr int kSample6 = 1500000;
    static constexpr int kSample7 = 2000000;
    static constexpr int kNumCategories = 9;
    // 各牌型分到的桶位权重（总和 100）。牌型越常见、越常做决策，权重越高。
    static constexpr int kCatWeight[kNumCategories] = {22, 26, 16, 10, 8, 7, 5, 3, 3};

    // 精确秩表：score 升序去重，cum[i] = 严格小于 score[i] 的组合数
    struct ScoreRanks {
        std::vector<int> score;
        std::vector<std::uint32_t> cum;
        std::uint32_t total = 0;
    };

    struct CatBounds {
        std::array<double, kNumCategories> lo{};
        std::array<double, kNumCategories> hi{};
    };

    StrengthModel() {
        buildPreflopTable();
        buildScoreRank(5);  // C(52,5) 精确枚举
        buildScoreRank(6);
        buildScoreRank(7);
        ensureAllocation(20);
    }

    double catMid(int numCards, HandCategory cat) const {
        const CatBounds& cb = catBounds_[std::clamp(numCards, 5, 7) - 5];
        const int c = static_cast<int>(cat);
        return 0.5 * (cb.lo[c] + cb.hi[c]);
    }

    // 按权重把 buckets 个桶位分配给九个牌型（最大余数法，总数精确等于 buckets）
    void ensureAllocation(int buckets) const {
        if (buckets < kNumCategories) buckets = kNumCategories;
        if (buckets == cachedBuckets_) return;
        cachedBuckets_ = buckets;
        totalBuckets_ = buckets;

        int extra[kNumCategories] = {};
        int frac[kNumCategories] = {};
        const int rem = buckets - kNumCategories;
        int assigned = 0;
        for (int c = 0; c < kNumCategories; ++c) {
            const int num = rem * kCatWeight[c];
            extra[c] = num / 100;
            frac[c] = num % 100;
            assigned += extra[c];
        }
        while (assigned < rem) {
            int bi = 0;
            for (int c = 1; c < kNumCategories; ++c) {
                if (frac[c] > frac[bi]) bi = c;
            }
            ++extra[bi];
            frac[bi] = -1;
            ++assigned;
        }

        int off = 0;
        for (int c = 0; c < kNumCategories; ++c) {
            sub_[c] = 1 + extra[c];
            offset_[c] = off;
            off += sub_[c];
        }
        offset_[kNumCategories] = off;
    }

    void buildPreflopTable();
    void buildScoreRank(int numCards);

    double percentileOfScore(int score, int numCards) const {
        const ScoreRanks& r = ranks_[std::clamp(numCards, 5, 7) - 5];
        if (r.total == 0) return 0.0;
        const auto it = std::lower_bound(r.score.begin(), r.score.end(), score);
        const std::size_t idx = static_cast<std::size_t>(it - r.score.begin());
        const std::uint32_t below = (idx < r.cum.size()) ? r.cum[idx] : r.total;
        return static_cast<double>(below) / static_cast<double>(r.total);
    }

    std::array<double, kPreflopClasses> preflopTable_{};
    std::array<ScoreRanks, 3> ranks_{};       // 5/6/7 张的精确秩表
    std::array<CatBounds, 3> catBounds_{};    // 5/6/7 张的牌型百分位边界

    mutable int cachedBuckets_ = -1;
    mutable int totalBuckets_ = 0;
    mutable std::array<int, kNumCategories> sub_{};
    mutable std::array<int, kNumCategories + 1> offset_{};
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
    // 169 类各取一个代表组合。同一类里的所有具体组合对随机对手的胜率完全相等
    // （花色置换对称），所以一个组合就够，误差只来自采样次数。
    std::mt19937_64 rng(20240518ULL);

    auto fill = [&](Card a, Card b) {
        const Card hole[2] = {a, b};
        double win = 0.0;
        for (int i = 0; i < kPreflopSamples; ++i) win += monteCarloWinOnce(hole, rng);
        preflopTable_[preflopClass(a, b)] = win / kPreflopSamples;
    };

    for (int r = 0; r < 13; ++r) fill(r * 4 + 2, r * 4 + 3);  // 对子
    for (int hi = 1; hi < 13; ++hi) {
        for (int lo = 0; lo < hi; ++lo) {
            fill(hi * 4 + 2, lo * 4 + 2);  // 同花
            fill(hi * 4 + 2, lo * 4 + 3);  // 非同花
        }
    }
}

inline void StrengthModel::buildScoreRank(int numCards) {
    std::vector<int> scores;

    if (numCards == 5) {
        // 5 张一共只有 2598960 种组合，直接穷举，分布是精确的
        scores.reserve(2598960);
        Card c[5];
        for (int a = 0; a < 52; ++a)
            for (int b = a + 1; b < 52; ++b)
                for (int d = b + 1; d < 52; ++d)
                    for (int e = d + 1; e < 52; ++e)
                        for (int f = e + 1; f < 52; ++f) {
                            c[0] = a; c[1] = b; c[2] = d; c[3] = e; c[4] = f;
                            scores.push_back(evaluateBest(c, 5));
                        }
    } else {
        const int samples = (numCards == 6) ? kSample6 : kSample7;
        scores.reserve(static_cast<std::size_t>(samples));
        std::mt19937_64 rng(0x5EED0000ULL + static_cast<std::uint64_t>(numCards));
        Card deck[kNumCards];
        for (int c = 0; c < kNumCards; ++c) deck[c] = c;
        for (int s = 0; s < samples; ++s) {
            for (int i = 0; i < numCards; ++i) {
                const int j = i + static_cast<int>(rng() % static_cast<std::uint64_t>(kNumCards - i));
                std::swap(deck[i], deck[j]);
            }
            scores.push_back(evaluateBest(deck, numCards));
        }
    }

    std::sort(scores.begin(), scores.end());
    const std::uint64_t total = static_cast<std::uint64_t>(scores.size());

    // 压缩成精确秩表：去重后的分数 + 每个分数之前有多少组合
    ScoreRanks& r = ranks_[numCards - 5];
    r.total = static_cast<std::uint32_t>(std::min<std::uint64_t>(total, 0xFFFFFFFFULL));
    r.score.clear();
    r.cum.clear();
    std::uint64_t below = 0;
    for (std::size_t i = 0; i < scores.size();) {
        const int v = scores[i];
        r.score.push_back(v);
        r.cum.push_back(static_cast<std::uint32_t>(std::min<std::uint64_t>(below, 0xFFFFFFFEULL)));
        std::size_t j = i;
        while (j < scores.size() && scores[j] == v) ++j;
        below += static_cast<std::uint64_t>(j - i);
        i = j;
    }

    // 九个牌型的百分位区间，直接由秩表得出（精确，不受任何量化影响）。
    // 分桶靠它给尾部牌型留出分辨率，而不是用等质量分位（那样两对以上全挤在末桶）。
    CatBounds& cb = catBounds_[numCards - 5];
    std::size_t start = 0;
    for (int c = 0; c < kNumCategories; ++c) {
        std::size_t i = start;
        while (i < r.score.size() && handCategoryOf(r.score[i]) == c) ++i;
        const std::uint64_t loBelow = (start < r.cum.size()) ? r.cum[start] : total;
        const std::uint64_t hiBelow = (i < r.cum.size()) ? r.cum[i] : total;
        cb.lo[c] = static_cast<double>(loBelow) / static_cast<double>(total);
        cb.hi[c] = static_cast<double>(hiBelow) / static_cast<double>(total);
        start = i;
    }
    cb.hi[kNumCategories - 1] = 1.0;  // 最顶端收口，避免浮点误差漏出 [0,1]
}

}  // namespace texas
