#pragma once

// 52 张牌的表示与牌力评估。
// card = rank * 4 + suit，rank: 0=2 ... 12=A，suit: 0=c 1=d 2=h 3=s

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

namespace texas {

using Card = int;

constexpr int kNumCards = 52;
constexpr int kScoreBase = 1048576;  // 16^5，用于从分数里取回牌型类别

inline int cardRank(Card c) { return c >> 2; }
inline int cardSuit(Card c) { return c & 3; }

inline std::string cardToString(Card c) {
    static const char kRanks[] = "23456789TJQKA";
    static const char kSuits[] = "cdhs";
    if (c < 0 || c >= kNumCards) return "--";
    return std::string(1, kRanks[cardRank(c)]) + kSuits[cardSuit(c)];
}

// 解析 "Ah" / "Td" / "2c" 这样的字符串，失败返回 -1
inline Card parseCard(const std::string& s) {
    if (s.size() != 2) return -1;
    static const char kRanks[] = "23456789TJQKA";
    static const char kSuits[] = "cdhs";
    int r = -1, u = -1;
    for (int i = 0; i < 13; ++i) {
        if (kRanks[i] == s[0] || (s[0] == 't' && kRanks[i] == 'T')) {
            r = i;
            break;
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (kSuits[i] == s[1]) {
            u = i;
            break;
        }
    }
    if (r < 0 || u < 0) return -1;
    return r * 4 + u;
}

enum class HandCategory : int {
    HighCard = 0,
    OnePair,
    TwoPair,
    ThreeOfAKind,
    Straight,
    Flush,
    FullHouse,
    FourOfAKind,
    StraightFlush,
};

// 5 张牌 -> 可比较的整数（越大越强）
inline int evaluate5(const Card* c) {
    int rc[13] = {};
    int sc[4] = {};
    for (int i = 0; i < 5; ++i) {
        ++rc[cardRank(c[i])];
        ++sc[cardSuit(c[i])];
    }

    const bool flush = (sc[0] == 5 || sc[1] == 5 || sc[2] == 5 || sc[3] == 5);

    int distinct = 0;
    int lo = -1;
    int hi = -1;
    for (int r = 0; r < 13; ++r) {
        if (rc[r] > 0) {
            ++distinct;
            if (lo < 0) lo = r;
            hi = r;
        }
    }

    bool straight = false;
    int straightHigh = -1;
    if (distinct == 5) {
        if (hi - lo == 4) {
            straight = true;
            straightHigh = hi;
        } else if (rc[12] && rc[0] && rc[1] && rc[2] && rc[3]) {
            // A-2-3-4-5，5 为高牌
            straight = true;
            straightHigh = 3;
        }
    }

    struct RankCount {
        int rank;
        int count;
    };
    RankCount list[5];
    int n = 0;
    for (int r = 12; r >= 0; --r) {
        if (rc[r] > 0) list[n++] = RankCount{r, rc[r]};
    }
    std::sort(list, list + n, [](const RankCount& a, const RankCount& b) {
        if (a.count != b.count) return a.count > b.count;
        return a.rank > b.rank;
    });

    int cat = 0;
    int vals[5] = {0, 0, 0, 0, 0};

    if (straight && flush) {
        cat = static_cast<int>(HandCategory::StraightFlush);
        vals[0] = straightHigh;
    } else if (list[0].count == 4) {
        cat = static_cast<int>(HandCategory::FourOfAKind);
        vals[0] = list[0].rank;
        vals[1] = list[1].rank;
    } else if (list[0].count == 3 && list[1].count == 2) {
        cat = static_cast<int>(HandCategory::FullHouse);
        vals[0] = list[0].rank;
        vals[1] = list[1].rank;
    } else if (flush) {
        cat = static_cast<int>(HandCategory::Flush);
        for (int i = 0; i < 5; ++i) vals[i] = list[i].rank;
    } else if (straight) {
        cat = static_cast<int>(HandCategory::Straight);
        vals[0] = straightHigh;
    } else if (list[0].count == 3) {
        cat = static_cast<int>(HandCategory::ThreeOfAKind);
        vals[0] = list[0].rank;
        vals[1] = list[1].rank;
        vals[2] = list[2].rank;
    } else if (list[0].count == 2 && list[1].count == 2) {
        cat = static_cast<int>(HandCategory::TwoPair);
        vals[0] = list[0].rank;
        vals[1] = list[1].rank;
        vals[2] = list[2].rank;
    } else if (list[0].count == 2) {
        cat = static_cast<int>(HandCategory::OnePair);
        for (int i = 0; i < 4; ++i) vals[i] = list[i].rank;
    } else {
        cat = static_cast<int>(HandCategory::HighCard);
        for (int i = 0; i < 5; ++i) vals[i] = list[i].rank;
    }

    int score = cat;
    for (int i = 0; i < 5; ++i) score = score * 16 + vals[i];
    return score;
}

inline int handCategoryOf(int score) { return score / kScoreBase; }

// 7 选 5 的 21 种组合索引
inline const std::array<std::array<int, 5>, 21>& sevenCardMasks() {
    static const std::array<std::array<int, 5>, 21> kTable = [] {
        std::array<std::array<int, 5>, 21> t{};
        int k = 0;
        for (int m = 0; m < 128 && k < 21; ++m) {
            int bits = 0;
            for (int i = 0; i < 7; ++i) {
                if (m & (1 << i)) ++bits;
            }
            if (bits != 5) continue;
            int j = 0;
            for (int i = 0; i < 7; ++i) {
                if (m & (1 << i)) t[k][j++] = i;
            }
            ++k;
        }
        return t;
    }();
    return kTable;
}

// 暴力参考实现：枚举 21 种 5 张组合取最大。保留下来给测试做等价性对照。
inline int evaluate7Brute(const Card* c) {
    int best = -1;
    for (const auto& m : sevenCardMasks()) {
        const Card h[5] = {c[m[0]], c[m[1]], c[m[2]], c[m[3]], c[m[4]]};
        const int s = evaluate5(h);
        if (s > best) best = s;
    }
    return best;
}

// ---------------------------------------------------------------- 快速评估
//
// 上面 evaluate7Brute 每次要做 21 次 evaluate5，而 evaluate5 内部还有一次
// std::sort 和两次栈上数组清零，实测 734ns/次。牌力评估是蒙特卡洛采样、
// 牌力表构建、胜率计算的公共热点，必须压下来。
//
// 快路径只用一次 O(n) 扫描建两张计数表，然后按牌型优先级逐条判定：
//   - 顺子：13 位 rank 掩码查 8192 项的表，O(1)
//   - 同花：花色计数 ≥5 即有，取的 5 张牌名直接下降扫描掩码
//   - 对子/三条/葫芦/四条：从 rank 计数表按降序收集
// 分数编码与 evaluate5 完全一致（cat * 16^5 + 递减 kicker），所以两者可以
// 直接互相比较，handCategoryOf 也照旧可用。

// mask 的 bit r 表示存在 rank r 的牌，返回该掩码里最大顺子的高牌 rank（无则 -1）
inline const std::array<std::int16_t, 8192>& straightHighTable() {
    static const std::array<std::int16_t, 8192> kTable = [] {
        std::array<std::int16_t, 8192> t{};
        for (int m = 0; m < 8192; ++m) {
            int sh = -1;
            for (int high = 12; high >= 4; --high) {
                const int run = 0x1F << (high - 4);
                if ((m & run) == run) {
                    sh = high;
                    break;
                }
            }
            // A-2-3-4-5：5 当高牌（rank 3）
            if (sh < 0 && (m & 0xF) == 0xF && (m & 0x1000) != 0) sh = 3;
            t[static_cast<std::size_t>(m)] = static_cast<std::int16_t>(sh);
        }
        return t;
    }();
    return kTable;
}

inline int packScore(int cat, const int* vals) {
    int s = cat;
    for (int i = 0; i < 5; ++i) s = s * 16 + vals[i];
    return s;
}

// n ∈ [5, 7]，分数与 max over C(n,5) 个 evaluate5 完全一致
inline int evaluateBest(const Card* c, int n) {
    int rc[13] = {};
    int sc[4] = {};
    int sm[4] = {};
    for (int i = 0; i < n; ++i) {
        const int r = c[i] >> 2;
        const int s = c[i] & 3;
        ++rc[r];
        ++sc[s];
        sm[s] |= 1 << r;
    }

    int quad = -1;
    int trip1 = -1, trip2 = -1;
    int pair1 = -1, pair2 = -1;
    int rmask = 0;
    for (int r = 12; r >= 0; --r) {
        const int cnt = rc[r];
        if (cnt == 0) continue;
        rmask |= 1 << r;
        if (cnt == 4) {
            if (quad < 0) quad = r;
        } else if (cnt == 3) {
            if (trip1 < 0) trip1 = r;
            else if (trip2 < 0) trip2 = r;
        } else if (cnt == 2) {
            if (pair1 < 0) pair1 = r;
            else if (pair2 < 0) pair2 = r;
        }
    }

    // 四条
    if (quad >= 0) {
        int kicker = -1;
        for (int r = 12; r >= 0; --r) {
            if (r != quad && rc[r] > 0) { kicker = r; break; }
        }
        const int v[5] = {quad, kicker < 0 ? 0 : kicker, 0, 0, 0};
        return packScore(static_cast<int>(HandCategory::FourOfAKind), v);
    }

    // 葫芦（两个三条也算）
    if (trip1 >= 0 && (trip2 >= 0 || pair1 >= 0)) {
        const int pairPart = std::max(trip2, pair1);
        const int v[5] = {trip1, pairPart, 0, 0, 0};
        return packScore(static_cast<int>(HandCategory::FullHouse), v);
    }

    // 同花 / 同花顺
    int flushSuit = -1;
    for (int s = 0; s < 4; ++s) {
        if (sc[s] >= 5) { flushSuit = s; break; }
    }
    if (flushSuit >= 0) {
        const int fm = sm[flushSuit];
        const int sh = straightHighTable()[static_cast<std::size_t>(fm)];
        if (sh >= 0) {
            const int v[5] = {sh, 0, 0, 0, 0};
            return packScore(static_cast<int>(HandCategory::StraightFlush), v);
        }
        int v[5] = {0, 0, 0, 0, 0};
        int k = 0;
        for (int r = 12; r >= 0 && k < 5; --r) {
            if ((fm >> r) & 1) v[k++] = r;
        }
        return packScore(static_cast<int>(HandCategory::Flush), v);
    }

    // 顺子
    const int sh = straightHighTable()[static_cast<std::size_t>(rmask)];
    if (sh >= 0) {
        const int v[5] = {sh, 0, 0, 0, 0};
        return packScore(static_cast<int>(HandCategory::Straight), v);
    }

    // 三条
    if (trip1 >= 0) {
        int v[5] = {trip1, 0, 0, 0, 0};
        int k = 1;
        for (int r = 12; r >= 0 && k < 3; --r) {
            if (r != trip1 && rc[r] > 0) v[k++] = r;
        }
        return packScore(static_cast<int>(HandCategory::ThreeOfAKind), v);
    }

    // 两对
    if (pair1 >= 0 && pair2 >= 0) {
        int v[5] = {pair1, pair2, 0, 0, 0};
        for (int r = 12; r >= 0; --r) {
            if (r != pair1 && r != pair2 && rc[r] > 0) { v[2] = r; break; }
        }
        return packScore(static_cast<int>(HandCategory::TwoPair), v);
    }

    // 一对
    if (pair1 >= 0) {
        int v[5] = {pair1, 0, 0, 0, 0};
        int k = 1;
        for (int r = 12; r >= 0 && k < 4; --r) {
            if (r != pair1 && rc[r] > 0) v[k++] = r;
        }
        return packScore(static_cast<int>(HandCategory::OnePair), v);
    }

    // 高牌
    int v[5] = {0, 0, 0, 0, 0};
    int k = 0;
    for (int r = 12; r >= 0 && k < 5; --r) {
        if (rc[r] > 0) v[k++] = r;
    }
    return packScore(static_cast<int>(HandCategory::HighCard), v);
}

inline int evaluate6(const Card* c) { return evaluateBest(c, 6); }
inline int evaluate7(const Card* c) { return evaluateBest(c, 7); }

}  // namespace texas
