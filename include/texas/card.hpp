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

inline int evaluate7(const Card* c) {
    int best = -1;
    for (const auto& m : sevenCardMasks()) {
        const Card h[5] = {c[m[0]], c[m[1]], c[m[2]], c[m[3]], c[m[4]]};
        const int s = evaluate5(h);
        if (s > best) best = s;
    }
    return best;
}

}  // namespace texas
