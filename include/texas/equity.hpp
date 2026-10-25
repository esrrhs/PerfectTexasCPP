#pragma once

// 蒙特卡洛胜率（equity）估算：给定自己的手牌和已知公共牌，
// 对随机对手手牌做采样，返回胜/平/负频率。

#include <cstdint>
#include <random>
#include <vector>

#include "card.hpp"

namespace texas {

struct EquityResult {
    double win = 0.0;
    double tie = 0.0;
    double lose = 0.0;

    // 习惯用法：平局按半个胜场计
    double equity() const { return win + 0.5 * tie; }
};

inline EquityResult equityMonteCarlo(const std::vector<Card>& hero,
                                     const std::vector<Card>& board,
                                     int opponents,
                                     int iterations,
                                     std::uint64_t seed = 42) {
    EquityResult res;
    if (hero.size() != 2 || board.size() > 5 || opponents < 1 || iterations <= 0) {
        return res;
    }

    bool used[kNumCards] = {};
    for (Card c : hero) {
        if (c >= 0 && c < kNumCards) used[c] = true;
    }
    for (Card c : board) {
        if (c >= 0 && c < kNumCards) used[c] = true;
    }

    std::vector<Card> deck;
    deck.reserve(kNumCards);
    for (int c = 0; c < kNumCards; ++c) {
        if (!used[c]) deck.push_back(c);
    }

    const int missingBoard = 5 - static_cast<int>(board.size());
    const int need = opponents * 2 + missingBoard;
    if (static_cast<int>(deck.size()) < need) return res;

    std::mt19937_64 rng(seed);
    std::vector<Card> fullBoard(5, 0);
    for (int i = 0; i < static_cast<int>(board.size()); ++i) fullBoard[i] = board[i];
    std::vector<Card> oppHole(static_cast<std::size_t>(opponents) * 2, 0);

    for (int it = 0; it < iterations; ++it) {
        // 部分洗牌：只需要前 need 张
        for (int i = 0; i < need; ++i) {
            std::uniform_int_distribution<int> dist(i, static_cast<int>(deck.size()) - 1);
            std::swap(deck[i], deck[dist(rng)]);
        }

        int idx = 0;
        for (int i = 0; i < opponents * 2; ++i) oppHole[i] = deck[idx++];
        for (int i = static_cast<int>(board.size()); i < 5; ++i) fullBoard[i] = deck[idx++];

        Card h7[7] = {hero[0], hero[1], fullBoard[0], fullBoard[1],
                      fullBoard[2], fullBoard[3], fullBoard[4]};
        const int heroScore = evaluate7(h7);

        bool lost = false;
        bool tied = false;
        for (int o = 0; o < opponents; ++o) {
            Card o7[7] = {oppHole[o * 2], oppHole[o * 2 + 1], fullBoard[0], fullBoard[1],
                          fullBoard[2], fullBoard[3], fullBoard[4]};
            const int s = evaluate7(o7);
            if (s > heroScore) {
                lost = true;
                break;
            }
            if (s == heroScore) tied = true;
        }

        if (lost) {
            res.lose += 1.0;
        } else if (tied) {
            res.tie += 1.0;
        } else {
            res.win += 1.0;
        }
    }

    const double inv = 1.0 / static_cast<double>(iterations);
    res.win *= inv;
    res.tie *= inv;
    res.lose *= inv;
    return res;
}

}  // namespace texas
