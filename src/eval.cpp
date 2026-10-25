#include "texas/eval.hpp"

#include <algorithm>
#include <cmath>

#include "texas/rollout.hpp"

namespace texas {

namespace {

EvalResult summarize(const std::vector<int>& scores) {
    EvalResult r;
    r.hands = static_cast<int>(scores.size());
    if (scores.empty()) return r;
    double sum = 0;
    double sq = 0;
    for (int v : scores) {
        sum += v;
        sq += double(v) * double(v);
    }
    const double n = double(scores.size());
    const double mean = sum / n;
    const double var = std::max(sq / n - mean * mean, 0.0);
    r.chipsPerHand = mean;
    r.stderr = std::sqrt(var / n);
    r.bbPer100 = mean / double(kBigBlind) * 100.0;
    return r;
}

// 两遍对局：模型坐小盲一遍、坐大盲一遍。同一批下标各自独立发牌，
// 方差靠样本量压，不靠配对（配对同牌要预发公共牌，这里先不做）。
EvalResult runBenchmark(const nn::Actor actors[2], int games, uint64_t seed, int equityIters,
                        int ruleSeat, int randomSeat) {
    std::vector<int> scores;
    scores.reserve(static_cast<size_t>(games) * 2);
    for (int pass = 0; pass < 2; ++pass) {
        RolloutConfig cfg;
        cfg.games = games;
        cfg.seed = seed + static_cast<uint64_t>(pass) * 7919u;
        cfg.historicalProb = 0.0f;
        cfg.ruleProb = 0.0f;
        cfg.randomProb = 0.0f;
        cfg.oracleReward = false;
        cfg.equityIters = equityIters;
        cfg.forceRuleSeat = (ruleSeat >= 0) ? (pass == 0 ? 1 : 0) : -1;
        cfg.forceRandomSeat = (randomSeat >= 0) ? (pass == 0 ? 1 : 0) : -1;
        cfg.handScores = &scores;

        ModelSet models;
        models.actor[0] = actors[0];
        models.actor[1] = actors[1];
        std::array<std::vector<Transition>, 2> tr;
        RolloutStats stats;
        collectRollout(models, nullptr, cfg, tr, stats);
    }
    return summarize(scores);
}

}  // namespace

EvalResult evalVsRule(const nn::Actor actors[2], int games, uint64_t seed, int equityIters) {
    return runBenchmark(actors, games, seed, equityIters, /*ruleSeat=*/1, /*randomSeat=*/-1);
}

EvalResult evalVsRandom(const nn::Actor actors[2], int games, uint64_t seed, int equityIters) {
    return runBenchmark(actors, games, seed, equityIters, /*ruleSeat=*/-1, /*randomSeat=*/1);
}

EvalResult evalSelfPlay(const nn::Actor actors[2], int games, uint64_t seed, int equityIters) {
    std::vector<int> scores;
    scores.reserve(static_cast<size_t>(games));
    RolloutConfig cfg;
    cfg.games = games;
    cfg.seed = seed;
    cfg.historicalProb = 0.0f;
    cfg.ruleProb = 0.0f;
    cfg.randomProb = 0.0f;
    cfg.oracleReward = false;
    cfg.equityIters = equityIters;
    cfg.handScores = &scores;

    ModelSet models;
    models.actor[0] = actors[0];
    models.actor[1] = actors[1];
    std::array<std::vector<Transition>, 2> tr;
    RolloutStats stats;
    collectRollout(models, nullptr, cfg, tr, stats);
    return summarize(scores);
}

}  // namespace texas
