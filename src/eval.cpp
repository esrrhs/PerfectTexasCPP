#include "texas/eval.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "texas/rollout.hpp"
#include "texas/strength.hpp"

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
    r.mbbPerGame = mean / double(kBigBlind) * 1000.0;  // ACPC 单位
    return r;
}

// 把团队的额外成员写进 rollout 配置。集成只作用于 ROLE_LATEST 座位，
// 也就是"被测 / 学习"的这一边。
void applyTeam(RolloutConfig& cfg, const EvalTeam& team) {
    for (int s = 0; s < 2; ++s) {
        const int n = std::min(team.extraCount[s], kMaxEnsemble);
        cfg.ensembleExtraCount[s] = n;
        for (int e = 0; e < n; ++e) cfg.ensembleExtra[s][e] = team.extra[s][e];
    }
    cfg.policyTemp = team.policyTemp;
    cfg.policyGreedy = team.policyGreedy;
    cfg.amountMid = team.amountMid;
    cfg.preflopRaiseMult = team.preflopRaiseMult;
    cfg.postflopBetMult = team.postflopBetMult;
    cfg.rangeEqSamples = team.rangeEqSamples;
    cfg.rangeEqIters = team.rangeEqIters;
    cfg.rangeSearchSamples = team.rangeSearchSamples;
    cfg.rangeSearchUniform = team.rangeSearchUniform;
    cfg.robustSearch = team.robustSearch;
    cfg.robustMean = team.robustMean;
    cfg.rangeSearchMinPot = team.rangeSearchMinPot;
    cfg.rangeSearchDefenseOnly = team.rangeSearchDefenseOnly;
    cfg.riverCfrIters = team.riverCfrIters;
    cfg.riverCfrMinPot = team.riverCfrMinPot;
    cfg.riverCfrMaxRaises = team.riverCfrMaxRaises;
    cfg.riverCfrAllin = team.riverCfrAllin;
    cfg.rangerAir = team.rangerAir;
    if (team.oppCfrIters > 0) {
        cfg.oppCfrProb = 1.0f;  // 评估路径：整局都用同一个对手，不混联赛
        cfg.oppCfrIters = team.oppCfrIters;
        cfg.oppCfrMinPot = team.oppCfrMinPot;
    }
}

// 两遍对局：模型坐小盲一遍、坐大盲一遍。两遍用**同一批发牌**，第二遍
// 不换牌，而是把座位对调（模型在大盲拿同一副牌的大盲手牌）。这样每个
// 配对样本里，模型在一边拿强牌、在另一边拿弱牌，发牌运气在配对和里抵消；
// 剩下的差才是"策略 / 位置"的贡献。这是 duplicate match 的做法。
//
// 先按对求和，再除以 2 得到每手期望。SE 由配对样本数 games 决定。
EvalResult runBenchmark(const EvalTeam& team, int games, uint64_t seed, int equityIters,
                        int ruleSeat, int randomSeat, int proSeat, int searchSeat,
                        int policySearchSamples = 0, int solidSeat = -1) {
    std::vector<HUState> deals(static_cast<size_t>(games));
    Rng dealRng(seed);
    for (HUState& d : deals) d = dealNewHand(dealRng);

    std::vector<std::vector<int>> passScores(2, std::vector<int>(static_cast<size_t>(games), 0));
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
        cfg.forceProSeat = (proSeat >= 0) ? (pass == 0 ? 1 : 0) : -1;
        cfg.forceSearchSeat = (searchSeat >= 0) ? (pass == 0 ? 1 : 0) : -1;
        cfg.forceSolidSeat = (solidSeat >= 0) ? (pass == 0 ? 1 : 0) : -1;
        cfg.handScoresByGame = &passScores[static_cast<size_t>(pass)];
        cfg.fixedDeals = &deals;
        applyTeam(cfg, team);
        if (policySearchSamples > 0 || team.rangeSearchSamples > 0) {
            cfg.policySearchSamples = policySearchSamples;
            cfg.policySearchActors[0] = team.actor[0];
            cfg.policySearchActors[1] = team.actor[1];
        }

        ModelSet models;
        models.actor[0] = *team.actor[0];
        models.actor[1] = *team.actor[1];
        std::array<std::vector<Transition>, 2> tr;
        RolloutStats stats;
        collectRollout(models, nullptr, cfg, tr, stats);
    }

    std::vector<int> pairs(static_cast<size_t>(games), 0);
    for (int i = 0; i < games; ++i) {
        pairs[static_cast<size_t>(i)] = passScores[0][static_cast<size_t>(i)] +
                                        passScores[1][static_cast<size_t>(i)];
    }
    EvalResult r = summarize(pairs);
    // pairs 是"两手牌的和"；换成每手口径，SE 保持按配对样本数计算。
    r.chipsPerHand /= 2.0;
    r.stderr /= 2.0;
    r.bbPer100 /= 2.0;
    r.mbbPerGame /= 2.0;
    r.hands = games * 2;
    return r;
}

}  // namespace

EvalResult runMatch(const EvalTeam& a, const nn::Actor b[2], int games, uint64_t seed,
                    int equityIters, int policySearchSamples = 0, float bTemp = 1.0f,
                    bool bGreedy = false, MatchLog* log = nullptr,
                    float bPreflopRaiseMult = 1.0f, float bPostflopBetMult = 1.0f) {
    std::vector<HUState> deals(static_cast<size_t>(games));
    Rng dealRng(seed);
    for (HUState& d : deals) d = dealNewHand(dealRng);

    std::vector<std::vector<int>> passScores(2, std::vector<int>(static_cast<size_t>(games), 0));
    std::vector<std::vector<int>> passPots(2, std::vector<int>(static_cast<size_t>(games), 0));
    std::vector<std::vector<int>> passSeats(2, std::vector<int>(static_cast<size_t>(games), 0));
    for (int pass = 0; pass < 2; ++pass) {
        RolloutConfig cfg;
        cfg.games = games;
        cfg.seed = seed + static_cast<uint64_t>(pass) * 7919u;
        cfg.historicalProb = 0.0f;
        cfg.ruleProb = 0.0f;
        cfg.randomProb = 0.0f;
        cfg.oracleReward = false;
        cfg.equityIters = equityIters;
        cfg.handPotsByGame = &passPots[static_cast<size_t>(pass)];
        cfg.handSeatsByGame = &passSeats[static_cast<size_t>(pass)];
        // 偶数遍：a 坐小盲，b 的 BB 网络坐大盲；奇数遍对调。
        cfg.forceActor = &b[pass == 0 ? 1 : 0];
        cfg.forceActorSeat = (pass == 0) ? 1 : 0;
        cfg.watchedSeat = (pass == 0) ? 0 : 1;
        cfg.handScoresByGame = &passScores[static_cast<size_t>(pass)];
        cfg.fixedDeals = &deals;
        applyTeam(cfg, a);
        cfg.forceActorTemp = bTemp;
        cfg.forceActorGreedy = bGreedy;
        cfg.forceActorPreflopRaiseMult = bPreflopRaiseMult;
        cfg.forceActorPostflopBetMult = bPostflopBetMult;
        if (policySearchSamples > 0 || a.rangeSearchSamples > 0) {
            cfg.policySearchSamples = policySearchSamples;
            cfg.policySearchActors[0] = a.actor[0];
            cfg.policySearchActors[1] = a.actor[1];
        }

        ModelSet models;
        models.actor[0] = *a.actor[0];
        models.actor[1] = *a.actor[1];
        std::array<std::vector<Transition>, 2> tr;
        RolloutStats stats;
        collectRollout(models, nullptr, cfg, tr, stats);
    }

    if (log != nullptr) {
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i < games; ++i) {
                log->pots.push_back(passPots[static_cast<size_t>(pass)][static_cast<size_t>(i)]);
                log->seats.push_back(passSeats[static_cast<size_t>(pass)][static_cast<size_t>(i)]);
                log->nets.push_back(passScores[static_cast<size_t>(pass)][static_cast<size_t>(i)]);
            }
        }
    }
    std::vector<int> pairs(static_cast<size_t>(games), 0);
    for (int i = 0; i < games; ++i) {
        pairs[static_cast<size_t>(i)] = passScores[0][static_cast<size_t>(i)] +
                                        passScores[1][static_cast<size_t>(i)];
    }
    EvalResult r = summarize(pairs);
    r.chipsPerHand /= 2.0;
    r.stderr /= 2.0;
    r.bbPer100 /= 2.0;
    r.mbbPerGame /= 2.0;
    r.hands = games * 2;
    // 分座位：pass 0 是被测方坐小盲，pass 1 是坐大盲。
    const EvalResult rsb = summarize(passScores[0]);
    const EvalResult rbb = summarize(passScores[1]);
    r.sbChipsPerHand = rsb.chipsPerHand;
    r.sbStderr = rsb.stderr;
    r.bbChipsPerHand = rbb.chipsPerHand;
    r.bbStderr = rbb.stderr;
    return r;
}

EvalResult evalMatch(const nn::Actor a[2], const nn::Actor b[2], int games, uint64_t seed,
                     int equityIters) {
    return runMatch(EvalTeam::of(a), b, games, seed, equityIters);
}

EvalResult evalTeamMatch(const EvalTeam& a, const nn::Actor b[2], int games, uint64_t seed,
                         int equityIters, float bTemp, bool bGreedy, MatchLog* log,
                         float bPreflopRaiseMult, float bPostflopBetMult) {
    return runMatch(a, b, games, seed, equityIters, 0, bTemp, bGreedy, log,
                    bPreflopRaiseMult, bPostflopBetMult);
}

void ShowdownStats::print(const char* tag, int writtenHands) const {
    const char* nm[9] = {"高牌", "一对", "两对", "三条", "顺子", "同花", "葫芦", "四条", "同花顺"};
    auto mean = [](double s, double sq, int n, double* se) {
        if (n <= 0) {
            if (se) *se = 0.0;
            return 0.0;
        }
        const double m = s / double(n);
        const double var = std::max(sq / double(n) - m * m, 0.0);
        if (se) *se = std::sqrt(var / double(n));
        return m;
    };
    double seSd = 0.0, seNsd = 0.0;
    const double mSd = mean(sdSum, sdSumSq, showdowns, &seSd);
    const double mNsd = mean(nsdSum, nsdSumSq, hands - showdowns, &seNsd);
    std::printf("摊牌诊断 %-8s %d 手（被测 %d）| 摊牌 %d（%.1f%%）胜 %d 平 %d = %.3f\n", tag,
                hands, writtenHands, showdowns, 100.0 * double(showdowns) / std::max(hands, 1),
                wins, ties, showdowns > 0 ? double(wins) / double(showdowns) : 0.0);
    std::printf("  摊牌 %+0.1f ± %.1f 筹码/手（贡献 %+0.1f）| 非摊牌 %+0.1f ± %.1f（贡献 %+0.1f）\n",
                mSd, seSd, hands > 0 ? mSd * double(showdowns) / double(hands) : 0.0, mNsd, seNsd,
                hands > 0 ? mNsd * double(hands - showdowns) / double(hands) : 0.0);
    std::string lhs, rhs;
    for (int i = 0; i < 9; ++i) {
        if (myCat[i] == 0 && oppCat[i] == 0) continue;
        if (!lhs.empty()) {
            lhs += " ";
            rhs += " ";
        }
        lhs += std::string(nm[i]) + " " +
               std::to_string(showdowns > 0 ? myCat[i] * 100 / showdowns : 0) + "%";
        rhs += std::string(nm[i]) + " " +
               std::to_string(showdowns > 0 ? oppCat[i] * 100 / showdowns : 0) + "%";
    }
    std::printf("  牌型 我们: %s\n  牌型 对手: %s\n", lhs.c_str(), rhs.c_str());
    static const char* bk[3] = {"<1500", "1500-4000", ">=4000"};
    for (int i = 0; i < 3; ++i) {
        if (bucketSd[i] == 0) continue;
        double se = 0.0;
        const double m = mean(bucketSdSum[i], bucketSdSumSq[i], bucketSd[i], &se);
        std::printf("  摊牌 pot %-9s n=%4d  %+0.1f ± %.1f 筹码/手\n", bk[i], bucketSd[i], m, se);
    }
}

EvalResult evalShowdownVs(const EvalTeam& team, const std::string& opp, int games, uint64_t seed,
                          int equityIters, ShowdownStats& out) {
    int ruleSeat = -1, randomSeat = -1, proSeat = -1, searchSeat = -1, solidSeat = -1;
    int rangerSeat = -1;
    int cfrSeat = -1;
    if (opp == "rule") ruleSeat = 1;
    else if (opp == "random") randomSeat = 1;
    else if (opp == "pro") proSeat = 1;
    else if (opp == "solid") solidSeat = 1;
    else if (opp == "ranger") rangerSeat = 1;
    else if (opp == "cfr") cfrSeat = 1;
    else if (opp == "search") searchSeat = 1;
    else {
        std::printf("evalShowdownVs: 未知对手 %s\n", opp.c_str());
        return EvalResult{};
    }

    std::vector<HUState> deals(static_cast<size_t>(games));
    Rng dealRng(seed);
    for (HUState& d : deals) d = dealNewHand(dealRng);

    std::vector<std::vector<int>> passScores(2, std::vector<int>(static_cast<size_t>(games), 0));
    for (int pass = 0; pass < 2; ++pass) {
        const size_t n = static_cast<size_t>(games);
        std::vector<int> sdFlag(n, 0), myScore(n, -1), oppScore(n, -1), pots(n, 0);
        RolloutConfig cfg;
        cfg.games = games;
        cfg.seed = seed + static_cast<uint64_t>(pass) * 7919u;
        cfg.historicalProb = 0.0f;
        cfg.ruleProb = 0.0f;
        cfg.randomProb = 0.0f;
        cfg.oracleReward = false;
        cfg.equityIters = equityIters;
        const int oppSeat = (pass == 0) ? 1 : 0;
        cfg.forceRuleSeat = (ruleSeat >= 0) ? oppSeat : -1;
        cfg.forceRandomSeat = (randomSeat >= 0) ? oppSeat : -1;
        cfg.forceProSeat = (proSeat >= 0) ? oppSeat : -1;
        cfg.forceSearchSeat = (searchSeat >= 0) ? oppSeat : -1;
        cfg.forceSolidSeat = (solidSeat >= 0) ? oppSeat : -1;
        cfg.forceRangerSeat = (rangerSeat >= 0) ? oppSeat : -1;
        cfg.forceCfrSeat = (cfrSeat >= 0) ? oppSeat : -1;
        cfg.watchedSeat = 1 - oppSeat;
        cfg.handScoresByGame = &passScores[static_cast<size_t>(pass)];
        cfg.handPotsByGame = &pots;
        cfg.handShowdownByGame = &sdFlag;
        cfg.handScoreByGame = &myScore;
        cfg.handOppScoreByGame = &oppScore;
        cfg.fixedDeals = &deals;
        applyTeam(cfg, team);

        ModelSet models;
        models.actor[0] = *team.actor[0];
        models.actor[1] = *team.actor[1];
        std::array<std::vector<Transition>, 2> tr;
        RolloutStats stats;
        collectRollout(models, nullptr, cfg, tr, stats);

        for (size_t i = 0; i < n; ++i) {
            const int net = passScores[static_cast<size_t>(pass)][i];
            ++out.hands;
            if (!sdFlag[i]) {
                out.nsdSum += double(net);
                out.nsdSumSq += double(net) * double(net);
                continue;
            }
            ++out.showdowns;
            out.sdSum += double(net);
            out.sdSumSq += double(net) * double(net);
            if (myScore[i] > oppScore[i]) ++out.wins;
            else if (myScore[i] == oppScore[i]) ++out.ties;
            const int mc = handCategoryOf(myScore[i]);
            const int oc = handCategoryOf(oppScore[i]);
            if (mc >= 0 && mc < 9) ++out.myCat[mc];
            if (oc >= 0 && oc < 9) ++out.oppCat[oc];
            const int pot = pots[i];
            const int b = (pot < 1500) ? 0 : (pot < 4000 ? 1 : 2);
            ++out.bucketSd[b];
            out.bucketSdSum[b] += double(net);
            out.bucketSdSumSq[b] += double(net) * double(net);
        }
    }

    std::vector<int> pairs(static_cast<size_t>(games), 0);
    for (int i = 0; i < games; ++i) {
        pairs[static_cast<size_t>(i)] = passScores[0][static_cast<size_t>(i)] +
                                        passScores[1][static_cast<size_t>(i)];
    }
    EvalResult r = summarize(pairs);
    r.chipsPerHand /= 2.0;
    r.stderr /= 2.0;
    r.bbPer100 /= 2.0;
    r.mbbPerGame /= 2.0;
    r.hands = games * 2;
    return r;
}

EvalResult evalVsRule(const nn::Actor actors[2], int games, uint64_t seed, int equityIters) {
    return runBenchmark(EvalTeam::of(actors), games, seed, equityIters, /*ruleSeat=*/1,
                        /*randomSeat=*/-1, /*proSeat=*/-1, /*searchSeat=*/-1);
}

EvalResult evalVsRandom(const nn::Actor actors[2], int games, uint64_t seed, int equityIters) {
    return runBenchmark(EvalTeam::of(actors), games, seed, equityIters, /*ruleSeat=*/-1,
                        /*randomSeat=*/1, /*proSeat=*/-1, /*searchSeat=*/-1);
}

EvalResult evalVsPro(const nn::Actor actors[2], int games, uint64_t seed, int equityIters) {
    return runBenchmark(EvalTeam::of(actors), games, seed, equityIters, /*ruleSeat=*/-1,
                        /*randomSeat=*/-1, /*proSeat=*/1, /*searchSeat=*/-1);
}

EvalResult evalVsSolid(const nn::Actor actors[2], int games, uint64_t seed, int equityIters) {
    return runBenchmark(EvalTeam::of(actors), games, seed, equityIters, /*ruleSeat=*/-1,
                        /*randomSeat=*/-1, /*proSeat=*/-1, /*searchSeat=*/-1,
                        /*policySearchSamples=*/0, /*solidSeat=*/1);
}

EvalResult evalVsSearch(const nn::Actor actors[2], int games, uint64_t seed, int equityIters) {
    return runBenchmark(EvalTeam::of(actors), games, seed, equityIters, /*ruleSeat=*/-1,
                        /*randomSeat=*/-1, /*proSeat=*/-1, /*searchSeat=*/1);
}

EvalResult evalTeamVsRule(const EvalTeam& team, int games, uint64_t seed, int equityIters) {
    return runBenchmark(team, games, seed, equityIters, /*ruleSeat=*/1, /*randomSeat=*/-1,
                        /*proSeat=*/-1, /*searchSeat=*/-1);
}

EvalResult evalTeamVsRandom(const EvalTeam& team, int games, uint64_t seed, int equityIters) {
    return runBenchmark(team, games, seed, equityIters, /*ruleSeat=*/-1, /*randomSeat=*/1,
                        /*proSeat=*/-1, /*searchSeat=*/-1);
}

EvalResult evalTeamVsPro(const EvalTeam& team, int games, uint64_t seed, int equityIters) {
    return runBenchmark(team, games, seed, equityIters, /*ruleSeat=*/-1, /*randomSeat=*/-1,
                        /*proSeat=*/1, /*searchSeat=*/-1);
}

EvalResult evalTeamVsSolid(const EvalTeam& team, int games, uint64_t seed, int equityIters) {
    return runBenchmark(team, games, seed, equityIters, /*ruleSeat=*/-1, /*randomSeat=*/-1,
                        /*proSeat=*/-1, /*searchSeat=*/-1, /*policySearchSamples=*/0,
                        /*solidSeat=*/1);
}

EvalResult evalTeamVsSearch(const EvalTeam& team, int games, uint64_t seed, int equityIters) {
    return runBenchmark(team, games, seed, equityIters, /*ruleSeat=*/-1, /*randomSeat=*/-1,
                        /*proSeat=*/-1, /*searchSeat=*/1);
}

EvalResult runMix(const EvalTeam& team, int games, uint64_t seed, int equityIters,
                  int policySearchSamples = 0) {
    std::vector<int> scores(static_cast<size_t>(games), 0);

    RolloutConfig cfg;
    cfg.games = games;
    cfg.seed = seed;
    cfg.historicalProb = 0.0f;
    cfg.ruleProb = 0.25f;
    cfg.proProb = 0.35f;
    cfg.searchProb = 0.30f;
    cfg.randomProb = 0.10f;
    cfg.oracleReward = false;
    cfg.equityIters = equityIters;
    cfg.handScoresByGame = &scores;
    cfg.watchLearningSeat = true;
    applyTeam(cfg, team);
    if (policySearchSamples > 0 || team.rangeSearchSamples > 0) {
        cfg.policySearchSamples = policySearchSamples;
        cfg.policySearchActors[0] = team.actor[0];
        cfg.policySearchActors[1] = team.actor[1];
    }

    ModelSet models;
    models.actor[0] = *team.actor[0];
    models.actor[1] = *team.actor[1];
    std::array<std::vector<Transition>, 2> tr;
    RolloutStats stats;
    collectRollout(models, nullptr, cfg, tr, stats);

    EvalResult r = summarize(scores);
    r.hands = games;
    return r;
}

EvalResult evalVsMix(const nn::Actor actors[2], int games, uint64_t seed, int equityIters) {
    return runMix(EvalTeam::of(actors), games, seed, equityIters);
}

EvalResult evalTeamVsMix(const EvalTeam& team, int games, uint64_t seed, int equityIters) {
    return runMix(team, games, seed, equityIters);
}

// 策略内搜索的评估入口：被测座位不直接采样，而是用 policySearchMove
// 做"用自己策略续演"的一步前瞻。samples = 每次决策抽的隐藏牌样本数。
EvalResult evalSearchTeamVsRule(const EvalTeam& team, int games, uint64_t seed, int samples,
                                int equityIters) {
    return runBenchmark(team, games, seed, equityIters, /*ruleSeat=*/1, /*randomSeat=*/-1,
                        /*proSeat=*/-1, /*searchSeat=*/-1, samples);
}

EvalResult evalSearchTeamVsMix(const EvalTeam& team, int games, uint64_t seed, int samples,
                               int equityIters) {
    return runMix(team, games, seed, equityIters, samples);
}

EvalResult evalSearchTeamMatch(const EvalTeam& a, const nn::Actor b[2], int games, uint64_t seed,
                               int samples, int equityIters) {
    return runMatch(a, b, games, seed, equityIters, samples);
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

// 团队版自对局：和上面的口径相同，但把 EvalTeam 的开关（温度、集成、
// rangeEq 等）也带进来——否则 --range-eq 在 --vs self 下会被静默忽略，
// 诊断时看到的就是"关了特征"的数，容易误判。
EvalResult evalTeamSelfPlay(const EvalTeam& team, int games, uint64_t seed, int equityIters) {
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
    applyTeam(cfg, team);
    cfg.handScores = &scores;

    ModelSet models;
    models.actor[0] = *team.actor[0];
    models.actor[1] = *team.actor[1];
    std::array<std::vector<Transition>, 2> tr;
    RolloutStats stats;
    collectRollout(models, nullptr, cfg, tr, stats);
    return summarize(scores);
}

}  // namespace texas
