// 自测：牌力评估、规则引擎、特征编码、PPO 数学、网络存盘往返。
//
// 硬断言优先用"不依赖训练"的性质：
//   - 任意随机对局必须零和、筹码守恒、必然终局
//   - 合法动作里必须恰好有一个跟注、加注额严格递增、最小加注不被违反
//   - 河牌的完美信息胜率必须与直接比牌一致
//   - 掩码 softmax 的概率和为 1，非法动作概率为 0

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "texas/card.hpp"
#include "texas/eval.hpp"
#include "texas/features.hpp"
#include "texas/hu.hpp"
#include "texas/nn.hpp"
#include "texas/ppo.hpp"
#include "texas/rollout.hpp"
#include "texas/slumbot.hpp"
#include "texas/strength.hpp"

namespace {

int g_failed = 0;
int g_passed = 0;

void section(const char* name) { std::printf("\n[%s]\n", name); }

// torch 的头文件里也有一个 CHECK 宏，先撤掉再定义自己的
#undef CHECK

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failed;                                                   \
        } else {                                                          \
            ++g_passed;                                                   \
        }                                                                 \
    } while (0)

#define CHECK_NEAR(actual, expect, eps)                                     \
    do {                                                                    \
        const double a_ = (actual);                                         \
        const double e_ = (expect);                                         \
        if (!(std::fabs(a_ - e_) <= (eps))) {                               \
            std::printf("  FAIL %s:%d  %s = %.6f, 期望 %.6f (容差 %.4f)\n",  \
                        __FILE__, __LINE__, #actual, a_, e_, (double)(eps));  \
            ++g_failed;                                                     \
        } else {                                                            \
            ++g_passed;                                                     \
        }                                                                   \
    } while (0)

texas::Card C(const char* s) { return texas::parseCard(s); }

// ---------------------------------------------------------------- 牌

void testCards() {
    section("牌与牌力评估");
    CHECK(C("2c") == 0);
    CHECK(C("Ah") == 12 * 4 + 2);
    CHECK(C("As") == 51);
    CHECK(C("Xx") < 0);

    // 快路径 vs 暴力枚举：n 张牌里取 C(n,5) 组合的最大值
    auto brute = [](const texas::Card* c, int n) {
        int best = -1;
        for (int a = 0; a < n; ++a)
            for (int b = a + 1; b < n; ++b)
                for (int c2 = b + 1; c2 < n; ++c2)
                    for (int d = c2 + 1; d < n; ++d)
                        for (int e = d + 1; e < n; ++e) {
                            const texas::Card h[5] = {c[a], c[b], c[c2], c[d], c[e]};
                            best = std::max(best, texas::evaluate5(h));
                        }
        return best;
    };

    std::mt19937_64 rng(12345);
    int mismatch = 0;
    for (int t = 0; t < 50000; ++t) {
        texas::Card deck[52];
        for (int i = 0; i < 52; ++i) deck[i] = i;
        for (int i = 51; i > 0; --i) std::swap(deck[i], deck[rng() % (i + 1)]);
        for (int n = 5; n <= 7; ++n) {
            if (texas::evaluateBest(deck, n) != brute(deck, n)) ++mismatch;
        }
        if (texas::evaluate7(deck) != texas::evaluate7Brute(deck)) ++mismatch;
    }
    CHECK(mismatch == 0);

    const auto& sm = texas::StrengthModel::instance();
    const double aa = sm.preflopEquity(C("Ah"), C("Ad"));
    const double kk = sm.preflopEquity(C("Kh"), C("Kd"));
    const double s72 = sm.preflopEquity(C("7h"), C("2d"));
    CHECK_NEAR(aa, 0.852, 0.015);
    CHECK_NEAR(kk, 0.824, 0.015);
    CHECK_NEAR(s72, 0.346, 0.020);
    CHECK(aa > kk && kk > s72);
}

// ---------------------------------------------------------------- 规则

texas::HUState rigged(const char* h0a, const char* h0b, const char* h1a, const char* h1b,
                      const char* b0 = nullptr, const char* b1 = nullptr,
                      const char* b2 = nullptr, const char* b3 = nullptr,
                      const char* b4 = nullptr, int boardN = 0) {
    texas::Card h0[2] = {C(h0a), C(h0b)};
    texas::Card h1[2] = {C(h1a), C(h1b)};
    texas::Card board[5] = {0, 0, 0, 0, 0};
    const char* bs[5] = {b0, b1, b2, b3, b4};
    for (int i = 0; i < 5; ++i) {
        if (bs[i]) board[i] = C(bs[i]);
    }
    return texas::makeState(h0, h1, board, boardN);
}

void testRules() {
    section("规则引擎");

    texas::Rng rng(7);
    texas::HUState s = rigged("Ah", "Ad", "Kh", "Kd");
    CHECK(s.pot() == texas::kSmallBlind + texas::kBigBlind);
    CHECK(s.toAct == 0);
    CHECK(s.contrib[0] == texas::kSmallBlind && s.contrib[1] == texas::kBigBlind);

    texas::LegalMove lm[texas::kNumActions];
    int n = texas::legalMoves(s, lm);
    CHECK(n >= 4);
    CHECK(lm[0].slot == texas::kSlotFold);
    int callCount = 0;
    int lastTo = -1;
    bool increasing = true;
    for (int i = 0; i < n; ++i) {
        if (lm[i].slot == texas::kSlotCheckCall) ++callCount;
        if (lm[i].mv.kind == texas::MK_RAISE) {
            if (lm[i].mv.to <= lastTo) increasing = false;
            lastTo = lm[i].mv.to;
            CHECK(lm[i].mv.to >= 2 * texas::kBigBlind);  // 最小加注 = 大盲 + 大盲
            CHECK(lm[i].mv.to > s.contrib[0]);
        }
    }
    CHECK(callCount == 1);
    CHECK(increasing);
    CHECK(lm[n - 1].slot == texas::kSlotAllIn);

    // 跟注 → 大盲获得行动权
    texas::Move call{};
    for (int i = 0; i < n; ++i) {
        if (lm[i].slot == texas::kSlotCheckCall) call = lm[i].mv;
    }
    texas::HUState s2 = s;
    texas::applyMove(s2, call, rng);
    CHECK(!s2.terminal());
    CHECK(s2.toAct == 1);
    CHECK(s2.pot() == 2 * texas::kBigBlind);

    // 大盲过牌 → 翻牌，公共牌 3 张，大盲先行动
    texas::legalMoves(s2, lm);
    texas::applyMove(s2, lm[0].mv, rng);
    CHECK(!s2.terminal());
    CHECK(s2.street == texas::ST_FLOP);
    CHECK(s2.boardN == 3);
    CHECK(s2.toAct == 1);
    CHECK(s2.contrib[0] == 0 && s2.contrib[1] == 0);

    // 小盲弃牌：大盲净赚 1
    {
        texas::HUState f = rigged("Ah", "Ad", "Kh", "Kd");
        texas::legalMoves(f, lm);
        texas::applyMove(f, lm[0].mv, rng);
        CHECK(f.terminal());
        CHECK(texas::payoff(f, 0) == -texas::kSmallBlind);
        CHECK(texas::payoff(f, 1) == texas::kSmallBlind);
        CHECK(texas::payoff(f, 0) + texas::payoff(f, 1) == 0);
    }

    // 翻牌前全下 → 大盲跟注后一次发完 5 张
    {
        texas::HUState allin = rigged("Ah", "Ad", "Kh", "Kd");
        n = texas::legalMoves(allin, lm);
        for (int i = 0; i < n; ++i) {
            if (lm[i].slot == texas::kSlotAllIn) texas::applyMove(allin, lm[i].mv, rng);
        }
        CHECK(!allin.terminal());
        CHECK(allin.toAct == 1);
        n = texas::legalMoves(allin, lm);
        texas::Move callAllIn{};
        for (int i = 0; i < n; ++i) {
            if (lm[i].slot == texas::kSlotCheckCall) callAllIn = lm[i].mv;
        }
        texas::applyMove(allin, callAllIn, rng);
        CHECK(allin.terminal());
        CHECK(allin.boardN == 5);
        CHECK(texas::payoff(allin, 0) + texas::payoff(allin, 1) == 0);
    }

    // 摊牌：AA 胜 KK；公共牌成同花顺则平分
    {
        texas::HUState sd = rigged("Ah", "Ad", "Kh", "Kd", "2c", "7d", "9s", "Jh", "4c", 5);
        CHECK(sd.boardN == 5);
        CHECK(texas::showdownWinner(sd) == 0);
    }
    {
        texas::HUState sd = rigged("2h", "3h", "4d", "5c", "Ah", "Kh", "Qh", "Jh", "Th", 5);
        CHECK(texas::showdownWinner(sd) == 2);
    }

    // 河牌的开天眼胜率 = 直接比牌；转牌在 (0.5, 1)
    {
        texas::HUState sd = rigged("Ah", "Ad", "Kh", "Kd", "2c", "7d", "9s", "Jh", "4c", 5);
        CHECK_NEAR(texas::perfectEquity(sd, 0), 1.0, 1e-9);
        CHECK_NEAR(texas::perfectEquity(sd, 1), 0.0, 1e-9);
    }
    {
        texas::HUState sd = rigged("Ah", "Ad", "Kh", "Kd", "2c", "7d", "9s", "Jh", nullptr, 4);
        const double eq = texas::perfectEquity(sd, 0);
        CHECK(eq > 0.6 && eq < 1.0);
        CHECK_NEAR(eq + texas::perfectEquity(sd, 1), 1.0, 1e-12);
    }

    // ---- 对齐 ACPC 的三条不变式 ----
    // 1) 筹码深度必须是 200BB（ACPC：stack 20000 / big blind 100）
    section("对齐 ACPC");
    CHECK(texas::kStartingStack / texas::kBigBlind == 200);
    CHECK(texas::kBigBlind == 2 * texas::kSmallBlind);
    // 2) 默认无加注次数上限
    CHECK(texas::defaultMaxRaises() == 0);
    {
        texas::HUState r = rigged("Ah", "Ad", "Kh", "Kd");
        CHECK(r.maxRaisesPerStreet == 0);
        int raises = 0;
        texas::LegalMove rl[texas::kNumActions];
        for (int it = 0; it < 40 && !r.terminal(); ++it) {
            const int n2 = texas::legalMoves(r, rl);
            if (n2 <= 0) break;
            // 每次都取最小的一档加注，逼出最长的加注序列
            int pick = -1;
            for (int i = 0; i < n2; ++i) {
                if (rl[i].mv.kind == texas::MK_RAISE) {
                    pick = i;
                    break;
                }
            }
            if (pick < 0) break;
            texas::applyMove(r, rl[pick].mv, rng);
            ++raises;
        }
        CHECK(raises > 3);  // 旧版 kMaxRaises=3 时这里永远到不了 4
    }
    // 3) 单位换算：永远弃牌 = 750 mbb/g（ACPC 文献的标尺）
    {
        const double chips = 0.5 * double(texas::kSmallBlind + texas::kBigBlind);
        const double mbbPerGame = chips / double(texas::kBigBlind) * 1000.0;
        CHECK(std::fabs(mbbPerGame - 750.0) < 1e-9);
    }
    texas::setDefaultMaxRaises(0);

    // 随机对局：必然终局、零和、筹码守恒
    int zeroSumFail = 0;
    int notTerminal = 0;
    int c0 = 0;
    int c1 = 0;
    long long decisions = 0;
    for (int t = 0; t < 20000; ++t) {
        texas::HUState g = texas::dealNewHand(rng);
        int guard = 0;
        while (!g.terminal() && guard < 1000) {
            texas::LegalMove mv[texas::kNumActions];
            const int k = texas::legalMoves(g, mv);
            if (k <= 0) break;
            texas::applyMove(g, mv[rng() % static_cast<size_t>(k)].mv, rng);
            ++guard;
            ++decisions;
        }
        if (!g.terminal()) ++notTerminal;
        if (texas::payoff(g, 0) + texas::payoff(g, 1) != 0) ++zeroSumFail;
        c0 += texas::payoff(g, 0);
        c1 += texas::payoff(g, 1);
    }
    CHECK(notTerminal == 0);
    CHECK(zeroSumFail == 0);
    CHECK(c0 + c1 == 0);
    std::printf("  随机对局 2 万手：零和违例 %d，平均决策数 %.2f，小盲总收益 %d\n", zeroSumFail,
                double(decisions) / 20000.0, c0);
}

// ---------------------------------------------------------------- 特征

void testFeatures() {
    section("特征编码");

    texas::Rng rng(99);
    texas::HUState s = texas::dealNewHand(rng);
    texas::Obs obs;
    texas::encodeObservation(s, obs, 128);

    int holeBits = 0;
    for (int i = 0; i < texas::kHoleDim; ++i) holeBits += obs.node[static_cast<size_t>(i)] > 0.5f;
    CHECK(holeBits == 2);
    int boardBits = 0;
    for (int i = 0; i < texas::kBoardDim; ++i)
        boardBits += obs.node[static_cast<size_t>(texas::kHoleDim + i)] > 0.5f;
    CHECK(boardBits == 0);

    CHECK(obs.equityPerfect >= 0.0f && obs.equityPerfect <= 1.0f);
    int extraBits = 0;
    for (int i = 0; i < texas::kHoleDim; ++i) extraBits += obs.extra[static_cast<size_t>(i)] > 0.5f;
    CHECK(extraBits == 2);

    float seqSum = 0;
    for (float v : obs.seq) seqSum += v;
    CHECK_NEAR(seqSum, 0.0, 1e-6);

    texas::LegalMove lm[texas::kNumActions];
    const int n = texas::legalMoves(s, lm);
    CHECK(n > 0);
    std::array<float, texas::kActionInput> af;
    for (int i = 0; i < n; ++i) {
        texas::encodeActionFeature(s, lm[i], af);
        int ones = 0;
        for (int k = 0; k < 4; ++k) ones += af[static_cast<size_t>(k)] > 0.5f;
        CHECK(ones == 1);
    }

    // 走到翻牌：历史非空、公共牌 3 张
    texas::Move mv[2];
    texas::legalMoves(s, lm);
    for (int i = 0; i < n; ++i) {
        if (lm[i].slot == texas::kSlotCheckCall) mv[0] = lm[i].mv;
    }
    texas::applyMove(s, mv[0], rng);
    texas::legalMoves(s, lm);
    texas::applyMove(s, lm[0].mv, rng);
    texas::Obs obs2;
    texas::encodeObservation(s, obs2, 32);
    float seqSum2 = 0;
    for (float v : obs2.seq) seqSum2 += v;
    CHECK(seqSum2 > 0.0f);
    int boardBits2 = 0;
    for (int i = 0; i < texas::kBoardDim; ++i)
        boardBits2 += obs2.node[static_cast<size_t>(texas::kHoleDim + i)] > 0.5f;
    CHECK(boardBits2 == 3);

    // 牌力抽象要能看见听牌：花听的实际百分位应该高于"当前最佳五张"的排名。
    {
        const texas::Card hole[2] = {C("Ah"), C("Kh")};
        const texas::Card flop[3] = {C("Qh"), C("7h"), C("2c")};
        const double raw = texas::StrengthModel::instance().postflopPercentile(hole, flop, 3);
        const float eff = texas::handStrength(hole, flop, 3);
        CHECK(eff > raw);
        CHECK(eff > 0.5f);

        // 河牌没有听牌，有效牌力必须退回原始百分位。
        const texas::Card river[5] = {C("Qh"), C("7h"), C("2c"), C("3s"), C("4d")};
        const double rawRiver = texas::StrengthModel::instance().postflopPercentile(hole, river, 5);
        const float effRiver = texas::handStrength(hole, river, 5);
        CHECK_NEAR(effRiver, rawRiver, 1e-6);
    }
}

// ---------------------------------------------------------------- PPO

void testPpo() {
    section("PPO 数学");

    float logits[8] = {0.5f, -1e9f, 1.5f, -1e9f, -1e9f, -1e9f, -1e9f, -1e9f};
    float probs[8];
    texas::ppo::maskedSoftmax(logits, 8, probs);
    double sum = 0;
    for (float p : probs) sum += p;
    CHECK_NEAR(sum, 1.0, 1e-6);
    CHECK_NEAR(probs[1], 0.0, 1e-12);
    CHECK(probs[2] > probs[0]);

    float bad[8];
    for (int i = 0; i < 8; ++i) bad[i] = -1e9f;
    texas::ppo::maskedSoftmax(bad, 8, probs);
    sum = 0;
    for (float p : probs) sum += p;
    CHECK_NEAR(sum, 1.0, 1e-6);

    std::vector<texas::Transition> tr(3);
    tr[0].gameId = 0;
    tr[0].reward = 1.0f;
    tr[0].value = 0.5f;
    tr[1].gameId = 0;
    tr[1].reward = 2.0f;
    tr[1].value = 0.25f;
    tr[2].gameId = 0;
    tr[2].reward = -3.0f;
    tr[2].value = 0.0f;
    tr[2].terminal = true;
    texas::ppo::assignEpisodeReturns(tr, 1.0f, 1.0f);
    CHECK_NEAR(tr[0].ret, 0.0, 1e-5);
    CHECK_NEAR(tr[0].adv, -0.5, 1e-5);
    CHECK_NEAR(tr[1].ret, -1.0, 1e-5);
    CHECK_NEAR(tr[2].ret, -3.0, 1e-5);

    std::vector<texas::Transition> tr2(2);
    tr2[0].gameId = 1;
    tr2[0].reward = 5.0f;
    tr2[1].gameId = 2;
    tr2[1].reward = -5.0f;
    tr2[1].terminal = true;
    texas::ppo::assignEpisodeReturns(tr2, 1.0f, 0.95f);
    CHECK_NEAR(tr2[0].ret, 5.0, 1e-5);
    CHECK_NEAR(tr2[1].ret, -5.0, 1e-5);
}

// ---------------------------------------------------------------- 网络

void testNetwork() {
    section("网络前向与存盘");

    texas::nn::ModelConfig mc;
    mc.hidden = 64;
    mc.lstmHidden = 32;
    texas::nn::Actor actor(std::make_shared<texas::nn::ActorImpl>(mc));
    texas::nn::Critic critic(std::make_shared<texas::nn::CriticImpl>(mc));
    actor->eval();
    critic->eval();

    const int64_t B = 3;
    const int64_t L = 7;
    torch::Tensor node = torch::rand({B, texas::kNodeInput});
    torch::Tensor seq = torch::rand({B, texas::kHistSteps, texas::kHistDim});
    torch::Tensor extra = torch::rand({B, texas::kExtraInput});
    torch::Tensor actFeat = torch::rand({L, texas::kActionInput});
    torch::Tensor actSample = torch::tensor({0, 0, 0, 1, 1, 2, 2});
    torch::Tensor actId = torch::tensor({0, 1, 3, 1, 5, 2, 7});

    torch::NoGradGuard ng;
    torch::Tensor logits = actor->forward(node, seq, actFeat, actSample, actId);
    CHECK(logits.size(0) == B);
    CHECK(logits.size(1) == texas::kNumActions);
    torch::Tensor values = critic->forward(node, seq, extra);
    CHECK(values.size(0) == B);

    const float* lp = logits.data_ptr<float>();
    CHECK(lp[0 * texas::kNumActions + 2] < -1e8f);
    CHECK(lp[0 * texas::kNumActions + 0] > -1e8f);

    const std::string path = "/tmp/perfecttexas_test_actor.pt";
    actor->saveTo(path);
    texas::nn::Actor actor2(std::make_shared<texas::nn::ActorImpl>(mc));
    CHECK(actor2->loadFrom(path));
    torch::Tensor logits2 = actor2->forward(node, seq, actFeat, actSample, actId);
    const float* lp2 = logits2.data_ptr<float>();
    double maxDiff = 0;
    for (int64_t i = 0; i < B * texas::kNumActions; ++i) {
        if (lp[i] < -1e8f && lp2[i] < -1e8f) continue;
        maxDiff = std::max(maxDiff, std::fabs(double(lp[i] - lp2[i])));
    }
    CHECK(maxDiff < 1e-6);
}

// 脚本教师：分支续演的价值必须确定、有限、看牌。
void testTeacher() {
    section("脚本教师");

    const texas::Card opp[2] = {C("Kc"), C("Qc")};
    // 公共牌在开局一次发定；makeState 必须带上完整的五张，否则全下后的
    // 摊牌会拿到五张 0 号牌。
    const texas::Card board[5] = {C("3s"), C("4d"), C("8c"), C("9h"), C("Jh")};

    auto evalAt = [&](const char* a, const char* b) {
        const texas::Card hole[2] = {C(a), C(b)};
        const texas::HUState s = texas::makeState(hole, opp, board, 0);
        std::array<float, texas::kNumActions> q;
        texas::teacherActionValues(s, 0, q);
        return std::make_pair(s, q);
    };

    auto [sAA, qAA] = evalAt("Ah", "Ad");
    auto [s72, q72] = evalAt("7h", "2d");

    // 弃牌的价值是确定值：-已经投入的筹码 / 起始筹码。
    CHECK_NEAR(qAA[texas::kSlotFold], -double(texas::kSmallBlind) / double(texas::kStartingStack),
               1e-6);

    // 每个合法 slot 都有有限值；非法 slot 留 0。
    texas::LegalMove lm[texas::kNumActions];
    const int n = texas::legalMoves(sAA, lm);
    std::array<bool, texas::kNumActions> legal{};
    for (int i = 0; i < n; ++i) legal[lm[i].slot] = true;
    for (int slot = 0; slot < texas::kNumActions; ++slot) {
        if (!legal[slot]) {
            CHECK(qAA[slot] == 0.0f);
            continue;
        }
        CHECK(std::isfinite(qAA[slot]));
        CHECK(qAA[slot] >= -1.01f && qAA[slot] <= 1.01f);
    }

    // 同样的局面重复算，结果逐位一致（教师是确定性的，没有发牌/随机性方差）。
    std::array<float, texas::kNumActions> qAgain;
    texas::teacherActionValues(sAA, 0, qAgain);
    for (int slot = 0; slot < texas::kNumActions; ++slot) {
        CHECK(qAgain[slot] == qAA[slot]);
    }

    // 教师必须看牌：同样的对手底牌和公共牌，AA 的全下价值高于 72o。
    CHECK(qAA[texas::kSlotAllIn] > q72[texas::kSlotAllIn]);
    std::printf("  AA jam=%.4f fold=%.4f | 72o jam=%.4f fold=%.4f\n",
                qAA[texas::kSlotAllIn], qAA[texas::kSlotFold], q72[texas::kSlotAllIn],
                q72[texas::kSlotFold]);
}

void testLeague() {
    section("两级历史池");
    using texas::HistoricalPool;
    using texas::Snapshot;

    auto mk = [](int u) {
        auto s = std::make_shared<Snapshot>();
        s->update = u;
        return s;
    };

    // 1) recent 严格只留最后 recentCap 个，且保持时间顺序
    HistoricalPool pool;
    const int recentCap = 8;
    const int archiveCap = 16;
    for (int u = 1; u <= 40; ++u) pool.push(mk(u), recentCap, archiveCap);
    CHECK(int(pool.recent.size()) == recentCap);
    CHECK(int(pool.archive.size()) <= archiveCap);
    CHECK(!pool.archive.empty());
    CHECK(pool.size() == pool.recent.size() + pool.archive.size());
    bool recentOrdered = true;
    for (size_t i = 1; i < pool.recent.size(); ++i) {
        if (pool.recent[i]->update <= pool.recent[i - 1]->update) recentOrdered = false;
    }
    CHECK(recentOrdered);
    CHECK(pool.recent.back()->update == 40);
    CHECK(pool.recent.front()->update == 40 - recentCap + 1);

    // 2) archive 永不整体丢弃最老的快照，且时刻递增
    CHECK(pool.archive.front()->update == 1);
    CHECK(pool.archive.back()->update == 40);
    bool archiveOrdered = true;
    for (size_t i = 1; i < pool.archive.size(); ++i) {
        if (pool.archive[i]->update <= pool.archive[i - 1]->update) archiveOrdered = false;
    }
    CHECK(archiveOrdered);

    // 3) 隔一个丢一个 = 保留间隔翻倍：archive 从老到新是"越老越稀、越新越密"，
    //    所以相邻时间差序列应当非递增（8,8,4,4,2,2,1,1,…），而不是均匀间隔。
    bool gapsNonIncreasing = true;
    for (size_t i = 2; i < pool.archive.size(); ++i) {
        const int g1 = pool.archive[i - 1]->update - pool.archive[i - 2]->update;
        const int g2 = pool.archive[i]->update - pool.archive[i - 1]->update;
        if (g2 > g1) gapsNonIncreasing = false;
    }
    CHECK(gapsNonIncreasing);

    // 4) at() 的边界必须是 nullptr，不能越界读
    CHECK(pool.at(0, 0) == pool.recent[0].get());
    CHECK(pool.at(1, 0) == pool.archive[0].get());
    CHECK(pool.at(0, -1) == nullptr);
    CHECK(pool.at(0, recentCap) == nullptr);
    CHECK(pool.at(1, int(pool.archive.size())) == nullptr);
    CHECK(pool.at(7, 0) == nullptr);

    // 5) recentCap = 0 时只有 archive
    HistoricalPool onlyArchive;
    for (int u = 1; u <= 5; ++u) onlyArchive.push(mk(u), 0, archiveCap);
    CHECK(onlyArchive.recent.empty());
    CHECK(onlyArchive.archive.size() == 5);
    CHECK(!onlyArchive.empty());

    // 6) 空池
    HistoricalPool emptyPool;
    CHECK(emptyPool.empty());
    CHECK(emptyPool.size() == 0);
    CHECK(emptyPool.at(0, 0) == nullptr);
}

// ---------------------------------------------------------------- 动作级集成

// 集成把成员模型的 masked softmax 概率平均后再采样。把同一个模型（同权重
// 的克隆）当额外成员时，平均不改变分布，走集成路径的逐手结果必须和单模型
// **逐位相同**——这一条同时钉住"多跑几遍前向不会改变采样流"。
void testEnsemble() {
    section("动作级集成");

    texas::nn::ModelConfig mc;
    texas::nn::Actor actors[2];
    texas::nn::Actor dup[2];
    texas::nn::Critic critics[2];
    for (int s = 0; s < 2; ++s) {
        actors[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        critics[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
        dup[s] = texas::nn::cloneActor(actors[s]);
    }

    const texas::EvalResult single = texas::evalVsRule(actors, 60, 9911, 8);

    // 1) 空团队 = 原来的单模型路径
    const texas::EvalResult solo = texas::evalTeamVsRule(texas::EvalTeam::of(actors), 60, 9911, 8);
    CHECK_NEAR(solo.chipsPerHand, single.chipsPerHand, 1e-12);
    CHECK_NEAR(solo.stderr, single.stderr, 1e-12);

    // 2) 同权重克隆当额外成员：概率平均是恒等变换，结果必须逐位一致
    texas::EvalTeam team = texas::EvalTeam::of(actors);
    for (int s = 0; s < 2; ++s) {
        team.extra[s][0] = &dup[s];
        team.extraCount[s] = 1;
    }
    const texas::EvalResult ens = texas::evalTeamVsRule(team, 60, 9911, 8);
    CHECK(ens.hands == single.hands);
    CHECK_NEAR(ens.chipsPerHand, single.chipsPerHand, 1e-12);
    CHECK_NEAR(ens.stderr, single.stderr, 1e-12);
    CHECK_NEAR(ens.bbPer100, single.bbPer100, 1e-12);

    // 3) 团队成员的 critic 不参与：集成只影响采样，不影响记录口径
    CHECK(ens.hands == 60 * 2);
}

// ---------------------------------------------------------------- 尖自对局

// selfSharpProb=1 时每局都必须有一个座位按部署温度当对手，另一个座位照常
// 学习；两边的转移都要记下来（和快照对手同一口径），不能让牌局空转。
void testSharpSelfPlay() {
    section("尖自对局");

    texas::nn::ModelConfig mc;
    texas::ModelSet models;
    for (int s = 0; s < 2; ++s) {
        models.actor[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        models.critic[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
    }

    texas::RolloutConfig cfg;
    cfg.games = 12;
    cfg.seed = 777;
    cfg.historicalProb = 0.0f;
    cfg.ruleProb = 0.0f;
    cfg.randomProb = 0.0f;
    cfg.proProb = 0.0f;
    cfg.searchProb = 0.0f;
    cfg.selfSharpProb = 1.0f;
    cfg.selfSharpTemp = 0.1f;
    cfg.oracleReward = false;
    cfg.teacher = false;

    std::array<std::vector<texas::Transition>, 2> tr;
    texas::RolloutStats stats;
    texas::collectRollout(models, nullptr, cfg, tr, stats);

    CHECK(stats.games == 12);
    CHECK(stats.vsSelfSharp == 12);  // 每局都有一个尖对手座位
    CHECK(stats.vsLatest == 0);
    CHECK(!tr[0].empty());
    CHECK(!tr[1].empty());
    // 动作槽必须在合法范围内，logp 有限：尖对手的转移也走了正常提取路径。
    for (int s = 0; s < 2; ++s) {
        for (const texas::Transition& t : tr[static_cast<size_t>(s)]) {
            CHECK(t.action >= 0 && t.action < texas::kNumActions);
            CHECK(std::isfinite(t.logp));
        }
    }
}

// ---------------------------------------------------------------- 联赛里的冻结对手

// mixedFrozen + extraOpponentProb=1：每局都应该是"固定对手 vs 当前策略"，
// 但走联赛路径（不是纯 BR），统计里 vsExtra 必须等于局数。
void testMixedFrozen() {
    section("联赛冻结对手");

    texas::nn::ModelConfig mc;
    texas::ModelSet models;
    texas::nn::Actor extra[2];
    for (int s = 0; s < 2; ++s) {
        models.actor[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        models.critic[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
        extra[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
    }

    texas::RolloutConfig cfg;
    cfg.games = 10;
    cfg.seed = 4321;
    cfg.historicalProb = 0.0f;
    cfg.ruleProb = 0.0f;
    cfg.randomProb = 0.0f;
    cfg.proProb = 0.0f;
    cfg.searchProb = 0.0f;
    cfg.mixedFrozen = true;
    cfg.extraOpponentProb = 1.0f;
    cfg.fixedOpponent[0] = &extra[0];
    cfg.fixedOpponent[1] = &extra[1];
    cfg.fixedOpponentTemp = 0.1f;
    cfg.oracleReward = false;
    cfg.teacher = false;

    std::array<std::vector<texas::Transition>, 2> tr;
    texas::RolloutStats stats;
    texas::collectRollout(models, nullptr, cfg, tr, stats);

    CHECK(stats.games == 10);
    CHECK(stats.vsExtra == 10);
    CHECK(stats.vsLatest == 0);
    CHECK(!tr[0].empty());
    CHECK(!tr[1].empty());
}

// ---------------------------------------------------------------- 学习座位偏置

// learnerBBProb=1：联赛局（这里全是规则机器人）里学习座位永远坐 BB，
// 规则机器人的 SB 座位不记录样本，所以 out[0] 必须为空、out[1] 非空。
void testLearnerSeatBias() {
    section("学习座位偏置");

    texas::nn::ModelConfig mc;
    texas::ModelSet models;
    for (int s = 0; s < 2; ++s) {
        models.actor[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        models.critic[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
    }

    auto run = [&](float bbProb) {
        texas::RolloutConfig cfg;
        cfg.games = 8;
        cfg.seed = 2468;
        cfg.historicalProb = 0.0f;
        cfg.ruleProb = 1.0f;
        cfg.randomProb = 0.0f;
        cfg.proProb = 0.0f;
        cfg.searchProb = 0.0f;
        cfg.learnerBBProb = bbProb;
        cfg.oracleReward = false;
        cfg.teacher = false;
        std::array<std::vector<texas::Transition>, 2> tr;
        texas::RolloutStats stats;
        texas::collectRollout(models, nullptr, cfg, tr, stats);
        return std::make_pair(tr, stats);
    };

    const auto allBb = run(1.0f);
    CHECK(allBb.second.vsRule == 8);
    CHECK(allBb.first[0].empty());
    CHECK(!allBb.first[1].empty());

    const auto allSb = run(0.0f);
    CHECK(allSb.second.vsRule == 8);
    CHECK(!allSb.first[0].empty());
    CHECK(allSb.first[1].empty());
}

// ---------------------------------------------------------------- 同牌对调

// duplicateDeals：每两局成一对，底牌互换、其他（公共牌/筹码/行动权）相同。
void testDuplicateDeals() {
    section("同牌对调");

    texas::nn::ModelConfig mc;
    texas::ModelSet models;
    for (int s = 0; s < 2; ++s) {
        models.actor[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        models.critic[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
    }

    texas::RolloutConfig cfg;
    cfg.games = 8;
    cfg.seed = 12345;
    cfg.historicalProb = 0.0f;
    cfg.ruleProb = 0.0f;
    cfg.randomProb = 0.0f;
    cfg.proProb = 0.0f;
    cfg.searchProb = 0.0f;
    cfg.oracleReward = false;
    cfg.teacher = false;
    cfg.duplicateDeals = true;

    std::vector<texas::HUState> deals;
    cfg.initialDeals = &deals;
    std::array<std::vector<texas::Transition>, 2> tr;
    texas::RolloutStats stats;
    texas::collectRollout(models, nullptr, cfg, tr, stats);

    CHECK(deals.size() == 8);
    for (size_t i = 0; i + 1 < deals.size(); i += 2) {
        const texas::HUState& a = deals[i];
        const texas::HUState& b = deals[i + 1];
        CHECK(a.hole[0][0] == b.hole[1][0]);
        CHECK(a.hole[0][1] == b.hole[1][1]);
        CHECK(a.hole[1][0] == b.hole[0][0]);
        CHECK(a.hole[1][1] == b.hole[0][1]);
        for (int k = 0; k < 5; ++k) CHECK(a.board[k] == b.board[k]);
        for (int s = 0; s < 2; ++s) {
            CHECK(a.stack[s] == b.stack[s]);
            CHECK(a.total[s] == b.total[s]);
        }
        CHECK(a.toAct == b.toAct);
        CHECK(a.street == b.street);
    }
}

// ---------------------------------------------------------------- 策略内搜索

// 搜索返回的必须是合法动作：同类型槽位、金额落在该槽位的合法区间里。
// 这条不变式保证"部署时搜索"不会把非法动作打到真实牌局上。
void testPolicySearch() {
    section("策略内搜索");

    texas::nn::ModelConfig mc;
    texas::nn::Actor actors[2];
    for (int s = 0; s < 2; ++s) {
        actors[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
    }

    texas::Rng rng(4242);
    int illegal = 0;
    int checked = 0;
    for (int t = 0; t < 40; ++t) {
        texas::HUState g = texas::dealNewHand(rng);
        const int steps = int(rng() % 6);
        for (int i = 0; i < steps && !g.terminal(); ++i) {
            texas::LegalMove lm[texas::kNumActions];
            const int n = texas::legalMoves(g, lm);
            if (n <= 0) break;
            texas::applyMove(g, lm[rng() % static_cast<size_t>(n)].mv, rng);
        }
        if (g.terminal()) continue;
        texas::LegalMove lm[texas::kNumActions];
        const int n = texas::legalMoves(g, lm);
        if (n <= 0) continue;
        const texas::Move mv = texas::policySearchMove(actors, actors + 1, g, g.toAct, 2, rng, 4);
        ++checked;
        bool ok = false;
        for (int i = 0; i < n; ++i) {
            if (lm[i].mv.kind != mv.kind) continue;
            if (mv.kind == texas::MK_RAISE || mv.kind == texas::MK_ALLIN) {
                ok = (mv.to >= int(lm[i].mv.to) && mv.to <= lm[i].toHi);
            } else {
                ok = true;
            }
            if (ok) break;
        }
        if (!ok) ++illegal;
    }
    CHECK(illegal == 0);
    CHECK(checked > 10);
}

// ---------------------------------------------------------------- Slumbot 状态重建

// Slumbot 的动作串（b200c/kb400）必须能无歧义地还原成本地 HUState：
// 本街投入、底池、行动权、公共牌张数都要对上。
void testSlumbotState() {
    section("Slumbot 状态重建");

    texas::Rng rng(2026);
    texas::HUState st;

    // 翻前：SB 加注到 200，BB 跟注；翻牌 BB 过牌，SB 下注 400，轮到 BB。
    CHECK(texas::buildSlumbotState("b200c/kb400",
                                   "Qs", "8h", {"Kd", "7c", "2s"}, /*clientPos=*/0, rng, st));
    CHECK(st.street == texas::ST_FLOP);
    CHECK(st.boardN == 3);
    CHECK(st.hole[1][0] == texas::parseCard("Qs"));
    CHECK(st.hole[1][1] == texas::parseCard("8h"));
    CHECK(st.toAct == 1);                    // BB = 我们
    CHECK(int(st.contrib[0]) == 400);        // SB 本街下注 400
    CHECK(int(st.contrib[1]) == 0);
    CHECK(st.pot() == 800);                  // 400（翻前）+ 400（翻牌）
    CHECK(int(st.total[0]) == 600);
    CHECK(int(st.total[1]) == 200);

    // 翻前直接全下跟注：b20000c///（ACPC 里空街用 / 表示）
    CHECK(texas::buildSlumbotState("b20000c///", "Ah", "Ad", {}, /*clientPos=*/0, rng, st));
    CHECK(st.finished == 1);
    CHECK(int(st.total[0]) == texas::kStartingStack);
    CHECK(int(st.total[1]) == texas::kStartingStack);
    CHECK(st.pot() == 2 * texas::kStartingStack);

    // 非法输入必须拒绝，而不是构造出一个坏状态
    texas::HUState bad;
    CHECK(!texas::buildSlumbotState("b200c/kb400", "Qs", "Qs", {"Kd"}, 0, rng, bad));
    CHECK(!texas::buildSlumbotState("b200c/kb400", "Qs", "8h", {"Qs"}, 0, rng, bad));
    CHECK(!texas::buildSlumbotState("zzz", "Qs", "8h", {}, 0, rng, bad));

    // 重建后的状态可以直接喂给 policyMove，返回的必须是合法动作
    if (texas::buildSlumbotState("b200c/kb400", "Qs", "8h", {"Kd", "7c", "2s"}, 0, rng, st)) {
        texas::nn::ModelConfig mc;
        texas::nn::Actor a[2];
        for (int s = 0; s < 2; ++s) {
            a[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        }
        const texas::Move mv = texas::policyMove(&a[0], &a[1], st, rng, 0.1f);
        texas::LegalMove lm[texas::kNumActions];
        const int n = texas::legalMoves(st, lm);
        bool ok = false;
        for (int i = 0; i < n; ++i) {
            if (lm[i].mv.kind != mv.kind) continue;
            if (mv.kind == texas::MK_RAISE || mv.kind == texas::MK_ALLIN) {
                ok = (mv.to >= int(lm[i].mv.to) && mv.to <= lm[i].toHi);
            } else {
                ok = true;
            }
            if (ok) break;
        }
        CHECK(ok);
    }
}

// ---------------------------------------------------------------- 全下 EV 调整

// 河牌前全下用全下时的胜率折算底池：AA 打 KK 翻前全下，调整值应接近
// 0.82 × 40000 − 20000；河牌全下或弃牌则不调整。
void testAllinAdjusted() {
    section("全下 EV 调整");

    texas::Card aa[2] = {texas::parseCard("As"), texas::parseCard("Ah")};
    texas::Card kk[2] = {texas::parseCard("Ks"), texas::parseCard("Kh")};
    texas::Card board[5] = {texas::parseCard("2c"), texas::parseCard("7d"),
                            texas::parseCard("9s"), texas::parseCard("Jc"),
                            texas::parseCard("3d")};

    // 翻前全下：b20000c（小盲全下，大盲跟）。clientPos=1 = 我们是小盲。
    double adj = -123.0;
    CHECK(texas::allinAdjustedWinnings("b20000c", aa, kk, board, /*clientPos=*/1,
                                       texas::kStartingStack, adj));
    CHECK(adj > 10000.0 && adj < 15000.0);  // 0.817×40000−20000 ≈ 12700

    // 弃牌结束：没有全下，不调整，adj 保持原值。
    double keep = -456.0;
    CHECK(!texas::allinAdjustedWinnings("f", aa, kk, board, 1, -50, keep));
    CHECK_NEAR(keep, -456.0, 1e-12);

    // 转牌全下且我们拿着皇家同花顺：胜率 1，调整值 = 底池 − 投入。
    texas::Card royal[2] = {texas::parseCard("Th"), texas::parseCard("2c")};
    texas::Card nines[2] = {texas::parseCard("9s"), texas::parseCard("9d")};
    texas::Card royalBoard[5] = {texas::parseCard("Ah"), texas::parseCard("Kh"),
                                 texas::parseCard("Qh"), texas::parseCard("Jh"),
                                 texas::parseCard("3d")};
    // 三条街投满 20000，底池 40000；胜率 1 → 调整值 = 40000−20000。
    double lock = 0.0;
    CHECK(texas::allinAdjustedWinnings("b200c/kb1000c/kb20000c", royal, nines, royalBoard,
                                       /*clientPos=*/1, 2 * texas::kStartingStack, lock));
    CHECK_NEAR(lock, double(texas::kStartingStack), 1e-9);
}

// ---------------------------------------------------------------- 任意金额

// ACPC 的规则是"任何不小于最小加注的整数金额"。这一节钉住三件事：
//   1) 引擎接受任意合法整数（不是几个预设档位）；
//   2) 尺寸带首尾相接铺满整个合法区间，不重不漏；
//   3) 带内落点连续，任意整数都可能被取到。
void testAnyAmount() {
    section("任意下注金额");

    texas::Rng rng(2026);
    texas::HUState s = rigged("Ah", "Ad", "Kh", "Kd");  // 翻前，小盲先动

    // 1) 合法区间：最小加注 = 2 个大盲，上限 = 全下
    CHECK(texas::minRaiseTo(s) == 2 * texas::kBigBlind);
    CHECK(texas::maxRaiseTo(s) == texas::kStartingStack);

    // 2) 任意整数金额都合法
    CHECK(texas::canRaiseTo(s, 2 * texas::kBigBlind));
    CHECK(texas::canRaiseTo(s, 1234));
    CHECK(texas::canRaiseTo(s, 7777));
    CHECK(texas::canRaiseTo(s, texas::kStartingStack));
    CHECK(!texas::canRaiseTo(s, 2 * texas::kBigBlind - 1));
    CHECK(!texas::canRaiseTo(s, texas::kStartingStack + 1));
    CHECK(!texas::canRaiseTo(s, 0));
    CHECK(!texas::canRaiseTo(s, -5));

    // 3) 真能执行：加注到 1234，之后的最小加注跟着 1234 重算
    {
        texas::HUState g = s;
        const texas::Move m = texas::makeRaise(g, 1234);
        CHECK(m.kind == texas::MK_RAISE);
        CHECK(int(m.to) == 1234);
        texas::applyMove(g, m, rng);
        CHECK(g.contrib[0] == 1234);
        CHECK(g.lastRaiseTo == 1234);
        CHECK(g.lastRaiseSize == 1234 - texas::kBigBlind);
        CHECK(g.toAct == 1);
        const int nextMin = 1234 + (1234 - texas::kBigBlind);
        CHECK(texas::minRaiseTo(g) == nextMin);
        CHECK(texas::canRaiseTo(g, nextMin));
        CHECK(!texas::canRaiseTo(g, nextMin - 1));
    }

    // 3b) ACPC 的"短全下"：加注额不够最小加注时，把全部筹码推进去仍然合法
    {
        texas::HUState g = s;
        texas::applyMove(g, texas::makeRaise(g, 15000), rng);  // 小盲加到 15000
        CHECK(g.toAct == 1);
        const int hi = texas::maxRaiseTo(g);
        const int lo = texas::minRaiseTo(g);
        CHECK(lo == hi);              // 凑不出完整最小加注 -> 最小加注被钳到全部筹码
        CHECK(texas::isRaisePossible(g));
        CHECK(texas::canRaiseTo(g, hi));   // 全下这一档仍然合法（ACPC 的短全下）
        CHECK(!texas::canRaiseTo(g, hi - 1));
        CHECK(texas::makeRaise(g, hi).kind == texas::MK_ALLIN);
        CHECK(texas::makeRaise(g, hi).to == hi);
    }

    // 4) makeRaise 的钳位与全下退化
    CHECK(texas::makeRaise(s, 1).to == 2 * texas::kBigBlind);
    CHECK(texas::makeRaise(s, texas::kStartingStack).kind == texas::MK_ALLIN);
    CHECK(texas::makeRaise(s, texas::kStartingStack + 500).kind == texas::MK_ALLIN);

    // 5) 尺寸带：首尾相接、铺满 [最小加注, 全下)，槽位唯一
    {
        texas::LegalMove lm[texas::kNumActions];
        const int n = texas::legalMoves(s, lm);
        CHECK(n <= texas::kNumActions);

        bool slotUsed[texas::kNumActions] = {false};
        int bands = 0;
        int prevHi = -1;
        for (int i = 0; i < n; ++i) {
            CHECK(!slotUsed[lm[i].slot]);
            slotUsed[lm[i].slot] = true;
            if (lm[i].mv.kind != texas::MK_RAISE) continue;
            if (bands == 0) {
                CHECK(lm[i].mv.to == texas::minRaiseTo(s));
            } else {
                CHECK(lm[i].mv.to == prevHi + 1);  // 不重不漏
            }
            CHECK(lm[i].toHi >= lm[i].mv.to);
            prevHi = lm[i].toHi;
            ++bands;
        }
        CHECK(bands > 1);
        CHECK(bands <= texas::kMaxSizes);
        CHECK(prevHi == texas::maxRaiseTo(s) - 1);  // 带只到全下前一档
    }

    // 6) 一个决策点上"可达金额"是连续的，不是有限几个档位
    {
        texas::LegalMove lm[texas::kNumActions];
        const int n = texas::legalMoves(s, lm);
        std::set<int> reach;
        int bands = 0;
        for (int i = 0; i < n; ++i) {
            if (lm[i].mv.kind != texas::MK_RAISE) continue;
            ++bands;
            for (int k = 0; k <= 400; ++k) {
                reach.insert(texas::amountInBand(int(lm[i].mv.to), lm[i].toHi, double(k) / 400.0));
            }
        }
        CHECK(bands >= 10);            // 旧版只有 5 个固定档
        CHECK(reach.size() > 1000);    // 旧版这里最多 5 个金额
        CHECK(reach.count(1234) == 1);  // 随手一个数都能打出来
        CHECK(reach.count(2 * texas::kBigBlind) == 1);
        CHECK(reach.count(texas::kStartingStack - 1) == 1);
    }

    // 7) 带内落点：单调、端点可达、能取到几千个不同金额
    {
        const int lo = 500;
        const int hi = 5000;
        CHECK(texas::amountInBand(lo, hi, 0.0) == lo);
        CHECK(texas::amountInBand(lo, hi, 1.0) == hi);

        std::set<int> vals;
        int prev = 0;
        bool mono = true;
        for (int i = 0; i <= 10000; ++i) {
            const int a = texas::amountInBand(lo, hi, double(i) / 10000.0);
            if (a < prev) mono = false;
            if (a < lo || a > hi) mono = false;
            prev = a;
            vals.insert(a);
        }
        CHECK(mono);
        CHECK(vals.size() > 3000);  // 上限 4501 个整数，覆盖了大部分
        CHECK(vals.count(lo) == 1);
        CHECK(vals.count(hi) == 1);
    }

    // 8) 胜率机器人的动作也始终合法（它会打任意金额，且比规则机器人更爱加注）
    {
        int illegal = 0;
        for (int t = 0; t < 400; ++t) {
            texas::HUState g = texas::dealNewHand(rng);
            const int steps = int(rng() % 6);
            for (int i = 0; i < steps && !g.terminal(); ++i) {
                texas::LegalMove lm[texas::kNumActions];
                const int n = texas::legalMoves(g, lm);
                if (n <= 0) break;
                texas::applyMove(g, lm[rng() % static_cast<size_t>(n)].mv, rng);
            }
            if (g.terminal()) continue;
            const texas::Move m = texas::proAgentMove(g, rng);
            const int need = texas::toCall(g, g.toAct);
            if (m.kind == texas::MK_RAISE && !texas::canRaiseTo(g, m.to)) ++illegal;
            if (m.kind == texas::MK_ALLIN && m.to != texas::maxRaiseTo(g)) ++illegal;
            if (m.kind == texas::MK_CHECK && need > 0) ++illegal;
        }
        CHECK(illegal == 0);
    }

    // 9) 规则机器人的金额始终合法，且随底池连续变化（不是固定档位）
    {
        std::set<int> amts;
        int illegal = 0;
        for (int t = 0; t < 3000; ++t) {
            texas::HUState g = texas::dealNewHand(rng);
            const int steps = int(rng() % 6);
            for (int i = 0; i < steps && !g.terminal(); ++i) {
                texas::LegalMove lm[texas::kNumActions];
                const int n = texas::legalMoves(g, lm);
                if (n <= 0) break;
                texas::applyMove(g, lm[rng() % static_cast<size_t>(n)].mv, rng);
            }
            if (g.terminal()) continue;
            const texas::Move m = texas::ruleAgentMove(g, rng);
            if (m.kind == texas::MK_RAISE || m.kind == texas::MK_ALLIN) {
                if (!texas::canRaiseTo(g, m.to)) ++illegal;
                amts.insert(int(m.to));
            }
        }
        CHECK(illegal == 0);
        CHECK(amts.size() > 10);
    }

    // 10) 跟注型稳健机器人：动作合法，且风格确实是"宽跟注"（跟注多于弃牌）
    {
        int illegal = 0, calls = 0, folds = 0;
        for (int t = 0; t < 400; ++t) {
            texas::HUState g = texas::dealNewHand(rng);
            const int steps = int(rng() % 6);
            for (int i = 0; i < steps && !g.terminal(); ++i) {
                texas::LegalMove lm[texas::kNumActions];
                const int n = texas::legalMoves(g, lm);
                if (n <= 0) break;
                texas::applyMove(g, lm[rng() % static_cast<size_t>(n)].mv, rng);
            }
            if (g.terminal()) continue;
            const texas::Move m = texas::solidAgentMove(g, rng);
            const int need = texas::toCall(g, g.toAct);
            if (m.kind == texas::MK_RAISE && !texas::canRaiseTo(g, m.to)) ++illegal;
            if (m.kind == texas::MK_ALLIN && m.to != texas::maxRaiseTo(g)) ++illegal;
            if (m.kind == texas::MK_CHECK && need > 0) ++illegal;
            if (m.kind == texas::MK_CALL) ++calls;
            if (m.kind == texas::MK_FOLD) ++folds;
        }
        CHECK(illegal == 0);
        CHECK(calls > folds);
    }

    // 11) 按尺寸弃牌机器人：动作合法；翻牌后面对过牌永远过牌
    //     （把"要不要用大尺寸诈唬"完全留给被测策略）。
    {
        int illegal = 0, checkedTo = 0, passive = 0;
        for (int t = 0; t < 800; ++t) {
            texas::HUState g = texas::dealNewHand(rng);
            const int steps = int(rng() % 8);
            for (int i = 0; i < steps && !g.terminal(); ++i) {
                texas::LegalMove lm[texas::kNumActions];
                const int n = texas::legalMoves(g, lm);
                if (n <= 0) break;
                texas::applyMove(g, lm[rng() % static_cast<size_t>(n)].mv, rng);
            }
            if (g.terminal()) continue;
            const texas::Move m = texas::folderAgentMove(g, rng);
            const int need = texas::toCall(g, g.toAct);
            if (m.kind == texas::MK_RAISE && !texas::canRaiseTo(g, m.to)) ++illegal;
            if (m.kind == texas::MK_ALLIN && m.to != texas::maxRaiseTo(g)) ++illegal;
            if (m.kind == texas::MK_CHECK && need > 0) ++illegal;
            if (g.street > 0 && need == 0) {
                ++checkedTo;
                if (m.kind == texas::MK_CHECK) ++passive;
            }
        }
        CHECK(illegal == 0);
        CHECK(checkedTo > 20);
        CHECK(passive == checkedTo);
    }

    // 12) 紧全下机器人：动作始终合法；只用顶范围推/接全下。
    {
        int illegal = 0;
        for (int t = 0; t < 300; ++t) {
            texas::HUState g = texas::dealNewHand(rng);
            const int steps = int(rng() % 8);
            for (int i = 0; i < steps && !g.terminal(); ++i) {
                texas::LegalMove lm[texas::kNumActions];
                const int n = texas::legalMoves(g, lm);
                if (n <= 0) break;
                texas::applyMove(g, lm[rng() % static_cast<size_t>(n)].mv, rng);
            }
            if (g.terminal()) continue;
            const texas::Move m = texas::tightAgentMove(g, rng);
            const int need = texas::toCall(g, g.toAct);
            if (m.kind == texas::MK_RAISE && !texas::canRaiseTo(g, m.to)) ++illegal;
            if (m.kind == texas::MK_ALLIN && m.to != texas::maxRaiseTo(g)) ++illegal;
            if (m.kind == texas::MK_CHECK && need > 0) ++illegal;
        }
        CHECK(illegal == 0);
    }

    // 12b) AA 翻前面对开牌要加注；72o 面对 3-bet 要弃牌。
    {
        const texas::Card aa[2] = {C("Ah"), C("Ad")};
        const texas::Card junk[2] = {C("7c"), C("2d")};
        const texas::Card other[2] = {C("Ks"), C("Qh")};
        texas::Move open{};
        open.kind = texas::MK_RAISE;
        open.to = 300;

        texas::HUState g = texas::makeState(other, aa);  // SB=其他, BB=AA
        texas::applyMove(g, open, rng);
        const texas::Move m = texas::tightAgentMove(g, rng);
        CHECK(m.kind == texas::MK_RAISE || m.kind == texas::MK_ALLIN);

        texas::HUState h = texas::makeState(junk, other);  // SB=72o
        texas::applyMove(h, open, rng);
        texas::Move threeBet{};
        threeBet.kind = texas::MK_RAISE;
        threeBet.to = 900;
        texas::applyMove(h, threeBet, rng);
        const texas::Move f = texas::tightAgentMove(h, rng);
        CHECK(f.kind == texas::MK_FOLD);
    }

    // 12c) 河牌坚果无人下注要下注；河牌一对面对全下要弃牌。
    {
        const texas::Card nuts[2] = {C("6h"), C("7h")};
        const texas::Card weak[2] = {C("Kd"), C("Qd")};
        const texas::Card other[2] = {C("As"), C("Ac")};
        texas::Card board[5] = {C("2c"), C("3d"), C("4h"), C("5s"), C("9c")};

        texas::HUState g = texas::makeState(nuts, other, board, 5);
        texas::Move call{};
        call.kind = texas::MK_CALL;
        call.to = 100;
        texas::applyMove(g, call, rng);
        for (int st = 0; st < 3; ++st) {
            texas::Move chk{};
            chk.kind = texas::MK_CHECK;
            chk.to = g.contrib[g.toAct];
            texas::applyMove(g, chk, rng);
            chk.to = g.contrib[g.toAct];
            texas::applyMove(g, chk, rng);
        }
        CHECK(g.street == texas::ST_RIVER);
        CHECK(texas::toCall(g, g.toAct) == 0);
        const texas::Move bet = texas::tightAgentMove(g, rng);
        CHECK(bet.kind == texas::MK_RAISE || bet.kind == texas::MK_ALLIN);

        texas::HUState h = texas::makeState(other, weak, board, 5);
        texas::Move call2 = call;
        texas::applyMove(h, call2, rng);
        for (int st = 0; st < 3; ++st) {
            texas::Move chk{};
            chk.kind = texas::MK_CHECK;
            chk.to = h.contrib[h.toAct];
            texas::applyMove(h, chk, rng);
            chk.to = h.contrib[h.toAct];
            texas::applyMove(h, chk, rng);
        }
        // 河牌 BB（弱牌）先行动：SB 先全下，BB 面对全下只能弃。
        const texas::Move jam = texas::makeRaise(h, int(h.contrib[h.toAct]) + int(h.stack[h.toAct]));
        texas::applyMove(h, jam, rng);
        CHECK(h.toAct == 1);
        CHECK(texas::toCall(h, 1) > 0);
        const texas::Move f = texas::tightAgentMove(h, rng);
        CHECK(f.kind == texas::MK_FOLD);
    }

    // 13b) 陷阱机器人：动作合法；翻前从不加注（强牌也只跟）。
    {
        int illegal = 0, preflopRaises = 0, preflopDecisions = 0;
        for (int t = 0; t < 300; ++t) {
            texas::HUState g = texas::dealNewHand(rng);
            const int steps = int(rng() % 8);
            for (int i = 0; i < steps && !g.terminal(); ++i) {
                texas::LegalMove lm[texas::kNumActions];
                const int n = texas::legalMoves(g, lm);
                if (n <= 0) break;
                texas::applyMove(g, lm[rng() % static_cast<size_t>(n)].mv, rng);
            }
            if (g.terminal()) continue;
            const bool pre = (g.street == texas::ST_PREFLOP);
            const texas::Move m = texas::trapperAgentMove(g, rng);
            const int need = texas::toCall(g, g.toAct);
            if (m.kind == texas::MK_RAISE && !texas::canRaiseTo(g, m.to)) ++illegal;
            if (m.kind == texas::MK_ALLIN && m.to != texas::maxRaiseTo(g)) ++illegal;
            if (m.kind == texas::MK_CHECK && need > 0) ++illegal;
            if (pre) {
                ++preflopDecisions;
                if (m.kind == texas::MK_RAISE || m.kind == texas::MK_ALLIN) ++preflopRaises;
            }
        }
        CHECK(illegal == 0);
        CHECK(preflopDecisions > 50);
        CHECK(preflopRaises == 0);
    }

    // 13c) 价值型强对手（shark）：动作合法；翻前不 5-bet 诈唬；翻后两对不弃、
    // 空气弃大注、无人下注时空气过牌。
    {
        int illegal = 0;
        for (int t = 0; t < 300; ++t) {
            texas::HUState g = texas::dealNewHand(rng);
            const int steps = int(rng() % 8);
            for (int i = 0; i < steps && !g.terminal(); ++i) {
                texas::LegalMove lm[texas::kNumActions];
                const int n = texas::legalMoves(g, lm);
                if (n <= 0) break;
                texas::applyMove(g, lm[rng() % static_cast<size_t>(n)].mv, rng);
            }
            if (g.terminal()) continue;
            const texas::Move m = texas::sharkAgentMove(g, rng);
            const int need = texas::toCall(g, g.toAct);
            if (m.kind == texas::MK_RAISE && !texas::canRaiseTo(g, m.to)) ++illegal;
            if (m.kind == texas::MK_ALLIN && m.to != texas::maxRaiseTo(g)) ++illegal;
            if (m.kind == texas::MK_CHECK && need > 0) ++illegal;
        }
        CHECK(illegal == 0);

        // 翻前 5-bet 后用中等牌（QJ offsuit）：只跟或弃，绝不 6-bet。
        {
            const texas::Card mine[2] = {C("Qd"), C("Jc")};
            const texas::Card other[2] = {C("Ah"), C("Kd")};
            texas::HUState g = texas::makeState(mine, other);
            const int seq[4] = {200, 600, 1600, 4000};  // SB 开、BB 3-bet、SB 4-bet、BB 5-bet
            for (int k = 0; k < 4; ++k) {
                const texas::Move r = texas::makeRaise(g, seq[k]);
                CHECK(r.kind == texas::MK_RAISE || r.kind == texas::MK_ALLIN);
                texas::applyMove(g, r, rng);
            }
            CHECK(g.toAct == 0);
            CHECK(g.numRaises >= 3);
            const texas::Move m = texas::sharkAgentMove(g, rng);
            CHECK(m.kind != texas::MK_RAISE);
            CHECK(m.kind != texas::MK_ALLIN);
        }

        // 翻后两对面对 3/4 池下注：不弃（跟或加）。
        {
            const texas::Card strong[2] = {C("Ah"), C("9d")};
            const texas::Card weak[2] = {C("7c"), C("2d")};
            texas::Card board[5] = {C("Ad"), C("9s"), C("5s"), C("2h"), C("6h")};
            texas::HUState g = texas::makeState(strong, weak, board, 5);
            texas::Move call{};
            call.kind = texas::MK_CALL;
            call.to = 100;
            texas::applyMove(g, call, rng);
            texas::Move chk{};
            chk.kind = texas::MK_CHECK;
            chk.to = g.contrib[1];
            texas::applyMove(g, chk, rng);
            CHECK(g.street == texas::ST_FLOP);
            // 底池 200，下注 100（半池）：两对必须继续（赔率 <= 0.30 即调用）。
            const texas::Move bet = texas::makeRaise(g, int(g.contrib[1]) + 100);
            texas::applyMove(g, bet, rng);
            CHECK(g.toAct == 0);
            const texas::Move m = texas::sharkAgentMove(g, rng);
            CHECK(m.kind == texas::MK_CALL || m.kind == texas::MK_RAISE ||
                  m.kind == texas::MK_ALLIN);
        }

        // 翻后空气面对 3/4 池下注：必须弃。
        {
            const texas::Card air[2] = {C("7c"), C("2d")};
            const texas::Card other[2] = {C("Ah"), C("Kd")};
            texas::Card board[5] = {C("Kc"), C("Qs"), C("8h"), C("3d"), C("5c")};
            texas::HUState g = texas::makeState(air, other, board, 5);
            texas::Move call{};
            call.kind = texas::MK_CALL;
            call.to = 100;
            texas::applyMove(g, call, rng);
            texas::Move chk{};
            chk.kind = texas::MK_CHECK;
            chk.to = g.contrib[1];
            texas::applyMove(g, chk, rng);
            CHECK(g.street == texas::ST_FLOP);
            // 底池 200，下注 150（3/4 池）。
            const texas::Move bet = texas::makeRaise(g, int(g.contrib[1]) + 150);
            texas::applyMove(g, bet, rng);
            CHECK(g.toAct == 0);
            const texas::Move m = texas::sharkAgentMove(g, rng);
            CHECK(m.kind == texas::MK_FOLD);
        }

        // 翻后无人下注 + 空气：过牌（不空气开火）。
        {
            const texas::Card air[2] = {C("7c"), C("2d")};
            const texas::Card other[2] = {C("Ah"), C("Kd")};
            texas::Card board[5] = {C("Kc"), C("Qs"), C("8h"), C("3d"), C("5c")};
            texas::HUState g = texas::makeState(air, other, board, 5);
            texas::Move call{};
            call.kind = texas::MK_CALL;
            call.to = 100;
            texas::applyMove(g, call, rng);
            texas::Move chk{};
            chk.kind = texas::MK_CHECK;
            chk.to = g.contrib[1];
            texas::applyMove(g, chk, rng);
            texas::Move chk2{};
            chk2.kind = texas::MK_CHECK;
            chk2.to = g.contrib[1];
            texas::applyMove(g, chk2, rng);
            CHECK(g.toAct == 0);
            CHECK(texas::toCall(g, 0) == 0);
            const texas::Move m = texas::sharkAgentMove(g, rng);
            CHECK(m.kind == texas::MK_CHECK);
        }
    }

    // 13a) 小注骚扰机器人：动作合法；翻后无人下注时高频打小注。
    {
        int illegal = 0;
        for (int t = 0; t < 300; ++t) {
            texas::HUState g = texas::dealNewHand(rng);
            const int steps = int(rng() % 8);
            for (int i = 0; i < steps && !g.terminal(); ++i) {
                texas::LegalMove lm[texas::kNumActions];
                const int n = texas::legalMoves(g, lm);
                if (n <= 0) break;
                texas::applyMove(g, lm[rng() % static_cast<size_t>(n)].mv, rng);
            }
            if (g.terminal()) continue;
            const texas::Move m = texas::spammerAgentMove(g, rng);
            const int need = texas::toCall(g, g.toAct);
            if (m.kind == texas::MK_RAISE && !texas::canRaiseTo(g, m.to)) ++illegal;
            if (m.kind == texas::MK_ALLIN && m.to != texas::maxRaiseTo(g)) ++illegal;
            if (m.kind == texas::MK_CHECK && need > 0) ++illegal;
        }
        CHECK(illegal == 0);

        // 河牌无人下注：200 次里应有约 65% 的下注（与牌力无关）。
        const texas::Card ha[2] = {C("Ah"), C("Kd")};
        const texas::Card hb[2] = {C("Qs"), C("Jh")};
        texas::Card board[5] = {C("2c"), C("7d"), C("9h"), C("Ts"), C("3c")};
        texas::HUState g = texas::makeState(ha, hb, board, 5);
        texas::Move call{};
        call.kind = texas::MK_CALL;
        call.to = 100;
        texas::applyMove(g, call, rng);
        for (int st = 0; st < 3; ++st) {
            texas::Move chk{};
            chk.kind = texas::MK_CHECK;
            chk.to = g.contrib[g.toAct];
            texas::applyMove(g, chk, rng);
            chk.to = g.contrib[g.toAct];
            texas::applyMove(g, chk, rng);
        }
        CHECK(g.street == texas::ST_RIVER);
        CHECK(texas::toCall(g, g.toAct) == 0);
        int bets = 0;
        for (int t = 0; t < 200; ++t) {
            const texas::Move m = texas::spammerAgentMove(g, rng);
            if (m.kind == texas::MK_RAISE || m.kind == texas::MK_ALLIN) ++bets;
        }
        CHECK(bets > 100);
    }

    // 13) 全下纪律 guard：弱牌面对大额全下要弃，强牌不弃。
    {
        const texas::Card weak[2] = {C("Kd"), C("Qd")};
        const texas::Card strong[2] = {C("Ah"), C("Ad")};
        const texas::Card other[2] = {C("Qh"), C("Jh")};
        auto preflopJam = [&](const texas::Card* mine) {
            texas::HUState g = texas::makeState(mine, other);  // 我们是 SB
            texas::Move call{};
            call.kind = texas::MK_CALL;
            call.to = 100;
            texas::applyMove(g, call, rng);
            const texas::Move jam =
                texas::makeRaise(g, int(g.contrib[g.toAct]) + int(g.stack[g.toAct]));
            texas::applyMove(g, jam, rng);
            return g;
        };
        texas::HUState w = preflopJam(weak);
        CHECK(w.toAct == 0);
        CHECK(texas::allinGuardFold(w, 0.75f, 3, 0.5f));
        texas::HUState sp = preflopJam(strong);
        CHECK(!texas::allinGuardFold(sp, 0.75f, 3, 0.5f));

        // 翻后：河牌弱牌面对 SB 全下要弃，坚果不弃。
        const texas::Card nuts[2] = {C("6h"), C("7h")};
        texas::Card board[5] = {C("2c"), C("3d"), C("4h"), C("5s"), C("9c")};
        auto riverJam = [&](const texas::Card* bbHole) {
            texas::HUState g = texas::makeState(other, bbHole, board, 5);  // BB = 被测手牌
            texas::Move call{};
            call.kind = texas::MK_CALL;
            call.to = 100;
            texas::applyMove(g, call, rng);
            for (int st = 0; st < 3; ++st) {
                texas::Move chk{};
                chk.kind = texas::MK_CHECK;
                chk.to = g.contrib[g.toAct];
                texas::applyMove(g, chk, rng);
                chk.to = g.contrib[g.toAct];
                texas::applyMove(g, chk, rng);
            }
            const texas::Move jam =
                texas::makeRaise(g, int(g.contrib[g.toAct]) + int(g.stack[g.toAct]));
            texas::applyMove(g, jam, rng);
            return g;
        };
        texas::HUState wr = riverJam(weak);
        CHECK(wr.toAct == 1);
        CHECK(texas::allinGuardFold(wr, 0.75f, 3, 0.5f));
        texas::HUState nr = riverJam(nuts);
        CHECK(!texas::allinGuardFold(nr, 0.75f, 3, 0.5f));
    }
}

}  // namespace

int main() {
    testCards();
    testRules();
    testFeatures();
    testAnyAmount();
    testPpo();
    testNetwork();
    testTeacher();
    testLeague();
    testEnsemble();
    testSharpSelfPlay();
    testMixedFrozen();
    testLearnerSeatBias();
    testDuplicateDeals();
    testPolicySearch();
    testSlumbotState();
    testAllinAdjusted();
    std::printf("\n%d 通过, %d 失败\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
