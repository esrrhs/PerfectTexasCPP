// 自测：牌力评估、规则引擎、特征编码、PPO 数学、网络存盘往返。
//
// 硬断言优先用"不依赖训练"的性质：
//   - 任意随机对局必须零和、筹码守恒、必然终局
//   - 合法动作里必须恰好有一个跟注、加注额严格递增、最小加注不被违反
//   - 河牌的完美信息胜率必须与直接比牌一致
//   - 掩码 softmax 的概率和为 1，非法动作概率为 0

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "texas/card.hpp"
#include "texas/features.hpp"
#include "texas/hu.hpp"
#include "texas/nn.hpp"
#include "texas/ppo.hpp"
#include "texas/rollout.hpp"
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
    CHECK(s.pot() == 3);
    CHECK(s.toAct == 0);
    CHECK(s.contrib[0] == 1 && s.contrib[1] == 2);

    texas::LegalMove lm[16];
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
            CHECK(lm[i].mv.to >= 4);
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
    CHECK(s2.pot() == 4);

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
        CHECK(texas::payoff(f, 0) == -1);
        CHECK(texas::payoff(f, 1) == 1);
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

    // 随机对局：必然终局、零和、筹码守恒
    int zeroSumFail = 0;
    int notTerminal = 0;
    int c0 = 0;
    int c1 = 0;
    long long decisions = 0;
    for (int t = 0; t < 20000; ++t) {
        texas::HUState g = texas::dealNewHand(rng);
        int guard = 0;
        while (!g.terminal() && guard < 200) {
            texas::LegalMove mv[16];
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

    texas::LegalMove lm[16];
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

}  // namespace

int main() {
    testCards();
    testRules();
    testFeatures();
    testPpo();
    testNetwork();
    std::printf("\n%d 通过, %d 失败\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
