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
#include <set>
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

// 两级历史池：recent 滚动淘汰，archive 几何间隔保留。
// 这里只验结构性质，不建真网络（Snapshot 里的 actor 留空即可）。
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

    // 8) 规则机器人的金额始终合法，且随底池连续变化（不是固定档位）
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
}

}  // namespace

int main() {
    testCards();
    testRules();
    testFeatures();
    testAnyAmount();
    testPpo();
    testNetwork();
    testLeague();
    std::printf("\n%d 通过, %d 失败\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
