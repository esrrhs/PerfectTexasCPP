// 自测：牌力评估、胜率、CFR 收敛性、均衡策略性质。
// 关键的硬断言用解析解做对照：
//   - Kuhn 先手博弈值 = -1/18
//   - Kuhn 均衡策略的已知结构（α / 3α 关系、Q 从不主动下注等）
//   - 均衡策略的 exploitability 必须显著低于均匀策略

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "texas/br.hpp"
#include "texas/card.hpp"
#include "texas/cfr.hpp"
#include "texas/equity.hpp"
#include "texas/holdem.hpp"
#include "texas/kuhn.hpp"
#include "texas/leduc.hpp"
#include "texas/mccfr.hpp"

namespace {

int g_failed = 0;
int g_passed = 0;
const char* g_section = "";

void section(const char* name) {
    g_section = name;
    std::printf("\n[%s]\n", name);
}

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);     \
            ++g_failed;                                                      \
        } else {                                                             \
            ++g_passed;                                                      \
        }                                                                    \
    } while (0)

#define CHECK_NEAR(actual, expect, eps)                                       \
    do {                                                                      \
        const double a_ = (actual);                                           \
        const double e_ = (expect);                                           \
        if (!(std::fabs(a_ - e_) <= (eps))) {                                 \
            std::printf("  FAIL %s:%d  %s = %.6f, 期望 %.6f (容差 %.4f)\n",    \
                        __FILE__, __LINE__, #actual, a_, e_, (double)(eps));   \
            ++g_failed;                                                       \
        } else {                                                              \
            ++g_passed;                                                       \
        }                                                                     \
    } while (0)

texas::Card C(const char* s) { return texas::parseCard(s); }

int scoreOf5(const char* a, const char* b, const char* c, const char* d, const char* e) {
    const texas::Card h[5] = {C(a), C(b), C(c), C(d), C(e)};
    for (int i = 0; i < 5; ++i) {
        if (h[i] < 0) {
            std::printf("  牌面解析失败: %s/%s/%s/%s/%s\n", a, b, c, d, e);
            return -1;
        }
    }
    return texas::evaluate5(h);
}

int categoryOf5(const char* a, const char* b, const char* c, const char* d, const char* e) {
    return texas::handCategoryOf(scoreOf5(a, b, c, d, e));
}

void testCards() {
    section("牌面与牌力");

    CHECK(C("Ah") == 12 * 4 + 2);
    CHECK(C("2c") == 0);
    CHECK(C("As") == 51);
    CHECK(C("Td") == 8 * 4 + 1);
    CHECK(C("Xx") < 0);

    const int royal = categoryOf5("Ah", "Kh", "Qh", "Jh", "Th");
    const int straightFlush = categoryOf5("9h", "8h", "7h", "6h", "5h");
    const int quads = categoryOf5("As", "Ah", "Ad", "Ac", "Kd");
    const int fullHouse = categoryOf5("As", "Ah", "Ad", "Kc", "Kd");
    const int flush = categoryOf5("Ah", "Kh", "9h", "7h", "2h");
    const int straight = categoryOf5("9c", "Th", "Jd", "Qs", "Kc");
    const int trips = categoryOf5("As", "Ah", "Ad", "Kc", "Qd");
    const int twoPair = categoryOf5("As", "Ah", "Kc", "Kd", "Qs");
    const int onePair = categoryOf5("As", "Ah", "Kc", "Qd", "Js");
    const int high = categoryOf5("As", "Kh", "Qc", "Jd", "9s");

    CHECK(royal == static_cast<int>(texas::HandCategory::StraightFlush));
    CHECK(straightFlush == static_cast<int>(texas::HandCategory::StraightFlush));
    CHECK(quads == static_cast<int>(texas::HandCategory::FourOfAKind));
    CHECK(fullHouse == static_cast<int>(texas::HandCategory::FullHouse));
    CHECK(flush == static_cast<int>(texas::HandCategory::Flush));
    CHECK(straight == static_cast<int>(texas::HandCategory::Straight));
    CHECK(trips == static_cast<int>(texas::HandCategory::ThreeOfAKind));
    CHECK(twoPair == static_cast<int>(texas::HandCategory::TwoPair));
    CHECK(onePair == static_cast<int>(texas::HandCategory::OnePair));
    CHECK(high == static_cast<int>(texas::HandCategory::HighCard));

    // 皇家同花顺 > 普通同花顺
    CHECK(scoreOf5("Ah", "Kh", "Qh", "Jh", "Th") > scoreOf5("9h", "8h", "7h", "6h", "5h"));
    // 类别严格递增
    CHECK(scoreOf5("Ah", "Kh", "Qh", "Jh", "Th") > scoreOf5("As", "Ah", "Ad", "Ac", "Kd"));
    CHECK(quads > fullHouse);
    CHECK(fullHouse > flush);
    CHECK(flush > straight);
    CHECK(straight > trips);
    CHECK(trips > twoPair);
    CHECK(twoPair > onePair);
    CHECK(onePair > high);

    // 轮子 A-2-3-4-5 是顺子，且 6 高的顺子更大
    CHECK(categoryOf5("Ac", "2d", "3h", "4s", "5c") == static_cast<int>(texas::HandCategory::Straight));
    CHECK(scoreOf5("2c", "3d", "4h", "5s", "6c") > scoreOf5("Ac", "2d", "3h", "4s", "5c"));

    // 同花 > 顺子（同一副牌里不能同时成立，这里用不同组合对比类别优先级）
    CHECK(flush > straight);

    // 踢脚牌：KK 带 A > KK 带 Q
    CHECK(scoreOf5("Ks", "Kh", "Ac", "Qd", "Js") > scoreOf5("Kc", "Kd", "Qc", "Jd", "9s"));

    // 7 选 5：应挑出同花顺
    {
        const texas::Card h[7] = {C("As"), C("Ks"), C("Qs"), C("Js"), C("Ts"), C("2c"), C("2d")};
        CHECK(texas::handCategoryOf(texas::evaluate7(h)) ==
              static_cast<int>(texas::HandCategory::StraightFlush));
    }
    // 7 选 5：四条优先于葫芦
    {
        const texas::Card h[7] = {C("As"), C("Ah"), C("Ad"), C("Ac"), C("Kd"), C("Kc"), C("2h")};
        const int s = texas::evaluate7(h);
        CHECK(texas::handCategoryOf(s) == static_cast<int>(texas::HandCategory::FourOfAKind));
    }
    // 7 选 5：同花顺要能穿过 6 张同花的干扰
    {
        const texas::Card h[7] = {C("9s"), C("8s"), C("7s"), C("6s"), C("5s"), C("2s"), C("Kd")};
        CHECK(texas::handCategoryOf(texas::evaluate7(h)) ==
              static_cast<int>(texas::HandCategory::StraightFlush));
    }

    std::printf("  OK 牌型 10 类、轮子顺子、踢脚、7选5 均正确\n");
}

void testEquity() {
    section("蒙特卡洛胜率");

    const std::vector<texas::Card> none;
    const auto aa = texas::equityMonteCarlo({C("Ah"), C("As")}, none, 1, 40000, 7);
    const auto kk = texas::equityMonteCarlo({C("Kh"), C("Ks")}, none, 1, 40000, 7);
    const auto s72 = texas::equityMonteCarlo({C("7h"), C("2c")}, none, 1, 40000, 7);

    std::printf("  AA vs 随机 = %.2f%%, KK = %.2f%%, 72o = %.2f%%\n",
                aa.equity() * 100, kk.equity() * 100, s72.equity() * 100);

    // 已知理论值：AA vs 随机约 85.2%，KK 约 82.4%，72o 约 34.6%
    CHECK_NEAR(aa.equity(), 0.852, 0.012);
    CHECK_NEAR(kk.equity(), 0.824, 0.012);
    CHECK_NEAR(s72.equity(), 0.346, 0.015);
    CHECK(aa.equity() > kk.equity());
    CHECK(kk.equity() > s72.equity());
    CHECK_NEAR(aa.win + aa.tie + aa.lose, 1.0, 1e-9);

    // 已知 5 张公共牌时的胜率：AA 在 A 高的干燥面上应接近必胜
    const auto flopped = texas::equityMonteCarlo({C("Ah"), C("As")},
                                                 {C("Ad"), C("7c"), C("2h")}, 1, 20000, 11);
    std::printf("  AhAs 在 Ad7c2h 面上 = %.2f%%\n", flopped.equity() * 100);
    CHECK(flopped.equity() > 0.95);
}

const std::vector<double>* kuhnSigmaAt(const texas::Strategy& sigma, int card, const std::string& hist) {
    texas::KuhnGame::State st;
    st.priv[0] = card;
    st.priv[1] = card;
    st.hist = hist;
    const auto it = sigma.find(texas::KuhnGame::infosetId(st));
    if (it == sigma.end()) return nullptr;
    return &it->second;
}

double kuhnProb(const texas::Strategy& sigma, int card, const std::string& hist, int action) {
    const std::vector<double>* v = kuhnSigmaAt(sigma, card, hist);
    if (!v || action >= static_cast<int>(v->size())) return -1.0;
    return (*v)[static_cast<std::size_t>(action)];
}

void testKuhn() {
    section("Kuhn Poker / CFR+ 收敛");

    texas::CFRSolver<texas::KuhnGame> solver;
    solver.iterate(20000);
    const texas::Strategy sigma = solver.averageStrategy();
    const double expl = texas::exploitability<texas::KuhnGame>(sigma);
    const double value = solver.expectedValue(sigma);

    std::printf("  iter=%d  infosets=%zu  exploitability=%.6f  value=%+.6f (理论 -1/18=%.6f)\n",
                solver.iterations(), solver.numInfosets(), expl, value, -1.0 / 18.0);

    // 12 个信息集：P0 3 张牌 × 1 个历史 + P1 3 张牌 × 2 个历史
    CHECK(solver.numInfosets() == 12);
    // 博弈值必须收敛到 -1/18
    CHECK_NEAR(value, -1.0 / 18.0, 0.004);
    // 可利用度应接近 0
    CHECK(expl < 0.01);

    section("Kuhn Poker / 均衡策略结构");
    // card: 0=J 1=Q 2=K；action: 0=pass 1=bet
    const double p0_j_bet = kuhnProb(sigma, 0, "", 1);
    const double p0_q_bet = kuhnProb(sigma, 1, "", 1);
    const double p0_k_bet = kuhnProb(sigma, 2, "", 1);
    const double p1_j_bet = kuhnProb(sigma, 0, "p", 1);
    const double p1_q_bet = kuhnProb(sigma, 1, "p", 1);
    const double p1_k_bet = kuhnProb(sigma, 2, "p", 1);
    const double p0_j_call = kuhnProb(sigma, 0, "pb", 1);
    const double p0_q_call = kuhnProb(sigma, 1, "pb", 1);
    const double p0_k_call = kuhnProb(sigma, 2, "pb", 1);
    const double p1_j_call = kuhnProb(sigma, 0, "b", 1);
    const double p1_q_call = kuhnProb(sigma, 1, "b", 1);
    const double p1_k_call = kuhnProb(sigma, 2, "b", 1);

    const double alpha = p0_j_bet;
    std::printf("  α(P0 拿 J 下注)=%.4f  3α=%.4f  P0 拿 K 下注=%.4f\n", alpha, 3 * alpha, p0_k_bet);
    std::printf("  P0 面对下注跟注: J=%.3f Q=%.3f K=%.3f\n", p0_j_call, p0_q_call, p0_k_call);
    std::printf("  P1 过牌后下注  : J=%.3f Q=%.3f K=%.3f\n", p1_j_bet, p1_q_bet, p1_k_bet);
    std::printf("  P1 面对下注跟注: J=%.3f Q=%.3f K=%.3f\n", p1_j_call, p1_q_call, p1_k_call);

    CHECK(p0_j_bet >= -0.01 && p0_j_bet <= 0.36);  // α ∈ [0, 1/3]
    CHECK_NEAR(p0_q_bet, 0.0, 0.03);               // Q 永不主动下注
    CHECK_NEAR(p0_k_bet, 3.0 * alpha, 0.09);       // K 的下注频率是 J 的三倍
    CHECK_NEAR(p1_j_bet, 1.0 / 3.0, 0.07);         // 过牌后 J 诈唬 1/3
    CHECK_NEAR(p1_q_bet, 0.0, 0.05);
    CHECK_NEAR(p1_k_bet, 1.0, 0.05);
    CHECK_NEAR(p0_j_call, 0.0, 0.05);              // J 面对下注必弃
    CHECK_NEAR(p0_k_call, 1.0, 0.05);              // K 必跟
    CHECK_NEAR(p1_j_call, 0.0, 0.05);
    CHECK_NEAR(p1_k_call, 1.0, 0.05);
    CHECK_NEAR(p1_q_call, 1.0 / 3.0, 0.09);
}

void testExploitabilityBaseline() {
    section("可利用度基准：均匀策略 vs 均衡策略");

    // 均匀策略（所有信息集等概率）是一个明确的弱策略
    texas::Strategy uniform;
    texas::KuhnGame::State probe;
    for (int p = 0; p < 2; ++p) {
        for (int card = 0; card < 3; ++card) {
            const char* histsP0[] = {"", "pb"};
            const char* histsP1[] = {"p", "b"};
            const char** hists = (p == 0) ? histsP0 : histsP1;
            for (int h = 0; h < 2; ++h) {
                probe.priv[0] = card;
                probe.priv[1] = card;
                probe.hist = hists[h];
                if (texas::KuhnGame::playerToAct(probe) != p) continue;
                uniform[texas::KuhnGame::infosetId(probe)] = {0.5, 0.5};
            }
        }
    }
    const double uniformExpl = texas::exploitability<texas::KuhnGame>(uniform);
    std::printf("  均匀策略 exploitability = %.4f\n", uniformExpl);
    CHECK(uniformExpl > 0.05);  // 明显可被剥削

    texas::CFRSolver<texas::KuhnGame> solver;
    solver.iterate(20000);
    const double trainedExpl = texas::exploitability<texas::KuhnGame>(solver.averageStrategy());
    std::printf("  训练后 exploitability = %.4f（下降 %.0f 倍）\n", trainedExpl,
                uniformExpl / std::max(trainedExpl, 1e-9));
    CHECK(trainedExpl < uniformExpl * 0.2);  // 至少好 5 倍
}

void testLeduc() {
    section("Leduc Hold'em / CFR+ 收敛");

    texas::CFRSolver<texas::LeducGame> solver;
    solver.iterate(15000);
    const texas::Strategy sigma = solver.averageStrategy();
    const double expl = texas::exploitability<texas::LeducGame>(sigma);
    const double value = solver.expectedValue(sigma);

    std::printf("  iter=%d  infosets=%zu  exploitability=%.6f  value=%+.6f (标准值 -0.0856)\n",
                solver.iterations(), solver.numInfosets(), expl, value);

    // 288 = 按 rank 合并同 rank 双牌后的精确信息集数量
    CHECK(solver.numInfosets() == 288);
    // 博弈值必须收敛到 Leduc 的已知解 -0.0856
    CHECK_NEAR(value, -0.0856, 0.005);
    CHECK(expl < 0.035);

    // 再练一轮，可利用度必须继续下降
    solver.iterate(5000);
    const texas::Strategy sigma2 = solver.averageStrategy();
    const double expl2 = texas::exploitability<texas::LeducGame>(sigma2);
    const double value2 = solver.expectedValue(sigma2);
    std::printf("  iter=%d  exploitability=%.6f  value=%+.6f\n", solver.iterations(), expl2,
                value2);
    CHECK(expl2 < expl);
    CHECK_NEAR(value2, -0.0856, 0.004);
}

void testVanillaVsPlus() {
    section("vanilla CFR 与 CFR+ 对比");

    texas::CFRSolver<texas::KuhnGame> plusSolver(texas::CFRSolver<texas::KuhnGame>::Options(true, true));
    texas::CFRSolver<texas::KuhnGame> vanilla(texas::CFRSolver<texas::KuhnGame>::Options(false, false));
    plusSolver.iterate(2000);
    vanilla.iterate(2000);

    const double ePlus = texas::exploitability<texas::KuhnGame>(plusSolver.averageStrategy());
    const double eVani = texas::exploitability<texas::KuhnGame>(vanilla.averageStrategy());
    std::printf("  2000 轮后: CFR+ = %.6f, vanilla = %.6f\n", ePlus, eVani);

    // 两者都应可用（收敛到均衡）
    CHECK(ePlus < 0.02);
    CHECK(eVani < 0.03);
    // CFR+ 通常更快
    CHECK(ePlus <= eVani * 2.0);
}

// ---------------- 无限注德州 ----------------

void testHoldemRules() {
    section("无限注德州 / 盲注与合法动作");

    texas::HoldemGame::State s = texas::HoldemGame::init(2);
    CHECK(s.contrib[0] == texas::HoldemGame::kSmallBlind);
    CHECK(s.contrib[1] == texas::HoldemGame::kBigBlind);
    CHECK(s.player == 0);  // 单挑由按钮/小盲先行动
    CHECK(s.street == 0);
    CHECK(s.boardCount == 0);

    texas::Move mv[texas::HoldemGame::kMaxActions];
    int n = texas::HoldemGame::actions(s, mv, texas::HoldemGame::kMaxActions);
    CHECK(n == 5);  // 弃牌 + 跟注 + 两个下注尺度 + all-in
    CHECK(mv[0].kind == texas::Move::Fold);
    CHECK(mv[1].kind == texas::Move::CheckCall);
    CHECK(mv[2].kind == texas::Move::Raise);
    CHECK(mv[3].kind == texas::Move::Raise);
    CHECK(mv[3].amount > mv[2].amount);
    CHECK(mv[4].kind == texas::Move::Raise);
    CHECK(mv[4].amount == texas::HoldemGame::kBuyIn);
    CHECK(mv[2].amount <= texas::HoldemGame::kBuyIn);

    // 小盲补齐，大盲仍有选择权（可以过牌或加注）
    const auto s1 = texas::HoldemGame::next(s, texas::Move{texas::Move::CheckCall, 0});
    CHECK(s1.contrib[0] == 2);
    CHECK(s1.player == 1);
    CHECK(s1.street == 0);

    // 大盲过牌 => 翻牌，等发牌
    const auto s2 = texas::HoldemGame::next(s1, texas::Move{texas::Move::CheckCall, 0});
    CHECK(s2.street == 1);
    CHECK(s2.pendingDeal == 3);
    CHECK(texas::HoldemGame::needsChance(s2));
    CHECK(!texas::HoldemGame::isTerminal(s2));

    // 发翻牌后由大盲（非按钮）先行动
    std::mt19937_64 rng(1);
    texas::HoldemGame::State s3 = s2;
    texas::HoldemGame::deal(s3, rng);
    CHECK(s3.boardCount == 3);
    CHECK(!texas::HoldemGame::needsChance(s3));
    CHECK(s3.player == 1);

    // 无人下注时不允许弃牌（弃牌只在面对下注时合法）
    texas::Move f[6];
    const int fn = texas::HoldemGame::actions(s3, f, 6);
    CHECK(fn == 4);  // 过牌 + 两个下注尺度 + all-in
    CHECK(f[0].kind == texas::Move::CheckCall);
    for (int i = 0; i < fn; ++i) CHECK(f[i].kind != texas::Move::Fold);

    std::printf("  OK 盲注、行动顺序、翻牌发牌、无人下注不可弃牌\n");
}

void testHoldemActionSpace() {
    section("无限注德州 / 动作集参数化");

    const int NA = texas::HoldemGame::kMaxActions;
    const int BUY = texas::HoldemGame::kBuyIn;

    // 默认动作集必须与早期写死的尺寸**逐位一致**，否则所有已训模型和基准快照都失效。
    // 翻牌前 maxRC=2，2.5x -> 5，4x -> 8；早期写的是 maxRC + (maxRC*3)/2 和 maxRC*4。
    {
        const texas::HoldemGame::State s = texas::HoldemGame::init(2);
        texas::Move mv[NA];
        texas::ActionSpace def;
        const int n = texas::HoldemGame::actions(s, mv, NA, def);
        CHECK(n == 5);
        CHECK(mv[2].amount == 5);
        CHECK(mv[3].amount == 8);
        CHECK(mv[4].amount == BUY);
        // 显式指定 2.5x/4x 应得到完全相同的结果
        texas::ActionSpace same;
        same.preflopMul1000[0] = 2500;
        same.preflopMul1000[1] = 4000;
        CHECK(same.preflopCount == 2);
        texas::Move mv2[NA];
        const int n2 = texas::HoldemGame::actions(s, mv2, NA, same);
        CHECK(n2 == n);
        for (int i = 0; i < n && i < n2; ++i) {
            CHECK(mv2[i].kind == mv[i].kind);
            CHECK(mv2[i].amount == mv[i].amount);
        }
        // 单参重载等于默认动作集
        texas::Move mv3[NA];
        const int n3 = texas::HoldemGame::actions(s, mv3, NA);
        CHECK(n3 == n);
        for (int i = 0; i < n && i < n3; ++i) CHECK(mv3[i].amount == mv[i].amount);
    }

    // 改尺寸档数：1 档 / 3 档
    {
        const texas::HoldemGame::State s = texas::HoldemGame::init(2);
        texas::Move mv[NA];
        texas::ActionSpace one;
        one.preflopCount = 1;
        CHECK(texas::HoldemGame::actions(s, mv, NA, one) == 4);  // 弃 + 跟 + 1 档 + allin
        CHECK(mv[2].amount == 5);
        CHECK(mv[3].amount == BUY);

        texas::ActionSpace three;
        three.preflopCount = 3;
        three.preflopMul1000[2] = 6000;
        const int n = texas::HoldemGame::actions(s, mv, NA, three);
        CHECK(n == 6);  // 弃 + 跟 + 3 档 + allin
        CHECK(mv[2].amount == 5);
        CHECK(mv[3].amount == 8);
        CHECK(mv[4].amount == 12);
        CHECK(mv[5].amount == BUY);

        // 5 档才会填满 kMaxActions（弃 + 跟 + 5 档 + allin）
        texas::ActionSpace five;
        five.preflopCount = 5;
        five.preflopMul1000[2] = 6000;
        five.preflopMul1000[3] = 9000;
        five.preflopMul1000[4] = 12000;
        const int n5 = texas::HoldemGame::actions(s, mv, NA, five);
        CHECK(n5 == NA);
        CHECK(mv[NA - 1].amount == BUY);  // all-in 在最后一位
    }

    // 去掉独立的 all-in 动作
    {
        const texas::HoldemGame::State s = texas::HoldemGame::init(2);
        texas::Move mv[NA];
        texas::ActionSpace noJam;
        noJam.includeAllIn = false;
        const int n = texas::HoldemGame::actions(s, mv, NA, noJam);
        CHECK(n == 4);
        for (int i = 0; i < n; ++i) CHECK(mv[i].amount < BUY);
    }

    // 翻牌后的底池比例：一池就是底池大小，半池是一半
    {
        texas::HoldemGame::State s = texas::HoldemGame::init(2);
        s = texas::HoldemGame::next(s, texas::Move{texas::Move::Raise, 8});       // 小盲加到 8
        s = texas::HoldemGame::next(s, texas::Move{texas::Move::CheckCall, 0});   // 大盲跟
        texas::Card board[5] = {C("Ks"), C("8c"), C("3d"), 0, 0};
        texas::HoldemGame::reveal(s, board);
        CHECK(s.street == 1);
        CHECK(texas::HoldemGame::pot(s) == 16);
        CHECK(s.player == 1);  // 翻牌后大盲先行动

        texas::Move mv[NA];
        const int n = texas::HoldemGame::actions(s, mv, NA);
        CHECK(n == 4);  // 过牌 + 半池 + 一池 + allin
        const int base = s.contrib[1];
        CHECK(mv[1].amount == base + 8);   // 半池 = 16/2
        CHECK(mv[2].amount == base + 16);  // 一池 = 16
        CHECK(mv[3].amount == BUY);

        // 换成 {1/3 池, 3/4 池}：尺寸必须跟着变
        texas::ActionSpace alt;
        alt.postflopFrac1000[0] = 333;
        alt.postflopFrac1000[1] = 750;
        texas::Move mv2[NA];
        const int n2 = texas::HoldemGame::actions(s, mv2, NA, alt);
        CHECK(n2 == 4);
        CHECK(mv2[1].amount == base + 5);   // 16*333/1000 = 5
        CHECK(mv2[2].amount == base + 12);  // 16*750/1000 = 12
        CHECK(mv2[1].amount < mv[1].amount);
        CHECK(mv2[2].amount < mv[2].amount);
    }

    // 动作集要能跟着模型存盘：读盘后这个求解器代表的就是存盘时那个模型
    {
        texas::ActionSpace space;
        space.preflopCount = 1;
        space.postflopCount = 1;
        space.includeAllIn = false;
        const char* path = "texas_cfr_test_space.bin";
        texas::HoldemSolver src(20, 999ULL, texas::CfrVariant::DCFR, true);
        src.setActionSpace(space);
        src.iterate(3000);
        CHECK(src.save(path));

        texas::HoldemSolver dst(20, 1ULL, texas::CfrVariant::DCFR, true);
        CHECK(dst.load(path));
        CHECK(dst.abstraction().actionSpace.preflopCount == 1);
        CHECK(dst.abstraction().actionSpace.postflopCount == 1);
        CHECK(!dst.abstraction().actionSpace.includeAllIn);
        // 键必须还能对上：读回来的一手牌策略与存盘前逐位相同
        texas::HoldemGame::State st = texas::HoldemGame::init(2);
        st.hole[0][0] = C("Ac");
        st.hole[0][1] = C("Ad");
        const std::vector<double> a = src.averageStrategy(st);
        const std::vector<double> b = dst.averageStrategy(st);
        CHECK(a.size() == b.size());
        if (a.size() == b.size()) {
            for (std::size_t i = 0; i < a.size(); ++i) CHECK_NEAR(a[i], b[i], 1e-12);
        }
        std::remove(path);
    }

    std::printf("  OK 动作集参数化（默认逐位不变 / 档数 / 去 all-in / 底池比例 / 随模型存盘）\n");
}

// 名字 ↔ 动作集必须是同一个双射。CLI 打印的那一行按内容反查名字，
// 若表和名字能各自漂移，就会出现"动作集 three：翻牌前 2 档"这种自相矛盾
// 却不报错的输出——这正是改 --sizes 时踩到的东西。
void testActionSpaceNaming() {
    section("无限注德州 / 动作集命名的往返");

    const auto& table = texas::actionPresets();
    CHECK(table.size() >= 6);

    // 每个名字都能解析，且解析回来的动作集反查仍是同一个名字
    std::set<std::string> names;
    for (const texas::ActionPreset& p : table) {
        texas::ActionSpace sp;
        CHECK(texas::actionPreset(p.name, sp));
        CHECK(texas::sameActionSpace(sp, p.space));
        CHECK(std::string(texas::actionSpaceName(sp)) == p.name);
        names.insert(p.name);
    }
    CHECK(names.size() == table.size());  // 没有重名

    // 认不出来的名字不能悄悄变成某个预设
    texas::ActionSpace fallback;
    fallback.preflopCount = 3;
    CHECK(!texas::actionPreset("nope", fallback));
    CHECK(texas::sameActionSpace(fallback, texas::ActionSpace{}));

    // 手工改一位就不再属于任何预设
    texas::ActionSpace odd;
    odd.preflopMul1000[0] = 2750;
    CHECK(std::string(texas::actionSpaceName(odd)) == "custom");

    // 只看 actions() 真读的字段：把尾部未使用的槽位改成噪声，名字不能变
    texas::ActionSpace tail = texas::ActionSpace{};
    tail.preflopMul1000[2] = 7777;   // preflopCount=2，这个槽不被读
    tail.postflopFrac1000[2] = 8888;
    CHECK(std::string(texas::actionSpaceName(tail)) == "default");

    std::printf("  OK 动作集命名往返（%zu 个预设 / 未知名字 / 未读槽位不影响命名）\n",
                table.size());
}

// 存盘格式 v4 到 v5 的兼容。v4 把 kMaxActions / kMaxSizes 这两个编译期常量写死进了
// 文件布局（每条记录 6 个动作、每个动作数组 3 个数），所以动作数上限一提就必须留一条
// 按**当时**常量解释旧文件的路径——否则旧文件不会被拒绝，而是被错位读成别的数据。
void testHoldemSaveV4Compat() {
    section("无限注德州 / 旧存盘格式兼容");

    const char* v4path = "texas_cfr_test_v4.bin";
    const char* v5path = "texas_cfr_test_v5.bin";
    {
        std::FILE* f = std::fopen(v4path, "wb");
        CHECK(f != nullptr);
        if (f == nullptr) return;
        const std::uint32_t magic = 0x54434652u;
        const std::uint32_t ver = 4;
        const std::uint64_t cnt = 1;
        const std::uint64_t iters = 1000;
        const std::int32_t buckets = 20;
        const std::int32_t sized = 1;
        const std::int32_t allin = 1;
        // v4 的动作数组是 3 个数（那时 kMaxSizes = 3）
        const std::int32_t pre[3] = {2500, 4000, 2500};
        const std::int32_t preCount = 2;
        const std::int32_t post[3] = {500, 1000, 500};
        const std::int32_t postCount = 2;
        std::fwrite(&magic, sizeof(magic), 1, f);
        std::fwrite(&ver, sizeof(ver), 1, f);
        std::fwrite(&cnt, sizeof(cnt), 1, f);
        std::fwrite(&iters, sizeof(iters), 1, f);
        std::fwrite(&buckets, sizeof(buckets), 1, f);
        std::fwrite(&sized, sizeof(sized), 1, f);
        std::fwrite(pre, sizeof(std::int32_t), 3, f);
        std::fwrite(&preCount, sizeof(preCount), 1, f);
        std::fwrite(post, sizeof(std::int32_t), 3, f);
        std::fwrite(&postCount, sizeof(postCount), 1, f);
        std::fwrite(&allin, sizeof(allin), 1, f);
        // 一条记录：key + n + 6 个遗憾 + 6 个平均策略（n 只用到前 5 个）
        const std::uint64_t key = 0x0123456789abcdefULL;
        const std::uint32_t n = 5;
        double regret[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 999.0};
        double avg[6] = {6.0, 7.0, 8.0, 9.0, 10.0, 999.0};
        std::fwrite(&key, sizeof(key), 1, f);
        std::fwrite(&n, sizeof(n), 1, f);
        std::fwrite(regret, sizeof(double), 6, f);
        std::fwrite(avg, sizeof(double), 6, f);
        std::fclose(f);
    }

    texas::HoldemSolver s(20, 1ULL, texas::CfrVariant::DCFR, true);
    CHECK(s.load(v4path));
    CHECK(s.numInfosets() == 1);
    // 动作集要从 3 个数的数组里正确恢复到 4 槽结构
    CHECK(s.abstraction().actionSpace.preflopCount == 2);
    CHECK(s.abstraction().actionSpace.postflopCount == 2);
    CHECK(s.abstraction().actionSpace.preflopMul1000[1] == 4000);
    CHECK(s.abstraction().actionSpace.postflopFrac1000[1] == 1000);

    // 存回去必须是新版本号：读旧写新，格式不会一直背着历史
    CHECK(s.save(v5path));
    {
        std::FILE* f = std::fopen(v5path, "rb");
        CHECK(f != nullptr);
        if (f != nullptr) {
            std::uint32_t magic = 0;
            std::uint32_t ver = 0;
            CHECK(std::fread(&magic, sizeof(magic), 1, f) == 1);
            CHECK(std::fread(&ver, sizeof(ver), 1, f) == 1);
            CHECK(magic == 0x54434652u);
            CHECK(ver == 6);
            std::fclose(f);
        }
    }

    // 反过来：v5 文件必须能被本版本读回，且动作集一致
    {
        texas::HoldemSolver t(20, 2ULL, texas::CfrVariant::DCFR, true);
        CHECK(t.load(v5path));
        CHECK(t.numInfosets() == 1);
        CHECK(texas::sameActionSpace(t.abstraction().actionSpace,
                                     s.abstraction().actionSpace));
    }

    std::remove(v4path);
    std::remove(v5path);
    std::printf("  OK 旧存盘兼容（v4 按当时的常量解释 / 读旧写新 / v5 往返）\n");
}

void testHoldemShowdown() {
    section("无限注德州 / 摊牌与弃牌");

    // 两人都过牌到河牌，AA 对 KK
    texas::HoldemGame::State s = texas::HoldemGame::init(2);
    s.hole[0][0] = C("Ac");
    s.hole[0][1] = C("Ad");
    s.hole[1][0] = C("Kc");
    s.hole[1][1] = C("Kd");
    s.board[0] = C("2c");
    s.board[1] = C("7d");
    s.board[2] = C("9s");
    s.board[3] = C("Jc");
    s.board[4] = C("4h");
    s.boardCount = 5;
    s.street = 3;
    s.finished = true;
    s.contrib[0] = 10;
    s.contrib[1] = 10;

    CHECK(texas::HoldemGame::isTerminal(s));
    const double u0 = texas::HoldemGame::utilityFrom(s, 0);
    const double u1 = texas::HoldemGame::utilityFrom(s, 1);
    std::printf("  AA vs KK 摊牌: P0=%+.1f P1=%+.1f\n", u0, u1);
    CHECK_NEAR(u0, 10.0, 0.01);   // 赢走对方投入
    CHECK_NEAR(u0 + u1, 0.0, 1e-9);

    // 换成 P0 拿 5h3d（纯高牌，且不撞公共牌），应该输
    s.hole[0][0] = C("5h");
    s.hole[0][1] = C("3d");
    CHECK_NEAR(texas::HoldemGame::utilityFrom(s, 0), -10.0, 0.01);

    // 一方弃牌：弃牌方输掉自己已投入的部分
    texas::HoldemGame::State f = texas::HoldemGame::init(2);
    f.contrib[0] = 30;
    f.contrib[1] = 10;
    f.folded[0] = true;
    CHECK_NEAR(texas::HoldemGame::utilityFrom(f, 0), -30.0, 0.01);
    CHECK_NEAR(texas::HoldemGame::utilityFrom(f, 1), 30.0, 0.01);

    std::printf("  OK 摊牌比牌、弃牌结算、零和\n");
}

void testHoldemSidePot() {
    section("无限注德州 / 边池分配");

    // 三人：P0 只投 50（短筹码），P1/P2 各投 100
    // 主池 150（三人争）、边池 100（P1/P2 争）
    texas::HoldemGame::State s;
    s.numPlayers = 3;
    s.board[0] = C("2c");
    s.board[1] = C("7d");
    s.board[2] = C("9s");
    s.board[3] = C("Jc");
    s.board[4] = C("4h");
    s.boardCount = 5;
    s.street = 3;
    s.finished = true;
    s.contrib[0] = 50;
    s.contrib[1] = 100;
    s.contrib[2] = 100;

    // P0 最强（AA）但只有 50；P1 次之（KK）；P2 最弱（高牌）
    s.hole[0][0] = C("Ac");
    s.hole[0][1] = C("Ad");
    s.hole[1][0] = C("Kc");
    s.hole[1][1] = C("Kd");
    s.hole[2][0] = C("5c");
    s.hole[2][1] = C("6d");

    double pay[6];
    texas::HoldemGame::payouts(s, pay);
    std::printf("  边池: P0=%+.1f P1=%+.1f P2=%+.1f\n", pay[0], pay[1], pay[2]);

    CHECK_NEAR(pay[0], 100.0, 0.01);  // 赢主池 150，净 +100
    CHECK_NEAR(pay[1], 0.0, 0.01);    // 赢边池 100 又付出 100
    CHECK_NEAR(pay[2], -100.0, 0.01); // 全输
    CHECK_NEAR(pay[0] + pay[1] + pay[2], 0.0, 1e-9);

    // 让 P1 最强：主池和边池都归 P1
    s.hole[1][0] = C("Ac");
    s.hole[1][1] = C("Ah");
    s.hole[0][0] = C("Kc");
    s.hole[0][1] = C("Kd");
    texas::HoldemGame::payouts(s, pay);
    std::printf("  P1 最强: P0=%+.1f P1=%+.1f P2=%+.1f\n", pay[0], pay[1], pay[2]);
    CHECK_NEAR(pay[1], 150.0, 0.01);  // 赢下全部 250，净 +150
    CHECK_NEAR(pay[0], -50.0, 0.01);
    CHECK_NEAR(pay[2], -100.0, 0.01);
    CHECK_NEAR(pay[0] + pay[1] + pay[2], 0.0, 1e-9);

    std::printf("  OK 主池/边池分层结算，三人零和\n");
}

void testHoldemScoreCache() {
    section("无限注德州 / 摊牌牌力缓存");

    // 6 张牌路径（翻牌 + 两张底牌）改用 evaluate6 快路径，必须与
    // "枚举 6 选 5 的所有子集取最大"逐位一致
    std::mt19937_64 rng(20241025ULL);
    int mismatch6 = 0;
    for (int t = 0; t < 20000; ++t) {
        texas::Card deck[texas::kNumCards];
        for (int i = 0; i < texas::kNumCards; ++i) deck[i] = i;
        for (int i = 0; i < 6; ++i) {
            const int j = i + static_cast<int>(rng() % static_cast<std::uint64_t>(texas::kNumCards - i));
            std::swap(deck[i], deck[j]);
        }
        texas::HoldemGame::State s = texas::HoldemGame::init(2);
        s.hole[0][0] = deck[0];
        s.hole[0][1] = deck[1];
        for (int i = 0; i < 4; ++i) s.board[i] = deck[2 + i];
        s.boardCount = 4;  // 2 + 4 = 6 张

        const int fast = texas::HoldemGame::handScore(s, 0);
        int brute = -1;
        texas::Card all6[6] = {deck[0], deck[1], deck[2], deck[3], deck[4], deck[5]};
        for (int skip = 0; skip < 6; ++skip) {
            texas::Card h5[5];
            int k = 0;
            for (int i = 0; i < 6; ++i) {
                if (i != skip) h5[k++] = all6[i];
            }
            brute = std::max(brute, texas::evaluate5(h5));
        }
        if (fast != brute) ++mismatch6;
    }
    std::printf("  6 张牌快路径 vs 枚举子集：20000 组，不一致 %d 处\n", mismatch6);
    CHECK(mismatch6 == 0);

    // 结算缓存必须完全等价。这里刻意用"引擎自己走出的合法对局"而不是手搓随机状态：
    // 手搓的投入额可能让弃牌者的投入高于存活者，那种局面在真实对局里不可达，
    // 结算本身也不守恒，拿它测缓存只会测出噪声。
    // 每局用一个唯一 stamp —— 缓存的约定就是"同一 stamp 内底牌和公共牌固定"，
    // 违反这个约定得到错误结果不是 bug，所以测试必须遵守它。
    texas::HoldemGame::ScoreCache cache;
    int stamp = 0;
    int mismatchPay = 0;
    int badSum = 0;
    int showdowns = 0;
    std::mt19937_64 gameRng(8675309ULL);
    for (int g = 0; g < 20000; ++g) {
        ++stamp;  // 每局唯一
        const int np = 2 + static_cast<int>(gameRng() % 2);
        texas::HoldemGame::State s = texas::HoldemGame::init(np);
        texas::Card deck[texas::kNumCards];
        const int dn = texas::HoldemGame::buildDeck(s, deck);
        int k = 0;
        for (int p = 0; p < np; ++p) {
            for (int j = 0; j < 2; ++j) {
                const int i = k + static_cast<int>(gameRng() % static_cast<std::uint64_t>(dn - k));
                std::swap(deck[k], deck[i]);
                s.hole[p][j] = deck[k++];
            }
        }
        // 随机合法走子，偏向跟注/过牌，让牌局有机会打到摊牌
        for (int step = 0; step < 60 && !texas::HoldemGame::isTerminal(s); ++step) {
            if (texas::HoldemGame::needsChance(s)) {
                texas::HoldemGame::deal(s, gameRng);
                continue;
            }
            texas::Move mv[6];
            const int n = texas::HoldemGame::actions(s, mv, 6);
            if (n <= 0) break;
            int pick = static_cast<int>(gameRng() % static_cast<std::uint64_t>(n));
            for (int a = 0; a < n; ++a) {
                if (mv[a].kind == texas::Move::CheckCall && gameRng() % 10 < 6) {
                    pick = a;
                    break;
                }
            }
            s = texas::HoldemGame::next(s, mv[pick]);
        }
        if (!texas::HoldemGame::isTerminal(s)) continue;

        double noCache[6] = {};
        double cold[6] = {};
        double hot[6] = {};
        texas::HoldemGame::payouts(s, noCache);
        texas::HoldemGame::payouts(s, cold, &cache, stamp);  // 首次：填缓存
        texas::HoldemGame::payouts(s, hot, &cache, stamp);   // 再次：命中缓存
        double sum = 0.0;
        for (int p = 0; p < np; ++p) {
            if (noCache[p] != cold[p] || noCache[p] != hot[p]) ++mismatchPay;
            sum += noCache[p];
        }
        if (std::fabs(sum) > 1e-9) ++badSum;
        if (s.boardCount == 5 && texas::HoldemGame::numActive(s) > 1) ++showdowns;
    }
    std::printf("  合法对局结算：20000 局（含摊牌 %d 局），缓存不一致 %d 处，非零和 %d 处\n",
                showdowns, mismatchPay, badSum);
    CHECK(mismatchPay == 0);
    CHECK(badSum == 0);
    CHECK(showdowns > 2000);  // 保证确实覆盖到了摊牌路径

    std::printf("  OK 摊牌牌力缓存（6 张快路径等价 / 结算零和 / 缓存不改变结果）\n");
}

void testHoldemAllIn() {
    section("无限注德州 / all-in 一次发完");

    texas::HoldemGame::State s = texas::HoldemGame::init(2);
    s.hole[0][0] = C("Ac");
    s.hole[0][1] = C("Ad");
    s.hole[1][0] = C("Kc");
    s.hole[1][1] = C("Kd");

    // 小盲直接 all-in
    texas::Move mv[6];
    const int n = texas::HoldemGame::actions(s, mv, 6);
    int allIn = -1;
    for (int i = 0; i < n; ++i) {
        if (mv[i].kind == texas::Move::Raise && mv[i].amount == texas::HoldemGame::kBuyIn) {
            allIn = i;
        }
    }
    CHECK(allIn >= 0);

    const auto s1 = texas::HoldemGame::next(s, mv[allIn]);
    CHECK(s1.contrib[0] == texas::HoldemGame::kBuyIn);
    CHECK(s1.allIn[0]);

    // 大盲跟注 all-in
    const auto s2 = texas::HoldemGame::next(s1, texas::Move{texas::Move::CheckCall, 0});
    CHECK(s2.contrib[1] == texas::HoldemGame::kBuyIn);
    CHECK(s2.street == 3);
    CHECK(s2.pendingDeal == 5);  // 一次把翻牌/转牌/河牌全发完
    CHECK(texas::HoldemGame::needsChance(s2));

    std::mt19937_64 rng(3);
    texas::HoldemGame::State s3 = s2;
    texas::HoldemGame::deal(s3, rng);
    CHECK(s3.boardCount == 5);
    CHECK(texas::HoldemGame::isTerminal(s3));

    double pay[6];
    texas::HoldemGame::payouts(s3, pay);
    CHECK_NEAR(pay[0] + pay[1], 0.0, 1e-9);
    CHECK_NEAR(std::fabs(pay[0]), static_cast<double>(texas::HoldemGame::kBuyIn), 0.01);

    std::printf("  OK all-in 后一次发完 5 张公共牌并摊牌\n");
}

void testHoldemAbstraction() {
    section("无限注德州 / 牌力抽象");

    // ---- 评测器：快路径必须与暴力实现逐位等价 ----
    // 5 张穷举 C(52,5) 全量对照
    {
        long long bad = 0;
        texas::Card c[5];
        for (int a = 0; a < 52; ++a)
        for (int b = a + 1; b < 52; ++b)
        for (int d = b + 1; d < 52; ++d)
        for (int e = d + 1; e < 52; ++e)
        for (int f = e + 1; f < 52; ++f) {
            c[0] = a; c[1] = b; c[2] = d; c[3] = e; c[4] = f;
            if (texas::evaluateBest(c, 5) != texas::evaluate5(c)) ++bad;
        }
        std::printf("  C(52,5) 穷举 %d 组，快路径与 evaluate5 不一致 %lld 处\n", 2598960, bad);
        CHECK(bad == 0);
    }
    // 7 张随机对照 21 组合暴力
    {
        std::mt19937_64 rng(777);
        long long bad = 0;
        for (int it = 0; it < 300000; ++it) {
            texas::Card d[52];
            for (int i = 0; i < 52; ++i) d[i] = i;
            for (int i = 0; i < 7; ++i) std::swap(d[i], d[i + static_cast<int>(rng() % (52 - i))]);
            texas::Card c[7] = {d[0], d[1], d[2], d[3], d[4], d[5], d[6]};
            if (texas::evaluateBest(c, 7) != texas::evaluate7Brute(c)) ++bad;
        }
        std::printf("  7 张随机 30 万组，快路径与暴力不一致 %lld 处\n", bad);
        CHECK(bad == 0);
    }

    // ---- 翻牌前：169 类必须各自独立，不能被桶合并 ----
    const texas::StrengthModel& model = texas::StrengthModel::instance();
    const int clsAA = texas::StrengthModel::preflopClass(C("Ac"), C("Ad"));
    const int clsKK = texas::StrengthModel::preflopClass(C("Kc"), C("Kd"));
    const int clsAKs = texas::StrengthModel::preflopClass(C("Ah"), C("Kh"));
    const int clsAKo = texas::StrengthModel::preflopClass(C("Ah"), C("Kd"));
    const int cls72 = texas::StrengthModel::preflopClass(C("7h"), C("2d"));
    std::printf("  类号: AA=%d KK=%d AKs=%d AKo=%d 72o=%d\n", clsAA, clsKK, clsAKs, clsAKo, cls72);
    CHECK(clsAA != clsKK);    // 老版本这两手会被压进同一个桶
    CHECK(clsAKs != clsAKo);  // 同花/非同花不能混
    CHECK(clsAA != cls72);
    // 花色置换不变
    CHECK(texas::StrengthModel::preflopClass(C("Ac"), C("Ad")) ==
          texas::StrengthModel::preflopClass(C("Ah"), C("As")));
    CHECK(texas::StrengthModel::preflopClass(C("Ah"), C("Kh")) ==
          texas::StrengthModel::preflopClass(C("As"), C("Ks")));

    // 胜率排序：AA > KK > QQ，同花 > 非同花
    const double eqAA = model.preflopEquity(C("Ac"), C("Ad"));
    const double eqKK = model.preflopEquity(C("Kc"), C("Kd"));
    const double eqQQ = model.preflopEquity(C("Qc"), C("Qd"));
    const double eqAKs = model.preflopEquity(C("Ah"), C("Kh"));
    const double eqAKo = model.preflopEquity(C("Ah"), C("Kd"));
    const double eq72 = model.preflopEquity(C("7h"), C("2d"));
    std::printf("  胜率: AA=%.4f KK=%.4f QQ=%.4f AKs=%.4f AKo=%.4f 72o=%.4f\n", eqAA, eqKK, eqQQ,
                eqAKs, eqAKo, eq72);
    CHECK(eqAA > eqKK);
    CHECK(eqKK > eqQQ);
    CHECK(eqAKs > eqAKo);
    CHECK(eqAKs > eq72);
    CHECK(eqAA > 0.83 && eqAA < 0.87);  // AA 对单随机对手约 85%

    // ---- 翻牌后：潜力修正必须把听牌从"当前牌力排名"里救出来 ----
    // 同一块翻牌面，AhTh 和 AdTd 的"当前最佳五张"完全相同（都是 A 高牌），
    // 所以百分位严格相等；但 AhTh 有同花听牌，修正后必须严格更强。
    {
        const texas::Card board[3] = {C("Kh"), C("7h"), C("2s")};
        const texas::Card draw[2] = {C("Ah"), C("Th")};
        const texas::Card dry[2] = {C("Ad"), C("Td")};
        const double pDraw = model.postflopPercentile(draw, board, 3);
        const double pDry = model.postflopPercentile(dry, board, 3);
        const double sDraw = model.postflopStrength(draw, board, 3);
        const double sDry = model.postflopStrength(dry, board, 3);
        const texas::DrawOuts outs = texas::StrengthModel::computeOuts(draw, board, 3);
        std::printf("  K72 面: 同花听牌 pct=%.3f str=%.3f (outs 花%d 顺%d 共%d)\n", pDraw, sDraw,
                    outs.flush, outs.straight, outs.total);
        std::printf("          无听牌   pct=%.3f str=%.3f\n", pDry, sDry);
        CHECK(std::fabs(pDraw - pDry) < 1e-12);  // 当前牌力确实一模一样
        CHECK(outs.flush == 9);                  // 同花听牌 9 个出路
        CHECK(sDraw > sDry + 0.10);              // 潜力修正必须显著拉开
        CHECK(sDraw > pDraw);                    // 听牌只会上调，不会下调
    }
    // 两头顺听牌 8 个出路，卡顺 4 个
    {
        const texas::Card board[3] = {C("9c"), C("8d"), C("2s")};
        const texas::Card oesd[2] = {C("Th"), C("7c")};
        const texas::Card gut[2] = {C("Th"), C("6c")};
        const texas::DrawOuts a = texas::StrengthModel::computeOuts(oesd, board, 3);
        const texas::DrawOuts b = texas::StrengthModel::computeOuts(gut, board, 3);
        std::printf("  98 2 面: T7o 顺子outs=%d | T6o 顺子outs=%d\n", a.straight, b.straight);
        CHECK(a.straight == 8);
        CHECK(b.straight == 4);
    }
    // 河牌没有后续牌，修正必须退化成百分位本身
    {
        const texas::Card board[5] = {C("Kh"), C("7h"), C("2s"), C("9d"), C("3c")};
        const texas::Card hole[2] = {C("Ah"), C("Th")};
        CHECK_NEAR(model.postflopStrength(hole, board, 5),
                   model.postflopPercentile(hole, board, 5), 1e-12);
    }
    // 百分位必须随牌力单调
    {
        const texas::Card board[5] = {C("Kh"), C("7h"), C("2s"), C("9d"), C("3c")};
        const texas::Card nuts[2] = {C("Ks"), C("Kd")};  // 三条 K
        const texas::Card weak[2] = {C("4h"), C("5d")};  // 高牌
        CHECK(model.postflopPercentile(nuts, board, 5) >
              model.postflopPercentile(weak, board, 5));
    }

    // ---- 分桶：九个牌型必须落在互不重叠的桶区间里 ----
    // 等质量分桶会把两对以上（只占 8.2%）全塞进最后一个桶，模型分不清超对和
    // 暗三条。这里逐个牌型构造一手牌，验证桶号严格单调递增。
    {
        struct Case {
            const char* name;
            texas::HandCategory cat;
            const char* h1;
            const char* h2;
            const char* b1;
            const char* b2;
            const char* b3;
        };
        const Case cases[] = {
            {"高牌", texas::HandCategory::HighCard, "9h", "5h", "Ks", "7c", "2d"},
            {"一对", texas::HandCategory::OnePair, "Kd", "5h", "Ks", "7c", "2d"},
            {"两对", texas::HandCategory::TwoPair, "Kd", "7h", "Ks", "7c", "2d"},
            {"三条", texas::HandCategory::ThreeOfAKind, "7d", "7s", "Ks", "7c", "2d"},
            {"顺子", texas::HandCategory::Straight, "7h", "6h", "Ts", "9c", "8d"},
            {"同花", texas::HandCategory::Flush, "Ah", "5h", "Kh", "7h", "2h"},
            {"葫芦", texas::HandCategory::FullHouse, "7h", "7s", "Ks", "Kc", "7d"},
            {"四条", texas::HandCategory::FourOfAKind, "Kh", "Kd", "Ks", "Kc", "7d"},
            {"同花顺", texas::HandCategory::StraightFlush, "Th", "6h", "9h", "8h", "7h"},
        };
        for (int B : {20, 50}) {
            int prev = -1;
            bool ok = true;
            std::printf("  桶=%d 各牌型桶位:", B);
            for (const auto& c : cases) {
                const texas::Card hole[2] = {C(c.h1), C(c.h2)};
                const texas::Card board[3] = {C(c.b1), C(c.b2), C(c.b3)};
                // 先确认构造出来的确实是目标牌型
                texas::Card all[5] = {C(c.h1), C(c.h2), C(c.b1), C(c.b2), C(c.b3)};
                CHECK(texas::handCategoryOf(texas::evaluate5(all)) == static_cast<int>(c.cat));
                const int bk = model.postflopBucket(hole, board, 3, B);
                std::printf(" %s=%d", c.name, bk);
                if (bk <= prev) ok = false;
                prev = bk;
            }
            std::printf("\n");
            CHECK(ok);
        }
        // 超对和暗三条绝对不能同桶（老版本 10 桶时这里就是坏的）
        const texas::Card b[3] = {C("Ks"), C("7c"), C("2d")};
        const texas::Card over[2] = {C("Ac"), C("Ah")};
        const texas::Card set[2] = {C("7d"), C("7s")};
        CHECK(model.postflopBucket(over, b, 3, 20) != model.postflopBucket(set, b, 3, 20));
        CHECK(model.postflopBucket(set, b, 3, 20) > model.postflopBucket(over, b, 3, 20));
    }

    std::printf("  OK 牌力抽象（评测器等价 / 翻牌前 169 类 / 听牌潜力修正 / 牌型分桶）\n");
}

void testHoldemInfosetPartition() {
    section("无限注德州 / 信息集划分：应对方必须看得见下注额");

    texas::StrengthModel::instance();

    // 造一手固定的牌，避免桶把键搅乱（翻牌前用的是 preflopClass，与下注无关）
    auto fresh = [] {
        texas::HoldemGame::State s = texas::HoldemGame::init(2);
        s.hole[0][0] = C("Ac");
        s.hole[0][1] = C("Ad");
        s.hole[1][0] = C("7h");
        s.hole[1][1] = C("2d");
        return s;
    };

    // 小盲加注到不同尺寸后，大盲面对的必须是不同信息集。
    // 这里的 5 / 8 / 200 是 2.5x、4x 和全下，是三个完全不同的局面。
    texas::HoldemAbstraction sized;               // 默认 sizedHistory = true
    texas::HoldemAbstraction flat;
    flat.sizedHistory = false;
    CHECK(sized.sizedHistory);
    CHECK(!flat.sizedHistory);

    {
        const int totals[3] = {5, 8, 200};
        std::uint64_t ks[3];
        std::uint64_t kf[3];
        int contribs[3];
        for (int i = 0; i < 3; ++i) {
            const texas::HoldemGame::State s =
                texas::HoldemGame::next(fresh(), texas::Move{texas::Move::Raise, totals[i]});
            ks[i] = sized.infosetId(s);
            kf[i] = flat.infosetId(s);
            contribs[i] = s.contrib[0];
            std::printf("  加注到 %-3d -> 投入 %-3d 看得见键 %016llx  看不见键 %016llx\n", totals[i],
                        contribs[i], static_cast<unsigned long long>(ks[i]),
                        static_cast<unsigned long long>(kf[i]));
        }
        // 三个局面确实不同（否则这个测试本身没意义）
        CHECK(contribs[0] != contribs[1] && contribs[1] != contribs[2] && contribs[0] != contribs[2]);
        // 看不见尺寸时三者塌成同一个键——这正是那个缺陷的表现
        CHECK(kf[0] == kf[1] && kf[1] == kf[2]);
        // 看得见时必须三者互不相同
        CHECK(ks[0] != ks[1]);
        CHECK(ks[1] != ks[2]);
        CHECK(ks[0] != ks[2]);
    }

    // 同一个决策节点下的三个加注动作，字符历史都是 'r'，但局面不同。
    // 这条断言直接盯着"历史里只有字符"这个失信息的机制。
    {
        texas::HoldemGame::State s = fresh();
        texas::Move mv[texas::HoldemGame::kMaxActions];
        const int na = texas::HoldemGame::actions(s, mv, texas::HoldemGame::kMaxActions);
        int raiseCount = 0;
        int distinctSized = 0;
        int distinctFlat = 0;
        int distinctChars = 0;
        std::uint64_t firstSized = 0;
        std::uint64_t firstFlat = 0;
        std::string firstStr;
        for (int i = 0; i < na; ++i) {
            if (mv[i].kind != texas::Move::Raise) continue;
            const texas::HoldemGame::State t = texas::HoldemGame::next(s, mv[i]);
            const std::uint64_t ks = sized.infosetId(t);
            const std::uint64_t kf = flat.infosetId(t);
            const std::string str = t.histString();
            if (raiseCount == 0) {
                firstSized = ks;
                firstFlat = kf;
                firstStr = str;
            } else {
                if (ks != firstSized) ++distinctSized;
                if (kf != firstFlat) ++distinctFlat;
                if (str != firstStr) ++distinctChars;
            }
            ++raiseCount;
        }
        std::printf("  同一节点的加注动作 %d 个\n", raiseCount);
        // 引擎当前给出的加注动作数：2.5x / 4x / 全下
        CHECK(raiseCount == 3);
        // 可读历史字符串完全一样（都是 "r"），这正是缺陷的机制
        CHECK(distinctChars == 0);
        CHECK(distinctSized == 2);  // 三个带尺寸键两两不同
        CHECK(distinctFlat == 0);   // 看不见尺寸时全部相同
    }

    // 不变的恒等式：没有加注的历史上，两个哈希必须逐位相同。
    // 这条保证"抹掉尺寸"只在加注处丢信息，不会莫名其妙地额外改动别的东西。
    {
        int checked = 0;
        int mismatch = 0;
        // 只有过牌/跟注/弃牌的路径
        texas::HoldemGame::State s = fresh();
        for (int i = 0; i < 12 && !texas::HoldemGame::isTerminal(s); ++i) {
            if (texas::HoldemGame::needsChance(s)) {
                texas::Card b[5] = {C("Ks"), C("7c"), C("2d"), C("9h"), C("3s")};
                texas::HoldemGame::reveal(s, b);
            } else {
                texas::HoldemGame::State t =
                    texas::HoldemGame::next(s, texas::Move{texas::Move::CheckCall, 0});
                // CheckCall 在引擎内部可能是跟注（'c'）也可能是过牌（'k'），两者都不碰尺寸哈希
                if (t.histHash != t.histSizedHash) ++mismatch;
                ++checked;
                s = t;
            }
        }
        std::printf("  无加注路径上检查了 %d 步，两哈希不一致 %d 处\n", checked, mismatch);
        CHECK(checked > 3);
        CHECK(mismatch == 0);
    }

    // 哈希格式一旦改变，所有已存的模型文件都会读不出来（键全变），所以把定义钉在这里。
    // 下面这段是 pushHist 的规格说明，不是"顺手抄一遍实现"。
    {
        const texas::HoldemGame::State s =
            texas::HoldemGame::next(fresh(), texas::Move{texas::Move::Raise, 8});
        std::uint64_t h = 0;
        for (const char* p = "r"; *p != '\0'; ++p) {
            h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(*p));
            h *= 1099511628211ULL;
        }
        std::printf("  手动折算出无尺寸哈希 %016llx，引擎 %016llx\n",
                    static_cast<unsigned long long>(h),
                    static_cast<unsigned long long>(s.histHash));
        CHECK(s.histHash == h);
        // 有尺寸的那个必须与之不同（否则说明 mixRaiseSize 没被调用）
        CHECK(s.histSizedHash != s.histHash);
    }

    std::printf("  OK 信息集划分（尺寸可见 / 抹掉尺寸时塌缩 / 恒等式 / 哈希格式固定）\n");
}

void testHoldemSolverVariants() {
    section("无限注德州 / 遗憾折扣与模型存盘");

    // DCFR 的核心好处是同一轮数下"当前策略"离均衡更近，用遗憾上界直接量。
    //
    // 阈值随信息划分变过：修好"看不见下注额"的缺陷后，树大了 15 倍（3.8 万 -> 59 万
    // 信息集），10 万轮摊薄到每个信息集上就不够了，比值从 <0.55 抬到 0.72。
    // 这不是 DCFR 变差——比值随预算单调下降：10万=0.720 20万=0.638 50万=0.554 100万=0.503，
    // 且同预算对打 DCFR 在 10万/20万/50万/100万 分别 +1.96/+1.32/+1.08/+1.06 筹码每手
    // （6.7~9.0 sigma）。所以这里断言 0.80，并把上界随预算下降的趋势交给下面第二个块管。
    // 10 万轮 20 桶、6 个种子的实测比值区间是 0.706~0.747，取 0.80 留了余量。
    {
        texas::HoldemSolver plus(20, 20240518ULL, texas::CfrVariant::CFRPlus);
        texas::HoldemSolver dcfr(20, 20240518ULL, texas::CfrVariant::DCFR);
        plus.iterate(100000);
        dcfr.iterate(100000);
        std::printf("  10 万轮遗憾上界: CFR+=%.2f DCFR=%.2f（比值 %.2f）\n", plus.regretBound(),
                    dcfr.regretBound(), dcfr.regretBound() / plus.regretBound());
        CHECK(dcfr.regretBound() < plus.regretBound() * 0.80);
    }

    // 遗憾上界应随轮数下降，否则说明折扣因子算错了（比如把 t 写成常数）
    {
        texas::HoldemSolver a(20, 777ULL, texas::CfrVariant::DCFR);
        texas::HoldemSolver b(20, 777ULL, texas::CfrVariant::DCFR);
        a.iterate(40000);
        b.iterate(160000);
        std::printf("  遗憾上界 4万=%.2f 16万=%.2f\n", a.regretBound(), b.regretBound());
        CHECK(b.regretBound() < a.regretBound());
    }

    // 存盘 / 读盘的往返必须逐位一致，否则"用冻结快照当基准对手"这件事就不可靠
    {
        const char* path = "texas_cfr_test_model.bin";
        texas::HoldemSolver src(20, 424242ULL, texas::CfrVariant::DCFR);
        src.iterate(20000);
        CHECK(src.save(path));

        texas::HoldemSolver dst(20, 1ULL, texas::CfrVariant::DCFR);
        CHECK(dst.load(path));
        CHECK(dst.numInfosets() == src.numInfosets());
        // 轮数也要存下来，否则读盘后遗憾上界算不出来、断点续训的折扣因子也会错
        CHECK(dst.iterations() == src.iterations());

        // 取若干手牌比对平均策略
        const char* hands[][2] = {{"Ac", "Ad"}, {"7h", "2d"}, {"Kh", "Qh"}, {"9c", "8c"}};
        int mismatch = 0;
        for (auto& h : hands) {
            texas::HoldemGame::State st = texas::HoldemGame::init(2);
            st.hole[0][0] = C(h[0]);
            st.hole[0][1] = C(h[1]);
            const std::vector<double> a = src.averageStrategy(st);
            const std::vector<double> b = dst.averageStrategy(st);
            if (a.size() != b.size()) {
                ++mismatch;
                continue;
            }
            for (std::size_t i = 0; i < a.size(); ++i) {
                if (std::fabs(a[i] - b[i]) > 1e-12) ++mismatch;
            }
        }
        CHECK(mismatch == 0);

        // 读一个不存在或损坏的文件必须失败而不是崩溃
        texas::HoldemSolver bad(20, 1ULL, texas::CfrVariant::DCFR);
        CHECK(!bad.load("texas_cfr_no_such_model.bin"));
        std::remove(path);
    }

    // 两边策略完全相同，座位对称对打的结果应该贴近 0
    {
        texas::HoldemSolver a(20, 20240518ULL, texas::CfrVariant::DCFR);
        a.iterate(30000);
        const texas::MatchResult r = texas::headToHeadBothSeats(a, a, 20000, 99ULL);
        std::printf("  自我对打 %+0.4f ± %.4f 筹码/手（应贴近 0）\n", r.heroPerHand,
                    1.96 * r.stderr);
        CHECK(std::fabs(r.heroPerHand) < 4.0 * r.stderr + 0.05);
    }

    std::printf("  OK 遗憾折扣（DCFR 上界更低）/ 存盘往返一致 / 自我对打归零\n");
}

void testHoldemMCCFR() {
    section("无限注德州 / MCCFR 求解");

    texas::HoldemSolver solver(20, 20240518ULL);
    const int rounds = 500000;
    solver.iterate(rounds);
    std::printf("  %d 轮后 infosets=%zu\n", rounds, solver.numInfosets());
    CHECK(solver.numInfosets() > 100);

    // 翻牌前小盲面对大盲：强牌应该比弱牌更常加注、更少弃牌
    auto probe = [&](const char* a, const char* b) {
        texas::HoldemGame::State st = texas::HoldemGame::init(2);
        st.hole[0][0] = C(a);
        st.hole[0][1] = C(b);
        return solver.averageStrategy(st);
    };
    const std::vector<double> pAA = probe("Ac", "Ad");
    const std::vector<double> p72 = probe("7h", "2d");

    auto sumRaise = [](const std::vector<double>& p) {
        double r = 0.0;
        for (std::size_t i = 2; i < p.size(); ++i) r += p[i];
        return r;
    };
    const double raiseAA = sumRaise(pAA);
    const double raise72 = sumRaise(p72);
    const double foldAA = pAA.empty() ? 1.0 : pAA[0];
    const double fold72 = p72.empty() ? 0.0 : p72[0];

    std::printf("  翻牌前: AA raise=%.3f fold=%.3f | 72o raise=%.3f fold=%.3f\n", raiseAA, foldAA,
                raise72, fold72);
    // 阈值由实测支撑，不是估计。8 个种子各跑 50 万轮（含尺寸可见的信息划分）：
    //   AA 弃牌率最高 0.0020；72o 弃牌最低 0.921；raiseAA > raise72 在 8/8 个种子上成立，
    //   最紧的一次是 5.84 倍。
    // 但"raiseAA 是 raise72 的几倍"**仍然不稳定**：同样 8 个种子，比值分布在 5.84 到 1084
    // 之间（种子 2024 是 5.84，种子 999983 是 1084——分母已经接近 0）。所以这里只断言
    // 序关系和弃牌率，不断言倍数：早先写过 `raiseAA > raise72 * 2.0`，那条会随随机流假警报。
    CHECK(raiseAA > raise72);  // 8/8 成立，最紧的一次是 5.84 倍
    CHECK(foldAA < 0.01);      // AA 几乎不弃牌（8 个种子最高 0.0020）
    CHECK(fold72 > 0.85);      // 72o 大部分直接弃（8 个种子最低 0.921）
}

}  // namespace

int main() {
    std::printf("PerfectTexasCPP 自测开始\n");
    std::printf("================================================\n");

    testCards();
    testEquity();
    testKuhn();
    testExploitabilityBaseline();
    testLeduc();
    testVanillaVsPlus();
    testHoldemRules();
    testHoldemActionSpace();
    testActionSpaceNaming();
    testHoldemSaveV4Compat();
    testHoldemShowdown();
    testHoldemSidePot();
    testHoldemScoreCache();
    testHoldemAllIn();
    testHoldemAbstraction();
    testHoldemInfosetPartition();
    testHoldemSolverVariants();
    testHoldemMCCFR();

    std::printf("\n================================================\n");
    std::printf("通过 %d 项，失败 %d 项\n", g_passed, g_failed);
    if (g_failed == 0) {
        std::printf("全部通过\n");
        return 0;
    }
    std::printf("存在失败项\n");
    return 1;
}
