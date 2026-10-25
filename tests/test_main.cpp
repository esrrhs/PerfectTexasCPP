// 自测：牌力评估、胜率、CFR 收敛性、均衡策略性质。
// 关键的硬断言用解析解做对照：
//   - Kuhn 先手博弈值 = -1/18
//   - Kuhn 均衡策略的已知结构（α / 3α 关系、Q 从不主动下注等）
//   - 均衡策略的 exploitability 必须显著低于均匀策略

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "texas/br.hpp"
#include "texas/card.hpp"
#include "texas/cfr.hpp"
#include "texas/equity.hpp"
#include "texas/kuhn.hpp"
#include "texas/leduc.hpp"

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

}  // namespace

int main() {
    std::printf("texas_cfr 自测开始\n");
    std::printf("================================================\n");

    testCards();
    testEquity();
    testKuhn();
    testExploitabilityBaseline();
    testLeduc();
    testVanillaVsPlus();

    std::printf("\n================================================\n");
    std::printf("通过 %d 项，失败 %d 项\n", g_passed, g_failed);
    if (g_failed == 0) {
        std::printf("全部通过\n");
        return 0;
    }
    std::printf("存在失败项\n");
    return 1;
}
