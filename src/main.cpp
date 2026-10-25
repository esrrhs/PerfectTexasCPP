// texas_cfr 命令行入口。
//
//   texas_cli solve kuhn  [--iters N] [--report M] [--csv FILE] [--vanilla] [--dump N]
//   texas_cli solve leduc [--iters N] [--report M] [--csv FILE] [--vanilla] [--dump N]
//   texas_cli holdem      [--iters N] [--buckets N] [--variant dcfr|cfrplus]
//                         [--save FILE] [--load FILE] [--seed N] [--ev-hands N]
//   texas_cli match --a FILE --b FILE [--hands N] [--seed N]
//   texas_cli equity --hero AhKh [--board QhJh2c] [--opp 1] [--iters 200000]
//   texas_cli demo
//
// solve  训练一个 CFR 求解器，周期性报告可利用度与博弈值
// holdem 训练无限注德州求解器，可存盘成基准
// match  两个已存盘策略对打，报每手净收益与标准误（判断谁更强）
// equity 手牌胜率（蒙特卡洛）

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "texas/br.hpp"
#include "texas/card.hpp"
#include "texas/cfr.hpp"
#include "texas/equity.hpp"
#include "texas/holdem.hpp"
#include "texas/kuhn.hpp"
#include "texas/leduc.hpp"
#include "texas/mccfr.hpp"
#include "texas/strength.hpp"

namespace {

struct Args {
    std::string cmd;
    std::string game;
    int iters = -1;
    int report = -1;
    int dump = -1;
    std::string csv;
    bool vanilla = false;
    std::string hero;
    std::string board;
    int opp = 1;
    int buckets = -1;
    std::string variant = "dcfr";
    std::string save;
    std::string load;
    std::string fileA;
    std::string fileB;
    int hands = -1;
    unsigned long long seed = 20240518ULL;
    int evHands = -1;
};

void usage() {
    std::printf(
        "texas_cfr —— 德州扑克不完全信息博弈求解器（CFR / CFR+ / DCFR）\n"
        "\n"
        "用法:\n"
        "  texas_cli solve kuhn  [--iters N] [--report M] [--csv FILE] [--vanilla] [--dump N]\n"
        "  texas_cli solve leduc [--iters N] [--report M] [--csv FILE] [--vanilla] [--dump N]\n"
        "  texas_cli holdem      [--iters N] [--buckets N] [--variant dcfr|cfrplus]\n"
        "                        [--save FILE] [--load FILE] [--seed N] [--ev-hands N]\n"
        "  texas_cli match --a FILE --b FILE [--hands N] [--seed N]\n"
        "  texas_cli equity --hero AhKh [--board QhJh2c] [--opp 1] [--iters 200000]\n"
        "  texas_cli demo\n"
        "\n"
        "选项:\n"
        "  --iters N    迭代轮数（kuhn 默认 20000，leduc 默认 10000，holdem 默认 500000）\n"
        "  --report M   每隔多少轮报告一次（默认 iters/10）\n"
        "  --csv FILE   把 (iter,exploitability,value) 曲线写成 CSV\n"
        "  --vanilla    用原始 CFR 而不是 CFR+\n"
        "  --dump N     打印前 N 个信息集的平均策略\n"
        "  --hero HAND  自己的两张手牌，如 AhKh\n"
        "  --board STR  公共牌，如 QhJh2c（可空）\n"
        "  --opp N      对手人数（默认 1）\n"
        "  --buckets N  翻牌后分桶数（默认 20，最小 9；越大越细但收敛越慢）\n"
        "  --variant V  遗憾累计方式：dcfr（默认）或 cfrplus\n"
        "  --save FILE  训练完把策略存盘，可当作以后的基准对手\n"
        "  --load FILE  从存盘读入策略（给了就跳过训练；续训时 --variant 要与原文件一致）\n"
        "  --seed N     随机种子（默认 20240518）\n"
        "  --ev-hands N 对均匀随机对手的评估手数（默认 20000，0 表示跳过）\n"
        "  --hands N    match 的对局手数（默认 150000，两个座位各打一遍）\n"
        "  --a / --b    match 的双方策略文件（--a 为被测方，收益从它视角计）\n");
}

int parseInt(const char* s, int fallback) {
    if (!s) return fallback;
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s) return fallback;
    return static_cast<int>(v);
}

Args parseArgs(int argc, char** argv) {
    Args a;
    if (argc >= 2) a.cmd = argv[1];
    if (argc >= 3 && a.cmd == "solve") a.game = argv[2];

    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        const bool hasNext = (i + 1 < argc);
        if (k == "--iters" && hasNext) {
            a.iters = parseInt(argv[++i], a.iters);
        } else if (k == "--report" && hasNext) {
            a.report = parseInt(argv[++i], a.report);
        } else if (k == "--dump" && hasNext) {
            a.dump = parseInt(argv[++i], a.dump);
        } else if (k == "--csv" && hasNext) {
            a.csv = argv[++i];
        } else if (k == "--vanilla") {
            a.vanilla = true;
        } else if (k == "--hero" && hasNext) {
            a.hero = argv[++i];
        } else if (k == "--board" && hasNext) {
            a.board = argv[++i];
        } else if (k == "--opp" && hasNext) {
            a.opp = parseInt(argv[++i], a.opp);
        } else if (k == "--buckets" && hasNext) {
            a.buckets = parseInt(argv[++i], a.buckets);
        } else if (k == "--variant" && hasNext) {
            a.variant = argv[++i];
        } else if (k == "--save" && hasNext) {
            a.save = argv[++i];
        } else if (k == "--load" && hasNext) {
            a.load = argv[++i];
        } else if (k == "--a" && hasNext) {
            a.fileA = argv[++i];
        } else if (k == "--b" && hasNext) {
            a.fileB = argv[++i];
        } else if (k == "--hands" && hasNext) {
            a.hands = parseInt(argv[++i], a.hands);
        } else if (k == "--ev-hands" && hasNext) {
            a.evHands = parseInt(argv[++i], a.evHands);
        } else if (k == "--seed" && hasNext) {
            a.seed = std::strtoull(argv[++i], nullptr, 10);
        }
    }
    return a;
}

std::vector<texas::Card> parseCards(const std::string& s) {
    std::vector<texas::Card> out;
    for (std::size_t i = 0; i + 1 < s.size(); i += 2) {
        const texas::Card c = texas::parseCard(s.substr(i, 2));
        if (c >= 0) out.push_back(c);
    }
    return out;
}

template <class G>
void printStrategy(const texas::CFRSolver<G>& solver,
                   const texas::Strategy& sigma,
                   int maxCount) {
    std::vector<std::pair<std::string, texas::InfoKey>> rows;
    for (const auto& kv : solver.names()) rows.emplace_back(kv.second, kv.first);
    std::sort(rows.begin(), rows.end());

    const int limit = (maxCount < 0) ? static_cast<int>(rows.size())
                                     : std::min<int>(maxCount, static_cast<int>(rows.size()));
    std::printf("\n  平均策略（前 %d / %zu 个信息集）:\n", limit, rows.size());
    for (int i = 0; i < limit; ++i) {
        const texas::InfoKey key = rows[static_cast<std::size_t>(i)].second;
        const auto itS = sigma.find(key);
        const auto itI = solver.infosets().find(key);
        if (itS == sigma.end() || itI == solver.infosets().end()) continue;
        std::printf("    %-34s", rows[static_cast<std::size_t>(i)].first.c_str());
        for (int j = 0; j < itI->second.n; ++j) {
            std::printf(" %c=%.3f", itI->second.actChars[j], itS->second[static_cast<std::size_t>(j)]);
        }
        std::printf("\n");
    }
}

template <class G>
int runSolve(const Args& a) {
    using Solver = texas::CFRSolver<G>;
    const bool isKuhn = (std::is_same<G, texas::KuhnGame>::value);
    const int iters = (a.iters > 0) ? a.iters : (isKuhn ? 20000 : 10000);
    const int report = (a.report > 0) ? a.report : std::max(1, iters / 10);
    const int dump = (a.dump >= 0) ? a.dump : (isKuhn ? -1 : 24);

    const typename Solver::Options opt(!a.vanilla, !a.vanilla);
    Solver solver(opt);

    std::ofstream csv;
    if (!a.csv.empty()) {
        csv.open(a.csv);
        if (csv) csv << "iter,exploitability,value\n";
    }

    std::printf("\n=== %s | %s | 对手 %d 人 ===\n", isKuhn ? "Kuhn Poker" : "Leduc Hold'em",
                a.vanilla ? "vanilla CFR" : "CFR+ (alternating, linear avg)", 1);
    std::printf("  %10s  %14s  %12s  %10s\n", "iter", "exploitability", "value(P0)", "infosets");

    int done = 0;
    while (done < iters) {
        const int chunk = std::min(report, iters - done);
        solver.iterate(chunk);
        done += chunk;

        const texas::Strategy sigma = solver.averageStrategy();
        const double expl = texas::exploitability<G>(sigma);
        const double val = solver.expectedValue(sigma);
        std::printf("  %10d  %14.6f  %12.6f  %10zu\n", done, expl, val, solver.numInfosets());
        if (csv) csv << done << "," << expl << "," << val << "\n";
    }

    printStrategy(solver, solver.averageStrategy(), dump);
    if (!a.csv.empty()) std::printf("\n  CSV 已写入 %s\n", a.csv.c_str());
    return 0;
}

int runEquity(const Args& a) {
    const std::vector<texas::Card> hero = parseCards(a.hero);
    const std::vector<texas::Card> board = parseCards(a.board);
    if (hero.size() != 2) {
        std::printf("错误: --hero 需要正好两张牌，例如 AhKh\n");
        return 1;
    }
    const int iters = (a.iters > 0) ? a.iters : 200000;

    std::printf("\n=== 胜率估算 ===\n");
    std::printf("  手牌   : %s %s\n", texas::cardToString(hero[0]).c_str(),
                texas::cardToString(hero[1]).c_str());
    std::printf("  公共牌 : ");
    if (board.empty()) {
        std::printf("(无)\n");
    } else {
        for (texas::Card c : board) std::printf("%s ", texas::cardToString(c).c_str());
        std::printf("\n");
    }
    std::printf("  对手   : %d 人\n", a.opp);
    std::printf("  采样   : %d 次\n", iters);

    const texas::EquityResult r = texas::equityMonteCarlo(hero, board, a.opp, iters);
    std::printf("\n  胜 %.2f%%   平 %.2f%%   负 %.2f%%   ->  equity %.2f%%\n", r.win * 100.0,
                r.tie * 100.0, r.lose * 100.0, r.equity() * 100.0);
    return 0;
}

int runDemo() {
    std::printf("\n########## Kuhn Poker：CFR+ 收敛过程 ##########\n");
    {
        texas::CFRSolver<texas::KuhnGame> solver;
        double prevExpl = 1e9;
        for (int step = 1; step <= 6; ++step) {
            solver.iterate(step == 1 ? 100 : 500);
            const texas::Strategy sigma = solver.averageStrategy();
            const double expl = texas::exploitability<texas::KuhnGame>(sigma);
            const double val = solver.expectedValue(sigma);
            std::printf("  iter=%6d  expl=%.6f  value=%+.6f  (-1/18 = %.6f)\n", solver.iterations(),
                        expl, val, -1.0 / 18.0);
            prevExpl = expl;
        }
        std::printf("  最终误差: |value - (-1/18)| = %.6f\n",
                    std::fabs(solver.expectedValue(solver.averageStrategy()) + 1.0 / 18.0));
        (void)prevExpl;
    }

    std::printf("\n########## 手牌胜率 ##########\n");
    {
        const texas::Card aa[2] = {texas::parseCard("Ah"), texas::parseCard("As")};
        const texas::Card kk[2] = {texas::parseCard("Kh"), texas::parseCard("Ks")};
        const texas::Card s72[2] = {texas::parseCard("7h"), texas::parseCard("2c")};
        const std::vector<texas::Card> none;
        const auto rAA = texas::equityMonteCarlo({aa[0], aa[1]}, none, 1, 50000);
        const auto rKK = texas::equityMonteCarlo({kk[0], kk[1]}, none, 1, 50000);
        const auto r72 = texas::equityMonteCarlo({s72[0], s72[1]}, none, 1, 50000);
        std::printf("  AA vs 随机 : %.2f%%\n", rAA.equity() * 100.0);
        std::printf("  KK vs 随机 : %.2f%%\n", rKK.equity() * 100.0);
        std::printf("  72o vs 随机: %.2f%%\n", r72.equity() * 100.0);
    }
    return 0;
}

texas::CfrVariant parseVariant(const std::string& s) {
    if (s == "cfrplus" || s == "cfr+" || s == "plus") return texas::CfrVariant::CFRPlus;
    return texas::CfrVariant::DCFR;
}

const char* variantName(texas::CfrVariant v) {
    return v == texas::CfrVariant::DCFR ? "DCFR" : "CFR+";
}

int runHoldem(const Args& a) {
    const int iters = (a.iters > 0) ? a.iters : 500000;
    const int buckets = (a.buckets > 0) ? a.buckets : 20;
    const int report = (a.report > 0) ? a.report : std::max(1, iters / 10);
    const int evHands = (a.evHands >= 0) ? a.evHands : 20000;
    const texas::CfrVariant variant = parseVariant(a.variant);

    texas::StrengthModel::instance();  // 预热牌力表
    texas::HoldemSolver solver(buckets, a.seed, variant);

    std::printf("\n=== 无限注德州 Hold'em | 单挑 100BB | MCCFR (external sampling) ===\n");
    std::printf("  翻牌后 %d 桶 | 抽象: 翻牌前 169 类全分辨率 + 翻牌后潜力修正百分位 | %s\n",
                buckets, variantName(variant));

    bool loaded = false;
    if (!a.load.empty()) {
        loaded = solver.load(a.load.c_str());
        if (!loaded) {
            std::printf("错误: 读不到策略文件 %s\n", a.load.c_str());
            return 1;
        }
        std::printf("  已读入 %s：infosets=%zu 遗憾上界=%.4f（跳过训练）\n", a.load.c_str(),
                    solver.numInfosets(), solver.regretBound());
    }

    if (!loaded) {
        std::printf("  %10s  %10s  %10s  %12s\n", "iter", "infosets", "轮/s", "遗憾上界");
        int done = 0;
        while (done < iters) {
            const int chunk = std::min(report, iters - done);
            const auto t0 = std::chrono::steady_clock::now();
            solver.iterate(chunk);
            const auto t1 = std::chrono::steady_clock::now();
            done += chunk;
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::printf("  %10d  %10zu  %10.0f  %12.4f\n", done, solver.numInfosets(),
                        ms > 0 ? chunk / (ms / 1000.0) : 0.0, solver.regretBound());
        }
        if (!a.save.empty()) {
            if (solver.save(a.save.c_str())) {
                std::printf("\n  策略已存盘: %s\n", a.save.c_str());
            } else {
                std::printf("\n  错误: 存盘失败 %s\n", a.save.c_str());
            }
        }
    }

    std::printf("\n  翻牌前（小盲/按钮面对大盲）平均策略:\n");
    auto show = [&](const char* label, const char* c1, const char* c2) {
        texas::HoldemGame::State st = texas::HoldemGame::init(2);
        st.hole[0][0] = texas::parseCard(c1);
        st.hole[0][1] = texas::parseCard(c2);
        const std::vector<double> p = solver.averageStrategy(st);
        texas::Move mv[texas::HoldemGame::kMaxActions];
        const int n = texas::HoldemGame::actions(st, mv, texas::HoldemGame::kMaxActions);
        std::printf("    %-6s", label);
        for (int i = 0; i < n && i < static_cast<int>(p.size()); ++i) {
            if (mv[i].kind == texas::Move::Fold) {
                std::printf("  fold=%.2f", p[static_cast<std::size_t>(i)]);
            } else if (mv[i].kind == texas::Move::CheckCall) {
                std::printf("  call=%.2f", p[static_cast<std::size_t>(i)]);
            } else {
                std::printf("  r%d=%.2f", mv[i].amount, p[static_cast<std::size_t>(i)]);
            }
        }
        std::printf("\n");
    };
    show("AA", "Ac", "Ad");
    show("KK", "Kc", "Kd");
    show("AKs", "Ah", "Kh");
    show("22", "2c", "2d");
    show("72o", "7h", "2d");

    // 策略质量标尺：打均匀随机对手的平均净收益（不是可利用度，只做横向比较）
    if (evHands > 0) {
        const double evSb = solver.evVsUniform(evHands, 0);
        const double evBb = solver.evVsUniform(evHands, 1);
        std::printf("\n  对均匀随机对手的每手平均净收益（%d 手）:\n", evHands);
        std::printf("    小盲/按钮位 %.3f 筹码 | 大盲位 %.3f 筹码\n", evSb, evBb);
    }
    return 0;
}

// 两个已存盘策略对打。这是唯一能验证"变强"的手段：自对局在零和对称游戏里恒为 0，
// 打均匀随机对手又太软，只有跟一个冻结的、同量级的策略对打才有分辨力。
int runMatch(const Args& a) {
    if (a.fileA.empty() || a.fileB.empty()) {
        std::printf("错误: match 需要 --a FILE 和 --b FILE\n");
        return 1;
    }
    const int hands = (a.hands > 0) ? a.hands : 150000;

    texas::HoldemSolver A(20, a.seed);
    texas::HoldemSolver B(20, a.seed);
    if (!A.load(a.fileA.c_str())) {
        std::printf("错误: 读不到策略文件 %s\n", a.fileA.c_str());
        return 1;
    }
    if (!B.load(a.fileB.c_str())) {
        std::printf("错误: 读不到策略文件 %s\n", a.fileB.c_str());
        return 1;
    }

    std::printf("\n=== 策略对打 ===\n");
    std::printf("  A（被测）: %s  infosets=%zu 遗憾上界=%.4f\n", a.fileA.c_str(), A.numInfosets(),
                A.regretBound());
    std::printf("  B（基准）: %s  infosets=%zu 遗憾上界=%.4f\n", a.fileB.c_str(), B.numInfosets(),
                B.regretBound());
    std::printf("  手数: %d（两个座位各 %d 手，消除位置优势）\n", hands * 2, hands);

    const texas::MatchResult r = texas::headToHeadBothSeats(A, B, hands, a.seed * 7919ULL + 17ULL);
    const double ci = 1.96 * r.stderr;
    std::printf("\n  A 每手净收益: %+0.4f 筹码  (95%% 置信区间 ±%.4f)\n", r.heroPerHand, ci);
    if (std::fabs(r.heroPerHand) <= ci) {
        std::printf("  结论: 差异不显著（区间含 0）\n");
    } else {
        std::printf("  结论: A 显著%s，约 %s %.4f 筹码/手\n", r.heroPerHand > 0 ? "更强" : "更弱",
                    r.heroPerHand > 0 ? "多赢" : "少赢", std::fabs(r.heroPerHand));
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parseArgs(argc, argv);

    if (a.cmd.empty() || a.cmd == "--help" || a.cmd == "-h") {
        usage();
        return a.cmd.empty() ? 1 : 0;
    }
    if (a.cmd == "demo") return runDemo();
    if (a.cmd == "equity") return runEquity(a);
    if (a.cmd == "holdem") return runHoldem(a);
    if (a.cmd == "match") return runMatch(a);
    if (a.cmd == "solve") {
        if (a.game == "kuhn") return runSolve<texas::KuhnGame>(a);
        if (a.game == "leduc") return runSolve<texas::LeducGame>(a);
        std::printf("错误: 未知游戏 '%s'（可选 kuhn / leduc）\n", a.game.c_str());
        return 1;
    }

    std::printf("错误: 未知命令 '%s'\n\n", a.cmd.c_str());
    usage();
    return 1;
}
