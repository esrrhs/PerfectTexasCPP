// PerfectTexasCPP 命令行入口。
//
//   perfecttexas train  [--updates N] [--games N] [--out DIR] [--resume DIR]
//   perfecttexas eval   --resume DIR [--vs rule|random|self] [--hands N]
//   perfecttexas bench  [--hands N]
//
// train  自对局 PPO 训练（不完全信息 actor + 开天眼 critic）
// eval   拿存盘模型对固定基准打对局，报每手净收益与标准误
// bench  未训练策略打规则机器人，作为强度下限

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "texas/eval.hpp"
#include "texas/features.hpp"
#include "texas/hu.hpp"
#include "texas/learner.hpp"
#include "texas/nn.hpp"
#include "texas/ppo.hpp"
#include "texas/strength.hpp"
#include "texas/rollout.hpp"

namespace {

const char* argValue(int argc, char** argv, const char* key, const char* def) {
    for (int i = 0; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return def;
}

bool hasFlag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) return true;
    }
    return false;
}

int argInt(int argc, char** argv, const char* key, int def) {
    const char* v = argValue(argc, argv, key, nullptr);
    return v ? std::atoi(v) : def;
}

double argDouble(int argc, char** argv, const char* key, double def) {
    const char* v = argValue(argc, argv, key, nullptr);
    return v ? std::atof(v) : def;
}

std::string argStr(int argc, char** argv, const char* key, const std::string& def) {
    const char* v = argValue(argc, argv, key, nullptr);
    return v ? std::string(v) : def;
}

void usage() {
    std::printf(
        "PerfectTexasCPP —— 单挑无限注德州的自对局强化学习（PPO + 完美信息蒸馏）\n"
        "\n"
        "用法:\n"
        "  perfecttexas train [--updates N] [--games N] [--lr F] [--hidden N] [--lstm N]\n"
        "                     [--out DIR] [--resume DIR] [--mps] [--threads N]\n"
        "                     [--league F] [--rule-prob F] [--random-prob F]\n"
        "                     [--archive-ratio F] [--recent N] [--archive N]\n"
        "                     [--oracle-iters N]\n"
        "                     [--eval-every N] [--eval-games N]\n"
        "                     [--oracle-coef F] [--no-oracle] [--seed N]\n"
        "  perfecttexas eval  --resume DIR [--vs rule|random|self] [--hands N]\n"
        "  perfecttexas probe  [--resume DIR]\n"
        "  perfecttexas bench  [--hands N]\n"
        "\n"
        "说明:\n"
        "  --league F      一局里对手是历史快照的概率（默认 0.30）\n"
        "  --rule-prob F   一局里对手是规则机器人的概率（默认 0.15）\n"
        "  --random-prob F 一局里对手是均匀随机的概率（默认 0.05）\n"
        "  --archive-ratio F 抽历史对手时用归档快照的比例（默认 0.30）\n"
        "  --recent N      历史池的滚动窗口容量（默认 8）\n"
        "  --archive N     历史池的归档容量，满了隔一个丢一个（默认 16）\n"
        "  --oracle-coef F 开天眼稠密奖励的系数（默认 0.05，0 等于关掉）\n"
        "  --no-oracle     关闭开天眼稠密奖励（只保留 critic 的完美信息价值）\n");
}

void saveModels(const texas::nn::Actor* actors, const texas::nn::Critic* critics,
                const std::string& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    for (int s = 0; s < 2; ++s) {
        actors[s]->saveTo(dir + "/actor" + std::to_string(s) + ".pt");
        critics[s]->saveTo(dir + "/critic" + std::to_string(s) + ".pt");
    }
}

bool loadModels(texas::nn::Actor* actors, texas::nn::Critic* critics, const std::string& dir) {
    if (!std::filesystem::exists(dir + "/actor0.pt")) return false;
    bool ok = true;
    for (int s = 0; s < 2; ++s) {
        ok = actors[s]->loadFrom(dir + "/actor" + std::to_string(s) + ".pt") && ok;
        ok = critics[s]->loadFrom(dir + "/critic" + std::to_string(s) + ".pt") && ok;
    }
    return ok;
}

torch::Device pickDevice(int argc, char** argv) {
    if (hasFlag(argc, argv, "--mps") && torch::mps::is_available()) return torch::kMPS;
    return torch::kCPU;
}

// ---------------------------------------------------------------- train

int cmdTrain(int argc, char** argv) {
    const int updates = argInt(argc, argv, "--updates", 200);
    const int games = argInt(argc, argv, "--games", 512);
    const std::string outDir = argStr(argc, argv, "--out", "ckpt");
    const std::string resumeDir = argStr(argc, argv, "--resume", "");
    const int seed = argInt(argc, argv, "--seed", 20240518);
    const int evalEvery = argInt(argc, argv, "--eval-every", 10);
    const int evalGames = argInt(argc, argv, "--eval-games", 512);
    const int recentCap = argInt(argc, argv, "--recent", 8);    // 历史池：滚动窗口
    const int archiveCap = argInt(argc, argv, "--archive", 16); // 历史池：几何间隔存档
    const int hwThreads = argInt(argc, argv, "--threads", 0);

    const torch::Device device = pickDevice(argc, argv);
    if (hwThreads > 0) torch::set_num_threads(hwThreads);

    texas::nn::ModelConfig mc;
    mc.hidden = argInt(argc, argv, "--hidden", 256);
    mc.lstmHidden = argInt(argc, argv, "--lstm", 128);

    texas::ModelSet models;
    for (int s = 0; s < 2; ++s) {
        models.actor[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        models.critic[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
    }

    if (!resumeDir.empty()) {
        if (loadModels(models.actor, models.critic, resumeDir)) {
            std::printf("resumed from %s\n", resumeDir.c_str());
        } else {
            std::printf("!! resume failed: %s (starting fresh)\n", resumeDir.c_str());
        }
    }

    texas::ppo::Config pcfg;
    pcfg.lr = static_cast<float>(argDouble(argc, argv, "--lr", 1e-3));
    pcfg.clip = static_cast<float>(argDouble(argc, argv, "--clip", 0.2));
    pcfg.entCoef = static_cast<float>(argDouble(argc, argv, "--entropy", 0.01));
    pcfg.epochs = argInt(argc, argv, "--epochs", 4);
    pcfg.minibatch = argInt(argc, argv, "--minibatch", 128);

    std::unique_ptr<texas::Learner> learners[2];
    for (int s = 0; s < 2; ++s) {
        learners[s] = std::make_unique<texas::Learner>(models.actor[s], models.critic[s], pcfg, device);
        if (!resumeDir.empty()) learners[s]->loadOptimizers(resumeDir);
    }

    texas::HistoricalPool pool;
    texas::RolloutConfig rcfg;
    rcfg.games = games;
    rcfg.historicalProb = static_cast<float>(argDouble(argc, argv, "--league", 0.30));
    rcfg.ruleProb = static_cast<float>(argDouble(argc, argv, "--rule-prob", 0.15));
    rcfg.randomProb = static_cast<float>(argDouble(argc, argv, "--random-prob", 0.05));
    rcfg.archiveProb = static_cast<float>(argDouble(argc, argv, "--archive-ratio", 0.30));
    rcfg.oracleReward = !hasFlag(argc, argv, "--no-oracle");
    rcfg.oracleCoef = static_cast<float>(argDouble(argc, argv, "--oracle-coef", 0.05));
    rcfg.equityIters = argInt(argc, argv, "--oracle-iters", 32);

    const float selfProb = std::max(
        0.0f, 1.0f - (rcfg.historicalProb + rcfg.ruleProb + rcfg.randomProb));
    std::printf("PerfectTexasCPP train | 单挑 NLHE %d 筹码 | 节点特征 %d + 开天眼 %d + 序列 %d\n",
                texas::kStartingStack, texas::kNodeInput, texas::kExtraInput, texas::kSeqInput);
    std::printf("  hidden=%lld lstm=%lld | device=%s | games/update=%d | oracle=%.2f\n",
                static_cast<long long>(mc.hidden), static_cast<long long>(mc.lstmHidden),
                device.type() == torch::kMPS ? "mps" : "cpu", games,
                rcfg.oracleReward ? rcfg.oracleCoef : 0.0f);
    // 联赛配比：每局只有一个座位是对手，另一个一定是当前策略。
    std::printf("  league: 自对局 %.2f | 历史快照 %.2f (archive %.2f) | 规则 %.2f | 随机 %.2f"
                "  [池 recent=%d archive=%d]\n",
                selfProb, rcfg.historicalProb, rcfg.archiveProb, rcfg.ruleProb, rcfg.randomProb,
                recentCap, archiveCap);
    std::printf("  %6s %7s %7s %6s %6s %6s %8s %8s %7s %6s %7s | %s\n", "update", "games", "sbWin%",
                "sd%", "jam%", "fold%", "pgLoss", "vLoss", "entropy", "kl", "upd/s", "batch");

    const auto t0 = std::chrono::steady_clock::now();
    for (int u = 1; u <= updates; ++u) {
        rcfg.seed = static_cast<uint64_t>(seed) * 1000003ULL + static_cast<uint64_t>(u);
        std::array<std::vector<texas::Transition>, 2> tr;
        texas::RolloutStats stats;
        texas::collectRollout(models, pool.empty() ? nullptr : &pool, rcfg, tr, stats);

        double pg = 0, vl = 0, ent = 0, kl = 0, retMean = 0, retStd = 0, valMean = 0;
        for (int s = 0; s < 2; ++s) {
            if (tr[s].empty()) continue;
            texas::ppo::assignEpisodeReturns(tr[s], pcfg.gamma, pcfg.lambda);
            texas::BatchTensors batch;
            texas::buildBatch(tr[s], batch);
            const texas::ppo::Stats st = learners[s]->update(batch);
            pg += st.pgLoss;
            vl += st.vLoss;
            ent += st.entropy;
            kl += st.kl;
            retMean += st.retMean;
            retStd += st.retStd;
            valMean += st.valueMean;
        }

        const auto now = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(now - t0).count();
        const double winPct =
            stats.games > 0 ? 100.0 * double(stats.sbWins) / double(stats.games) : 0.0;
        const double sdPct =
            stats.games > 0 ? 100.0 * double(stats.showdowns) / double(stats.games) : 0.0;
        const double jamPct = stats.decisions > 0
                                  ? 100.0 * double(stats.allinDecisions) / double(stats.decisions)
                                  : 0.0;
        const double foldPct = stats.decisions > 0
                                   ? 100.0 * double(stats.folds) / double(stats.decisions)
                                   : 0.0;
        std::printf(
            "  %6d %7d %7.1f %6.1f %6.1f %6.1f %8.4f %8.4f %7.3f %6.1e %7.2f | ret %.3f±%.3f val %.3f\n",
            u, stats.games, winPct, sdPct, jamPct, foldPct, pg / 2, vl / 2, ent / 2, kl / 2,
            double(u) / secs, retMean / 2, retStd / 2, valMean / 2);
        std::fflush(stdout);

        if (evalEvery > 0 && (u % evalEvery == 0 || u == updates)) {
            const texas::EvalResult vr = texas::evalVsRule(models.actor, evalGames, 12345 + u);
            const texas::EvalResult vn = texas::evalVsRandom(models.actor, evalGames, 54321 + u);
            const texas::EvalResult vs = texas::evalSelfPlay(models.actor, evalGames / 2, 777 + u);
            // 两个基准一起报：只报 vs 规则机器人时，一个不看牌的固定下注模式也能刷到
            // +11.5 筹码/手（见 README 第五节）。
            const int g = std::max(1, stats.games);
            std::printf("         [eval] rule %+6.2f±%.2f | random %+6.2f±%.2f | self %+5.2f±%.2f"
                        " | 对手 自%3d%% 近%2d%% 档%2d%% 规%2d%% 随%2d%%\n",
                        vr.chipsPerHand, vr.stderr, vn.chipsPerHand, vn.stderr, vs.chipsPerHand,
                        vs.stderr, 100 * stats.vsLatest / g, 100 * stats.vsRecent / g,
                        100 * stats.vsArchive / g, 100 * stats.vsRule / g,
                        100 * stats.vsRandom / g);
            std::fflush(stdout);

            auto snap = std::make_shared<texas::Snapshot>();
            snap->update = u;
            for (int s = 0; s < 2; ++s) {
                snap->actor[s] = texas::nn::cloneActor(models.actor[s]);
                snap->actor[s]->eval();
            }
            // 快照同时进 recent（滚动）和 archive（几何间隔）。recent 让对手记得
            // 自己最近什么样，archive 防止策略跑偏后再也遇不到早期的自己。
            pool.push(snap, recentCap, archiveCap);

            saveModels(models.actor, models.critic, outDir);
            for (int s = 0; s < 2; ++s) learners[s]->saveOptimizers(outDir);
        }
    }

    saveModels(models.actor, models.critic, outDir);
    for (int s = 0; s < 2; ++s) learners[s]->saveOptimizers(outDir);
    std::printf("models saved to %s\n", outDir.c_str());
    return 0;
}

// ---------------------------------------------------------------- eval

int cmdEval(int argc, char** argv) {
    const std::string dir = argStr(argc, argv, "--resume", "ckpt");
    const std::string vs = argStr(argc, argv, "--vs", "rule");
    const int hands = argInt(argc, argv, "--hands", 2000);
    const int seed = argInt(argc, argv, "--seed", 90001);

    texas::nn::ModelConfig mc;
    texas::nn::Actor actors[2];
    texas::nn::Critic critics[2];
    for (int s = 0; s < 2; ++s) {
        actors[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        critics[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
    }
    if (!loadModels(actors, critics, dir)) {
        std::printf("cannot load model from %s\n", dir.c_str());
        return 1;
    }

    texas::EvalResult r;
    if (vs == "random") {
        r = texas::evalVsRandom(actors, hands, static_cast<uint64_t>(seed));
    } else if (vs == "self") {
        r = texas::evalSelfPlay(actors, hands, static_cast<uint64_t>(seed));
    } else {
        r = texas::evalVsRule(actors, hands, static_cast<uint64_t>(seed));
    }
    std::printf("%s: %d 手, %+0.3f ± %.3f 筹码/手, %+0.1f bb/100\n", vs.c_str(), r.hands,
                r.chipsPerHand, r.stderr, r.bbPer100);
    return 0;
}

// ---------------------------------------------------------------- probe

// 打印某个具体局面上各动作的平均策略概率。用来看"学到的策略像不像会打牌"，
// 但注意：策略形状不是判据，对打胜率才是（见 README）。
void probeOne(const texas::nn::Actor& actor, const char* label, const texas::HUState& st) {
    texas::LegalMove lm[16];
    const int n = texas::legalMoves(st, lm);
    if (n <= 0) {
        std::printf("  %-28s (无合法动作)\n", label);
        return;
    }
    const int64_t B = 1;
    torch::Tensor nodeT = torch::zeros({B, texas::kNodeInput}, torch::kFloat32);
    torch::Tensor seqT = torch::zeros({B, texas::kHistSteps, texas::kHistDim}, torch::kFloat32);
    torch::Tensor featT = torch::zeros({n, texas::kActionInput}, torch::kFloat32);
    torch::Tensor sampleT = torch::zeros({n}, torch::kLong);
    torch::Tensor idT = torch::zeros({n}, torch::kLong);

    texas::Obs obs;
    texas::encodeObservation(st, obs, 256);
    std::copy(obs.node.begin(), obs.node.end(), nodeT.data_ptr<float>());
    std::copy(obs.seq.begin(), obs.seq.end(), seqT.data_ptr<float>());
    for (int i = 0; i < n; ++i) {
        std::array<float, texas::kActionInput> af;
        texas::encodeActionFeature(st, lm[i], af);
        std::copy(af.begin(), af.end(), featT.data_ptr<float>() + i * texas::kActionInput);
        sampleT.data_ptr<int64_t>()[i] = 0;
        idT.data_ptr<int64_t>()[i] = lm[i].slot;
    }

    torch::NoGradGuard ng;
    torch::Tensor logits = actor->forward(nodeT, seqT, featT, sampleT, idT);
    const float* lp = logits.data_ptr<float>();
    float probs[texas::kNumActions];
    texas::ppo::maskedSoftmax(lp, texas::kNumActions, probs);

    std::printf("  %-24s [str=%.2f eq=%.2f]", label,
                double(obs.node[texas::kHoleDim + texas::kBoardDim + 14]),
                double(obs.equityPerfect));
    char buf[32];
    for (int i = 0; i < n; ++i) {
        const texas::Move& mv = lm[i].mv;
        if (lm[i].slot == texas::kSlotFold) std::snprintf(buf, sizeof buf, "fold");
        else if (lm[i].slot == texas::kSlotCheckCall) std::snprintf(buf, sizeof buf, "call");
        else if (lm[i].slot == texas::kSlotAllIn) std::snprintf(buf, sizeof buf, "jam%d", int(mv.to));
        else std::snprintf(buf, sizeof buf, "r%d", int(mv.to));
        std::printf(" %s=%.2f", buf, probs[lm[i].slot]);
    }
    std::printf("\n");
}

int cmdProbe(int argc, char** argv) {
    const std::string dir = argStr(argc, argv, "--resume", "ckpt");
    texas::nn::ModelConfig mc;
    texas::nn::Actor actors[2];
    texas::nn::Critic critics[2];
    for (int s = 0; s < 2; ++s) {
        actors[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        critics[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
    }
    if (!loadModels(actors, critics, dir)) {
        std::printf("cannot load model from %s\n", dir.c_str());
        return 1;
    }

    auto C = [](const char* t) { return texas::parseCard(t); };
    struct Case {
        const char* label;
        const char* a0;
        const char* b0;
        const char* a1;
        const char* b1;
        const char* board[5];
        int boardN;
    };
    const Case cases[] = {
        {"AA 翻牌前(先行动)", "Ah", "Ad", "7c", "2d", {nullptr, nullptr, nullptr, nullptr, nullptr}, 0},
        {"KK 翻牌前(先行动)", "Kh", "Kd", "7c", "2d", {nullptr, nullptr, nullptr, nullptr, nullptr}, 0},
        {"AKs 翻牌前(先行动)", "Ah", "Kh", "7c", "2d", {nullptr, nullptr, nullptr, nullptr, nullptr}, 0},
        {"72o 翻牌前(先行动)", "7h", "2d", "Ac", "Kc", {nullptr, nullptr, nullptr, nullptr, nullptr}, 0},
        {"AA 在 K72 面(先行动)", "Ah", "Ad", "Kc", "Qd", {"Kd", "7s", "2c", nullptr, nullptr}, 3},
        {"72o 在 K72 面(先行动)", "7h", "2d", "Ac", "Qc", {"Kd", "7s", "2c", nullptr, nullptr}, 3},
    };

    std::printf("平均策略（动作后金额，jam=全下）:\n");
    for (const Case& c : cases) {
        texas::Card h0[2] = {C(c.a0), C(c.b0)};
        texas::Card h1[2] = {C(c.a1), C(c.b1)};
        texas::Card board[5] = {0, 0, 0, 0, 0};
        for (int i = 0; i < 5; ++i) {
            if (c.board[i]) board[i] = C(c.board[i]);
        }
        texas::HUState st = texas::makeState(h0, h1, board, c.boardN);
        probeOne(actors[0], c.label, st);
    }
    return 0;
}

// ---------------------------------------------------------------- sanity

// 端到端体检：把优势换成一个已知与牌力强相关的合成信号，看 actor 能不能学会
// "强牌加注、弱牌弃牌"。如果连这个都学不会，问题在 actor/PPO 链路；如果学得会，
// 说明链路没问题，真实训练里的困难来自优势的噪声（下注树的方差）。
int cmdSanity(int argc, char** argv) {
    const int steps = argInt(argc, argv, "--steps", 300);
    texas::nn::ModelConfig mc;
    texas::ModelSet models;
    for (int s = 0; s < 2; ++s) {
        models.actor[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        models.critic[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
    }
    texas::RolloutConfig rc;
    rc.games = 256;
    rc.seed = 4242;
    rc.historicalProb = 0.0f;
    rc.ruleProb = 0.0f;
    rc.randomProb = 0.0f;
    std::array<std::vector<texas::Transition>, 2> tr;
    texas::RolloutStats st;
    texas::collectRollout(models, nullptr, rc, tr, st);

    // 合成优势：强牌加注给正优势、弱牌弃牌给正优势。
    // 关键是优势必须"依赖动作"，否则每个动作被同等放大，政策学不到偏好。
    for (int s = 0; s < 2; ++s) {
        for (auto& t : tr[s]) {
            const float strength = t.node[texas::kHoleDim + texas::kBoardDim + 14];
            const float centered = 2.0f * (strength - 0.5f);
            float adv = 0.0f;
            for (const auto& a : t.actions) {
                if (a.slot != t.action) continue;
                if (a.mv.kind == texas::MK_RAISE || a.mv.kind == texas::MK_ALLIN) adv = centered;
                else if (a.mv.kind == texas::MK_FOLD) adv = -centered;
            }
            t.adv = adv;
            t.ret = adv;
        }
    }
    std::printf("样本 %zu / %zu，合成优势 |adv| 均值 ≈ 0.5\n", tr[0].size(), tr[1].size());

    texas::ppo::Config pcfg;
    pcfg.lr = 1e-3f;
    texas::Learner learner(models.actor[0], models.critic[0], pcfg, torch::kCPU);
    texas::BatchTensors batch;
    texas::buildBatch(tr[0], batch);
    for (int u = 0; u < steps; ++u) learner.update(batch);

    std::printf("训练 %d 轮合成优势后：\n", steps);
    auto C = [](const char* t) { return texas::parseCard(t); };
    const char* hands[][2] = {{"Ah", "Ad"}, {"7h", "2d"}, {"Kh", "Kd"}, {"8c", "5d"}};
    const char* names[] = {"AA", "72o", "KK", "85o"};
    for (int i = 0; i < 4; ++i) {
        texas::Card h0[2] = {C(hands[i][0]), C(hands[i][1])};
        texas::Card h1[2] = {C("Qc"), C("Jc")};
        texas::HUState state = texas::makeState(h0, h1, nullptr, 0);
        probeOne(models.actor[0], names[i], state);
    }
    return 0;
}

// ---------------------------------------------------------------- critfit

// 单独验证 critic 的拟合能力：采一批自对局数据，然后把价值网在这批固定数据上
// 反复训练。如果 RMSE 降不下去，问题在价值网结构或优化；如果能降下去，问题
// 就在"每次只有 2000 个新样本、目标还在漂移"的训练循环上。
int cmdCritFit(int argc, char** argv) {
    const int steps = argInt(argc, argv, "--steps", 1500);
    texas::nn::ModelConfig mc;
    mc.hidden = argInt(argc, argv, "--hidden", 256);
    mc.lstmHidden = argInt(argc, argv, "--lstm", 128);

    texas::ModelSet models;
    for (int s = 0; s < 2; ++s) {
        models.actor[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        models.critic[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
    }
    texas::RolloutConfig rc;
    rc.games = argInt(argc, argv, "--games", 256);
    rc.seed = 42;
    rc.historicalProb = 0.0f;
    rc.ruleProb = 0.0f;
    rc.randomProb = 0.0f;
    std::array<std::vector<texas::Transition>, 2> tr;
    texas::RolloutStats st;
    texas::collectRollout(models, nullptr, rc, tr, st);
    std::printf("transitions: %zu / %zu\n", tr[0].size(), tr[1].size());
    texas::ppo::assignEpisodeReturns(tr[0], 1.0f, 0.95f);
    texas::BatchTensors b;
    texas::buildBatch(tr[0], b);
    std::printf("ret mean %.4f std %.4f | oldValue mean %.4f std %.4f\n",
                b.ret.mean().item<double>(), b.ret.std().item<double>(),
                b.oldValue.mean().item<double>(), b.oldValue.std().item<double>());

    auto opt = torch::optim::Adam(models.critic[0]->parameters(), torch::optim::AdamOptions(1e-3));
    for (int it = 1; it <= steps; ++it) {
        opt.zero_grad();
        auto v = models.critic[0]->forward(b.node, b.seq, b.extra).squeeze(-1);
        auto loss = 0.5 * (v - b.ret).pow(2).mean();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(models.critic[0]->parameters(), 5.0);
        opt.step();
        if (it % 100 == 0) {
            const double mse = loss.item<double>() * 2.0;
            std::printf("  step %4d  rmse %.4f  predStd %.4f\n", it, std::sqrt(mse),
                        v.detach().std().item<double>());
        }
    }
    return 0;
}

// ---------------------------------------------------------------- bench

int cmdBench(int argc, char** argv) {
    const int hands = argInt(argc, argv, "--hands", 2000);
    texas::nn::ModelConfig mc;
    texas::nn::Actor actors[2];
    for (int s = 0; s < 2; ++s) {
        actors[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
    }
    const texas::EvalResult r = texas::evalVsRule(actors, hands, 4242);
    std::printf("未训练策略 vs 规则机器人: %+0.3f ± %.3f 筹码/手 (%+0.1f bb/100)\n",
                r.chipsPerHand, r.stderr, r.bbPer100);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string cmd = argc > 1 ? argv[1] : "help";
    if (cmd == "train") return cmdTrain(argc, argv);
    if (cmd == "eval") return cmdEval(argc, argv);
    if (cmd == "probe") return cmdProbe(argc, argv);
    if (cmd == "sanity") return cmdSanity(argc, argv);
    if (cmd == "critfit") return cmdCritFit(argc, argv);
    if (cmd == "bench") return cmdBench(argc, argv);
    usage();
    return 0;
}
