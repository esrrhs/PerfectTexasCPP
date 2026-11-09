// R10 第 1 步离线数据集：自我对局 + 每个决策点的「对手范围」真值。
//
// 每打出一个动作，就在**动作前状态**上枚举全部合法对手底牌组合，用该座位
// 策略在该动作槽上的概率当似然，乘进这个座位的运行后验；发牌/推进后，若
// 牌局未结束，就把「下一个行动者所面对的状态」编码成上下文，配上对手
// （= 1 - 下一个行动者）的后验作为标签写出一条样本。
//
// 标签口径与 rollout.cpp 的 rangeEquityFor 完全一致：似然前向固定喂
// rangeEq=0.5，组合空间排除「已知底牌 + 公共牌」。区别只是不抽样，而是
// 穷举全部组合（翻前 1225、翻牌 1081、转牌 1035、河牌 990）。
//
// 用法：
//   ptx_range_data --resume DIR --games N --seed S --out FILE [--temp 1.0]

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "texas/features.hpp"
#include "texas/hu.hpp"
#include "texas/nn.hpp"
#include "texas/ppo.hpp"
#include "texas/rollout.hpp"
#include "texas/strength.hpp"

using namespace texas;

namespace {

constexpr int kCtxDim = kNodeInput + kSeqInput;  // 532
constexpr int kNClass = StrengthModel::kPreflopClasses;
constexpr int kBatch = 512;

struct Combo {
    int a = 0;
    int b = 0;
    int idx = 0;   // 0..1325
    int cls = 0;   // 169 类
};

std::vector<Combo> gCombos;

void buildCombos() {
    gCombos.clear();
    gCombos.reserve(1326);
    for (int a = 0; a < kNumCards; ++a) {
        for (int b = a + 1; b < kNumCards; ++b) {
            Combo c;
            c.a = a;
            c.b = b;
            c.idx = int(gCombos.size());
            c.cls = StrengthModel::preflopClass(static_cast<Card>(a), static_cast<Card>(b));
            gCombos.push_back(c);
        }
    }
}

// 枚举「未被 hero 底牌 + 公共牌挡住」的对手组合。
std::vector<Combo> availableCombos(const HUState& s, int heroSeat) {
    bool used[kNumCards] = {};
    used[s.hole[heroSeat][0]] = true;
    used[s.hole[heroSeat][1]] = true;
    for (int i = 0; i < s.boardN && i < 5; ++i) used[s.board[i]] = true;
    std::vector<Combo> out;
    out.reserve(1225);
    for (const Combo& c : gCombos) {
        if (used[c.a] || used[c.b]) continue;
        out.push_back(c);
    }
    return out;
}

void encodeContext(const HUState& s, std::vector<float>& ctx) {
    Obs obs;
    encodeObservation(s, obs, 0.5f);
    // 去掉一切能泄露「行动者自己底牌」的输入：底牌 104 维 + 标量里的
    // 牌力百分位 / 听牌出路 / 翻前类号。剩下的全是公共信息。
    for (int i = 0; i < kHoleDim; ++i) obs.node[i] = 0.0f;
    float* x = obs.node.data() + kHoleDim + kBoardDim;
    x[14] = 0.0f;  // handStrength
    x[15] = 0.0f;  // outs
    x[21] = 0.0f;  // preflop class
    ctx.assign(kCtxDim, 0.0f);
    std::copy(obs.node.begin(), obs.node.end(), ctx.begin());
    std::copy(obs.seq.begin(), obs.seq.end(), ctx.begin() + kNodeInput);
}

// 对一个动作前状态枚举全部可用组合，返回每个组合在 slot 上的策略概率。
std::vector<float> slotLikelihood(const HUState& pre, int seat, int slot,
                                  const nn::Actor& actor, float temp) {
    LegalMove lm[kNumActions];
    const int n = legalMoves(pre, lm);
    std::vector<float> like;
    if (n <= 0) return like;

    std::vector<ActionOption> opts(static_cast<size_t>(n));
    for (int k = 0; k < n; ++k) {
        opts[static_cast<size_t>(k)].slot = lm[k].slot;
        opts[static_cast<size_t>(k)].mv = lm[k].mv;
        opts[static_cast<size_t>(k)].hi = lm[k].toHi;
        encodeActionFeature(pre, lm[k], opts[static_cast<size_t>(k)].feat);
    }

    const std::vector<Combo> avail = availableCombos(pre, 1 - seat);
    like.assign(avail.size(), 0.0f);
    if (avail.empty()) return like;

    std::vector<float> node(avail.size() * kNodeInput);
    std::vector<float> seq(avail.size() * kSeqInput);
    for (size_t m = 0; m < avail.size(); ++m) {
        HUState st = pre;
        st.hole[seat][0] = static_cast<uint8_t>(avail[m].a);
        st.hole[seat][1] = static_cast<uint8_t>(avail[m].b);
        Obs obs;
        encodeObservation(st, obs, 0.5f);
        std::copy(obs.node.begin(), obs.node.end(), node.begin() + m * kNodeInput);
        std::copy(obs.seq.begin(), obs.seq.end(), seq.begin() + m * kSeqInput);
    }

    torch::NoGradGuard ng;
    for (size_t base = 0; base < avail.size(); base += kBatch) {
        const size_t B = std::min<size_t>(kBatch, avail.size() - base);
        const int64_t L = int64_t(B) * n;
        auto nodeT = torch::from_blob(node.data() + base * kNodeInput,
                                      {int64_t(B), kNodeInput},
                                      torch::TensorOptions().dtype(torch::kFloat32)).clone();
        auto seqT = torch::from_blob(seq.data() + base * kSeqInput,
                                     {int64_t(B), kHistSteps, kHistDim},
                                     torch::TensorOptions().dtype(torch::kFloat32)).clone();
        auto featT = torch::empty({L, kActionInput}, torch::TensorOptions().dtype(torch::kFloat32));
        auto sampleT = torch::empty({L}, torch::TensorOptions().dtype(torch::kLong));
        auto idT = torch::empty({L}, torch::TensorOptions().dtype(torch::kLong));
        auto rqT = torch::full({int64_t(B)}, 0.5f, torch::TensorOptions().dtype(torch::kFloat32));
        float* pf = featT.data_ptr<float>();
        int64_t* ps = sampleT.data_ptr<int64_t>();
        int64_t* pi = idT.data_ptr<int64_t>();
        for (size_t m = 0; m < B; ++m) {
            for (int k = 0; k < n; ++k) {
                const int64_t row = int64_t(m) * n + k;
                std::copy(opts[static_cast<size_t>(k)].feat.begin(),
                          opts[static_cast<size_t>(k)].feat.end(), pf + row * kActionInput);
                ps[row] = int64_t(m);
                pi[row] = int64_t(lm[k].slot);
            }
        }
        torch::Tensor logits = actor->forward(nodeT, seqT, featT, sampleT, idT, rqT);
        const float* lp = logits.data_ptr<float>();
        for (size_t m = 0; m < B; ++m) {
            float probs[kNumActions];
            const float* row = lp + m * kNumActions;
            float scaled[kNumActions];
            for (int k = 0; k < kNumActions; ++k) scaled[k] = row[k] / temp;
            ppo::maskedSoftmax(scaled, kNumActions, probs);
            like[base + m] = std::max(1e-9f, probs[slot]);
        }
    }
    return like;
}

struct Rec {
    uint8_t street = 0;
    uint8_t toAct = 0;
    uint8_t targetSeat = 0;
    uint8_t slot = 0;
    uint8_t trueClass = 0;
    uint8_t boardN = 0;
    uint8_t board[5] = {};
    uint8_t oppHole[2] = {};
    int32_t pot = 0;
    int32_t toCallAmt = 0;
    int32_t effStack = 0;
    float betRatio = 0.0f;  // 刚打出的动作：增量 / 动作前底池
    uint64_t gameSeed = 0;
    int32_t gameIdx = 0;
    int32_t decisionIdx = 0;
    std::array<float, kCtxDim> ctx{};
    std::array<float, kNClass> strength{};
    std::array<float, kNClass> target{};
};

bool writeHeader(std::FILE* fp, int count) {
    char magic[8] = {'P', 'T', 'X', 'R', 'G', '1', 0, 0};
    int32_t ver = 1;
    int32_t ctxDim = kCtxDim;
    int32_t nClass = kNClass;
    return std::fwrite(magic, 1, 8, fp) == 8 && std::fwrite(&ver, 4, 1, fp) == 1 &&
           std::fwrite(&ctxDim, 4, 1, fp) == 1 && std::fwrite(&nClass, 4, 1, fp) == 1 &&
           std::fwrite(&count, 4, 1, fp) == 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir;
    std::string out;
    int games = 500;
    uint64_t seed = 20261109ULL;
    float temp = 1.0f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--resume") dir = next();
        else if (a == "--out") out = next();
        else if (a == "--games") games = std::atoi(next().c_str());
        else if (a == "--seed") seed = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--temp") temp = float(std::atof(next().c_str()));
    }
    if (dir.empty() || out.empty()) {
        std::printf("用法: ptx_range_data --resume DIR --games N --seed S --out FILE [--temp F]\n");
        return 2;
    }

    nn::Actor actor[2] = {nn::Actor(std::make_shared<nn::ActorImpl>(nn::ModelConfig{})),
                          nn::Actor(std::make_shared<nn::ActorImpl>(nn::ModelConfig{}))};
    for (int s = 0; s < 2; ++s) {
        actor[s]->eval();
        if (!actor[s]->loadFrom(dir + "/actor" + std::to_string(s) + ".pt")) {
            std::printf("加载失败: %s/actor%d.pt\n", dir.c_str(), s);
            return 1;
        }
    }
    torch::set_num_threads(8);
    buildCombos();

    std::FILE* fp = std::fopen(out.c_str(), "wb");
    if (!fp) {
        std::printf("无法写入 %s\n", out.c_str());
        return 1;
    }
    if (!writeHeader(fp, 0)) return 1;

    Rng rng(seed);
    int written = 0;
    long long decisions = 0;
    std::array<std::array<double, 1326>, 2> w;  // 每座位运行后验
    std::array<bool, 2> seen = {false, false};

    for (int g = 0; g < games; ++g) {
        HUState s = dealNewHand(rng);
        w[0].fill(0.0);
        w[1].fill(0.0);
        seen[0] = seen[1] = false;
        int decisionIdx = 0;

        while (!s.finished && decisionIdx < 512) {
            ++decisionIdx;
            ++decisions;
            const int seat = s.toAct;
            const int opp = 1 - seat;
            const HUState pre = s;
            const Move mv = policyMove(&actor[0], &actor[1], s, rng, temp, 0.5f);
            const int slot = slotOfMove(pre, mv);
            if (slot < 0) break;
            const int potBefore = pre.pot();
            const int delta = std::max<int>(0, int(mv.to) - int(pre.contrib[seat]));

            applyMove(s, mv, rng);
            if (s.finished) break;

            // 似然：该座位在动作前状态上对各组合打这个槽的概率。
            const std::vector<Combo> avail = availableCombos(pre, opp);
            const std::vector<float> like = slotLikelihood(pre, seat, slot, actor[seat], temp);
            if (!like.empty()) {
                if (!seen[seat]) {
                    for (const Combo& c : avail) w[seat][c.idx] = 1.0;
                    seen[seat] = true;
                }
                for (size_t i = 0; i < avail.size(); ++i) {
                    w[seat][avail[i].idx] *= std::max(1e-9, double(like[i]));
                }
                double mx = 0.0;
                for (const Combo& c : avail) mx = std::max(mx, w[seat][c.idx]);
                if (mx > 0.0) {
                    for (const Combo& c : avail) w[seat][c.idx] /= mx;
                }
            }

            // 记录：下一个行动者面对的对手（= 1 - s.toAct）的范围。
            const int targetSeat = 1 - s.toAct;
            if (!seen[targetSeat]) continue;
            const std::vector<Combo> cur = availableCombos(s, 1 - targetSeat);
            double mass = 0.0;
            for (const Combo& c : cur) mass += w[targetSeat][c.idx];
            if (!(mass > 0.0)) continue;

            Rec rec;
            rec.street = s.street;
            rec.toAct = s.toAct;
            rec.targetSeat = static_cast<uint8_t>(targetSeat);
            rec.slot = static_cast<uint8_t>(slot);
            rec.boardN = s.boardN;
            for (int i = 0; i < s.boardN && i < 5; ++i) rec.board[i] = s.board[i];
            rec.oppHole[0] = s.hole[targetSeat][0];
            rec.oppHole[1] = s.hole[targetSeat][1];
            rec.trueClass = static_cast<uint8_t>(StrengthModel::preflopClass(
                static_cast<Card>(s.hole[targetSeat][0]),
                static_cast<Card>(s.hole[targetSeat][1])));
            rec.pot = s.pot();
            rec.toCallAmt = toCall(s, s.toAct);
            rec.effStack = std::min<int>(s.stack[0], s.stack[1]);
            rec.betRatio = (potBefore > 0) ? float(delta) / float(potBefore) : 0.0f;
            rec.gameSeed = seed + static_cast<uint64_t>(g);
            rec.gameIdx = g;
            rec.decisionIdx = decisionIdx;

            std::vector<float> ctx;
            encodeContext(s, ctx);
            std::copy(ctx.begin(), ctx.end(), rec.ctx.begin());

            std::array<double, kNClass> ssum{};
            std::array<double, kNClass> scnt{};
            for (const Combo& c : cur) {
                Card hole[2] = {static_cast<Card>(c.a), static_cast<Card>(c.b)};
                Card board[5] = {0, 0, 0, 0, 0};
                for (int i = 0; i < s.boardN && i < 5; ++i) board[i] = static_cast<Card>(s.board[i]);
                const float st = handStrength(hole, board, s.boardN);
                ssum[static_cast<size_t>(c.cls)] += st;
                scnt[static_cast<size_t>(c.cls)] += 1.0;
                rec.target[static_cast<size_t>(c.cls)] +=
                    static_cast<float>(w[targetSeat][c.idx] / mass);
            }
            // 归一化到 1（可用组合按类求和已有质量，浮点误差再校一次）。
            double tsum = 0.0;
            for (int c = 0; c < kNClass; ++c) {
                rec.strength[static_cast<size_t>(c)] =
                    (scnt[static_cast<size_t>(c)] > 0.0)
                        ? float(ssum[static_cast<size_t>(c)] / scnt[static_cast<size_t>(c)])
                        : 0.5f;
                tsum += rec.target[static_cast<size_t>(c)];
            }
            if (tsum > 0.0) {
                for (int c = 0; c < kNClass; ++c) {
                    rec.target[static_cast<size_t>(c)] = float(rec.target[static_cast<size_t>(c)] / tsum);
                }
            }

            std::fwrite(&rec.street, 1, 1, fp);
            std::fwrite(&rec.toAct, 1, 1, fp);
            std::fwrite(&rec.targetSeat, 1, 1, fp);
            std::fwrite(&rec.slot, 1, 1, fp);
            std::fwrite(&rec.trueClass, 1, 1, fp);
            std::fwrite(&rec.boardN, 1, 1, fp);
            std::fwrite(rec.board, 1, 5, fp);
            std::fwrite(rec.oppHole, 1, 2, fp);
            std::fwrite(&rec.pot, 4, 1, fp);
            std::fwrite(&rec.toCallAmt, 4, 1, fp);
            std::fwrite(&rec.effStack, 4, 1, fp);
            std::fwrite(&rec.betRatio, 4, 1, fp);
            std::fwrite(&rec.gameSeed, 8, 1, fp);
            std::fwrite(&rec.gameIdx, 4, 1, fp);
            std::fwrite(&rec.decisionIdx, 4, 1, fp);
            std::fwrite(rec.ctx.data(), 4, kCtxDim, fp);
            std::fwrite(rec.strength.data(), 4, kNClass, fp);
            std::fwrite(rec.target.data(), 4, kNClass, fp);
            ++written;
        }
        if ((g + 1) % 200 == 0) {
            std::printf("games %d/%d  decisions %lld  records %d\n", g + 1, games, decisions,
                        written);
            std::fflush(stdout);
        }
    }

    std::fseek(fp, 20, SEEK_SET);  // magic 8 + ver 4 + ctxDim 4 + nClass 4
    std::fwrite(&written, 4, 1, fp);
    std::fclose(fp);
    std::printf("写出 %s: %d 条（%lld 个决策，%d 局）\n", out.c_str(), written, decisions, games);
    return 0;
}
