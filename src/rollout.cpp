#include "texas/rollout.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <map>

#include "texas/ppo.hpp"

namespace texas {

namespace {

enum RoleType : uint8_t { ROLE_LATEST = 0, ROLE_SNAPSHOT = 1, ROLE_RULE = 2, ROLE_RANDOM = 3 };

struct GameCtx {
    HUState st;
    uint8_t role[2] = {ROLE_LATEST, ROLE_LATEST};
    int snapIdx[2] = {-1, -1};
    int lastIdx[2] = {-1, -1};
    float phi[2] = {0.0f, 0.0f};  // 上次开天眼势能
    uint8_t eqStamp = 255;        // 胜率缓存对应的公共牌张数
    float eq0 = 0.5f;
    bool active = true;
};

// 一条街内底池不会改变胜率（发牌只在街道切换时发生），所以开天眼胜率按
// boardN 缓存。seat 1 的胜率严格等于 1 - seat 0（平局双方各算半胜）。
float equityFor(GameCtx& g, int seat, int iters) {
    if (g.eqStamp != g.st.boardN) {
        g.eq0 = float(perfectEquity(g.st, 0, iters));
        g.eqStamp = g.st.boardN;
    }
    return (seat == 0) ? g.eq0 : 1.0f - g.eq0;
}

// 开天眼势能 Φ = (胜率 × 底池 − 自己已投入) / 起始筹码。
//
// 这是"从此刻起双方都不再下注时的期望净收益"，所以它是一个真正的势函数：
//   - 弃牌不改变底池也不改变投入 → ΔΦ = 0，不会凭空奖励弃牌
//   - 跟注 b：ΔΦ = b·(2·eq − 1)，胜率过半才有正收益
// 早先版本用的是 Φ = 底池 × (2·eq − 1)，少了"自己已投入"这一项，于是对手
// 往底池里下注就会把 Φ 推高 —— 弃一手 60% 胜率的牌也能拿到正奖励。实测的
// 后果是策略塌成"从不弃牌、摊牌率 44% → 80%"。归一化到起始筹码是为了让
// 奖励落在 O(1)：原始筹码量级（±200）会让价值网 MSE 上千，梯度全被它吃掉。
float phiOf(const HUState& s, int seat, float eq) {
    return (float(eq) * float(s.pot()) - float(s.total[seat])) / float(kStartingStack);
}

size_t totalOptions(const std::vector<Transition>& tmp) {
    size_t n = 0;
    for (const auto& t : tmp) n += t.actions.size();
    return n;
}

void writeTensors(const std::vector<Transition>& tmp, torch::Tensor& nodeT, torch::Tensor& seqT,
                  torch::Tensor& extraT, torch::Tensor& actFeatT, torch::Tensor& actSampleT,
                  torch::Tensor& actIdT, bool withExtra) {
    const int64_t B = static_cast<int64_t>(tmp.size());
    const size_t L = totalOptions(tmp);
    nodeT = torch::empty({B, kNodeInput}, torch::kFloat32);
    seqT = torch::empty({B, kHistSteps, kHistDim}, torch::kFloat32);
    extraT = torch::empty({B, withExtra ? kExtraInput : 1}, torch::kFloat32);
    actFeatT = torch::empty({static_cast<int64_t>(L), kActionInput}, torch::kFloat32);
    actSampleT = torch::empty({static_cast<int64_t>(L)}, torch::kLong);
    actIdT = torch::empty({static_cast<int64_t>(L)}, torch::kLong);

    float* pNode = nodeT.data_ptr<float>();
    float* pSeq = seqT.data_ptr<float>();
    float* pExtra = extraT.data_ptr<float>();
    float* pFeat = actFeatT.data_ptr<float>();
    int64_t* pSample = actSampleT.data_ptr<int64_t>();
    int64_t* pId = actIdT.data_ptr<int64_t>();

    int64_t off = 0;
    for (int64_t i = 0; i < B; ++i) {
        const Transition& t = tmp[static_cast<size_t>(i)];
        std::copy(t.node.begin(), t.node.end(), pNode + i * kNodeInput);
        std::copy(t.seq.begin(), t.seq.end(), pSeq + i * kSeqInput);
        if (withExtra) {
            std::copy(t.extra.begin(), t.extra.end(), pExtra + i * kExtraInput);
        }
        for (const ActionOption& a : t.actions) {
            std::copy(a.feat.begin(), a.feat.end(), pFeat + off * kActionInput);
            pSample[off] = i;
            pId[off] = a.slot;
            ++off;
        }
    }
}

}  // namespace

std::pair<int, float> sampleAction(const float* logits, const std::vector<ActionOption>& acts,
                                   Rng& rng) {
    float probs[kNumActions];
    ppo::maskedSoftmax(logits, kNumActions, probs);
    double sum = 0.0;
    for (const ActionOption& a : acts) sum += probs[a.slot];
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (!(sum > 0.0)) {
        const int k = static_cast<int>(rng() % acts.size());
        return {acts[static_cast<size_t>(k)].slot, -std::log(double(acts.size()))};
    }
    double r = u(rng) * sum;
    for (const ActionOption& a : acts) {
        const double p = probs[a.slot];
        if (r < p || &a == &acts.back()) {
            return {a.slot, float(std::log(p / sum))};
        }
        r -= p;
    }
    return {acts.back().slot, float(std::log(probs[acts.back().slot] / sum))};
}

Move ruleAgentMove(const HUState& s, Rng& rng) {
    LegalMove mv[16];
    const int n = legalMoves(s, mv);
    if (n == 0) return Move{};

    const int me = s.toAct;
    Card hole[2] = {static_cast<Card>(s.hole[me][0]), static_cast<Card>(s.hole[me][1])};
    Card board[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < s.boardN && i < 5; ++i) board[i] = static_cast<Card>(s.board[i]);
    const float strength = handStrength(hole, board, s.boardN);
    const int need = toCall(s, me);
    const int pot = s.pot();
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    const float jitter = u(rng);

    auto findSlot = [&](int slot, Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == slot) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };
    auto checkCall = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotCheckCall) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };
    auto fold = [&](Move& out) {
        for (int i = 0; i < n; ++i) {
            if (mv[i].slot == kSlotFold) {
                out = mv[i].mv;
                return true;
            }
        }
        return false;
    };

    Move out{};
    if (need == 0) {
        // 无人下注：按牌力决定下注尺寸，弱牌偶尔诈唬
        const float betThreshold = 0.58f - 0.22f * jitter;
        if (strength > betThreshold) {
            const int want = strength > 0.85f ? slotOfSize(4) : (strength > 0.7f ? slotOfSize(3) : slotOfSize(1));
            if (findSlot(want, out)) return out;
        }
        checkCall(out);
        return out;
    }

    const float odds = float(need) / float(pot + need);
    if (strength > 0.80f && jitter > 0.55f) {
        const int want = strength > 0.93f ? kSlotAllIn : slotOfSize(3);
        if (findSlot(want, out)) return out;
    }
    if (strength > odds + 0.10f && jitter > 0.12f) {
        checkCall(out);
        return out;
    }
    if (strength < odds * 0.55f) {
        if (fold(out)) return out;
    }
    checkCall(out);
    return out;
}

void collectRollout(const ModelSet& latest, const HistoricalPool* pool, const RolloutConfig& cfg,
                    std::array<std::vector<Transition>, 2>& out, RolloutStats& stats) {
    torch::NoGradGuard noGrad;  // 采样只做前向，不需要建图
    Rng rng(cfg.seed);
    const int N = std::max(1, cfg.games);
    std::vector<GameCtx> gs(static_cast<size_t>(N));
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::uniform_real_distribution<double> uu(0.0, 1.0);

    out[0].clear();
    out[1].clear();

    for (int i = 0; i < N; ++i) {
        GameCtx& g = gs[static_cast<size_t>(i)];
        g.st = dealNewHand(rng);
        for (int seat = 0; seat < 2; ++seat) {
            g.role[seat] = ROLE_LATEST;
            if (cfg.forceRuleSeat == seat) {
                g.role[seat] = ROLE_RULE;
                continue;
            }
            if (cfg.forceRandomSeat == seat) {
                g.role[seat] = ROLE_RANDOM;
                continue;
            }
            const float r = u(rng);
            if (pool && !pool->empty() && r < cfg.historicalProb) {
                g.role[seat] = ROLE_SNAPSHOT;
                g.snapIdx[seat] = static_cast<int>(rng() % pool->items.size());
            } else if (r < cfg.historicalProb + cfg.ruleProb) {
                g.role[seat] = ROLE_RULE;
            }
        }
        for (int seat = 0; seat < 2; ++seat) {
            g.phi[seat] = phiOf(g.st, seat, equityFor(g, seat, cfg.equityIters));
        }
        g.active = true;
    }

    int remaining = N;
    auto finishGame = [&](GameCtx& g) {
        for (int p = 0; p < 2; ++p) {
            if (g.lastIdx[p] < 0) continue;
            auto& t = out[static_cast<size_t>(p)][static_cast<size_t>(g.lastIdx[p])];
            t.reward += float(payoff(g.st, p)) * cfg.rewardScale;
            t.terminal = true;
        }
        if (g.st.folded == 255) ++stats.showdowns;
        {
            const int w = (g.st.folded != 255) ? (1 - int(g.st.folded)) : showdownWinner(g.st);
            if (w == 0) ++stats.sbWins;
            else if (w == 1) ++stats.bbWins;
            else ++stats.splits;
        }
        stats.sbScore += payoff(g.st, 0);
        stats.bbScore += payoff(g.st, 1);
        if (cfg.handScores) {
            const int watched = (cfg.forceRuleSeat == 0 || cfg.forceRandomSeat == 0) ? 1 : 0;
            cfg.handScores->push_back(payoff(g.st, watched));
        }
        stats.games += 1;
        g.active = false;
        --remaining;
    };

    auto runBatch = [&](const nn::Actor& actor, const nn::Critic* critic, int seat,
                        const std::vector<int>& idx) {
        if (idx.empty()) return;
        const int64_t B = static_cast<int64_t>(idx.size());
        std::vector<Transition> tmp(static_cast<size_t>(B));

        for (int64_t i = 0; i < B; ++i) {
            GameCtx& g = gs[static_cast<size_t>(idx[static_cast<size_t>(i)])];
            Transition& t = tmp[static_cast<size_t>(i)];
            const float eq = equityFor(g, seat, cfg.equityIters);
            Obs obs;
            encodeObservation(g.st, obs, eq);
            t.node = obs.node;
            t.seq = obs.seq;
            t.extra = obs.extra;

            LegalMove lm[16];
            const int n = legalMoves(g.st, lm);
            t.actions.resize(static_cast<size_t>(n));
            for (int k = 0; k < n; ++k) {
                ActionOption& a = t.actions[static_cast<size_t>(k)];
                a.slot = lm[k].slot;
                a.mv = lm[k].mv;
                encodeActionFeature(g.st, lm[k], a.feat);
            }
            t.gameId = idx[static_cast<size_t>(i)];
        }

        torch::Tensor nodeT, seqT, extraT, actFeatT, actSampleT, actIdT;
        writeTensors(tmp, nodeT, seqT, extraT, actFeatT, actSampleT, actIdT, critic != nullptr);

        torch::Tensor logits = actor->forward(nodeT, seqT, actFeatT, actSampleT, actIdT);
        const float* pLogits = logits.data_ptr<float>();
        torch::Tensor values;
        const float* pValues = nullptr;
        if (critic) {
            values = (*critic)->forward(nodeT, seqT, extraT).squeeze(-1);
            pValues = values.data_ptr<float>();
        }

        for (int64_t i = 0; i < B; ++i) {
            const int gi = idx[static_cast<size_t>(i)];
            GameCtx& g = gs[static_cast<size_t>(gi)];
            Transition& t = tmp[static_cast<size_t>(i)];
            auto [slot, logp] = sampleAction(pLogits + i * kNumActions, t.actions, rng);

            const ActionOption* opt = &t.actions.back();
            for (const ActionOption& a : t.actions) {
                if (a.slot == slot) opt = &a;
            }
            t.action = slot;
            t.logp = logp;
            if (pValues) {
                const float v = pValues[i];
                t.value = std::isfinite(v) ? v : 0.0f;
            }

            applyMove(g.st, opt->mv, rng);
            ++stats.decisions;
            if (opt->mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (opt->mv.kind == MK_FOLD) ++stats.folds;

            if (cfg.oracleReward) {
                const float eq = equityFor(g, seat, cfg.equityIters);
                const float phi = phiOf(g.st, seat, eq);
                t.reward += cfg.oracleCoef * (phi - g.phi[seat]);
                g.phi[seat] = phi;
            }

            const int index = static_cast<int>(out[static_cast<size_t>(seat)].size());
            out[static_cast<size_t>(seat)].push_back(std::move(t));
            g.lastIdx[seat] = index;
            if (g.st.terminal()) finishGame(g);
        }
    };

    auto runRandom = [&](int seat, const std::vector<int>& idx) {
        (void)seat;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            LegalMove lm[16];
            const int n = legalMoves(g.st, lm);
            if (n <= 0) {
                g.active = false;
                --remaining;
                continue;
            }
            const int pick = static_cast<int>(rng() % static_cast<size_t>(n));
            applyMove(g.st, lm[pick].mv, rng);
            if (lm[pick].mv.kind == MK_FOLD) ++stats.folds;
            if (g.st.terminal()) finishGame(g);
        }
    };

    auto runRule = [&](int seat, const std::vector<int>& idx) {
        (void)seat;
        for (int gi : idx) {
            GameCtx& g = gs[static_cast<size_t>(gi)];
            const Move mv = ruleAgentMove(g.st, rng);
            applyMove(g.st, mv, rng);
            if (mv.kind == MK_FOLD) ++stats.folds;
            if (mv.kind == MK_ALLIN) ++stats.allinDecisions;
            if (g.st.terminal()) finishGame(g);
        }
    };

    while (remaining > 0) {
        for (int seat = 0; seat < 2; ++seat) {
            std::vector<int> latestIdx;
            std::map<int, std::vector<int>> snapIdx;
            std::vector<int> ruleIdx;
            std::vector<int> randomIdx;

            for (int i = 0; i < N; ++i) {
                GameCtx& g = gs[static_cast<size_t>(i)];
                if (!g.active || g.st.toAct != seat) continue;
                switch (g.role[seat]) {
                    case ROLE_LATEST: latestIdx.push_back(i); break;
                    case ROLE_SNAPSHOT: snapIdx[g.snapIdx[seat]].push_back(i); break;
                    case ROLE_RANDOM: randomIdx.push_back(i); break;
                    default: ruleIdx.push_back(i); break;
                }
            }

            runBatch(latest.actor[seat], &latest.critic[seat], seat, latestIdx);
            for (auto& [sidx, idx] : snapIdx) {
                if (sidx < 0 || sidx >= static_cast<int>(pool->items.size())) {
                    runBatch(latest.actor[seat], nullptr, seat, idx);
                    continue;
                }
                runBatch(pool->items[static_cast<size_t>(sidx)]->actor[seat], nullptr, seat, idx);
            }
            runRule(seat, ruleIdx);
            runRandom(seat, randomIdx);
        }
    }
}

}  // namespace texas
