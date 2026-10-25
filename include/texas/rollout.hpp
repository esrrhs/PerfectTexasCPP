#pragma once

// 自对局采样：一批牌局按座位锁步推进，同一座位所有待决策的牌局做一次批量
// 前向。每个座位一条独立的 transition 流，交给 PPO 的 GAE 用。

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "texas/features.hpp"
#include "texas/hu.hpp"
#include "texas/nn.hpp"

namespace texas {

struct ActionOption {
    int slot = 0;
    Move mv;
    std::array<float, kActionInput> feat{};
};

struct Transition {
    std::array<float, kNodeInput> node{};
    std::array<float, kSeqInput> seq{};
    std::array<float, kExtraInput> extra{};
    std::vector<ActionOption> actions;  // 本节点的全部合法动作
    int action = -1;                    // 采样到的 slot
    int gameId = -1;
    bool terminal = false;
    float logp = 0.0f;
    float value = 0.0f;
    float reward = 0.0f;
    float adv = 0.0f;
    float ret = 0.0f;
};

struct Snapshot {
    nn::Actor actor[2];
    int update = 0;
};

struct HistoricalPool {
    std::vector<std::shared_ptr<Snapshot>> items;
    bool empty() const { return items.empty(); }
};

struct ModelSet {
    nn::Actor actor[2];
    nn::Critic critic[2];
};

struct RolloutConfig {
    int games = 256;
    uint64_t seed = 20240518ULL;
    float historicalProb = 0.25f;  // 对手是历史快照的概率
    float ruleProb = 0.05f;        // 对手是规则机器人的概率
    bool oracleReward = true;      // 开天眼稠密奖励
    float oracleCoef = 0.05f;
    int equityIters = 32;          // 开天眼胜率的抽样次数
    float rewardScale = 1.0f / float(kStartingStack);  // 筹码 -> 起始筹码的倍数
    int forceRuleSeat = -1;        // 评估用：强制某个座位打规则机器人
    int forceRandomSeat = -1;      // 评估用：强制某个座位均匀随机
    std::vector<int>* handScores = nullptr;  // 评估用：逐手记录"被测座位"的净收益
};

struct RolloutStats {
    int games = 0;
    int sbWins = 0;   // 小盲收池（含摊牌）
    int bbWins = 0;   // 大盲收池
    int splits = 0;
    long long sbScore = 0;
    long long bbScore = 0;
    long long decisions = 0;
    long long showdowns = 0;
    long long allinDecisions = 0;
    long long folds = 0;
};

void collectRollout(const ModelSet& latest, const HistoricalPool* pool, const RolloutConfig& cfg,
                    std::array<std::vector<Transition>, 2>& out, RolloutStats& stats);

// 从 logits 里按合法动作采样，返回 (slot, log 概率)。
std::pair<int, float> sampleAction(const float* logits, const std::vector<ActionOption>& acts,
                                   Rng& rng);

// 规则机器人：牌力百分位 + 底池赔率的启发式。作为评估锚点与联赛对手。
Move ruleAgentMove(const HUState& s, Rng& rng);

}  // namespace texas
