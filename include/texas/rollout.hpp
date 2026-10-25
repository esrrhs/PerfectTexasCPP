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

// 两级历史池，对应 PerfectDouCPP 的 HistoricalPool：
//
//   recent  —— 最近 N 个快照，滚动淘汰。管"战术演化的最近几步"，
//              让对手还记得自己上一分钟长什么样。
//   archive —— 几何间隔保留的旧快照，永不整体丢弃。管"灾难性遗忘"：
//              只有 recent 时，池子永远只覆盖最近 N 轮，策略一旦跑偏就再也
//              遇不到早期的自己，很容易在同一个坑里反复振荡。
//
// archive 满了不是从头丢，而是**隔一个丢一个**——保留的间隔翻倍，
// 于是越老的快照越稀疏、但永不消失（1,2,4,8,… 的几何间隔）。
struct HistoricalPool {
    std::vector<std::shared_ptr<Snapshot>> recent;
    std::vector<std::shared_ptr<Snapshot>> archive;

    bool empty() const { return recent.empty() && archive.empty(); }
    size_t size() const { return recent.size() + archive.size(); }

    // tier: 0 = recent, 1 = archive
    const Snapshot* at(int tier, int index) const;

    // 推入一个新快照并维护两个队列的容量。
    void push(const std::shared_ptr<Snapshot>& snap, int recentCap, int archiveCap);
};

struct ModelSet {
    nn::Actor actor[2];
    nn::Critic critic[2];
};

struct RolloutConfig {
    int games = 256;
    uint64_t seed = 20240518ULL;
    // 联赛配比。三个概率都是"**这一局**有一个座位是这类对手"的概率，
    // 另一个座位一定是当前策略（见 collectRollout 的按局分配）。
    float historicalProb = 0.30f;  // 历史快照
    float ruleProb = 0.15f;        // 规则机器人
    float randomProb = 0.05f;      // 均匀随机
    float archiveProb = 0.30f;     // 抽历史对手时，抽 archive 而不是 recent 的比例
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
    // 本批里各类对手实际出现的手数（按局计，不含被强制的评估局）
    int vsLatest = 0;
    int vsRecent = 0;
    int vsArchive = 0;
    int vsRule = 0;
    int vsRandom = 0;
};

void collectRollout(const ModelSet& latest, const HistoricalPool* pool, const RolloutConfig& cfg,
                    std::array<std::vector<Transition>, 2>& out, RolloutStats& stats);

// 从 logits 里按合法动作采样，返回 (slot, log 概率)。
std::pair<int, float> sampleAction(const float* logits, const std::vector<ActionOption>& acts,
                                   Rng& rng);

// 规则机器人：牌力百分位 + 底池赔率的启发式。作为评估锚点与联赛对手。
Move ruleAgentMove(const HUState& s, Rng& rng);

}  // namespace texas
