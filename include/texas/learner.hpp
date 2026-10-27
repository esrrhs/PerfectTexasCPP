#pragma once

// PPO 学习器：策略裁剪损失 + 价值回归 + 熵正则 + KL 早停。
// 两个优化器（actor / critic）独立，梯度裁剪阈值也分开——critic 的梯度在
// 筹码量级上，用策略的 0.5 会把价值网彻底压死。

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "texas/nn.hpp"
#include "texas/ppo.hpp"
#include "texas/rollout.hpp"

namespace texas {

struct BatchTensors {
    torch::Tensor node;        // [B, kNodeInput]
    torch::Tensor seq;         // [B, kHistSteps, kHistDim]
    torch::Tensor extra;       // [B, kExtraInput]
    torch::Tensor actFeat;     // [L, kActionInput]
    torch::Tensor actSample;   // [L]
    torch::Tensor actId;       // [L]
    torch::Tensor actions;     // [B]
    torch::Tensor oldLogp;     // [B]
    torch::Tensor adv;         // [B]
    torch::Tensor ret;         // [B]
    torch::Tensor oldValue;    // [B]
    torch::Tensor teacherQ;    // [B, kNumActions] 脚本教师动作价值
    torch::Tensor teacherValid;  // [B] 1 = 该样本有教师标注
    torch::Tensor auxLabel;    // [B] 开天眼真实胜率（范围胜率信念头的回归标签）
    torch::Tensor ruleSlot;    // [B] 规则教师目标槽位，-1 = 无
};

// 把若干条 transition 流拼成一批张量。isTerminal 的样本照常参与训练。
void buildBatch(const std::vector<Transition>& tr, BatchTensors& batch, float advNormEps = 1e-6f);

// 教师回放用的紧凑样本：只保留 actor 输入、合法动作和教师 Q。
// 动作特征也存下来，蒸馏时不需要重新构造状态。
struct TeacherSample {
    std::array<float, kNodeInput> node{};
    std::array<float, kSeqInput> seq{};
    std::array<float, kNumActions> q{};
    std::vector<ActionOption> actions;
};

// 价值回放的紧凑样本：价值网输入 + 当时的回报目标。
// 回报是那条轨迹的 GAE 回报，作为回归目标仍然有效；跨轮回放让 critic 在
// 更宽的状态分布上多跑几轮，压住"只拟合当前这一批"的过拟合。实测价值网
// 与回报的相关系数长期只有 0.2~0.4，优势估计因此很吵（见 README）。
struct CriticSample {
    std::array<float, kNodeInput> node{};
    std::array<float, kSeqInput> seq{};
    std::array<float, kExtraInput> extra{};
    float ret = 0.0f;
};

class Learner {
public:
    Learner(nn::Actor actor, nn::Critic critic, const ppo::Config& cfg, torch::Device device);

    ppo::Stats update(BatchTensors& batch);

    // 共享 actor 训练用：把策略更新与价值更新拆开。共享 actor 时两个座位
    // 共用一个 Adam，所以 actor 部分只能由一个 Learner 调；critic 仍分座位。
    ppo::Stats updateActorOnly(BatchTensors& batch);
    double updateCriticOnly(BatchTensors& batch, int epochs);

    // 教师回放：在最近 N 轮的带标注样本上多跑几步纯蒸馏。
    // 返回平均交叉熵（未乘系数），用于日志。只更新 actor。
    double distill(const std::deque<TeacherSample>& samples, int steps, float coef, float temp,
                   int minibatch);

    // 价值回放：在最近 N 轮的价值样本上只更新 critic。返回平均价值损失。
    // 每轮在 update() 之后调用，相当于给 critic 一个跨轮的小型经验池。
    double updateCriticReplay(const std::deque<CriticSample>& samples, int epochs, int minibatch);

    void setLr(float lr);
    void saveOptimizers(const std::string& dir) const;
    void loadOptimizers(const std::string& dir);

    nn::Actor actor;
    nn::Critic critic;
    // 可选的冻结基座（评估模式、不加梯度）。只用于状态条件信任域的 KL。
    const nn::Actor* anchor = nullptr;
    ppo::Config config;
    torch::Device device;
    std::unique_ptr<torch::optim::Adam> actorOpt;
    std::unique_ptr<torch::optim::Adam> criticOpt;
};

}  // namespace texas
