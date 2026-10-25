#pragma once

// PPO 学习器：策略裁剪损失 + 价值回归 + 熵正则 + KL 早停。
// 两个优化器（actor / critic）独立，梯度裁剪阈值也分开——critic 的梯度在
// 筹码量级上，用策略的 0.5 会把价值网彻底压死。

#include <memory>
#include <string>

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
};

// 把若干条 transition 流拼成一批张量。isTerminal 的样本照常参与训练。
void buildBatch(const std::vector<Transition>& tr, BatchTensors& batch, float advNormEps = 1e-6f);

class Learner {
public:
    Learner(nn::Actor actor, nn::Critic critic, const ppo::Config& cfg, torch::Device device);

    ppo::Stats update(BatchTensors& batch);

    void setLr(float lr);
    void saveOptimizers(const std::string& dir) const;
    void loadOptimizers(const std::string& dir);

    nn::Actor actor;
    nn::Critic critic;
    ppo::Config config;
    torch::Device device;
    std::unique_ptr<torch::optim::Adam> actorOpt;
    std::unique_ptr<torch::optim::Adam> criticOpt;
};

}  // namespace texas
