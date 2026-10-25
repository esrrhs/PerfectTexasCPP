#pragma once

// 网络结构，与 PerfectDouCPP 同构：
//
// Actor（不完全信息策略网）
//   LSTM(kSeqInput -> lstmHidden) 吃 12 步行为序列
//   node = concat(节点特征, LSTM 末态)            [B, 388 + lstmHidden]
//   joint = concat(node[动作所属样本], 动作特征)   [L, 388 + lstmHidden + 16]
//   MLP: 4 层 ReLU（最后一层 2 倍宽）-> 每个合法动作一个分数
//   scatter 到 [B, 8] 的 logits，非法动作为 -1e9
//
// Critic（开天眼价值网）
//   同一条 LSTM + 节点特征，再拼上对手底牌与完美信息胜率（108 维）
//   4 层 MLP -> 标量价值

#include <torch/torch.h>

#include <string>

#include "texas/features.hpp"

namespace texas {
namespace nn {

struct ModelConfig {
    int64_t hidden = 256;
    int64_t lstmHidden = 128;
};

class ActorImpl : public torch::nn::Module {
public:
    explicit ActorImpl(const ModelConfig& cfg = ModelConfig{});

    // node: [B, kNodeInput]  seq: [B, kHistSteps, kHistDim]
    // actFeat: [L, kActionInput]  actSample: [L]  actId: [L]
    // 返回 logits: [B, kNumActions]
    torch::Tensor forward(const torch::Tensor& node, const torch::Tensor& seq,
                          const torch::Tensor& actFeat, const torch::Tensor& actSample,
                          const torch::Tensor& actId) const;

    void saveTo(const std::string& path) const;
    bool loadFrom(const std::string& path);

    ModelConfig config;
    mutable torch::nn::LSTM lstm{nullptr};
    mutable torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr}, l4{nullptr}, head{nullptr};
};
TORCH_MODULE(Actor);

class CriticImpl : public torch::nn::Module {
public:
    explicit CriticImpl(const ModelConfig& cfg = ModelConfig{});

    // node: [B, kNodeInput]  seq: [B, kHistSteps, kHistDim]  extra: [B, kExtraInput]
    torch::Tensor forward(const torch::Tensor& node, const torch::Tensor& seq,
                          const torch::Tensor& extra) const;

    void saveTo(const std::string& path) const;
    bool loadFrom(const std::string& path);

    ModelConfig config;
    mutable torch::nn::LSTM lstm{nullptr};
    mutable torch::nn::Linear c1{nullptr}, c2{nullptr}, c3{nullptr}, c4{nullptr}, out{nullptr};
};
TORCH_MODULE(Critic);

// 深拷贝一份同结构的网络（多线程自对局每个线程一份，避免共享可变状态）。
Actor cloneActor(const Actor& src);
Critic cloneCritic(const Critic& src);

}  // namespace nn
}  // namespace texas
