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
#include <utility>

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
    // rangeEq: [B] 动作条件化的"我方 vs 对手范围"胜率（rollout 组合级计算）。
    //          未提供时按 0.5 中性值喂入；新支路末层零初始化，所以旧检查点
    //          行为不变。部署/评估路径如果要复现训练时的输入，必须显式传入。
    // 返回 logits: [B, kNumActions]
    torch::Tensor forward(const torch::Tensor& node, const torch::Tensor& seq,
                          const torch::Tensor& actFeat, const torch::Tensor& actSample,
                          const torch::Tensor& actId,
                          const torch::Tensor& rangeEq = torch::Tensor()) const;

    // 训练用：同时返回策略 logits 与"对手范围胜率"预测 eqHat [B]。
    // eqHat 只用不完全信息（node + 序列）估计，标签是开天眼的真实胜率；
    // 另有一条残差支路把 eqHat 与动作特征结合，直接修正每个动作的分数。
    std::pair<torch::Tensor, torch::Tensor> forwardWithAux(
        const torch::Tensor& node, const torch::Tensor& seq, const torch::Tensor& actFeat,
        const torch::Tensor& actSample, const torch::Tensor& actId,
        const torch::Tensor& rangeEq = torch::Tensor()) const;

    // 对手建模头的三个输出：logits [B, kNumActions]、eqHat [B]、oppHat [B]。
    // oppHat 同样只用不完全信息，回归标签是"对手当前牌力百分位"——这是策略
    // 在翻后大池最缺的量（我的手牌 vs 对手范围），不是我自己牌力的重复。
    struct AuxOutputs {
        torch::Tensor logits;
        torch::Tensor eqHat;
        torch::Tensor oppHat;
    };
    AuxOutputs forwardAll(const torch::Tensor& node, const torch::Tensor& seq,
                          const torch::Tensor& actFeat, const torch::Tensor& actSample,
                          const torch::Tensor& actId,
                          const torch::Tensor& rangeEq = torch::Tensor()) const;

    void saveTo(const std::string& path) const;
    // 旧检查点没有 aux_* 参数：严格加载失败后按参数名逐个读取，缺的跳过。
    bool loadFrom(const std::string& path);

    ModelConfig config;
    mutable torch::nn::LSTM lstm{nullptr};
    mutable torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr}, l4{nullptr}, head{nullptr};
    // 范围胜率信念头 + 信念残差评分支路（新参数，旧检查点加载时保持初始值，
    // 残差支路末层零初始化，所以旧模型的行为与改动前完全一致）。
    mutable torch::nn::Linear aux1{nullptr}, aux2{nullptr};
    mutable torch::nn::Linear auxAct1{nullptr}, auxAct2{nullptr};
    // 对手建模头：预测"对手当前牌力百分位"。残差支路把 oppHat 与动作特征
    // 拼接后加到动作分数上（末层零初始化，旧检查点行为不变）。
    mutable torch::nn::Linear opp1{nullptr}, opp2{nullptr};
    mutable torch::nn::Linear oppAct1{nullptr}, oppAct2{nullptr};
    // 范围胜率残差支路：输入的 rangeEq 由 rollout 在组合级算出（按对手最近
    // 动作做贝叶斯更新 + 我方的摊牌胜率），不是网络自己回归出来的标量。
    // 与动作特征拼接后修正动作分数；末层零初始化，旧检查点行为不变。
    mutable torch::nn::Linear rngAct1{nullptr}, rngAct2{nullptr};
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
