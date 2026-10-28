#include "texas/nn.hpp"

#include <unistd.h>

#include <atomic>
#include <filesystem>

namespace texas {
namespace nn {

ActorImpl::ActorImpl(const ModelConfig& cfg) : config(cfg) {
    torch::nn::LSTMOptions opts(kHistDim, cfg.lstmHidden);
    opts.batch_first(true);
    lstm = register_module("lstm", torch::nn::LSTM(opts));

    const int64_t nodeDim = kNodeInput + cfg.lstmHidden;
    const int64_t jointDim = nodeDim + kActionInput;

    l1 = register_module("l1", torch::nn::Linear(jointDim, cfg.hidden));
    l2 = register_module("l2", torch::nn::Linear(cfg.hidden, cfg.hidden));
    l3 = register_module("l3", torch::nn::Linear(cfg.hidden, cfg.hidden));
    l4 = register_module("l4", torch::nn::Linear(cfg.hidden, cfg.hidden * 2));
    head = register_module("head", torch::nn::Linear(cfg.hidden * 2, 1));

    // 范围胜率信念头：把"对手底牌"从输入里拿掉，只靠公开信息回归开天眼胜率。
    // 线上部署时它就是在估计"我的牌 vs 对手范围"的胜率，是策略最缺的那个量。
    aux1 = register_module("aux1", torch::nn::Linear(nodeDim, 128));
    aux2 = register_module("aux2", torch::nn::Linear(128, 1));
    torch::nn::init::normal_(aux2->weight, 0.0, 0.01);
    torch::nn::init::zeros_(aux2->bias);

    // 信念残差支路：把 eqHat 与动作特征一起过一个小 MLP，直接给每个动作
    // 加一个分数修正（"胜率这么低还打这么大"）。末层零初始化 => 旧模型不变。
    auxAct1 = register_module("aux_act1", torch::nn::Linear(kActionInput + 1, 64));
    auxAct2 = register_module("aux_act2", torch::nn::Linear(64, 1));
    torch::nn::init::zeros_(auxAct2->weight);
    torch::nn::init::zeros_(auxAct2->bias);

    // 对手建模头：与信念头同构，但回归标签是"对手当前牌力百分位"。
    // 残差支路独立一条（不扩 aux_act1 的输入维度），这样带 aux 的旧检查点
    // 仍然能按原形状加载；所有新参数都在零初始化路径上（末层为零）。
    opp1 = register_module("opp1", torch::nn::Linear(nodeDim, 128));
    opp2 = register_module("opp2", torch::nn::Linear(128, 1));
    torch::nn::init::normal_(opp2->weight, 0.0, 0.01);
    torch::nn::init::zeros_(opp2->bias);
    oppAct1 = register_module("opp_act1", torch::nn::Linear(kActionInput + 1, 32));
    oppAct2 = register_module("opp_act2", torch::nn::Linear(32, 1));
    torch::nn::init::zeros_(oppAct2->weight);
    torch::nn::init::zeros_(oppAct2->bias);

    // 范围胜率残差支路：rangeEq 由 rollout 采样对手底牌、按对手最近动作做
    // 贝叶斯重加权后算出（组合级计算，网络自己做不到）。这里只负责把
    // "动作 x 胜率"接进打分。末层零初始化 => 旧检查点行为逐位不变。
    rngAct1 = register_module("rng_act1", torch::nn::Linear(kActionInput + 1, 32));
    rngAct2 = register_module("rng_act2", torch::nn::Linear(32, 1));
    torch::nn::init::zeros_(rngAct2->weight);
    torch::nn::init::zeros_(rngAct2->bias);

    // 最后一层零初始化：开局所有动作分数相同，策略从均匀分布出发
    torch::nn::init::zeros_(head->weight);
    torch::nn::init::zeros_(head->bias);
}

std::pair<torch::Tensor, torch::Tensor> ActorImpl::forwardWithAux(
    const torch::Tensor& node, const torch::Tensor& seq, const torch::Tensor& actFeat,
    const torch::Tensor& actSample, const torch::Tensor& actId,
    const torch::Tensor& rangeEq) const {
    AuxOutputs out = forwardAll(node, seq, actFeat, actSample, actId, rangeEq);
    return {out.logits, out.eqHat};
}

ActorImpl::AuxOutputs ActorImpl::forwardAll(
    const torch::Tensor& node, const torch::Tensor& seq, const torch::Tensor& actFeat,
    const torch::Tensor& actSample, const torch::Tensor& actId,
    const torch::Tensor& rangeEq) const {
    auto lstmOut = lstm->forward(seq);
    torch::Tensor hLast = std::get<0>(std::get<1>(lstmOut)).squeeze(0);
    torch::Tensor x = torch::cat({node, hLast}, /*dim=*/1);

    // 不完全信息胜率预测（detach 后进残差支路：信念头只由回归损失塑形，
    // 策略梯度不去扭曲校准）。
    torch::Tensor eqHat = torch::sigmoid(aux2->forward(torch::relu(aux1->forward(x))))
                              .squeeze(-1);

    const int64_t nodeDim = kNodeInput + config.lstmHidden;
    torch::Tensor wNode = l1->weight.slice(1, 0, nodeDim);
    torch::Tensor wAct = l1->weight.slice(1, nodeDim);

    // 分解成两次矩阵乘：节点部分只算一次，再按动作所属样本 gather
    torch::Tensor nodeProj = torch::addmm(l1->bias, x, wNode.t());
    torch::Tensor actProj = torch::mm(actFeat, wAct.t());
    torch::Tensor sampleIdx = actSample.to(torch::kLong);

    torch::Tensor f1 = torch::relu(nodeProj.index_select(0, sampleIdx) + actProj);
    torch::Tensor f2 = torch::relu(l2->forward(f1));
    torch::Tensor f3 = torch::relu(l3->forward(f2));
    torch::Tensor f4 = torch::relu(l4->forward(f3));
    torch::Tensor scores = head->forward(f4).squeeze(-1);

    // 残差支路：eqHat 是这个动作所属样本的预测胜率；与动作特征拼接后
    // 给出每动作一个标量修正。eqHat.detach() 保证校准目标不被策略梯度污染。
    torch::Tensor eqPerAction =
        eqHat.detach().index_select(0, sampleIdx).unsqueeze(1);
    torch::Tensor aj = torch::cat({actFeat, eqPerAction}, /*dim=*/1);
    scores = scores + auxAct2->forward(torch::relu(auxAct1->forward(aj))).squeeze(-1);

    // 对手建模残差支路：同一个动作在"对手范围很强"和"对手范围很弱"两种
    // 判断下应该得到不同的分数（对手强 -> 大注更差）。oppHat 同样 detach。
    torch::Tensor oppHat = torch::sigmoid(opp2->forward(torch::relu(opp1->forward(x))))
                               .squeeze(-1);
    torch::Tensor oppPerAction =
        oppHat.detach().index_select(0, sampleIdx).unsqueeze(1);
    torch::Tensor oj = torch::cat({actFeat, oppPerAction}, /*dim=*/1);
    scores = scores + oppAct2->forward(torch::relu(oppAct1->forward(oj))).squeeze(-1);

    // 范围胜率残差支路：rangeEq 是 rollout 用"对手最近动作 + 采样底牌"算出的
    // vs-range 摊牌胜率（组合级后验加权），随状态变化。它不参与梯度（是输入
    // 特征），与动作特征拼接后给每个动作一个修正。
    torch::Tensor rngPerAction;
    if (rangeEq.defined() && rangeEq.numel() == node.size(0)) {
        rngPerAction = rangeEq.to(node.device()).index_select(0, sampleIdx).unsqueeze(1);
    } else {
        rngPerAction = torch::full({actFeat.size(0), 1}, 0.5f,
                                   torch::TensorOptions().dtype(torch::kFloat32).device(node.device()));
    }
    torch::Tensor rj = torch::cat({actFeat, rngPerAction}, /*dim=*/1);
    scores = scores + rngAct2->forward(torch::relu(rngAct1->forward(rj))).squeeze(-1);

    const int64_t B = node.size(0);
    torch::Tensor logits = torch::full({B, kNumActions}, -1e9f,
                                       torch::TensorOptions().dtype(torch::kFloat32).device(node.device()));
    logits.index_put_({sampleIdx, actId.to(torch::kLong)}, scores);
    return AuxOutputs{logits, eqHat, oppHat};
}

torch::Tensor ActorImpl::forward(const torch::Tensor& node, const torch::Tensor& seq,
                                 const torch::Tensor& actFeat, const torch::Tensor& actSample,
                                 const torch::Tensor& actId, const torch::Tensor& rangeEq) const {
    return forwardWithAux(node, seq, actFeat, actSample, actId, rangeEq).first;
}

void ActorImpl::saveTo(const std::string& path) const {
    torch::serialize::OutputArchive archive;
    Module::save(archive);
    archive.save_to(path);
}

bool ActorImpl::loadFrom(const std::string& path) {
    // 新检查点：参数齐全，走严格加载（递归子模块）。
    // 旧检查点：没有 aux_* 子模块，严格加载会抛异常；此时按旧架构的固定
    // 子模块名逐个读取，缺的 aux_* 保留初始化值（残差支路末层为零 =>
    // 旧模型行为与改动前完全一致）。
    {
        torch::serialize::InputArchive archive;
        try {
            archive.load_from(path);
            Module::load(archive);
            return true;
        } catch (const std::exception&) {
            // fall through
        }
    }
    try {
        torch::NoGradGuard ng;
        torch::serialize::InputArchive old;
        old.load_from(path);
        int loaded = 0;
        auto loadMod = [&](const std::string& name, const torch::nn::Module& m) {
            try {
                torch::serialize::InputArchive sub;
                old.read(name, sub);
                for (const auto& p : m.named_parameters()) {
                    torch::Tensor t = p.value().clone();
                    sub.read(p.key(), t);
                    p.value().copy_(t);
                    ++loaded;
                }
            } catch (const std::exception&) {
                // 该子模块在旧检查点里不存在：保留初始化值（新增支路末层为零）。
            }
        };
        loadMod("lstm", *lstm);
        loadMod("l1", *l1);
        loadMod("l2", *l2);
        loadMod("l3", *l3);
        loadMod("l4", *l4);
        loadMod("head", *head);
        loadMod("aux1", *aux1);
        loadMod("aux2", *aux2);
        loadMod("aux_act1", *auxAct1);
        loadMod("aux_act2", *auxAct2);
        loadMod("opp1", *opp1);
        loadMod("opp2", *opp2);
        loadMod("opp_act1", *oppAct1);
        loadMod("opp_act2", *oppAct2);
        loadMod("rng_act1", *rngAct1);
        loadMod("rng_act2", *rngAct2);
        std::printf("  actor: 旧检查点兼容加载 %d 个参数（缺失的新支路保持初始值）\n", loaded);
        return loaded > 0;
    } catch (const std::exception& e) {
        std::printf("  actor loadFrom 失败: %s\n", e.what());
        return false;
    }
}

CriticImpl::CriticImpl(const ModelConfig& cfg) : config(cfg) {
    torch::nn::LSTMOptions opts(kHistDim, cfg.lstmHidden);
    opts.batch_first(true);
    lstm = register_module("lstm", torch::nn::LSTM(opts));

    const int64_t allDim = kNodeInput + cfg.lstmHidden + kExtraInput;
    c1 = register_module("c1", torch::nn::Linear(allDim, cfg.hidden));
    c2 = register_module("c2", torch::nn::Linear(cfg.hidden, cfg.hidden));
    c3 = register_module("c3", torch::nn::Linear(cfg.hidden, cfg.hidden));
    c4 = register_module("c4", torch::nn::Linear(cfg.hidden, cfg.hidden));
    out = register_module("out", torch::nn::Linear(cfg.hidden, 1));

    torch::nn::init::zeros_(out->weight);
    torch::nn::init::zeros_(out->bias);
}

torch::Tensor CriticImpl::forward(const torch::Tensor& node, const torch::Tensor& seq,
                                  const torch::Tensor& extra) const {
    auto lstmOut = lstm->forward(seq);
    torch::Tensor hLast = std::get<0>(std::get<1>(lstmOut)).squeeze(0);
    torch::Tensor x = torch::cat({node, hLast, extra}, /*dim=*/1);

    torch::Tensor q1 = torch::relu(c1->forward(x));
    torch::Tensor q2 = torch::relu(c2->forward(q1));
    torch::Tensor q3 = torch::relu(c3->forward(q2));
    torch::Tensor q4 = torch::relu(c4->forward(q3));
    return out->forward(q4);
}

void CriticImpl::saveTo(const std::string& path) const {
    torch::serialize::OutputArchive archive;
    Module::save(archive);
    archive.save_to(path);
}

bool CriticImpl::loadFrom(const std::string& path) {
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        Module::load(archive);
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

namespace {

std::string uniqueTempPath() {
    static std::atomic<int> counter{0};
    return "/tmp/perfecttexas_clone_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter.fetch_add(1)) + ".pt";
}

}  // namespace

Actor cloneActor(const Actor& src) {
    const std::string tmp = uniqueTempPath();
    {
        torch::serialize::OutputArchive ar;
        src->save(ar);
        ar.save_to(tmp);
    }
    Actor dst(std::make_shared<ActorImpl>(src->config));
    torch::serialize::InputArchive in;
    in.load_from(tmp);
    dst->load(in);
    std::filesystem::remove(tmp);
    dst->to(src->parameters()[0].device());
    dst->eval();
    return dst;
}

Critic cloneCritic(const Critic& src) {
    const std::string tmp = uniqueTempPath();
    {
        torch::serialize::OutputArchive ar;
        src->save(ar);
        ar.save_to(tmp);
    }
    Critic dst(std::make_shared<CriticImpl>(src->config));
    torch::serialize::InputArchive in;
    in.load_from(tmp);
    dst->load(in);
    std::filesystem::remove(tmp);
    dst->to(src->parameters()[0].device());
    dst->eval();
    return dst;
}

}  // namespace nn
}  // namespace texas
