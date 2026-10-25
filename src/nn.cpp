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

    // 最后一层零初始化：开局所有动作分数相同，策略从均匀分布出发
    torch::nn::init::zeros_(head->weight);
    torch::nn::init::zeros_(head->bias);
}

torch::Tensor ActorImpl::forward(const torch::Tensor& node, const torch::Tensor& seq,
                                 const torch::Tensor& actFeat, const torch::Tensor& actSample,
                                 const torch::Tensor& actId) const {
    auto lstmOut = lstm->forward(seq);
    torch::Tensor hLast = std::get<0>(std::get<1>(lstmOut)).squeeze(0);
    torch::Tensor x = torch::cat({node, hLast}, /*dim=*/1);

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

    const int64_t B = node.size(0);
    torch::Tensor logits = torch::full({B, kNumActions}, -1e9f,
                                       torch::TensorOptions().dtype(torch::kFloat32).device(node.device()));
    logits.index_put_({sampleIdx, actId.to(torch::kLong)}, scores);
    return logits;
}

void ActorImpl::saveTo(const std::string& path) const {
    torch::serialize::OutputArchive archive;
    Module::save(archive);
    archive.save_to(path);
}

bool ActorImpl::loadFrom(const std::string& path) {
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        Module::load(archive);
        return true;
    } catch (const std::exception& e) {
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
