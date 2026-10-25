#include "texas/learner.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace texas {

void buildBatch(const std::vector<Transition>& tr, BatchTensors& batch, float advNormEps) {
    const int64_t N = static_cast<int64_t>(tr.size());
    size_t L = 0;
    for (const auto& t : tr) L += t.actions.size();

    batch.node = torch::empty({N, kNodeInput}, torch::kFloat32);
    batch.seq = torch::empty({N, kHistSteps, kHistDim}, torch::kFloat32);
    batch.extra = torch::empty({N, kExtraInput}, torch::kFloat32);
    batch.actFeat = torch::empty({static_cast<int64_t>(L), kActionInput}, torch::kFloat32);
    batch.actSample = torch::empty({static_cast<int64_t>(L)}, torch::kLong);
    batch.actId = torch::empty({static_cast<int64_t>(L)}, torch::kLong);
    batch.actions = torch::zeros({N}, torch::kLong);
    batch.oldLogp = torch::zeros({N}, torch::kFloat32);
    batch.adv = torch::zeros({N}, torch::kFloat32);
    batch.ret = torch::zeros({N}, torch::kFloat32);
    batch.oldValue = torch::zeros({N}, torch::kFloat32);

    float* pNode = batch.node.data_ptr<float>();
    float* pSeq = batch.seq.data_ptr<float>();
    float* pExtra = batch.extra.data_ptr<float>();
    float* pFeat = batch.actFeat.data_ptr<float>();
    int64_t* pSample = batch.actSample.data_ptr<int64_t>();
    int64_t* pId = batch.actId.data_ptr<int64_t>();
    int64_t* pAct = batch.actions.data_ptr<int64_t>();
    float* pLogp = batch.oldLogp.data_ptr<float>();
    float* pAdv = batch.adv.data_ptr<float>();
    float* pRet = batch.ret.data_ptr<float>();
    float* pVal = batch.oldValue.data_ptr<float>();

    int64_t off = 0;
    double sum = 0.0;
    double sq = 0.0;
    for (int64_t i = 0; i < N; ++i) {
        const Transition& t = tr[static_cast<size_t>(i)];
        std::copy(t.node.begin(), t.node.end(), pNode + i * kNodeInput);
        std::copy(t.seq.begin(), t.seq.end(), pSeq + i * kSeqInput);
        std::copy(t.extra.begin(), t.extra.end(), pExtra + i * kExtraInput);
        for (const ActionOption& a : t.actions) {
            std::copy(a.feat.begin(), a.feat.end(), pFeat + off * kActionInput);
            pSample[off] = i;
            pId[off] = a.slot;
            ++off;
        }
        pAct[i] = t.action;
        pLogp[i] = t.logp;
        pAdv[i] = t.adv;
        pRet[i] = t.ret;
        pVal[i] = t.value;
        sum += t.adv;
        sq += double(t.adv) * double(t.adv);
    }

    // 批内优势标准化
    const double mean = sum / double(N);
    const double var = std::max(sq / double(N) - mean * mean, 0.0);
    const float inv = 1.0f / float(std::sqrt(var + advNormEps));
    for (int64_t i = 0; i < N; ++i) {
        pAdv[i] = (pAdv[i] - float(mean)) * inv;
    }
}

Learner::Learner(nn::Actor actor_, nn::Critic critic_, const ppo::Config& cfg, torch::Device device_)
    : actor(actor_), critic(critic_), config(cfg), device(device_) {
    actor->to(device);
    critic->to(device);
    actor->train();
    critic->train();

    torch::optim::AdamOptions aOpts(config.lr);
    aOpts.betas(std::make_tuple(0.9, 0.999)).eps(1e-5);
    actorOpt = std::make_unique<torch::optim::Adam>(actor->parameters(), aOpts);

    torch::optim::AdamOptions cOpts(config.lr);
    cOpts.betas(std::make_tuple(0.9, 0.999)).eps(1e-5);
    criticOpt = std::make_unique<torch::optim::Adam>(critic->parameters(), cOpts);
}

void Learner::setLr(float lr) {
    config.lr = lr;
    for (auto& g : actorOpt->param_groups()) {
        static_cast<torch::optim::AdamOptions&>(g.options()).lr(lr);
    }
    for (auto& g : criticOpt->param_groups()) {
        static_cast<torch::optim::AdamOptions&>(g.options()).lr(lr);
    }
}

ppo::Stats Learner::update(BatchTensors& batch) {
    const int64_t N = batch.node.size(0);
    ppo::Stats stats;
    if (N == 0) return stats;

    actor->train();
    critic->train();

    // 回报分布（同一批数据在两个 epoch 之间不变，取一次即可）
    {
        const auto ret = batch.ret;
        stats.retMean = ret.mean().item<double>();
        stats.retStd = ret.std().item<double>();
        if (batch.oldValue.numel() > 0) stats.valueMean = batch.oldValue.mean().item<double>();
    }

    const int64_t mbSize = std::min<int64_t>(config.minibatch, N);
    const int64_t nBatches = (N + mbSize - 1) / mbSize;

    double pgSum = 0, vSum = 0, entSum = 0, klSum = 0, clipSum = 0;
    int total = 0;
    int vTotal = 0;

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        torch::Tensor indices = torch::randperm(N, torch::TensorOptions().dtype(torch::kLong));
        double epochKL = 0;
        int epochSamples = 0;

        for (int64_t b = 0; b < nBatches; ++b) {
            const int64_t start = b * mbSize;
            const int64_t end = std::min(start + mbSize, N);
            const int64_t curB = end - start;
            torch::Tensor mbIdx = indices.slice(0, start, end).to(device);

            torch::Tensor nodeMb = batch.node.index_select(0, mbIdx);
            torch::Tensor seqMb = batch.seq.index_select(0, mbIdx);
            torch::Tensor extraMb = batch.extra.index_select(0, mbIdx);
            torch::Tensor actMb = batch.actions.index_select(0, mbIdx).to(device);
            torch::Tensor oldLogpMb = batch.oldLogp.index_select(0, mbIdx).to(device);
            torch::Tensor advMb = batch.adv.index_select(0, mbIdx).to(device);
            torch::Tensor retMb = batch.ret.index_select(0, mbIdx).to(device);

            // 动作特征按线性表映射到 minibatch 内的样本号
            torch::Tensor lookup = torch::full({N}, -1, torch::TensorOptions().dtype(torch::kLong));
            lookup.index_put_({mbIdx.cpu()}, torch::arange(curB, torch::TensorOptions().dtype(torch::kLong)));
            torch::Tensor mapped = lookup.index_select(0, batch.actSample);
            torch::Tensor mask = mapped >= 0;
            torch::Tensor actIdx = mask.nonzero().squeeze(-1);
            if (actIdx.numel() == 0) continue;

            torch::Tensor actFeatMb = batch.actFeat.index_select(0, actIdx).to(device);
            torch::Tensor actSampleMb = mapped.index_select(0, actIdx).to(device);
            torch::Tensor actIdMb = batch.actId.index_select(0, actIdx).to(device);

            // ---- actor ----
            torch::Tensor logits = actor->forward(nodeMb, seqMb, actFeatMb, actSampleMb, actIdMb);
            torch::Tensor logProbs = torch::log_softmax(logits, 1);
            torch::Tensor probs = torch::softmax(logits, 1);
            torch::Tensor curLogp = logProbs.gather(1, actMb.unsqueeze(1)).squeeze(1);

            torch::Tensor logRatio = curLogp - oldLogpMb;
            torch::Tensor ratio = torch::exp(logRatio);
            torch::Tensor approxKL = torch::nan_to_num((ratio - 1.0f) - logRatio, 0.0);
            const double mbKL = approxKL.mean().item<double>();
            epochKL += mbKL * double(curB);
            epochSamples += static_cast<int>(curB);

            torch::Tensor surr1 = ratio * advMb;
            torch::Tensor surr2 = torch::clamp(ratio, 1.0f - config.clip, 1.0f + config.clip) * advMb;
            torch::Tensor pgLoss = -torch::min(surr1, surr2).mean();
            torch::Tensor entropy =
                -(probs * torch::nan_to_num(logProbs, 0.0)).sum(1).mean();
            torch::Tensor actorLoss = pgLoss - config.entCoef * entropy;

            actorOpt->zero_grad();
            actorLoss.backward();
            torch::nn::utils::clip_grad_norm_(actor->parameters(), config.maxGradNorm);
            actorOpt->step();

            // ---- critic ----
            torch::Tensor values = critic->forward(nodeMb, seqMb, extraMb).squeeze(-1);
            torch::Tensor vLoss = config.vfCoef * 0.5f * (values - retMb).pow(2).mean();

            criticOpt->zero_grad();
            vLoss.backward();
            torch::nn::utils::clip_grad_norm_(critic->parameters(), config.criticMaxGradNorm);
            criticOpt->step();

            pgSum += pgLoss.item<double>() * double(curB);
            vSum += vLoss.item<double>() * double(curB);
            vTotal += static_cast<int>(curB);
            entSum += entropy.item<double>() * double(curB);
            klSum += mbKL * double(curB);
            torch::Tensor clipped =
                ((ratio > 1.0f + config.clip) | (ratio < 1.0f - config.clip)).to(torch::kFloat32);
            clipSum += clipped.sum().item<double>();
            total += static_cast<int>(curB);
        }

        stats.epochsCompleted = epoch + 1;
        if (epochSamples > 0 && (epochKL / epochSamples) > config.targetKL * 1.5) break;
    }

    // 额外的价值网轮数：只更新 critic，不动 actor
    if (config.criticExtraEpochs > 0 && N > 0) {
        for (int epoch = 0; epoch < config.criticExtraEpochs; ++epoch) {
            torch::Tensor perm = torch::randperm(N, torch::TensorOptions().dtype(torch::kLong));
            for (int64_t b = 0; b < nBatches; ++b) {
                const int64_t start = b * mbSize;
                const int64_t end = std::min(start + mbSize, N);
                torch::Tensor mbIdx = perm.slice(0, start, end);
                torch::Tensor nodeMb = batch.node.index_select(0, mbIdx);
                torch::Tensor seqMb = batch.seq.index_select(0, mbIdx);
                torch::Tensor extraMb = batch.extra.index_select(0, mbIdx);
                torch::Tensor retMb = batch.ret.index_select(0, mbIdx);
                torch::Tensor values = critic->forward(nodeMb, seqMb, extraMb).squeeze(-1);
                torch::Tensor vLoss = config.vfCoef * 0.5f * (values - retMb).pow(2).mean();
                criticOpt->zero_grad();
                vLoss.backward();
                torch::nn::utils::clip_grad_norm_(critic->parameters(), config.criticMaxGradNorm);
                criticOpt->step();
                vSum += vLoss.item<double>() * double(end - start);
                vTotal += static_cast<int>(end - start);
            }
        }
    }

    if (total > 0) {
        stats.pgLoss = pgSum / total;
        stats.vLoss = vTotal > 0 ? vSum / vTotal : 0.0;
        stats.entropy = entSum / total;
        stats.kl = klSum / total;
        stats.clipFraction = clipSum / total;
    }
    return stats;
}

void Learner::saveOptimizers(const std::string& dir) const {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    {
        torch::serialize::OutputArchive a;
        actorOpt->save(a);
        a.save_to(dir + "/actor_opt.pt");
    }
    {
        torch::serialize::OutputArchive c;
        criticOpt->save(c);
        c.save_to(dir + "/critic_opt.pt");
    }
}

void Learner::loadOptimizers(const std::string& dir) {
    if (std::filesystem::exists(dir + "/actor_opt.pt")) {
        torch::serialize::InputArchive a;
        a.load_from(dir + "/actor_opt.pt");
        actorOpt->load(a);
    }
    if (std::filesystem::exists(dir + "/critic_opt.pt")) {
        torch::serialize::InputArchive c;
        c.load_from(dir + "/critic_opt.pt");
        criticOpt->load(c);
    }
}

}  // namespace texas
