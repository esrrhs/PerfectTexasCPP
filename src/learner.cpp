#include "texas/learner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <tuple>

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
    batch.teacherQ = torch::zeros({N, kNumActions}, torch::kFloat32);
    batch.teacherValid = torch::zeros({N}, torch::kFloat32);
    batch.auxLabel = torch::zeros({N}, torch::kFloat32);
    batch.ruleSlot = torch::full({N}, -1, torch::kLong);

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
    float* pTq = batch.teacherQ.data_ptr<float>();
    float* pTv = batch.teacherValid.data_ptr<float>();
    float* pAux = batch.auxLabel.data_ptr<float>();
    int64_t* pRule = batch.ruleSlot.data_ptr<int64_t>();

    int64_t off = 0;
    double sum = 0.0;
    double sq = 0.0;
    for (int64_t i = 0; i < N; ++i) {
        const Transition& t = tr[static_cast<size_t>(i)];
        std::copy(t.node.begin(), t.node.end(), pNode + i * kNodeInput);
        std::copy(t.seq.begin(), t.seq.end(), pSeq + i * kSeqInput);
        std::copy(t.extra.begin(), t.extra.end(), pExtra + i * kExtraInput);
        if (t.hasTeacher) {
            std::copy(t.teacherQ.begin(), t.teacherQ.end(), pTq + i * kNumActions);
            pTv[i] = 1.0f;
        }
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
        pAux[i] = t.extra[kHoleDim];  // 开天眼胜率（对手底牌之后第 0 个标量）
        pRule[i] = t.ruleSlot;
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
    cOpts.weight_decay(config.criticWeightDecay);
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
        stats.advStd = batch.adv.std().item<double>();
        if (batch.oldValue.numel() > 0) {
            stats.valueMean = batch.oldValue.mean().item<double>();
            stats.valueStd = batch.oldValue.std().item<double>();
        }
        if (batch.oldValue.numel() > 1 && batch.ret.numel() > 1) {
            torch::Tensor rc = batch.ret - batch.ret.mean();
            torch::Tensor vc = batch.oldValue - batch.oldValue.mean();
            torch::Tensor denom = rc.std() * vc.std();
            stats.valueCorr = ((rc * vc).mean() / denom.clamp_min(1e-8)).item<double>();
        }
    }

    const int64_t mbSize = std::min<int64_t>(config.minibatch, N);
    const int64_t nBatches = (N + mbSize - 1) / mbSize;

    double pgSum = 0, vSum = 0, entSum = 0, klSum = 0, clipSum = 0;
    double teaSum = 0, teaMaxSum = 0, teaEntSum = 0;
    double auxSum = 0, auxMaeSum = 0, auxPredSum = 0, auxLabSum = 0;
    int teaTotal = 0;
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
            torch::Tensor logits;
            torch::Tensor eqHat;
            std::tie(logits, eqHat) =
                actor->forwardWithAux(nodeMb, seqMb, actFeatMb, actSampleMb, actIdMb);
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

            // ---- 状态条件信任域 ----
            // 非目标状态（pot < anchorMaxPot）上把策略钉在冻结基座上：
            // KL(π_new || π_anchor)。这样大池教师/辅助信号只能改大池，
            // 不会把中池已经验证过的打法一起带走。
            if (anchor != nullptr && config.anchorCoef > 0.0f) {
                torch::Tensor aLogits;
                {
                    torch::NoGradGuard ng;
                    aLogits = (*anchor)->forward(nodeMb, seqMb, actFeatMb, actSampleMb, actIdMb);
                }
                torch::Tensor aLogp = torch::log_softmax(aLogits, 1);
                torch::Tensor kl = (probs.detach() * (logProbs - aLogp)).sum(1);
                torch::Tensor pot =
                    nodeMb.select(1, kHoleDim + kBoardDim) * float(kStartingStack);
                torch::Tensor mask = (pot < config.anchorMaxPot).to(torch::kFloat32);
                torch::Tensor klLoss = (kl * mask).sum() / mask.sum().clamp_min(1.0f);
                actorLoss = actorLoss + config.anchorCoef * klLoss;
            }

            // ---- 规则教师蒸馏（河牌大池 -> 1.5x 等外部范围规则） ----
            if (config.ruleCoef > 0.0f && batch.ruleSlot.numel() > 0) {
                torch::Tensor rs = batch.ruleSlot.index_select(0, mbIdx).to(device);
                torch::Tensor valid = (rs >= 0).to(torch::kFloat32);
                const float nValid = valid.sum().item<float>();
                if (nValid > 0.5f) {
                    torch::Tensor ce =
                        -logProbs.gather(1, rs.clamp_min(0).unsqueeze(1)).squeeze(1);
                    torch::Tensor rloss = (ce * valid).sum() / valid.sum().clamp_min(1.0f);
                    actorLoss = actorLoss + config.ruleCoef * rloss;
                }
            }

            // ---- 范围胜率辅助头 ----
            // 目标 = 开天眼真实胜率（batch.auxLabel，取自观测 extra[0]）。
            // 只用不完全信息回归它，逼 actor 的共享表示学"对手范围有多强"。
            // 残差支路用的是 detach 的 eqHat，所以策略梯度不会反向污染校准。
            if (config.auxEquityCoef > 0.0f) {
                torch::Tensor auxReal = batch.auxLabel.index_select(0, mbIdx).to(device);
                torch::Tensor diff = eqHat - auxReal;
                torch::Tensor auxLoss = diff.pow(2).mean();
                actorLoss = actorLoss + config.auxEquityCoef * auxLoss;
                auxSum += config.auxEquityCoef * auxLoss.item<double>() * double(curB);
                auxMaeSum += diff.abs().mean().item<double>() * double(curB);
                auxPredSum += eqHat.mean().item<double>() * double(curB);
                auxLabSum += auxReal.mean().item<double>() * double(curB);
            }

            // ---- 脚本教师蒸馏 ----
            // Q 是"同一副牌、只换我们的动作"的分支续演结果，所以动作间差异
            // 不含发牌方差。softmax(Q/T) 作为目标分布，交叉熵只灌 actor，
            // 不进 PPO 优势（避免教师偏差污染价值基线）。
            if (config.teacherCoef > 0.0f && batch.teacherValid.numel() > 0) {
                torch::Tensor valid = batch.teacherValid.index_select(0, mbIdx).to(device);
                const float nValid = valid.sum().item<float>();
                if (nValid > 0.5f) {
                    // 逐状态标准化：动作选择只取决于同一局面内 Q 的相对大小，
                    // 绝对筹码尺度只会把每个目标都压成"接近均匀"。用合法动作的
                    // 标准差归一后，温度才是一个有意义的旋钮。
                    torch::Tensor legalMask = logits.detach() > -1e8f;
                    torch::Tensor legalF = legalMask.to(torch::kFloat32);
                    torch::Tensor cnt = legalF.sum(1, /*keepdim=*/true).clamp_min(1.0f);
                    torch::Tensor raw = batch.teacherQ.index_select(0, mbIdx).to(device);
                    torch::Tensor mean = (raw * legalF).sum(1, /*keepdim=*/true) / cnt;
                    torch::Tensor centered = (raw - mean) * legalF;
                    torch::Tensor scale =
                        (centered.pow(2).sum(1, /*keepdim=*/true) / cnt).sqrt().clamp_min(1e-3f);
                    torch::Tensor tq = centered / scale / config.teacherTemp;
                    // 非法 slot 的 Q 无意义，按 logits 的掩码一起抹掉。
                    tq = tq.masked_fill(~legalMask, -1e9f);
                    torch::Tensor tlog = torch::log_softmax(tq, 1);
                    torch::Tensor tprob = torch::exp(tlog);
                    torch::Tensor ce = -(tprob * logProbs).sum(1);
                    torch::Tensor teaLoss = (ce * valid).sum() / valid.sum().clamp_min(1.0f);
                    actorLoss = actorLoss + config.teacherCoef * teaLoss;
                    teaSum += teaLoss.item<double>() * double(nValid);
                    teaMaxSum +=
                        (std::get<0>(tprob.max(1)) * valid).sum().item<double>();
                    teaEntSum +=
                        (-(tprob * tlog).sum(1) * valid).sum().item<double>();
                    teaTotal += static_cast<int>(nValid);
                }
            }

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
        stats.teacherLoss = teaTotal > 0 ? teaSum / double(teaTotal) : 0.0;
        stats.teacherEntropy = teaTotal > 0 ? teaEntSum / double(teaTotal) : 0.0;
        stats.teacherMax = teaTotal > 0 ? teaMaxSum / double(teaTotal) : 0.0;
        stats.auxLoss = auxSum / double(total);
        stats.auxMae = auxMaeSum / double(total);
        stats.auxPred = auxPredSum / double(total);
        stats.auxLabel = auxLabSum / double(total);
    }
    return stats;
}

ppo::Stats Learner::updateActorOnly(BatchTensors& batch) {
    const int64_t N = batch.node.size(0);
    ppo::Stats stats;
    if (N == 0) return stats;
    actor->train();

    {
        const auto ret = batch.ret;
        stats.retMean = ret.mean().item<double>();
        stats.retStd = ret.std().item<double>();
        stats.advStd = batch.adv.std().item<double>();
        if (batch.oldValue.numel() > 0) {
            stats.valueMean = batch.oldValue.mean().item<double>();
            stats.valueStd = batch.oldValue.std().item<double>();
        }
        if (batch.oldValue.numel() > 1 && batch.ret.numel() > 1) {
            torch::Tensor rc = batch.ret - batch.ret.mean();
            torch::Tensor vc = batch.oldValue - batch.oldValue.mean();
            torch::Tensor denom = rc.std() * vc.std();
            stats.valueCorr = ((rc * vc).mean() / denom.clamp_min(1e-8)).item<double>();
        }
    }

    const int64_t mbSize = std::min<int64_t>(config.minibatch, N);
    const int64_t nBatches = (N + mbSize - 1) / mbSize;
    double pgSum = 0, entSum = 0, klSum = 0, clipSum = 0;
    double teaSum = 0, teaMaxSum = 0, teaEntSum = 0;
    double auxSum = 0, auxMaeSum = 0, auxPredSum = 0, auxLabSum = 0;
    int teaTotal = 0, total = 0;

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
            torch::Tensor actMb = batch.actions.index_select(0, mbIdx).to(device);
            torch::Tensor oldLogpMb = batch.oldLogp.index_select(0, mbIdx).to(device);
            torch::Tensor advMb = batch.adv.index_select(0, mbIdx).to(device);

            torch::Tensor lookup = torch::full({N}, -1, torch::TensorOptions().dtype(torch::kLong));
            lookup.index_put_({mbIdx.cpu()}, torch::arange(curB, torch::TensorOptions().dtype(torch::kLong)));
            torch::Tensor mapped = lookup.index_select(0, batch.actSample);
            torch::Tensor mask = mapped >= 0;
            torch::Tensor actIdx = mask.nonzero().squeeze(-1);
            if (actIdx.numel() == 0) continue;

            torch::Tensor actFeatMb = batch.actFeat.index_select(0, actIdx).to(device);
            torch::Tensor actSampleMb = mapped.index_select(0, actIdx).to(device);
            torch::Tensor actIdMb = batch.actId.index_select(0, actIdx).to(device);

            torch::Tensor logits;
            torch::Tensor eqHat;
            std::tie(logits, eqHat) =
                actor->forwardWithAux(nodeMb, seqMb, actFeatMb, actSampleMb, actIdMb);
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
            torch::Tensor entropy = -(probs * torch::nan_to_num(logProbs, 0.0)).sum(1).mean();
            torch::Tensor actorLoss = pgLoss - config.entCoef * entropy;

            // ---- 状态条件信任域 ----
            // 非目标状态（pot < anchorMaxPot）上把策略钉在冻结基座上：
            // KL(π_new || π_anchor)。这样大池教师/辅助信号只能改大池，
            // 不会把中池已经验证过的打法一起带走。
            if (anchor != nullptr && config.anchorCoef > 0.0f) {
                torch::Tensor aLogits;
                {
                    torch::NoGradGuard ng;
                    aLogits = (*anchor)->forward(nodeMb, seqMb, actFeatMb, actSampleMb, actIdMb);
                }
                torch::Tensor aLogp = torch::log_softmax(aLogits, 1);
                torch::Tensor kl = (probs.detach() * (logProbs - aLogp)).sum(1);
                torch::Tensor pot =
                    nodeMb.select(1, kHoleDim + kBoardDim) * float(kStartingStack);
                torch::Tensor mask = (pot < config.anchorMaxPot).to(torch::kFloat32);
                torch::Tensor klLoss = (kl * mask).sum() / mask.sum().clamp_min(1.0f);
                actorLoss = actorLoss + config.anchorCoef * klLoss;
            }

            // ---- 范围胜率辅助头 ----
            // 目标 = 开天眼真实胜率（batch.auxLabel，取自观测 extra[0]）。
            // 只用不完全信息回归它，逼 actor 的共享表示学"对手范围有多强"。
            // 残差支路用的是 detach 的 eqHat，所以策略梯度不会反向污染校准。
            if (config.auxEquityCoef > 0.0f) {
                torch::Tensor auxReal = batch.auxLabel.index_select(0, mbIdx).to(device);
                torch::Tensor diff = eqHat - auxReal;
                torch::Tensor auxLoss = diff.pow(2).mean();
                actorLoss = actorLoss + config.auxEquityCoef * auxLoss;
                auxSum += config.auxEquityCoef * auxLoss.item<double>() * double(curB);
                auxMaeSum += diff.abs().mean().item<double>() * double(curB);
                auxPredSum += eqHat.mean().item<double>() * double(curB);
                auxLabSum += auxReal.mean().item<double>() * double(curB);
            }

            if (config.teacherCoef > 0.0f && batch.teacherValid.numel() > 0) {
                torch::Tensor valid = batch.teacherValid.index_select(0, mbIdx).to(device);
                const float nValid = valid.sum().item<float>();
                if (nValid > 0.5f) {
                    torch::Tensor legalMask = logits.detach() > -1e8f;
                    torch::Tensor legalF = legalMask.to(torch::kFloat32);
                    torch::Tensor cnt = legalF.sum(1, /*keepdim=*/true).clamp_min(1.0f);
                    torch::Tensor raw = batch.teacherQ.index_select(0, mbIdx).to(device);
                    torch::Tensor mean = (raw * legalF).sum(1, /*keepdim=*/true) / cnt;
                    torch::Tensor centered = (raw - mean) * legalF;
                    torch::Tensor scale =
                        (centered.pow(2).sum(1, /*keepdim=*/true) / cnt).sqrt().clamp_min(1e-3f);
                    torch::Tensor tq = centered / scale / config.teacherTemp;
                    tq = tq.masked_fill(~legalMask, -1e9f);
                    torch::Tensor tlog = torch::log_softmax(tq, 1);
                    torch::Tensor tprob = torch::exp(tlog);
                    torch::Tensor ce = -(tprob * logProbs).sum(1);
                    torch::Tensor teaLoss = (ce * valid).sum() / valid.sum().clamp_min(1.0f);
                    actorLoss = actorLoss + config.teacherCoef * teaLoss;
                    teaSum += teaLoss.item<double>() * double(nValid);
                    teaMaxSum += (std::get<0>(tprob.max(1)) * valid).sum().item<double>();
                    teaEntSum += (-(tprob * tlog).sum(1) * valid).sum().item<double>();
                    teaTotal += static_cast<int>(nValid);
                }
            }

            actorOpt->zero_grad();
            actorLoss.backward();
            torch::nn::utils::clip_grad_norm_(actor->parameters(), config.maxGradNorm);
            actorOpt->step();

            pgSum += pgLoss.item<double>() * double(curB);
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

    if (total > 0) {
        stats.pgLoss = pgSum / total;
        stats.entropy = entSum / total;
        stats.kl = klSum / total;
        stats.clipFraction = clipSum / total;
        stats.teacherLoss = teaTotal > 0 ? teaSum / double(teaTotal) : 0.0;
        stats.teacherEntropy = teaTotal > 0 ? teaEntSum / double(teaTotal) : 0.0;
        stats.teacherMax = teaTotal > 0 ? teaMaxSum / double(teaTotal) : 0.0;
        stats.auxLoss = auxSum / double(total);
        stats.auxMae = auxMaeSum / double(total);
        stats.auxPred = auxPredSum / double(total);
        stats.auxLabel = auxLabSum / double(total);
    }
    return stats;
}

double Learner::updateCriticOnly(BatchTensors& batch, int epochs) {
    const int64_t N = batch.node.size(0);
    if (N == 0 || epochs <= 0) return 0.0;
    critic->train();
    const int64_t mbSize = std::min<int64_t>(config.minibatch, N);
    const int64_t nBatches = (N + mbSize - 1) / mbSize;
    double vSum = 0.0;
    int vTotal = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        torch::Tensor perm = torch::randperm(N, torch::TensorOptions().dtype(torch::kLong));
        for (int64_t b = 0; b < nBatches; ++b) {
            const int64_t start = b * mbSize;
            const int64_t end = std::min(start + mbSize, N);
            torch::Tensor mbIdx = perm.slice(0, start, end).to(device);
            torch::Tensor values =
                critic->forward(batch.node.index_select(0, mbIdx), batch.seq.index_select(0, mbIdx),
                                batch.extra.index_select(0, mbIdx))
                    .squeeze(-1);
            torch::Tensor retMb = batch.ret.index_select(0, mbIdx).to(device);
            torch::Tensor vLoss = config.vfCoef * 0.5f * (values - retMb).pow(2).mean();
            criticOpt->zero_grad();
            vLoss.backward();
            torch::nn::utils::clip_grad_norm_(critic->parameters(), config.criticMaxGradNorm);
            criticOpt->step();
            vSum += vLoss.item<double>() * double(end - start);
            vTotal += static_cast<int>(end - start);
        }
    }
    return vTotal > 0 ? vSum / double(vTotal) : 0.0;
}

double Learner::distill(const std::deque<TeacherSample>& samples, int steps, float coef,
                        float temp, int minibatch) {
    const int64_t B = static_cast<int64_t>(samples.size());
    if (B < 8 || steps <= 0 || coef <= 0.0f) return 0.0;

    size_t L = 0;
    for (const TeacherSample& s : samples) L += s.actions.size();
    if (L == 0) return 0.0;

    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor nodeT = torch::empty({B, kNodeInput}, f32);
    torch::Tensor seqT = torch::empty({B, kHistSteps, kHistDim}, f32);
    torch::Tensor qT = torch::zeros({B, kNumActions}, f32);
    torch::Tensor legalT = torch::zeros({B, kNumActions}, f32);
    torch::Tensor actFeatT = torch::empty({static_cast<int64_t>(L), kActionInput}, f32);
    torch::Tensor actSampleT = torch::empty({static_cast<int64_t>(L)}, torch::kLong);
    torch::Tensor actIdT = torch::empty({static_cast<int64_t>(L)}, torch::kLong);

    float* pNode = nodeT.data_ptr<float>();
    float* pSeq = seqT.data_ptr<float>();
    float* pQ = qT.data_ptr<float>();
    float* pLegal = legalT.data_ptr<float>();
    float* pFeat = actFeatT.data_ptr<float>();
    int64_t* pSample = actSampleT.data_ptr<int64_t>();
    int64_t* pId = actIdT.data_ptr<int64_t>();

    int64_t off = 0;
    int64_t i = 0;
    for (const TeacherSample& s : samples) {
        std::copy(s.node.begin(), s.node.end(), pNode + i * kNodeInput);
        std::copy(s.seq.begin(), s.seq.end(), pSeq + i * kSeqInput);
        std::copy(s.q.begin(), s.q.end(), pQ + i * kNumActions);
        for (const ActionOption& a : s.actions) {
            std::copy(a.feat.begin(), a.feat.end(), pFeat + off * kActionInput);
            pSample[off] = i;
            pId[off] = a.slot;
            pLegal[i * kNumActions + a.slot] = 1.0f;
            ++off;
        }
        ++i;
    }

    actor->train();
    const int64_t mbSize = std::max<int64_t>(1, std::min<int64_t>(minibatch, B));
    const int64_t nBatches = (B + mbSize - 1) / mbSize;
    const float t = std::max(temp, 1e-3f);

    double ceSum = 0.0;
    int64_t total = 0;
    for (int step = 0; step < steps; ++step) {
        torch::Tensor perm = torch::randperm(B, torch::TensorOptions().dtype(torch::kLong));
        for (int64_t b = 0; b < nBatches; ++b) {
            const int64_t start = b * mbSize;
            const int64_t end = std::min(start + mbSize, B);
            const int64_t curB = end - start;
            torch::Tensor mbIdx = perm.slice(0, start, end);

            torch::Tensor nodeMb = nodeT.index_select(0, mbIdx).to(device);
            torch::Tensor seqMb = seqT.index_select(0, mbIdx).to(device);
            torch::Tensor lookup = torch::full({B}, -1, torch::TensorOptions().dtype(torch::kLong));
            lookup.index_put_({mbIdx}, torch::arange(curB, torch::TensorOptions().dtype(torch::kLong)));
            torch::Tensor mapped = lookup.index_select(0, actSampleT);
            torch::Tensor mask = mapped >= 0;
            torch::Tensor actIdx = mask.nonzero().squeeze(-1);
            if (actIdx.numel() == 0) continue;

            torch::Tensor actFeatMb = actFeatT.index_select(0, actIdx).to(device);
            torch::Tensor actSampleMb = mapped.index_select(0, actIdx).to(device);
            torch::Tensor actIdMb = actIdT.index_select(0, actIdx).to(device);

            torch::Tensor logits = actor->forward(nodeMb, seqMb, actFeatMb, actSampleMb, actIdMb);
            torch::Tensor logProbs = torch::log_softmax(logits, 1);

            torch::Tensor legalF = legalT.index_select(0, mbIdx).to(device);
            torch::Tensor raw = qT.index_select(0, mbIdx).to(device);
            torch::Tensor cnt = legalF.sum(1, /*keepdim=*/true).clamp_min(1.0f);
            torch::Tensor mean = (raw * legalF).sum(1, /*keepdim=*/true) / cnt;
            torch::Tensor centered = (raw - mean) * legalF;
            torch::Tensor scale =
                (centered.pow(2).sum(1, /*keepdim=*/true) / cnt).sqrt().clamp_min(1e-3f);
            torch::Tensor tq = centered / scale / t;
            tq = tq.masked_fill(legalF < 0.5f, -1e9f);
            torch::Tensor tlog = torch::log_softmax(tq, 1);
            torch::Tensor ce = -(torch::exp(tlog) * logProbs).sum(1).mean();

            actorOpt->zero_grad();
            (coef * ce).backward();
            torch::nn::utils::clip_grad_norm_(actor->parameters(), config.maxGradNorm);
            actorOpt->step();

            ceSum += ce.item<double>() * double(curB);
            total += curB;
        }
    }
    return total > 0 ? ceSum / double(total) : 0.0;
}

double Learner::updateCriticReplay(const std::deque<CriticSample>& samples, int epochs,
                                   int minibatch) {
    const int64_t B = static_cast<int64_t>(samples.size());
    if (B < 8 || epochs <= 0) return 0.0;

    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor nodeT = torch::empty({B, kNodeInput}, f32);
    torch::Tensor seqT = torch::empty({B, kHistSteps, kHistDim}, f32);
    torch::Tensor extraT = torch::empty({B, kExtraInput}, f32);
    torch::Tensor retT = torch::empty({B}, f32);

    float* pNode = nodeT.data_ptr<float>();
    float* pSeq = seqT.data_ptr<float>();
    float* pExtra = extraT.data_ptr<float>();
    float* pRet = retT.data_ptr<float>();
    int64_t i = 0;
    for (const CriticSample& s : samples) {
        std::copy(s.node.begin(), s.node.end(), pNode + i * kNodeInput);
        std::copy(s.seq.begin(), s.seq.end(), pSeq + i * kSeqInput);
        std::copy(s.extra.begin(), s.extra.end(), pExtra + i * kExtraInput);
        pRet[i] = s.ret;
        ++i;
    }

    critic->train();
    const int64_t mbSize = std::max<int64_t>(1, std::min<int64_t>(minibatch, B));
    const int64_t nBatches = (B + mbSize - 1) / mbSize;
    double sum = 0.0;
    int64_t total = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        torch::Tensor perm = torch::randperm(B, torch::TensorOptions().dtype(torch::kLong));
        for (int64_t b = 0; b < nBatches; ++b) {
            const int64_t start = b * mbSize;
            const int64_t end = std::min(start + mbSize, B);
            torch::Tensor mbIdx = perm.slice(0, start, end).to(device);
            torch::Tensor values =
                critic->forward(nodeT.index_select(0, mbIdx), seqT.index_select(0, mbIdx),
                                extraT.index_select(0, mbIdx))
                    .squeeze(-1);
            torch::Tensor retMb = retT.index_select(0, mbIdx).to(device);
            torch::Tensor vLoss = config.vfCoef * 0.5f * (values - retMb).pow(2).mean();
            criticOpt->zero_grad();
            vLoss.backward();
            torch::nn::utils::clip_grad_norm_(critic->parameters(), config.criticMaxGradNorm);
            criticOpt->step();
            sum += vLoss.item<double>() * double(end - start);
            total += end - start;
        }
    }
    return total > 0 ? sum / double(total) : 0.0;
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
    // 旧检查点的 Adam 状态没有 aux_* 参数；状态张量数量/形状对不上时直接
    // 放弃恢复（相当于热启动但动量从零开始），不影响模型权重本身。
    if (std::filesystem::exists(dir + "/actor_opt.pt")) {
        try {
            torch::serialize::InputArchive a;
            a.load_from(dir + "/actor_opt.pt");
            actorOpt->load(a);
        } catch (const std::exception&) {
            std::printf("  !! actor optimizer state 与当前参数不匹配，动量从零开始\n");
        }
    }
    if (std::filesystem::exists(dir + "/critic_opt.pt")) {
        try {
            torch::serialize::InputArchive c;
            c.load_from(dir + "/critic_opt.pt");
            criticOpt->load(c);
        } catch (const std::exception&) {
            std::printf("  !! critic optimizer state 与当前参数不匹配，动量从零开始\n");
        }
    }
}

}  // namespace texas
