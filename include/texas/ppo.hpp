#pragma once

// PPO 的数学部分：GAE 优势估计与带掩码的 softmax。
// 和 PerfectDouCPP 的实现是同一套（γ=1 的回合制牌类游戏）。

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace texas {
namespace ppo {

struct Config {
    float lr = 3e-4f;
    float clip = 0.2f;
    float entCoef = 0.01f;
    float vfCoef = 0.5f;
    int epochs = 4;
    // 策略更新之后额外给 critic 的轮数。critic 学不准时，价值基线会把"稳输
    // 一小注"的弃牌算成好棋 —— 实测会出现"全弃"塌缩（见 README）。
    int criticExtraEpochs = 8;
    int minibatch = 512;
    float maxGradNorm = 0.5f;
    float criticMaxGradNorm = 5.0f;
    float gamma = 1.0f;
    float lambda = 0.95f;
    float targetKL = 0.03f;
};

struct Stats {
    double retMean = 0;   // 回报均值 / 标准差：判断价值网在学什么
    double retStd = 0;
    double valueMean = 0;
    double pgLoss = 0;
    double vLoss = 0;
    double entropy = 0;
    double kl = 0;
    double clipFraction = 0;
    int epochsCompleted = 0;
};

template <typename TransitionT>
void assignEpisodeReturns(std::vector<TransitionT>& tr, float gamma, float lambda) {
    const int N = static_cast<int>(tr.size());
    if (N == 0) return;

    std::unordered_map<int, int> gidMap;
    std::vector<int> local(static_cast<size_t>(N), -1);
    int numGames = 0;
    for (int i = 0; i < N; ++i) {
        const int gid = tr[static_cast<size_t>(i)].gameId;
        if (gid < 0) continue;
        auto it = gidMap.find(gid);
        if (it == gidMap.end()) {
            it = gidMap.emplace(gid, numGames++).first;
        }
        local[static_cast<size_t>(i)] = it->second;
    }

    std::vector<int> head(static_cast<size_t>(numGames), -1);
    std::vector<int> next(static_cast<size_t>(N), -1);
    for (int i = 0; i < N; ++i) {
        const int g = local[static_cast<size_t>(i)];
        if (g < 0) continue;
        next[static_cast<size_t>(i)] = head[static_cast<size_t>(g)];
        head[static_cast<size_t>(g)] = i;
    }

    const float gaeFactor = gamma * lambda;
    for (int g = 0; g < numGames; ++g) {
        int curr = head[static_cast<size_t>(g)];
        float gae = 0.0f;
        float nextValue = 0.0f;
        bool first = true;
        while (curr >= 0) {
            auto& t = tr[static_cast<size_t>(curr)];
            const float vNext = first ? 0.0f : nextValue;
            const float delta = t.reward + gamma * vNext - t.value;
            gae = delta + (t.terminal ? 0.0f : gaeFactor * gae);
            t.adv = gae;
            t.ret = t.adv + t.value;
            nextValue = t.value;
            first = false;
            curr = next[static_cast<size_t>(curr)];
        }
    }
}

inline void maskedSoftmax(const float* logits, int n, float* probs) {
    float mx = -1e30f;
    for (int a = 0; a < n; ++a) {
        if (logits[a] > mx) mx = logits[a];
    }
    float sum = 0.0f;
    for (int a = 0; a < n; ++a) {
        const float v = (logits[a] < -1e8f || !std::isfinite(logits[a]))
                            ? 0.0f
                            : std::exp(logits[a] - mx);
        probs[a] = v;
        sum += v;
    }
    if (sum > 0.0f && std::isfinite(sum)) {
        const float inv = 1.0f / sum;
        for (int a = 0; a < n; ++a) probs[a] *= inv;
    } else {
        const float uni = 1.0f / static_cast<float>(n);
        for (int a = 0; a < n; ++a) probs[a] = uni;
    }
}

}  // namespace ppo
}  // namespace texas
