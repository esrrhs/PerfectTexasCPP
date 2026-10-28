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
    // 策略更新之后额外给 critic 的轮数。早期版本用 8，理由是价值基线不准时
    // PPO 会把"稳输一小注"的弃牌算成好棋。但 600 轮同种子 A/B（见 README
    // 第八节第 12 条）显示：额外 epoch 越多，critic 越贴上一批数据、跨批泛化
    // 越差；0 反而 mix 最高、速度还快 1.5 倍。默认改成 0。
    int criticExtraEpochs = 0;
    int minibatch = 512;
    float maxGradNorm = 0.5f;
    float criticMaxGradNorm = 5.0f;
    // critic 的 L2 正则。离线诊断（critfit 的 8:2 留出集）显示：不加正则时
    // 价值网在几百步内就能把训练集记住，留出集相关 0.65；加到 1e-3 后留出集
    // 相关 0.85、RMSE 0.20（回报标准差 0.34）。线上每轮只有约 2000 条样本，
    // 过拟合正是价值基线吵（c≈0.2~0.4）的直接原因。
    float criticWeightDecay = 0.0f;
    float gamma = 1.0f;
    float lambda = 0.95f;
    float targetKL = 0.03f;
    // 脚本教师蒸馏：把 teacherQ / temp 的 softmax 当作动作分布，对 actor
    // 加一项交叉熵。它不进 PPO 优势，只负责把"看牌"这件事灌进策略。
    float teacherCoef = 0.0f;
    // Q 已在学习器里按状态标准化（减均值、除合法动作标准差），所以温度
    // 是"相对动作差"的尺度：0.5 比较软，0.1 接近 one-hot。
    float teacherTemp = 0.5f;
    // 范围胜率辅助头：用不完全信息回归开天眼胜率，MSE 系数。0 = 关闭
    // （关闭时信念头仍参与前向，但残差支路末层零初始化，策略等价于旧模型）。
    float auxEquityCoef = 0.0f;
    // 对手建模头：用不完全信息回归"对手当前牌力百分位"，MSE 系数。0 = 关闭。
    // 只塑形信念头与残差支路；策略梯度不会被校准目标污染（输入已 detach）。
    float oppModelCoef = 0.0f;
    // 状态条件信任域：对 pot < anchorMaxPot 的样本加 KL(新策略 || 冻结基座)。
    // 大池教师只会改大池，中池/小池保持基座，避免第 75 条那种"大池赚、
    // 中池亏"的风格漂移。
    float anchorCoef = 0.0f;
    float anchorMaxPot = 1500.0f;
    // 规则教师蒸馏系数：把"河牌大池 -> 1.5x"这类由外部范围导出的规则
    // 写进策略（交叉熵），不依赖部署 override。
    float ruleCoef = 0.0f;
};

struct Stats {
    double retMean = 0;   // 回报均值 / 标准差：判断价值网在学什么
    double retStd = 0;
    double valueMean = 0;
    double valueStd = 0;
    double valueCorr = 0;  // 价值预测与回报的相关系数；低了说明基线没学到东西
    double advStd = 0;
    double pgLoss = 0;
    double vLoss = 0;
    double entropy = 0;
    double kl = 0;
    double clipFraction = 0;
    double teacherLoss = 0;  // 教师交叉熵（未乘系数），看蒸馏有没有在收敛
    double teacherEntropy = 0;  // 教师目标分布自身的熵；CE - H = KL(policy || teacher)
    double teacherMax = 0;   // 教师目标分布的 Top-1 概率均值（均匀 = 1/n，尖 = 接近 1）
    double auxLoss = 0;      // 范围胜率回归 MSE
    double auxMae = 0;       // 平均绝对误差（和胜率同量纲，便于看校准）
    double auxPred = 0;      // 预测均值
    double auxLabel = 0;     // 标签均值（两者接近说明没有系统性偏移）
    double oppLoss = 0;      // 对手牌力回归 MSE
    double oppMae = 0;
    double oppPred = 0;
    double oppLabel = 0;
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
