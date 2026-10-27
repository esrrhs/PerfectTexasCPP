#pragma once

// 强度评估。零和对称游戏里"自对局恒为 0"，所以必须有固定基准：
//   规则机器人（牌力百分位 + 底池赔率）—— 相当于 PerfectDou 里 DouZero 的角色
//   均匀随机 —— 只做下限标尺
// 每个基准跑两遍（模型坐小盲 / 坐大盲），合并成模型每手净收益。

#include <cstdint>
#include <vector>

#include "texas/rollout.hpp"

namespace texas {

struct EvalResult {
    int hands = 0;
    double chipsPerHand = 0.0;
    double stderr = 0.0;
    double bbPer100 = 0.0;   // 每百手大盲数
    double mbbPerGame = 0.0; // ACPC 标准口径：每手千分之一大盲（1 bb/100 = 10 mbb/g）
    // 同牌对调拆成两个座位看：a 坐小盲 / 坐大盲时的每手净收益。
    // 两个座位不对称，合并数字会掩盖"某一侧明显更弱"。
    double sbChipsPerHand = 0.0;
    double sbStderr = 0.0;
    double bbChipsPerHand = 0.0;
    double bbStderr = 0.0;
};

// 动作级集成：每个座位一组 actor，actor 是主模型，extra 是额外成员。
// 每次决策各前向一遍，把 masked softmax 的概率平均后采样。和参数平均（soup）
// 不同，它不要求两个检查点落在同一个损失盆地，所以跨 seed / 跨课程的模型
// 可以直接组合。评估开销 = 成员数 × 单模型前向。
// 逐手记录（对打用）：被测座位的净收益 + 终局底池 + 座位。
// 用来按底池分桶对照"内部自对局 vs 外部基准"的分布。
struct MatchLog {
    std::vector<int> seats;
    std::vector<int> pots;
    std::vector<int> nets;
};

struct EvalTeam {
    const nn::Actor* actor[2] = {nullptr, nullptr};
    const nn::Actor* extra[2][kMaxEnsemble] = {};
    int extraCount[2] = {0, 0};
    // 动作选择：默认 temp=1 且按分布采样（和训练一致）。temp<1 更尖，
    // greedy=true 取 argmax。这两种都会改变"策略"本身，所以只用于评估/对局。
    float policyTemp = 1.0f;
    bool policyGreedy = false;
    // 尺寸带内的金额固定取对数中点（去掉带内随机）。
    bool amountMid = false;
    // 翻前加注尺寸放大系数（部署剥削用，默认 1）。
    float preflopRaiseMult = 1.0f;
    // 后翻牌下注/加注尺寸放大系数（部署剥削用，默认 1）。
    float postflopBetMult = 1.0f;

    // 单模型团队：和原来的 evalVs* 行为完全一致。
    static EvalTeam of(const nn::Actor actors[2]) {
        EvalTeam t;
        t.actor[0] = &actors[0];
        t.actor[1] = &actors[1];
        return t;
    }
};

EvalResult evalVsRule(const nn::Actor actors[2], int games, uint64_t seed, int equityIters = 24);
EvalResult evalVsRandom(const nn::Actor actors[2], int games, uint64_t seed, int equityIters = 24);
EvalResult evalVsPro(const nn::Actor actors[2], int games, uint64_t seed, int equityIters = 24);
EvalResult evalVsSolid(const nn::Actor actors[2], int games, uint64_t seed, int equityIters = 24);
EvalResult evalVsSearch(const nn::Actor actors[2], int games, uint64_t seed, int equityIters = 24);
// 混合基准：每局随机抽规则/胜率/搜索/随机之一当对手，座位也随机。
// 这是比单个脚本更接近"成熟 AI 组合"的评估，不需要配对发牌。
EvalResult evalVsMix(const nn::Actor actors[2], int games, uint64_t seed, int equityIters = 24);
EvalResult evalSelfPlay(const nn::Actor actors[2], int games, uint64_t seed, int equityIters = 24);

// 动作级集成的评估入口：语义与上面完全相同，只是被测策略换成 EvalTeam。
EvalResult evalTeamVsRule(const EvalTeam& team, int games, uint64_t seed, int equityIters = 24);
EvalResult evalTeamVsRandom(const EvalTeam& team, int games, uint64_t seed, int equityIters = 24);
EvalResult evalTeamVsPro(const EvalTeam& team, int games, uint64_t seed, int equityIters = 24);
EvalResult evalTeamVsSolid(const EvalTeam& team, int games, uint64_t seed, int equityIters = 24);
EvalResult evalTeamVsSearch(const EvalTeam& team, int games, uint64_t seed, int equityIters = 24);
EvalResult evalTeamVsMix(const EvalTeam& team, int games, uint64_t seed, int equityIters = 24);

// 策略内搜索的评估入口：被测座位用 policySearchMove（一步前瞻 + 策略续演）。
// samples = 每个决策抽的隐藏牌样本数；越大越准，开销线性增加。
EvalResult evalSearchTeamVsRule(const EvalTeam& team, int games, uint64_t seed, int samples,
                                int equityIters = 24);
EvalResult evalSearchTeamVsMix(const EvalTeam& team, int games, uint64_t seed, int samples,
                               int equityIters = 24);

// 两个检查点对打（同牌对调）：返回 a 对 b 的每手净收益。
// 只适合同一套特征口径的检查点（特征改过之后旧 ckpt 的语义会变）。
EvalResult evalMatch(const nn::Actor a[2], const nn::Actor b[2], int games, uint64_t seed,
                     int equityIters = 24);
EvalResult evalTeamMatch(const EvalTeam& a, const nn::Actor b[2], int games, uint64_t seed,
                         int equityIters = 24, float bTemp = 1.0f, bool bGreedy = false,
                         MatchLog* log = nullptr, float bPreflopRaiseMult = 1.0f,
                         float bPostflopBetMult = 1.0f);
// a 侧用策略内搜索，b 侧照常按策略采样：测"搜索是否真的比原策略强"。
EvalResult evalSearchTeamMatch(const EvalTeam& a, const nn::Actor b[2], int games, uint64_t seed,
                               int samples, int equityIters = 24);

}  // namespace texas
