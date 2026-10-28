#pragma once

// 强度评估。零和对称游戏里"自对局恒为 0"，所以必须有固定基准：
//   规则机器人（牌力百分位 + 底池赔率）—— 相当于 PerfectDou 里 DouZero 的角色
//   均匀随机 —— 只做下限标尺
// 每个基准跑两遍（模型坐小盲 / 坐大盲），合并成模型每手净收益。

#include <cstdint>
#include <string>
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
    // rangeEq 特征采样数（必须与训练时一致，否则残差支路输入分布不同）。
    int rangeEqSamples = 0;
    int rangeEqIters = 48;
    // 河牌范围条件化搜索的对手底牌样本数（0 = 关）。见 policySearchMove。
    int rangeSearchSamples = 0;
    // 消融：河牌搜索不按对手动作似然加权（均匀隐藏牌）。
    bool rangeSearchUniform = false;
    // 鲁棒范围搜索：根动作 EV 取"自身 actor / solid / shark"三种对手模型的最坏情况。
    bool robustSearch = false;
    bool robustMean = false;
    // 河牌搜索的最小底池（筹码，0 = 全部）。
    int rangeSearchMinPot = 0;
    // 只在要跟注的河牌决策上用搜索（面对下注的防守）。
    bool rangeSearchDefenseOnly = false;
    // 河牌 CFR+ 子博弈求解器（README #105）：>0 = 河牌决策改用范围均衡动作。
    int riverCfrIters = 0;
    int riverCfrMinPot = 0;
    int riverCfrMaxRaises = 1;
    bool riverCfrAllin = true;
    // 「读范围机器人」的对手范围里空气（诈唬）占比，见 RolloutConfig::rangerAir。
    float rangerAir = 0.28f;
    // 对手座位用河牌 CFR 均衡回应（见 RolloutConfig::oppCfrProb）。0 = 关。
    int oppCfrIters = 0;
    int oppCfrMinPot = 1500;

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
// 团队版自对局：温度 / 集成 / rangeEq 等开关都会生效（否则 --vs self 会
// 静默忽略 --range-eq，诊断时容易把"关了特征"的数当成"开了特征"）。
EvalResult evalTeamSelfPlay(const EvalTeam& team, int games, uint64_t seed,
                            int equityIters = 24);

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

// 摊牌诊断（README #107）。外部 12000 手上，赤字全部来自摊牌：摊牌只赢 20.4%、
// 每手 −1574 筹码，而我们摊牌时 22% 是高牌（对手 7%）。这个结构在 12000 手
// 外部采样里是 6σ 级，但外部测量慢、方差大。这里把它做成**内部**可复现的
// 测量：对任意脚本对手打 N 手，按"摊牌/非摊牌"拆收益，并统计双方牌型分布。
// 9 个桶对应 `handCategoryOf`：0 高牌、1 一对、2 两对、3 三条、4 顺子、
// 5 同花、6 葫芦、7 四条、8 同花顺。
struct ShowdownStats {
    int hands = 0;
    int showdowns = 0;
    int wins = 0;
    int ties = 0;
    double sdSum = 0.0;      // 摊牌手的净收益和
    double sdSumSq = 0.0;
    double nsdSum = 0.0;     // 非摊牌手的净收益和
    double nsdSumSq = 0.0;
    int myCat[9] = {};
    int oppCat[9] = {};
    // 按终局底池分桶的摊牌收益（0: <1500, 1: 1500-4000, 2: >=4000）。
    int bucketSd[3] = {};
    double bucketSdSum[3] = {};
    double bucketSdSumSq[3] = {};
    void print(const char* tag, int writtenHands) const;
};

// 对指定脚本对手（rule/random/pro/solid/search）打 games 手（两个座位各一遍），
// 同时输出常规 EvalResult 与摊牌诊断。opp 不合法时 out 保持空、返回 EV 为 0。
EvalResult evalShowdownVs(const EvalTeam& team, const std::string& opp, int games,
                          uint64_t seed, int equityIters, ShowdownStats& out);

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
