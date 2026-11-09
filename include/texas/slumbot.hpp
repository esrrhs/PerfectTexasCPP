#pragma once

// Slumbot 公开 API 客户端：用真实的成熟 AI 来测强度。
//
// Slumbot 的规则和本引擎一致：盲注 50/100、起始筹码 20000（200BB）、每手重置。
// 协议（https://www.slumbot.com，/slumbot/api/...）：
//   POST /slumbot/api/new_hand  {"token": "<上次响应里的 token>"}
//   POST /slumbot/api/act       {"token": "...", "incr": "c"}
// 响应里 action 是完整的本手动作串（b200c/kb400/...），winnings 出现时本手结束。
// incr 语义：k = check，c = call，f = fold，b<金额> = 本街累计投入额。
//
// 我们只把它当**对手**与**计分器**：策略仍然来自本地 actor。

#include <cstdint>
#include <string>
#include <vector>

#include "texas/hu.hpp"
#include "texas/nn.hpp"
#include "texas/rollout.hpp"

namespace texas {

struct SlumbotConfig {
    int hands = 200;
    uint64_t seed = 20261026ULL;
    float temp = 0.1f;     // 部署提取温度（当前冠军是 0.1）
    int delayMs = 150;     // 每步之间的礼貌延迟（并发会话会被限流，必须串行）
    std::string host = "slumbot.com";
    // 诊断用：actor = 用神经网络；fold = 面对下注就弃、否则过牌；
    // random = 合法动作里均匀随机。后两者用来校验客户端记账（永远弃牌
    // 的期望是 -75 筹码/手，大小盲交替）。
    std::string policy = "actor";
    // 可选：逐手 CSV 日志
    // (hand,client_pos,winnings,adjusted,pot,allin_adjusted,action)。
    // pot 来自终局响应的 won_pot（带符号，取绝对值），按底池大小诊断漏洞；
    // action 是完整动作串，用来定位"哪条街、弃牌还是跟注"在漏。
    std::string logPath;
    // 部署剥削：3-bet（面对开牌时的再加注）金额放大系数（>=1，默认 1）。
    float preflop3betMult = 1.0f;
    // 部署剥削：后翻牌下注/加注金额放大系数（>=1，默认 1）。
    float postflopBetMult = 1.0f;
    // 河牌大注剥削：河牌无人下注且底池 >= minPot 时，直接用 1.5x 池下注。
    // 外部 900 手河牌数据：>=1500 池的 533 个决策里 92% 的最优动作是 >1.3x，
    // Slumbot 面对 >1.3x 的河牌下注弃 76.8%（纯诈唬只需 64% 才保本）。
    float riverOverbetPot = 0.0f;   // 0 = 关闭
    int riverOverbetMinPot = 1500;
    // 部署"全下纪律"guard：面对会打光筹码的全下时，只用手牌百分位足够高的
    // 范围跟注。外部数据（2763 手 solid2 + 1538 手 tight1）：我们的全下跟注
    // 平均调整后 EV 约 -1BB/手，Slumbot 的全下几乎总是顶范围。
    // prePct 作用于翻前（对随机手牌胜率），postPct 作用于翻后（牌力百分位）。
    float allinGuardPrePct = 0.0f;  // 翻前胜率门槛（0 = 关闭）
    int allinGuardPostCat = -1;     // 翻后最低牌力类别（-1 = 关闭；3 = 三条）
    // 翻前进攻上限：手牌胜率 < preflopAggroPct 时，面对 numRaises >=
    // preflopAggroMaxRaises 的局面不再加注（改成跟注/过牌）。
    // 外部路径分析：我们的 4-bet/5-bet 常用 pct 0.43~0.66 的牌，把底池
    // 建到 200BB 再被顶范围接住；这条规则直接掐掉"用中等牌建大池"。
    float preflopAggroPct = 0.0f;   // 0 = 关闭
    int preflopAggroMaxRaises = 2;  // 允许到 3-bet（numRaises < 2 时不受限）
    // BB 面对开牌（numRaises==1）的 3-bet 门槛：翻前手牌百分位低于它时不 3-bet，
    // 改成跟注。注意：早期 200~400 手分桶曾显示 0.45~0.60 档 3-bet 亏 2~4.5 BB/手，
    // 12k 手重算后只有 −0.9~−1.1，与跟注 EV（−0.86~−1.42）几乎相同 ⇒ 增量≈0。
    // 保留开关作对照，默认关闭（README #113(d)）。0 = 关闭。
    float bb3betMinPct = 0.0f;
    // 翻牌进攻上限：牌力百分位 < flopAggroPct 时不加注（改成跟注/过牌）。
    // 路径分析：翻牌用 pct 0.40~0.60 的成牌/听牌加注，是把底池建到 200BB
    // 的第二大来源。
    float flopAggroPct = 0.0f;      // 0 = 关闭
    float allinGuardMinRisk = 0.5f;  // 跟注额 >= 剩余筹码的这个比例才触发
    // 部署一致性：训练 rollout 每个决策都会算 rangeEq 并喂给范围残差支路，
    // 而部署路径早先固定喂 0.5 —— 同一份权重在两边看到的输入不同。>0 时
    // 按训练同一口径（对手最近动作似然 + 采样底牌 + 摊牌胜率）真算。
    // 对手模型用我们自己（训练里也是"对手=当前策略"）。
    int rangeEqSamples = 0;  // 0 = 关闭（喂中性 0.5）
    int rangeEqIters = 48;
    // rangeEq 的对手似然来源：0 = 我们的策略（训练同口径）；1 = 经验模型
    // （Slumbot 日志拟合的"动作类型 × 手牌百分位"密度，仅部署有意义）。
    int rangeModel = 0;
    // 转牌 CFR+ 子博弈求解器：>0 = 转牌决策改用范围均衡动作（叶子 = 发河牌
    // 后 check-down 摊牌；对手范围 = 1128 个组合按对手最近动作的策略似然）。
    // 与 eval 的 --turn-cfr 同一实现；0 = 关闭（默认，转牌走网络策略）。
    int turnCfrIters = 0;
    int turnCfrMinPot = 0;  // 只在底池 >= 该值（筹码）的转牌决策上求解（0=全部）
    // 河牌 CFR+ 子博弈（与内部 eval 的 --river-cfr 同一实现）。河牌求解便宜
    // （1081 组合、每手 ~20ms），可以在外部整段开。0 = 关闭。
    int riverCfrIters = 0;
    int riverCfrMinPot = 0;
    int riverCfrMaxRaises = 1;
    bool riverCfrAllin = true;
    // R38：SB 首次开牌的目标尺寸（筹码；0 = 关闭）。外部 12k 实测：我们 SB
    // 开牌集中在 3.0~3.6BB，而 Slumbot 只加到 1.8BB。SB 被 3-bet 弃牌时直接
    // 损失开牌额，开牌越大，弃牌成本越高（1,404 手 3-bet 底池合计 −0.449 BB/手）。
    int sbOpenTo = 0;
    // R38：面对 3-bet / 4-bet（翻前 numRaises >= 2）的继续门槛，两个座位都适用。
    // 外部 12k 实测：SB 被 3-bet 后 弃 −3.7 / 跟 −6.9 / 4-bet −9.2 BB/手；
    // BB 被 4-bet 后 弃 −5.0 / 跟 −24.4 / 5-bet −77 BB/手 → 除顶端范围外继续全是亏的。
    // 手牌「对随机胜率」>= pct 才继续，否则弃牌。
    float preflopFoldVs3betPct = 0.0f;
    // R39：翻牌无注可跟（need==0）时的高频下注限制。外部 12k 实测：SB 开牌被跟后
    // 我们翻牌 c-bet 92.7%（EV −0.766 BB/手），同口径 Slumbot 只 c-bet 59.9%。
    // 牌力百分位 < pct 时把下注改成过牌（0 = 关闭）。
    float flopCheckPct = 0.0f;
    // R40：BB 面对开牌（numRaises==1）时的加宽 3-bet。外部 12k 实测：BB 弃 −1.00 /
    // 跟 −1.19 / 3-bet −0.085 BB/手 → 用弃/跟的牌去 3-bet 是唯一便宜选项。
    // 牌力百分位 < pct 时把弃牌/跟注改成 3-bet（0 = 关闭）。
    float bb3betBelowPct = 0.0f;
    int bb3betTo = 0;  // 加宽 3-bet 的目标尺寸（筹码；0 = 取 3×跟注额）
    // R41：河牌「无注可跟」的诈唬尺寸上限。外部 12k 逐手精确对账（我们与对手底牌全知）：
    // 河牌下注 779 手里 eq<0.50 的 417 手，改成过牌可回收 +0.286 BB/手；其中
    // >1.5× 池的 50 手单独值 +0.157，而 ≤0.5× 池只有 +0.025（几乎中性）。
    // 即「诈唬可以打，但别用大尺寸」：手牌百分位 < pct 时把下注压到 ≤ ratio×底池。
    float riverBluffPct = 0.0f;      // 0 = 关闭
    float riverBluffMaxRatio = 0.5f; // 压到的目标尺寸（占下注前底池）
    // R42：翻前被 3-bet/4-bet 后不再加注（4-bet/5-bet 一律改跟注）。外部 12k 实测：
    // SB 被 3-bet 后 4-bet −9.24 vs 跟 −6.89；而 4-bet 是 200BB 全下底池的唯一起点
    // （4bet+ 打到河牌的 36 手 −84 BB/手，合计 −0.239 BB/手）→ 不加注就整段消失。
    bool preflopNo4bet = false;
    // R44：转牌「无注可跟」的弱牌下注尺寸上限（R41 机制外推到转牌）。12k 逐手对账：
    // 转牌 2,301 次开池下注里 1,330 次 eq<0.5，全部压到 ≤0.6× 池的粗算净 +0.06 BB/手
    // （弱牌 +0.186、价值 −0.125；有强度门槛时可只取弱牌那部分）。
    float turnBluffPct = 0.0f;
    float turnBluffMaxRatio = 0.6f;
};

struct SlumbotResult {
    int hands = 0;      // 完成并有 winnings 的手数（被网络错误中止的手不计分）
    int attempted = 0;  // 发起的手数；attempted - hands = 中止手数
    int riverOverbets = 0;  // 河牌大注 override 触发次数（诊断用）
    int allinGuardFolds = 0;  // 全下纪律 guard 强制弃牌次数（诊断用）
    int preflopAggroCapped = 0;  // 翻前进攻上限触发次数（诊断用）
    int bb3betCapped = 0;        // BB 中等牌 3-bet 改成跟注的次数（诊断用）
    int flopAggroCapped = 0;     // 翻牌进攻上限触发次数（诊断用）
    int sbOpenCapped = 0;        // R38：SB 开牌尺寸被压小的次数（诊断用）
    int preflopVs3betFolds = 0;  // R38：SB 面对 3-bet 被改成弃牌的次数（诊断用）
    int flopCheckCapped = 0;     // R39：翻牌下注被改成过牌的次数（诊断用）
    int bb3betWidened = 0;       // R40：BB 弃/跟被改成 3-bet 的次数（诊断用）
    int riverBluffCapped = 0;    // R41：河牌诈唬尺寸被压小的次数（诊断用）
    int preflopCappedToCall = 0; // R42：翻前 3-bet/4-bet 后加注被改成跟注的次数
    int turnBluffCapped = 0;     // R44：转牌弱牌下注被压小的次数（诊断用）
    int rangeEqDecisions = 0;    // 真实算过 rangeEq 的决策数（诊断用）
    int turnCfrDecisions = 0;    // 转牌子博弈求解器接管的决策数（诊断用）
    int riverCfrDecisions = 0;   // 河牌子博弈求解器接管的决策数（诊断用）
    long long winnings = 0;  // 净筹码（正 = 我们赢）
    double chipsStderr = 0.0;  // 每手净收益的标准误（按逐手结果算）
    // 河牌前全下的手用全下时胜率折算底池（"EV 调整"），剔掉发牌运气。
    // 没调整的手仍用实际 winnings。它和 winnings 的差只是方差，不是偏差。
    double adjustedWinnings = 0.0;
    double adjustedStderr = 0.0;
    int allinAdjusted = 0;  // 被调整的手数
    int wins = 0;
    int losses = 0;
    int pushes = 0;
    int errors = 0;
};

// 打 cfg.hands 手，返回净收益。网络错误会中止并记录在 errors 里。
SlumbotResult playSlumbot(const nn::Actor actors[2], const SlumbotConfig& cfg);

// 河牌前双方全下时，用全下那一刻的精确胜率（枚举剩余公共牌）折算底池，
// 得到剔掉发牌运气后的净收益。返回 true 表示这手确实被调整了；否则
// adjusted 不变（调用方用实际 winnings）。
bool allinAdjustedWinnings(const std::string& action, const Card mine[2], const Card bot[2],
                           const Card board[5], int clientPos, long long winnings,
                           double& adjusted);

// 从 ACPC 风格的动作串 + 已知公共牌重建本地 HUState（用于选动作）。
// clientPos：0 = 我们是 BB，1 = 我们是 SB。失败返回 false。
bool buildSlumbotState(const std::string& action, const std::string& hole0,
                       const std::string& hole1, const std::vector<std::string>& board,
                       int clientPos, Rng& rng, HUState& out);

// 重建"对手（clientPos 座位）最近一次决策"：动作前状态 + 动作槽。
// 部署时算 rangeEq 需要它（与训练 rollout 的 g.lastAct 同一口径）。
bool slumbotLastOppDecision(const std::string& action, const std::string& hole0,
                            const std::string& hole1, const std::vector<std::string>& board,
                            int clientPos, Rng& rng, RangeObs& out);

// 同时重建对手与我们自己（1-clientPos 座位）的最近一次决策。转牌子博弈
// 求解器需要**双方**范围（均衡策略取决于双方范围，见 riverCfr 的 ownRange）。
bool slumbotLastDecisions(const std::string& action, const std::string& hole0,
                          const std::string& hole1, const std::vector<std::string>& board,
                          int clientPos, Rng& rng, RangeObs& lastOpp, RangeObs& lastSelf);

// 全下纪律：面对会打光筹码的全下，且跟注额 >= minRisk * 剩余筹码时，
// 手牌百分位低于阈值就返回 true（应当弃牌）。翻前用 prePct（对随机胜率），
// 翻后用 postPct（handStrength 百分位）。任一阈值为 0 表示该街不启用。
bool allinGuardFold(const HUState& st, float prePct, int postCat, float minRisk);

}  // namespace texas
