#pragma once

// 自对局采样：一批牌局按座位锁步推进，同一座位所有待决策的牌局做一次批量
// 前向。每个座位一条独立的 transition 流，交给 PPO 的 GAE 用。

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "texas/features.hpp"
#include "texas/hu.hpp"
#include "texas/nn.hpp"

namespace texas {

struct ActionOption {
    int slot = 0;
    Move mv;   // 尺寸带时，mv.to 是带的下端点（抽象动作的锚）
    int hi = 0;  // 尺寸带的上端点；带内任何整数金额合法
    std::array<float, kActionInput> feat{};
};

struct Transition {
    std::array<float, kNodeInput> node{};
    std::array<float, kSeqInput> seq{};
    std::array<float, kExtraInput> extra{};
    std::vector<ActionOption> actions;  // 本节点的全部合法动作
    int action = -1;                    // 采样到的 slot
    int gameId = -1;
    bool terminal = false;
    float logp = 0.0f;
    float value = 0.0f;
    float reward = 0.0f;
    float adv = 0.0f;
    float ret = 0.0f;
    // 脚本教师的反事实动作价值（同牌分支续演，单位 = 起始筹码）。
    // 只有 hasTeacher 为真的样本参与蒸馏损失；索引是 slot，非法 slot 填 0。
    std::array<float, kNumActions> teacherQ{};
    bool hasTeacher = false;
    // 规则教师目标槽位（-1 = 无）。目前用于"河牌大池无人下注 -> 1.5x"，
    // 把外部范围诊断得到的剥削规则蒸馏进策略，而不是长期靠部署 override。
    int ruleSlot = -1;
};

struct Snapshot {
    nn::Actor actor[2];
    int update = 0;
};

// 两级历史池，对应 PerfectDouCPP 的 HistoricalPool：
//
//   recent  —— 最近 N 个快照，滚动淘汰。管"战术演化的最近几步"，
//              让对手还记得自己上一分钟长什么样。
//   archive —— 几何间隔保留的旧快照，永不整体丢弃。管"灾难性遗忘"：
//              只有 recent 时，池子永远只覆盖最近 N 轮，策略一旦跑偏就再也
//              遇不到早期的自己，很容易在同一个坑里反复振荡。
//
// archive 满了不是从头丢，而是**隔一个丢一个**——保留的间隔翻倍，
// 于是越老的快照越稀疏、但永不消失（1,2,4,8,… 的几何间隔）。
struct HistoricalPool {
    std::vector<std::shared_ptr<Snapshot>> recent;
    std::vector<std::shared_ptr<Snapshot>> archive;

    bool empty() const { return recent.empty() && archive.empty(); }
    size_t size() const { return recent.size() + archive.size(); }

    // tier: 0 = recent, 1 = archive
    const Snapshot* at(int tier, int index) const;

    // 推入一个新快照并维护两个队列的容量。
    void push(const std::shared_ptr<Snapshot>& snap, int recentCap, int archiveCap);
};

struct ModelSet {
    nn::Actor actor[2];
    nn::Critic critic[2];
};

// 动作级集成最多叠加的额外模型数（不含主模型）。
constexpr int kMaxEnsemble = 4;

struct RolloutConfig {
    int games = 256;
    uint64_t seed = 20240518ULL;
    // 联赛配比。三个概率都是"**这一局**有一个座位是这类对手"的概率，
    // 另一个座位一定是当前策略（见 collectRollout 的按局分配）。
    float historicalProb = 0.30f;  // 历史快照
    float ruleProb = 0.15f;        // 规则机器人
    float randomProb = 0.05f;      // 均匀随机
    float proProb = 0.0f;          // 胜率机器人（风格与规则机器人不同）
    float searchProb = 0.0f;       // 搜索机器人（教师 argmax，风格又不同）
    // 跟注型"稳健"机器人：宽跟注、少诈唬、只用强牌加注。存在的理由是
    // 外部日志显示 Slumbot 的 BB 有 70% 决策是 check/call，而其余脚本对手
    // 都会对加注弃牌——缺了这层压力，策略会学成"3-bet/弃牌 + 翻牌全下注"。
    float solidProb = 0.0f;
    // 「紧全下」机器人：只用顶范围推/接全下，其余过牌/弃牌。外部明细显示
    // 我们的赤字全部来自全下决策（翻前用 K3s/A7o 接 200BB 全下、河牌用
    // 一对跟大注）；联赛里的脚本对手都爱全下诈唬，策略因此学会了宽接。
    // 这个机器人提供反向压力：全下几乎总是强牌，教策略"全下=紧范围"。
    float tightProb = 0.0f;
    // 「陷阱」机器人：强牌在前几条街只跟注/过牌，河牌才发力，且从不弃顶范围。
    // 外部路径分析显示我们的 4-bet/5-bet 常用 pct 0.43~0.66 的牌把底池建到
    // 200BB；这个对手专门"接住"这些牌，逼策略学会控池。
    float trapperProb = 0.0f;
    // 「价值型强对手」机器人：翻前紧凶（只用强牌 3-bet/4-bet，从不 5-bet 诈唬），
    // 翻后只用两对及以上的成牌建大池/加注，弱牌面对大注直接弃。外部 14.3k 手
    // 复盘显示赤字全在 0.4% 的 200BB 全下手，且都是我们自己用中等牌建池；
    // 现有脚本对手（solid 宽跟、trapper 只河牌、spammer 被加注就弃）都不惩罚
    // 建池。这个对手逼策略学会控池。
    float sharkProb = 0.0f;
    // 「小注骚扰」机器人：翻后频繁下注 1/3 池（不看牌力），逼策略学会
    // 防守小注。外部分析：BB 在 300-600 小池面对下注弃了 151 次，
    // 弃牌时中位胜率 0.414 vs 赔率 0.250——系统性地过度弃牌。
    float spammerProb = 0.0f;
    // 「按尺寸弃牌」机器人（模仿 Slumbot 面对下注的尺寸-弃牌曲线）。
    float folderProb = 0.0f;
    // 规则蒸馏：河牌、底池 >= ruleMinPot、无人下注时，给最接近
    // ruleTargetRatio 倍池的加注槽位打标签。
    float ruleMinPot = 0.0f;
    float ruleTargetRatio = 1.5f;
    // 大池专用对手：底池 >= bigPotThreshold 时，对手座位改用 bigPotOpponent
    // （不是换一整局，而是逐决策切换）。用途：把我们赤字最大的"≥20BB 大池"
    // 交给真实成熟 AI 的克隆模型去施压。0 = 关。
    const nn::Actor* bigPotOpponent[2] = {nullptr, nullptr};
    float bigPotThreshold = 0.0f;
    float bigPotTemp = 1.0f;
    // 尖自对局：这一局的对手座位用**当前权重**按 selfSharpTemp 提取动作
    // （部署用 T=0.1）。和"自对局"一样消耗 wSelf 的配额，但对手不留噪声动作。
    float selfSharpProb = 0.0f;
    float selfSharpTemp = 0.1f;
    // 同牌对调：相邻两局用**同一副牌**、两个座位各打一次（只交换底牌，
    // 位置/筹码不变）。牌运成分在成对样本里抵消，降低优势估计的方差。
    bool duplicateDeals = false;
    float archiveProb = 0.30f;     // 抽历史对手时，抽 archive 而不是 recent 的比例
    bool oracleReward = true;      // 开天眼稠密奖励
    float oracleCoef = 0.05f;
    int equityIters = 32;          // 开天眼胜率的抽样次数
    // 价值网是否吃"开天眼"特征（对手底牌 + 完美胜率）。默认 true。
    // 关掉之后 critic 只能看到 actor 也看得到的信息，它拟合的是信息集上的
    // 条件期望 E[return | 自己的信息]，函数更平滑、样本效率更高，但方差更大。
    // 两种 baseline 都是合法的（不依赖动作），这是方差 / 泛化的取舍。
    bool criticOracleExtra = true;
    // 脚本教师：对每个决策点枚举合法动作，用同一副牌做分支续演，得到
    // 低方差的动作价值 Q。Q 只用于蒸馏 actor（不进 PPO 优势），见 README。
    bool teacher = false;
    float teacherProb = 1.0f;
    // 只给"决策时底池 >= teacherMinPot"的样本做教师标注（0 = 不限制）。
    // 把蒸馏预算对准大池漏点：教师开天眼，在大池的"是否被 beat"上比
    // 不完全信息的 actor 可靠；小池样本则完全不标，省算力也避免带偏风格。
    float teacherMinPot = 0.0f;       // 每个决策点算教师的概率（控制开销）
    int teacherSamples = 1;         // >1：对隐藏牌（对手底牌 + 未发公共牌）取平均
    int teacherStyles = 3;          // 2~4：紧/标准/松规则，4 再加翻前胜率机器人

    float rewardScale = 1.0f / float(kStartingStack);  // 筹码 -> 起始筹码的倍数
    int forceRuleSeat = -1;        // 评估用：强制某个座位打规则机器人
    int forceRandomSeat = -1;      // 评估用：强制某个座位均匀随机
    int forceProSeat = -1;         // 评估用：强制某个座位打"胜率机器人"
    int forceSearchSeat = -1;      // 评估用：强制某个座位打"搜索机器人"（教师 argmax）
    int forceSolidSeat = -1;       // 评估用：强制某个座位打"跟注型稳健机器人"
    int searchSamples = 16;        // 搜索机器人对隐藏牌平均的样本数
    int searchStyles = 3;          // 搜索机器人用的教师续演风格数
    // 评估用：强制某个座位使用任意一个外部 actor（两个检查点对打）。
    const nn::Actor* forceActor = nullptr;
    int forceActorSeat = -1;
    // 动作级集成（评估 / 对打用）：ROLE_LATEST 座位上除 latest.actor[seat] 之外，
    // 再把 ensembleExtra 里的冻结模型各前向一遍，把各自 masked softmax 的**概率**
    // 平均后采样。平均的是概率而不是 logits / 参数：参数平均（soup）要求权重在
    // 同一盆地，概率平均没有这个前提，跨 seed、跨课程训练的模型也能直接合。
    // 只在评估路径生效——训练仍走单模型，PPO 的 logp 与梯度口径不变。
    const nn::Actor* ensembleExtra[2][kMaxEnsemble] = {};
    int ensembleExtraCount[2] = {0, 0};
    // 评估用：动作选择温度与贪心。temp<1 让策略更尖（接近 best response），
    // greedy=true 直接取 argmax。训练永远用 temp=1 + 采样（重要性比值要求
    // 行为策略就是待优化的随机策略）。
    float policyTemp = 1.0f;
    bool policyGreedy = false;
    // 评估用：尺寸带内的金额是否固定取对数中点（不再带内随机）。
    // 策略选的是槽，金额默认在带内对数均匀；固定中点能去掉这部分随机性。
    bool amountMid = false;
    // 部署专用：翻前加注额乘以这个系数（>=1），用来剥削"对大尺寸弃牌率高"
    // 的对手（Slumbot 对 >1.8x 池的加注弃 68~74%）。训练默认 1.0。
    float preflopRaiseMult = 1.0f;
    float forceActorPreflopRaiseMult = 1.0f;
    // 部署专用：后翻牌下注/加注额乘以这个系数（>=1）。外部日志显示
    // Slumbot 对 >1.3x 池的下注弃 67~78%，而纯诈唬只需 56.5% 弃牌率。
    float postflopBetMult = 1.0f;
    float forceActorPostflopBetMult = 1.0f;
    // 对打时对手（forceActor）侧的动作选择温度/贪心。默认与训练一致。
    float forceActorTemp = 1.0f;
    bool forceActorGreedy = false;
    // 评估用：把 ROLE_LATEST 座位换成策略内搜索（见 policySearchMove）。
    // 0 = 关；>0 = 每次决策的隐藏牌样本数。搜索用的策略由 policySearchActors 给出。
    int policySearchSamples = 0;
    int policySearchEquityIters = 8;
    const nn::Actor* policySearchActors[2] = {nullptr, nullptr};
    // 训练用：每局随机挑一个座位放这个冻结对手，另一座位一定是当前策略。
    // 两个指针通常来自同一个检查点的 SB/BB 网络；用来训练 best response。
    const nn::Actor* fixedOpponent[2] = {nullptr, nullptr};
    // 冻结对手的动作选择（温度/贪心）。测"尖策略是否更容易被 BR 打穿"时，
    // 让对手用和部署相同的提取方式。
    float fixedOpponentTemp = 1.0f;
    bool fixedOpponentGreedy = false;
    // mixedFrozen=true 时，fixedOpponent 不再独占所有对局（BR 模式），
    // 而是作为联赛里的一类对手，占比 extraOpponentProb。用来"在正常联赛里
    // 加入当前冠军"，而不是退化成纯 BR。
    bool mixedFrozen = false;
    float extraOpponentProb = 0.0f;
    // 真双 oracle：同时把多个 BR 利用者放进联赛，每局等概率抽一个。
    // 单个 BR 依次补洞会放大可利用性（README 第 80 条），必须同时防住所有。
    static constexpr int kMaxExtraOpponents = 8;
    const nn::Actor* extraOpponents[kMaxExtraOpponents][2] = {};
    int extraOpponentCount = 0;
    // 联赛局里"学习座位坐大盲"的概率（默认 0.5）。BB 的决策更难、也是
    // 历史上最容易塌缩的座位，偏置到 BB 可以给它更多样本。
    float learnerBBProb = 0.5f;
    int watchedSeat = -1;          // 评估用：记录哪个座位的净收益；-1 = 自动推断
    // 评估用：记录"学习座位"（ROLE_LATEST）的净收益。混合基准每局的座位随机。
    bool watchLearningSeat = false;
    std::vector<int>* handScores = nullptr;  // 评估用：逐手记录"被测座位"的净收益
    // 评估用：按**局号**记录被测座位的净收益。牌局是乱序结束的，
    // handScores 的 push 顺序不可靠；配对评估必须按局号对齐。
    std::vector<int>* handScoresByGame = nullptr;
    // 评估用：逐手记录终局底池与被测座位，用于按底池分桶诊断（和外部日志同口径）。
    std::vector<int>* handPotsByGame = nullptr;
    std::vector<int>* handSeatsByGame = nullptr;
    // 评估用：固定发牌（同牌对调）。为空时正常随机发牌。
    const std::vector<HUState>* fixedDeals = nullptr;
    // 测试用：按局号记录每局的初始发牌（验证同牌对调等）。为空时不记录。
    std::vector<HUState>* initialDeals = nullptr;
};

struct RolloutStats {
    int games = 0;
    int sbWins = 0;   // 小盲收池（含摊牌）
    int bbWins = 0;   // 大盲收池
    int splits = 0;
    long long sbScore = 0;
    long long bbScore = 0;
    long long decisions = 0;
    long long showdowns = 0;
    long long allinDecisions = 0;
    long long folds = 0;
    // 本批里各类对手实际出现的手数（按局计，不含被强制的评估局）
    int vsLatest = 0;
    int vsRecent = 0;
    int vsArchive = 0;
    int vsRule = 0;
    int vsRandom = 0;
    int vsPro = 0;
    int vsSearch = 0;
    int vsSolid = 0;      // 跟注型稳健机器人
    int vsTight = 0;      // 紧全下机器人
    int vsTrapper = 0;    // 陷阱机器人
    int vsShark = 0;      // 价值型强对手机器人
    int vsSpammer = 0;    // 小注骚扰机器人
    int vsFolder = 0;     // 按尺寸弃牌机器人
    int vsSelfSharp = 0;  // 尖自对局（对手按部署温度提取）
    int vsExtra = 0;      // 联赛里的额外冻结对手（mixedFrozen 模式）
};

void collectRollout(const ModelSet& latest, const HistoricalPool* pool, const RolloutConfig& cfg,
                    std::array<std::vector<Transition>, 2>& out, RolloutStats& stats);

// 从 logits 里按合法动作采样，返回 (slot, log 概率)。
std::pair<int, float> sampleAction(const float* logits, const std::vector<ActionOption>& acts,
                                   Rng& rng);

// 规则机器人：牌力百分位 + 底池赔率的启发式。作为评估锚点与联赛对手。
Move ruleAgentMove(const HUState& s, Rng& rng);

// 胜率机器人：用"对随机手牌的蒙特卡洛胜率"替代百分位，翻前用 169 类表。
// 它比规则机器人强（会正确 semi-bluff / 跟听牌），作为更高一档的评估锚点。
Move proAgentMove(const HUState& s, Rng& rng);

// 搜索机器人：对隐藏牌平均后的教师 Q 取 argmax。它不偷看对手底牌，
// 但每一步都做一次"同牌分支续演"的搜索，是比胜率机器人更强的评估锚点。
Move searchAgentMove(const HUState& s, Rng& rng, int samples = 16, int styles = 3);

// 跟注型"稳健"机器人：宽跟注、少诈唬、只用强牌加注。用来提供"被跟注"的
// 对抗压力（其余脚本对手都会对加注弃牌）。
Move solidAgentMove(const HUState& s, Rng& rng);
// 紧全下机器人：只用顶范围推/接全下（其余过牌/弃牌），与 solid 的
// "宽跟注"互补，教策略不要用弱牌接大额全下。
Move tightAgentMove(const HUState& s, Rng& rng);
// 陷阱机器人：强牌只跟注/过牌（河牌才下注），不弃顶范围。
Move trapperAgentMove(const HUState& s, Rng& rng);
// 价值型强对手：翻前紧凶、翻后只用两对+ 建池，弱牌面对大注直接弃。
Move sharkAgentMove(const HUState& s, Rng& rng);
// 小注骚扰机器人：翻后高频下注 1/3 池（不看牌力），教策略防守小注。
Move spammerAgentMove(const HUState& s, Rng& rng);
Move folderAgentMove(const HUState& s, Rng& rng);

// 策略内搜索（部署用）：对每个合法动作，用同一批隐藏牌抽样（对手底牌 +
// 未发公共牌）各做一次完整续演，续演双方都用这套 actor 自己的策略。
// 取平均回报最大的动作。
//
// 这是"策略改进"算子：续演策略与根策略相同，所以期望上不会比直接按策略
// 行动更差；样本数越多，估计越接近一步前瞻的 Q。代价是每个决策点要做
// samples × 合法动作数 条 rollout（批量前向，一次决策约几十毫秒）。
Move policySearchMove(const nn::Actor* sb, const nn::Actor* bb, const HUState& s, int seat,
                      int samples, Rng& rng, int equityIters = 8);

// 部署用：给任意 HUState 选一个动作（编码 + 前向 + 温度采样 + 带内金额）。
// 外部接口（Slumbot 等）与本地评估共用这一条路径。
Move policyMove(const nn::Actor* sb, const nn::Actor* bb, const HUState& s, Rng& rng,
                float temp = 0.1f);

// 脚本教师：枚举 s 的每个合法动作，用同一副牌（同一对手底牌、同一公共牌）
// 分支续演到终局，返回每个 slot 的动作价值（净筹码 / 起始筹码）。
// 分支之间只差我们的动作，续演策略固定，所以差的方差远小于终局回报本身。
// 三条续演风格（0.2 / 0.5 / 0.8 jitter）平均，避免只针对某一种机器人节奏。
//
// samples > 1 时，对"信息集里看不见的牌"（对手底牌 + 未发公共牌）重复采样
// 并取平均。这一步是必须的：actor 看不到对手底牌，若只对某一手具体对手牌
// 蒸馏，隐藏牌一换最优动作就会翻转，网络学到的会是"一半全下、一半弃牌"的
// 混合，而不是在期望值上占优的那个动作。
void teacherActionValues(const HUState& s, int seat, std::array<float, kNumActions>& q,
                         int samples = 1, int styles = 3);

}  // namespace texas
