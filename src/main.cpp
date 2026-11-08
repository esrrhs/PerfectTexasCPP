// PerfectTexasCPP 命令行入口。
//
//   perfecttexas train  [--updates N] [--games N] [--out DIR] [--resume DIR]
//   perfecttexas eval   --resume DIR [--vs rule|random|self] [--hands N]
//   perfecttexas bench  [--hands N]
//
// train  自对局 PPO 训练（不完全信息 actor + 开天眼 critic）
// eval   拿存盘模型对固定基准打对局，报每手净收益与标准误
// bench  未训练策略分别打规则机器人和随机策略，作为强度下限

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <torch/torch.h>

#include "texas/eval.hpp"
#include "texas/features.hpp"
#include "texas/hu.hpp"
#include "texas/learner.hpp"
#include "texas/nn.hpp"
#include "texas/ppo.hpp"
#include "texas/strength.hpp"
#include "texas/rollout.hpp"
#include "texas/slumbot.hpp"
#include "texas/blueprint.hpp"
#include "texas/blueprint_br.hpp"

namespace {

const char* argValue(int argc, char** argv, const char* key, const char* def) {
    for (int i = 0; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return def;
}

bool hasFlag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) return true;
    }
    return false;
}

int argInt(int argc, char** argv, const char* key, int def) {
    const char* v = argValue(argc, argv, key, nullptr);
    return v ? std::atoi(v) : def;
}

double argDouble(int argc, char** argv, const char* key, double def) {
    const char* v = argValue(argc, argv, key, nullptr);
    return v ? std::atof(v) : def;
}

std::string argStr(int argc, char** argv, const char* key, const std::string& def) {
    const char* v = argValue(argc, argv, key, nullptr);
    return v ? std::string(v) : def;
}

// blueprint 长训的迭代数会超过 int 上限，单独用 64 位解析。
uint64_t argU64(int argc, char** argv, const char* key, uint64_t def) {
    const char* v = argValue(argc, argv, key, nullptr);
    return v ? std::strtoull(v, nullptr, 10) : def;
}

// 收集同一个 key 的所有取值（可重复传参，例如多个 --ensemble DIR）。
std::vector<std::string> argValues(int argc, char** argv, const char* key) {
    std::vector<std::string> out;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) out.emplace_back(argv[i + 1]);
    }
    return out;
}

void usage() {
    std::printf(
        "PerfectTexasCPP —— 单挑无限注德州的自对局强化学习（PPO + 完美信息蒸馏）\n"
        "\n"
        "用法:\n"
        "  perfecttexas train [--updates N] [--games N] [--lr F] [--hidden N] [--lstm N]\n"
        "                     [--out DIR] [--resume DIR] [--freeze-opponent DIR] [--threads N]\n"
        "                     [--device auto|cpu|cuda|mps] [--mps]（采样恒在 CPU；设备只跑 PPO 更新）\n"
        "                     [--freeze-temp F] [--freeze-greedy]\n"
        "                     [--shared-actor]\n"
        "                     [--workers N] [--league F] [--rule-prob F] [--random-prob F]\n"
        "                     [--pro-prob F] [--search-prob F] [--search-samples N]\n"
        "                     [--solid-prob F] [--tight-prob F] [--trapper-prob F]\n"
        "                     [--shark-prob F] [--spammer-prob F] [--maniac-prob F] [--ranger-prob F]\n"
        "                     [--ranger-air F]\n"
        "                     [--archive-ratio F] [--recent N] [--archive N]\n"
        "                     [--max-raises N] [--oracle-iters N]\n"
        "                     [--eval-every N] [--eval-games N]\n"
        "                     [--oracle-coef F] [--no-oracle] [--seed N]\n"
        "                     [--entropy F] [--entropy-final F]\n"
        "                     [--teacher-coef F] [--teacher-temp F] [--teacher-prob F]\n"
        "                     [--teacher-samples N] [--teacher-styles N] [--teacher-decay N]\n"
        "                     [--teacher-replay N] [--teacher-replay-steps N]\n"
        "                     [--teacher-replay-coef F]\n"
        "                     [--critic-replay N] [--critic-replay-epochs N]\n"
        "                     [--critic-weight-decay F] [--critic-extra-epochs N]\n"
        "                     [--critic-info-set]\n"
        "                     [--range-eq N] [--range-eq-iters N]\n"
        "                     [--search-opp-prob F] [--search-opp-samples N]\n"
        "                     [--expert-prob F] [--expert-samples N]\n"
        "  perfecttexas eval  --resume DIR [--ensemble DIR]... [--vs rule|random|pro|search|solid|mix|self]\n"
        "                     [--hands N] [--seed N] [--policy-search N] [--range-search N]\n"
        "                     [--temp F] [--greedy]\n"
        "                     [--amount-mid] [--hidden N] [--lstm N]\n"
        "  perfecttexas match  --a DIR [--ensemble DIR]... --b DIR [--hands N] [--seed N]\n"
        "                     [--search-a N] [--range-search N]\n"
        "                     [--a-temp F] [--a-greedy] [--b-temp F] [--b-greedy]\n"
        "                     [--a-amount-mid] [--a-sb DIR] [--a-bb DIR] [--hidden N] [--lstm N]\n"
        "                     [--b-hidden N] [--b-lstm N]（B 侧网络尺寸，默认同 A）\n"
        "  perfecttexas soup   --a DIR --b DIR --out DIR [--alpha F]\n"
        "  perfecttexas botmatch [--a pro|rule|search|random] [--b ...] [--hands N] [--seed N]\n"
        "  perfecttexas probe  [--resume DIR] [--teacher-q]\n"
        "  perfecttexas distill [--steps N] [--teacher-samples N]\n"
        "  perfecttexas slumbot [--resume DIR] [--hands N] [--temp F] [--seed N] [--delay-ms N]\n"
        "                       [--policy actor|fold|random] [--log FILE.csv]\n"
        "                       [--allin-guard-pre F] [--allin-guard-post F]\n"
        "                       [--range-eq N] [--range-eq-iters N]\n"
        "                       [--river-cfr N] [--river-cfr-min-pot N] [--river-cfr-raises N]\n"
        "  perfecttexas slumbot-analyze --csv FILE.csv  逐决策算事后胜率，定位漏点\n"
        "  perfecttexas allin-audit --csv FILE.csv...  全下决策审计（按座位/街/推接）\n"
        "  perfecttexas slumbot-merge --csv FILE.csv... 合并多段外部日志的调整后 EV ± SE\n"
        "  perfecttexas commit-audit --csv FILE.csv... 弃牌前投入审计（按投入深度/弃牌街）\n"
        "  perfecttexas slumbot-decisions --csv FILE.csv... 导出 Slumbot 逐决策样本（含手牌百分位）\n"
        "  perfecttexas showdown --resume DIR --vs solid|rule|pro|search|random|ranger|cfr\n"
        "                     [--hands N] [--seed N] [--temp F] [--range-eq N]\n"
        "                     [--ranger-air F] [--opp-cfr-iters N] [--opp-cfr-min-pot N]\n"
        "                     摊牌诊断：按摊牌/非摊牌拆收益 + 双方摊牌牌型分布\n"
        "  perfecttexas slumbot-fit --csv FILE.csv [--csv ...] --out DIR [--resume DIR]\n"
        "                     [--epochs N] [--lr F] [--holdout-every N]\n"
        "                     用日志里 Slumbot 的动作做行为克隆，得到它的对手模型\n"
        "  perfecttexas policy-stats --resume DIR [--hands N] [--temp F]\n"
        "                     自对局动作分布（翻前开牌/BB 应对/翻牌下注频率）\n"
        "  perfecttexas bench-device [--iters N] [--batch N]...  CPU/MPS/CUDA 前后向基准\n"
        "  perfecttexas bench  [--hands N]\n"
        "  perfecttexas blueprint-train [--iters N] [--out DIR] [--resume DIR] [--seed N]\n"
        "                     [--buckets N] [--raise-cap N] [--save-every N] [--log-every N]\n"
        "                     离线表格 MCCFR 训练 CFR blueprint（检查点含 regret/平均策略/RNG）\n"
        "  perfecttexas blueprint-br --bp DIR [--samples N] [--seed N]\n"
        "                     抽样最优应对，量 blueprint 可利用度（G1）\n"
        "  perfecttexas blueprint-distill --bp DIR --resume DIR --out DIR [--steps N]\n"
        "                     [--eval-every N] [--patience N] [--eval-only]\n"
        "                     把 blueprint 平均策略监督蒸馏进 actor（G2，留出集早停）\n"
        "  perfecttexas blueprint-eval --bp DIR [--hands N] [--seed N]\n"
        "                     blueprint 查表策略对内置基准的强度诊断\n"
        "\n"
        "说明:\n"
        "  --device auto|cpu|cuda|mps  训练设备（默认 auto：CUDA > MPS > CPU）。注意\n"
        "                  rollout/评估始终在 CPU，只有 PPO 更新跑在设备上；小 batch 在\n"
        "                  MPS 上反而更慢，默认 minibatch 因此按设备取 512（CPU 仍是 128）\n"
        "  --freeze-opponent DIR 只用该检查点当对手，训练 best response（关掉联赛）\n"
        "  --freeze-temp F / --freeze-greedy  冻结对手侧的动作提取（默认同训练）\n"
        "  --league F      一局里对手是历史快照的概率（默认 0.30）\n"
        "  --rule-prob F   一局里对手是规则机器人的概率（默认 0.15）\n"
        "  --random-prob F 一局里对手是均匀随机的概率（默认 0.05）\n"
        "  --pro-prob F    一局里对手是胜率机器人的概率（默认 0，风格不同于规则机器人）\n"
        "  --search-prob F 一局里对手是搜索机器人（教师 argmax）的概率（默认 0）\n"
        "  --search-samples N 搜索机器人对隐藏牌平均的样本数（默认 8）\n"
        "  --solid-prob F  一局里对手是「跟注型稳健机器人」的概率（默认 0）\n"
        "                  （宽跟注、少诈唬、只用强牌加注；补上被跟注的对抗压力）\n"
        "  --spammer-prob F 一局里对手是「小注骚扰机器人」的概率（默认 0）\n"
        "                  （翻后高频 1/3 池下注，逼策略防守小注）\n"
        "  --maniac-prob F 一局里对手是「过度下注机器人」的概率（默认 0）\n"
        "                  （翻后 1.5~3 倍池超大尺寸开火含诈唬、宽跟，惩罚\n"
        "                   把大加注当坚果 + 用弱牌超池诈唬）\n"
        "  --trapper-prob F 一局里对手是「陷阱机器人」的概率（默认 0）\n"
        "                  （强牌只跟注/过牌，河牌才发力；惩罚用中等牌建大池）\n"
        "  --tight-prob F  一局里对手是「紧全下机器人」的概率（默认 0）\n"
        "                  （只用顶范围推/接全下，教策略面对全下收紧）\n"
        "  --shark-prob F  一局里对手是「价值型强对手」的概率（默认 0）\n"
        "                  （翻前紧凶、翻后只用两对+ 建池；惩罚中等牌建大池）\n"
        "  --ranger-prob F 一局里对手是「读范围机器人」的概率（默认 0）\n"
        "                  （跟注/弃牌按对手范围 = 当前牌面最强 keep%% 组件的条件\n"
        "                   胜率；惩罚薄价值下注与宽跟注，补上其余脚本的范围盲）\n"
        "  --ranger-air F  读范围机器人假设我们范围里的空气（诈唬）占比（默认 0.28；\n"
        "                  0 = 纯强牌范围，会让它过度弃牌、教出「多诈唬」）\n"
        "  --folder-prob F 一局里对手是「按尺寸弃牌机器人」的概率（默认 0）\n"
        "  --opp-cfr-prob F 一局里对手是「河牌 CFR 均衡回应」的概率（默认 0；其它街按\n"
        "                   当前策略采样，河牌按范围均衡解，成本随 --opp-cfr-iters 线性）\n"
        "  --opp-cfr-iters N 对手河牌求解的迭代次数（默认 120；0 = 关闭）\n"
        "  --opp-cfr-min-pot N 只有底池 >= 该值才求解（默认 1500）\n"
        "  --big-pot-opponent DIR 底池 >= --big-pot-threshold（默认 4000）时改用该模型\n"
        "                     当对手（只换对手座位，用于压制大池赤字）\n"
        "  --self-sharp-prob F 一局里对手是**当前策略按部署温度**（默认 T=0.1）的概率\n"
        "  --self-sharp-temp F 尖自对局的提取温度（默认 0.1）\n"
        "  --extra-opponent DIR 把该检查点作为一个固定对手加进联赛（不是纯 BR）\n"
        "  --extra-opponent-prob F 它在联赛里的占比（默认 0.10）\n"
        "  --extra-opponent-temp F 它的动作提取温度（默认 0.1，和部署一致）\n"
        "  --learner-bb-prob F 联赛局里学习座位坐 BB 的概率（默认 0.5；>0.5 给 BB 更多样本）\n"
        "  --duplicate-deals 相邻两局同牌打对调座位（降低牌运方差；偶数局成对）\n"
        "  --archive-ratio F 抽历史对手时用归档快照的比例（默认 0.30）\n"
        "  --recent N      历史池的滚动窗口容量（默认 8）\n"
        "  --archive N     历史池的归档容量，满了隔一个丢一个（默认 16）\n"
        "  --max-raises N  每街加注次数上限（默认 0 = 无上限，对齐 ACPC）\n"
        "  --oracle-coef F 开天眼稠密奖励的系数（默认 0.05，0 等于关掉）\n"
        "  --aux-equity-coef F 范围胜率信念头的回归系数（默认 0 = 关闭）\n"
        "  --opp-model-coef F 对手牌力建模头的回归系数（默认 0 = 关闭）\n"
        "  --range-eq N    范围胜率特征：每个决策采样 N 个对手组合，按对手最近动作\n"
        "                  的策略似然重加权后算 vs-range 摊牌胜率（默认 0 = 关闭）\n"
        "  --range-eq-iters N rangeEq 摊牌胜率的蒙特卡洛次数（默认 48；转/河精确）\n"
        "  --range-model M  rangeEq 的对手似然：policy（默认）| empirical（Slumbot 经验表）\n"
        "  --turn-cfr N    转牌决策用 CFR+ 子博弈求解（叶子=发河牌后 check-down 摊牌；\n"
        "                 0=关闭，默认走网络策略；--turn-cfr-min-pot N 只在底池>=N 时求解）\n"
        "  --teacher-min-pot F 只对底池 >= F 的决策做教师标注（默认 0 = 全部）\n"
        "  --anchor-model DIR --anchor-coef F 非大池状态对冻结基座加 KL 信任域\n"
        "  --river-rule-coef F --river-rule-min-pot N 河牌大池 -> 1.5x 下注的规则蒸馏\n"
        "  --preflop-raise-mult F / --a-... / --b-... 翻前加注额放大系数（部署剥削）\n"
        "  --postflop-bet-mult F / --a-... / --b-... 后翻牌下注额放大系数（部署剥削）\n"
        "  --no-oracle     关闭开天眼稠密奖励（只保留 critic 的完美信息价值）\n"
        "  --teacher-coef F 脚本教师蒸馏系数（默认 0 = 关；>0 时启用教师）\n"
        "  --teacher-temp F 教师动作价值的 softmax 温度（默认 0.5；Q 已按状态标准化）\n"
        "  --teacher-prob F 翻后决策点算教师的概率（默认 1.0；翻前总是算）\n"
        "  --teacher-samples N 教师对隐藏牌（对手底牌/未发公共牌）平均的样本数（默认 8）\n"
        "  --teacher-styles N 教师续演风格数 2~4（默认 3；4 = 再加翻前胜率机器人）\n"
        "  --teacher-replay N 教师标注回放容量/座位（默认 0 = 关；推荐 4096）\n"
        "  --teacher-replay-steps N 每轮在回放上多跑的纯蒸馏 minibatch 轮数（默认 2）\n"
        "  --teacher-replay-coef F 回放蒸馏系数相对 --teacher-coef 的倍率（默认 0.5）\n"
        "  --teacher-decay N 在前 N 轮把教师系数线性衰减到 0（默认 0 = 不衰减）\n"
        "  --critic-replay N 价值回放容量/座位（默认 0 = 关；每轮在跨轮池上多跑 critic）\n"
        "  --critic-replay-epochs N 每轮价值回放的 epoch 数（默认 2）\n"
        "  --critic-weight-decay F critic 的 L2 正则（默认 0；离线留出集上 1e-3 明显更准）\n"
        "  --critic-extra-epochs N 策略更新后 critic 额外跑的 epoch 数（默认 0）\n"
        "  --critic-info-set critic 只用 actor 也看得到的信息（去掉对手底牌/完美胜率）\n"
        "  --entropy-final F 熵系数从 --entropy 线性退火到 F（默认等于 --entropy）\n"
        "  --policy-temp F 训练采样温度（默认 1.0；部署温度一致时可用同值训练）\n"
        "  --shared-actor  SB/BB 共用一套 actor（位置由节点特征区分，critic 仍分座位）\n"
        "  --ensemble DIR  动作级集成：可重复，被测座位把主模型与这些检查点的动作\n"
        "                  概率平均后采样（最多 4 个额外模型）\n"
        "  --workers N     采样线程数（默认 1；按局并行；瓶颈在学习器，实测约 1.4×）\n"
        "  --policy-search N / --search-a N  策略内搜索：每步抽 N 组隐藏牌，用当前策略\n"
        "                  把每个合法动作续演到底，取期望回报最大的动作（部署用）\n"
        "  --range-search N  河牌范围条件化搜索：只在河牌上用 N 组对手底牌做一步前瞻，\n"
        "                  样本按对手最近动作的策略似然加权（后验范围），比均匀\n"
        "                  隐藏牌搜索更接近 vs 范围的 EV；其它街仍用原策略\n"
        "  --robust-search  鲁棒范围搜索：同一批对手底牌分别按自身/solid/shark 三种\n"
        "                  模型续演，每个根动作取最坏情况 EV（配合 --range-search N）\n"
        "  --robust-mean    把上面的最坏情况换成三模型等权平均 EV\n"
        "  --range-search-min-pot N  河牌搜索只对底池 >= N 筹码的决策生效（0=全部）\n"
        "  --range-search-defense-only  河牌搜索只用于面对下注（toCall>0）的防守决策\n"
        "  --river-cfr N     河牌 CFR+ 子博弈：迭代 N 轮求范围均衡，用自己这张牌的\n"
        "                  均衡动作替换网络动作（对手范围 = 1081 个组合按对手最近\n"
        "                  动作的策略似然加权）。minimax 口径，和上面的剥削搜索互补\n"
        "  --river-cfr-min-pot N  只在底池 >= N 筹码的河牌决策上求解（0=全部）\n"
        "  --river-cfr-raises N   抽象树里首次下注后的加注次数（默认 1）\n"
        "  --river-cfr-no-allin   抽象树里去掉全下尺寸\n"
        "  --turn-cfr N      转牌 CFR+ 子博弈：同 --river-cfr 口径（对手范围 = 1128\n"
        "                  个组合），叶子 = 发河牌后 check-down 摊牌（河牌全枚举\n"
        "                  精确；≥1 人全下时整个叶子精确，双方有筹码时忽略河牌\n"
        "                  下注街，是已知近似）\n"
        "  --turn-cfr-min-pot N  只在底池 >= N 筹码的转牌决策上求解（0=全部）\n"
        "  --turn-cfr-raises N   抽象树里首次下注后的加注次数（默认 1）\n"
        "  --turn-cfr-no-allin   抽象树里去掉全下尺寸\n"
        "  --expert-prob F / --expert-samples N  专家迭代：自对局的河牌决策按概率 F 用\n"
        "                  范围搜索的 Q 当教师标签，经 --teacher-coef 蒸馏回策略\n"
        "  --temp F / --a-temp F  评估时把动作 logits 除以 F（<1 更尖，默认 1）\n"
        "  --greedy / --a-greedy  评估时直接取 argmax（不采样）\n"
        "  --b-temp F / --b-greedy  对打时对手侧的动作选择（默认与训练一致）\n"
        "  --amount-mid / --a-amount-mid  尺寸带金额固定取对数中点（默认带内随机）\n"
        "  slumbot          用公开 API 打真实成熟 AI；规则 50/100、200BB，按筹码/手报净收益\n"
        "  --allin-guard-pre F   面对全下时翻前只用胜率 >= F 的范围跟（0=关，建议 0.75）\n"
        "  --allin-guard-post-cat N 翻后跟全下要求的最低牌力（0高牌..3三条；-1=关，建议 3）\n"
        "  --allin-guard-min-risk F 跟注额 >= 剩余筹码×F 才触发（默认 0.5）\n"
        "  --preflop-aggro-pct F 手牌胜率低于 F 时，面对 >=N 次加注不再加注（0=关）\n"
        "  --bb-3bet-min-pct F  BB 面对开牌时，手牌百分位低于 F 不 3-bet（改成跟注，0=关）\n"
        "  --preflop-aggro-max-raises N 允许的加注次数上限（默认 2，即允许 3-bet）\n"
        "  --flop-aggro-pct F 翻牌牌力百分位低于 F 时不加注（0=关，建议 0.65）\n"
        "  --policy fold|random  slumbot 的诊断策略（fold 的期望是 -75 筹码/手）\n");
}

void saveModels(const texas::nn::Actor* actors, const texas::nn::Critic* critics,
                const std::string& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    for (int s = 0; s < 2; ++s) {
        actors[s]->saveTo(dir + "/actor" + std::to_string(s) + ".pt");
        critics[s]->saveTo(dir + "/critic" + std::to_string(s) + ".pt");
    }
}

bool loadModels(texas::nn::Actor* actors, texas::nn::Critic* critics, const std::string& dir) {
    if (!std::filesystem::exists(dir + "/actor0.pt")) return false;
    bool ok = true;
    for (int s = 0; s < 2; ++s) {
        ok = actors[s]->loadFrom(dir + "/actor" + std::to_string(s) + ".pt") && ok;
        ok = critics[s]->loadFrom(dir + "/critic" + std::to_string(s) + ".pt") && ok;
    }
    return ok;
}

// 只加载 actor（动作级集成的额外成员不需要 critic）。
bool loadActors(texas::nn::Actor* actors, const std::string& dir) {
    if (!std::filesystem::exists(dir + "/actor0.pt")) return false;
    bool ok = true;
    for (int s = 0; s < 2; ++s) {
        ok = actors[s]->loadFrom(dir + "/actor" + std::to_string(s) + ".pt") && ok;
    }
    return ok;
}

// 把 --ensemble 目录里的检查点加载成一个动作级集成团队。指针稳定性：先
// reserve 再 push_back，之后才取元素地址。
bool buildTeam(texas::EvalTeam& team, const texas::nn::Actor actors[2],
               const std::vector<std::string>& ensDirs,
               std::vector<std::array<texas::nn::Actor, 2>>& storage,
               const texas::nn::ModelConfig& mc) {
    team = texas::EvalTeam::of(actors);
    if (static_cast<int>(ensDirs.size()) > texas::kMaxEnsemble) {
        std::printf("--ensemble 最多 %d 个（当前 %d）\n", texas::kMaxEnsemble,
                    static_cast<int>(ensDirs.size()));
        return false;
    }
    storage.clear();
    storage.reserve(ensDirs.size());
    for (const std::string& dir : ensDirs) {
        std::array<texas::nn::Actor, 2> pair;
        for (int s = 0; s < 2; ++s) {
            pair[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        }
        if (!loadActors(pair.data(), dir)) {
            std::printf("cannot load actors from %s\n", dir.c_str());
            return false;
        }
        storage.push_back(std::move(pair));
    }
    for (size_t i = 0; i < storage.size(); ++i) {
        for (int s = 0; s < 2; ++s) team.extra[s][i] = &storage[i][s];
    }
    for (int s = 0; s < 2; ++s) team.extraCount[s] = static_cast<int>(storage.size());
    return true;
}

torch::Device pickDevice(int argc, char** argv) {
    const std::string spec = argStr(argc, argv, "--device", "");
    if (spec == "cpu") return torch::kCPU;
    if (spec == "cuda") {
        if (!torch::cuda::is_available()) {
            std::printf("!! --device cuda 不可用（未编译 CUDA 或没有 GPU），回退 cpu\n");
            return torch::kCPU;
        }
        return torch::kCUDA;
    }
    if (spec == "mps" || hasFlag(argc, argv, "--mps")) {
        if (!torch::mps::is_available()) {
            std::printf("!! --device mps 不可用（LibTorch 无 MPS 后端），回退 cpu\n");
            return torch::kCPU;
        }
        return torch::kMPS;
    }
    // auto：有 CUDA 用 CUDA，否则 CPU。**不自动选 MPS**：bench-device 实测
    // 256×128 网络在 M1 Pro 上 MPS 端到端比 CPU 慢 2~5×（小 kernel 启动 +
    // 每步同步），只有显式 --device mps 才用。
    if (torch::cuda::is_available()) return torch::kCUDA;
    return torch::kCPU;
}

const char* deviceName(const torch::Device& d) {
    switch (d.type()) {
        case torch::kCUDA: return "cuda";
        case torch::kMPS: return "mps";
        default: return "cpu";
    }
}

// 训练/推理设备基准：同一套 Actor 在 CPU 与 MPS/CUDA 上跑不同 batch 的
// 前向与"前向+反向+优化器步"，用来决定 --device 的默认值和最优 minibatch。
// 小 batch 会被 kernel 启动/同步开销吃掉，必须实测，不能假设 GPU 一定更快。
int cmdBenchDevice(int argc, char** argv) {
    const int iters = argInt(argc, argv, "--iters", 8);
    texas::nn::ModelConfig mc;
    mc.hidden = argInt(argc, argv, "--hidden", 256);
    mc.lstmHidden = argInt(argc, argv, "--lstm", 128);
    std::printf("torch %s | cpu_threads=%d | cuda=%d | mps=%d\n", TORCH_VERSION,
                torch::get_num_threads(), int(torch::cuda::is_available()),
                int(torch::mps::is_available()));
    std::vector<int64_t> batches;
    for (const std::string& tok : argValues(argc, argv, "--batch")) {
        batches.push_back(std::atoll(tok.c_str()));
    }
    if (batches.empty()) batches = {32, 64, 128, 256, 512, 1024, 2048};

    std::vector<torch::Device> devs{torch::kCPU};
    if (torch::mps::is_available()) devs.push_back(torch::kMPS);
    if (torch::cuda::is_available()) devs.push_back(torch::kCUDA);

    auto makeTensors = [&](int64_t B, const torch::Device& dev) {
        const int64_t L = B * 5;
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
        torch::Tensor node = torch::rand({B, texas::kNodeInput}, opts);
        torch::Tensor seq = torch::rand({B, texas::kHistSteps, texas::kHistDim}, opts);
        torch::Tensor feat = torch::rand({L, texas::kActionInput}, opts);
        torch::Tensor sample = torch::randint(0, B, {L},
                            torch::TensorOptions().dtype(torch::kLong).device(dev));
        torch::Tensor id = torch::randint(0, texas::kNumActions, {L},
                          torch::TensorOptions().dtype(torch::kLong).device(dev));
        torch::Tensor rq = torch::rand({B}, opts);
        return std::make_tuple(node, seq, feat, sample, id, rq);
    };

    std::printf("\n%-5s %6s %14s %14s\n", "dev", "batch", "forward ms", "train ms");
    for (const torch::Device& dev : devs) {
        for (int64_t B : batches) {
            texas::nn::Actor actor(std::make_shared<texas::nn::ActorImpl>(mc));
            actor->to(dev);
            actor->train();
            auto opt = std::make_unique<torch::optim::Adam>(
                actor->parameters(), torch::optim::AdamOptions(1e-4));
            auto tup = makeTensors(B, dev);
            auto& node = std::get<0>(tup);
            auto& seq = std::get<1>(tup);
            auto& feat = std::get<2>(tup);
            auto& sample = std::get<3>(tup);
            auto& id = std::get<4>(tup);
            auto& rq = std::get<5>(tup);

            auto sync = [&]() {
                if (dev.type() == torch::kMPS) torch::mps::synchronize();
                if (dev.type() == torch::kCUDA) torch::cuda::synchronize();
            };
            // 预热：第一次前向会触发内核编译/内存分配，不计入。
            for (int k = 0; k < 2; ++k) {
                actor->forward(node, seq, feat, sample, id, rq);
                sync();
            }
            // 吞吐口径：一段内不逐次同步（GPU 是异步队列），只在段末同步一次。
            // 真实训练要的是吞吐，不是单次延迟。
            sync();
            auto t1 = std::chrono::steady_clock::now();
            for (int k = 0; k < iters; ++k) {
                actor->forward(node, seq, feat, sample, id, rq);
            }
            sync();
            const double fwdMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1)
                    .count() / double(iters);

            sync();
            auto t2 = std::chrono::steady_clock::now();
            for (int k = 0; k < iters; ++k) {
                auto logits = actor->forward(node, seq, feat, sample, id, rq);
                auto loss = logits.sum();
                opt->zero_grad();
                loss.backward();
                opt->step();
            }
            sync();
            const double trMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2)
                    .count() / double(iters);
            std::printf("%-5s %6lld %14.3f %14.3f\n", deviceName(dev),
                        static_cast<long long>(B), fwdMs, trMs);
        }
    }
    return 0;
}


// ---------------------------------------------------------------- train

int cmdTrain(int argc, char** argv) {
    const int updates = argInt(argc, argv, "--updates", 200);
    const int games = argInt(argc, argv, "--games", 512);
    const std::string outDir = argStr(argc, argv, "--out", "ckpt");
    const std::string resumeDir = argStr(argc, argv, "--resume", "");
    const std::string freezeDir = argStr(argc, argv, "--freeze-opponent", "");
    const int seed = argInt(argc, argv, "--seed", 20240518);
    const int evalEvery = argInt(argc, argv, "--eval-every", 10);
    const int evalGames = argInt(argc, argv, "--eval-games", 512);
    const int recentCap = argInt(argc, argv, "--recent", 8);    // 历史池：滚动窗口
    const int archiveCap = argInt(argc, argv, "--archive", 16); // 历史池：几何间隔存档
    const int hwThreads = argInt(argc, argv, "--threads", 0);
    // 采样线程数：一局一条独立牌局，按局并行；每线程用不同 seed 采一部分局，
    // 再合并成同一批 transition。torch 的 CPU 前向是只读的，多个线程可以共享
    // 同一套模型参数（历史池在采样期间也不变）。线程数不能超过局数，
    // 否则有的 worker 分到 0 局、gameId 偏移和实际局数会对不上。
    int workers = std::max(1, std::min(argInt(argc, argv, "--workers", 1), games));
    // 共享 actor：两个座位共用一套策略网络（位置由节点特征区分）。数据翻倍、
    // 参数减半，代价是 SB/BB 不能再各自特化。critic 仍然分座位。
    const bool sharedActor = hasFlag(argc, argv, "--shared-actor");

    const torch::Device device = pickDevice(argc, argv);
    if (hwThreads > 0) torch::set_num_threads(hwThreads);
    // 采样始终在 CPU 上跑（行动模型留在 CPU），所以 MPS/CUDA 不再限制 --workers。

    // 预热牌力模型：桶分配是懒初始化的 mutable 缓存，第一次调用会写表。
    // 多线程采样时必须先在主线程把它初始化好，否则首轮会竞争。
    {
        const texas::Card hole[2] = {texas::parseCard("Ah"), texas::parseCard("Kd")};
        const texas::Card board[3] = {texas::parseCard("2c"), texas::parseCard("7s"),
                                      texas::parseCard("Jh")};
        (void)texas::handStrength(hole, board, 3);
    }

    texas::nn::ModelConfig mc;
    mc.hidden = argInt(argc, argv, "--hidden", 256);
    mc.lstmHidden = argInt(argc, argv, "--lstm", 128);

    texas::ModelSet models;
    for (int s = 0; s < 2; ++s) {
        models.actor[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        models.critic[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
    }

    if (!resumeDir.empty()) {
        if (loadModels(models.actor, models.critic, resumeDir)) {
            std::printf("resumed from %s\n", resumeDir.c_str());
        } else {
            std::printf("!! resume failed: %s (starting fresh)\n", resumeDir.c_str());
        }
    }

    if (sharedActor) {
        // 从两个座位的参数平均出发，然后把 actor1 指到同一个模块。之后再训练
        // 就是一个网络同时吃两个座位的数据；评估时存盘会写两份相同的 actor。
        torch::NoGradGuard ng;
        auto p0 = models.actor[0]->parameters();
        auto p1 = models.actor[1]->parameters();
        if (!p0.empty() && p0.size() == p1.size()) {
            for (size_t i = 0; i < p0.size(); ++i) {
                p0[i].data().mul_(0.5).add_(p1[i].data(), 0.5);
            }
        }
        models.actor[1] = models.actor[0];
        std::printf("shared actor: SB/BB 共用一套参数（初始化 = 两座位平均）\n");
    }

    // 冻结对手：训练 best response 时，每局随机把这个检查点放到一个座位。
    texas::nn::Actor frozenActor[2];
    texas::nn::Critic frozenCritic[2];
    bool hasFrozen = false;
    if (!freezeDir.empty()) {
        for (int s = 0; s < 2; ++s) {
            frozenActor[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
            frozenCritic[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
        }
        if (!loadModels(frozenActor, frozenCritic, freezeDir)) {
            std::printf("cannot load frozen opponent from %s\n", freezeDir.c_str());
            return 1;
        }
        for (int s = 0; s < 2; ++s) frozenActor[s]->eval();
        hasFrozen = true;
        std::printf("freeze opponent: %s (联赛关闭，只打这个对手)\n", freezeDir.c_str());
    }

    // 联赛里额外加入固定对手（通常是外部冠军或 BR 利用者），总占比由
    // --extra-opponent-prob 控制、各对手等分；--extra-opponent 可重复。
    // 和 --freeze-opponent 的纯 BR 模式互斥。
    using ExtraArr = std::array<texas::nn::Actor, texas::RolloutConfig::kMaxExtraOpponents>;
    std::array<ExtraArr, 2> extraActor;
    const std::vector<std::string> extraDirs = argValues(argc, argv, "--extra-opponent");
    int extraCount = 0;
    bool hasExtra = false;
    if (!extraDirs.empty()) {
        if (static_cast<int>(extraDirs.size()) > texas::RolloutConfig::kMaxExtraOpponents) {
            std::printf("!! --extra-opponent 最多 %d 个\n",
                        texas::RolloutConfig::kMaxExtraOpponents);
            return 1;
        }
        for (const std::string& dir : extraDirs) {
            texas::nn::Actor pair[2];
            texas::nn::Critic extraCritic[2];
            for (int s = 0; s < 2; ++s) {
                pair[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
                extraCritic[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
            }
            if (!loadModels(pair, extraCritic, dir)) {
                std::printf("cannot load extra opponent from %s\n", dir.c_str());
                return 1;
            }
            for (int s = 0; s < 2; ++s) {
                pair[s]->eval();
                extraActor[static_cast<size_t>(s)][static_cast<size_t>(extraCount)] = pair[s];
            }
            ++extraCount;
            std::printf("extra opponent #%d: %s\n", extraCount, dir.c_str());
        }
        hasExtra = true;
        std::printf("extra opponents: %d 个，联赛总占比 %.2f（各等分）\n", extraCount,
                    argDouble(argc, argv, "--extra-opponent-prob", 0.10));
    }

    texas::ppo::Config pcfg;
    pcfg.lr = static_cast<float>(argDouble(argc, argv, "--lr", 1e-3));
    pcfg.clip = static_cast<float>(argDouble(argc, argv, "--clip", 0.2));
    pcfg.entCoef = static_cast<float>(argDouble(argc, argv, "--entropy", 0.01));
    // 熵系数从 --entropy 线性退火到 --entropy-final（默认等于 --entropy）。
    // 训练末期把分布收紧，部署时 T=1 就接近"尖策略"，不必靠温度后处理。
    const float entropy0 = pcfg.entCoef;
    const float entropyFinal =
        static_cast<float>(argDouble(argc, argv, "--entropy-final", pcfg.entCoef));
    pcfg.epochs = argInt(argc, argv, "--epochs", 4);
    // minibatch 默认值：CPU 128（历史配方）；设备上 512（bench-device 实测
    // MPS 在 batch 512 的每样本训练吞吐约为 128 的 2 倍，同时保留足够多的
    // 优化步数；更大的 1024/2048 吞吐更高，但会改变 PPO 的更新节奏）。
    pcfg.minibatch = argInt(argc, argv, "--minibatch",
                            device.type() == torch::kCPU ? 128 : 1024);
    pcfg.criticWeightDecay = static_cast<float>(argDouble(argc, argv, "--critic-weight-decay", 0.0));
    // critic 在每轮策略更新之后额外跑的 epoch 数。默认 0：600 轮同种子 A/B
    // 显示额外 epoch 越多、跨批泛化越差（README 第八节第 12 条）。
    pcfg.criticExtraEpochs = argInt(argc, argv, "--critic-extra-epochs", 0);
    pcfg.teacherCoef = static_cast<float>(argDouble(argc, argv, "--teacher-coef", 0.0));
    pcfg.teacherTemp = static_cast<float>(argDouble(argc, argv, "--teacher-temp", 0.5));
    pcfg.auxEquityCoef = static_cast<float>(argDouble(argc, argv, "--aux-equity-coef", 0.0));
    pcfg.oppModelCoef = static_cast<float>(argDouble(argc, argv, "--opp-model-coef", 0.0));
    pcfg.anchorCoef = static_cast<float>(argDouble(argc, argv, "--anchor-coef", 0.0));
    pcfg.anchorMaxPot = static_cast<float>(argDouble(argc, argv, "--anchor-max-pot", 1500.0));
    pcfg.ruleCoef = static_cast<float>(argDouble(argc, argv, "--river-rule-coef", 0.0));
    const int teacherDecay = argInt(argc, argv, "--teacher-decay", 0);
    const float teacherCoef0 = pcfg.teacherCoef;
    const int teacherReplay = argInt(argc, argv, "--teacher-replay", 0);
    const int teacherReplaySteps = argInt(argc, argv, "--teacher-replay-steps", 2);
    const float teacherReplayCoef =
        static_cast<float>(argDouble(argc, argv, "--teacher-replay-coef", 0.5));
    // 价值回放：cross-round critic 经验池。0 = 关。
    const int criticReplay = argInt(argc, argv, "--critic-replay", 0);
    const int criticReplayEpochs = argInt(argc, argv, "--critic-replay-epochs", 2);

    // 状态条件信任域的冻结基座（只读、eval、不与 learner 共享参数）。
    texas::nn::Actor anchorActor[2];
    const std::string anchorDir = argStr(argc, argv, "--anchor-model", "");
    if (!anchorDir.empty()) {
        for (int s = 0; s < 2; ++s) {
            anchorActor[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        }
        if (!loadActors(anchorActor, anchorDir)) {
            std::printf("cannot load anchor model from %s\n", anchorDir.c_str());
            return 1;
        }
        for (int s = 0; s < 2; ++s) {
            anchorActor[s]->to(device);
            anchorActor[s]->eval();
        }
        std::printf("anchor: %s | coef=%.2f | 只锚 pot<%.0f\n", anchorDir.c_str(),
                    pcfg.anchorCoef, pcfg.anchorMaxPot);
    }

    // 设备与行动模型分离：rollout / 评估始终用 CPU 上的 models（避免每个决策
    // 都做一次 H2D 拷贝，也避免 MPS 在多线程采样下不安全）；Learner 用一份
    // 设备副本训练，每轮更新后把权重同步回 CPU。
    //
    // benchmark（bench-device，Apple M1 Pro 16 核 GPU）：forward 在 batch 128
    // 时 MPS 是 CPU 的 ~3.8×；训练步在 batch 128 只有 ~1.3×，batch 1024 才
    // ~2.1×。所以 GPU 的收益主要来自"更大的 minibatch + 推理"，不是小 batch。
    const bool onDevice = device.type() != torch::kCPU;
    texas::nn::Actor trainActor[2];
    texas::nn::Critic trainCritic[2];
    for (int s = 0; s < 2; ++s) {
        if (onDevice) {
            trainActor[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
            trainCritic[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
        } else {
            trainActor[s] = models.actor[s];
            trainCritic[s] = models.critic[s];
        }
    }
    if (sharedActor && onDevice) trainActor[1] = trainActor[0];

    std::unique_ptr<texas::Learner> learners[2];
    for (int s = 0; s < 2; ++s) {
        learners[s] = std::make_unique<texas::Learner>(trainActor[s], trainCritic[s], pcfg, device);
        if (!anchorDir.empty()) learners[s]->anchor = &anchorActor[s];
        if (!resumeDir.empty()) learners[s]->loadOptimizers(resumeDir);
    }

    // libtorch 的 C++ 前端没有 state_dict；同架构的参数顺序是确定的，按序 copy_。
    // copy_ 支持跨设备（CPU -> MPS/CUDA），所以同步两个方向都用同一个函数。
    auto copyParams = [](torch::nn::Module& dst, const torch::nn::Module& src) {
        auto dp = dst.parameters();
        auto sp = src.parameters();
        if (dp.size() != sp.size()) return false;
        torch::NoGradGuard ng;
        for (size_t i = 0; i < dp.size(); ++i) dp[i].data().copy_(sp[i].data());
        return true;
    };
    if (onDevice) {
        for (int s = 0; s < 2; ++s) {
            copyParams(*trainActor[s], *models.actor[s]);
            copyParams(*trainCritic[s], *models.critic[s]);
        }
        if (device.type() == torch::kMPS) torch::mps::synchronize();
        if (device.type() == torch::kCUDA) torch::cuda::synchronize();
    }

    // 每轮更新后把设备副本写回 CPU 行动模型；MPS/CUDA 先同步再拷，避免读到
    // 还没算完的队列。权重只有 ~10MB，开销远小于一轮 rollout。
    auto syncBackToCpu = [&]() {
        if (!onDevice) return;
        if (device.type() == torch::kMPS) torch::mps::synchronize();
        if (device.type() == torch::kCUDA) torch::cuda::synchronize();
        for (int s = 0; s < 2; ++s) {
            copyParams(*models.actor[s], *trainActor[s]);
            copyParams(*models.critic[s], *trainCritic[s]);
        }
    };

    texas::HistoricalPool pool;
    texas::RolloutConfig rcfg;
    rcfg.games = games;
    rcfg.historicalProb = static_cast<float>(argDouble(argc, argv, "--league", 0.30));
    rcfg.ruleProb = static_cast<float>(argDouble(argc, argv, "--rule-prob", 0.15));
    rcfg.randomProb = static_cast<float>(argDouble(argc, argv, "--random-prob", 0.05));
    rcfg.proProb = static_cast<float>(argDouble(argc, argv, "--pro-prob", 0.0));
    rcfg.searchProb = static_cast<float>(argDouble(argc, argv, "--search-prob", 0.0));
    rcfg.solidProb = static_cast<float>(argDouble(argc, argv, "--solid-prob", 0.0));
    rcfg.tightProb = static_cast<float>(argDouble(argc, argv, "--tight-prob", 0.0));
    rcfg.trapperProb = static_cast<float>(argDouble(argc, argv, "--trapper-prob", 0.0));
    rcfg.sharkProb = static_cast<float>(argDouble(argc, argv, "--shark-prob", 0.0));
    rcfg.spammerProb = static_cast<float>(argDouble(argc, argv, "--spammer-prob", 0.0));
    rcfg.maniacProb = static_cast<float>(argDouble(argc, argv, "--maniac-prob", 0.0));
    rcfg.rangerProb = static_cast<float>(argDouble(argc, argv, "--ranger-prob", 0.0));
    rcfg.rangerAir = static_cast<float>(argDouble(argc, argv, "--ranger-air", 0.28));
    rcfg.oppCfrProb = static_cast<float>(argDouble(argc, argv, "--opp-cfr-prob", 0.0));
    rcfg.oppCfrIters = argInt(argc, argv, "--opp-cfr-iters", 120);
    rcfg.oppCfrMinPot = argInt(argc, argv, "--opp-cfr-min-pot", 1500);
    rcfg.folderProb = static_cast<float>(argDouble(argc, argv, "--folder-prob", 0.0));
    rcfg.searchSamples = argInt(argc, argv, "--search-samples", 8);
    // 大池对手：把底池 >= 阈值的对手决策换成指定的克隆模型。
    texas::nn::Actor bigPotActor[2];
    const std::string bigPotDir = argStr(argc, argv, "--big-pot-opponent", "");
    if (!bigPotDir.empty()) {
        for (int s = 0; s < 2; ++s) {
            bigPotActor[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        }
        if (!loadActors(bigPotActor, bigPotDir)) {
            std::printf("cannot load big-pot opponent from %s\n", bigPotDir.c_str());
            return 1;
        }
        for (int s = 0; s < 2; ++s) bigPotActor[s]->eval();
        rcfg.bigPotOpponent[0] = &bigPotActor[0];
        rcfg.bigPotOpponent[1] = &bigPotActor[1];
        rcfg.bigPotThreshold =
            static_cast<float>(argDouble(argc, argv, "--big-pot-threshold", 4000.0));
        rcfg.bigPotTemp = static_cast<float>(argDouble(argc, argv, "--big-pot-temp", 1.0));
        std::printf("big-pot opponent: %s (pot >= %.0f, T=%.2f)\n", bigPotDir.c_str(),
                    rcfg.bigPotThreshold, rcfg.bigPotTemp);
    }


    rcfg.selfSharpProb = static_cast<float>(argDouble(argc, argv, "--self-sharp-prob", 0.0));
    rcfg.selfSharpTemp = static_cast<float>(argDouble(argc, argv, "--self-sharp-temp", 0.1));
    rcfg.duplicateDeals = hasFlag(argc, argv, "--duplicate-deals");
    rcfg.archiveProb = static_cast<float>(argDouble(argc, argv, "--archive-ratio", 0.30));
    rcfg.oracleReward = !hasFlag(argc, argv, "--no-oracle");
    rcfg.criticOracleExtra = !hasFlag(argc, argv, "--critic-info-set");
    rcfg.oracleCoef = static_cast<float>(argDouble(argc, argv, "--oracle-coef", 0.05));
    rcfg.equityIters = argInt(argc, argv, "--oracle-iters", 32);
    // 范围胜率特征：组合级计算（对手底牌采样 + 动作似然重加权 + 摊牌胜率）。
    rcfg.rangeEqSamples = argInt(argc, argv, "--range-eq", 0);
    rcfg.rangeEqIters = argInt(argc, argv, "--range-eq-iters", 48);
    // 范围搜索剥削者：对手座位按概率换成"用最新策略做河牌后验加权前瞻"的
    // 近似 BR。这是相对当前策略的近似 best response，用来压缩可利用空间。
    rcfg.searchOppProb = static_cast<float>(argDouble(argc, argv, "--search-opp-prob", 0.0));
    rcfg.searchOppSamples = argInt(argc, argv, "--search-opp-samples", 4);
    // 专家迭代：河牌把范围条件化搜索的 Q 当教师标签蒸馏回策略（只标自对局）。
    rcfg.expertProb = static_cast<float>(argDouble(argc, argv, "--expert-prob", 0.0));
    rcfg.expertSamples = argInt(argc, argv, "--expert-samples", 8);
    rcfg.learnerBBProb = static_cast<float>(argDouble(argc, argv, "--learner-bb-prob", 0.5));
    if (hasFrozen) {
        rcfg.historicalProb = 0.0f;
        rcfg.ruleProb = 0.0f;
        rcfg.randomProb = 0.0f;
        rcfg.proProb = 0.0f;
        rcfg.searchProb = 0.0f;
        rcfg.solidProb = 0.0f;
        rcfg.tightProb = 0.0f;
        rcfg.trapperProb = 0.0f;
        rcfg.sharkProb = 0.0f;
        rcfg.spammerProb = 0.0f;
        rcfg.maniacProb = 0.0f;
        rcfg.selfSharpProb = 0.0f;
        rcfg.fixedOpponent[0] = &frozenActor[0];
        rcfg.fixedOpponent[1] = &frozenActor[1];
        rcfg.fixedOpponentTemp = static_cast<float>(argDouble(argc, argv, "--freeze-temp", 1.0));
        rcfg.fixedOpponentGreedy = hasFlag(argc, argv, "--freeze-greedy");
        if (rcfg.fixedOpponentTemp != 1.0f || rcfg.fixedOpponentGreedy) {
            std::printf("  freeze extraction: temp=%.2f greedy=%d\n", rcfg.fixedOpponentTemp,
                        rcfg.fixedOpponentGreedy ? 1 : 0);
        }
    } else if (hasExtra) {
        rcfg.mixedFrozen = true;
        rcfg.extraOpponentProb =
            static_cast<float>(argDouble(argc, argv, "--extra-opponent-prob", 0.10));
        rcfg.extraOpponentCount = extraCount;
        for (int e = 0; e < extraCount; ++e) {
            for (int s = 0; s < 2; ++s) {
                rcfg.extraOpponents[e][s] = &extraActor[static_cast<size_t>(s)][static_cast<size_t>(e)];
            }
        }
        rcfg.fixedOpponentTemp = static_cast<float>(argDouble(argc, argv, "--extra-opponent-temp", 0.1));
        rcfg.fixedOpponentGreedy = false;
    }
    // 训练采样温度：默认 1.0（重要性比值要求行为策略=待优化策略）。
    // 改它等于把"待优化的策略"换成 τ 温度版本——部署用 τ 时可以试同温度训练。
    rcfg.policyTemp = static_cast<float>(argDouble(argc, argv, "--policy-temp", 1.0));
    // 专家迭代用河牌搜索标签，与脚本教师共用同一条蒸馏损失。打开专家时
    // 脚本教师整体关掉（含翻前），这样实验里只有"搜索改进"这一个标签来源，
    // 归因不会和脚本教师混在一起；需要两者叠加时把 --teacher-coef 单独设。
    rcfg.teacher = pcfg.teacherCoef > 0.0f && rcfg.expertProb <= 0.0f;
    rcfg.teacherProb = static_cast<float>(argDouble(argc, argv, "--teacher-prob", 1.0));
    rcfg.teacherSamples = argInt(argc, argv, "--teacher-samples", 8);
    rcfg.teacherStyles = argInt(argc, argv, "--teacher-styles", 3);
    rcfg.ruleMinPot = static_cast<float>(argDouble(argc, argv, "--river-rule-min-pot", 0.0));
    rcfg.ruleTargetRatio = static_cast<float>(argDouble(argc, argv, "--river-rule-ratio", 1.5));
    rcfg.teacherMinPot = static_cast<float>(argDouble(argc, argv, "--teacher-min-pot", 0.0));

    // 加注次数上限：0 = 无上限，和 ACPC 一致。非 0 时退回旧版"每街 N 次"的口径。
    const int maxRaises = argInt(argc, argv, "--max-raises", texas::kNoRaiseCap);
    texas::setDefaultMaxRaises(maxRaises);

    const float selfProb = std::max(0.0f, 1.0f - (rcfg.historicalProb + rcfg.ruleProb +
                                                  rcfg.randomProb + rcfg.proProb +
                                                  rcfg.searchProb + rcfg.solidProb +
                                                  rcfg.tightProb + rcfg.trapperProb +
                                                  rcfg.sharkProb +
                                                  rcfg.spammerProb + rcfg.maniacProb +
                                                  rcfg.rangerProb +
                                                  rcfg.oppCfrProb +
                                                  rcfg.folderProb +
                                                  rcfg.selfSharpProb +
                                                  rcfg.extraOpponentProb));
    const std::string capText =
        (maxRaises == 0) ? std::string("无（ACPC 口径）") : std::to_string(maxRaises) + " 次/街";
    std::printf("PerfectTexasCPP train | 单挑 NLHE %d 筹码 = %dBB | 节点特征 %d + 开天眼 %d + 序列 %d\n",
                texas::kStartingStack, texas::kStartingStack / texas::kBigBlind, texas::kNodeInput,
                texas::kExtraInput, texas::kSeqInput);
    std::printf("  blinds %d/%d | 加注上限 %s | oracle=%.2f\n", texas::kSmallBlind,
                texas::kBigBlind, capText.c_str(),
                rcfg.oracleReward ? rcfg.oracleCoef : 0.0f);
    if (pcfg.auxEquityCoef > 0.0f) {
        std::printf("  aux equity head: coef=%.3f（不完全信息 -> 开天眼胜率）\n",
                    pcfg.auxEquityCoef);
    }
    if (pcfg.oppModelCoef > 0.0f) {
        std::printf("  opp model head: coef=%.3f（不完全信息 -> 对手牌力百分位）\n",
                    pcfg.oppModelCoef);
    }
    if (rcfg.rangeEqSamples > 0) {
        std::printf("  rangeEq: samples=%d iters=%d（组合级后验范围胜率，作为输入特征）\n",
                    rcfg.rangeEqSamples, rcfg.rangeEqIters);
    }
    if (rcfg.searchOppProb > 0.0f) {
        std::printf("  search opponent: prob=%.3f samples=%d（河牌范围条件化搜索，近似 BR）\n",
                    rcfg.searchOppProb, rcfg.searchOppSamples);
    }
    std::printf("  hidden=%lld lstm=%lld | device=%s | games/update=%d | workers=%d\n",
                static_cast<long long>(mc.hidden), static_cast<long long>(mc.lstmHidden),
                deviceName(device), games, workers);
    if (rcfg.teacher) {
        std::printf("  teacher: coef=%.2f temp=%.3f prob=%.2f samples=%d styles=%d decay=%d\n",
                    pcfg.teacherCoef, pcfg.teacherTemp, rcfg.teacherProb, rcfg.teacherSamples,
                    rcfg.teacherStyles, teacherDecay);
        if (rcfg.teacherMinPot > 0.0f) {
            std::printf("  teacher 只标底池 >= %.0f 的决策\n", rcfg.teacherMinPot);
        }
        if (pcfg.ruleCoef > 0.0f) {
            std::printf("  规则教师: 河牌 >= %.0f 池 -> %.2fx 下注，coef=%.2f\n",
                        rcfg.ruleMinPot, rcfg.ruleTargetRatio, pcfg.ruleCoef);
        }
        if (teacherReplay > 0) {
            std::printf("  teacher replay: cap=%d/seat steps=%d coef×%.2f\n", teacherReplay,
                        teacherReplaySteps, teacherReplayCoef);
        }
    }
    if (rcfg.expertProb > 0.0f) {
        std::printf("  expert iteration: prob=%.2f samples=%d coef=%.2f temp=%.2f（只标自对局河牌）\n",
                    rcfg.expertProb, rcfg.expertSamples, pcfg.teacherCoef, pcfg.teacherTemp);
        if (pcfg.teacherCoef <= 0.0f) {
            std::printf("  [警告] --expert-prob>0 但 --teacher-coef=0，蒸馏损失不会生效\n");
        }
    }
    std::array<std::deque<texas::TeacherSample>, 2> replay;
    std::array<std::deque<texas::CriticSample>, 2> criticMemory;
    if (criticReplay > 0) {
        std::printf("  critic replay: cap=%d/seat epochs=%d\n", criticReplay, criticReplayEpochs);
    }
    // 联赛配比：每局只有一个座位是对手，另一个一定是当前策略。
    if (hasFrozen) {
        std::printf("  league: 冻结对手（每局随机座位） | 池 recent=%d archive=%d\n", recentCap,
                    archiveCap);
    } else {
        std::printf(
            "  league: 自对局 %.2f | 尖自对局 %.2f(T=%.2f) | 冻结 %.2f(T=%.2f)"
            " | 历史快照 %.2f (archive %.2f) | 规则 %.2f | 随机 %.2f | 胜率 %.2f"
            " | 搜索 %.2f | 稳健 %.2f | 紧全下 %.2f | 陷阱 %.2f | 强价值 %.2f"
            " | 小注骚扰 %.2f | 过度下注 %.2f | 读范围 %.2f | 按尺寸弃牌 %.2f | 河牌CFR %.2f  [池 recent=%d archive=%d]\n",
            selfProb, rcfg.selfSharpProb, rcfg.selfSharpTemp, rcfg.extraOpponentProb,
            rcfg.fixedOpponentTemp, rcfg.historicalProb, rcfg.archiveProb, rcfg.ruleProb,
            rcfg.randomProb, rcfg.proProb, rcfg.searchProb, rcfg.solidProb, rcfg.tightProb,
            rcfg.trapperProb, rcfg.sharkProb, rcfg.spammerProb, rcfg.maniacProb,
            rcfg.rangerProb, rcfg.folderProb, rcfg.oppCfrProb, recentCap, archiveCap);
    }
    std::printf("  %6s %7s %7s %6s %6s %6s %8s %8s %7s %6s %7s | %s\n", "update", "games", "sbWin%",
                "sd%", "jam%", "fold%", "pgLoss", "vLoss", "entropy", "kl", "upd/s", "batch");

    // 多线程采样的 per-worker 缓冲（复用，避免每轮分配）。
    struct WorkerBatch {
        std::array<std::vector<texas::Transition>, 2> tr;
        texas::RolloutStats stats;
        int games = 0;
        uint64_t seed = 0;
    };
    std::vector<WorkerBatch> workerBatches(static_cast<size_t>(workers));
    auto addStats = [](texas::RolloutStats& a, const texas::RolloutStats& b) {
        a.games += b.games;
        a.sbWins += b.sbWins;
        a.bbWins += b.bbWins;
        a.splits += b.splits;
        a.sbScore += b.sbScore;
        a.bbScore += b.bbScore;
        a.decisions += b.decisions;
        a.showdowns += b.showdowns;
        a.allinDecisions += b.allinDecisions;
        a.folds += b.folds;
        a.vsLatest += b.vsLatest;
        a.vsRecent += b.vsRecent;
        a.vsArchive += b.vsArchive;
        a.vsRule += b.vsRule;
        a.vsRandom += b.vsRandom;
        a.vsPro += b.vsPro;
        a.vsSearch += b.vsSearch;
        a.vsSolid += b.vsSolid;
        a.vsTight += b.vsTight;
        a.vsTrapper += b.vsTrapper;
        a.vsShark += b.vsShark;
        a.vsSpammer += b.vsSpammer;
        a.vsManiac += b.vsManiac;
        a.vsFolder += b.vsFolder;
        a.vsSelfSharp += b.vsSelfSharp;
        a.vsExtra += b.vsExtra;
    };

    const auto t0 = std::chrono::steady_clock::now();
    for (int u = 1; u <= updates; ++u) {
        rcfg.seed = static_cast<uint64_t>(seed) * 1000003ULL + static_cast<uint64_t>(u);
        // 教师退火：前 teacherDecay 轮线性降到 0，之后纯自对局 PPO 接管。
        if (teacherDecay > 0) {
            const float scale = std::max(0.0f, 1.0f - float(u - 1) / float(teacherDecay));
            for (int s = 0; s < 2; ++s) learners[s]->config.teacherCoef = teacherCoef0 * scale;
        }
        if (entropyFinal != entropy0) {
            const float f = (updates > 1) ? float(u - 1) / float(updates - 1) : 1.0f;
            for (int s = 0; s < 2; ++s) {
                learners[s]->config.entCoef = entropy0 + (entropyFinal - entropy0) * f;
            }
        }
        std::array<std::vector<texas::Transition>, 2> tr;
        texas::RolloutStats stats;
        if (workers == 1) {
            texas::collectRollout(models, pool.empty() ? nullptr : &pool, rcfg, tr, stats);
        } else {
            // 按局切给多个线程。每线程独立 seed，所以数据分布一致、样本不重复。
            int base = rcfg.games / workers;
            int rem = rcfg.games % workers;
            std::vector<std::thread> threads;
            threads.reserve(static_cast<size_t>(workers));
            for (int w = 0; w < workers; ++w) {
                WorkerBatch& wb = workerBatches[static_cast<size_t>(w)];
                wb.games = base + (rem > 0 ? 1 : 0);
                if (rem > 0) --rem;
                wb.seed = rcfg.seed + static_cast<uint64_t>(w) * 1000003ULL;
                threads.emplace_back([&, w]() {
                    WorkerBatch& self = workerBatches[static_cast<size_t>(w)];
                    texas::RolloutConfig wc = rcfg;
                    wc.games = self.games;
                    wc.seed = self.seed;
                    texas::collectRollout(models, pool.empty() ? nullptr : &pool, wc, self.tr,
                                          self.stats);
                });
            }
            for (std::thread& th : threads) th.join();
            // gameId 加上累计偏移：GAE 按 gameId 分组，跨 worker 不能撞号。
            int offset = 0;
            for (int w = 0; w < workers; ++w) {
                WorkerBatch& wb = workerBatches[static_cast<size_t>(w)];
                for (int s = 0; s < 2; ++s) {
                    for (texas::Transition& t : wb.tr[static_cast<size_t>(s)]) t.gameId += offset;
                    tr[static_cast<size_t>(s)].insert(
                        tr[static_cast<size_t>(s)].end(),
                        std::make_move_iterator(wb.tr[static_cast<size_t>(s)].begin()),
                        std::make_move_iterator(wb.tr[static_cast<size_t>(s)].end()));
                    wb.tr[static_cast<size_t>(s)].clear();
                }
                offset += wb.games;
                addStats(stats, wb.stats);
                wb.stats = texas::RolloutStats{};
            }
        }

        double pg = 0, vl = 0, ent = 0, kl = 0, tea = 0, teaH = 0, teaM = 0, rtea = 0, retMean = 0,
               retStd = 0, valMean = 0, valStd = 0, valCorr = 0, rcrit = 0;
        double auxMse = 0, auxMae = 0, auxPred = 0, auxLab = 0;
        double oppMse = 0, oppMae = 0, oppPred = 0, oppLab = 0;

        std::array<texas::BatchTensors, 2> batches;
        std::array<texas::ppo::Stats, 2> sts;
        std::array<bool, 2> hasBatch = {false, false};
        for (int s = 0; s < 2; ++s) {
            if (tr[s].empty()) continue;
            texas::ppo::assignEpisodeReturns(tr[s], pcfg.gamma, pcfg.lambda);
            texas::buildBatch(tr[s], batches[s]);
            hasBatch[s] = true;
        }

        if (sharedActor) {
            // 共享 actor：两个座位的 batch 依次走同一个 Adam（等价于把座位
            // 当成同一策略的两半数据），critic 仍各训各的。
            for (int s = 0; s < 2; ++s) {
                if (!hasBatch[s]) continue;
                sts[s] = learners[0]->updateActorOnly(batches[s]);
                // 非共享路径的 critic 在主更新里先跑 epochs 轮，再加额外轮数；
                // 这里要显式补上主轮数，否则 extra=0 时 critic 完全不训练。
                sts[s].vLoss = learners[s]->updateCriticOnly(
                    batches[s], pcfg.epochs + pcfg.criticExtraEpochs);
            }
            // 教师回放：两个座位的样本进同一个池，只更新共享 actor。
            if (teacherReplay > 0 && learners[0]->config.teacherCoef > 0.0f) {
                for (int s = 0; s < 2; ++s) {
                    if (!hasBatch[s]) continue;
                    for (const texas::Transition& t : tr[s]) {
                        if (!t.hasTeacher) continue;
                        texas::TeacherSample ts;
                        ts.node = t.node;
                        ts.seq = t.seq;
                        ts.q = t.teacherQ;
                        ts.rangeEq = t.rangeEq;
                        ts.actions = t.actions;
                        replay[0].push_back(std::move(ts));
                        while (static_cast<int>(replay[0].size()) > teacherReplay * 2) {
                            replay[0].pop_front();
                        }
                    }
                }
                const float rcoef = learners[0]->config.teacherCoef * teacherReplayCoef;
                if (rcoef > 0.0f && replay[0].size() >= 8) {
                    rtea += learners[0]->distill(replay[0], teacherReplaySteps, rcoef,
                                                 pcfg.teacherTemp, pcfg.minibatch);
                }
            }
            if (criticReplay > 0) {
                for (int s = 0; s < 2; ++s) {
                    if (!hasBatch[s]) continue;
                    for (const texas::Transition& t : tr[s]) {
                        texas::CriticSample cs;
                        cs.node = t.node;
                        cs.seq = t.seq;
                        cs.extra = t.extra;
                        cs.ret = t.ret;
                        criticMemory[s].push_back(std::move(cs));
                        while (static_cast<int>(criticMemory[s].size()) > criticReplay) {
                            criticMemory[s].pop_front();
                        }
                    }
                    rcrit += learners[s]->updateCriticReplay(criticMemory[s], criticReplayEpochs,
                                                             pcfg.minibatch);
                }
            }
        } else {
            for (int s = 0; s < 2; ++s) {
                if (!hasBatch[s]) continue;
                sts[s] = learners[s]->update(batches[s]);
                // 价值回放：把这一轮的 (状态, 回报) 存进跨轮池，再在池上多跑几轮
                // critic。actor 更新完之后才做，避免影响本轮的重要性比值。
                if (criticReplay > 0) {
                    for (const texas::Transition& t : tr[s]) {
                        texas::CriticSample cs;
                        cs.node = t.node;
                        cs.seq = t.seq;
                        cs.extra = t.extra;
                        cs.ret = t.ret;
                        criticMemory[s].push_back(std::move(cs));
                        while (static_cast<int>(criticMemory[s].size()) > criticReplay) {
                            criticMemory[s].pop_front();
                        }
                    }
                    rcrit += learners[s]->updateCriticReplay(criticMemory[s], criticReplayEpochs,
                                                             pcfg.minibatch);
                }
                if (teacherReplay > 0 && learners[s]->config.teacherCoef > 0.0f) {
                    for (const texas::Transition& t : tr[s]) {
                        if (!t.hasTeacher) continue;
                        texas::TeacherSample ts;
                        ts.node = t.node;
                        ts.seq = t.seq;
                        ts.q = t.teacherQ;
                        ts.rangeEq = t.rangeEq;
                        ts.actions = t.actions;
                        replay[s].push_back(std::move(ts));
                        while (static_cast<int>(replay[s].size()) > teacherReplay) {
                            replay[s].pop_front();
                        }
                    }
                    const float rcoef = learners[s]->config.teacherCoef * teacherReplayCoef;
                    if (rcoef > 0.0f && replay[s].size() >= 8) {
                        rtea += learners[s]->distill(replay[s], teacherReplaySteps, rcoef,
                                                     pcfg.teacherTemp, pcfg.minibatch);
                    }
                }
            }
        }

        for (int s = 0; s < 2; ++s) {
            if (!hasBatch[s]) continue;
            const texas::ppo::Stats& st = sts[s];
            pg += st.pgLoss;
            vl += st.vLoss;
            ent += st.entropy;
            kl += st.kl;
            tea += st.teacherLoss;
            teaH += st.teacherEntropy;
            teaM += st.teacherMax;
            retMean += st.retMean;
            retStd += st.retStd;
            valMean += st.valueMean;
            valStd += st.valueStd;
            valCorr += st.valueCorr;
            auxMse += st.auxLoss;
            auxMae += st.auxMae;
            auxPred += st.auxPred;
            auxLab += st.auxLabel;
            oppMse += st.oppLoss;
            oppMae += st.oppMae;
            oppPred += st.oppPred;
            oppLab += st.oppLabel;
        }

        // 设备副本 → CPU 行动模型：下一轮 rollout 与 eval 都读 CPU 版。
        syncBackToCpu();

        const auto now = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(now - t0).count();
        const double winPct =
            stats.games > 0 ? 100.0 * double(stats.sbWins) / double(stats.games) : 0.0;
        const double sdPct =
            stats.games > 0 ? 100.0 * double(stats.showdowns) / double(stats.games) : 0.0;
        const double jamPct = stats.decisions > 0
                                  ? 100.0 * double(stats.allinDecisions) / double(stats.decisions)
                                  : 0.0;
        const double foldPct = stats.decisions > 0
                                   ? 100.0 * double(stats.folds) / double(stats.decisions)
                                   : 0.0;
        std::printf(
            "  %6d %7d %7.1f %6.1f %6.1f %6.1f %8.4f %8.4f %7.3f %6.1e %7.2f | ret %.3f±%.3f"
            " val %.3f±%.3f c%.2f tea %.2f/%.2f/%.2f rce %.2f cr %.3f aux %.4f/%.3f %.2f/%.2f"
            " opp %.4f/%.3f %.2f/%.2f\n",
            u, stats.games, winPct, sdPct, jamPct, foldPct, pg / 2, vl / 2, ent / 2, kl / 2,
            double(u) / secs, retMean / 2, retStd / 2, valMean / 2, valStd / 2, valCorr / 2,
            tea / 2, teaH / 2, teaM / 2, rtea / 2, rcrit / 2,
            auxMse / 2, auxMae / 2, auxPred / 2, auxLab / 2,
            oppMse / 2, oppMae / 2, oppPred / 2, oppLab / 2);
        std::fflush(stdout);

        if (evalEvery > 0 && (u % evalEvery == 0 || u == updates)) {
            const texas::EvalResult vr = texas::evalVsRule(models.actor, evalGames, 12345 + u);
            const texas::EvalResult vn = texas::evalVsRandom(models.actor, evalGames, 54321 + u);
            const texas::EvalResult vs = texas::evalSelfPlay(models.actor, evalGames / 2, 777 + u);
            texas::EvalResult vp, vse;
            if (rcfg.proProb > 0.0f) {
                vp = texas::evalVsPro(models.actor, evalGames, 13579 + u);
            }
            if (rcfg.searchProb > 0.0f) {
                vse = texas::evalVsSearch(models.actor, evalGames, 24680 + u);
            }
            // 多个基准一起报：只报 vs 规则机器人时，一个不看牌的固定下注模式也能刷到
            // +11.5 筹码/手（见 README 第五节）。
            const int g = std::max(1, stats.games);
            std::printf("         [eval] rule %+6.2f±%.2f | random %+6.2f±%.2f", vr.chipsPerHand,
                        vr.stderr, vn.chipsPerHand, vn.stderr);
            if (rcfg.proProb > 0.0f) {
                std::printf(" | pro %+6.2f±%.2f", vp.chipsPerHand, vp.stderr);
            }
            if (rcfg.searchProb > 0.0f) {
                std::printf(" | search %+6.2f±%.2f", vse.chipsPerHand, vse.stderr);
            }
            std::printf(
                " | self %+5.2f±%.2f | 对手 自%3d%% 尖%2d%% 冻%2d%% 近%2d%% 档%2d%% 规%2d%%"
                " 随%2d%% 胜%2d%% 搜%2d%% 稳%2d%% 弃%2d%%\n",
                vs.chipsPerHand, vs.stderr, 100 * stats.vsLatest / g,
                100 * stats.vsSelfSharp / g, 100 * stats.vsExtra / g,
                100 * stats.vsRecent / g, 100 * stats.vsArchive / g, 100 * stats.vsRule / g,
                100 * stats.vsRandom / g, 100 * stats.vsPro / g, 100 * stats.vsSearch / g,
                100 * stats.vsSolid / g, 100 * stats.vsFolder / g);
            std::fflush(stdout);

            auto snap = std::make_shared<texas::Snapshot>();
            snap->update = u;
            for (int s = 0; s < 2; ++s) {
                snap->actor[s] = texas::nn::cloneActor(models.actor[s]);
                snap->actor[s]->eval();
            }
            // 快照同时进 recent（滚动）和 archive（几何间隔）。recent 让对手记得
            // 自己最近什么样，archive 防止策略跑偏后再也遇不到早期的自己。
            pool.push(snap, recentCap, archiveCap);

            saveModels(models.actor, models.critic, outDir);
            for (int s = 0; s < 2; ++s) learners[s]->saveOptimizers(outDir);
        }
    }

    saveModels(models.actor, models.critic, outDir);
    for (int s = 0; s < 2; ++s) learners[s]->saveOptimizers(outDir);
    std::printf("models saved to %s\n", outDir.c_str());
    return 0;
}

// ---------------------------------------------------------------- eval

int cmdEval(int argc, char** argv) {
    const std::string dir = argStr(argc, argv, "--resume", "ckpt");
    const std::string vs = argStr(argc, argv, "--vs", "rule");
    const int hands = argInt(argc, argv, "--hands", 2000);
    const int seed = argInt(argc, argv, "--seed", 90001);
    const std::vector<std::string> ensDirs = argValues(argc, argv, "--ensemble");
    const int policySearch = argInt(argc, argv, "--policy-search", 0);

    // 网络尺寸必须与检查点一致；评估大网络时要显式传 --hidden/--lstm。
    texas::nn::ModelConfig mc;
    mc.hidden = argInt(argc, argv, "--hidden", 256);
    mc.lstmHidden = argInt(argc, argv, "--lstm", 128);
    texas::nn::Actor actors[2];
    texas::nn::Critic critics[2];
    for (int s = 0; s < 2; ++s) {
        actors[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        critics[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
    }
    if (!loadModels(actors, critics, dir)) {
        std::printf("cannot load model from %s\n", dir.c_str());
        return 1;
    }

    texas::EvalTeam team;
    std::vector<std::array<texas::nn::Actor, 2>> ensStorage;
    if (!buildTeam(team, actors, ensDirs, ensStorage, mc)) return 1;
    team.policyTemp = static_cast<float>(argDouble(argc, argv, "--temp", 1.0));
    team.policyGreedy = hasFlag(argc, argv, "--greedy");
    team.amountMid = hasFlag(argc, argv, "--amount-mid");
    team.preflopRaiseMult =
        static_cast<float>(argDouble(argc, argv, "--preflop-raise-mult", 1.0));
    team.postflopBetMult =
        static_cast<float>(argDouble(argc, argv, "--postflop-bet-mult", 1.0));
    team.rangeEqSamples = argInt(argc, argv, "--range-eq", 0);
    team.rangeEqIters = argInt(argc, argv, "--range-eq-iters", 48);
    team.rangeSearchSamples = argInt(argc, argv, "--range-search", 0);
    team.rangeSearchUniform = hasFlag(argc, argv, "--range-search-uniform");
    team.robustSearch = hasFlag(argc, argv, "--robust-search");
    team.robustMean = hasFlag(argc, argv, "--robust-mean");
    team.rangeSearchMinPot = argInt(argc, argv, "--range-search-min-pot", 0);
    team.rangeSearchDefenseOnly = hasFlag(argc, argv, "--range-search-defense-only");
    team.riverCfrIters = argInt(argc, argv, "--river-cfr", 0);
    team.riverCfrMinPot = argInt(argc, argv, "--river-cfr-min-pot", 0);
    team.riverCfrMaxRaises = argInt(argc, argv, "--river-cfr-raises", 1);
    if (hasFlag(argc, argv, "--river-cfr-no-allin")) team.riverCfrAllin = false;
    team.turnCfrIters = argInt(argc, argv, "--turn-cfr", 0);
    team.turnCfrMinPot = argInt(argc, argv, "--turn-cfr-min-pot", 0);
    team.turnCfrMaxRaises = argInt(argc, argv, "--turn-cfr-raises", 1);
    if (hasFlag(argc, argv, "--turn-cfr-no-allin")) team.turnCfrAllin = false;

    texas::EvalResult r;
    if (policySearch > 0) {
        if (vs == "mix") {
            r = texas::evalSearchTeamVsMix(team, hands, static_cast<uint64_t>(seed), policySearch);
        } else if (vs == "rule") {
            r = texas::evalSearchTeamVsRule(team, hands, static_cast<uint64_t>(seed), policySearch);
        } else {
            std::printf("--policy-search 只支持 --vs rule|mix\n");
            return 1;
        }
    } else if (vs == "random") {
        r = texas::evalTeamVsRandom(team, hands, static_cast<uint64_t>(seed));
    } else if (vs == "self") {
        r = texas::evalTeamSelfPlay(team, hands, static_cast<uint64_t>(seed));
    } else if (vs == "pro") {
        r = texas::evalTeamVsPro(team, hands, static_cast<uint64_t>(seed));
    } else if (vs == "search") {
        r = texas::evalTeamVsSearch(team, hands, static_cast<uint64_t>(seed));
    } else if (vs == "solid") {
        r = texas::evalTeamVsSolid(team, hands, static_cast<uint64_t>(seed));
    } else if (vs == "mix") {
        r = texas::evalTeamVsMix(team, hands, static_cast<uint64_t>(seed));
    } else {
        r = texas::evalTeamVsRule(team, hands, static_cast<uint64_t>(seed));
    }
    std::string label = vs;
    if (!ensDirs.empty()) {
        label += "+ens" + std::to_string(ensDirs.size() + 1);
    }
    if (policySearch > 0) {
        label += "+search" + std::to_string(policySearch);
    }
    if (team.rangeSearchSamples > 0) {
        label += "+rsearch" + std::to_string(team.rangeSearchSamples);
    }
    if (team.robustSearch) label += "+robust";
    if (team.robustMean) label += "mean";
    if (team.rangeSearchMinPot > 0) label += "+minpot" + std::to_string(team.rangeSearchMinPot);
    if (team.rangeSearchDefenseOnly) label += "+defonly";
    if (team.riverCfrIters > 0) label += "+rcfr" + std::to_string(team.riverCfrIters);
    if (team.riverCfrMinPot > 0) label += "+rcfrmin" + std::to_string(team.riverCfrMinPot);
    if (team.turnCfrIters > 0) label += "+tcfr" + std::to_string(team.turnCfrIters);
    if (team.turnCfrMinPot > 0) label += "+tcfrmin" + std::to_string(team.turnCfrMinPot);
    if (team.policyGreedy) {
        label += "+greedy";
    } else if (team.policyTemp != 1.0f) {
        label += "+T" + std::to_string(team.policyTemp);
    }
    if (team.amountMid) label += "+mid";
    std::printf("%s: %d 手, %+0.3f ± %.3f 筹码/手, %+0.1f bb/100, %+0.1f mbb/g\n", label.c_str(),
                r.hands, r.chipsPerHand, r.stderr, r.bbPer100, r.mbbPerGame);
    return 0;
}

// ---------------------------------------------------------------- match

// 两个检查点同牌对调对打：--a 的收益为正表示 a 更强。
// --ensemble DIR 可重复，给 a 侧叠加动作级集成成员。
int cmdMatch(int argc, char** argv) {
    const std::string dirA = argStr(argc, argv, "--a", "");
    const std::string dirB = argStr(argc, argv, "--b", "");
    const int hands = argInt(argc, argv, "--hands", 4000);
    const int seed = argInt(argc, argv, "--seed", 24680);
    const std::vector<std::string> ensDirs = argValues(argc, argv, "--ensemble");
    // a 侧用策略内搜索（每次决策抽 N 组隐藏牌，用自己策略续演到底）。
    const int searchA = argInt(argc, argv, "--search-a", 0);
    if (dirA.empty() || dirB.empty()) {
        std::printf("match 需要 --a DIR --b DIR\n");
        return 1;
    }

    // 两边可以是不同尺寸的网络（大网络 vs 旧检查点）：--hidden/--lstm 给 A，
    // --b-hidden/--b-lstm 给 B，未显式给时沿用 A 的尺寸。
    texas::nn::ModelConfig mc;
    mc.hidden = argInt(argc, argv, "--hidden", 256);
    mc.lstmHidden = argInt(argc, argv, "--lstm", 128);
    texas::nn::ModelConfig mcB = mc;
    mcB.hidden = argInt(argc, argv, "--b-hidden", static_cast<int>(mc.hidden));
    mcB.lstmHidden = argInt(argc, argv, "--b-lstm", static_cast<int>(mc.lstmHidden));
    texas::nn::Actor a[2], b[2];
    texas::nn::Critic ca[2], cb[2];
    for (int s = 0; s < 2; ++s) {
        a[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        b[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mcB));
        ca[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
        cb[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mcB));
    }
    if (!loadModels(a, ca, dirA)) {
        std::printf("cannot load model from %s\n", dirA.c_str());
        return 1;
    }
    if (!loadModels(b, cb, dirB)) {
        std::printf("cannot load model from %s\n", dirB.c_str());
        return 1;
    }

    // 分座位选模型：--a-sb / --a-bb 可以分别用另一个检查点的对应座位网络。
    // SB/BB 本来就是两套独立参数，混搭是合法的（特征口径完全相同）。
    const std::string aSb = argStr(argc, argv, "--a-sb", "");
    const std::string aBb = argStr(argc, argv, "--a-bb", "");
    auto loadSeat = [&](texas::nn::Actor& dst, const std::string& dir, int seat) {
        texas::nn::Actor tmp[2];
        for (int s = 0; s < 2; ++s) {
            tmp[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        }
        if (!loadActors(tmp, dir)) {
            std::printf("cannot load actors from %s\n", dir.c_str());
            return false;
        }
        dst = tmp[seat];
        return true;
    };
    if (!aSb.empty() && !loadSeat(a[0], aSb, 0)) return 1;
    if (!aBb.empty() && !loadSeat(a[1], aBb, 1)) return 1;

    texas::EvalTeam team;
    std::vector<std::array<texas::nn::Actor, 2>> ensStorage;
    if (!buildTeam(team, a, ensDirs, ensStorage, mc)) return 1;
    team.policyTemp = static_cast<float>(argDouble(argc, argv, "--a-temp", 1.0));
    team.policyGreedy = hasFlag(argc, argv, "--a-greedy");
    team.amountMid = hasFlag(argc, argv, "--a-amount-mid");
    team.preflopRaiseMult =
        static_cast<float>(argDouble(argc, argv, "--a-preflop-raise-mult", 1.0));
    team.postflopBetMult =
        static_cast<float>(argDouble(argc, argv, "--a-postflop-bet-mult", 1.0));
    team.rangeEqSamples = argInt(argc, argv, "--range-eq", 0);
    team.rangeEqIters = argInt(argc, argv, "--range-eq-iters", 48);
    team.rangeSearchSamples = argInt(argc, argv, "--range-search", 0);
    team.rangeSearchUniform = hasFlag(argc, argv, "--range-search-uniform");
    team.robustSearch = hasFlag(argc, argv, "--robust-search");
    team.robustMean = hasFlag(argc, argv, "--robust-mean");
    team.rangeSearchMinPot = argInt(argc, argv, "--range-search-min-pot", 0);
    team.rangeSearchDefenseOnly = hasFlag(argc, argv, "--range-search-defense-only");
    team.riverCfrIters = argInt(argc, argv, "--river-cfr", 0);
    team.riverCfrMinPot = argInt(argc, argv, "--river-cfr-min-pot", 0);
    team.riverCfrMaxRaises = argInt(argc, argv, "--river-cfr-raises", 1);
    if (hasFlag(argc, argv, "--river-cfr-no-allin")) team.riverCfrAllin = false;
    team.turnCfrIters = argInt(argc, argv, "--turn-cfr", 0);
    team.turnCfrMinPot = argInt(argc, argv, "--turn-cfr-min-pot", 0);
    team.turnCfrMaxRaises = argInt(argc, argv, "--turn-cfr-raises", 1);
    if (hasFlag(argc, argv, "--turn-cfr-no-allin")) team.turnCfrAllin = false;

    texas::MatchLog mlog;
    const texas::EvalResult r =
        (searchA > 0) ? texas::evalSearchTeamMatch(team, b, hands, static_cast<uint64_t>(seed),
                                                   searchA)
                      : texas::evalTeamMatch(team, b, hands, static_cast<uint64_t>(seed), 24,
                                             static_cast<float>(argDouble(argc, argv, "--b-temp", 1.0)),
                                             hasFlag(argc, argv, "--b-greedy"), &mlog,
                                             static_cast<float>(argDouble(
                                                 argc, argv, "--b-preflop-raise-mult", 1.0)),
                                             static_cast<float>(argDouble(
                                                 argc, argv, "--b-postflop-bet-mult", 1.0)));
    // --log FILE：逐手写 seat,pot,net（同牌对调的两遍都写），用来按底池分桶。
    // 这样内部对打可以和外部 Slumbot 日志用同一套桶做对照。
    const std::string matchLog = argStr(argc, argv, "--log", "");
    if (!matchLog.empty()) {
        if (FILE* fp = std::fopen(matchLog.c_str(), "w")) {
            std::fprintf(fp, "seat,pot,net\n");
            for (size_t i = 0; i < mlog.pots.size(); ++i) {
                std::fprintf(fp, "%d,%d,%d\n", mlog.seats[i], mlog.pots[i], mlog.nets[i]);
            }
            std::fclose(fp);
        }
    }
    std::string labelA = dirA;
    if (!aSb.empty()) labelA += "[sb=" + aSb + "]";
    if (!aBb.empty()) labelA += "[bb=" + aBb + "]";
    if (!ensDirs.empty()) {
        labelA += "+ens" + std::to_string(ensDirs.size() + 1);
    }
    if (searchA > 0) {
        labelA += "+search" + std::to_string(searchA);
    }
    if (team.rangeSearchSamples > 0) {
        labelA += "+rsearch" + std::to_string(team.rangeSearchSamples);
    }
    if (team.robustSearch) labelA += "+robust";
    if (team.robustMean) labelA += "mean";
    if (team.rangeSearchMinPot > 0) labelA += "+minpot" + std::to_string(team.rangeSearchMinPot);
    if (team.rangeSearchDefenseOnly) labelA += "+defonly";
    if (team.riverCfrIters > 0) labelA += "+rcfr" + std::to_string(team.riverCfrIters);
    if (team.turnCfrIters > 0) labelA += "+tcfr" + std::to_string(team.turnCfrIters);
    if (team.policyGreedy) {
        labelA += "+greedy";
    } else if (team.policyTemp != 1.0f) {
        labelA += "+T" + std::to_string(team.policyTemp);
    }
    if (team.amountMid) labelA += "+mid";
    std::printf("match %s vs %s: %d 手, %+0.3f ± %.3f 筹码/手, %+0.1f bb/100\n",
                labelA.c_str(), dirB.c_str(), r.hands, r.chipsPerHand, r.stderr, r.bbPer100);
    std::printf("  分座位：小盲 %+0.3f ± %.3f | 大盲 %+0.3f ± %.3f 筹码/手\n",
                r.sbChipsPerHand, r.sbStderr, r.bbChipsPerHand, r.bbStderr);
    return 0;
}

// ---------------------------------------------------------------- soup

// 检查点平均（model soup）：把两个同结构 actor 的参数按 alpha 线性插值。
// 只用于评估；critic 从 a 复制，优化器不复制。
int cmdSoup(int argc, char** argv) {
    const std::string dirA = argStr(argc, argv, "--a", "");
    const std::string dirB = argStr(argc, argv, "--b", "");
    const std::string outDir = argStr(argc, argv, "--out", "");
    const float alpha = static_cast<float>(argDouble(argc, argv, "--alpha", 0.5));
    if (dirA.empty() || dirB.empty() || outDir.empty()) {
        std::printf("soup 需要 --a DIR --b DIR --out DIR\n");
        return 1;
    }

    texas::nn::ModelConfig mc;
    mc.hidden = argInt(argc, argv, "--hidden", 256);
    mc.lstmHidden = argInt(argc, argv, "--lstm", 128);
    std::error_code ec;
    std::filesystem::create_directories(outDir, ec);
    for (int s = 0; s < 2; ++s) {
        texas::nn::Actor a(std::make_shared<texas::nn::ActorImpl>(mc));
        texas::nn::Actor b(std::make_shared<texas::nn::ActorImpl>(mc));
        texas::nn::Critic ca(std::make_shared<texas::nn::CriticImpl>(mc));
        texas::nn::Critic cb(std::make_shared<texas::nn::CriticImpl>(mc));
        if (!a->loadFrom(dirA + "/actor" + std::to_string(s) + ".pt") ||
            !b->loadFrom(dirB + "/actor" + std::to_string(s) + ".pt")) {
            std::printf("cannot load actors from %s / %s\n", dirA.c_str(), dirB.c_str());
            return 1;
        }
        if (!ca->loadFrom(dirA + "/critic" + std::to_string(s) + ".pt")) {
            std::printf("cannot load critic from %s\n", dirA.c_str());
            return 1;
        }
        {
            torch::NoGradGuard ng;
            auto pa = a->parameters();
            auto pb = b->parameters();
            if (pa.size() != pb.size()) {
                std::printf("parameter size mismatch\n");
                return 1;
            }
            for (size_t i = 0; i < pa.size(); ++i) {
                pa[i].data().mul_(alpha).add_(pb[i].data(), 1.0 - alpha);
            }
        }
        a->saveTo(outDir + "/actor" + std::to_string(s) + ".pt");
        ca->saveTo(outDir + "/critic" + std::to_string(s) + ".pt");
    }
    std::printf("soup: %s × %.2f + %s × %.2f -> %s\n", dirA.c_str(), alpha, dirB.c_str(),
                1.0 - alpha, outDir.c_str());
    return 0;
}

// ---------------------------------------------------------------- botmatch

// 两个脚本机器人对打（胜率机器人 vs 规则机器人），用来确认高一级的基准
// 确实更强。返回值的正负是"胜率机器人"的每手净收益。
int cmdBotMatch(int argc, char** argv) {
    const int games = argInt(argc, argv, "--hands", 2000);
    const int seed = argInt(argc, argv, "--seed", 2024);
    const std::string typeA = argStr(argc, argv, "--a", "pro");
    const std::string typeB = argStr(argc, argv, "--b", "rule");
    const int searchSamples = argInt(argc, argv, "--search-samples", 8);

    auto botMove = [&](const std::string& type, texas::HUState& st, texas::Rng& rng) {
        if (type == "random") {
            texas::LegalMove lm[texas::kNumActions];
            const int n = texas::legalMoves(st, lm);
            if (n <= 0) return texas::Move{};
            const int pick = static_cast<int>(rng() % static_cast<size_t>(n));
            texas::Move mv = lm[pick].mv;
            if (mv.kind == texas::MK_RAISE && lm[pick].toHi > int(mv.to)) {
                std::uniform_real_distribution<double> u(0.0, 1.0);
                mv.to = static_cast<int16_t>(
                    texas::amountInBand(int(mv.to), lm[pick].toHi, u(rng)));
            }
            return mv;
        }
        if (type == "search") return texas::searchAgentMove(st, rng, searchSamples, 3);
        if (type == "pro") return texas::proAgentMove(st, rng);
        return texas::ruleAgentMove(st, rng);
    };

    std::vector<texas::HUState> deals(static_cast<size_t>(games));
    texas::Rng dealRng(static_cast<uint64_t>(seed));
    for (texas::HUState& d : deals) d = texas::dealNewHand(dealRng);

    std::vector<int> pairs(static_cast<size_t>(games), 0);
    texas::Rng rng(static_cast<uint64_t>(seed) + 17);
    for (int pass = 0; pass < 2; ++pass) {
        const int aSeat = (pass == 0) ? 0 : 1;
        for (int i = 0; i < games; ++i) {
            texas::HUState st = deals[static_cast<size_t>(i)];
            int guard = 0;
            while (!st.finished && guard++ < 128) {
                const int seat = st.toAct;
                const texas::Move mv = (seat == aSeat) ? botMove(typeA, st, rng)
                                                       : botMove(typeB, st, rng);
                texas::applyMove(st, mv, rng);
            }
            pairs[static_cast<size_t>(i)] += texas::payoff(st, aSeat);
        }
    }

    double sum = 0, sq = 0;
    for (int v : pairs) {
        sum += double(v);
        sq += double(v) * double(v);
    }
    const double n = double(games);
    const double mean = sum / n / 2.0;  // 两手牌的和 -> 每手
    const double var = std::max(sq / n - (sum / n) * (sum / n), 0.0);
    const double se = std::sqrt(var / n) / 2.0;
    std::printf("%s vs %s: %d 手, %+0.3f ± %.3f 筹码/手, %+0.1f bb/100\n", typeA.c_str(),
                typeB.c_str(), games * 2, mean, se, mean / double(texas::kBigBlind) * 100.0);
    return 0;
}

// ---------------------------------------------------------------- probe

// 打印某个具体局面上各动作的平均策略概率。用来看"学到的策略像不像会打牌"，
// 但注意：策略形状不是判据，对打胜率才是（见 README）。
void probeOne(const texas::nn::Actor& actor, const char* label, const texas::HUState& st,
              bool printTeacherQ = false, int teacherSamples = 1) {
    texas::LegalMove lm[texas::kNumActions];
    const int n = texas::legalMoves(st, lm);
    if (n <= 0) {
        std::printf("  %-28s (无合法动作)\n", label);
        return;
    }
    const int64_t B = 1;
    torch::Tensor nodeT = torch::zeros({B, texas::kNodeInput}, torch::kFloat32);
    torch::Tensor seqT = torch::zeros({B, texas::kHistSteps, texas::kHistDim}, torch::kFloat32);
    torch::Tensor featT = torch::zeros({n, texas::kActionInput}, torch::kFloat32);
    torch::Tensor sampleT = torch::zeros({n}, torch::kLong);
    torch::Tensor idT = torch::zeros({n}, torch::kLong);

    texas::Obs obs;
    texas::encodeObservation(st, obs, 256);
    std::copy(obs.node.begin(), obs.node.end(), nodeT.data_ptr<float>());
    std::copy(obs.seq.begin(), obs.seq.end(), seqT.data_ptr<float>());
    for (int i = 0; i < n; ++i) {
        std::array<float, texas::kActionInput> af;
        texas::encodeActionFeature(st, lm[i], af);
        std::copy(af.begin(), af.end(), featT.data_ptr<float>() + i * texas::kActionInput);
        sampleT.data_ptr<int64_t>()[i] = 0;
        idT.data_ptr<int64_t>()[i] = lm[i].slot;
    }

    torch::NoGradGuard ng;
    torch::Tensor logits = actor->forward(nodeT, seqT, featT, sampleT, idT);
    const float* lp = logits.data_ptr<float>();
    float probs[texas::kNumActions];
    texas::ppo::maskedSoftmax(lp, texas::kNumActions, probs);

    std::printf("  %-24s [str=%.2f eq=%.2f]", label,
                double(obs.node[texas::kHoleDim + texas::kBoardDim + 14]),
                double(obs.equityPerfect));
    char buf[32];
    for (int i = 0; i < n; ++i) {
        const texas::Move& mv = lm[i].mv;
        if (lm[i].slot == texas::kSlotFold) std::snprintf(buf, sizeof buf, "fold");
        else if (lm[i].slot == texas::kSlotCheckCall) std::snprintf(buf, sizeof buf, "call");
        else if (lm[i].slot == texas::kSlotAllIn) std::snprintf(buf, sizeof buf, "jam");
        else std::snprintf(buf, sizeof buf, "b%d-%d", int(mv.to), lm[i].toHi);
        std::printf(" %s=%.2f", buf, probs[lm[i].slot]);
    }
    std::printf("\n");

    if (printTeacherQ) {
        std::array<float, texas::kNumActions> q;
        texas::teacherActionValues(st, st.toAct, q, teacherSamples);
        std::printf("    teacherQ:");
        for (int i = 0; i < n; ++i) {
            const texas::Move& mv = lm[i].mv;
            if (lm[i].slot == texas::kSlotFold) std::snprintf(buf, sizeof buf, "fold");
            else if (lm[i].slot == texas::kSlotCheckCall) std::snprintf(buf, sizeof buf, "call");
            else if (lm[i].slot == texas::kSlotAllIn) std::snprintf(buf, sizeof buf, "jam");
            else std::snprintf(buf, sizeof buf, "b%d", int(mv.to));
            std::printf(" %s=%.4f", buf, q[lm[i].slot]);
        }
        std::printf("\n");
    }
}

int cmdProbe(int argc, char** argv) {
    const std::string dir = argStr(argc, argv, "--resume", "ckpt");
    texas::nn::ModelConfig mc;
    texas::nn::Actor actors[2];
    texas::nn::Critic critics[2];
    for (int s = 0; s < 2; ++s) {
        actors[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        critics[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
    }
    if (!loadModels(actors, critics, dir)) {
        std::printf("cannot load model from %s\n", dir.c_str());
        return 1;
    }

    auto C = [](const char* t) { return texas::parseCard(t); };
    struct Case {
        const char* label;
        const char* a0;
        const char* b0;
        const char* a1;
        const char* b1;
        const char* board[5];
        int boardN;
    };
    // 公共牌必须一次给满五张：前 boardN 张是已发出的，后面的是未发出的
    // runout（只影响全下后的摊牌与教师价值，不影响当前策略的输入）。
    const Case cases[] = {
        {"AA 翻牌前(先行动)", "Ah", "Ad", "7c", "2d", {"3h", "4s", "8d", "9c", "Js"}, 0},
        {"KK 翻牌前(先行动)", "Kh", "Kd", "7c", "2d", {"3h", "4s", "8d", "9c", "Js"}, 0},
        {"AKs 翻牌前(先行动)", "Ah", "Kh", "7c", "2d", {"3h", "4s", "8d", "9c", "Js"}, 0},
        {"72o 翻牌前(先行动)", "7h", "2d", "Ac", "Kc", {"3h", "4s", "8d", "9c", "Js"}, 0},
        {"AA 在 K72 面(先行动)", "Ah", "Ad", "Kc", "Qd", {"Kd", "7s", "2c", "4h", "9s"}, 3},
        {"72o 在 K72 面(先行动)", "7h", "2d", "Ac", "Qc", {"Kd", "7s", "2c", "4h", "9s"}, 3},
    };

    const bool showTeacherQ = hasFlag(argc, argv, "--teacher-q");
    const int teacherSamples = argInt(argc, argv, "--teacher-samples", 16);
    std::printf("平均策略（动作后金额，jam=全下）:\n");
    for (const Case& c : cases) {
        texas::Card h0[2] = {C(c.a0), C(c.b0)};
        texas::Card h1[2] = {C(c.a1), C(c.b1)};
        texas::Card board[5] = {0, 0, 0, 0, 0};
        for (int i = 0; i < 5; ++i) {
            if (c.board[i]) board[i] = C(c.board[i]);
        }
        texas::HUState st = texas::makeState(h0, h1, board, c.boardN);
        probeOne(actors[0], c.label, st, showTeacherQ, teacherSamples);
    }
    return 0;
}

// ---------------------------------------------------------------- sanity

// 端到端体检：把优势换成一个已知与牌力强相关的合成信号，看 actor 能不能学会
// "强牌加注、弱牌弃牌"。如果连这个都学不会，问题在 actor/PPO 链路；如果学得会，
// 说明链路没问题，真实训练里的困难来自优势的噪声（下注树的方差）。
int cmdSanity(int argc, char** argv) {
    const int steps = argInt(argc, argv, "--steps", 300);
    texas::nn::ModelConfig mc;
    texas::ModelSet models;
    for (int s = 0; s < 2; ++s) {
        models.actor[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        models.critic[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
    }
    texas::RolloutConfig rc;
    rc.games = 256;
    rc.seed = 4242;
    rc.historicalProb = 0.0f;
    rc.ruleProb = 0.0f;
    rc.randomProb = 0.0f;
    std::array<std::vector<texas::Transition>, 2> tr;
    texas::RolloutStats st;
    texas::collectRollout(models, nullptr, rc, tr, st);

    // 合成优势：强牌加注给正优势、弱牌弃牌给正优势。
    // 关键是优势必须"依赖动作"，否则每个动作被同等放大，政策学不到偏好。
    for (int s = 0; s < 2; ++s) {
        for (auto& t : tr[s]) {
            const float strength = t.node[texas::kHoleDim + texas::kBoardDim + 14];
            const float centered = 2.0f * (strength - 0.5f);
            float adv = 0.0f;
            for (const auto& a : t.actions) {
                if (a.slot != t.action) continue;
                if (a.mv.kind == texas::MK_RAISE || a.mv.kind == texas::MK_ALLIN) adv = centered;
                else if (a.mv.kind == texas::MK_FOLD) adv = -centered;
            }
            t.adv = adv;
            t.ret = adv;
        }
    }
    std::printf("样本 %zu / %zu，合成优势 |adv| 均值 ≈ 0.5\n", tr[0].size(), tr[1].size());

    texas::ppo::Config pcfg;
    pcfg.lr = 1e-3f;
    texas::Learner learner(models.actor[0], models.critic[0], pcfg, torch::kCPU);
    texas::BatchTensors batch;
    texas::buildBatch(tr[0], batch);
    for (int u = 0; u < steps; ++u) learner.update(batch);

    std::printf("训练 %d 轮合成优势后：\n", steps);
    auto C = [](const char* t) { return texas::parseCard(t); };
    const char* hands[][2] = {{"Ah", "Ad"}, {"7h", "2d"}, {"Kh", "Kd"}, {"8c", "5d"}};
    const char* names[] = {"AA", "72o", "KK", "85o"};
    for (int i = 0; i < 4; ++i) {
        texas::Card h0[2] = {C(hands[i][0]), C(hands[i][1])};
        texas::Card h1[2] = {C("Qc"), C("Jc")};
        texas::HUState state = texas::makeState(h0, h1, nullptr, 0);
        probeOne(models.actor[0], names[i], state);
    }
    return 0;
}

// ---------------------------------------------------------------- distill

// 教师蒸馏的端到端体检：固定一批带教师标注的样本，把 PPO 优势清零，只跑
// 蒸馏损失。如果 CE 学不到教师目标熵 H，问题在优化/梯度链路；如果 CE 能
// 降到 H 附近，说明在线训练里剩下的问题是样本分布和教师信号噪声。
int cmdDistill(int argc, char** argv) {
    const int steps = argInt(argc, argv, "--steps", 400);
    const int games = argInt(argc, argv, "--games", 384);
    const int samples = argInt(argc, argv, "--teacher-samples", 8);
    const int styles = argInt(argc, argv, "--teacher-styles", 3);
    const float temp = static_cast<float>(argDouble(argc, argv, "--teacher-temp", 0.5));

    texas::nn::ModelConfig mc;
    texas::ModelSet models;
    for (int s = 0; s < 2; ++s) {
        models.actor[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        models.critic[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
    }

    texas::RolloutConfig rc;
    rc.games = games;
    rc.seed = 4242;
    rc.historicalProb = 0.0f;
    rc.ruleProb = 0.0f;
    rc.randomProb = 0.0f;
    rc.teacher = true;
    rc.teacherProb = 0.25f;
    rc.teacherSamples = samples;
    rc.teacherStyles = styles;
    rc.oracleReward = false;
    std::array<std::vector<texas::Transition>, 2> tr;
    texas::RolloutStats st;
    texas::collectRollout(models, nullptr, rc, tr, st);
    texas::ppo::assignEpisodeReturns(tr[0], 1.0f, 0.95f);
    texas::BatchTensors batch;
    texas::buildBatch(tr[0], batch);
    batch.adv.zero_();  // 只测教师蒸馏，不跑 PPO

    int labeled = 0;
    for (const auto& t : tr[0]) labeled += t.hasTeacher ? 1 : 0;
    std::printf("样本 %zu / %zu（seats 0/1），其中 seat0 带教师标注 %d 条\n", tr[0].size(),
                tr[1].size(), labeled);

    texas::ppo::Config pcfg;
    pcfg.lr = static_cast<float>(argDouble(argc, argv, "--lr", 1e-3));
    pcfg.epochs = argInt(argc, argv, "--epochs", 4);
    pcfg.minibatch = argInt(argc, argv, "--minibatch", 128);
    pcfg.entCoef = 0.0f;
    pcfg.teacherCoef = 1.0f;
    pcfg.teacherTemp = temp;
    pcfg.criticExtraEpochs = 0;

    texas::Learner learner(models.actor[0], models.critic[0], pcfg, torch::kCPU);
    for (int u = 1; u <= steps; ++u) {
        const texas::ppo::Stats s = learner.update(batch);
        if (u % 25 == 0 || u == steps) {
            std::printf("  step %4d  CE %.3f  H %.3f  KL %.3f  top1 %.2f  ent %.3f\n", u,
                        s.teacherLoss, s.teacherEntropy, s.teacherLoss - s.teacherEntropy,
                        s.teacherMax, s.entropy);
        }
    }

    models.actor[0]->eval();
    const texas::Card opp[2] = {texas::parseCard("Kc"), texas::parseCard("Qc")};
    const texas::Card board[5] = {texas::parseCard("3h"), texas::parseCard("4s"),
                                  texas::parseCard("8d"), texas::parseCard("9c"),
                                  texas::parseCard("Js")};
    struct Hand { const char* name; const char* a; const char* b; };
    const Hand hands[] = {{"AA", "Ah", "Ad"}, {"72o", "7h", "2d"}, {"AKs", "Ah", "Kh"},
                          {"85o", "8c", "5d"}};
    for (const Hand& h : hands) {
        const texas::Card hole[2] = {texas::parseCard(h.a), texas::parseCard(h.b)};
        const texas::HUState state = texas::makeState(hole, opp, board, 0);
        probeOne(models.actor[0], h.name, state);
    }
    return 0;
}

// ---------------------------------------------------------------- critfit

// 单独验证 critic 的拟合能力：采一批自对局数据，然后把价值网在这批固定数据上
// 反复训练。如果 RMSE 降不下去，问题在价值网结构或优化；如果能降下去，问题
// 就在"每次只有 2000 个新样本、目标还在漂移"的训练循环上。
int cmdCritFit(int argc, char** argv) {
    const int steps = argInt(argc, argv, "--steps", 1500);
    const std::string resumeDir = argStr(argc, argv, "--resume", "");
    texas::nn::ModelConfig mc;
    mc.hidden = argInt(argc, argv, "--hidden", 256);
    mc.lstmHidden = argInt(argc, argv, "--lstm", 128);

    texas::ModelSet models;
    for (int s = 0; s < 2; ++s) {
        models.actor[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        models.critic[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
    }
    if (!resumeDir.empty() && !loadModels(models.actor, models.critic, resumeDir)) {
        std::printf("cannot load model from %s\n", resumeDir.c_str());
        return 1;
    }
    texas::RolloutConfig rc;
    rc.games = argInt(argc, argv, "--games", 256);
    rc.seed = static_cast<uint64_t>(argInt(argc, argv, "--seed", 42));
    rc.historicalProb = 0.0f;
    rc.ruleProb = static_cast<float>(argDouble(argc, argv, "--rule-prob", 0.0));
    rc.randomProb = static_cast<float>(argDouble(argc, argv, "--random-prob", 0.0));
    rc.proProb = static_cast<float>(argDouble(argc, argv, "--pro-prob", 0.0));
    rc.searchProb = static_cast<float>(argDouble(argc, argv, "--search-prob", 0.0));
    rc.searchSamples = argInt(argc, argv, "--search-samples", 8);
    std::array<std::vector<texas::Transition>, 2> tr;
    texas::RolloutStats st;
    texas::collectRollout(models, nullptr, rc, tr, st);
    std::printf("transitions: %zu / %zu\n", tr[0].size(), tr[1].size());
    texas::ppo::assignEpisodeReturns(tr[0], 1.0f, 0.95f);
    texas::BatchTensors b;
    texas::buildBatch(tr[0], b);
    std::printf("ret mean %.4f std %.4f | oldValue mean %.4f std %.4f\n",
                b.ret.mean().item<double>(), b.ret.std().item<double>(),
                b.oldValue.mean().item<double>(), b.oldValue.std().item<double>());

    auto opt = torch::optim::Adam(models.critic[0]->parameters(), torch::optim::AdamOptions(1e-3));
    const double wd = argDouble(argc, argv, "--weight-decay", 0.0);
    for (auto& g : opt.param_groups()) {
        static_cast<torch::optim::AdamOptions&>(g.options()).weight_decay(wd);
    }
    // 8:2 切分：训练集和留出集分开报。只报训练集 RMSE 会把"记住这批样本"
    // 当成"学会了价值"，这正是线上 c 只有 0.3 却看不出原因的地方。
    const int64_t N = b.node.size(0);
    torch::manual_seed(static_cast<uint64_t>(argInt(argc, argv, "--seed", 42)));
    torch::Tensor perm = torch::randperm(N, torch::TensorOptions().dtype(torch::kLong));
    const int64_t nTrain = std::max<int64_t>(8, (N * 8) / 10);
    torch::Tensor trIdx = perm.slice(0, 0, nTrain);
    torch::Tensor teIdx = perm.slice(0, nTrain, N);
    torch::Tensor nodeTr = b.node.index_select(0, trIdx);
    torch::Tensor seqTr = b.seq.index_select(0, trIdx);
    torch::Tensor extraTr = b.extra.index_select(0, trIdx);
    torch::Tensor retTr = b.ret.index_select(0, trIdx);
    torch::Tensor nodeTe = b.node.index_select(0, teIdx);
    torch::Tensor seqTe = b.seq.index_select(0, teIdx);
    torch::Tensor extraTe = b.extra.index_select(0, teIdx);
    torch::Tensor retTe = b.ret.index_select(0, teIdx);
    // step 0：先量一下"未拟合前"的留出集表现，这才是线上 critic 的真实水平。
    {
        torch::NoGradGuard ng;
        auto vTe = models.critic[0]->forward(nodeTe, seqTe, extraTe).squeeze(-1);
        const double rmseTe = std::sqrt((vTe - retTe).pow(2).mean().item<double>());
        auto rc = retTe - retTe.mean();
        auto vc = vTe - vTe.mean();
        const double corr =
            (rc * vc).mean().item<double>() /
            std::max(1e-8, (rc.std() * vc.std()).item<double>());
        std::printf("  step    0  rmse(holdout) %.4f  corr %.3f  predStd %.4f\n", rmseTe, corr,
                    vTe.std().item<double>());
    }
    for (int it = 1; it <= steps; ++it) {
        opt.zero_grad();
        auto v = models.critic[0]->forward(nodeTr, seqTr, extraTr).squeeze(-1);
        auto loss = 0.5 * (v - retTr).pow(2).mean();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(models.critic[0]->parameters(), 5.0);
        opt.step();
        if (it % 100 == 0) {
            torch::NoGradGuard ng;
            auto vTe = models.critic[0]->forward(nodeTe, seqTe, extraTe).squeeze(-1);
            const double rmseTe =
                std::sqrt(2.0 * 0.5 * (vTe - retTe).pow(2).mean().item<double>());
            auto rc = retTe - retTe.mean();
            auto vc = vTe - vTe.mean();
            const double corr =
                (rc * vc).mean().item<double>() /
                std::max(1e-8, (rc.std() * vc.std()).item<double>());
            std::printf("  step %4d  rmse(train) %.4f  rmse(holdout) %.4f  corr %.3f  predStd %.4f\n",
                        it, std::sqrt(2.0 * loss.item<double>()), rmseTe, corr,
                        vTe.std().item<double>());
        }
    }
    return 0;
}

// ---------------------------------------------------------------- bench

// 本地自对局的动作分布统计：回答"策略在关键节点到底选了什么"。
// 现有评估只给 EV，看不出风格（例如 BB 面对开牌是否从不跟注）。
int cmdPolicyStats(int argc, char** argv) {
    const std::string dir = argStr(argc, argv, "--resume", "ckpt_v12");
    const int hands = argInt(argc, argv, "--hands", 3000);
    const float temp = static_cast<float>(argDouble(argc, argv, "--temp", 0.1));
    const int seed = argInt(argc, argv, "--seed", 20240518);
    texas::nn::ModelConfig mc;
    mc.hidden = argInt(argc, argv, "--hidden", 256);
    mc.lstmHidden = argInt(argc, argv, "--lstm", 128);
    texas::nn::Actor actors[2];
    for (int s = 0; s < 2; ++s) {
        actors[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
    }
    if (!loadActors(actors, dir)) {
        std::printf("cannot load actors from %s\n", dir.c_str());
        return 1;
    }

    struct Cnt {
        int n = 0, fold = 0, call = 0, raise = 0;
        double toSum = 0.0;  // 加注额 / 底池（只有加注样本）
    };
    auto add = [](Cnt& c, const texas::Move& mv, int pot) {
        ++c.n;
        if (mv.kind == texas::MK_FOLD) {
            ++c.fold;
        } else if (mv.kind == texas::MK_CHECK || mv.kind == texas::MK_CALL) {
            ++c.call;
        } else {
            ++c.raise;
            if (pot > 0) c.toSum += double(mv.to) / double(pot);
        }
    };
    Cnt sbOpen, bbVsOpen, sbVs3bet, bbVs4bet, bbFlopLead, bbFlopVsBet, sbFlopVsBet;
    Cnt sbRiverLead, bbRiverLead, sbRiverVsBet, bbRiverVsBet;
    texas::Rng rng(static_cast<uint64_t>(seed));
    for (int h = 0; h < hands; ++h) {
        texas::HUState s = texas::dealNewHand(rng);
        int guard = 0;
        while (!s.finished && guard++ < 96) {
            const int seat = s.toAct;
            const int need = texas::toCall(s, seat);
            const int pot = s.pot();
            const texas::Move mv = texas::policyMove(&actors[0], &actors[1], s, rng, temp);
            if (s.street == texas::ST_PREFLOP) {
                if (seat == 0 && s.numRaises == 0) {
                    add(sbOpen, mv, pot);  // 翻前第一次行动（面对盲注）
                } else if (seat == 1 && need > 0 && s.numRaises <= 1) {
                    add(bbVsOpen, mv, pot);
                } else if (seat == 1 && need > 0 && s.numRaises >= 2) {
                    add(bbVs4bet, mv, pot);  // BB 3-bet 后被 4-bet
                } else if (seat == 0 && need > 0 && s.numRaises >= 2) {
                    add(sbVs3bet, mv, pot);
                }
            } else if (s.street == texas::ST_FLOP) {
                if (seat == 1 && need == 0) {
                    add(bbFlopLead, mv, pot);
                } else if (seat == 1) {
                    add(bbFlopVsBet, mv, pot);
                } else if (seat == 0 && need > 0) {
                    add(sbFlopVsBet, mv, pot);
                }
            }
            else if (s.street == texas::ST_RIVER) {
                if (seat == 0 && need == 0) add(sbRiverLead, mv, pot);
                else if (seat == 0) add(sbRiverVsBet, mv, pot);
                else if (need == 0) add(bbRiverLead, mv, pot);
                else add(bbRiverVsBet, mv, pot);
            }
            texas::applyMove(s, mv, rng);
        }
    }
    auto pct = [](int a, int n) { return n > 0 ? 100.0 * double(a) / double(n) : 0.0; };
    auto show = [&](const char* name, const Cnt& c) {
        std::printf("  %-14s n=%5d | fold %5.1f%% | call %5.1f%% | raise %5.1f%%", name, c.n,
                    pct(c.fold, c.n), pct(c.call, c.n), pct(c.raise, c.n));
        if (c.raise > 0) std::printf(" | 加注额/底池均值 %.2f", c.toSum / c.raise);
        std::printf("\n");
    };
    std::printf("policy-stats: %s | T=%.2f | %d 手\n", dir.c_str(), temp, hands);
    show("SB 翻前开牌", sbOpen);
    show("BB 面对开牌", bbVsOpen);
    show("SB 面对3-bet", sbVs3bet);
    show("BB 面对4-bet", bbVs4bet);
    show("BB 翻牌先动", bbFlopLead);
    show("BB 翻牌面对注", bbFlopVsBet);
    show("SB 翻牌面对注", sbFlopVsBet);
    show("SB 河牌先动", sbRiverLead);
    show("BB 河牌先动", bbRiverLead);
    show("SB 河牌面对注", sbRiverVsBet);
    show("BB 河牌面对注", bbRiverVsBet);
    return 0;
}

int cmdBench(int argc, char** argv) {
    const int hands = argInt(argc, argv, "--hands", 2000);
    texas::nn::ModelConfig mc;
    texas::nn::Actor actors[2];
    for (int s = 0; s < 2; ++s) {
        actors[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
    }
    const texas::EvalResult r = texas::evalVsRule(actors, hands, 4242);
    std::printf("未训练策略 vs 规则机器人: %+0.3f ± %.3f 筹码/手 (%+0.1f bb/100, %+0.1f mbb/g)\n",
                r.chipsPerHand, r.stderr, r.bbPer100, r.mbbPerGame);
    const texas::EvalResult rr = texas::evalVsRandom(actors, hands, 4243);
    std::printf("未训练策略 vs 随机策略:   %+0.3f ± %.3f 筹码/手 (%+0.1f bb/100, %+0.1f mbb/g)\n",
                rr.chipsPerHand, rr.stderr, rr.bbPer100, rr.mbbPerGame);
    std::printf("  （对齐 ACPC：%d 筹码 = %dBB，永远弃牌 = %+0.1f mbb/g）\n",
                texas::kStartingStack, texas::kStartingStack / texas::kBigBlind,
                -0.5 * double(texas::kSmallBlind + texas::kBigBlind) /
                    double(texas::kBigBlind) * 1000.0);
    return 0;
}

// ---------------------------------------------------------------- slumbot 日志分析

namespace {

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool inQuote = false;
    for (char c : line) {
        if (c == '"') {
            inQuote = !inQuote;
            continue;
        }
        if (c == ',' && !inQuote) {
            out.push_back(cur);
            cur.clear();
            continue;
        }
        cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

std::vector<std::string> splitWs(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ' ' || c == '\t' || c == '\r') {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

}  // namespace

// 读 `slumbot --log` 的 CSV，逐手重放，在每个"我们的决策"处用**双方已知底牌**
// 算事后真实胜率，按街统计"高胜率弃牌 / 低胜率跟注"的 EV 损失。
// 这是漏洞定位工具：事后胜率不是策略的期望胜率，但系统性模式仍然有意义。
int cmdSlumbotAnalyze(int argc, char** argv) {
    const std::string path = argStr(argc, argv, "--csv", "");
    if (path.empty()) {
        std::printf("slumbot-analyze 需要 --csv FILE\n");
        return 1;
    }
    std::ifstream in(path);
    if (!in) {
        std::printf("cannot open %s\n", path.c_str());
        return 1;
    }
    const std::string dumpSeat = argStr(argc, argv, "--dump-seat", "");
    const int dumpPotLo = argInt(argc, argv, "--dump-pot-lo", 0);
    const int dumpPotHi = argInt(argc, argv, "--dump-pot-hi", 1 << 30);
    const int dumpLimit = argInt(argc, argv, "--dump-limit", 20);
    const std::string dumpBig = argStr(argc, argv, "--dump-big", "");  // push|calloff|fold
    int dumped = 0;

    struct StreetAgg {
        int decisions = 0, folds = 0, calls = 0, raises = 0;
        int badFolds = 0, badCalls = 0;
        double foldLoss = 0.0, callLoss = 0.0;
        double sumEqFold = 0.0, sumOddsFold = 0.0;
        double sumEqCall = 0.0, sumOddsCall = 0.0;
    };
    StreetAgg agg[5];
    // 按"决策时底池"分桶：外部日志显示 1500–4000 是 5σ 级漏洞，这里直接
    // 把同样的统计放到底池桶里，避免被街道混杂掩盖。
    StreetAgg byPot[5];
    StreetAgg byPotSeat[2][5];  // [我们的座位][底池桶]，座位 0 = SB、1 = BB
    const int potEdge[4] = {300, 600, 1500, 4000};
    auto potBucket = [&](int pot) {
        for (int i = 0; i < 4; ++i) {
            if (pot < potEdge[i]) return i;
        }
        return 4;
    };
    // 逐手结果：按 (座位, 终局底池桶, 是否摊牌) 统计调整后净收益。
    // 直接回答"这个桶的赤字是弃牌漏还是摊牌漏"。
    struct HandAgg {
        int n = 0;
        double sum = 0.0;
    };
    HandAgg handAgg[2][5][2];
    // Slumbot 的翻前应对（按它面对的加注次数分桶）：直接看它的弃牌率，
    // 决定我们的 3-bet / 4-bet 频率是否值得加码。
    struct PreAgg {
        int n = 0, folds = 0, calls = 0, raises = 0, allins = 0;
    };
    PreAgg preAgg[4];
    // 再加一层"面对的下注有多大"：直接看 Slumbot 的弃牌率是否随我们的
    // 加注尺寸变化（决策尺寸是对它最便宜的剥削杠杆）。
    PreAgg sizeAgg[3][4];
    // 后翻牌：对手面对我们最近一次下注/加注的反应，按"下注额/下注前底池"分桶。
    // 外部日志里只测过翻前尺寸，后翻牌的尺寸——弃牌率曲线一直是空白。
    PreAgg postSize[4][4];
    struct PendingBet {
        bool valid = false;
        int street = -1;
        int potBefore = 0;
        int amount = 0;
    } pendingBet;
    std::vector<std::string> decLog;  // 当前这手我们的决策描述（--dump 用）
    int lines = 0, replayed = 0, decisions = 0, handsSeen = 0;
    // 深筹码全下（>=90BB）由谁发起、结果如何：区分"我们推"和"我们接"。
    struct BigAgg {
        int n = 0;
        double sum = 0.0;
    };
    BigAgg bigPush[2], bigCallOff[2], bigFoldTo[2];  // [座位]
    // 我们 SB 的翻前开牌尺寸 vs 整手调整后收益：直接测"开多大更赚"。
    BigAgg openSize[4];
    BigAgg openLimp;
    std::string line;
    std::getline(in, line);  // header
    texas::Rng rng(7);
    while (std::getline(in, line)) {
        ++lines;
        const auto f = splitCsvLine(line);
        if (f.size() < 10 || f[6].empty()) continue;
        const int clientPos = std::atoi(f[1].c_str());
        const int mySeat = (clientPos == 0) ? 1 : 0;
        const auto hole = splitWs(f[7]);
        const auto bot = splitWs(f[8]);
        const auto board = splitWs(f[9]);
        if (hole.size() < 2 || bot.size() < 2) continue;
        texas::Card mine[2] = {texas::parseCard(hole[0].c_str()),
                               texas::parseCard(hole[1].c_str())};
        texas::Card theirs[2] = {texas::parseCard(bot[0].c_str()),
                                 texas::parseCard(bot[1].c_str())};
        texas::Card bd[5] = {0, 0, 0, 0, 0};
        bool ok = mine[0] >= 0 && mine[1] >= 0 && theirs[0] >= 0 && theirs[1] >= 0;
        for (size_t k = 0; ok && k < board.size() && k < 5; ++k) {
            bd[k] = texas::parseCard(board[k].c_str());
            ok = bd[k] >= 0;
        }
        if (!ok) continue;

        texas::HUState st = (clientPos == 0) ? texas::makeState(theirs, mine, bd, 0)
                                             : texas::makeState(mine, theirs, bd, 0);
        ++replayed;
        ++handsSeen;
        decLog.clear();
        int bigSeat = -1;
        int openBucket = -1;
        bool openLimpHand = false;
        int lastSeat = -1;
        bool lastWasFold = false;
        bool lastWasCall = false;
        const std::string& action = f[6];
        size_t i = 0;
        while (i < action.size()) {
            const char c = action[i++];
            if (c == '/') continue;
            texas::Move m{};
            if (c == 'k') {
                m.kind = texas::MK_CHECK;
                m.to = st.contrib[st.toAct];
            } else if (c == 'c') {
                m.kind = texas::MK_CALL;
                m.to = st.contrib[1 - st.toAct];
            } else if (c == 'f') {
                m.kind = texas::MK_FOLD;
                m.to = st.contrib[st.toAct];
            } else if (c == 'b') {
                int v = 0;
                const size_t start = i;
                while (i < action.size() && action[i] >= '0' && action[i] <= '9') {
                    v = v * 10 + (action[i] - '0');
                    ++i;
                }
                if (i == start) break;
                m.kind = texas::MK_RAISE;
                m.to = static_cast<int16_t>(v);
            } else {
                break;
            }

            if (st.toAct != mySeat && st.street == texas::ST_PREFLOP) {
                const int need = texas::toCall(st, st.toAct);
                if (need > 0) {
                    PreAgg& p = preAgg[std::min<int>(st.numRaises, 3)];
                    ++p.n;
                    {
                        const int cur = int(st.contrib[1 - st.toAct]);
                        const int potBefore = st.pot() - (cur - int(st.contrib[st.toAct]));
                        const double ratio =
                            (potBefore > 0) ? double(cur) / double(potBefore) : 0.0;
                        const int sb = (ratio < 0.8) ? 0 : (ratio < 1.2 ? 1 : (ratio < 1.8 ? 2 : 3));
                        PreAgg& q = sizeAgg[std::min<int>(st.numRaises, 2)][sb];
                        ++q.n;
                        if (m.kind == texas::MK_FOLD) ++q.folds;
                        if (m.kind == texas::MK_CHECK || m.kind == texas::MK_CALL) ++q.calls;
                        if (m.kind == texas::MK_RAISE || m.kind == texas::MK_ALLIN) ++q.raises;
                    }
                    if (m.kind == texas::MK_FOLD) {
                        ++p.folds;
                    } else if (m.kind == texas::MK_CHECK || m.kind == texas::MK_CALL) {
                        ++p.calls;
                    } else {
                        ++p.raises;
                    }
                    if (m.kind == texas::MK_ALLIN ||
                        (m.kind == texas::MK_RAISE &&
                         int(m.to) >= texas::maxRaiseTo(st, st.toAct))) {
                        ++p.allins;
                    }
                }
            }
            // 后翻牌尺寸反应：对手正面对我们上一笔下注时，先记账再清空。
            if (st.toAct != mySeat && pendingBet.valid &&
                pendingBet.street == st.street && pendingBet.amount > 0) {
                if (m.kind == texas::MK_FOLD || m.kind == texas::MK_CALL ||
                    m.kind == texas::MK_RAISE || m.kind == texas::MK_ALLIN) {
                    const double ratio =
                        (pendingBet.potBefore > 0)
                            ? double(pendingBet.amount) / double(pendingBet.potBefore)
                            : 0.0;
                    const int sb = (ratio < 0.5) ? 0 : (ratio < 0.8 ? 1 : (ratio < 1.3 ? 2 : 3));
                    PreAgg& q = postSize[st.street][sb];
                    ++q.n;
                    if (m.kind == texas::MK_FOLD) ++q.folds;
                    if (m.kind == texas::MK_CALL) ++q.calls;
                    if (m.kind == texas::MK_RAISE || m.kind == texas::MK_ALLIN) ++q.raises;
                }
                pendingBet.valid = false;
            }
            if (st.toAct == mySeat) {
                const int street = st.street;
                const int pot = st.pot();
                const int need = texas::toCall(st, mySeat);
                const double eq = texas::perfectEquity(st, mySeat, 160);
                const double odds = (need > 0) ? double(need) / double(pot + need) : 0.0;
                ++decisions;
                auto add = [&](StreetAgg& a) {
                    ++a.decisions;
                    if (m.kind == texas::MK_FOLD) {
                        ++a.folds;
                        a.sumEqFold += eq;
                        a.sumOddsFold += odds;
                        if (eq > odds + 0.10) {
                            ++a.badFolds;
                            a.foldLoss += eq * double(pot + need) - double(need);
                        }
                    } else if (m.kind == texas::MK_CALL || m.kind == texas::MK_CHECK) {
                        ++a.calls;
                        a.sumEqCall += eq;
                        a.sumOddsCall += odds;
                        if (m.kind == texas::MK_CALL && eq < odds - 0.10) {
                            ++a.badCalls;
                            a.callLoss += (odds - eq) * double(pot + need);
                        }
                    } else {
                        ++a.raises;
                    }
                };
                add(agg[street]);
                add(byPot[potBucket(pot)]);
                add(byPotSeat[mySeat][potBucket(pot)]);
                if (mySeat == 0 && street == 0 && st.numRaises == 0) {
                    if (m.kind == texas::MK_CALL) {
                        openLimpHand = true;
                    } else if (m.kind == texas::MK_RAISE) {
                        const int cur = int(m.to);
                        const double ratio = (pot > 0) ? double(cur) / double(pot) : 0.0;
                        const int sb = (ratio < 1.5) ? 0 : (ratio < 1.8 ? 1 : (ratio < 2.2 ? 2 : 3));
                        ++openSize[sb].n;
                        openBucket = sb;
                    }
                }
                if (!dumpSeat.empty()) {
                    char buf[160];
                    std::snprintf(buf, sizeof(buf),
                                  "    s%d pot%5d need%5d eq%.3f odds%.3f %s%s", street, pot, need,
                                  eq, odds, texas::moveToString(st, m).c_str(),
                                  m.kind == texas::MK_RAISE ? (" to " + std::to_string(m.to)).c_str()
                                                            : "");
                    decLog.emplace_back(buf);
                }
            }
            lastSeat = int(st.toAct);
            lastWasFold = (m.kind == texas::MK_FOLD);
            lastWasCall = (m.kind == texas::MK_CALL);
            if ((m.kind == texas::MK_RAISE || m.kind == texas::MK_ALLIN) && int(m.to) >= 9000) {
                bigSeat = lastSeat;
            }
            // 记录本街的最后一笔主动下注（对手下一动作就是它的反应）。
            if (st.toAct == mySeat && st.street > texas::ST_PREFLOP &&
                (m.kind == texas::MK_RAISE || m.kind == texas::MK_ALLIN)) {
                const int amount = int(m.to) - int(st.contrib[mySeat]);
                if (amount > 0) {
                    pendingBet.valid = true;
                    pendingBet.street = st.street;
                    pendingBet.potBefore = st.pot();
                    pendingBet.amount = amount;
                }
            }
            texas::applyMove(st, m, rng);
            if (st.finished) break;
        }
        if (!dumpSeat.empty() && dumped < dumpLimit) {
            const bool seatOk = (dumpSeat == "all") ||
                                (dumpSeat == "sb" && mySeat == 0) ||
                                (dumpSeat == "bb" && mySeat == 1);
            bool bigOk = true;
            if (dumpBig == "push") {
                bigOk = (bigSeat == mySeat);
            } else if (dumpBig == "calloff") {
                bigOk = (bigSeat >= 0 && bigSeat != mySeat && lastWasCall);
            } else if (dumpBig == "fold") {
                bigOk = (bigSeat >= 0 && bigSeat != mySeat && lastWasFold);
            }
            const int finalPot = std::atoi(f[4].c_str());
            if (seatOk && bigOk && finalPot >= dumpPotLo && finalPot <= dumpPotHi) {
                ++dumped;
                std::printf("hand#%d seat=%s finalPot=%d adj=%s | %s | hole %s | bot %s | board %s\n",
                            handsSeen, mySeat == 0 ? "SB" : "BB", finalPot, f[3].c_str(),
                            f[6].c_str(), f[7].c_str(), f[8].c_str(), f[9].c_str());
                for (const std::string& d : decLog) std::printf("%s\n", d.c_str());
            }
        }
        if (f.size() >= 5) {
            const int finalPot = std::atoi(f[4].c_str());
            const double adj = std::atof(f[3].c_str());
            if (openBucket >= 0) openSize[openBucket].sum += adj;
            if (openLimpHand) ++openLimp.n;
            if (openLimpHand) openLimp.sum += adj;
            if (bigSeat == mySeat) {
                BigAgg& a = bigPush[mySeat];
                ++a.n;
                a.sum += adj;
            } else if (bigSeat >= 0 && bigSeat != mySeat) {
                BigAgg& a = lastWasFold ? bigFoldTo[mySeat]
                                        : (lastWasCall ? bigCallOff[mySeat] : bigCallOff[mySeat]);
                ++a.n;
                a.sum += adj;
            }
            const bool showdown = !action.empty() && action.back() != 'f';
            HandAgg& h = handAgg[mySeat][potBucket(finalPot)][showdown ? 1 : 0];
            ++h.n;
            h.sum += adj;
        }
    }

    const char* names[5] = {"preflop", "flop", "turn", "river", "?"};
    std::printf("slumbot-analyze: %s | %d 行，重放 %d 手，%d 个决策\n", path.c_str(), lines,
                replayed, decisions);
    std::printf("  %-8s %6s %6s %6s %6s | %8s %8s | %6s %6s | %8s %8s\n", "street", "dec",
                "folds", "calls", "raises", "eq@fold", "odds@fold", "bFolds", "loss",
                "bCalls", "loss");
    double totalFoldLoss = 0.0, totalCallLoss = 0.0;
    int totalBadFolds = 0, totalBadCalls = 0;
    for (int s = 0; s < 4; ++s) {
        const StreetAgg& a = agg[s];
        if (a.decisions == 0) continue;
        const double eqF = a.folds ? a.sumEqFold / a.folds : 0.0;
        const double odF = a.folds ? a.sumOddsFold / a.folds : 0.0;
        std::printf("  %-8s %6d %6d %6d %6d | %8.3f %8.3f | %6d %8.0f | %6d %8.0f\n", names[s],
                    a.decisions, a.folds, a.calls, a.raises, eqF, odF, a.badFolds, a.foldLoss,
                    a.badCalls, a.callLoss);
        totalFoldLoss += a.foldLoss;
        totalCallLoss += a.callLoss;
        totalBadFolds += a.badFolds;
        totalBadCalls += a.badCalls;
    }
    std::printf("  合计：坏弃 %d 个（事后 EV 损失 %+0.0f 筹码），坏跟 %d 个（%+0.0f 筹码）\n",
                totalBadFolds, -totalFoldLoss, totalBadCalls, -totalCallLoss);

    // 同一批决策按底池分桶（决策时的底池，不是终局底池）。
    const char* potNames[5] = {"0-300", "300-600", "600-1500", "1500-4000", ">4000"};
    std::printf("  按底池分桶：\n");
    std::printf("  %-10s %6s %6s %6s %6s | %8s %8s | %6s %8s | %6s %8s\n", "pot", "dec", "folds",
                "calls", "raises", "eq@fold", "odds@fold", "bFolds", "loss", "bCalls", "loss");
    for (int seat = 0; seat < 2; ++seat) {
        std::printf("  [%s 座位]\n", seat == 0 ? "SB" : "BB");
        for (int b = 0; b < 5; ++b) {
            const StreetAgg& a = (seat == 0) ? byPotSeat[0][b] : byPotSeat[1][b];
            if (a.decisions == 0) continue;
            const double eqF = a.folds ? a.sumEqFold / a.folds : 0.0;
            const double odF = a.folds ? a.sumOddsFold / a.folds : 0.0;
            std::printf("  %-10s %6d %6d %6d %6d | %8.3f %8.3f | %6d %8.0f | %6d %8.0f\n",
                        potNames[b], a.decisions, a.folds, a.calls, a.raises, eqF, odF, a.badFolds,
                        -a.foldLoss, a.badCalls, -a.callLoss);
        }
    }
    (void)byPot;
    std::printf("  我们 SB 开牌尺寸 vs 整手收益（调整后）：\n");
    {
        const char* on[4] = {"<1.5", "1.5-1.8", "1.8-2.2", ">2.2"};
        for (int b = 0; b < 4; ++b) {
            if (openSize[b].n == 0) continue;
            std::printf("    %-8s n=%4d %+9.1f/手\n", on[b], openSize[b].n,
                        openSize[b].sum / openSize[b].n);
        }
        if (openLimp.n > 0) {
            std::printf("    limp     n=%4d %+9.1f/手\n", openLimp.n, openLimp.sum / openLimp.n);
        }
    }
    std::printf("  深全下（单笔 >=90BB）归属与结果（调整后）：\n");
    for (int seat = 0; seat < 2; ++seat) {
        const BigAgg& push = bigPush[seat];
        const BigAgg& calloff = bigCallOff[seat];
        const BigAgg& foldto = bigFoldTo[seat];
        std::printf("    %s: 我们推 n=%3d %+9.1f/手 | 我们接 n=%3d %+9.1f/手 | 我们弃 n=%3d %+9.1f/手\n",
                    seat == 0 ? "SB" : "BB", push.n, push.n ? push.sum / push.n : 0.0, calloff.n,
                    calloff.n ? calloff.sum / calloff.n : 0.0, foldto.n,
                    foldto.n ? foldto.sum / foldto.n : 0.0);
    }
    std::printf("  Slumbot 的翻前应对（面对第 N 次加注时）:\n");
    for (int r = 1; r < 4; ++r) {
        const PreAgg& p = preAgg[r];
        if (p.n == 0) continue;
        std::printf("    numRaises=%d: n=%4d | fold %5.1f%% | call %5.1f%% | raise %5.1f%%"
                    "（其中全下 %d）\n",
                    r, p.n, 100.0 * p.folds / p.n, 100.0 * p.calls / p.n, 100.0 * p.raises / p.n,
                    p.allins);
    }
    // 逐手结果表：同一底池桶里，弃牌结束的手 vs 摊牌的手各亏多少。
    const char* sizeNames[4] = {"<0.8", "0.8-1.2", "1.2-1.8", ">1.8"};
    std::printf("  Slumbot 弃牌率 vs 我们加注尺寸（加注额/加注前底池）：\n");
    for (int r = 1; r <= 2; ++r) {
        std::printf("    numRaises=%d:", r);
        for (int b = 0; b < 4; ++b) {
            const PreAgg& q = sizeAgg[r][b];
            if (q.n == 0) continue;
            std::printf("  %s n=%3d fold %5.1f%%", sizeNames[b], q.n, 100.0 * q.folds / q.n);
        }
        std::printf("\n");
    }
    const char* postSizeNames[4] = {"<0.5", "0.5-0.8", "0.8-1.3", ">1.3"};
    std::printf("  我们的后翻牌下注尺寸 vs Slumbot 反应（下注额/下注前底池）：\n");
    for (int stree = 1; stree <= 3; ++stree) {
        std::printf("    %s:", names[stree]);
        for (int b = 0; b < 4; ++b) {
            const PreAgg& q = postSize[stree][b];
            if (q.n == 0) continue;
            std::printf("  %s n=%4d fold %5.1f%% call %5.1f%% raise %5.1f%%", postSizeNames[b],
                        q.n, 100.0 * q.folds / q.n, 100.0 * q.calls / q.n,
                        100.0 * q.raises / q.n);
        }
        std::printf("\n");
    }
    std::printf("  逐手结果（调整后，按终局底池/座位/摊牌）：\n");
    std::printf("  %-4s %-10s %6s %8s %6s %8s\n", "seat", "pot", "hands", "非摊牌", "hands", "摊牌");
    for (int seat = 0; seat < 2; ++seat) {
        for (int b = 0; b < 5; ++b) {
            const HandAgg& noSd = handAgg[seat][b][0];
            const HandAgg& sd = handAgg[seat][b][1];
            if (noSd.n + sd.n == 0) continue;
            std::printf("  %-4s %-10s %6d %+8.1f %6d %+8.1f\n", seat == 0 ? "SB" : "BB",
                        potNames[b], noSd.n, noSd.n ? noSd.sum / noSd.n : 0.0, sd.n,
                        sd.n ? sd.sum / sd.n : 0.0);
        }
    }
    return 0;
}

// 全下决策审计：外部日志里 `allin_adjusted=1` 的手，按（座位 × 街 × 推/接）
// 汇总调整后 EV、投入和手牌百分位。把"赤字全部来自全下决策"从定性结论变成
// 可复现指标；训练改动（tight 对手）后可直接对比同一张表。
int cmdAllinAudit(int argc, char** argv) {
    const std::vector<std::string> csvs = argValues(argc, argv, "--csv");
    if (csvs.empty()) {
        std::printf("allin-audit 需要 --csv FILE（可重复）\n");
        return 1;
    }
    struct Agg {
        int n = 0;
        double sumAdj = 0.0;
        double sumInvested = 0.0;
        std::vector<float> pct;
    };
    const char* streetNames0[4] = {"preflop", "flop", "turn", "river"};
    // [座位][街][0 = 我们推, 1 = 我们接]
    Agg agg[2][4][2];
    int handsSeen = 0, allinHands = 0, replayed = 0, badReplay = 0;
    double totalAdj = 0.0;
    double allinAdj = 0.0;
    for (const std::string& path : csvs) {
        std::ifstream in(path);
        if (!in) {
            std::printf("打不开 %s\n", path.c_str());
            continue;
        }
        std::string line;
        std::getline(in, line);  // header
        texas::Rng rng(11);
        while (std::getline(in, line)) {
            const auto f = splitCsvLine(line);
            if (f.size() < 10 || f[6].empty()) continue;
            const int clientPos = std::atoi(f[1].c_str());
            const int mySeat = (clientPos == 0) ? 1 : 0;
            const auto hole = splitWs(f[7]);
            const auto bot = splitWs(f[8]);
            const auto boardS = splitWs(f[9]);
            if (hole.size() < 2 || bot.size() < 2) continue;
            texas::Card mine[2] = {texas::parseCard(hole[0].c_str()),
                                   texas::parseCard(hole[1].c_str())};
            texas::Card theirs[2] = {texas::parseCard(bot[0].c_str()),
                                     texas::parseCard(bot[1].c_str())};
            texas::Card bd[5] = {0, 0, 0, 0, 0};
            bool ok = mine[0] >= 0 && mine[1] >= 0 && theirs[0] >= 0 && theirs[1] >= 0;
            for (size_t k = 0; ok && k < boardS.size() && k < 5; ++k) {
                bd[k] = texas::parseCard(boardS[k].c_str());
                ok = bd[k] >= 0;
            }
            if (!ok) continue;
            ++replayed;
            ++handsSeen;
            totalAdj += std::atof(f[3].c_str());
            if (std::atoi(f[5].c_str()) == 0) continue;

            texas::HUState st = (clientPos == 0) ? texas::makeState(theirs, mine, bd, 0)
                                                 : texas::makeState(mine, theirs, bd, 0);
            const std::string& action = f[6];
            size_t i = 0;
            bool found = false;
            std::vector<std::string> path;  // 本手里我们每个决策点的状态（--path）
            while (i < action.size()) {
                const char c = action[i++];
                texas::Move m{};
                if (c == '/') continue;
                if (c == 'k') {
                    m.kind = texas::MK_CHECK;
                    m.to = st.contrib[st.toAct];
                } else if (c == 'c') {
                    m.kind = texas::MK_CALL;
                    m.to = st.contrib[1 - st.toAct];
                } else if (c == 'f') {
                    m.kind = texas::MK_FOLD;
                    m.to = st.contrib[st.toAct];
                } else if (c == 'b') {
                    int v = 0;
                    const size_t start = i;
                    while (i < action.size() && action[i] >= '0' && action[i] <= '9') {
                        v = v * 10 + (action[i] - '0');
                        ++i;
                    }
                    if (i == start) break;
                    m.kind = texas::MK_RAISE;
                    m.to = static_cast<int16_t>(v);
                } else {
                    break;
                }
                if (hasFlag(argc, argv, "--path") && st.toAct == mySeat) {
                    const int bN = (st.street == texas::ST_PREFLOP) ? 0 : int(st.street) + 2;
                    const float pp = texas::handStrength(mine, bd, bN);
                    char buf[160];
                    std::snprintf(buf, sizeof(buf), "%s pot%d need%d pct%.2f %s",
                                  streetNames0[st.street], st.pot(), texas::toCall(st, st.toAct), pp,
                                  texas::moveToString(st, m).c_str());
                    path.emplace_back(buf);
                }
                texas::applyMove(st, m, rng);
                if (st.allin[0] && st.allin[1] && st.street < texas::ST_RIVER) {
                    found = true;
                    break;
                }
                if (st.finished) break;
            }
            if (!found) {
                ++badReplay;
                continue;
            }
            ++allinHands;
            allinAdj += std::atof(f[3].c_str());
            // 双方全下时最后一步一定是跟注：跟注方是我们 → 我们接，否则我们推。
            const int lastSeat = int(st.hist[st.histLen - 1].seat);
            const int mode = (lastSeat == mySeat) ? 1 : 0;
            const int street = int(st.street);
            const int boardN = (street == texas::ST_PREFLOP) ? 0 : street + 2;
            Agg& a = agg[mySeat][street][mode];
            ++a.n;
            const double adj = std::atof(f[3].c_str());
            a.sumAdj += adj;
            a.sumInvested += double(st.total[mySeat]);
            const float pct = texas::handStrength(mine, bd, boardN);
            a.pct.push_back(pct);
            int cards[7];
            int nc = 0;
            cards[nc++] = mine[0];
            cards[nc++] = mine[1];
            for (int bc = 0; bc < boardN; ++bc) cards[nc++] = bd[bc];
            const int cat = texas::handCategoryOf(texas::evaluateBest(cards, nc));
            if (hasFlag(argc, argv, "--path")) {
                std::printf("  hand %s vs %s board %s | adj %+.0f | %s\n", f[7].c_str(),
                            f[8].c_str(), f[9].c_str(), adj, f[6].c_str());
                for (const std::string& ln : path) std::printf("      %s\n", ln.c_str());
            }
            if (hasFlag(argc, argv, "--detail")) {
                std::printf(
                    "    seat=%s street=%-7s mode=%s adj=%+8.0f invested=%6d pct=%.2f cat=%d %s %s\n",
                    mySeat == 0 ? "SB" : "BB", streetNames0[street],
                    mode == 1 ? "接" : "推", adj, int(st.total[mySeat]), pct, cat, f[7].c_str(),
                    f[9].c_str());
            }
        }
    }

    const char* streetNames[4] = {"preflop", "flop", "turn", "river"};
    const char* modeNames[2] = {"我们推", "我们接"};
    std::printf("allin-audit: %d 个文件, 重放 %d 手, 全下手 %d 手（重放失败 %d）\n",
                int(csvs.size()), replayed, allinHands, badReplay);
    std::printf("  %-4s %-8s %-8s %5s %10s %9s %7s %7s\n", "seat", "street", "mode", "n",
                "调整EV/手", "投入/手", "pct中位", "pct均值");
    for (int seat = 0; seat < 2; ++seat) {
        for (int street = 0; street < 4; ++street) {
            for (int mode = 0; mode < 2; ++mode) {
                Agg& a = agg[seat][street][mode];
                if (a.n == 0) continue;
                std::sort(a.pct.begin(), a.pct.end());
                const float med = a.pct[a.pct.size() / 2];
                double sumPct = 0.0;
                for (float v : a.pct) sumPct += v;
                std::printf("  %-4s %-8s %-8s %5d %+10.1f %9.1f %7.2f %7.2f\n",
                            seat == 0 ? "SB" : "BB", streetNames[street], modeNames[mode], a.n,
                            a.sumAdj / a.n, a.sumInvested / a.n, med, sumPct / a.n);
            }
        }
    }
    if (handsSeen > 0) {
        const double allinPerHand = allinAdj / handsSeen / 100.0;
        const double restPerHand = (totalAdj - allinAdj) / handsSeen / 100.0;
        std::printf(
            "  全下手贡献 %.2f BB/手 | 非全下手 %.2f BB/手 | 合计 %.2f BB/手（%d 手）\n",
            allinPerHand, restPerHand, totalAdj / handsSeen / 100.0, handsSeen);
    }
    return 0;
}

// 合并多段外部日志的裁决读数：与 slumbot 运行结束同公式（总体方差），
// 只读 winnings/adjusted 两列；裁决只看调整后的 BB/手 ± SE。
int cmdSlumbotMerge(int argc, char** argv) {
    const std::vector<std::string> csvs = argValues(argc, argv, "--csv");
    if (csvs.empty()) {
        std::printf("slumbot-merge 需要 --csv FILE（可重复）\n");
        return 1;
    }
    struct Agg {
        long long n = 0;
        double sumW = 0.0, sumWSq = 0.0, sumA = 0.0, sumASq = 0.0;
        void add(double w, double a) {
            ++n; sumW += w; sumWSq += w * w; sumA += a; sumASq += a * a;
        }
        void print(const char* label) const {
            if (n < 2) { std::printf("  %s: %lld 手（不足）\n", label, n); return; }
            const double d = double(n);
            const double mw = sumW / d, ma = sumA / d;
            const double sew = std::sqrt(std::max(sumWSq / d - mw * mw, 0.0) / d);
            const double sea = std::sqrt(std::max(sumASq / d - ma * ma, 0.0) / d);
            std::printf("  %s: %lld 手 | 原始 %+.2f ± %.2f 筹码/手 | 调整后 %+.2f ± %.2f 筹码/手"
                        " = %+.3f ± %.3f BB/手\n",
                        label, n, mw, sew, ma, sea, ma / 100.0, sea / 100.0);
        }
    };
    Agg tot;
    for (const std::string& path : csvs) {
        std::ifstream in(path);
        if (!in) {
            std::printf("打不开 %s\n", path.c_str());
            return 1;
        }
        std::string line;
        std::getline(in, line);  // header
        if (line.rfind("hand,client_pos,winnings,adjusted,", 0) != 0) {
            std::printf("%s: 表头不是 slumbot 日志格式\n", path.c_str());
            return 1;
        }
        Agg a;
        while (std::getline(in, line)) {
            const auto f = splitCsvLine(line);
            if (f.size() < 10) continue;
            a.add(std::atof(f[2].c_str()), std::atof(f[3].c_str()));
        }
        a.print(path.c_str());
        tot.n += a.n; tot.sumW += a.sumW; tot.sumWSq += a.sumWSq;
        tot.sumA += a.sumA; tot.sumASq += a.sumASq;
    }
    tot.print("合并");
    return 0;
}

// 承诺审计：外部日志里"我们投了一大笔之后弃牌"的手，按投入深度和弃牌街分组。
// #111(c) 显示 rng1 的赤字 62% 来自这类手（投 76BB 再弃），但当时是一次性脚本；
// 这个子命令把同一测量变成可复现指标，训练/部署改动前后可直接对比。
int cmdCommitAudit(int argc, char** argv) {
    const std::vector<std::string> csvs = argValues(argc, argv, "--csv");
    if (csvs.empty()) {
        std::printf("commit-audit 需要 --csv FILE（可重复）\n");
        return 1;
    }
    const int kNumBuckets = 5;
    const int bucketLoBB[kNumBuckets] = {0, 10, 30, 60, 100};
    const char* bucketNames[kNumBuckets] = {"<10BB", "10-30BB", "30-60BB", "60-100BB", ">=100BB"};
    struct Agg {
        int n = 0;
        double sumAdj = 0.0;
        double sumSq = 0.0;
        double sumInvested = 0.0;
    };
    Agg bucket[kNumBuckets];
    Agg foldStreet[4];
    Agg deepByStreet[4];  // 投入 >=30BB 且在该街弃牌
    Agg totalSeat[2];     // 全部手（不只弃牌）按座位，用于算调整后 EV ± SE
    Agg preRaise[4];      // 我方翻前最高加注层级：0 无 / 1 open / 2 3-bet / 3 4-bet / 4 5-bet+
    const char* preRaiseNames[5] = {"无加注", "open", "3-bet", "4-bet", "5-bet+"};
    // 翻前首轮响应：SB 开牌（弃/limp/加注）与 BB 面对开牌（弃/跟/3-bet）。
    Agg sbOpen[3];
    Agg bbVsOpen[3];
    const char* sbOpenNames[3] = {"弃牌", "limp", "加注"};
    const char* bbRespNames[3] = {"弃牌", "跟注", "3-bet"};
    Agg bb3betPct[4];  // BB 3-bet 按翻前手牌百分位分桶
    const char* bb3betPctNames[4] = {"<0.45", "0.45-0.60", "0.60-0.75", ">=0.75"};
    Agg sbOpenPct[4];  // SB 开牌加注按翻前手牌百分位分桶
    const char* sbOpenPctNames[4] = {"<0.30", "0.30-0.50", "0.50-0.70", ">=0.70"};
    Agg bbFoldPct[4];  // BB 面对开牌弃牌的手牌百分位分布（判断是否过度弃牌）
    const char* bbFoldPctNames[4] = {"<0.40", "0.40-0.55", "0.55-0.70", ">=0.70"};
    // 终局底池 × 摊牌/非摊牌（与 README #111(d) 同口径，底池单位筹码）
    Agg termSd[3], termNoSd[3];
    const char* potBucketNames[3] = {"<1500", "1500-4000", ">=4000"};
    int handsSeen = 0, foldedHands = 0, badReplay = 0;
    double totalAdj = 0.0, totalSq = 0.0;
    for (const std::string& path : csvs) {
        std::ifstream in(path);
        if (!in) {
            std::printf("打不开 %s\n", path.c_str());
            continue;
        }
        std::string line;
        std::getline(in, line);  // header
        texas::Rng rng(11);
        while (std::getline(in, line)) {
            const auto f = splitCsvLine(line);
            if (f.size() < 10 || f[6].empty()) continue;
            const int clientPos = std::atoi(f[1].c_str());
            const int mySeat = (clientPos == 0) ? 1 : 0;
            const auto hole = splitWs(f[7]);
            const auto bot = splitWs(f[8]);
            const auto boardS = splitWs(f[9]);
            if (hole.size() < 2 || bot.size() < 2) continue;
            texas::Card mine[2] = {texas::parseCard(hole[0].c_str()),
                                   texas::parseCard(hole[1].c_str())};
            texas::Card theirs[2] = {texas::parseCard(bot[0].c_str()),
                                     texas::parseCard(bot[1].c_str())};
            texas::Card bd[5] = {0, 0, 0, 0, 0};
            bool ok = mine[0] >= 0 && mine[1] >= 0 && theirs[0] >= 0 && theirs[1] >= 0;
            for (size_t k = 0; ok && k < boardS.size() && k < 5; ++k) {
                bd[k] = texas::parseCard(boardS[k].c_str());
                ok = bd[k] >= 0;
            }
            if (!ok) continue;
            ++handsSeen;
            const double adj = std::atof(f[3].c_str());
            totalAdj += adj;
            totalSq += adj * adj;
            {
                Agg& ts = totalSeat[mySeat];
                ++ts.n;
                ts.sumAdj += adj;
                ts.sumSq += adj * adj;
            }

            texas::HUState st = (clientPos == 0) ? texas::makeState(theirs, mine, bd, 0)
                                                 : texas::makeState(mine, theirs, bd, 0);
            const std::string& action = f[6];
            size_t i = 0;
            int foldedStreet = -1;
            int preLevel = 0;  // 0 = 没加注；1..4 = open / 3-bet / 4-bet / 5-bet+
            int sbOpenAct = -1, bbRespAct = -1;
            while (i < action.size() && !st.finished) {
                const char c = action[i++];
                texas::Move m{};
                if (c == '/') continue;
                if (c == 'k') {
                    m.kind = texas::MK_CHECK;
                    m.to = st.contrib[st.toAct];
                } else if (c == 'c') {
                    m.kind = texas::MK_CALL;
                    m.to = st.contrib[1 - st.toAct];
                } else if (c == 'f') {
                    m.kind = texas::MK_FOLD;
                    m.to = st.contrib[st.toAct];
                } else if (c == 'b') {
                    int v = 0;
                    const size_t start = i;
                    while (i < action.size() && action[i] >= '0' && action[i] <= '9') {
                        v = v * 10 + (action[i] - '0');
                        ++i;
                    }
                    if (i == start) break;
                    m.kind = texas::MK_RAISE;
                    m.to = static_cast<int16_t>(v);
                } else {
                    break;
                }
                if (st.toAct == mySeat) {
                    if (m.kind == texas::MK_FOLD) foldedStreet = int(st.street);
                    if (st.street == texas::ST_PREFLOP) {
                        if (mySeat == 0 && sbOpenAct < 0 && st.numRaises == 0 &&
                            texas::toCall(st, mySeat) > 0) {
                            sbOpenAct = (m.kind == texas::MK_FOLD)
                                            ? 0
                                            : (m.kind == texas::MK_CALL ? 1 : 2);
                        }
                        if (mySeat == 1 && bbRespAct < 0 && st.numRaises >= 1 &&
                            texas::toCall(st, mySeat) > 0) {
                            bbRespAct = (m.kind == texas::MK_FOLD)
                                            ? 0
                                            : (m.kind == texas::MK_CALL ? 1 : 2);
                        }
                        if ((m.kind == texas::MK_RAISE || m.kind == texas::MK_ALLIN)) {
                            const int lv = std::min(4, int(st.numRaises) + 1);
                            if (lv > preLevel) preLevel = lv;
                        }
                    }
                }
                texas::applyMove(st, m, rng);
            }
            {
                Agg& pr = preRaise[preLevel];
                ++pr.n;
                pr.sumAdj += adj;
                pr.sumSq += adj * adj;
            }
            if (sbOpenAct >= 0) {
                Agg& so = sbOpen[sbOpenAct];
                ++so.n;
                so.sumAdj += adj;
                so.sumSq += adj * adj;
                if (sbOpenAct == 2) {
                    const float pp = texas::handStrength(mine, bd, 0);
                    const int pb = pp < 0.30f ? 0 : (pp < 0.50f ? 1 : (pp < 0.70f ? 2 : 3));
                    Agg& so2 = sbOpenPct[pb];
                    ++so2.n;
                    so2.sumAdj += adj;
                    so2.sumSq += adj * adj;
                }
            }
            if (st.finished) {
                const int finalPot = st.pot();
                const int pbi = finalPot < 1500 ? 0 : (finalPot < 4000 ? 1 : 2);
                Agg& ta = (st.folded == 255) ? termSd[pbi] : termNoSd[pbi];
                ++ta.n;
                ta.sumAdj += adj;
                ta.sumSq += adj * adj;
            }
            if (bbRespAct >= 0) {
                Agg& br = bbVsOpen[bbRespAct];
                ++br.n;
                br.sumAdj += adj;
                br.sumSq += adj * adj;
                if (bbRespAct == 2) {
                    const float pp = texas::handStrength(mine, bd, 0);
                    const int pb = pp < 0.45f ? 0 : (pp < 0.60f ? 1 : (pp < 0.75f ? 2 : 3));
                    Agg& b3 = bb3betPct[pb];
                    ++b3.n;
                    b3.sumAdj += adj;
                    b3.sumSq += adj * adj;
                } else if (bbRespAct == 0) {
                    const float pp = texas::handStrength(mine, bd, 0);
                    const int pb = pp < 0.40f ? 0 : (pp < 0.55f ? 1 : (pp < 0.70f ? 2 : 3));
                    Agg& bf = bbFoldPct[pb];
                    ++bf.n;
                    bf.sumAdj += adj;
                    bf.sumSq += adj * adj;
                }
            }
            if (st.folded != mySeat) continue;
            if (foldedStreet < 0) {
                ++badReplay;
                continue;
            }
            ++foldedHands;
            const int invested = int(st.total[mySeat]);
            const double investedBB = double(invested) / 100.0;
            int bi = 0;
            for (int b = kNumBuckets - 1; b >= 0; --b) {
                if (investedBB >= bucketLoBB[b]) {
                    bi = b;
                    break;
                }
            }
            Agg& a = bucket[bi];
            ++a.n;
            a.sumAdj += adj;
            a.sumSq += adj * adj;
            a.sumInvested += invested;
            Agg& fs = foldStreet[foldedStreet];
            ++fs.n;
            fs.sumAdj += adj;
            fs.sumSq += adj * adj;
            fs.sumInvested += invested;
            if (investedBB >= 30.0) {
                Agg& ds = deepByStreet[foldedStreet];
                ++ds.n;
                ds.sumAdj += adj;
                ds.sumSq += adj * adj;
                ds.sumInvested += invested;
            }
        }
    }
    const char* streetNames[4] = {"preflop", "flop", "turn", "river"};
    std::printf("commit-audit: %d 个文件, 重放 %d 手（重放失败 %d）, 我们弃牌 %d 手\n",
                int(csvs.size()), handsSeen, badReplay, foldedHands);
    std::printf("  %-10s %5s %7s %10s %10s %10s\n", "投入桶", "n", "占比", "投入/手",
                "调整EV/手", "贡献BB/手");
    for (int b = 0; b < kNumBuckets; ++b) {
        const Agg& a = bucket[b];
        if (a.n == 0) continue;
        std::printf("  %-10s %5d %6.2f%% %10.1f %+10.1f %+10.2f\n", bucketNames[b], a.n,
                    handsSeen ? 100.0 * a.n / handsSeen : 0.0, a.sumInvested / a.n,
                    a.sumAdj / a.n, handsSeen ? a.sumAdj / handsSeen / 100.0 : 0.0);
    }
    std::printf("  %-10s %5s %7s %10s %10s %10s\n", "弃牌街", "n", "占比", "投入/手",
                "调整EV/手", "贡献BB/手");
    for (int s = 0; s < 4; ++s) {
        const Agg& a = foldStreet[s];
        if (a.n == 0) continue;
        std::printf("  %-10s %5d %6.2f%% %10.1f %+10.1f %+10.2f\n", streetNames[s], a.n,
                    handsSeen ? 100.0 * a.n / handsSeen : 0.0, a.sumInvested / a.n,
                    a.sumAdj / a.n, handsSeen ? a.sumAdj / handsSeen / 100.0 : 0.0);
    }
    Agg deep;
    for (int s = 0; s < 4; ++s) {
        deep.n += deepByStreet[s].n;
        deep.sumAdj += deepByStreet[s].sumAdj;
        deep.sumInvested += deepByStreet[s].sumInvested;
    }
    if (handsSeen > 0) {
        std::printf("  投入>=30BB 后弃牌: %d 手（%.2f%%）, 投入/手 %.1f, 调整EV/手 %+.1f, "
                    "贡献 %+.2f BB/手\n",
                    deep.n, 100.0 * deep.n / handsSeen,
                    deep.n ? deep.sumInvested / deep.n : 0.0,
                    deep.n ? deep.sumAdj / deep.n : 0.0, deep.sumAdj / handsSeen / 100.0);
        const double mean = totalAdj / handsSeen;
        double se = 0.0;
        if (handsSeen > 1) {
            const double var = (totalSq - handsSeen * mean * mean) / (handsSeen - 1);
            se = var > 0.0 ? std::sqrt(var / handsSeen) : 0.0;
        }
        std::printf("  合计 %+.2f ± %.2f BB/手（%d 手）| %+.1f ± %.1f 筹码/手\n",
                    mean / 100.0, se / 100.0, handsSeen, mean, se);
        std::printf("  %-3s %6s %11s %13s\n", "座位", "手数", "筹码/手", "BB/手");
        for (int s = 0; s < 2; ++s) {
            const Agg& ts = totalSeat[s];
            if (ts.n == 0) continue;
            const double m = ts.sumAdj / ts.n;
            double e = 0.0;
            if (ts.n > 1) {
                const double v = (ts.sumSq - ts.n * m * m) / (ts.n - 1);
                e = v > 0.0 ? std::sqrt(v / ts.n) : 0.0;
            }
            std::printf("  %-3s %6d %+11.1f %+7.2f ± %.2f\n", s == 0 ? "SB" : "BB", ts.n, m,
                        m / 100.0, e / 100.0);
        }
        std::printf("  %-7s %6s %7s %11s %13s\n", "翻前最高", "手数", "占比", "筹码/手", "BB/手");
        for (int lv = 0; lv <= 4; ++lv) {
            const Agg& pr = preRaise[lv];
            if (pr.n == 0) continue;
            const double m = pr.sumAdj / pr.n;
            double e = 0.0;
            if (pr.n > 1) {
                const double v = (pr.sumSq - pr.n * m * m) / (pr.n - 1);
                e = v > 0.0 ? std::sqrt(v / pr.n) : 0.0;
            }
            std::printf("  %-7s %6d %6.2f%% %+11.1f %+7.2f ± %.2f\n", preRaiseNames[lv], pr.n,
                        100.0 * pr.n / handsSeen, m, m / 100.0, e / 100.0);
        }
        auto printRow = [&](const char* label, const Agg& a) {
            if (a.n == 0) return;
            const double m = a.sumAdj / a.n;
            double e = 0.0;
            if (a.n > 1) {
                const double v = (a.sumSq - a.n * m * m) / (a.n - 1);
                e = v > 0.0 ? std::sqrt(v / a.n) : 0.0;
            }
            std::printf("  %-9s %6d %6.2f%% %+11.1f %+7.2f ± %.2f\n", label, a.n,
                        100.0 * a.n / handsSeen, m, m / 100.0, e / 100.0);
        };
        std::printf("  %-9s %6s %7s %11s %13s\n", "SB 首轮", "手数", "占比", "筹码/手",
                    "BB/手");
        for (int k = 0; k < 3; ++k) printRow(sbOpenNames[k], sbOpen[k]);
        if (sbOpen[2].n > 0) {
            std::printf("  %-9s %6s %7s %11s %13s\n", "SB 开牌", "手数", "占开牌",
                        "筹码/手", "BB/手");
            for (int k = 0; k < 4; ++k) {
                const Agg& a = sbOpenPct[k];
                if (a.n == 0) continue;
                const double m = a.sumAdj / a.n;
                double e = 0.0;
                if (a.n > 1) {
                    const double v = (a.sumSq - a.n * m * m) / (a.n - 1);
                    e = v > 0.0 ? std::sqrt(v / a.n) : 0.0;
                }
                std::printf("  %-9s %6d %6.2f%% %+11.1f %+7.2f ± %.2f\n", sbOpenPctNames[k],
                            a.n, 100.0 * a.n / sbOpen[2].n, m, m / 100.0, e / 100.0);
            }
        }
        std::printf("  %-9s %6s %7s %11s %13s\n", "BB 对开牌", "手数", "占比", "筹码/手",
                    "BB/手");
        for (int k = 0; k < 3; ++k) printRow(bbRespNames[k], bbVsOpen[k]);
        if (bbVsOpen[0].n > 0) {
            std::printf("  %-9s %6s %7s %11s %13s\n", "BB 弃牌", "手数", "占弃牌",
                        "筹码/手", "BB/手");
            for (int k = 0; k < 4; ++k) {
                const Agg& a = bbFoldPct[k];
                if (a.n == 0) continue;
                std::printf("  %-9s %6d %6.2f%% %+11.1f %+7.2f\n", bbFoldPctNames[k], a.n,
                            100.0 * a.n / bbVsOpen[0].n, a.sumAdj / a.n, -1.0);
            }
        }
        std::printf("  %-10s %6s %12s %6s %12s\n", "终局底池", "摊牌n", "摊牌筹码/手",
                    "非摊牌n", "非摊牌筹码/手");
        for (int b = 0; b < 3; ++b) {
            const Agg& sdA = termSd[b];
            const Agg& nsA = termNoSd[b];
            std::printf("  %-10s %6d %+12.1f %6d %+12.1f\n", potBucketNames[b], sdA.n,
                        sdA.n ? sdA.sumAdj / sdA.n : 0.0, nsA.n,
                        nsA.n ? nsA.sumAdj / nsA.n : 0.0);
        }
        if (bbVsOpen[2].n > 0) {
            std::printf("  %-9s %6s %7s %11s %13s\n", "BB 3-bet", "手数", "占3bet",
                        "筹码/手", "BB/手");
            for (int k = 0; k < 4; ++k) {
                const Agg& a = bb3betPct[k];
                if (a.n == 0) continue;
                const double m = a.sumAdj / a.n;
                double e = 0.0;
                if (a.n > 1) {
                    const double v = (a.sumSq - a.n * m * m) / (a.n - 1);
                    e = v > 0.0 ? std::sqrt(v / a.n) : 0.0;
                }
                std::printf("  %-9s %6d %6.2f%% %+11.1f %+7.2f ± %.2f\n", bb3betPctNames[k],
                            a.n, 100.0 * a.n / bbVsOpen[2].n, m, m / 100.0, e / 100.0);
            }
        }
    }
    return 0;
}

// 导出 Slumbot 的逐决策样本：每个"对手决策点"一行，带它当时的真实手牌百分位。
// 用途：拟合"动作 → 手牌强度分布"的经验范围模型（rangeEq 现在用我们自己的
// 策略当对手模型，这是它对外失真的根源之一）。输出 CSV 到 stdout。
int cmdSlumbotDecisions(int argc, char** argv) {
    const std::vector<std::string> csvs = argValues(argc, argv, "--csv");
    if (csvs.empty()) {
        std::printf("slumbot-decisions 需要 --csv FILE（可重复）\n");
        return 1;
    }
    std::printf("hand,seat,street,pot,to_call,num_raises,act,to,pct,board_n\n");
    int rows = 0, hands = 0, badReplay = 0;
    for (const std::string& path : csvs) {
        std::ifstream in(path);
        if (!in) {
            std::printf("打不开 %s\n", path.c_str());
            continue;
        }
        std::string line;
        std::getline(in, line);
        texas::Rng rng(11);
        while (std::getline(in, line)) {
            const auto f = splitCsvLine(line);
            if (f.size() < 10 || f[6].empty()) continue;
            const int clientPos = std::atoi(f[1].c_str());
            const int mySeat = (clientPos == 0) ? 1 : 0;
            const int oppSeat = 1 - mySeat;
            const auto hole = splitWs(f[7]);
            const auto bot = splitWs(f[8]);
            const auto boardS = splitWs(f[9]);
            if (hole.size() < 2 || bot.size() < 2) continue;
            texas::Card mine[2] = {texas::parseCard(hole[0].c_str()),
                                   texas::parseCard(hole[1].c_str())};
            texas::Card theirs[2] = {texas::parseCard(bot[0].c_str()),
                                     texas::parseCard(bot[1].c_str())};
            texas::Card bd[5] = {0, 0, 0, 0, 0};
            bool ok = mine[0] >= 0 && mine[1] >= 0 && theirs[0] >= 0 && theirs[1] >= 0;
            for (size_t k = 0; ok && k < boardS.size() && k < 5; ++k) {
                bd[k] = texas::parseCard(boardS[k].c_str());
                ok = bd[k] >= 0;
            }
            if (!ok) continue;
            ++hands;
            texas::HUState st = (clientPos == 0) ? texas::makeState(theirs, mine, bd, 0)
                                                 : texas::makeState(mine, theirs, bd, 0);
            const std::string& action = f[6];
            size_t i = 0;
            bool broke = false;
            while (i < action.size() && !st.finished) {
                const char c = action[i++];
                texas::Move m{};
                if (c == '/') continue;
                if (c == 'k') {
                    m.kind = texas::MK_CHECK;
                    m.to = st.contrib[st.toAct];
                } else if (c == 'c') {
                    m.kind = texas::MK_CALL;
                    m.to = st.contrib[1 - st.toAct];
                } else if (c == 'f') {
                    m.kind = texas::MK_FOLD;
                    m.to = st.contrib[st.toAct];
                } else if (c == 'b') {
                    int v = 0;
                    const size_t start = i;
                    while (i < action.size() && action[i] >= '0' && action[i] <= '9') {
                        v = v * 10 + (action[i] - '0');
                        ++i;
                    }
                    if (i == start) {
                        broke = true;
                        break;
                    }
                    m.kind = texas::MK_RAISE;
                    m.to = static_cast<int16_t>(v);
                } else {
                    broke = true;
                    break;
                }
                if (st.toAct == oppSeat) {
                    const int boardN = (st.street == texas::ST_PREFLOP) ? 0 : int(st.street) + 2;
                    const float pct = texas::handStrength(theirs, bd, boardN);
                    const int need = texas::toCall(st, oppSeat);
                    const char kind = m.kind == texas::MK_FOLD   ? 'f'
                                      : m.kind == texas::MK_CALL ? 'c'
                                      : m.kind == texas::MK_CHECK ? 'k'
                                                                  : 'b';
                    std::printf("%s,%d,%d,%d,%d,%d,%c,%d,%.4f,%d\n", f[0].c_str(), oppSeat,
                                int(st.street), st.pot(), need, int(st.numRaises), kind,
                                int(m.to), pct, boardN);
                    ++rows;
                }
                texas::applyMove(st, m, rng);
            }
            if (broke) ++badReplay;
        }
    }
    std::fprintf(stderr, "slumbot-decisions: %d 手，%d 行，重放失败 %d\n", hands, rows,
                 badReplay);
    return 0;
}

// ---------------------------------------------------------------- 快速环：外部状态决策评估

// 在**外部真实牌局状态**上给策略打分。外部亏损集中在转/河牌的跟注-弃牌
// （README 按街分解：两座位在翻前/翻牌都是正收益，亏损全在到河牌的摊牌），
// 所以这里把 Slumbot 日志里的每一个"我们的决策"原样回放：状态固定（同一批
// 手牌、同一副公共牌、同一个对手底牌），用枚举剩余公共牌算精确胜率，得到
// "弃牌 / 跟注"两个动作的事后 EV，再看策略在这两个动作上的概率质量。
//
// 指标 = 策略相对事后最优的可回收 EV 损失（筹码/手，除以 100 即 BB/手）。
// 事后最优用到了对手底牌，绝对数值是上界，只能在**同一批固定状态**上做 A/B。
// 优点：完全本地、秒级、无发牌方差（状态固定，输出对权重是确定性的）。
int cmdSlumbotPolicyEval(int argc, char** argv) {
    const std::vector<std::string> csvs = argValues(argc, argv, "--csv");
    const std::string dir = argStr(argc, argv, "--resume", "");
    const int maxHands = argInt(argc, argv, "--hands", 0);
    const float temp = static_cast<float>(argDouble(argc, argv, "--temp", 0.5));
    if (csvs.empty() || dir.empty()) {
        std::printf("slumbot-policy-eval 需要 --resume DIR --csv FILE（可重复）\n");
        return 1;
    }
    texas::nn::ModelConfig mc;
    mc.hidden = argInt(argc, argv, "--hidden", 256);
    mc.lstmHidden = argInt(argc, argv, "--lstm", 128);
    texas::nn::Actor actors[2];
    for (int s = 0; s < 2; ++s) {
        actors[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
    }
    if (!loadActors(actors, dir)) {
        std::printf("cannot load actors from %s\n", dir.c_str());
        return 1;
    }
    for (int s = 0; s < 2; ++s) actors[s]->eval();

    // 某个固定局面上的策略概率。与 policyMove 同一条前向路径（T 温度、rangeEq
    // 取部署默认 0.5）。
    auto policyProbs = [&](const texas::HUState& st, const texas::LegalMove* lm, int n,
                           float* probs) {
        const int seat = st.toAct;
        torch::NoGradGuard ng;
        torch::Tensor nodeT = torch::zeros({1, texas::kNodeInput}, torch::kFloat32);
        torch::Tensor seqT =
            torch::zeros({1, texas::kHistSteps, texas::kHistDim}, torch::kFloat32);
        torch::Tensor featT = torch::zeros({n, texas::kActionInput}, torch::kFloat32);
        torch::Tensor sampleT = torch::zeros({n}, torch::kLong);
        torch::Tensor idT = torch::zeros({n}, torch::kLong);
        texas::Obs obs;
        texas::encodeObservation(st, obs, 24);
        std::copy(obs.node.begin(), obs.node.end(), nodeT.data_ptr<float>());
        std::copy(obs.seq.begin(), obs.seq.end(), seqT.data_ptr<float>());
        for (int i = 0; i < n; ++i) {
            std::array<float, texas::kActionInput> af;
            texas::encodeActionFeature(st, lm[i], af);
            std::copy(af.begin(), af.end(), featT.data_ptr<float>() + i * texas::kActionInput);
            idT.data_ptr<int64_t>()[i] = lm[i].slot;
        }
        torch::Tensor logits = actors[seat]->forward(nodeT, seqT, featT, sampleT, idT);
        const float* lp = logits.data_ptr<float>();
        float scaled[texas::kNumActions];
        for (int i = 0; i < texas::kNumActions; ++i) {
            scaled[i] = (temp == 1.0f) ? lp[i] : lp[i] / temp;
        }
        texas::ppo::maskedSoftmax(scaled, texas::kNumActions, probs);
        return true;
    };

    // 双方底牌已知、公共牌已发 boardN 张时的精确摊牌胜率（枚举剩余公共牌）。
    auto exactEquity = [](const texas::Card* mine, const texas::Card* theirs,
                          const texas::Card* board, int boardN) {
        bool used[texas::kNumCards] = {};
        used[mine[0]] = used[mine[1]] = used[theirs[0]] = used[theirs[1]] = true;
        texas::Card known[5];
        for (int k = 0; k < boardN; ++k) {
            known[k] = board[k];
            used[board[k]] = true;
        }
        texas::Card deck[texas::kNumCards];
        int nDeck = 0;
        for (int c = 0; c < texas::kNumCards; ++c) {
            if (!used[c]) deck[nDeck++] = static_cast<texas::Card>(c);
        }
        double win = 0.0, tie = 0.0, total = 0.0;
        auto evalOne = [&](const texas::Card* run, int nRun) {
            texas::Card a[7], b[7];
            a[0] = mine[0];
            a[1] = mine[1];
            b[0] = theirs[0];
            b[1] = theirs[1];
            int n = 2;
            for (int k = 0; k < boardN; ++k) {
                a[n] = b[n] = known[k];
                ++n;
            }
            for (int k = 0; k < nRun; ++k) {
                a[n] = b[n] = run[k];
                ++n;
            }
            const int s0 = texas::evaluate7(a);
            const int s1 = texas::evaluate7(b);
            if (s0 > s1) win += 1.0;
            else if (s0 == s1) tie += 1.0;
            total += 1.0;
        };
        const int need = 5 - boardN;
        if (need <= 0) {
            evalOne(nullptr, 0);
        } else if (need == 1) {
            for (int x = 0; x < nDeck; ++x) evalOne(&deck[x], 1);
        } else if (need == 2) {
            for (int x = 0; x < nDeck; ++x) {
                for (int y = x + 1; y < nDeck; ++y) {
                    const texas::Card run[2] = {deck[x], deck[y]};
                    evalOne(run, 2);
                }
            }
        } else {
            return -1.0;  // 翻前不做（太贵，且外部亏损不在翻前）
        }
        return (total > 0) ? (win + 0.5 * tie) / total : 0.5;
    };

    struct Agg {
        int n = 0;       // 可评决策（s 有跟注压力且策略质量落在弃/跟）
        int hits = 0;    // argmax(弃/跟) 命中事后最优
        int skipped = 0; // 有跟注压力但策略质量几乎全在加注/弃跟之外
        double loss = 0.0;
        double lossSq = 0.0;
        double raiseMass = 0.0;
    };
    Agg total, byStreet[5], byBucket[5];
    const int potEdge[4] = {300, 600, 1500, 4000};
    auto bucketOf = [&](int pot) {
        int b = 0;
        while (b < 4 && pot >= potEdge[b]) ++b;
        return b;
    };

    int handsDone = 0, badReplay = 0, oppDecisions = 0;
    for (const std::string& path : csvs) {
        std::ifstream in(path);
        if (!in) {
            std::printf("打不开 %s\n", path.c_str());
            continue;
        }
        std::string line;
        std::getline(in, line);
        texas::Rng rng(11);
        while (std::getline(in, line)) {
            if (maxHands > 0 && handsDone >= maxHands) break;
            const auto f = splitCsvLine(line);
            if (f.size() < 10 || f[6].empty()) continue;
            const int clientPos = std::atoi(f[1].c_str());
            const int mySeat = (clientPos == 0) ? 1 : 0;
            const auto hole = splitWs(f[7]);
            const auto bot = splitWs(f[8]);
            const auto boardS = splitWs(f[9]);
            if (hole.size() < 2 || bot.size() < 2) continue;
            texas::Card mine[2] = {texas::parseCard(hole[0].c_str()),
                                   texas::parseCard(hole[1].c_str())};
            texas::Card theirs[2] = {texas::parseCard(bot[0].c_str()),
                                     texas::parseCard(bot[1].c_str())};
            texas::Card bd[5] = {0, 0, 0, 0, 0};
            bool ok = mine[0] >= 0 && mine[1] >= 0 && theirs[0] >= 0 && theirs[1] >= 0;
            for (size_t k = 0; ok && k < boardS.size() && k < 5; ++k) {
                bd[k] = texas::parseCard(boardS[k].c_str());
                ok = bd[k] >= 0;
            }
            if (!ok) continue;
            ++handsDone;
            texas::HUState st = (clientPos == 0) ? texas::makeState(theirs, mine, bd, 0)
                                                 : texas::makeState(mine, theirs, bd, 0);
            const std::string& action = f[6];
            size_t i = 0;
            bool broke = false;
            while (i < action.size() && !st.finished) {
                const char c = action[i++];
                texas::Move m{};
                if (c == '/') continue;
                if (c == 'k') {
                    m.kind = texas::MK_CHECK;
                    m.to = st.contrib[st.toAct];
                } else if (c == 'c') {
                    m.kind = texas::MK_CALL;
                    m.to = st.contrib[1 - st.toAct];
                } else if (c == 'f') {
                    m.kind = texas::MK_FOLD;
                    m.to = st.contrib[st.toAct];
                } else if (c == 'b') {
                    int v = 0;
                    const size_t start = i;
                    while (i < action.size() && action[i] >= '0' && action[i] <= '9') {
                        v = v * 10 + (action[i] - '0');
                        ++i;
                    }
                    if (i == start) {
                        broke = true;
                        break;
                    }
                    m.kind = texas::MK_RAISE;
                    m.to = static_cast<int16_t>(v);
                } else {
                    broke = true;
                    break;
                }
                // 应用动作**之前**给我们的策略打分（状态与外部日志完全一致）。
                if (st.toAct != mySeat) {
                    ++oppDecisions;
                } else if (st.street >= texas::ST_FLOP) {
                    texas::LegalMove lm[texas::kNumActions];
                    const int n = texas::legalMoves(st, lm);
                    if (n > 0) {
                        float probs[texas::kNumActions] = {};
                        policyProbs(st, lm, n, probs);
                        float pFold = 0.0f, pCall = 0.0f;
                        bool hasFold = false, hasCall = false;
                        for (int k = 0; k < n; ++k) {
                            if (lm[k].slot == texas::kSlotFold) {
                                pFold = probs[lm[k].slot];
                                hasFold = true;
                            } else if (lm[k].slot == texas::kSlotCheckCall) {
                                pCall = probs[lm[k].slot];
                                hasCall = true;
                            }
                        }
                        const int need = texas::toCall(st, mySeat);
                        const double pFC = double(pFold) + double(pCall);
                        if (need > 0 && hasFold && hasCall && pFC > 1e-3) {
                            const int boardN = int(st.street) + 2;
                            const double eq = exactEquity(mine, theirs, bd, boardN);
                            if (eq >= 0.0) {
                                const int contrib = int(st.contrib[mySeat]);
                                const int pot = st.pot();
                                const double evFold = -double(contrib);
                                const double evCall =
                                    eq * double(pot + need) - double(contrib + need);
                                const double best = std::max(evFold, evCall);
                                const double loss = (double(pFold) * (best - evFold) +
                                                     double(pCall) * (best - evCall)) /
                                                    pFC;
                                const bool hitBest =
                                    (evCall >= evFold) ? (pCall >= pFold) : (pFold >= pCall);
                                Agg* as[2] = {&total, &byStreet[int(st.street)]};
                                Agg* ab = &byBucket[bucketOf(pot)];
                                for (Agg* a : as) {
                                    ++a->n;
                                    a->loss += loss;
                                    a->lossSq += loss * loss;
                                    a->raiseMass += 1.0 - pFC;
                                    if (hitBest) ++a->hits;
                                }
                                ++ab->n;
                                ab->loss += loss;
                                ab->lossSq += loss * loss;
                                ab->raiseMass += 1.0 - pFC;
                                if (hitBest) ++ab->hits;
                            }
                        } else {
                            ++total.skipped;
                            ++byStreet[int(st.street)].skipped;
                        }
                    }
                }
                texas::applyMove(st, m, rng);
                if (st.finished) break;
            }
            if (broke) ++badReplay;
        }
    }

    const char* streetNames[4] = {"flop", "turn", "river", "?"};
    auto show = [&](const char* name, const Agg& a) {
        if (a.n <= 0) {
            std::printf("  %-9s n=0\n", name);
            return;
        }
        const double mean = a.loss / double(a.n);
        const double var = std::max(0.0, a.lossSq / double(a.n) - mean * mean);
        const double se = std::sqrt(var / double(a.n));
        std::printf("  %-9s dec=%5d | 命中 %5.1f%% | 事后损失 %7.1f ± %5.1f 筹码/决策 "
                    "| %+6.2f BB/手归一 | 加注质量 %.2f\n",
                    name, a.n, 100.0 * double(a.hits) / double(a.n), mean, se,
                    handsDone > 0 ? a.loss / double(handsDone) / 100.0 : 0.0, a.raiseMass / a.n);
    };
    std::printf("slumbot-policy-eval: %s | T=%.2f | %d 手，重放失败 %d | 对手决策 %d\n",
                dir.c_str(), double(temp), handsDone, badReplay, oppDecisions);
    show("合计", total);
    std::printf("  按街：\n");
    for (int s = 1; s <= 3; ++s) show(streetNames[s - 1], byStreet[s]);
    std::printf("  按底池：\n");
    const char* potNames[5] = {"0-300", "300-600", "600-1500", "1500-4000", ">4000"};
    for (int b = 0; b < 5; ++b) show(potNames[b], byBucket[b]);
    return 0;
}

// 把 Slumbot 的逐手日志变成行为克隆数据集，并从现有检查点微调出"对手模型"。
// 动机：内部基准和外部强度的相关性很弱，唯一已知的成熟 AI 是 Slumbot。
// 想赢它，先得有一份它的策略（哪怕是近似），之后才能做 BR / 放进联赛。
//
// 只训练 actor（Slumbot 的动作标签是 slot），critic 从 --resume 原样拷贝，
// 这样输出目录可以直接喂给 `train --freeze-opponent`。
// 河牌范围诊断：从外部日志里提取"双方在河牌的手牌百分位 + 动作"，
// 按"是否面对下注/下注尺寸/底池"分桶。用途：
//   1) 看 Slumbot 面对我们下注时，弃牌/跟注/加注各是什么牌力
//      （跟注范围的强度决定我们下注的 EV，而不只是弃牌率）；
//   2) 看我们自己面对下注时，弃牌/跟注的牌力是否错位
//      （大池摊牌亏损的直接来源）。
// 百分位来自 handStrength(hole, board, 5)，是精确排名，不是分位点近似。
// 摊牌诊断：对我们的模型 vs 某个脚本对手打 N 手，按"摊牌/非摊牌"拆收益，
// 并给出双方的摊牌牌型分布。外部 12000 手（README #107）显示赤字全部来自
// 摊牌（胜率 20.4%、我们 22% 是高牌），但外部一轮要跑几小时。这个子命令把
// 同一测量搬到本地：几十秒就出数，用来当后续改动的闸门指标。
int cmdShowdown(int argc, char** argv) {
    const std::string dir = argStr(argc, argv, "--resume", "");
    const std::string opp = argStr(argc, argv, "--vs", "solid");
    const int hands = argInt(argc, argv, "--hands", 2000);
    const int seed = argInt(argc, argv, "--seed", 90001);
    texas::EvalTeam team;
    texas::nn::ModelConfig mc;
    mc.hidden = argInt(argc, argv, "--hidden", 256);
    mc.lstmHidden = argInt(argc, argv, "--lstm", 128);
    static texas::nn::Actor actors[2];
    if (!dir.empty()) {
        for (int s = 0; s < 2; ++s) {
            actors[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        }
        if (!loadActors(actors, dir)) {
            std::printf("cannot load actors from %s\n", dir.c_str());
            return 1;
        }
        team.actor[0] = &actors[0];
        team.actor[1] = &actors[1];
    }
    team.policyTemp = static_cast<float>(argDouble(argc, argv, "--temp", 0.5));
    team.policyGreedy = hasFlag(argc, argv, "--greedy");
    team.amountMid = hasFlag(argc, argv, "--amount-mid");
    team.rangeEqSamples = argInt(argc, argv, "--range-eq", 0);
    team.rangeEqIters = argInt(argc, argv, "--range-eq-iters", 48);
    team.rangerAir = static_cast<float>(argDouble(argc, argv, "--ranger-air", 0.28));
    if (opp == "cfr") {
        team.oppCfrIters = argInt(argc, argv, "--opp-cfr-iters", 120);
        team.oppCfrMinPot = argInt(argc, argv, "--opp-cfr-min-pot", 1500);
    }

    texas::ShowdownStats sd;
    const texas::EvalResult r = texas::evalShowdownVs(team, opp, hands,
                                                     static_cast<uint64_t>(seed),
                                                     argInt(argc, argv, "--equity-iters", 24),
                                                     sd);
    std::printf("showdown %s vs %s | T=%.2f\n", dir.c_str(), opp.c_str(), team.policyTemp);
    std::printf("  总 EV %+0.1f ± %.1f 筹码/手 (%.1f BB/100)\n", r.chipsPerHand, r.stderr,
                r.bbPer100);
    sd.print(opp.c_str(), r.hands);
    return 0;
}

int cmdRiverRanges(int argc, char** argv) {
    const std::vector<std::string> csvs = argValues(argc, argv, "--csv");
    if (csvs.empty()) {
        std::printf("river-ranges 需要 --csv FILE（可重复）\n");
        return 1;
    }
    // 样本外检验：前 split 比例的手只用来建范围表，后面的手才评估 EV。
    const double split = argDouble(argc, argv, "--split", 1.0);

    struct RiverSample {
        int handId = 0;
        int actorSide = 0;  // 0 = 我们（客户端座位）, 1 = 对手
        int facing = 0;     // 1 = 面对下注
        int sizeBucket = 0; // 下注额/下注前底池：<0.5 / 0.5-0.8 / 0.8-1.3 / >1.3
        int potBucket = 0;  // 0 = <1500, 1 = >=1500
        int action = 0;     // 0 = fold, 1 = call/check, 2 = raise/bet
        int potBefore = 0;
        int amount = 0;     // 本动作投入的额外筹码（下注/跟注额）
        float pct = 0.0f;
    };
    std::vector<RiverSample> samples;
    int handId = 0, handsSeen = 0, riverDecisions = 0;
    texas::Rng rng(7);

    for (const std::string& path : csvs) {
        std::ifstream in(path);
        if (!in) {
            std::printf("cannot open %s\n", path.c_str());
            return 1;
        }
        std::string line;
        std::getline(in, line);  // header
        while (std::getline(in, line)) {
            const auto f = splitCsvLine(line);
            if (f.size() < 10 || f[6].empty()) continue;
            const int clientPos = std::atoi(f[1].c_str());
            const int mySeat = (clientPos == 0) ? 1 : 0;
            const auto hole = splitWs(f[7]);
            const auto bot = splitWs(f[8]);
            const auto board = splitWs(f[9]);
            if (hole.size() < 2 || bot.size() < 2 || board.size() < 5) continue;
            texas::Card mine[2] = {texas::parseCard(hole[0].c_str()),
                                   texas::parseCard(hole[1].c_str())};
            texas::Card theirs[2] = {texas::parseCard(bot[0].c_str()),
                                     texas::parseCard(bot[1].c_str())};
            texas::Card bd[5] = {0, 0, 0, 0, 0};
            bool ok = mine[0] >= 0 && mine[1] >= 0 && theirs[0] >= 0 && theirs[1] >= 0;
            for (size_t k = 0; ok && k < 5; ++k) {
                bd[k] = texas::parseCard(board[k].c_str());
                ok = bd[k] >= 0;
            }
            if (!ok) continue;

            texas::HUState st = (clientPos == 0) ? texas::makeState(theirs, mine, bd, 0)
                                                 : texas::makeState(mine, theirs, bd, 0);
            ++handsSeen;
            ++handId;

            const std::string& action = f[6];
            size_t i = 0;
            while (i < action.size()) {
                const char c = action[i++];
                if (c == '/') continue;
                texas::Move m{};
                if (c == 'k') {
                    m.kind = texas::MK_CHECK;
                    m.to = st.contrib[st.toAct];
                } else if (c == 'c') {
                    m.kind = texas::MK_CALL;
                    m.to = st.contrib[1 - st.toAct];
                } else if (c == 'f') {
                    m.kind = texas::MK_FOLD;
                    m.to = st.contrib[st.toAct];
                } else if (c == 'b') {
                    int v = 0;
                    const size_t start = i;
                    while (i < action.size() && action[i] >= '0' && action[i] <= '9') {
                        v = v * 10 + (action[i] - '0');
                        ++i;
                    }
                    if (i == start) break;
                    m.kind = texas::MK_RAISE;
                    m.to = static_cast<int16_t>(v);
                } else {
                    break;
                }

                if (st.street == texas::ST_RIVER && st.boardN == 5) {
                    const int actor = st.toAct;
                    const int need = texas::toCall(st, actor);
                    const int pot = st.pot();
                    const int amount = int(m.to) - int(st.contrib[actor]);
                    float betRatio = 0.0f;
                    if (need > 0) {
                        betRatio = float(need) / float(std::max(1, pot - need));
                    } else if (m.kind == texas::MK_RAISE || m.kind == texas::MK_ALLIN) {
                        betRatio = float(amount) / float(std::max(1, pot));
                    }
                    const int sb = (betRatio < 0.5f) ? 0
                                   : (betRatio < 0.8f) ? 1
                                   : (betRatio < 1.3f) ? 2 : 3;
                    RiverSample rs;
                    rs.handId = handId;
                    rs.actorSide = (actor == mySeat) ? 0 : 1;
                    rs.facing = (need > 0) ? 1 : 0;
                    rs.sizeBucket = sb;
                    rs.potBucket = (pot >= 1500) ? 1 : 0;
                    rs.action = (m.kind == texas::MK_FOLD) ? 0
                                : (m.kind == texas::MK_CHECK || m.kind == texas::MK_CALL) ? 1 : 2;
                    const texas::Card* hh = (actor == mySeat) ? mine : theirs;
                    rs.pct = texas::handStrength(hh, bd, 5);
                    rs.potBefore = pot;
                    rs.amount = (need > 0) ? need : ((m.kind == texas::MK_RAISE) ? amount : 0);
                    samples.push_back(rs);
                    ++riverDecisions;
                }
                texas::applyMove(st, m, rng);
                if (st.finished) break;
            }
        }
    }

    const int splitId = int(double(handsSeen) * split);
    if (split < 1.0) {
        std::printf("样本外：前 %d 手建表，后 %d 手评估\n", splitId, handsSeen - splitId);
    }
    const char* sideName[2] = {"我们", "对手"};
    const char* potName[2] = {"<1500", ">=1500"};
    const char* sizeName[4] = {"<0.5x", "0.5-0.8x", "0.8-1.3x", ">1.3x"};
    std::printf("river-ranges: %d 个文件, %d 手, %d 个河牌决策\n", int(csvs.size()), handsSeen,
                riverDecisions);

    auto printGroup = [&](int side, int facing, int pb, int sb) {
        std::vector<float> foldP, callP, raiseP;
        int folds = 0, calls = 0, raises = 0;
        for (const RiverSample& r : samples) {
            if (r.actorSide != side || r.facing != facing || r.potBucket != pb ||
                r.sizeBucket != sb) {
                continue;
            }
            if (r.action == 0) {
                ++folds;
                foldP.push_back(r.pct);
            } else if (r.action == 1) {
                ++calls;
                callP.push_back(r.pct);
            } else {
                ++raises;
                raiseP.push_back(r.pct);
            }
        }
        const int n = folds + calls + raises;
        if (n == 0) return;
        auto q = [](std::vector<float>& v, double p) {
            if (v.empty()) return 0.0f;
            std::sort(v.begin(), v.end());
            const size_t idx = std::min(v.size() - 1,
                                        static_cast<size_t>(p * double(v.size())));
            return v[idx];
        };
        std::printf(
            "  %s %s %s %s: n=%4d | fold %5.1f%% call %5.1f%% raise %5.1f%%"
            " | pct(fold) %.2f/%.2f pct(call) %.2f/%.2f pct(raise) %.2f/%.2f\n",
            sideName[side], potName[pb], facing ? "面对下注" : "无人下注", sizeName[sb], n,
            100.0 * folds / n, 100.0 * calls / n, 100.0 * raises / n, q(foldP, 0.5), q(foldP, 0.75),
            q(callP, 0.5), q(callP, 0.75), q(raiseP, 0.5), q(raiseP, 0.75));
    };

    std::printf("[对手（Slumbot）面对我们的下注]\n");
    for (int pb = 0; pb < 2; ++pb) {
        for (int sb = 0; sb < 4; ++sb) printGroup(1, 1, pb, sb);
    }
    std::printf("[对手主动下注 / 过牌]\n");
    for (int pb = 0; pb < 2; ++pb) {
        for (int sb = 0; sb < 4; ++sb) printGroup(1, 0, pb, sb);
    }
    std::printf("[我们面对对手下注]\n");
    for (int pb = 0; pb < 2; ++pb) {
        for (int sb = 0; sb < 4; ++sb) printGroup(0, 1, pb, sb);
    }
    std::printf("[我们无人下注 / 主动下注]\n");
    for (int pb = 0; pb < 2; ++pb) {
        for (int sb = 0; sb < 4; ++sb) printGroup(0, 0, pb, sb);
    }

    // ---- 河牌 EV：用测到的对手范围，算"过牌 vs 各种尺寸下注" ----
    // 对每个我们面对过牌（facing==0）的河牌决策，用同底池桶、同尺寸桶的
    // 对手响应范围（排除本手）估计每个动作的 EV，再和实际动作比较。
    // 这是"如果按外部范围打，能多赢多少"的直接估计（不含加注后的再决策，
    // 所以对下注略保守：对手加注时按我们损失下注额计）。
    auto winProb = [](float pct, const std::vector<float>& range, int excludeHand,
                      const std::vector<int>& handIds) {
        double w = 0.0;
        int n = 0;
        for (size_t i = 0; i < range.size(); ++i) {
            if (handIds[i] == excludeHand) continue;
            if (range[i] < pct) w += 1.0;
            else if (range[i] == pct) w += 0.5;
            ++n;
        }
        return n > 0 ? w / double(n) : 0.5;
    };

    for (int pb = 0; pb < 2; ++pb) {
        // 对手面对下注（同尺寸桶）的 fold/call/raise 比例 + 跟注范围
        struct Resp {
            int n = 0, folds = 0, calls = 0, raises = 0;
            std::vector<float> callPct;
            std::vector<int> callIds;
        } resp[4];
        for (const RiverSample& r : samples) {
            if (r.actorSide != 1 || r.facing != 1 || r.potBucket != pb) continue;
            if (r.handId > splitId) continue;
            Resp& q = resp[r.sizeBucket];
            ++q.n;
            if (r.action == 0) {
                ++q.folds;
            } else if (r.action == 1) {
                ++q.calls;
                q.callPct.push_back(r.pct);
                q.callIds.push_back(r.handId);
            } else {
                ++q.raises;
            }
        }
        // 对手 check-behind 范围
        std::vector<float> checkPct;
        std::vector<int> checkIds;
        for (const RiverSample& r : samples) {
            if (r.actorSide == 1 && r.facing == 0 && r.action == 1 && r.potBucket == pb &&
                r.handId <= splitId) {
                checkPct.push_back(r.pct);
                checkIds.push_back(r.handId);
            }
        }
        if (checkPct.empty()) continue;

        double actualSum = 0.0, bestSum = 0.0;
        int nDec = 0;
        int bestCount[5] = {0, 0, 0, 0, 0};  // 0=check, 1..4 = 尺寸桶
        int actualCount[5] = {0, 0, 0, 0, 0};
        double gapByActual[5] = {0, 0, 0, 0, 0};
        for (const RiverSample& r : samples) {
            if (r.actorSide != 0 || r.facing != 0 || r.potBucket != pb) continue;
            if (r.handId <= splitId) continue;
            const double P = double(r.potBefore);
            const double evCheck = winProb(r.pct, checkPct, r.handId, checkIds) * P;
            double best = evCheck;
            int bestSb = -1;
            for (int sb = 0; sb < 4; ++sb) {
                const Resp& q = resp[sb];
                if (q.n < 10) continue;
                const double B = P * (sb == 0 ? 0.35 : sb == 1 ? 0.65 : sb == 2 ? 1.05 : 1.5);
                const double f = double(q.folds) / q.n;
                const double c = double(q.calls) / q.n;
                const double rr = double(q.raises) / q.n;
                const double w = winProb(r.pct, q.callPct, r.handId, q.callIds);
                const double ev = f * P + c * (w * (P + 2.0 * B) - B) + rr * (-B);
                if (ev > best) {
                    best = ev;
                    bestSb = sb;
                }
            }
            double actual = evCheck;
            if (r.action == 2) {
                const Resp& q = resp[r.sizeBucket];
                if (q.n >= 10) {
                    const double B = double(r.amount);
                    const double f = double(q.folds) / q.n;
                    const double c = double(q.calls) / q.n;
                    const double rr = double(q.raises) / q.n;
                    const double w = winProb(r.pct, q.callPct, r.handId, q.callIds);
                    actual = f * P + c * (w * (P + 2.0 * B) - B) + rr * (-B);
                }
            }
            actualSum += actual;
            bestSum += best;
            ++nDec;
            ++bestCount[bestSb + 1];
            const int asb = (r.action == 2) ? (r.sizeBucket + 1) : 0;
            ++actualCount[asb];
            gapByActual[asb] += (best - actual);
        }
        if (nDec > 0) {
            std::printf("  河牌 EV [pot %s]: %d 个决策 | 实际 %.0f / 最优 %.0f 筹码/决策"
                        "（差 %.0f = %.2f BB/手，若每手一次）\n",
                        potName[pb], nDec, actualSum / nDec, bestSum / nDec,
                        (bestSum - actualSum) / nDec, (bestSum - actualSum) / nDec / 100.0);
            const char* actName[5] = {"check", "<0.5x", "0.5-0.8x", "0.8-1.3x", ">1.3x"};
            std::printf("    最优动作分布:");
            for (int k = 0; k < 5; ++k) {
                if (bestCount[k] > 0) std::printf(" %s %d", actName[k], bestCount[k]);
            }
            std::printf(" | 实际动作分布:");
            for (int k = 0; k < 5; ++k) {
                if (actualCount[k] > 0) {
                    std::printf(" %s %d(差 %.0f/个)", actName[k], actualCount[k],
                                gapByActual[k] / actualCount[k]);
                }
            }
            std::printf("\n");
        }
    }
    return 0;
}

int cmdSlumbotFit(int argc, char** argv) {
    std::vector<std::string> csvs = argValues(argc, argv, "--csv");
    // --holdout-csv：整份文件只做留出。跨会话留出比同批交错切分更诚实
    // （同一手牌的状态不会同时出现在两边）。
    const std::vector<std::string> holdCsvs = argValues(argc, argv, "--holdout-csv");
    if (csvs.empty()) {
        std::printf("slumbot-fit 需要 --csv FILE（可重复）\n");
        return 1;
    }
    struct CsvSrc {
        std::string path;
        bool holdout = false;
    };
    std::vector<CsvSrc> srcs;
    for (const auto& p : csvs) srcs.push_back({p, false});
    for (const auto& p : holdCsvs) srcs.push_back({p, true});
    const std::string init = argStr(argc, argv, "--resume", "");
    const std::string out = argStr(argc, argv, "--out", "");
    if (out.empty()) {
        std::printf("slumbot-fit 需要 --out DIR\n");
        return 1;
    }
    const int epochs = argInt(argc, argv, "--epochs", 8);
    const double lr = argDouble(argc, argv, "--lr", 2e-4);
    const int batchSize = argInt(argc, argv, "--batch", 128);
    const int holdoutEvery = std::max(2, argInt(argc, argv, "--holdout-every", 5));
    // 类别加权：BB 座位 68% 的标签是 check/call，普通 CE 会退化成"一直跟注"。
    // 加权后模型会预测更多 fold/raise，更接近真实分布（作为模拟器才可用）。
    const double cwPower = argDouble(argc, argv, "--class-weight-power", 0.0);
    const long long seed = argInt(argc, argv, "--seed", 424242);

    // ---- 1. 重放日志，抽 Slumbot 的决策 ----
    std::vector<texas::Transition> trainSet[2], holdSet[2];
    int hands = 0, lines = 0, held = 0;
    texas::Rng rng(7);
    for (const CsvSrc& src : srcs) {
        const std::string& path = src.path;
        std::ifstream in(path);
        if (!in) {
            std::printf("cannot open %s\n", path.c_str());
            return 1;
        }
        std::string line;
        std::getline(in, line);  // header
        while (std::getline(in, line)) {
            ++lines;
            const auto f = splitCsvLine(line);
            if (f.size() < 10 || f[6].empty()) continue;
            const int clientPos = std::atoi(f[1].c_str());
            const int mySeat = (clientPos == 0) ? 1 : 0;
            const int botSeat = 1 - mySeat;
            const auto hole = splitWs(f[7]);
            const auto bot = splitWs(f[8]);
            const auto board = splitWs(f[9]);
            if (hole.size() < 2 || bot.size() < 2) continue;
            texas::Card mine[2] = {texas::parseCard(hole[0].c_str()),
                                   texas::parseCard(hole[1].c_str())};
            texas::Card theirs[2] = {texas::parseCard(bot[0].c_str()),
                                     texas::parseCard(bot[1].c_str())};
            texas::Card bd[5] = {0, 0, 0, 0, 0};
            bool ok = mine[0] >= 0 && mine[1] >= 0 && theirs[0] >= 0 && theirs[1] >= 0;
            for (size_t k = 0; ok && k < board.size() && k < 5; ++k) {
                bd[k] = texas::parseCard(board[k].c_str());
                ok = bd[k] >= 0;
            }
            if (!ok) continue;

            texas::HUState st = (clientPos == 0) ? texas::makeState(theirs, mine, bd, 0)
                                                 : texas::makeState(mine, theirs, bd, 0);
            const bool isHold = src.holdout || (hands % holdoutEvery) == 0;
            if (isHold) ++held;
            ++hands;

            const std::string& action = f[6];
            size_t i = 0;
            while (i < action.size()) {
                const char c = action[i++];
                if (c == '/') continue;
                texas::Move m{};
                if (c == 'k') {
                    m.kind = texas::MK_CHECK;
                    m.to = st.contrib[st.toAct];
                } else if (c == 'c') {
                    m.kind = texas::MK_CALL;
                    m.to = st.contrib[1 - st.toAct];
                } else if (c == 'f') {
                    m.kind = texas::MK_FOLD;
                    m.to = st.contrib[st.toAct];
                } else if (c == 'b') {
                    int v = 0;
                    const size_t start = i;
                    while (i < action.size() && action[i] >= '0' && action[i] <= '9') {
                        v = v * 10 + (action[i] - '0');
                        ++i;
                    }
                    if (i == start) break;
                    m.kind = texas::MK_RAISE;
                    m.to = static_cast<int16_t>(v);
                } else {
                    break;
                }

                if (st.toAct == botSeat) {
                    texas::LegalMove lm[texas::kNumActions];
                    const int n = texas::legalMoves(st, lm);
                    if (n > 0) {
                        texas::Transition tr;
                        texas::Obs obs;
                        texas::encodeObservation(
                            st, obs, float(texas::perfectEquity(st, botSeat, 24)));
                        tr.node = obs.node;
                        tr.seq = obs.seq;
                        tr.extra = obs.extra;
                        tr.actions.resize(static_cast<size_t>(n));
                        for (int k = 0; k < n; ++k) {
                            tr.actions[static_cast<size_t>(k)].slot = lm[k].slot;
                            tr.actions[static_cast<size_t>(k)].mv = lm[k].mv;
                            tr.actions[static_cast<size_t>(k)].hi = lm[k].toHi;
                            texas::encodeActionFeature(st, lm[k],
                                                       tr.actions[static_cast<size_t>(k)].feat);
                        }
                        // 标签：把实际金额映回尺寸带（顶到全下就算 all-in）。
                        int label = texas::kSlotAllIn;
                        if (m.kind == texas::MK_FOLD) {
                            label = texas::kSlotFold;
                        } else if (m.kind == texas::MK_CHECK || m.kind == texas::MK_CALL) {
                            label = texas::kSlotCheckCall;
                        } else {
                            const int maxTo = texas::maxRaiseTo(st, st.toAct);
                            if (m.kind != texas::MK_ALLIN && int(m.to) < maxTo) {
                                for (int k = 0; k < n; ++k) {
                                    if (lm[k].slot >= texas::kSlotSize0 &&
                                        lm[k].slot < texas::kSlotAllIn &&
                                        int(m.to) >= int(lm[k].mv.to) && int(m.to) <= lm[k].toHi) {
                                        label = lm[k].slot;
                                        break;
                                    }
                                }
                            }
                        }
                        tr.action = label;
                        (isHold ? holdSet[botSeat] : trainSet[botSeat]).push_back(std::move(tr));
                    }
                }
                texas::applyMove(st, m, rng);
                if (st.finished) break;
            }
        }
    }
    std::printf("slumbot-fit: %zu 文件 | %d 行 | 重放 %d 手（留出 %d）\n", csvs.size(), lines,
                hands, held);
    for (int s = 0; s < 2; ++s) {
        std::printf("  seat%d: train %zu 样本, holdout %zu 样本\n", s, trainSet[s].size(),
                    holdSet[s].size());
    }

    // ---- 2. 建网络（可从 --resume 热启动）----
    texas::nn::ModelConfig mc;
    mc.hidden = argInt(argc, argv, "--hidden", 256);
    mc.lstmHidden = argInt(argc, argv, "--lstm", 128);
    texas::nn::Actor actors[2];
    for (int s = 0; s < 2; ++s) {
        actors[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
    }
    if (!init.empty() && !loadActors(actors, init)) {
        std::printf("cannot load init actors from %s\n", init.c_str());
        return 1;
    }

    auto buildBatch = [](const std::vector<texas::Transition>& src, size_t lo, size_t hi) {
        const int64_t B = static_cast<int64_t>(hi - lo);
        int64_t L = 0;
        for (size_t i = lo; i < hi; ++i) L += static_cast<int64_t>(src[i].actions.size());
        torch::Tensor node = torch::empty({B, texas::kNodeInput}, torch::kFloat32);
        torch::Tensor seq = torch::empty({B, texas::kHistSteps, texas::kHistDim}, torch::kFloat32);
        torch::Tensor feat = torch::empty({L, texas::kActionInput}, torch::kFloat32);
        torch::Tensor sample = torch::empty({L}, torch::kLong);
        torch::Tensor id = torch::empty({L}, torch::kLong);
        torch::Tensor label = torch::empty({B}, torch::kLong);
        float* pNode = node.data_ptr<float>();
        float* pSeq = seq.data_ptr<float>();
        float* pFeat = feat.data_ptr<float>();
        int64_t* pSample = sample.data_ptr<int64_t>();
        int64_t* pId = id.data_ptr<int64_t>();
        int64_t* pLabel = label.data_ptr<int64_t>();
        int64_t off = 0;
        for (int64_t i = 0; i < B; ++i) {
            const texas::Transition& t = src[lo + static_cast<size_t>(i)];
            std::copy(t.node.begin(), t.node.end(), pNode + i * texas::kNodeInput);
            std::copy(t.seq.begin(), t.seq.end(), pSeq + i * texas::kSeqInput);
            pLabel[i] = t.action;
            for (const texas::ActionOption& a : t.actions) {
                std::copy(a.feat.begin(), a.feat.end(), pFeat + off * texas::kActionInput);
                pSample[off] = i;
                pId[off] = a.slot;
                ++off;
            }
        }
        return std::make_tuple(node, seq, feat, sample, id, label);
    };

    auto accuracy = [&](texas::nn::Actor& actor, const std::vector<texas::Transition>& data) {
        if (data.empty()) return 0.0;
        int right = 0;
        torch::NoGradGuard ng;
        for (size_t lo = 0; lo < data.size(); lo += 512) {
            const size_t hi = std::min(data.size(), lo + 512);
            auto [node, seq, feat, sample, id, label] = buildBatch(data, lo, hi);
            torch::Tensor logits = actor->forward(node, seq, feat, sample, id);
            torch::Tensor pred = logits.argmax(1);
            right += int((pred == label).sum().item<int64_t>());
        }
        return double(right) / double(data.size());
    };

    // 平衡准确率：只在样本数 >=5 的类别上取平均召回。总体准确率会被
    // check/call 这个多数类淹没，看不出模型有没有学到 fold / raise。
    auto balancedAcc = [&](texas::nn::Actor& actor, const std::vector<texas::Transition>& data) {
        if (data.empty()) return 0.0;
        int total[texas::kNumActions] = {0}, hit[texas::kNumActions] = {0};
        torch::NoGradGuard ng;
        for (size_t lo = 0; lo < data.size(); lo += 512) {
            const size_t hi = std::min(data.size(), lo + 512);
            auto [node, seq, feat, sample, id, label] = buildBatch(data, lo, hi);
            torch::Tensor pred = actor->forward(node, seq, feat, sample, id).argmax(1);
            const int64_t* lp = label.data_ptr<int64_t>();
            const int64_t* pp = pred.data_ptr<int64_t>();
            for (int64_t i = 0; i < static_cast<int64_t>(hi - lo); ++i) {
                ++total[lp[i]];
                if (lp[i] == pp[i]) ++hit[lp[i]];
            }
        }
        double sum = 0.0;
        int classes = 0;
        for (int a = 0; a < texas::kNumActions; ++a) {
            if (total[a] < 5) continue;
            sum += double(hit[a]) / double(total[a]);
            ++classes;
        }
        return classes > 0 ? sum / classes : 0.0;
    };

    // ---- 3. 分座位微调（座位 0 = Slumbot 的 SB，座位 1 = 它的 BB）----
    // 先报标签分布和多数类基率：没有它，准确率数字无法判断模型有没有学到东西。
    for (int s = 0; s < 2; ++s) {
        const auto& data = trainSet[s];
        if (data.empty()) continue;
        int cnt[texas::kNumActions] = {0};
        for (const auto& t : data) {
            if (t.action >= 0 && t.action < texas::kNumActions) ++cnt[t.action];
        }
        int best = 0;
        for (int a = 0; a < texas::kNumActions; ++a) best = std::max(best, cnt[a]);
        std::printf("  座位%d 标签分布: fold %d, check/call %d, allin %d, 尺寸带 %d",
                    s, cnt[texas::kSlotFold], cnt[texas::kSlotCheckCall],
                    cnt[texas::kSlotAllIn],
                    int(data.size()) - cnt[texas::kSlotFold] - cnt[texas::kSlotCheckCall] -
                        cnt[texas::kSlotAllIn]);
        std::printf(" | 多数类基率 %.3f\n", double(best) / double(data.size()));
    }
    for (int s = 0; s < 2; ++s) {
        auto& data = trainSet[s];
        if (data.empty()) continue;
        torch::Tensor classW = torch::ones({texas::kNumActions});
        if (cwPower > 0.0) {
            int cnt[texas::kNumActions] = {0};
            for (const auto& t : data) {
                if (t.action >= 0 && t.action < texas::kNumActions) ++cnt[t.action];
            }
            float* w = classW.data_ptr<float>();
            const double n = double(data.size());
            int present = 0;
            for (int a = 0; a < texas::kNumActions; ++a) present += (cnt[a] > 0) ? 1 : 0;
            for (int a = 0; a < texas::kNumActions; ++a) {
                if (cnt[a] <= 0 || present <= 0) {
                    w[a] = 1.0f;
                    continue;
                }
                const double ideal = n / (double(present) * double(cnt[a]));
                w[a] = float(std::pow(ideal, cwPower));
                w[a] = std::min(std::max(w[a], 0.25f), 8.0f);
            }
        }
        std::vector<int> order(data.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = int(i);
        torch::optim::Adam opt(actors[s]->parameters(),
                               torch::optim::AdamOptions(lr).weight_decay(0.0));
        std::mt19937 shuf(static_cast<uint32_t>(seed + s));
        const double baseTrain = accuracy(actors[s], data);
        const double baseHold = accuracy(actors[s], holdSet[s]);
        std::printf("  座位%d 初始：train acc %.3f | holdout acc %.3f | 平衡 acc %.3f\n", s,
                    baseTrain, baseHold, balancedAcc(actors[s], holdSet[s]));
        for (int ep = 0; ep < epochs; ++ep) {
            std::shuffle(order.begin(), order.end(), shuf);
            double sumLoss = 0.0;
            int nb = 0;
            for (size_t lo = 0; lo < order.size(); lo += batchSize) {
                const size_t hi = std::min(order.size(), lo + batchSize);
                std::vector<texas::Transition> batch;
                batch.reserve(hi - lo);
                for (size_t i = lo; i < hi; ++i) batch.push_back(data[order[i]]);
                auto [node, seq, feat, sample, id, label] = buildBatch(batch, 0, batch.size());
                torch::Tensor logits = actors[s]->forward(node, seq, feat, sample, id);
                torch::Tensor lsm = torch::log_softmax(logits, 1);
                torch::Tensor nll = -lsm.gather(1, label.view({-1, 1})).squeeze(1);
                torch::Tensor loss = (cwPower > 0.0) ? (nll * classW.gather(0, label)).mean()
                                                     : nll.mean();
                opt.zero_grad();
                loss.backward();
                opt.step();
                sumLoss += loss.item<double>();
                ++nb;
            }
            std::printf("  座位%d epoch %d/%d: CE %.4f | train acc %.3f | holdout acc %.3f | 平衡 acc %.3f\n",
                        s, ep + 1, epochs, nb > 0 ? sumLoss / nb : 0.0, accuracy(actors[s], data),
                        accuracy(actors[s], holdSet[s]), balancedAcc(actors[s], holdSet[s]));
            if (ep + 1 == epochs && !holdSet[s].empty()) {
                // 动作分布对比：作为"对手模拟器"，预测分布比逐点准确率更要紧。
                int trueCnt[texas::kNumActions] = {0}, predCnt[texas::kNumActions] = {0};
                {
                    torch::NoGradGuard ng;
                    const auto& hd = holdSet[s];
                    for (size_t lo = 0; lo < hd.size(); lo += 512) {
                        const size_t hi = std::min(hd.size(), lo + 512);
                        auto [node, seq, feat, sample, id, label] = buildBatch(hd, lo, hi);
                        torch::Tensor pred = actors[s]->forward(node, seq, feat, sample, id).argmax(1);
                        const int64_t* lp = label.data_ptr<int64_t>();
                        const int64_t* pp = pred.data_ptr<int64_t>();
                        for (int64_t i = 0; i < static_cast<int64_t>(hi - lo); ++i) {
                            ++trueCnt[lp[i]];
                            ++predCnt[pp[i]];
                        }
                    }
                }
                auto grp = [](int a) {
                    if (a == texas::kSlotFold) return 0;
                    if (a == texas::kSlotCheckCall) return 1;
                    if (a == texas::kSlotAllIn) return 3;
                    return 2;
                };
                int tg[4] = {0, 0, 0, 0}, pg[4] = {0, 0, 0, 0};
                for (int a = 0; a < texas::kNumActions; ++a) {
                    tg[grp(a)] += trueCnt[a];
                    pg[grp(a)] += predCnt[a];
                }
                const double tn = std::max(1, int(holdSet[s].size()));
                std::printf("  座位%d 留出分布（真实 vs 预测）：fold %.1f/%.1f%% | call %.1f/%.1f%%"
                            " | raise %.1f/%.1f%% | allin %.1f/%.1f%%\n",
                            s, 100.0 * tg[0] / tn, 100.0 * pg[0] / tn, 100.0 * tg[1] / tn,
                            100.0 * pg[1] / tn, 100.0 * tg[2] / tn, 100.0 * pg[2] / tn,
                            100.0 * tg[3] / tn, 100.0 * pg[3] / tn);
            }
            std::fflush(stdout);
        }
    }

    // ---- 4. 存盘（critic 原样拷贝，方便直接当 --freeze-opponent）----
    std::error_code ec;
    std::filesystem::create_directories(out, ec);
    for (int s = 0; s < 2; ++s) {
        actors[s]->saveTo(out + "/actor" + std::to_string(s) + ".pt");
        if (!init.empty()) {
            std::filesystem::copy_file(init + "/critic" + std::to_string(s) + ".pt",
                                       out + "/critic" + std::to_string(s) + ".pt",
                                       std::filesystem::copy_options::overwrite_existing, ec);
        }
    }
    std::printf("slumbot-fit: 写出到 %s\n", out.c_str());
    return 0;
}

// 用 Slumbot 的公开 API 打真实成熟 AI。规则完全对齐本地引擎
// （盲注 50/100、200BB、每手重置），所以数字可以直接和本地基准对照。
int cmdSlumbot(int argc, char** argv) {
    const std::string dir = argStr(argc, argv, "--resume", "ckpt_v12");
    texas::SlumbotConfig cfg;
    cfg.hands = argInt(argc, argv, "--hands", 100);
    cfg.seed = static_cast<uint64_t>(argInt(argc, argv, "--seed", 20261026));
    cfg.temp = static_cast<float>(argDouble(argc, argv, "--temp", 0.1));
    cfg.delayMs = argInt(argc, argv, "--delay-ms", 150);
    cfg.host = argStr(argc, argv, "--host", "slumbot.com");
    cfg.policy = argStr(argc, argv, "--policy", "actor");
    cfg.logPath = argStr(argc, argv, "--log", "");
    cfg.preflop3betMult =
        static_cast<float>(argDouble(argc, argv, "--preflop-3bet-mult", 1.0));
    cfg.postflopBetMult =
        static_cast<float>(argDouble(argc, argv, "--postflop-bet-mult", 1.0));
    cfg.riverOverbetPot = static_cast<float>(argDouble(argc, argv, "--river-overbet", 0.0));
    cfg.riverOverbetMinPot = argInt(argc, argv, "--river-overbet-min-pot", 1500);
    cfg.allinGuardPrePct = static_cast<float>(argDouble(argc, argv, "--allin-guard-pre", 0.0));
    cfg.allinGuardPostCat = argInt(argc, argv, "--allin-guard-post-cat", -1);
    cfg.allinGuardMinRisk =
        static_cast<float>(argDouble(argc, argv, "--allin-guard-min-risk", 0.5));
    cfg.preflopAggroPct =
        static_cast<float>(argDouble(argc, argv, "--preflop-aggro-pct", 0.0));
    cfg.preflopAggroMaxRaises = argInt(argc, argv, "--preflop-aggro-max-raises", 2);
    cfg.bb3betMinPct = static_cast<float>(argDouble(argc, argv, "--bb-3bet-min-pct", 0.0));
    cfg.flopAggroPct = static_cast<float>(argDouble(argc, argv, "--flop-aggro-pct", 0.0));
    cfg.rangeEqSamples = argInt(argc, argv, "--range-eq", 0);
    cfg.rangeEqIters = argInt(argc, argv, "--range-eq-iters", 48);
    cfg.rangeModel =
        argStr(argc, argv, "--range-model", "policy") == "empirical" ? 1 : 0;
    cfg.turnCfrIters = argInt(argc, argv, "--turn-cfr", 0);
    cfg.turnCfrMinPot = argInt(argc, argv, "--turn-cfr-min-pot", 0);
    cfg.riverCfrIters = argInt(argc, argv, "--river-cfr", 0);
    cfg.riverCfrMinPot = argInt(argc, argv, "--river-cfr-min-pot", 0);
    cfg.riverCfrMaxRaises = argInt(argc, argv, "--river-cfr-raises", 1);
    if (hasFlag(argc, argv, "--river-cfr-no-allin")) cfg.riverCfrAllin = false;

    texas::nn::ModelConfig mc;
    mc.hidden = argInt(argc, argv, "--hidden", 256);
    mc.lstmHidden = argInt(argc, argv, "--lstm", 128);
    texas::nn::Actor actors[2];
    for (int s = 0; s < 2; ++s) {
        actors[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
    }
    // fold/random 诊断策略不需要网络，但加载失败也不该挡住它们。
    if (cfg.policy == "actor" && !loadActors(actors, dir)) {
        std::printf("cannot load actors from %s\n", dir.c_str());
        return 1;
    }

    std::printf("slumbot: %s vs %s | %d 手 | policy=%s T=%.2f | 盲注 %d/%d | 起始 %d\n",
                dir.c_str(), cfg.host.c_str(), cfg.hands, cfg.policy.c_str(), cfg.temp,
                texas::kSmallBlind, texas::kBigBlind, texas::kStartingStack);
    std::fflush(stdout);
    const auto t0 = std::chrono::steady_clock::now();
    const texas::SlumbotResult r = texas::playSlumbot(actors, cfg);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    if (r.hands <= 0) {
        std::printf("没有完成任何一手（errors=%d）\n", r.errors);
        return 1;
    }
    const double per = double(r.winnings) / double(r.hands);
    // 合并标准误也换算成 bb/100：每手 SE × 100 / 100。
    const double seBb100 = r.chipsStderr / double(texas::kBigBlind) * 100.0;
    const double adjPer = r.adjustedWinnings / double(r.hands);
    const double adjSeBb100 = r.adjustedStderr / double(texas::kBigBlind) * 100.0;
    std::printf(
        "\nslumbot 结果: %d/%d 手完成（中止 %d，错误 %d，河牌大注 %d，全下 guard %d，"
        "翻前限攻 %d，翻牌限攻 %d，BB 3-bet 限攻 %d，rangeEq 决策 %d，转牌求解 %d，"
        "河牌求解 %d）\n"
                "  净收益 %+lld 筹码 | %+0.1f ± %.1f 筹码/手 | %+0.1f ± %.1f bb/100 | "
                "胜 %d 负 %d 平 %d | %.1f 手/s\n",
                r.hands, r.attempted, r.attempted - r.hands, r.errors, r.riverOverbets,
                r.allinGuardFolds, r.preflopAggroCapped, r.flopAggroCapped, r.bb3betCapped,
                r.rangeEqDecisions, r.turnCfrDecisions, r.riverCfrDecisions, r.winnings, per,
                r.chipsStderr, per / double(texas::kBigBlind) * 100.0, seBb100, r.wins, r.losses,
                r.pushes, double(r.hands) / std::max(secs, 1e-6));
    std::printf("  全下调整: %+0.1f ± %.1f 筹码/手 | %+0.1f ± %.1f bb/100 | 调整 %d 手（%.0f%%）\n",
                adjPer, r.adjustedStderr,
                adjPer / double(texas::kBigBlind) * 100.0, adjSeBb100, r.allinAdjusted,
                100.0 * double(r.allinAdjusted) / double(r.hands));
    return 0;
}

// ---------------------------------------------------------------------------
// 离线 CFR blueprint（docs/CFR_BLUEPRINT_PLAN.md）
// ---------------------------------------------------------------------------

struct BlueprintMeta {
    uint64_t iters = 0;      // 已完成的总迭代数
    uint64_t seed = 0;
    int buckets = texas::kDefaultBuckets;
    int raiseCap = texas::kDefaultRaiseCap;
    int recall = 0;          // 1 = 压缩完美召回（每街 2 bit 加注历史入 key）
    int slim = 0;            // 1 = 瘦 key（底池桶/跨街历史不入 key，R2 对照）
};

bool writeBlueprintMeta(const std::string& dir, const BlueprintMeta& m) {
    FILE* f = std::fopen((dir + "/meta.txt").c_str(), "w");
    if (!f) return false;
    std::fprintf(f,
                 "iters %llu\nseed %llu\nbuckets %d\nraise_cap %d\nrecall %d\nslim %d\nstride %d\n",
                 static_cast<unsigned long long>(m.iters),
                 static_cast<unsigned long long>(m.seed), m.buckets, m.raiseCap, m.recall, m.slim,
                 texas::kAbstractStride);
    std::fclose(f);
    return true;
}

bool readBlueprintMetaFile(const std::string& path, BlueprintMeta& m) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    bool hasIters = false, hasB = false, hasR = false;
    while (std::getline(in, line)) {
        char k[32] = {};
        long long v = 0;
        if (std::sscanf(line.c_str(), "%31s %lld", k, &v) != 2) continue;
        if (std::strcmp(k, "iters") == 0) { m.iters = static_cast<uint64_t>(v); hasIters = true; }
        else if (std::strcmp(k, "seed") == 0) m.seed = static_cast<uint64_t>(v);
        else if (std::strcmp(k, "buckets") == 0) { m.buckets = static_cast<int>(v); hasB = true; }
        else if (std::strcmp(k, "raise_cap") == 0) { m.raiseCap = static_cast<int>(v); hasR = true; }
        else if (std::strcmp(k, "recall") == 0) m.recall = static_cast<int>(v);
        else if (std::strcmp(k, "slim") == 0) m.slim = static_cast<int>(v);
        else if (std::strcmp(k, "stride") == 0 && v != texas::kAbstractStride) return false;
    }
    return hasIters && hasB && hasR;
}

bool readBlueprintMeta(const std::string& dir, BlueprintMeta& m) {
    return readBlueprintMetaFile(dir + "/meta.txt", m);
}

bool saveBlueprintCheckpoint(const std::string& dir, const texas::RegretTable& tab,
                             const BlueprintMeta& meta, const texas::Rng& rng) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        std::printf("!! 无法创建检查点目录 %s\n", dir.c_str());
        return false;
    }
    if (!tab.save(dir + "/bp.bin")) return false;
    if (!writeBlueprintMeta(dir, meta)) return false;
    std::ofstream ro(dir + "/rng.txt");
    if (ro) ro << rng;  // mt19937_64 全状态，续训逐位可复现
    // 迭代快照：G1 要验证 BR 随迭代下降，必须留得住中间检查点。
    const std::string snapDir = dir + "/snapshots";
    std::filesystem::create_directories(snapDir, ec);
    const std::string tag = std::to_string(meta.iters);
    if (tab.save(snapDir + "/bp_" + tag + ".bin")) {
        FILE* sf = std::fopen((snapDir + "/meta_" + tag + ".txt").c_str(), "w");
        if (sf) {
            std::fprintf(sf, "iters %llu\nbuckets %d\nraise_cap %d\nrecall %d\nslim %d\n",
                         static_cast<unsigned long long>(meta.iters), meta.buckets, meta.raiseCap,
                         meta.recall, meta.slim);
            std::fclose(sf);
        }
    }
    return true;
}

int cmdBlueprintTrain(int argc, char** argv) {
    const uint64_t iters = argU64(argc, argv, "--iters", 20000000ULL);
    const uint64_t saveEvery = std::max<uint64_t>(1, argU64(argc, argv, "--save-every", 2000000ULL));
    const uint64_t logEvery = std::max<uint64_t>(1, argU64(argc, argv, "--log-every", 1000000ULL));
    const std::string outDir = argStr(argc, argv, "--out", "/Users/mingming/models/blueprint_v1");
    const std::string resumeDir = argStr(argc, argv, "--resume", "");
    int buckets = argInt(argc, argv, "--buckets", texas::kDefaultBuckets);
    int raiseCap = argInt(argc, argv, "--raise-cap", texas::kDefaultRaiseCap);
    int recall = argInt(argc, argv, "--recall", 0);  // 1 = 压缩完美召回
    int slim = argInt(argc, argv, "--key-slim", 0);  // 1 = 瘦 key（R2 对照）
    const uint64_t seed = argU64(argc, argv, "--seed", 20261107ULL);
    if (outDir.rfind("/tmp", 0) == 0) {
        std::printf("!! 纪律：blueprint 检查点禁止写 /tmp（会被系统清理），请用 --out 指到 ~/models\n");
        return 1;
    }
    if (iters == 0) { std::printf("--iters 必须 > 0\n"); return 1; }

    texas::StrengthModel& sm = texas::StrengthModel::instance();  // 预热秩表，避免算进吞吐
    texas::RegretTable tab(texas::kAbstractStride);
    texas::Rng rng(seed);
    BlueprintMeta meta;
    meta.seed = seed;
    meta.buckets = buckets;
    meta.raiseCap = raiseCap;
    meta.recall = recall;
    meta.slim = slim;

    if (!resumeDir.empty()) {
        BlueprintMeta rm;
        if (!readBlueprintMeta(resumeDir, rm)) {
            std::printf("!! 读不到 %s/meta.txt\n", resumeDir.c_str());
            return 1;
        }
        // 抽象口径决定 key 布局，不能混训：未显式传参时跟随检查点（与 br/distill 同口径）。
        if (argValue(argc, argv, "--buckets", nullptr) == nullptr) buckets = rm.buckets;
        if (argValue(argc, argv, "--raise-cap", nullptr) == nullptr) raiseCap = rm.raiseCap;
        if (argValue(argc, argv, "--key-slim", nullptr) == nullptr) slim = rm.slim;
        if (rm.buckets != buckets || rm.raiseCap != raiseCap || rm.slim != slim) {
            std::printf("!! 抽象不一致：检查点 B=%d R=%d slim=%d，命令行 B=%d R=%d slim=%d\n",
                        rm.buckets, rm.raiseCap, rm.slim, buckets, raiseCap, slim);
            return 1;
        }
        // 召回口径决定 key 布局，不能混训：未显式传 --recall 时跟随检查点。
        if (argValue(argc, argv, "--recall", nullptr) == nullptr) {
            recall = rm.recall;
        } else if (rm.recall != recall) {
            std::printf("!! 召回不一致：检查点 recall=%d，命令行 recall=%d\n", rm.recall, recall);
            return 1;
        }
        if (!tab.load(resumeDir + "/bp.bin")) {
            std::printf("!! 读不到 %s/bp.bin\n", resumeDir.c_str());
            return 1;
        }
        std::ifstream rin(resumeDir + "/rng.txt");
        if (rin) rin >> rng;
        else {
            std::printf("!! 缺 rng.txt，从 seed+iters 派生 RNG（续训非逐位复现）\n");
            rng.seed(seed + rm.iters);
        }
        meta = rm;
        meta.recall = recall;
        meta.slim = slim;
        std::printf("blueprint resume: %s @ %llu iters, %u infosets, recall=%d slim=%d\n",
                    resumeDir.c_str(), static_cast<unsigned long long>(rm.iters), tab.size(),
                    recall, slim);
    }

    // key 的牌桶字段只有 8 bit：B=0 静默退化、B>256 截断碰撞，都必须拒跑
    // （放在 resume 跟随之后，无效检查点也逃不过）。
    if (buckets < 1 || buckets > 256) {
        std::printf("!! --buckets 必须在 [1,256]（key 牌桶字段 8 bit），当前 %d\n", buckets);
        return 1;
    }

    texas::HUNLGame root(&rng, buckets, raiseCap, &sm, recall, slim);
    const uint64_t start = meta.iters;
    uint64_t done = 0;
    uint64_t nextSave = (start / saveEvery + 1) * saveEvery;
    const auto t0 = std::chrono::steady_clock::now();
    std::printf("blueprint-train: +%llu iters (总 %llu) | B=%d R=%d recall=%d slim=%d seed=%llu | out=%s\n",
                static_cast<unsigned long long>(iters),
                static_cast<unsigned long long>(start + iters), buckets, raiseCap, recall, slim,
                static_cast<unsigned long long>(seed), outDir.c_str());
    // 访问密度 = 累计迭代 / 当前表大小：ε 只作为诊断量看它的斜率，不再当采纳门（见计划书 §9.1）。
    std::printf("  %10s %10s %12s %12s %10s\n", "iters", "secs", "iter/s", "infosets", "访问/角");
    std::fflush(stdout);

    while (done < iters) {
        const uint64_t n = std::min(logEvery, iters - done);
        texas::mcfrTrain(root, tab, n, rng, start + done);
        done += n;
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double visitsPerInfoset =
            tab.size() ? static_cast<double>(start + done) / static_cast<double>(tab.size()) : 0.0;
        std::printf("  %10llu %10.1f %12.0f %12u %10.1f\n",
                    static_cast<unsigned long long>(start + done), sec,
                    static_cast<double>(done) / std::max(sec, 1e-6), tab.size(), visitsPerInfoset);
        std::fflush(stdout);
        if (start + done >= nextSave || done == iters) {
            meta.iters = start + done;
            if (!saveBlueprintCheckpoint(outDir, tab, meta, rng)) {
                std::printf("!! 检查点写盘失败: %s\n", outDir.c_str());
                return 1;
            }
            std::printf("  checkpoint -> %s (%llu iters)\n", outDir.c_str(),
                        static_cast<unsigned long long>(meta.iters));
            std::fflush(stdout);
            while (nextSave <= start + done) nextSave += saveEvery;
        }
    }
    std::printf("blueprint-train 完成: %llu iters, %u infosets, %.1f s\n",
                static_cast<unsigned long long>(start + done), tab.size(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
}

int cmdBlueprintBr(int argc, char** argv) {
    const std::string bp = argStr(argc, argv, "--bp", "/Users/mingming/models/blueprint_v1");
    const int samples = argInt(argc, argv, "--samples", 2000);
    const uint64_t seed = argU64(argc, argv, "--seed", 20261107ULL);
    const bool explicitB = argValue(argc, argv, "--buckets", nullptr) != nullptr;
    const bool explicitR = argValue(argc, argv, "--raise-cap", nullptr) != nullptr;
    const bool explicitRec = argValue(argc, argv, "--recall", nullptr) != nullptr;
    const bool explicitSlim = argValue(argc, argv, "--key-slim", nullptr) != nullptr;
    int buckets = argInt(argc, argv, "--buckets", texas::kDefaultBuckets);
    int raiseCap = argInt(argc, argv, "--raise-cap", texas::kDefaultRaiseCap);
    int recall = argInt(argc, argv, "--recall", 0);
    int slim = argInt(argc, argv, "--key-slim", 0);

    texas::RegretTable tab(texas::kAbstractStride);
    std::string binPath = bp;
    BlueprintMeta m;
    bool haveMeta = false;
    if (std::filesystem::is_directory(bp)) {
        binPath = bp + "/bp.bin";
        haveMeta = readBlueprintMeta(bp, m);
    } else {
        // 迭代快照：snapshots/bp_5000000.bin -> snapshots/meta_5000000.txt
        const std::filesystem::path p(bp);
        const std::string stem = p.stem().string();
        if (stem.rfind("bp_", 0) == 0) {
            haveMeta = readBlueprintMetaFile(
                (p.parent_path() / ("meta_" + stem.substr(3) + ".txt")).string(), m);
        }
    }
    if (haveMeta) {
        if (!explicitB) buckets = m.buckets;
        if (!explicitR) raiseCap = m.raiseCap;
        if (!explicitRec) recall = m.recall;
        if (!explicitSlim) slim = m.slim;
    }
    if (!tab.load(binPath)) {
        std::printf("!! 读不到 blueprint 表: %s\n", binPath.c_str());
        return 1;
    }

    texas::BrConfig cfg;
    cfg.samples = samples;
    cfg.seed = seed;
    cfg.buckets = buckets;
    cfg.raiseCap = raiseCap;
    cfg.recall = recall;
    cfg.slim = slim;
    cfg.byStreet = hasFlag(argc, argv, "--by-street");
    cfg.bothSeats = !hasFlag(argc, argv, "--one-seat");
    cfg.selfPlayCheck = hasFlag(argc, argv, "--self-play-check");
    cfg.maxNodesPerSample = argU64(argc, argv, "--max-nodes", 0);
    std::printf("blueprint-br: %s | %u infosets | %d 抽样 | B=%d R=%d recall=%d slim=%d seed=%llu | %s%s\n",
                binPath.c_str(), tab.size(), samples, buckets, raiseCap, recall, slim,
                static_cast<unsigned long long>(seed),
                cfg.bothSeats ? "双座位" : "单座位", cfg.byStreet ? " +按街分解" : "");
    if (haveMeta && tab.size()) {
        std::printf("  训练访问密度 = %.0f 次/角（%llu iters / %u infosets）\n",
                    static_cast<double>(m.iters) / static_cast<double>(tab.size()),
                    static_cast<unsigned long long>(m.iters), tab.size());
    }
    std::fflush(stdout);
    const auto t0 = std::chrono::steady_clock::now();
    const texas::BrResult r = texas::runBlueprintBR(tab, cfg);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!r.valid) {
        std::printf("!! BR 评估失败（抽样数或表 stride 不合法）\n");
        return 1;
    }
    std::printf("\nblueprint-br 结果: %llu 抽样, %.1f M 节点, %.1f 秒\n",
                static_cast<unsigned long long>(r.samples), double(r.nodes) / 1e6, sec);
    std::printf("  BR0 = %+0.2f ± %.2f 筹码/手\n  BR1 = %+0.2f ± %.2f 筹码/手\n",
                r.br0, r.se0, r.br1, r.se1);
    std::printf("  可利用度 BR0+BR1 = %+0.2f ± %.2f 筹码/手 = %+0.3f ± %.3f BB/手\n",
                r.exploitChips, std::sqrt(r.se0 * r.se0 + r.se1 * r.se1), r.exploitBb, r.seBb);
    if (cfg.byStreet) {
        static const char* kStreetName[4] = {"翻前", "翻牌", "转牌", "河牌"};
        std::printf("  按终局街分解：");
        for (int st = 0; st < 4; ++st)
            std::printf("%s %+.3f  ", kStreetName[st], r.exploitByStreetBb[st]);
        std::printf("BB/手\n");
    }
    // ε 是诊断量（对最坏对手的保证），不是采纳门；采纳看外部 12k 手 ≥ 基线 + 2σ（计划书 §9.1）。
    std::printf("  诊断（非采纳门）：ε = %+0.3f BB/手；同时看访问密度与 ε 的斜率\n", r.exploitBb);
    return 0;
}

// ---------------------------------------------------------------- 蒸馏（M4 / G2）

struct DistillData {
    std::deque<texas::TeacherSample> bySeat[2];
};

void makeDistillSample(const texas::HUState& s, const texas::RegretTable& tab, int buckets,
                       int raiseCap, texas::TeacherSample& ts, int recall = 0, int slim = 0) {
    float target[texas::kNumActions] = {};
    texas::abstractToSlots(s, tab, buckets, raiseCap, target, recall, slim);
    texas::Obs obs;
    const float eq = static_cast<float>(texas::perfectEquity(s, s.toAct, 24));
    texas::encodeObservation(s, obs, eq);
    ts.node = obs.node;
    ts.seq = obs.seq;
    ts.rangeEq = 0.5f;  // 与部署 --range-eq 0（policyMove 默认 0.5）对齐
    for (int i = 0; i < texas::kNumActions; ++i) ts.q[i] = 0.0f;
    texas::LegalMove lm[texas::kNumActions];
    const int m = texas::legalMoves(s, lm);
    ts.actions.clear();
    for (int j = 0; j < m; ++j) {
        texas::ActionOption o;
        o.slot = lm[j].slot;
        o.mv = lm[j].mv;
        o.hi = lm[j].toHi;
        texas::encodeActionFeature(s, lm[j], o.feat);
        ts.actions.push_back(o);
        ts.q[lm[j].slot] = target[lm[j].slot];
    }
}

texas::Move randomLegalMove(const texas::HUState& s, texas::Rng& rng) {
    texas::LegalMove lm[texas::kNumActions];
    const int m = texas::legalMoves(s, lm);
    if (m <= 0) {
        texas::Move mv;
        mv.kind = texas::MK_CHECK;
        mv.to = s.contrib[s.toAct];
        return mv;
    }
    const int j = static_cast<int>(rng() % m);
    texas::Move mv = lm[j].mv;
    if (mv.kind == texas::MK_RAISE && lm[j].toHi > int(mv.to)) {
        mv.to = static_cast<int16_t>(texas::amountInBand(
            int(mv.to), lm[j].toHi, std::generate_canonical<double, 32>(rng)));
    }
    return mv;
}

// blueprint 自对弈 + 随机发牌/随机动作混合采样决策点（计划书 §4.1）。
void generateDistillSamples(const texas::RegretTable& tab, int games, uint64_t seed,
                            int buckets, int raiseCap, float mixRandom, float bpTemp,
                            DistillData& out, int recall = 0, int slim = 0) {
    texas::Rng rng(seed);
    for (int g = 0; g < games; ++g) {
        texas::HUState s = texas::dealNewHand(rng);
        int steps = 0;
        while (!s.finished && steps++ < 80) {
            texas::TeacherSample ts;
            makeDistillSample(s, tab, buckets, raiseCap, ts, recall, slim);
            out.bySeat[s.toAct].push_back(std::move(ts));
            const bool randomBranch = std::generate_canonical<double, 32>(rng) < mixRandom;
            const texas::Move mv = randomBranch
                ? randomLegalMove(s, rng)
                : texas::sampleBlueprintMove(s, tab, buckets, raiseCap, rng, bpTemp, recall, slim);
            texas::applyMove(s, mv, rng);
        }
    }
}

struct FidelityResult {
    double top1 = 0.0;   // actor 与 blueprint 目标的 top-1 槽一致率（严格）
    double top1Tie = 0.0;  // 目标 top 并列时，actor 选任一并列槽都算一致
    double kl = 0.0;     // KL(target || actor)
    double ent = 0.0;    // 目标分布自身熵 H(target)，给 KL 一个相对尺度
    int count = 0;
    bool valid = false;
};

// 单座位留出集保真度。蒸馏每个 step 会走完整一遍数据，等价于一个 epoch，
// 所以训练中要按留出集挑最优点（见 cmdBlueprintDistill 的早停）。
FidelityResult measureFidelitySeat(texas::nn::Actor& actor,
                                   const std::deque<texas::TeacherSample>& samples) {
    FidelityResult r;
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
    {
        actor->eval();
        for (const texas::TeacherSample& ts : samples) {
            const int64_t L = static_cast<int64_t>(ts.actions.size());
            if (L <= 0) continue;
            torch::Tensor nodeT = torch::from_blob(
                const_cast<float*>(ts.node.data()), {1, texas::kNodeInput}, f32).clone();
            torch::Tensor seqT = torch::from_blob(
                const_cast<float*>(ts.seq.data()), {1, texas::kHistSteps, texas::kHistDim}, f32)
                .clone();
            torch::Tensor featT = torch::zeros({L, texas::kActionInput}, f32);
            torch::Tensor sampleT = torch::zeros({L}, torch::kLong);
            torch::Tensor idT = torch::zeros({L}, torch::kLong);
            float* pf = featT.data_ptr<float>();
            int64_t* pid = idT.data_ptr<int64_t>();
            for (int64_t j = 0; j < L; ++j) {
                std::copy(ts.actions[j].feat.begin(), ts.actions[j].feat.end(),
                          pf + j * texas::kActionInput);
                pid[j] = ts.actions[j].slot;
            }
            torch::Tensor rqT = torch::tensor({ts.rangeEq}, f32);
            torch::Tensor logits;
            {
                torch::NoGradGuard ng;
                logits = actor->forward(nodeT, seqT, featT, sampleT, idT, rqT);
            }
            float probs[texas::kNumActions] = {};
            texas::ppo::maskedSoftmax(logits.data_ptr<float>(), texas::kNumActions, probs);
            int aTop = -1, bTop = -1;
            double aBest = -1.0, bBest = -1.0, kl = 0.0, ent = 0.0;
            for (int a = 0; a < texas::kNumActions; ++a) {
                if (ts.q[a] > aBest) { aBest = ts.q[a]; aTop = a; }
                if (probs[a] > bBest) { bBest = probs[a]; bTop = a; }
                if (ts.q[a] > 1e-6) {
                    kl += double(ts.q[a]) * std::log(double(ts.q[a]) /
                                                     std::max(1e-9, double(probs[a])));
                    ent -= double(ts.q[a]) * std::log(double(ts.q[a]));
                }
            }
            if (aTop == bTop) r.top1 += 1.0;
            // 目标在多个槽上并列最大时（同一个抽象动作落到多个尺寸槽），
            // actor 选中任一并列槽都算保真，否则会低估蒸馏效果。
            if (ts.q[bTop] > aBest - 1e-6) r.top1Tie += 1.0;
            r.kl += kl;
            r.ent += ent;
            ++r.count;
        }
    }
    if (r.count > 0) {
        r.top1 /= r.count;
        r.top1Tie /= r.count;
        r.kl /= r.count;
        r.ent /= r.count;
        r.valid = true;
    }
    return r;
}

// 双座位合并读数（报告用）。按样本数加权，避免两座位样本量不同时被平均歪。
FidelityResult measureFidelity(texas::nn::Actor actors[2], const DistillData& data) {
    FidelityResult r;
    for (int seat = 0; seat < 2; ++seat) {
        const FidelityResult f = measureFidelitySeat(actors[seat], data.bySeat[seat]);
        r.top1 += f.top1 * f.count;
        r.top1Tie += f.top1Tie * f.count;
        r.kl += f.kl * f.count;
        r.ent += f.ent * f.count;
        r.count += f.count;
    }
    if (r.count > 0) {
        r.top1 /= r.count;
        r.top1Tie /= r.count;
        r.kl /= r.count;
        r.ent /= r.count;
        r.valid = true;
    }
    return r;
}

// 从 --bp（目录或快照 .bin）读抽象配置，避免与训练时不一致。
void resolveBlueprintPath(const std::string& bp, std::string& binPath, int& buckets,
                          int& raiseCap, int& recall, int& slim, bool explicitB, bool explicitR) {
    BlueprintMeta m;
    bool haveMeta = false;
    binPath = bp;
    if (std::filesystem::is_directory(bp)) {
        binPath = bp + "/bp.bin";
        haveMeta = readBlueprintMeta(bp, m);
    } else {
        const std::filesystem::path p(bp);
        const std::string stem = p.stem().string();
        if (stem.rfind("bp_", 0) == 0) {
            haveMeta = readBlueprintMetaFile(
                (p.parent_path() / ("meta_" + stem.substr(3) + ".txt")).string(), m);
        }
    }
    if (haveMeta) {
        if (!explicitB) buckets = m.buckets;
        if (!explicitR) raiseCap = m.raiseCap;
        recall = m.recall;
        slim = m.slim;
    }
}

int cmdBlueprintDistill(int argc, char** argv) {
    const std::string bp = argStr(argc, argv, "--bp", "/Users/mingming/models/blueprint_v3");
    const std::string resume = argStr(argc, argv, "--resume", "/Users/mingming/models/stn2");
    const std::string outDir = argStr(argc, argv, "--out", "/Users/mingming/models/bp_distill_v1");
    const int games = argInt(argc, argv, "--games", 1200);
    const int holdout = argInt(argc, argv, "--holdout", 300);
    const int steps = argInt(argc, argv, "--steps", 400);
    const int minibatch = argInt(argc, argv, "--minibatch", 256);
    const float lr = static_cast<float>(argDouble(argc, argv, "--lr", 3e-4));
    // stn2 的 actor logits 实测高达 ±500（PPO 训出的极尖策略），直接做 KL 蒸馏
    // 时 softmax 完全饱和、梯度推不动（CE 30~280）。只把四个输出末层的尺度压到
    // 1/N，保留 LSTM/前层/信念头的全部表示，再监督微调即可（1.0 = 不缩放）。
    const float logitScale = static_cast<float>(argDouble(argc, argv, "--logit-scale", 0.01));
    const float mixRandom = static_cast<float>(argDouble(argc, argv, "--mix-random", 0.30));
    const float bpTemp = static_cast<float>(argDouble(argc, argv, "--bp-temp", 1.0));
    const uint64_t seed = argU64(argc, argv, "--seed", 20261107ULL);
    const bool evalOnly = hasFlag(argc, argv, "--eval-only");
    if (outDir.rfind("/tmp", 0) == 0) {
        std::printf("!! 纪律：蒸馏输出禁止写 /tmp，请用 --out 指到 ~/models\n");
        return 1;
    }
    const bool explicitB = argValue(argc, argv, "--buckets", nullptr) != nullptr;
    const bool explicitR = argValue(argc, argv, "--raise-cap", nullptr) != nullptr;
    int buckets = argInt(argc, argv, "--buckets", texas::kDefaultBuckets);
    int raiseCap = argInt(argc, argv, "--raise-cap", texas::kDefaultRaiseCap);
    int recall = 0;
    int slim = 0;
    std::string binPath;
    resolveBlueprintPath(bp, binPath, buckets, raiseCap, recall, slim, explicitB, explicitR);

    texas::RegretTable tab(texas::kAbstractStride);
    if (!tab.load(binPath)) {
        std::printf("!! 读不到 blueprint 表: %s\n", binPath.c_str());
        return 1;
    }
    texas::nn::ModelConfig mc;
    mc.hidden = argInt(argc, argv, "--hidden", 256);
    mc.lstmHidden = argInt(argc, argv, "--lstm", 128);
    texas::nn::Actor actors[2];
    texas::nn::Critic critics[2];
    for (int s = 0; s < 2; ++s) {
        actors[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        critics[s] = texas::nn::Critic(std::make_shared<texas::nn::CriticImpl>(mc));
    }
    if (!loadModels(actors, critics, resume)) {
        std::printf("!! 读不到基座模型: %s\n", resume.c_str());
        return 1;
    }
    if (!evalOnly && logitScale > 0.0f && logitScale < 1.0f) {
        auto shrink = [logitScale](torch::nn::Linear& lin) {
            torch::NoGradGuard ng;
            lin->weight.mul_(logitScale);
            if (lin->bias.defined()) lin->bias.mul_(logitScale);
        };
        for (int s = 0; s < 2; ++s) {
            shrink(actors[s]->head);
            shrink(actors[s]->auxAct2);
            shrink(actors[s]->oppAct2);
            shrink(actors[s]->rngAct2);
        }
        std::printf("  输出末层 logit 尺度 × %.3f（保留全部特征层）\n", logitScale);
    }

    texas::ppo::Config pcfg;
    pcfg.lr = lr;
    pcfg.minibatch = minibatch;
    texas::Learner learners[2] = {
        texas::Learner(actors[0], critics[0], pcfg, torch::kCPU),
        texas::Learner(actors[1], critics[1], pcfg, torch::kCPU),
    };

    std::printf("blueprint-distill: bp=%s (%u infosets) | 基座=%s | B=%d R=%d\n",
                binPath.c_str(), tab.size(), resume.c_str(), buckets, raiseCap);
    std::printf("  采样 %d 局训练 + %d 局留出, 混合随机 %.2f, 蓝图温度 %.2f, seed=%llu\n",
                games, holdout, mixRandom, bpTemp, static_cast<unsigned long long>(seed));
    std::fflush(stdout);
    DistillData trainData, holdData;
    generateDistillSamples(tab, games, seed, buckets, raiseCap, mixRandom, bpTemp, trainData,
                           recall, slim);
    generateDistillSamples(tab, holdout, seed + 7919ULL, buckets, raiseCap, mixRandom, bpTemp,
                           holdData, recall, slim);
    std::printf("  样本: 座位0 %zu / 座位1 %zu；留出 %zu\n", trainData.bySeat[0].size(),
                trainData.bySeat[1].size(), holdData.bySeat[0].size() + holdData.bySeat[1].size());
    if (trainData.bySeat[0].size() < 8 || trainData.bySeat[1].size() < 8) {
        std::printf("!! 样本不足（每座位至少 8 条）\n");
        return 1;
    }
    std::fflush(stdout);

    // --eval-only：只对已有检查点（--resume 指向的目录）重算保真读数，
    // 用于给不同蒸馏版本出一致口径的对照表（同一留出集、同一 seed）。
    if (evalOnly) {
        const FidelityResult f = measureFidelity(actors, holdData);
        std::printf("\nG2 保真（留出 %d 决策点, 仅评估）:\n", f.count);
        std::printf("  top-1 一致率 %.3f（并列容忍 %.3f）\n", f.top1, f.top1Tie);
        std::printf("  KL(target||actor) %.4f / H(target) %.4f\n", f.kl, f.ent);
        std::printf("  H(target) 是目标分布自身熵，KL 相对它看：占比 %.1f%%\n",
                    100.0 * f.kl / std::max(1e-6, f.ent));
        return 0;
    }

    const FidelityResult before = measureFidelity(actors, holdData);
    // 早停：distill 的每个 step 都会完整扫一遍数据（= 一个 epoch），步数给多了
    // 会直接过拟合训练局（v2 实测 1500 步反而比 400 步差）。因此按留出集 KL
    // 分块选优，保留最优快照；--patience 个分块无改善即停。
    const int evalEvery = std::max(1, argInt(argc, argv, "--eval-every", 50));
    const int patience = std::max(1, argInt(argc, argv, "--patience", 4));
    const std::string bestPath[2] = {outDir + "/.best0.pt", outDir + "/.best1.pt"};
    for (int s = 0; s < 2; ++s) {
        std::error_code ec;
        std::filesystem::create_directories(outDir, ec);
        FidelityResult base = measureFidelitySeat(actors[s], holdData.bySeat[s]);
        double bestKl = base.kl;
        int bestStep = 0, sinceBest = 0, done = 0;
        actors[s]->saveTo(bestPath[s]);
        std::printf("  座位 %d: 起点 KL %.4f / top-1 %.3f\n", s, base.kl, base.top1);
        std::fflush(stdout);
        while (done < steps) {
            const int chunk = std::min(evalEvery, steps - done);
            learners[s].distill(trainData.bySeat[s], chunk, 1.0f, 1.0f, minibatch, true);
            done += chunk;
            const FidelityResult f = measureFidelitySeat(actors[s], holdData.bySeat[s]);
            std::printf("  座位 %d @ %d 步: 留出 KL %.4f / top-1 %.3f%s\n", s, done, f.kl,
                        f.top1, f.kl < bestKl - 1e-4 ? "  *" : "");
            std::fflush(stdout);
            if (f.kl < bestKl - 1e-4) {
                bestKl = f.kl;
                bestStep = done;
                sinceBest = 0;
                actors[s]->saveTo(bestPath[s]);
            } else if (++sinceBest >= patience) {
                std::printf("  座位 %d: 早停于 %d 步（最优 %d 步, KL %.4f）\n", s, done,
                            bestStep, bestKl);
                break;
            }
        }
        actors[s]->loadFrom(bestPath[s]);
        std::printf("  座位 %d: 采用 %d 步快照, 留出 KL %.4f\n", s, bestStep, bestKl);
    }
    const FidelityResult after = measureFidelity(actors, holdData);
    saveModels(actors, critics, outDir);
    for (int s = 0; s < 2; ++s) {
        std::error_code ec;
        std::filesystem::remove(bestPath[s], ec);  // 只留正式检查点
    }
    std::printf("\nG2 保真（留出 %d 决策点）:\n", after.count);
    std::printf("  top-1 一致率 %.3f -> %.3f（并列容忍 %.3f）(门 ≥ 0.700: %s)\n", before.top1,
                after.top1, after.top1Tie, after.top1 >= 0.70 ? "PASS" : "FAIL");
    std::printf("  KL(target||actor) %.4f -> %.4f (门 ≤ 0.100: %s)\n", before.kl, after.kl,
                after.kl <= 0.10 ? "PASS" : "FAIL");
    std::printf("  H(target) %.4f -> %.4f；KL 占目标熵 %.1f%%\n", before.ent, after.ent,
                100.0 * after.kl / std::max(1e-6, after.ent));
    std::printf("  模型已写入 %s\n", outDir.c_str());
    std::printf("  内部锚点请用: ./build_rl/perfecttexas eval --resume %s --vs rule|random|pro|search|solid|mix --hands 8000 --temp 0.5\n",
                outDir.c_str());
    return 0;
}

texas::Move scriptMove(const std::string& vs, const texas::HUState& s, texas::Rng& rng) {
    if (vs == "random") return randomLegalMove(s, rng);
    if (vs == "pro") return texas::proAgentMove(s, rng);
    if (vs == "solid") return texas::solidAgentMove(s, rng);
    if (vs == "search") return texas::searchAgentMove(s, rng, 16, 3);
    if (vs == "tight") return texas::tightAgentMove(s, rng);
    return texas::ruleAgentMove(s, rng);
}

int cmdBlueprintEval(int argc, char** argv) {
    const std::string bp = argStr(argc, argv, "--bp", "/Users/mingming/models/blueprint_v3");
    const std::string resume = argStr(argc, argv, "--resume", "");
    const int hands = argInt(argc, argv, "--hands", 2000);
    const uint64_t seed = argU64(argc, argv, "--seed", 90001ULL);
    const std::string vs = argStr(argc, argv, "--vs", "rule");
    const bool explicitB = argValue(argc, argv, "--buckets", nullptr) != nullptr;
    const bool explicitR = argValue(argc, argv, "--raise-cap", nullptr) != nullptr;
    int buckets = argInt(argc, argv, "--buckets", texas::kDefaultBuckets);
    int raiseCap = argInt(argc, argv, "--raise-cap", texas::kDefaultRaiseCap);
    int recall = 0;
    int slim = 0;
    std::string binPath;
    resolveBlueprintPath(bp, binPath, buckets, raiseCap, recall, slim, explicitB, explicitR);
    texas::RegretTable tab(texas::kAbstractStride);
    if (!tab.load(binPath)) {
        std::printf("!! 读不到 blueprint 表: %s\n", binPath.c_str());
        return 1;
    }

    if (!resume.empty()) {
        // 蒸馏后 actor 的保真复核（不需要训练，只看留出一致率/KL）。
        texas::nn::ModelConfig mc;
        mc.hidden = argInt(argc, argv, "--hidden", 256);
        mc.lstmHidden = argInt(argc, argv, "--lstm", 128);
        texas::nn::Actor actors[2];
        for (int s = 0; s < 2; ++s)
            actors[s] = texas::nn::Actor(std::make_shared<texas::nn::ActorImpl>(mc));
        if (!loadActors(actors, resume)) {
            std::printf("!! 读不到 actors: %s\n", resume.c_str());
            return 1;
        }
        DistillData data;
        generateDistillSamples(tab, std::max(1, hands / 8), seed + 7919ULL, buckets, raiseCap,
                               0.30f, 1.0f, data, recall, slim);
        const FidelityResult f = measureFidelity(actors, data);
        std::printf("blueprint-eval 保真: %s | %d 决策点 | top-1 %.3f (门 0.700) | KL %.4f (门 0.100)\n",
                    resume.c_str(), f.count, f.top1, f.kl);
        return 0;
    }

    // blueprint 查表策略直接对打内置脚本基准（与 eval 的 nn::Actor 路径独立）。
    texas::Rng rng(seed);
    double sum = 0.0, sq = 0.0;
    int n = 0, trainedHits = 0, decisions = 0;
    for (int h = 0; h < hands; ++h) {
        texas::HUState s = texas::dealNewHand(rng);
        const int bpSeat = h & 1;  // 交替座位，抵消盲注位置偏差
        int steps = 0;
        while (!s.finished && steps++ < 80) {
            texas::Move mv;
            if (s.toAct == bpSeat) {
                float abs[texas::kAbstractStride] = {};
                if (texas::queryBlueprint(s, tab, buckets, raiseCap, abs, recall, slim)) ++trainedHits;
                ++decisions;
                mv = texas::sampleBlueprintMove(s, tab, buckets, raiseCap, rng, 1.0f, recall, slim);
            } else {
                mv = scriptMove(vs, s, rng);
            }
            texas::applyMove(s, mv, rng);
        }
        const double u = double(texas::payoff(s, bpSeat));
        sum += u;
        sq += u * u;
        ++n;
    }
    const double mean = sum / n;
    const double var = std::max(0.0, sq / n - mean * mean);
    const double se = std::sqrt(var / n);
    std::printf("blueprint-eval 查表策略 vs %s: %d 手, %+0.2f ± %.2f 筹码/手 "
                "(%+0.2f ± %.2f BB/手) | 查表命中 %.1f%% (%d/%d 决策)\n",
                vs.c_str(), n, mean, se, mean / 100.0, se / 100.0,
                decisions > 0 ? 100.0 * trainedHits / decisions : 0.0, trainedHits, decisions);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string cmd = argc > 1 ? argv[1] : "help";
    if (cmd == "train") return cmdTrain(argc, argv);
    if (cmd == "eval") return cmdEval(argc, argv);
    if (cmd == "match") return cmdMatch(argc, argv);
    if (cmd == "soup") return cmdSoup(argc, argv);
    if (cmd == "botmatch") return cmdBotMatch(argc, argv);
    if (cmd == "probe") return cmdProbe(argc, argv);
    if (cmd == "distill") return cmdDistill(argc, argv);
    if (cmd == "sanity") return cmdSanity(argc, argv);
    if (cmd == "critfit") return cmdCritFit(argc, argv);
    if (cmd == "policy-stats") return cmdPolicyStats(argc, argv);
    if (cmd == "bench") return cmdBench(argc, argv);
    if (cmd == "slumbot") return cmdSlumbot(argc, argv);
    if (cmd == "bench-device") return cmdBenchDevice(argc, argv);
    if (cmd == "slumbot-analyze") return cmdSlumbotAnalyze(argc, argv);
    if (cmd == "allin-audit") return cmdAllinAudit(argc, argv);
    if (cmd == "slumbot-merge") return cmdSlumbotMerge(argc, argv);
    if (cmd == "commit-audit") return cmdCommitAudit(argc, argv);
    if (cmd == "slumbot-decisions") return cmdSlumbotDecisions(argc, argv);
    if (cmd == "slumbot-policy-eval") return cmdSlumbotPolicyEval(argc, argv);
    if (cmd == "river-ranges") return cmdRiverRanges(argc, argv);
    if (cmd == "showdown") return cmdShowdown(argc, argv);
    if (cmd == "slumbot-fit") return cmdSlumbotFit(argc, argv);
    if (cmd == "blueprint-train") return cmdBlueprintTrain(argc, argv);
    if (cmd == "blueprint-br") return cmdBlueprintBr(argc, argv);
    if (cmd == "blueprint-distill") return cmdBlueprintDistill(argc, argv);
    if (cmd == "blueprint-eval") return cmdBlueprintEval(argc, argv);
    usage();
    return 0;
}
