#pragma once

// 河牌子博弈 CFR+ 求解器（README #105 指定的主线）。
//
// 河牌是最后一条街：公共牌固定、后面不再发牌，所以整条街可以精确建模成一个
// 两人零和的不完全信息博弈：
//   - 双方的手牌空间 = 公共牌之外 47 张牌的所有两两组合，共 C(47,2) = 1081 个；
//   - 终局摊牌的收益只取决于两手牌的强弱（evaluate7 分数）和双方从当前节点
//     起的追加投入，没有运气成分；
//   - 动作 = 弃 / 过牌-跟注 / 几个底池比例的下注-加注（钳到全下）。
//
// 用 CFR+（regret matching+、交替更新、线性加权平均策略）求解，输出**根节点**
// 每个动作的 EV 与平均策略。求解只依赖局面与双方范围，不依赖网络，所以既能
// 当部署期的逐决策算子，也能当教师标签。
//
// 与 README #100~#104 的一步前瞻搜索的区别：那里的"对手"是一个具体脚本或
// 网络，每个样本一条续演线；这里的对手是一个**范围**，它在这棵下注树上按
// 均衡回应，所以诈唬-抓诈唬、blocker、尺寸选择是同一次求解出来的。

#include <array>
#include <vector>

#include "texas/hu.hpp"

namespace texas {

constexpr int kRiverComboCount = 47 * 46 / 2;  // 1081

struct RiverCombo {
    Card c0 = 0;
    Card c1 = 0;
    int rank = 0;  // evaluate7 分数：牌面固定时唯一决定摊牌胜负
};

// 枚举河牌手牌空间（公共牌之外的所有两两组合）。s.boardN 必须等于 5。
std::vector<RiverCombo> riverCombos(const HUState& s);

struct RiverCfrConfig {
    int iters = 120;   // 迭代轮数（每轮两个座位各更新一次）
    int maxRaises = 1; // 首次下注之后还允许的加注次数（1 = 下注→加注→跟/弃）
    // 下注/加注尺寸（相对底池）。去重后生效；2.0 提供超池尺寸。
    std::array<double, 3> betFracs = {0.5, 1.0, 2.0};
    bool allin = true;      // 额外加一个全下尺寸
    bool blockers = true;   // 摊牌矩阵扣掉双方手牌冲突（精确，代价是预处理）
};

// 根节点的一个动作：slot 只作参考（-1 = 求解器自造尺寸），mv 可直接执行。
struct RiverRootAction {
    int slot = -1;
    Move mv;
    double ev = 0.0;    // 该动作的 EV（筹码，相对当前节点；弃牌 = 0）
    double prob = 0.0;  // 平均策略里该动作的概率
};

struct RiverCfrResult {
    std::vector<RiverCombo> combos;
    std::vector<RiverRootAction> root;
    double rootValue = 0.0;  // 平均策略的根 EV
    double brGap = 0.0;      // BR0 + BR1 − 底池：可利用度上界（筹码），越小越接近均衡
    double br0 = 0.0;        // 座位 0 的最优反应值（诊断）
    double br1 = 0.0;        // 座位 1 的最优反应值（诊断）
    int nodes = 0;
    bool valid = false;
};

// 求解。oppRange / ownRange 是 1081 维权重（空 = 均匀）；内部做归一化。
// oppRange 会按 hero 的实际底牌扣掉冲突组合（对手不可能拿着我们的牌）。
// hero = 要出策略的座位（= s.toAct）。
RiverCfrResult riverCfrSolve(const HUState& s, int seat,
                             const std::vector<float>& oppRange = {},
                             const std::vector<float>& ownRange = {},
                             const RiverCfrConfig& cfg = {});

// 部署用：在河牌决策点求解并返回 EV 最大的合法动作。非河牌 / 求解失败返回
// false，调用方应回退到网络策略。oppRange / ownRange 是 1081 维范围权重
// （空 = 均匀）；ownRange 必须是**我们自己的**范围——均衡策略取决于双方范围，
// 用均匀范围会让求解器以为我们手里全是中等牌，从而系统性偏向小注/过牌。
// 给的范围会按我们实际底牌扣掉冲突组合。detail 非空时回填完整结果。
bool riverCfrMove(const HUState& s, const RiverCfrConfig& cfg, Move& out,
                  const std::vector<float>& oppRange = {},
                  const std::vector<float>& ownRange = {},
                  RiverCfrResult* detail = nullptr);

}  // namespace texas
