#pragma once

// 转牌子博弈 CFR+ 求解器（README 第八节"下一步"第 1 条：深度受限求解）。
//
// 转牌与河牌的区别：公共牌还差一张，本街打完后还要发河牌、还有一条街的
// 下注。叶子有两种：
//   1. check-down（默认）：值(i) = Σ_j 对手可达权重(j) · E_河牌[摊牌收益]。
//      河牌枚举精确（≤46 张），≥1 人全下时整个叶子精确；双方都有筹码时
//      忽略河牌下注街（已知近似）。
//   2. critic 价值头（--turn-cfr-critic-leaf）：值(i) = 发河牌后当前局面的
//      critic 价值（双方底牌给定 + 河牌采样）。critic 是开天眼价值网
//      （训练目标 = 终局回报 + 零均值稠密奖励，所以是金钱价值的无偏估计），
//      单位换算成筹码；hero 组合按牌力百分位分桶、每桶取代表组合前向。
//
// 摊牌矩阵复用 rivercfr 的"牌力排序 + 前缀和 + 单牌小前缀和"O(1) blocker
// 查询。整棵树的数学与 rivercfr 同一套口径：CFR+、反事实值不逐节点归一化、
// 根聚合做条件还原、brGap = BR0 + BR1 − 底池 ≥ 0。

#include <array>
#include <vector>

#include "texas/hu.hpp"
#include "texas/nn.hpp"

namespace texas {

constexpr int kTurnComboCount = 48 * 47 / 2;  // 1128

// 转牌手牌空间（4 张公共牌之外的所有两两组合）。河牌未知，所以不存 rank
// ——叶子求值按每张河牌分别排序。
struct TurnCombo {
    Card c0 = 0;
    Card c1 = 0;
};

std::vector<TurnCombo> turnCombos(const HUState& s);

struct TurnCfrConfig {
    int iters = 60;   // 迭代轮数（每轮两个座位各更新一次）
    int maxRaises = 1;  // 首次下注之后还允许的加注次数
    std::array<double, 3> betFracs = {0.5, 1.0, 2.0};
    bool allin = true;    // 额外加一个全下尺寸
    bool blockers = true; // 摊牌矩阵扣掉双方手牌冲突（精确）
};

struct TurnRootAction {
    int slot = -1;
    Move mv;
    double ev = 0.0;    // 该动作的 EV（筹码，相对当前节点；弃牌 = 0）
    double prob = 0.0;  // 平均策略里该动作的概率
};

struct TurnCfrResult {
    std::vector<TurnRootAction> root;
    double rootValue = 0.0;
    double brGap = 0.0;  // BR0 + BR1 − 底池（筹码），越小越接近均衡
    double br0 = 0.0;
    double br1 = 0.0;
    int nodes = 0;
    int combos = 0;
    bool valid = false;
};

// 求解转牌子博弈。oppRange / ownRange 与 turnCombos(s) 一一对应（空 = 均匀）；
// oppRange 会按 hero 的实际底牌扣掉冲突组合。
TurnCfrResult turnCfrSolve(const HUState& s, int seat,
                           const std::vector<float>& oppRange = {},
                           const std::vector<float>& ownRange = {},
                           const TurnCfrConfig& cfg = {});

// 部署用：在转牌决策点求解并返回 EV 最大的合法动作。非转牌 / 求解失败返回
// false，调用方应回退到网络策略。
bool turnCfrMove(const HUState& s, const TurnCfrConfig& cfg, Move& out,
                 const std::vector<float>& oppRange = {},
                 const std::vector<float>& ownRange = {},
                 TurnCfrResult* detail = nullptr);

}  // namespace texas
