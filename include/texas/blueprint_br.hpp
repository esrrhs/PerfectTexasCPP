#pragma once

// 离线 CFR blueprint 的抽样最优应对（BR）评估，G1 门的工具。
// 设计口径见 docs/CFR_BLUEPRINT_PLAN.md §3。
//
// 抽样公共状态（BR 座位底牌 + 完整 5 张公共牌 runout），对手底牌用 combo reach
// 向量表示；在抽象下注树上全宽遍历：BR 节点取 max_a，对手节点按各 combo 的
// blueprint 平均策略概率分流。因为对手策略只依赖牌力桶，reach 按桶聚合即可，
// 街切换时再按新街的桶重新展开/聚合（imperfect recall 的桶序列）。
//
// 注意这是「抽象内 BR」：BR 只能选抽象动作，且不支持跨街 perfect recall，
// 因此是可利用度的下界；但足以做「随迭代是否下降」和 0.5 BB/手 的门控。

#include <cstdint>

#include "texas/blueprint.hpp"

namespace texas {

struct BrConfig {
    int samples = 2000;         // 抽样数（每个抽样可对两个座位各算一次）
    uint64_t seed = 20261107;
    int buckets = kDefaultBuckets;
    int raiseCap = kDefaultRaiseCap;
    int recall = 0;             // 1 = 与压缩完美召回训练的表对齐（每街 2 bit）
    int slim = 0;               // 1 = 与瘦 key 表对齐（底池桶/跨街历史不入 key）
    bool bothSeats = true;      // 每抽样同时算 BR0/BR1（方差低，成本 2×）
    bool selfPlayCheck = false; // 诊断：BR 座位也按蓝图策略行动（不再是 max），
                                // 双座位收益之和应接近 0，用来校验遍历/结算自洽
    bool byStreet = false;      // 额外按「决策街」分解可利用度（每节点多一次查表）
    uint64_t maxNodesPerSample = 0;  // 0 = 不限；保护性预算
};

struct BrResult {
    double br0 = 0.0;           // 座位 0 的最优应对收益，筹码/手
    double br1 = 0.0;
    double se0 = 0.0;
    double se1 = 0.0;
    double exploitChips = 0.0;  // br0 + br1
    double exploitBb = 0.0;     // ÷100
    double seBb = 0.0;
    // byStreet=true 时有效：可利用度按决策街分解（预/翻/转/河，BB/手）
    double exploitByStreetBb[4] = {0.0, 0.0, 0.0, 0.0};
    uint64_t samples = 0;
    uint64_t nodes = 0;
    bool valid = false;
};

BrResult runBlueprintBR(const RegretTable& tab, const BrConfig& cfg);

}  // namespace texas
