#pragma once

// 特征工程：把"行为 + 底牌"编码成网络输入。
//
// 这是和 CFR 时代最大的区别。CFR 靠一个信息集键（街 + 牌力桶 + 下注历史哈希）
// 查表；这里全部塌成定长向量，交给网络自己学：
//
//   节点特征（行为者可见，388 维）
//     自己的两张底牌      2 x 52 = 104
//     公共牌              5 x 52 = 260
//     标量                24 个（底池、筹码、跟注额、位置、牌力百分位…）
//   历史序列（12 步 x 12 维，喂 LSTM）
//     每一步是一次动作：谁、哪条街、动作类型、金额、动作后的底池
//   开天眼特征（仅 critic，108 维）
//     对手两张底牌 104 + 完美信息胜率 / 底池赔率 / 边际 4
//   动作特征（16 维）
//     动作类型 one-hot + 金额相对底池 / 盲注 + 动作后的底池与筹码

#include <array>
#include <cstdint>

#include "texas/hu.hpp"

namespace texas {

constexpr int kHoleDim = 2 * kNumCards;    // 104
constexpr int kBoardDim = 5 * kNumCards;   // 260
constexpr int kNodeScalars = 24;
constexpr int kNodeInput = kHoleDim + kBoardDim + kNodeScalars;  // 388
constexpr int kExtraScalars = 4;
constexpr int kExtraInput = kHoleDim + kExtraScalars;            // 108
constexpr int kHistSteps = 12;
constexpr int kHistDim = 12;
constexpr int kSeqInput = kHistSteps * kHistDim;                 // 144
constexpr int kActionInput = 16;

struct Obs {
    std::array<float, kNodeInput> node{};
    std::array<float, kExtraInput> extra{};
    std::array<float, kSeqInput> seq{};
    float equityPerfect = 0.0f;  // 开天眼胜率，稠密奖励与统计复用
};

// 编码当前行动者的观测（node / extra / seq）。eqActing 是"开天眼"胜率，
// 由调用方传入以便复用（底池不变时胜率不变，一条街只算一次）。
void encodeObservation(const HUState& s, Obs& o, float eqActing);
void encodeObservation(const HUState& s, Obs& o, int equityIters = 64);

// 只编码开天眼胜率（不算整套特征时用，如稠密奖励）。
float oracleEquity(const HUState& s, int seat, int iters = 64);

// 编码一个合法动作的特征。
void encodeActionFeature(const HUState& s, const LegalMove& lm,
                         std::array<float, kActionInput>& out);

// 对手（非行动者）的牌力百分位，0~1。翻牌前用 169 类胜率表。
float handStrength(const Card* hole, const Card* board, int boardN);

}  // namespace texas
