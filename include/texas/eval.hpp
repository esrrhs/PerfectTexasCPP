#pragma once

// 强度评估。零和对称游戏里"自对局恒为 0"，所以必须有固定基准：
//   规则机器人（牌力百分位 + 底池赔率）—— 相当于 PerfectDou 里 DouZero 的角色
//   均匀随机 —— 只做下限标尺
// 每个基准跑两遍（模型坐小盲 / 坐大盲），合并成模型每手净收益。

#include <cstdint>
#include <vector>

#include "texas/nn.hpp"

namespace texas {

struct EvalResult {
    int hands = 0;
    double chipsPerHand = 0.0;
    double stderr = 0.0;
    double bbPer100 = 0.0;  // 每百手大盲数
};

EvalResult evalVsRule(const nn::Actor actors[2], int games, uint64_t seed, int equityIters = 24);
EvalResult evalVsRandom(const nn::Actor actors[2], int games, uint64_t seed, int equityIters = 24);
EvalResult evalSelfPlay(const nn::Actor actors[2], int games, uint64_t seed, int equityIters = 24);

}  // namespace texas
