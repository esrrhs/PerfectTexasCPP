#pragma once

// 单挑无限注德州（Heads-Up No-Limit Hold'em）规则引擎。
//
// 为强化学习自对局设计：State 是纯 POD、拷贝成本 ~120 字节，一局牌的全部
// 信息都在里面，不需要指针或堆分配。CFR 时代的多路/边池逻辑已经删掉——
// 单挑的结算只有"赢家通吃 / 平分"两种情形，代码少了三分之二，边界也少。
//
// 座位约定（单挑固定）：
//   0 = 小盲 SB（按钮位，翻牌前先行动，翻牌后后行动）
//   1 = 大盲 BB（翻牌前最后行动，翻牌后先行动）
//
// 筹码单位就是引擎的最小单位，盲注 1/2、起始 200（=100BB）。

#include <array>
#include <cstdint>
#include <random>
#include <string>

#include "texas/card.hpp"

namespace texas {

constexpr int kStartingStack = 200;
constexpr int kSmallBlind = 1;
constexpr int kBigBlind = 2;
constexpr int kMaxSizes = 5;     // 每街的尺寸档数
constexpr int kMaxRaises = 3;    // 每街最多 3 次主动加注（开池 / 3-bet / 4-bet）
constexpr int kMaxHist = 48;     // 一手牌的动作条数上限
constexpr int kNumActions = 8;   // 网络动作槽：fold / check-call / 5 档尺寸 / all-in

enum Street : uint8_t { ST_PREFLOP = 0, ST_FLOP = 1, ST_TURN = 2, ST_RIVER = 3 };
enum MoveKind : uint8_t {
    MK_FOLD = 0,
    MK_CHECK = 1,
    MK_CALL = 2,
    MK_RAISE = 3,
    MK_ALLIN = 4,
};

// 动作槽位映射（全局固定，与具体局面无关）：
//   0     fold
//   1     check / call
//   2..6  第 i 档尺寸（翻前 = 上次加注额的倍数，翻后 = 底池比例）
//   7     all-in
constexpr int kSlotFold = 0;
constexpr int kSlotCheckCall = 1;
constexpr int kSlotSize0 = 2;
constexpr int kSlotAllIn = 7;

inline int slotOfSize(int i) { return kSlotSize0 + i; }

struct Move {
    uint8_t kind = MK_CHECK;
    int16_t to = 0;  // 执行后本街的投入额
};

struct HistEntry {
    uint8_t seat = 0;
    uint8_t kind = MK_CHECK;
    uint8_t street = 0;
    int16_t to = 0;      // 本街投入额
    int16_t potAfter = 0;
};

struct LegalMove {
    int slot = 0;
    Move mv;
};

struct HUState {
    int16_t stack[2] = {kStartingStack, kStartingStack};  // 身后的筹码
    int16_t contrib[2] = {0, 0};                          // 本街已投入
    int16_t total[2] = {0, 0};                            // 整手牌已投入
    uint8_t hole[2][2] = {{0, 0}, {0, 0}};
    uint8_t board[5] = {0, 0, 0, 0, 0};
    uint8_t boardN = 0;
    uint8_t street = ST_PREFLOP;
    uint8_t toAct = 0;
    uint8_t lastAggressor = 255;
    uint8_t folded = 255;
    uint8_t allin[2] = {0, 0};
    uint8_t acted[2] = {0, 0};
    int16_t lastRaiseTo = 0;    // 跟注所需的投入额
    int16_t lastRaiseSize = 0;  // 上一次加注的增量，用于算最小加注
    uint8_t numRaises = 0;
    uint8_t finished = 0;
    uint8_t histLen = 0;
    HistEntry hist[kMaxHist];

    int pot() const { return int(total[0]) + int(total[1]); }
    int stackOf(int p) const { return stack[p]; }
    bool terminal() const { return finished != 0; }
};

using Rng = std::mt19937_64;

// 用指定底牌/公共牌构造状态（测试与配对评估用）。boardN = 0/3/4/5。
HUState makeState(const Card* hole0, const Card* hole1, const Card* board = nullptr,
                  int boardN = 0);

// 随机发牌开局。
HUState dealNewHand(Rng& rng);

// 当前行动者的跟注缺口。
inline int toCall(const HUState& s, int p) {
    int need = int(s.contrib[1 - p]) - int(s.contrib[p]);
    return need > 0 ? need : 0;
}

// 枚举当前局面的合法动作（含 slot）。返回条数。
int legalMoves(const HUState& s, LegalMove* out);

// 执行一个动作（内部会让机会节点发牌：本街结束就发下一条街）。
void applyMove(HUState& s, const Move& m, Rng& rng);

// 结算：玩家 p 的净收益（筹码）。终局前调用返回 0。
int payoff(const HUState& s, int p);

// 摊牌比牌，返回 0 / 1 / 2（2 表示平分）。非摊牌局面不合法。
int showdownWinner(const HUState& s);

// "开天眼"：双方底牌与公共牌全可见时的精确胜率（平局半胜）。
// 转牌/河牌精确枚举，翻牌前与翻牌用 iters 次抽样。
double perfectEquity(const HUState& s, int p, int iters = 64);

// 把动作转成可读文本，如 "fold" / "call 6" / "raise 18" / "allin 200"。
std::string moveToString(const HUState& s, const Move& m);

// 牌面字符串（测试 / 日志）。
std::string boardToString(const HUState& s);

}  // namespace texas
