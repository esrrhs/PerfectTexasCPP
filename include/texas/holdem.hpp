#pragma once

// 无限注德州扑克（No-Limit Texas Hold'em）规则引擎。
//
// 覆盖完整规则：52 张牌、四条街（翻牌前/翻牌/转牌/河牌）、无限注下注、
// all-in、边池分配、摊牌比牌。
//
// 支持 2~6 人。下注顺序按标准：翻牌前从大盲左手第一位开始（单挑时按钮/小盲先），
// 翻牌后从小盲之后开始。
//
// 说明：无限注的下注额是连续实数，必须做动作抽象才能求均衡，
// 抽象由 abstraction.hpp 提供，本文件只负责"合法动作是什么"。

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "card.hpp"

namespace texas {

struct Move {
    enum Kind : std::uint8_t {
        Fold = 0,       // 弃牌
        CheckCall = 1,  // 过牌 / 跟注
        Raise = 2,      // 下注 / 加注到 amount（该玩家的总投入）
    };
    Kind kind = CheckCall;
    int amount = 0;
};

// 动作抽象：一次性决定"每街给几档加注尺寸、各是多大"。
//
// 早先尺寸是写死在 actions() 里的，于是"哪种加注尺寸更好"这件事根本测不了——
// 一套二进制只能用一套动作集。做成参数之后可以训练多套动作集并互相对打。
//
// 上限由 kMaxActions 决定：弃牌 + 跟注占 2 个，all-in 占 1 个，最多剩 4 档尺寸。
// 翻前 4 档需要 7 个动作（有弃牌），翻后 4 档只需要 6 个（没有弃牌），所以 7 是
// "每街都能给 4 档"的最小上限。
struct ActionSpace {
    static constexpr int kMaxSizes = 4;

    // 翻牌前：加注到 (当前最大本街投入 × mul1000 / 1000)。
    // 默认 {2500, 4000} = 2.5 倍 / 4 倍，与早期的 maxRC + (maxRC*3)/2、maxRC*4
    // 逐位等价（对奇数 maxRC 也相等：x + floor(1.5x) == floor(2.5x)）。
    int preflopMul1000[kMaxSizes] = {2500, 4000, 2500, 2500};
    int preflopCount = 2;

    // 翻牌后：加注到 (当前最大 + 底池 × frac1000 / 1000)。
    // 默认 {500, 1000} = 半池 / 一池。
    int postflopFrac1000[kMaxSizes] = {500, 1000, 500, 500};
    int postflopCount = 2;

    // all-in 单独算一个动作，避免被比例尺寸吞掉
    bool includeAllIn = true;
};

// 两个动作集是否描述同一个博弈。只比 actions() 真正读的字段——直接 memcmp
// 会踩到结构体的填充字节，那几位是没有定义的。
inline bool sameActionSpace(const ActionSpace& x, const ActionSpace& y) {
    if (x.preflopCount != y.preflopCount || x.postflopCount != y.postflopCount) return false;
    if (x.includeAllIn != y.includeAllIn) return false;
    for (int i = 0; i < x.preflopCount; ++i) {
        if (x.preflopMul1000[i] != y.preflopMul1000[i]) return false;
    }
    for (int i = 0; i < x.postflopCount; ++i) {
        if (x.postflopFrac1000[i] != y.postflopFrac1000[i]) return false;
    }
    return true;
}

// 具名动作集。定义只有一份：名字→动作集与动作集→名字都读它。
//
// 这两张表各自漂移过一次，很安静：`--sizes three --load 某文件` 时动作集以文件为准
// （正确的行为），但打印用的是命令行那个字符串，于是输出写着 "three"、后面跟着
// "翻牌前 2 档"，看上去自相矛盾却没报错。往返测试（testActionSpaceNaming）钉住它。
struct ActionPreset {
    const char* name;
    ActionSpace space;
};

inline const std::vector<ActionPreset>& actionPresets() {
    static const std::vector<ActionPreset> table = [] {
        std::vector<ActionPreset> t;
        t.push_back({"default", ActionSpace{}});
        {
            ActionSpace s;
            s.preflopCount = 1;
            s.postflopCount = 1;
            t.push_back({"one", s});
        }
        {
            ActionSpace s;
            s.preflopCount = 3;
            s.preflopMul1000[0] = 2000;
            s.preflopMul1000[1] = 3000;
            s.preflopMul1000[2] = 5000;
            s.postflopCount = 3;
            s.postflopFrac1000[0] = 333;
            s.postflopFrac1000[1] = 750;
            s.postflopFrac1000[2] = 1500;
            t.push_back({"three", s});
        }
        {
            ActionSpace s;
            s.includeAllIn = false;
            t.push_back({"nojam", s});
        }
        {
            ActionSpace s;
            s.preflopMul1000[0] = 2000;
            s.preflopMul1000[1] = 3000;
            s.postflopFrac1000[0] = 333;
            s.postflopFrac1000[1] = 750;
            t.push_back({"small", s});
        }
        {
            ActionSpace s;
            s.preflopMul1000[0] = 3000;
            s.preflopMul1000[1] = 5000;
            s.postflopFrac1000[0] = 1000;
            s.postflopFrac1000[1] = 2000;
            t.push_back({"big", s});
        }
        // 六动作集的上限：弃 + 跟 + 3 档 + all-in = kMaxActions。
        // 前后各加一档，实测 +1.30 +- 0.22（见 README）。在动作数被 kMaxActions
        // 卡住之前，这是加大动作抽象能拿到的全部收益。
        {
            ActionSpace s;
            s.preflopCount = 3;
            s.preflopMul1000[0] = 3000;
            s.preflopMul1000[1] = 4000;
            s.preflopMul1000[2] = 5000;
            s.postflopCount = 3;
            s.postflopFrac1000[0] = 500;
            s.postflopFrac1000[1] = 1000;
            s.postflopFrac1000[2] = 2000;
            t.push_back({"wide", s});
        }
        // 翻前第四档。翻前没有牌力分桶（169 类全分辨率），动作粒度是唯一的抽象，
        // 所以这里加档还在赚钱；翻后加第四档相反——树炸 7 倍（111 万 → 781 万信息集）
        // 只换到 +0.48，等预算下不值。实测相对 wide +0.83~+1.01。
        {
            ActionSpace s;
            s.preflopCount = 4;
            s.preflopMul1000[0] = 2500;
            s.preflopMul1000[1] = 3000;
            s.preflopMul1000[2] = 4000;
            s.preflopMul1000[3] = 5000;
            s.postflopCount = 3;
            s.postflopFrac1000[0] = 500;
            s.postflopFrac1000[1] = 1000;
            s.postflopFrac1000[2] = 2000;
            t.push_back({"sharp", s});
        }
        return t;
    }();
    return table;
}

// 名字 → 动作集。返回 false 表示名字不认识（此时 out 是默认动作集）。
inline bool actionPreset(const std::string& name, ActionSpace& out) {
    for (const ActionPreset& p : actionPresets()) {
        if (name == p.name) {
            out = p.space;
            return true;
        }
    }
    out = ActionSpace{};
    return false;
}

// 动作集 → 名字。**按内容反查，不能回显用户输入**：读盘时动作集以文件里的值为准，
// 命令行那个字符串此刻已经不是实际在跑的东西了。认不出来就返回 "custom"。
inline const char* actionSpaceName(const ActionSpace& sp) {
    for (const ActionPreset& p : actionPresets()) {
        if (sameActionSpace(p.space, sp)) return p.name;
    }
    return "custom";
}

struct HoldemGame {
    static constexpr int kMaxPlayers = 6;
    static constexpr int kSmallBlind = 1;
    static constexpr int kBigBlind = 2;
    static constexpr int kBuyIn = 200;    // 100BB
    static constexpr int kMaxRaises = 3;  // 每条街最多 3 次加注（够打到 4-bet）
    static constexpr int kMaxActions = 7;

    struct State {
        int numPlayers = 2;
        Card hole[kMaxPlayers][2] = {};
        Card board[5] = {};
        int boardCount = 0;
        int street = 0;  // 0=preflop 1=flop 2=turn 3=river

        int contrib[kMaxPlayers] = {};
        int roundContrib[kMaxPlayers] = {};
        int raises = 0;
        int acts = 0;
        int player = 0;

        bool folded[kMaxPlayers] = {};
        bool allIn[kMaxPlayers] = {};

        int pendingDeal = 0;  // >0 表示待发公共牌张数（机会节点）
        bool finished = false;

        // 下注历史。早年这里用 std::string，结果每个节点 next() 都要深拷贝一次
        // （一次堆分配），而 infosetId 还要把整个字符串重新哈希一遍（O(长度)）。
        // MCCFR 每轮要经过十几个节点，这两项直接吃掉了大半吞吐。
        // 现在改成定长缓冲（拷贝退化成 memcpy）+ 增量哈希（O(1) 取键）。
        // 缓冲只服务于可读性/调试，信息集键以 histHash 为准，所以即使截断也不会串味。
        static constexpr int kHistCap = 64;
        char hist[kHistCap] = {};
        int histLen = 0;

        // 两个历史哈希，只在下注额这一维上不同：
        //   histHash      只含动作字符（'f' / 'c' / 'k' / 'r' / '/'）。加注额被抹掉。
        //   histSizedHash 在上面的基础上再混入每次加注后的本街投入额。
        // 为什么要同时维护：这两者对应两种信息划分，而"哪种更好"只能靠对打回答。
        // 让抽象挑一个（HoldemAbstraction::sizedHistory），就能在同一进程里训练两个
        // 求解器并用 headToHead 直接对打。多一个 uint64 的代价是每次动作一次乘加，
        // 在 63 ns/节点的量级上测不出来。
        std::uint64_t histHash = 0;
        std::uint64_t histSizedHash = 0;

        void pushHist(char c) {
            if (histLen < kHistCap - 1) hist[histLen++] = c;
            const std::uint64_t v = static_cast<std::uint64_t>(static_cast<unsigned char>(c));
            histHash ^= v;
            histHash *= 1099511628211ULL;
            histSizedHash ^= v;
            histSizedHash *= 1099511628211ULL;
        }

        // 加注额必须能被应对方看见。早先所有加注都只推一个 'r'，于是"加注到 5"、
        // "加注到 8" 和"加注到 200"产生完全相同的历史哈希 —— 应对方拿到的是**同一个
        // 信息集**，也就是它根本分不清面对的是小额加注还是 100BB 全下，只能对这两种
        // 截然不同的局面强行用同一套动作分布。这是比任何调参都严重的结构性缺陷。
        //
        // 这不是边角情况：枚举可达树，4442 个动作字符序列里有 3796 个（85.5%）背后
        // 藏着多个不同的筹码局面，平均每个藏着 11.1 个，最坏的一个藏 36 个。
        //
        // 混入的是加注后的本街投入。也可以混"这是第几个加注按钮"，两者对信息集的切分
        // 是等价的（尺寸由状态唯一决定，投入额与按钮号互相可推），选投入额是因为它
        // 不依赖动作集的定义——将来换掉下注尺寸表，这里不用跟着改。
        void mixRaiseSize(int roundContribAfter) {
            histSizedHash ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(roundContribAfter));
            histSizedHash *= 1099511628211ULL;
            histSizedHash ^= histSizedHash >> 29;
        }

        void setHist(const char* s) {
            histLen = 0;
            histHash = 0;
            histSizedHash = 0;
            for (; *s != '\0'; ++s) pushHist(*s);
        }

        std::string histString() const { return std::string(hist, hist + histLen); }
    };

    static State init(int numPlayers) {
        State s;
        s.numPlayers = numPlayers;
        s.contrib[0] = kSmallBlind;
        s.roundContrib[0] = kSmallBlind;
        s.contrib[1] = kBigBlind;
        s.roundContrib[1] = kBigBlind;
        // 翻牌前：单挑由按钮（小盲）先行动；多人由枪口位（大盲左手）先行动
        s.player = (numPlayers == 2) ? 0 : 2;
        return s;
    }

    static int maxRoundContrib(const State& s) {
        int m = 0;
        for (int i = 0; i < s.numPlayers; ++i) {
            if (!s.folded[i]) m = std::max(m, s.roundContrib[i]);
        }
        return m;
    }

    static int pot(const State& s) {
        int p = 0;
        for (int i = 0; i < s.numPlayers; ++i) p += s.contrib[i];
        return p;
    }

    static int numActive(const State& s) {
        int n = 0;
        for (int i = 0; i < s.numPlayers; ++i) {
            if (!s.folded[i]) ++n;
        }
        return n;
    }

    static int numCanAct(const State& s) {
        int n = 0;
        for (int i = 0; i < s.numPlayers; ++i) {
            if (!s.folded[i] && !s.allIn[i]) ++n;
        }
        return n;
    }

    static bool anyAllIn(const State& s) {
        for (int i = 0; i < s.numPlayers; ++i) {
            if (!s.folded[i] && s.allIn[i]) return true;
        }
        return false;
    }

    static bool isTerminal(const State& s) {
        if (numActive(s) <= 1) return true;
        return s.finished;
    }

    static bool needsChance(const State& s) { return s.pendingDeal > 0 && !s.finished; }

    static bool streetDone(const State& s) {
        int canAct = 0;
        int maxRC = 0;
        for (int i = 0; i < s.numPlayers; ++i) {
            if (s.folded[i]) continue;
            maxRC = std::max(maxRC, s.roundContrib[i]);
            if (!s.allIn[i]) ++canAct;
        }
        if (canAct == 0) return true;  // 全部 all-in，本街直接结束
        // 还能行动的玩家必须都已投入到位，否则还欠一个跟注/弃牌的决定
        for (int i = 0; i < s.numPlayers; ++i) {
            if (s.folded[i] || s.allIn[i]) continue;
            if (s.roundContrib[i] != maxRC) return false;
        }
        return s.acts >= canAct;
    }

    // ---------- 合法动作 ----------
    //
    // 加注尺寸（下注抽象）按街分开：
    //
    //   翻牌前：底池只有 3，用"底池比例"算出来的加注是加到 4/5，而下一个
    //           选项直接就是全下 200——中间 5 到 200 之间没有任何尺寸，模型
    //           想造大底池只能全下，策略会退化成"要么跛入要么推"。所以翻牌前
    //           按当前最大投入的倍数给尺寸（2.5x / 4x），开池、3-bet、4-bet
    //           自然形成阶梯。
    //
    //   翻牌后：按底池比例（半池 / 一池），这是标准下注尺度。
    //
    // 每条街允许 kMaxRaises 次加注，保证翻牌前的加注战（open → 3-bet →
    // 4-bet）真的能打到，而不是第二次加注之后就只能全下。
    static int actions(const State& s, Move* out, int maxOut, const ActionSpace& space) {
        if (isTerminal(s) || needsChance(s)) return 0;
        const int p = s.player;
        if (p < 0) return 0;
        const int maxRC = maxRoundContrib(s);
        const bool facing = s.roundContrib[p] < maxRC;
        const int stackLeft = kBuyIn - s.contrib[p];
        const int callNeed = maxRC - s.roundContrib[p];

        int n = 0;
        if (facing && n < maxOut) out[n++] = Move{Move::Fold, 0};
        if (n < maxOut) out[n++] = Move{Move::CheckCall, 0};

        if (s.raises < kMaxRaises && stackLeft > callNeed && n < maxOut) {
            // 以"本街投入"为单位给出目标，再换算成总投入
            auto emitRaise = [&](int roundTarget) {
                if (n >= maxOut) return;
                if (roundTarget <= maxRC) roundTarget = maxRC + kBigBlind;
                const int total = s.contrib[p] + (roundTarget - s.roundContrib[p]);
                if (total <= s.contrib[p]) return;
                const int amt = std::min(kBuyIn, total);
                // 去重：尺寸表一多，比例尺寸就容易在小底池里撞成同一个金额
                // （或者被 kBuyIn 一起截断）。两个一模一样的动作会让策略把质量
                // 分到两个看不出区别的槽里，白白浪费动作位，也让"每街 4 档"变成谎话。
                for (int i = 0; i < n; ++i) {
                    if (out[i].kind == Move::Raise && out[i].amount == amt) return;
                }
                out[n++] = Move{Move::Raise, amt};
            };

            if (s.street == 0) {
                for (int i = 0; i < space.preflopCount && i < ActionSpace::kMaxSizes && n < maxOut; ++i) {
                    emitRaise((maxRC * space.preflopMul1000[i]) / 1000);
                }
            } else {
                const int potNow = pot(s);
                for (int i = 0; i < space.postflopCount && i < ActionSpace::kMaxSizes && n < maxOut; ++i) {
                    emitRaise(maxRC + std::max(kBigBlind, (potNow * space.postflopFrac1000[i]) / 1000));
                }
            }
            if (space.includeAllIn && n < maxOut && s.contrib[p] < kBuyIn) {
                // 同一个去重：比例尺寸吃掉整个筹码时会正好落在 kBuyIn 上，
                // 不查就会和 all-in 变成两个一模一样的动作位。
                bool dup = false;
                for (int i = 0; i < n; ++i) {
                    if (out[i].kind == Move::Raise && out[i].amount == kBuyIn) dup = true;
                }
                if (!dup) out[n++] = Move{Move::Raise, kBuyIn};
            }
        }
        return n;
    }

    // 默认动作集，保持老调用点不变
    static int actions(const State& s, Move* out, int maxOut) {
        return actions(s, out, maxOut, ActionSpace{});
    }

    // ---------- 状态转移 ----------

    static State next(const State& s, const Move& m) {
        State t = s;
        const int p = s.player;
        const int maxRC = maxRoundContrib(s);

        if (m.kind == Move::Fold) {
            t.folded[p] = true;
            t.pushHist('f');
            t.player = -1;
            return t;
        }

        if (m.kind == Move::CheckCall) {
            const int delta = maxRC - s.roundContrib[p];
            t.contrib[p] += delta;
            t.roundContrib[p] = maxRC;
            ++t.acts;
            t.pushHist(delta > 0 ? 'c' : 'k');
        } else {
            const int targetTotal = std::min(m.amount, kBuyIn);
            const int delta = targetTotal - s.contrib[p];
            t.contrib[p] = targetTotal;
            t.roundContrib[p] = s.roundContrib[p] + delta;
            ++t.raises;
            ++t.acts;
            t.pushHist('r');
            t.mixRaiseSize(t.roundContrib[p]);
        }

        for (int i = 0; i < t.numPlayers; ++i) {
            if (!t.folded[i] && t.contrib[i] >= kBuyIn) t.allIn[i] = true;
        }

        t.player = -1;
        if (numActive(t) <= 1) return t;

        if (!streetDone(t)) {
            t.player = nextPlayer(t, p);
            return t;
        }

        // 本街结束
        for (int i = 0; i < t.numPlayers; ++i) t.roundContrib[i] = 0;
        t.raises = 0;
        t.acts = 0;
        t.pushHist('/');

        if (t.street == 3) {
            t.finished = true;  // 河牌打完，摊牌
            t.pendingDeal = 0;
        } else if (anyAllIn(t)) {
            t.pendingDeal = 5 - t.boardCount;  // 一次发完剩余公共牌
            t.street = 3;
        } else {
            t.street += 1;
            t.pendingDeal = (t.street == 1) ? 3 : 1;
        }
        return t;
    }

    static int nextPlayer(const State& s, int cur) {
        for (int i = 1; i <= s.numPlayers; ++i) {
            const int p = (cur + i) % s.numPlayers;
            if (!s.folded[p] && !s.allIn[p]) return p;
        }
        return -1;
    }

    // ---------- 发牌（机会节点） ----------

    // 未使用的牌（不含手牌和已发公共牌），按数组返回，避免发牌时堆分配
    static int buildDeck(const State& s, Card* out) {
        bool used[kNumCards] = {};
        for (int i = 0; i < s.numPlayers; ++i) {
            used[s.hole[i][0]] = true;
            used[s.hole[i][1]] = true;
        }
        for (int i = 0; i < s.boardCount; ++i) used[s.board[i]] = true;
        int n = 0;
        for (int c = 0; c < kNumCards; ++c) {
            if (!used[c]) out[n++] = c;
        }
        return n;
    }

    static std::vector<Card> remainingDeck(const State& s) {
        Card buf[kNumCards];
        const int n = buildDeck(s, buf);
        return std::vector<Card>(buf, buf + n);
    }

    // 机会节点收尾：这一街的牌发完了，决定下一个行动者或直接结束
    static void finishChance(State& s) {
        s.pendingDeal = 0;
        if (numCanAct(s) <= 1) {
            s.finished = true;
            s.player = -1;
        } else {
            s.player = nextPlayer(s, 0);  // 翻牌后由大盲先行动（单挑时小盲/按钮在后）
        }
    }

    template <class Rng>
    static void deal(State& s, Rng& rng) {
        // 发牌是每轮必经的热点，这里不碰堆：定长数组 + 部分 Fisher-Yates
        Card deck[kNumCards];
        const int dn = buildDeck(s, deck);
        const int need = s.pendingDeal;
        for (int i = 0; i < need && i < dn; ++i) {
            const int j = i + static_cast<int>(rng() % static_cast<std::uint64_t>(dn - i));
            std::swap(deck[i], deck[j]);
            s.board[s.boardCount++] = deck[i];
        }
        finishChance(s);
    }

    // 按"预先抽好的一整副公共牌"揭示接下来该发的牌，不重新抽。
    //
    // 这和逐街抽牌的联合分布完全一致（先均匀抽出 5 张、再逐街揭示，与逐街条件抽样
    // 等价），但整棵搜索树因此共用同一副公共牌——求解器可以据此缓存牌力桶。
    static void reveal(State& s, const Card* fullBoard) {
        const int need = s.pendingDeal;
        for (int i = 0; i < need && s.boardCount < 5; ++i) {
            s.board[s.boardCount] = fullBoard[s.boardCount];
            ++s.boardCount;
        }
        finishChance(s);
    }

    // ---------- 摊牌与边池 ----------

    static int handScore(const State& s, int p) {
        Card all[7];
        int n = 0;
        all[n++] = s.hole[p][0];
        all[n++] = s.hole[p][1];
        for (int i = 0; i < s.boardCount; ++i) all[n++] = s.board[i];
        if (n == 7) return evaluate7(all);
        if (n == 6) return evaluate6(all);
        if (n == 5) return evaluate5(all);
        return 0;
    }

    // 一手牌的牌力只取决于（底牌, 公共牌张数）—— 一轮 MCCFR 遍历里整副公共牌是
    // 固定的，同样的摊牌算力会被反复重算（实测每轮约 38 次 evaluate7）。
    // 这个缓存按 (玩家, 公共牌张数) 记住结果，把重算次数压到最多 6 次。
    // stamp 由调用方给（MCCFR 用迭代轮数），换了牌自然失效，不需要手动清理。
    struct ScoreCache {
        int stamp[kMaxPlayers][6];
        int score[kMaxPlayers][6];
        ScoreCache() {
            for (int p = 0; p < kMaxPlayers; ++p) {
                for (int b = 0; b < 6; ++b) {
                    stamp[p][b] = -1;
                    score[p][b] = 0;
                }
            }
        }
    };

    static int cachedHandScore(const State& s, int p, ScoreCache* cache, int stamp) {
        if (cache == nullptr || s.boardCount < 0 || s.boardCount > 5) return handScore(s, p);
        int* st = &cache->stamp[p][s.boardCount];
        if (*st == stamp) return cache->score[p][s.boardCount];
        const int v = handScore(s, p);
        *st = stamp;
        cache->score[p][s.boardCount] = v;
        return v;
    }

    // 每人净收益（零和）。按投入额分层结算边池。
    static void payouts(const State& s, double* out, ScoreCache* cache = nullptr,
                        int stamp = 0) {
        for (int i = 0; i < s.numPlayers; ++i) out[i] = 0.0;

        if (numActive(s) == 1) {
            const int total = pot(s);
            for (int i = 0; i < s.numPlayers; ++i) {
                if (s.folded[i]) {
                    out[i] = -static_cast<double>(s.contrib[i]);
                } else {
                    out[i] = static_cast<double>(total - s.contrib[i]);
                }
            }
            return;
        }

        int levels[kMaxPlayers];
        int levelCount = 0;
        for (int i = 0; i < s.numPlayers; ++i) {
            const int v = s.contrib[i];
            bool dup = false;
            for (int j = 0; j < levelCount; ++j) {
                if (levels[j] == v) dup = true;
            }
            if (!dup && v > 0) levels[levelCount++] = v;
        }
        std::sort(levels, levels + levelCount);

        int scores[kMaxPlayers];
        for (int i = 0; i < s.numPlayers; ++i) {
            scores[i] = s.folded[i] ? -1 : cachedHandScore(s, i, cache, stamp);
        }

        int prev = 0;
        for (int li = 0; li < levelCount; ++li) {
            const int lv = levels[li];
            const int slice = lv - prev;
            if (slice <= 0) continue;

            int potSlice = 0;
            for (int i = 0; i < s.numPlayers; ++i) {
                if (s.contrib[i] >= lv) potSlice += slice;
            }

            int bestScore = -1;
            for (int i = 0; i < s.numPlayers; ++i) {
                if (s.contrib[i] >= lv && !s.folded[i]) {
                    bestScore = std::max(bestScore, scores[i]);
                }
            }
            if (bestScore < 0) {
                prev = lv;
                continue;
            }
            int nw = 0;
            for (int i = 0; i < s.numPlayers; ++i) {
                if (s.contrib[i] >= lv && !s.folded[i] && scores[i] == bestScore) ++nw;
            }
            const double share = static_cast<double>(potSlice) / nw;
            for (int i = 0; i < s.numPlayers; ++i) {
                if (s.contrib[i] >= lv && !s.folded[i] && scores[i] == bestScore) out[i] += share;
            }
            prev = lv;
        }

        for (int i = 0; i < s.numPlayers; ++i) out[i] -= static_cast<double>(s.contrib[i]);
    }

    static double utilityFrom(const State& s, int player, ScoreCache* cache = nullptr,
                              int stamp = 0) {
        double pay[kMaxPlayers];
        payouts(s, pay, cache, stamp);
        return pay[player];
    }
};

}  // namespace texas
