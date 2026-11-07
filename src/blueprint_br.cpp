#include "texas/blueprint_br.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

#include "texas/card.hpp"
#include "texas/strength.hpp"

namespace texas {
namespace {

// 桶数上界：翻前 169 类，翻后最多 cfg.buckets。固定 256 是为了递归里用栈数组，
// 避免每个节点做一次堆分配（单次抽样约 2e5 节点）。
constexpr int kMaxGroups = 256;
using Reach = std::array<float, kMaxGroups>;

struct Combo {
    int c0 = 0;
    int c1 = 0;
};

struct BrCtx {
    const RegretTable* tab = nullptr;
    int buckets = kDefaultBuckets;
    int raiseCap = kDefaultRaiseCap;
    int brSeat = 0;
    int brBucket[4] = {0, 0, 0, 0};  // 自洽性检查用：BR 固定底牌在四条街的桶
    bool policyMode = false;
    Rng* rng = nullptr;
    std::vector<Combo> combos;
    std::vector<int> bucketOf[4];    // [street][combo]
    std::vector<int> groupCount[4];  // [street][bucket]
    std::vector<int> winCount[4], loseCount[4], tieCount[4];
    std::vector<char> valid;
    uint64_t nodes = 0;
    uint64_t nodeBudget = 0;
    bool truncated = false;
};

void buildCombos(BrCtx& c) {
    c.combos.clear();
    c.combos.reserve(kNumCards * (kNumCards - 1) / 2);
    for (int a = 0; a < kNumCards; ++a)
        for (int b = a + 1; b < kNumCards; ++b) c.combos.push_back({a, b});
}

// 为一个抽样预计算：合法 combo、四条街各自的牌力桶、摊牌胜负（用完整 5 张公共牌）。
void prepareSample(BrCtx& c, const HUState& root, int brSeat) {
    c.brSeat = brSeat;
    bool blocked[kNumCards] = {};
    blocked[root.hole[brSeat][0]] = true;
    blocked[root.hole[brSeat][1]] = true;
    Card board[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 5; ++i) {
        board[i] = Card(root.board[i]);
        blocked[root.board[i]] = true;
    }

    const size_t n = c.combos.size();
    c.valid.assign(n, 0);
    for (int st = 0; st < 4; ++st) {
        c.bucketOf[st].assign(n, -1);
        c.groupCount[st].assign(kMaxGroups, 0);
        c.winCount[st].assign(kMaxGroups, 0);
        c.loseCount[st].assign(kMaxGroups, 0);
        c.tieCount[st].assign(kMaxGroups, 0);
    }

    Card hero7[7], opp7[7];
    hero7[0] = Card(root.hole[brSeat][0]);
    hero7[1] = Card(root.hole[brSeat][1]);
    for (int i = 0; i < 5; ++i) {
        hero7[2 + i] = board[i];
        opp7[2 + i] = board[i];
    }
    const int heroScore = evaluate7(hero7);
    StrengthModel& sm = StrengthModel::instance();
    c.brBucket[ST_PREFLOP] = StrengthModel::preflopClass(hero7[0], hero7[1]);
    for (int st = ST_FLOP; st <= ST_RIVER; ++st) {
        const int boardN = (st == ST_FLOP) ? 3 : (st == ST_TURN ? 4 : 5);
        c.brBucket[st] = sm.postflopBucket(hero7, board, boardN, c.buckets);
    }

    int totalValid = 0;
    for (size_t i = 0; i < n; ++i) {
        const int c0 = c.combos[i].c0;
        const int c1 = c.combos[i].c1;
        if (blocked[c0] || blocked[c1]) continue;
        c.valid[i] = 1;
        ++totalValid;

        opp7[0] = c0;
        opp7[1] = c1;
        const int oppScore = evaluate7(opp7);
        const int res = heroScore > oppScore ? 1 : (heroScore < oppScore ? -1 : 0);

        const Card hole[2] = {c0, c1};
        for (int st = 0; st < 4; ++st) {
            int b = 0;
            if (st == ST_PREFLOP) {
                b = StrengthModel::preflopClass(c0, c1);
            } else {
                const int boardN = (st == ST_FLOP) ? 3 : (st == ST_TURN ? 4 : 5);
                b = sm.postflopBucket(hole, board, boardN, c.buckets);
            }
            if (b < 0 || b >= kMaxGroups) b = kMaxGroups - 1;
            c.bucketOf[st][i] = b;
            ++c.groupCount[st][b];
            if (res > 0) ++c.winCount[st][b];
            else if (res < 0) ++c.loseCount[st][b];
            else ++c.tieCount[st][b];
        }
    }
    (void)totalValid;
}

Reach initialReach(const BrCtx& c) {
    Reach r{};
    int total = 0;
    for (int b = 0; b < kMaxGroups; ++b) total += c.groupCount[ST_PREFLOP][b];
    if (total <= 0) return r;
    for (int b = 0; b < kMaxGroups; ++b)
        r[b] = static_cast<float>(c.groupCount[ST_PREFLOP][b]) / static_cast<float>(total);
    return r;
}

// 街切换：牌力桶会变（翻前 169 类 -> 翻后 8 桶），把按旧桶聚合的 reach 展开到
// 每个 combo，再按新街的桶重新聚合。同一旧桶内的 combo reach 恒相等，所以只需
// 一次除法即可展开；质量守恒，且与训练侧 imperfect-recall 的桶切换一致。
Reach remapStreet(const BrCtx& c, const Reach& gr, int oldSt, int newSt) {
    Reach out{};
    for (size_t i = 0; i < c.combos.size(); ++i) {
        if (!c.valid[i]) continue;
        const int ob = c.bucketOf[oldSt][i];
        const int cnt = c.groupCount[oldSt][ob];
        if (gr[ob] <= 0.0f || cnt <= 0) continue;
        const float perCombo = gr[ob] / static_cast<float>(cnt);
        out[c.bucketOf[newSt][i]] += perCombo;
    }
    return out;
}

double terminalValue(const HUState& s, const Reach& gr, const BrCtx& c) {
    const int pot = s.pot();
    const int tb = s.total[c.brSeat];
    if (s.folded != 255) {
        // 弃牌结算与对手牌无关，直接乘总到达质量。
        const double u = (s.folded == c.brSeat) ? -double(tb) : double(pot - tb);
        double mass = 0.0;
        for (int b = 0; b < kMaxGroups; ++b) mass += gr[b];
        return mass * u;
    }
    // 摊牌：平局时奇数筹码给 0 号位（与 HUState::payoff 一致）。
    const double tieChips = (c.brSeat == 0) ? double(pot - pot / 2) : double(pot / 2);
    const int st = s.street;
    double v = 0.0;
    for (int b = 0; b < kMaxGroups; ++b) {
        if (gr[b] <= 0.0f) continue;
        const int cnt = c.groupCount[st][b];
        if (cnt <= 0) continue;
        const double avg = (c.winCount[st][b] * double(pot - tb)
                            + c.loseCount[st][b] * (-double(tb))
                            + c.tieCount[st][b] * (tieChips - double(tb)))
                           / static_cast<double>(cnt);
        v += double(gr[b]) * avg;
    }
    return v;
}

double brWalk(const HUState& s, const Reach& gr, const uint8_t* seq, int seqLen, BrCtx& c) {
    ++c.nodes;
    if (s.finished) return terminalValue(s, gr, c);
    if (c.nodeBudget && c.nodes > c.nodeBudget) {
        c.truncated = true;
        return terminalValue(s, gr, c);
    }

    AbstractMove am[kAbstractStride];
    const int n = legalAbstract(s, c.raiseCap, am);
    if (n <= 0) return terminalValue(s, gr, c);
    bool legal[kAbstractStride] = {};
    for (int i = 0; i < n; ++i) legal[am[i].id] = true;

    const int p = s.toAct;
    const bool takeMax = (p == c.brSeat) && !c.policyMode;
    // 自洽性检查：BR 座位也按蓝图策略走，需要查自己固定底牌对应的桶。
    float brSig[kAbstractStride] = {};
    if (c.policyMode && p == c.brSeat) {
        const uint64_t key = packInfoset(s.street, s.pot(), c.brBucket[s.street], seq, seqLen,
                                         s.toAct);
        const int64_t idx = c.tab->find(key);
        if (idx >= 0) {
            c.tab->averageStrategy(static_cast<uint32_t>(idx), legal, brSig);
        } else {
            int cnt = 0;
            for (int a = 0; a < kAbstractStride; ++a) if (legal[a]) ++cnt;
            for (int a = 0; a < kAbstractStride; ++a)
                brSig[a] = (legal[a] && cnt > 0) ? 1.0f / cnt : 0.0f;
        }
    }
    // 对手节点：每个有质量的桶查一次表，拿到该桶的完整动作分布。查表次数与桶数
    // 成正比（翻后仅 8 次），而不是 combo 数 × 动作数。
    float sig[kMaxGroups][kAbstractStride] = {};
    if (p != c.brSeat) {
        for (int b = 0; b < kMaxGroups; ++b) {
            if (gr[b] <= 0.0f) continue;
            const uint64_t key = packInfoset(s.street, s.pot(), b, seq, seqLen, s.toAct);
            const int64_t idx = c.tab->find(key);
            if (idx >= 0) {
                c.tab->averageStrategy(static_cast<uint32_t>(idx), legal, sig[b]);
            } else {
                int cnt = 0;
                for (int a = 0; a < kAbstractStride; ++a) if (legal[a]) ++cnt;
                for (int a = 0; a < kAbstractStride; ++a)
                    sig[b][a] = (legal[a] && cnt > 0) ? 1.0f / cnt : 0.0f;
            }
        }
    }

    double ret = takeMax ? -1e18 : 0.0;
    for (int ai = 0; ai < n; ++ai) {
        const int id = am[ai].id;
        Reach childGr = gr;
        if (p != c.brSeat) {
            for (int b = 0; b < kMaxGroups; ++b) childGr[b] *= sig[b][id];
        }

        HUState child = s;
        applyMove(child, am[ai].mv, *c.rng);
        uint8_t childSeq[kMaxStreetSeq] = {};
        int childLen = 0;
        if (child.street == s.street) {
            for (int k = 0; k < seqLen; ++k) childSeq[k] = seq[k];
            childLen = seqLen;
            if (childLen < kMaxStreetSeq) childSeq[childLen++] = static_cast<uint8_t>(id);
        }
        if (child.street != s.street) childGr = remapStreet(c, childGr, s.street, child.street);

        const double v = brWalk(child, childGr, childSeq, childLen, c);
        if (takeMax) ret = std::max(ret, v);
        else if (c.policyMode && p == c.brSeat) ret += double(brSig[id]) * v;
        else ret += v;
    }
    return ret;
}

}  // namespace

BrResult runBlueprintBR(const RegretTable& tab, const BrConfig& cfg) {
    BrResult r;
    if (cfg.samples <= 0 || tab.stride() != kAbstractStride) return r;

    Rng rng(cfg.seed);
    BrCtx c;
    c.tab = &tab;
    c.buckets = cfg.buckets;
    c.raiseCap = cfg.raiseCap;
    c.rng = &rng;
    c.nodeBudget = cfg.maxNodesPerSample;
    c.policyMode = cfg.selfPlayCheck;
    buildCombos(c);

    double sum0 = 0.0, sq0 = 0.0, sum1 = 0.0, sq1 = 0.0;
    int n0 = 0, n1 = 0;
    for (int sample = 0; sample < cfg.samples; ++sample) {
        const HUState root = dealNewHand(rng);
        for (int seat = 0; seat < 2; ++seat) {
            if (!cfg.bothSeats && seat != (sample & 1)) continue;
            c.nodes = 0;
            c.truncated = false;
            prepareSample(c, root, seat);
            const Reach gr = initialReach(c);
            uint8_t seq[kMaxStreetSeq] = {};
            const double v = brWalk(root, gr, seq, 0, c);
            r.nodes += c.nodes;
            if (seat == 0) { sum0 += v; sq0 += v * v; ++n0; }
            else { sum1 += v; sq1 += v * v; ++n1; }
        }
        if ((sample + 1) % 100 == 0) {
            std::printf("  blueprint-br: %d/%d 抽样, %.1f M 节点\n", sample + 1, cfg.samples,
                        double(r.nodes) / 1e6);
            std::fflush(stdout);
        }
    }
    if (n0 == 0 || n1 == 0) return r;

    r.br0 = sum0 / n0;
    r.br1 = sum1 / n1;
    const double var0 = std::max(0.0, sq0 / n0 - r.br0 * r.br0);
    const double var1 = std::max(0.0, sq1 / n1 - r.br1 * r.br1);
    r.se0 = std::sqrt(var0 / n0);
    r.se1 = std::sqrt(var1 / n1);
    r.exploitChips = r.br0 + r.br1;
    r.exploitBb = r.exploitChips / 100.0;
    r.seBb = std::sqrt(r.se0 * r.se0 + r.se1 * r.se1) / 100.0;
    r.samples = static_cast<uint64_t>(cfg.samples);
    r.valid = true;
    return r;
}

}  // namespace texas
