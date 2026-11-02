#include "texas/turncfr.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "texas/features.hpp"

namespace texas {
namespace {

constexpr int kMaxActs = 8;
constexpr int kMaxDepth = 12;

enum NodeKind : uint8_t { NK_DECISION = 0, NK_FOLD_TERM = 1, NK_LEAF = 2 };

struct Node {
    uint8_t kind = NK_DECISION;
    int actor = -1;         // 决策节点：行动者座位
    int winner = -1;        // 弃牌终局：赢家座位
    int pot = 0;
    int bet[2] = {0, 0};    // 本街投入额（绝对值）
    int stack[2] = {0, 0};
    int extra[2] = {0, 0};  // bet − 根节点 bet
    int aggr = 0;
    int nActs = 0;
    Move mv[kMaxActs];
    int child[kMaxActs] = {};
    int parent = -1;        // 从根到本节点的路径（critic 叶子重放用）
    Move edgeMv{};
};

// 每张河牌的摊牌聚合结构：组合按"该河牌下的牌力"排序，前缀和 + 单牌小前缀和
// 做 O(1) blocker 查询（与 rivercfr 的 terminalValue 同款数学）。
struct RiverAux {
    std::vector<int> order;              // 组合下标，按 rank 升序
    std::vector<int> ltG, leG;           // 每个组合在 order 里 rank< / <= 的位置
    std::vector<int> ltC0, leC0, ltC1, leC1;  // 在对应单牌小表里的位置
    std::vector<std::vector<int>> cardList;   // 每张牌：含它的组合，按 rank 升序
    std::vector<std::vector<float>> prefixC;  // 每张牌的小前缀和（迭代期重填）
    std::vector<float> prefixG;               // 全局 rank 前缀和（迭代期重填）
    std::vector<float> cardTotal;             // 每张牌的质量和（迭代期重填）
    std::vector<int> containR;                // 含这张河牌的组合（迭代期清零）
    float total = 0.0f;
};

class TurnSolver {
public:
    TurnSolver(const HUState& s, int seat, const std::vector<float>& oppRange,
               const std::vector<float>& ownRange, const TurnCfrConfig& cfg)
        : s_(s), seat_(seat), cfg_(cfg) {
        combos_ = turnCombos(s);
        n_ = static_cast<int>(combos_.size());
        rootPot_ = s.pot();
        rootBet_[0] = int(s.contrib[0]);
        rootBet_[1] = int(s.contrib[1]);
        buildRivers();
        initRanges(oppRange, ownRange);
        nodes_.reserve(512);
        root_ = build(seat_, rootPot_, rootBet_[0], rootBet_[1], int(s.stack[0]),
                      int(s.stack[1]), 0, toCall(s, seat) > 0 ? 1 : 0,
                      std::max<int>(int(s.lastRaiseSize), kBigBlind));
        for (const Node& nd : nodes_) {
            if (nd.kind != NK_DECISION) continue;
            regret_.emplace_back(static_cast<size_t>(n_) * static_cast<size_t>(nd.nActs), 0.0f);
            avg_.emplace_back(static_cast<size_t>(n_) * static_cast<size_t>(nd.nActs), 0.0f);
        }
        slotOf_.assign(nodes_.size(), -1);
        int next = 0;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (nodes_[i].kind == NK_DECISION) slotOf_[i] = next++;
        }
        dBuf_.assign(static_cast<size_t>(kMaxDepth) * kMaxActs * n_, 0.0f);
        dReach_.assign(static_cast<size_t>(kMaxDepth) * n_, 0.0f);
        stratBuf_.assign(static_cast<size_t>(kMaxDepth) * kMaxActs * n_, 0.0f);
        rootOut_.assign(static_cast<size_t>(n_), 0.0f);
        v0_.assign(static_cast<size_t>(n_), 0.0f);
        v1_.assign(static_cast<size_t>(n_), 0.0f);
        wScratch_.assign(static_cast<size_t>(n_), 0.0f);
        heroIdx_ = -1;
        const Card h0 = static_cast<Card>(s.hole[seat_][0]);
        const Card h1 = static_cast<Card>(s.hole[seat_][1]);
        for (int i = 0; i < n_; ++i) {
            const int a = combos_[static_cast<size_t>(i)].c0, b = combos_[static_cast<size_t>(i)].c1;
            if ((a == h0 && b == h1) || (a == h1 && b == h0)) {
                heroIdx_ = i;
                break;
            }
        }
        for (int p = 0; p < 2; ++p) {
            const std::vector<float>& w = range_[1 - p];
            compat_[p].assign(static_cast<size_t>(n_), 0.0f);
            for (int i = 0; i < n_; ++i) {
                float tot = 0.0f;
                if (cfg_.blockers) {
                    float sum = 0.0f;
                    for (int j = 0; j < n_; ++j) {
                        const int a = combos_[static_cast<size_t>(j)].c0;
                        const int b = combos_[static_cast<size_t>(j)].c1;
                        const int ca = combos_[static_cast<size_t>(i)].c0;
                        const int cb = combos_[static_cast<size_t>(i)].c1;
                        if (a == ca || b == ca || a == cb || b == cb) continue;
                        sum += w[static_cast<size_t>(j)];
                    }
                    tot = sum;
                } else {
                    for (float x : w) tot += x;
                }
                compat_[p][static_cast<size_t>(i)] = std::max(tot, 0.0f);
            }
        }
    }

    double condValue(int hero, int i, const std::vector<float>& v) const {
        const float m = compat_[hero][static_cast<size_t>(i)];
        if (m <= 1e-6f) return 0.0;
        return double(v[static_cast<size_t>(i)]) / double(m);
    }

    TurnCfrResult run() {
        TurnCfrResult res;
        res.combos = n_;
        res.nodes = static_cast<int>(nodes_.size());
        if (heroIdx_ < 0 || n_ <= 0) return res;
        for (int t = 1; t <= cfg_.iters; ++t) {
            for (int hero = 0; hero < 2; ++hero) {
                cfr(root_, hero, range_[hero].data(), range_[1 - hero].data(), t, 0,
                    rootOut_.data());
            }
        }
        const Node& root = nodes_[root_];
        const int na = root.nActs;
        if (na <= 0) return res;

        std::vector<float> rootStrat(static_cast<size_t>(n_) * kMaxActs, 0.0f);
        avgStrategy(root_, rootStrat.data());

        double value = 0.0;
        for (int a = 0; a < na; ++a) {
            evalAvg(root.child[a], seat_, range_[1 - seat_].data(), false, 1, v0_.data());
            TurnRootAction ra;
            ra.slot = -1;
            ra.mv = root.mv[a];
            ra.ev = condValue(seat_, heroIdx_, v0_);
            ra.prob = double(rootStrat[static_cast<size_t>(heroIdx_) * kMaxActs + a]);
            value += ra.ev * ra.prob;
            res.root.push_back(ra);
        }
        res.rootValue = value;

        evalAvg(root_, 0, range_[1].data(), true, 0, v0_.data());
        evalAvg(root_, 1, range_[0].data(), true, 0, v1_.data());
        double br0 = 0.0, br1 = 0.0;
        for (int i = 0; i < n_; ++i) {
            br0 += double(range_[0][static_cast<size_t>(i)]) * condValue(0, i, v0_);
            br1 += double(range_[1][static_cast<size_t>(i)]) * condValue(1, i, v1_);
        }
        res.brGap = br0 + br1 - double(rootPot_);
        res.br0 = br0;
        res.br1 = br1;
        res.valid = true;
        return res;
    }

private:
    void buildRivers() {
        // 48 张非公共牌都可以当河牌；对组合 i 而言其中 2 张在自己的手里，
        // 平均时分母是 46（leafValue 里跳过冲突项）。
        bool used[kNumCards] = {};
        for (int i = 0; i < s_.boardN && i < 5; ++i) used[s_.board[i]] = true;
        rivers_.clear();
        for (int c = 0; c < kNumCards; ++c) {
            if (!used[c]) rivers_.push_back(c);
        }
        nRivers_ = static_cast<int>(rivers_.size());
        aux_.resize(static_cast<size_t>(nRivers_));
        Card board[5] = {0, 0, 0, 0, 0};
        for (int i = 0; i < 4; ++i) board[i] = static_cast<Card>(s_.board[i]);
        for (int ri = 0; ri < nRivers_; ++ri) {
            RiverAux& a = aux_[static_cast<size_t>(ri)];
            const int r = rivers_[static_cast<size_t>(ri)];
            board[4] = static_cast<Card>(r);
            std::vector<int> ranks(static_cast<size_t>(n_));
            for (int i = 0; i < n_; ++i) {
                Card h[7] = {combos_[static_cast<size_t>(i)].c0, combos_[static_cast<size_t>(i)].c1,
                             board[0], board[1], board[2], board[3], board[4]};
                ranks[static_cast<size_t>(i)] = evaluate7(h);
                if (combos_[static_cast<size_t>(i)].c0 == r || combos_[static_cast<size_t>(i)].c1 == r) {
                    a.containR.push_back(i);
                }
            }
            a.order.resize(static_cast<size_t>(n_));
            for (int i = 0; i < n_; ++i) a.order[static_cast<size_t>(i)] = i;
            std::sort(a.order.begin(), a.order.end(), [&](int x, int y) {
                if (ranks[static_cast<size_t>(x)] != ranks[static_cast<size_t>(y)]) {
                    return ranks[static_cast<size_t>(x)] < ranks[static_cast<size_t>(y)];
                }
                return x < y;
            });
            a.cardList.assign(kNumCards, {});
            for (int i : a.order) {
                a.cardList[static_cast<size_t>(combos_[static_cast<size_t>(i)].c0)].push_back(i);
                a.cardList[static_cast<size_t>(combos_[static_cast<size_t>(i)].c1)].push_back(i);
            }
            a.ltG.resize(static_cast<size_t>(n_));
            a.leG.resize(static_cast<size_t>(n_));
            a.ltC0.resize(static_cast<size_t>(n_));
            a.leC0.resize(static_cast<size_t>(n_));
            a.ltC1.resize(static_cast<size_t>(n_));
            a.leC1.resize(static_cast<size_t>(n_));
            for (int i = 0; i < n_; ++i) {
                const int rank = ranks[static_cast<size_t>(i)];
                a.ltG[static_cast<size_t>(i)] = lowerRank(a.order, ranks, rank);
                a.leG[static_cast<size_t>(i)] = upperRank(a.order, ranks, rank);
                a.ltC0[static_cast<size_t>(i)] = lowerRank(
                    a.cardList[static_cast<size_t>(combos_[static_cast<size_t>(i)].c0)], ranks, rank);
                a.leC0[static_cast<size_t>(i)] = upperRank(
                    a.cardList[static_cast<size_t>(combos_[static_cast<size_t>(i)].c0)], ranks, rank);
                a.ltC1[static_cast<size_t>(i)] = lowerRank(
                    a.cardList[static_cast<size_t>(combos_[static_cast<size_t>(i)].c1)], ranks, rank);
                a.leC1[static_cast<size_t>(i)] = upperRank(
                    a.cardList[static_cast<size_t>(combos_[static_cast<size_t>(i)].c1)], ranks, rank);
            }
            a.prefixG.assign(static_cast<size_t>(n_) + 1, 0.0f);
            a.cardTotal.assign(kNumCards, 0.0f);
            a.prefixC.resize(kNumCards);
            for (int c = 0; c < kNumCards; ++c) {
                a.prefixC[static_cast<size_t>(c)].assign(a.cardList[static_cast<size_t>(c)].size() + 1, 0.0f);
            }
        }
    }

    static int lowerRank(const std::vector<int>& v, const std::vector<int>& ranks, int rank) {
        int lo = 0, hi = static_cast<int>(v.size());
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (ranks[static_cast<size_t>(v[static_cast<size_t>(mid)])] < rank) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    static int upperRank(const std::vector<int>& v, const std::vector<int>& ranks, int rank) {
        int lo = 0, hi = static_cast<int>(v.size());
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (ranks[static_cast<size_t>(v[static_cast<size_t>(mid)])] <= rank) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    static void normalize(std::vector<float>& v) {
        double s = 0.0;
        for (float x : v) s += double(x);
        if (s <= 0.0) {
            const float u = v.empty() ? 0.0f : 1.0f / float(v.size());
            for (float& x : v) x = u;
            return;
        }
        for (float& x : v) x = float(double(x) / s);
    }

    void initRanges(const std::vector<float>& opp, const std::vector<float>& own) {
        for (int p = 0; p < 2; ++p) range_[p].assign(static_cast<size_t>(n_), 1.0f);
        if (static_cast<int>(own.size()) == n_) {
            for (int i = 0; i < n_; ++i) {
                range_[seat_][static_cast<size_t>(i)] = std::max(0.0f, own[static_cast<size_t>(i)]);
            }
        }
        if (static_cast<int>(opp.size()) == n_) {
            for (int i = 0; i < n_; ++i) {
                range_[1 - seat_][static_cast<size_t>(i)] = std::max(0.0f, opp[static_cast<size_t>(i)]);
            }
        }
        normalize(range_[seat_]);
        const Card h0 = static_cast<Card>(s_.hole[seat_][0]);
        const Card h1 = static_cast<Card>(s_.hole[seat_][1]);
        for (int i = 0; i < n_; ++i) {
            const int a = combos_[static_cast<size_t>(i)].c0;
            const int b = combos_[static_cast<size_t>(i)].c1;
            if (a == h0 || b == h0 || a == h1 || b == h1) {
                range_[1 - seat_][static_cast<size_t>(i)] = 0.0f;
            }
        }
        normalize(range_[1 - seat_]);
    }

    int addFoldTerm(int winner, int pot, int b0, int b1) {
        Node nd;
        nd.kind = NK_FOLD_TERM;
        nd.winner = winner;
        nd.pot = pot;
        nd.bet[0] = b0;
        nd.bet[1] = b1;
        nd.extra[0] = b0 - rootBet_[0];
        nd.extra[1] = b1 - rootBet_[1];
        nodes_.push_back(nd);
        return static_cast<int>(nodes_.size()) - 1;
    }

    // 本街结束：发河牌后（近似：立即）摊牌。≥1 人全下时是精确的（没有河牌
    // 下注街）；双方都还有筹码时是 check-down 近似。
    int addLeaf(int pot, int b0, int b1) {
        Node nd;
        nd.kind = NK_LEAF;
        nd.pot = pot;
        nd.bet[0] = b0;
        nd.bet[1] = b1;
        nd.extra[0] = b0 - rootBet_[0];
        nd.extra[1] = b1 - rootBet_[1];
        nodes_.push_back(nd);
        return static_cast<int>(nodes_.size()) - 1;
    }

    int build(int actor, int pot, int b0, int b1, int st0, int st1, int checks, int aggr,
              int lastRaiseSize) {
        const int idx = static_cast<int>(nodes_.size());
        {
            Node nd;
            nd.kind = NK_DECISION;
            nd.actor = actor;
            nd.pot = pot;
            nd.bet[0] = b0;
            nd.bet[1] = b1;
            nd.stack[0] = st0;
            nd.stack[1] = st1;
            nd.extra[0] = b0 - rootBet_[0];
            nd.extra[1] = b1 - rootBet_[1];
            nd.aggr = aggr;
            nodes_.push_back(nd);
        }
        const int me = actor, op = 1 - actor;
        Move mvs[kMaxActs];
        int kids[kMaxActs] = {};
        int na = 0;
        auto addAct = [&](Move mv, int child) {
            if (na >= kMaxActs) return;
            mvs[na] = mv;
            kids[na] = child;
            ++na;
        };
        const int betMe = (me == 0) ? b0 : b1;
        const int betOp = (me == 0) ? b1 : b0;
        const int stMe = (me == 0) ? st0 : st1;
        const int need = std::max(0, betOp - betMe);

        if (need == 0) {
            Move cmv{};
            cmv.kind = MK_CHECK;
            cmv.to = static_cast<int16_t>(betMe);
            int child = 0;
            if (checks >= 1) {
                child = addLeaf(pot, b0, b1);
            } else {
                child = build(op, pot, b0, b1, st0, st1, checks + 1, aggr, lastRaiseSize);
            }
            addAct(cmv, child);
            if (stMe > 0) {
                const int maxTo = betMe + stMe;
                std::vector<int> tos;
                for (double f : cfg_.betFracs) {
                    long long amt = static_cast<long long>(std::llround(f * double(pot)));
                    if (amt < kBigBlind) amt = kBigBlind;
                    int to = betMe + static_cast<int>(amt);
                    if (to > maxTo) to = maxTo;
                    if (to > betMe) tos.push_back(to);
                }
                if (cfg_.allin && maxTo > betMe) tos.push_back(maxTo);
                std::sort(tos.begin(), tos.end());
                tos.erase(std::unique(tos.begin(), tos.end()), tos.end());
                for (int to : tos) {
                    const int add = to - betMe;
                    const int nb0 = (me == 0) ? to : b0;
                    const int nb1 = (me == 1) ? to : b1;
                    const int ns0 = (me == 0) ? st0 - add : st0;
                    const int ns1 = (me == 1) ? st1 - add : st1;
                    const int child = build(op, pot + add, nb0, nb1, ns0, ns1, 0, aggr + 1,
                                            to - betOp);
                    Move bmv{};
                    bmv.kind = (add >= stMe) ? MK_ALLIN : MK_RAISE;
                    bmv.to = static_cast<int16_t>(to);
                    addAct(bmv, child);
                }
            }
        } else {
            Move fmv{};
            fmv.kind = MK_FOLD;
            fmv.to = static_cast<int16_t>(betMe);
            addAct(fmv, addFoldTerm(op, pot, b0, b1));

            const int callAmt = std::min(need, stMe);
            const int nbMe = betMe + callAmt;
            const int nb0 = (me == 0) ? nbMe : b0;
            const int nb1 = (me == 1) ? nbMe : b1;
            Move cmv{};
            cmv.kind = MK_CALL;
            cmv.to = static_cast<int16_t>(nbMe);
            addAct(cmv, addLeaf(pot + callAmt, nb0, nb1));

            const int maxTo = betMe + stMe;
            if (aggr < 1 + cfg_.maxRaises && maxTo > betOp) {
                const int minRaise = std::max(lastRaiseSize, kBigBlind);
                std::vector<int> tos;
                for (double f : cfg_.betFracs) {
                    long long ex = static_cast<long long>(
                        std::llround(f * double(pot + need)));
                    int raiseBy = static_cast<int>(std::max<long long>(minRaise, ex));
                    int to = betOp + raiseBy;
                    if (to > maxTo) to = maxTo;
                    if (to < betOp + minRaise && to != maxTo) continue;
                    if (to > betOp) tos.push_back(to);
                }
                if (cfg_.allin && maxTo > betOp) tos.push_back(maxTo);
                std::sort(tos.begin(), tos.end());
                tos.erase(std::unique(tos.begin(), tos.end()), tos.end());
                for (int to : tos) {
                    const int add = to - betMe;
                    const int nb0 = (me == 0) ? to : b0;
                    const int nb1 = (me == 1) ? to : b1;
                    const int ns0 = (me == 0) ? st0 - add : st0;
                    const int ns1 = (me == 1) ? st1 - add : st1;
                    const int child = build(op, pot + add, nb0, nb1, ns0, ns1, 0, aggr + 1,
                                            to - betOp);
                    Move rmv{};
                    rmv.kind = (add >= stMe) ? MK_ALLIN : MK_RAISE;
                    rmv.to = static_cast<int16_t>(to);
                    addAct(rmv, child);
                }
            }
        }

        Node& fin = nodes_[static_cast<size_t>(idx)];
        fin.nActs = na;
        for (int k = 0; k < na; ++k) {
            fin.mv[k] = mvs[k];
            fin.child[k] = kids[k];
        }
        return idx;
    }

    void fillPrefix(int ri, const float* w) {
        RiverAux& a = aux_[static_cast<size_t>(ri)];
        float total = 0.0f;
        a.prefixG[0] = 0.0f;
        for (int k = 0; k < n_; ++k) {
            const float x = w[static_cast<size_t>(a.order[static_cast<size_t>(k)])];
            total += x;
            a.prefixG[static_cast<size_t>(k) + 1] = a.prefixG[static_cast<size_t>(k)] + x;
        }
        a.total = total;
        for (int c = 0; c < kNumCards; ++c) {
            const std::vector<int>& lst = a.cardList[static_cast<size_t>(c)];
            std::vector<float>& p = a.prefixC[static_cast<size_t>(c)];
            p[0] = 0.0f;
            float s = 0.0f;
            for (size_t k = 0; k < lst.size(); ++k) {
                s += w[static_cast<size_t>(lst[k])];
                p[k + 1] = s;
            }
            a.cardTotal[static_cast<size_t>(c)] = s;
        }
    }

    // check-down 叶子：out[i] = Σ_j ro[j] · E_河牌[摊牌收益(i, j, 河牌)]。
    // 反事实值保持"原始单位"（不逐节点归一化，见 rivercfr #106(a) 的坑）。
    void leafValue(const Node& nd, int hero, const float* ro, float* out) {
        std::copy(ro, ro + n_, wScratch_.begin());
        for (int i = 0; i < n_; ++i) out[i] = 0.0f;
        const float potF = float(nd.pot);
        const float e = float(nd.extra[hero]);
        const float cWin = potF - e;
        const float cLose = -e;
        const float cTie = 0.5f * potF - e;
        // 分母 44 = 48 张非公共牌 − 组合 i 的 2 张 − 对手组合 j 的 2 张。
        // 对每个合法 j 都相同（j 冲突的河牌在 reach 里被清零），所以是常数。
        const float invR = 1.0f / float(nRivers_ - 4);
        for (int ri = 0; ri < nRivers_; ++ri) {
            const int r = rivers_[static_cast<size_t>(ri)];
            RiverAux& a = aux_[static_cast<size_t>(ri)];
            for (int c : a.containR) wScratch_[static_cast<size_t>(c)] = 0.0f;
            fillPrefix(ri, wScratch_.data());
            for (int i = 0; i < n_; ++i) {
                const TurnCombo& cm = combos_[static_cast<size_t>(i)];
                if (int(cm.c0) == r || int(cm.c1) == r) continue;  // 组合拿河牌，跳过
                float sLess = a.prefixG[static_cast<size_t>(a.ltG[static_cast<size_t>(i)])];
                float sLe = a.prefixG[static_cast<size_t>(a.leG[static_cast<size_t>(i)])];
                float tot = a.total;
                if (cfg_.blockers) {
                    sLess -= a.prefixC[static_cast<size_t>(cm.c0)][static_cast<size_t>(a.ltC0[static_cast<size_t>(i)])] +
                             a.prefixC[static_cast<size_t>(cm.c1)][static_cast<size_t>(a.ltC1[static_cast<size_t>(i)])];
                    sLe -= a.prefixC[static_cast<size_t>(cm.c0)][static_cast<size_t>(a.leC0[static_cast<size_t>(i)])] +
                           a.prefixC[static_cast<size_t>(cm.c1)][static_cast<size_t>(a.leC1[static_cast<size_t>(i)])];
                    sLe += wScratch_[static_cast<size_t>(i)];
                    tot -= a.cardTotal[static_cast<size_t>(cm.c0)] +
                           a.cardTotal[static_cast<size_t>(cm.c1)];
                    tot += wScratch_[static_cast<size_t>(i)];
                }
                const float sGt = tot - sLe;
                const float sTie = sLe - sLess;
                out[i] += invR * (cWin * sLess + cLose * sGt + cTie * sTie);
            }
            for (int c : a.containR) wScratch_[static_cast<size_t>(c)] = ro[static_cast<size_t>(c)];
        }
    }

    void strategyOf(int idx, float* strat) const {
        const Node& nd = nodes_[static_cast<size_t>(idx)];
        const int na = nd.nActs;
        const std::vector<float>& R = regret_[static_cast<size_t>(slotOf_[static_cast<size_t>(idx)])];
        for (int i = 0; i < n_; ++i) {
            const size_t base = static_cast<size_t>(i) * static_cast<size_t>(na);
            float sum = 0.0f;
            for (int a = 0; a < na; ++a) {
                const float r = R[base + static_cast<size_t>(a)];
                if (r > 0.0f) sum += r;
            }
            for (int a = 0; a < na; ++a) {
                float p;
                if (sum > 0.0f) {
                    p = std::max(0.0f, R[base + static_cast<size_t>(a)]) / sum;
                } else {
                    p = 1.0f / float(na);
                }
                strat[static_cast<size_t>(i) * kMaxActs + static_cast<size_t>(a)] = p;
            }
        }
    }

    void avgStrategy(int idx, float* strat) const {
        const Node& nd = nodes_[static_cast<size_t>(idx)];
        const int na = nd.nActs;
        const std::vector<float>& A = avg_[static_cast<size_t>(slotOf_[static_cast<size_t>(idx)])];
        for (int i = 0; i < n_; ++i) {
            const size_t base = static_cast<size_t>(i) * static_cast<size_t>(na);
            float sum = 0.0f;
            for (int a = 0; a < na; ++a) sum += A[base + static_cast<size_t>(a)];
            for (int a = 0; a < na; ++a) {
                float p;
                if (sum > 1e-12f) {
                    p = A[base + static_cast<size_t>(a)] / sum;
                } else {
                    p = 1.0f / float(na);
                }
                strat[static_cast<size_t>(i) * kMaxActs + static_cast<size_t>(a)] = p;
            }
        }
    }

    void cfr(int idx, int hero, const float* rh, const float* ro, int t, int depth, float* out) {
        const Node& nd = nodes_[static_cast<size_t>(idx)];
        if (nd.kind == NK_FOLD_TERM) {
            foldTerminal(nd, hero, ro, out);
            return;
        }
        if (nd.kind == NK_LEAF) {
            leafValue(nd, hero, ro, out);
            return;
        }
        const int na = nd.nActs;
        float* strat = stratBuf_.data() + static_cast<size_t>(depth) * kMaxActs * n_;
        strategyOf(idx, strat);
        float* kids = dBuf_.data() + static_cast<size_t>(depth) * kMaxActs * n_;
        float* reach = dReach_.data() + static_cast<size_t>(depth) * n_;
        if (nd.actor == hero) {
            for (int a = 0; a < na; ++a) {
                for (int i = 0; i < n_; ++i) {
                    reach[i] = rh[i] * strat[static_cast<size_t>(i) * kMaxActs + static_cast<size_t>(a)];
                }
                cfr(nd.child[a], hero, reach, ro, t, depth + 1, kids + static_cast<size_t>(a) * n_);
            }
            const int slot = slotOf_[static_cast<size_t>(idx)];
            float* R = regret_[static_cast<size_t>(slot)].data();
            float* A = avg_[static_cast<size_t>(slot)].data();
            for (int i = 0; i < n_; ++i) {
                float v = 0.0f;
                for (int a = 0; a < na; ++a) {
                    v += strat[static_cast<size_t>(i) * kMaxActs + static_cast<size_t>(a)] *
                         kids[static_cast<size_t>(a) * n_ + static_cast<size_t>(i)];
                }
                out[i] = v;
                const float w = float(t) * rh[i];
                const size_t base = static_cast<size_t>(i) * static_cast<size_t>(na);
                for (int a = 0; a < na; ++a) {
                    const float cf = kids[static_cast<size_t>(a) * n_ + static_cast<size_t>(i)];
                    R[base + static_cast<size_t>(a)] =
                        std::max(0.0f, R[base + static_cast<size_t>(a)] + cf - v);
                    A[base + static_cast<size_t>(a)] +=
                        w * strat[static_cast<size_t>(i) * kMaxActs + static_cast<size_t>(a)];
                }
            }
        } else {
            for (int a = 0; a < na; ++a) {
                for (int i = 0; i < n_; ++i) {
                    reach[i] = ro[i] * strat[static_cast<size_t>(i) * kMaxActs + static_cast<size_t>(a)];
                }
                cfr(nd.child[a], hero, rh, reach, t, depth + 1, kids + static_cast<size_t>(a) * n_);
            }
            for (int i = 0; i < n_; ++i) {
                float v = 0.0f;
                for (int a = 0; a < na; ++a) {
                    v += kids[static_cast<size_t>(a) * n_ + static_cast<size_t>(i)];
                }
                out[i] = v;
            }
        }
    }

    // 弃牌终局：out[i] = c · Σ_{j 与 i 不冲突} ro[j]。O(n)：先算单牌质量和，
    // 组合 i 的两张牌被扣两次、补回自己一次。
    void foldTerminal(const Node& nd, int hero, const float* ro, float* out) {
        const float potF = float(nd.pot);
        const float e = float(nd.extra[hero]);
        const float c = (hero == nd.winner) ? (potF - e) : (-e);
        float cardSum[kNumCards] = {};
        float total = 0.0f;
        for (int j = 0; j < n_; ++j) {
            const float w = ro[j];
            total += w;
            cardSum[static_cast<size_t>(combos_[static_cast<size_t>(j)].c0)] += w;
            cardSum[static_cast<size_t>(combos_[static_cast<size_t>(j)].c1)] += w;
        }
        for (int i = 0; i < n_; ++i) {
            float tot = total;
            if (cfg_.blockers) {
                tot -= cardSum[static_cast<size_t>(combos_[static_cast<size_t>(i)].c0)] +
                       cardSum[static_cast<size_t>(combos_[static_cast<size_t>(i)].c1)];
                tot += ro[i];
            }
            out[i] = c * std::max(tot, 0.0f);
        }
    }

    void evalAvg(int idx, int hero, const float* ro, bool br, int depth, float* out) {
        const Node& nd = nodes_[static_cast<size_t>(idx)];
        if (nd.kind == NK_FOLD_TERM) {
            foldTerminal(nd, hero, ro, out);
            return;
        }
        if (nd.kind == NK_LEAF) {
            leafValue(nd, hero, ro, out);
            return;
        }
        const int na = nd.nActs;
        float* strat = stratBuf_.data() + static_cast<size_t>(depth) * kMaxActs * n_;
        avgStrategy(idx, strat);
        float* kids = dBuf_.data() + static_cast<size_t>(depth) * kMaxActs * n_;
        float* reach = dReach_.data() + static_cast<size_t>(depth) * n_;
        const bool heroNode = (nd.actor == hero);
        if (heroNode && br) {
            for (int a = 0; a < na; ++a) {
                evalAvg(nd.child[a], hero, ro, true, depth + 1, kids + static_cast<size_t>(a) * n_);
            }
            for (int i = 0; i < n_; ++i) {
                float v = kids[i];
                for (int a = 1; a < na; ++a) {
                    v = std::max(v, kids[static_cast<size_t>(a) * n_ + static_cast<size_t>(i)]);
                }
                out[i] = v;
            }
        } else if (heroNode) {
            for (int a = 0; a < na; ++a) {
                evalAvg(nd.child[a], hero, ro, false, depth + 1, kids + static_cast<size_t>(a) * n_);
            }
            for (int i = 0; i < n_; ++i) {
                float v = 0.0f;
                for (int a = 0; a < na; ++a) {
                    v += strat[static_cast<size_t>(i) * kMaxActs + static_cast<size_t>(a)] *
                         kids[static_cast<size_t>(a) * n_ + static_cast<size_t>(i)];
                }
                out[i] = v;
            }
        } else {
            for (int a = 0; a < na; ++a) {
                for (int i = 0; i < n_; ++i) {
                    reach[i] = ro[i] * strat[static_cast<size_t>(i) * kMaxActs + static_cast<size_t>(a)];
                }
                evalAvg(nd.child[a], hero, reach, br, depth + 1, kids + static_cast<size_t>(a) * n_);
            }
            for (int i = 0; i < n_; ++i) {
                float v = 0.0f;
                for (int a = 0; a < na; ++a) {
                    v += kids[static_cast<size_t>(a) * n_ + static_cast<size_t>(i)];
                }
                out[i] = v;
            }
        }
    }

    HUState s_;
    int seat_ = 0;
    TurnCfrConfig cfg_;
    std::vector<TurnCombo> combos_;
    int n_ = 0;
    int rootBet_[2] = {0, 0};
    int rootPot_ = 0;
    int heroIdx_ = -1;

    std::vector<int> rivers_;
    int nRivers_ = 0;
    std::vector<RiverAux> aux_;

    std::vector<Node> nodes_;
    int root_ = 0;
    std::vector<int> slotOf_;
    std::vector<std::vector<float>> regret_;
    std::vector<std::vector<float>> avg_;

    std::vector<float> range_[2];
    std::vector<float> compat_[2];
    std::vector<float> dBuf_;
    std::vector<float> dReach_;
    std::vector<float> stratBuf_;
    std::vector<float> rootOut_;
    std::vector<float> v0_;
    std::vector<float> v1_;
    std::vector<float> wScratch_;
};

}  // namespace

std::vector<TurnCombo> turnCombos(const HUState& s) {
    bool used[kNumCards] = {};
    for (int i = 0; i < s.boardN && i < 5; ++i) {
        used[static_cast<size_t>(s.board[i])] = true;
    }
    Card rest[kNumCards];
    int nr = 0;
    for (int c = 0; c < kNumCards; ++c) {
        if (!used[static_cast<size_t>(c)]) rest[nr++] = static_cast<Card>(c);
    }
    std::vector<TurnCombo> out;
    out.reserve(static_cast<size_t>(kTurnComboCount));
    for (int i = 0; i < nr; ++i) {
        for (int j = i + 1; j < nr; ++j) {
            TurnCombo tc;
            tc.c0 = rest[i];
            tc.c1 = rest[j];
            out.push_back(tc);
        }
    }
    return out;
}

TurnCfrResult turnCfrSolve(const HUState& s, int seat, const std::vector<float>& oppRange,
                           const std::vector<float>& ownRange, const TurnCfrConfig& cfg) {
    TurnCfrResult bad;
    if (s.boardN != 4 || s.street != ST_TURN || s.finished || seat < 0 || seat > 1) return bad;
    if (s.stack[seat] <= 0) return bad;
    TurnSolver solver(s, seat, oppRange, ownRange, cfg);
    return solver.run();
}

bool turnCfrMove(const HUState& s, const TurnCfrConfig& cfg, Move& out,
                 const std::vector<float>& oppRange, const std::vector<float>& ownRange,
                 TurnCfrResult* detail) {
    if (s.boardN != 4 || s.street != ST_TURN || s.finished) return false;
    if (s.stack[s.toAct] <= 0) return false;
    TurnCfrResult r = turnCfrSolve(s, s.toAct, oppRange, ownRange, cfg);
    if (!r.valid || r.root.empty()) return false;
    size_t best = 0;
    for (size_t k = 1; k < r.root.size(); ++k) {
        if (r.root[k].ev > r.root[best].ev) best = k;
    }
    Move mv = r.root[best].mv;
    LegalMove lm[kNumActions];
    const int n = legalMoves(s, lm);
    if (n <= 0) return false;
    bool ok = false;
    if (mv.kind == MK_FOLD) {
        for (int i = 0; i < n; ++i) {
            if (lm[i].slot == kSlotFold) {
                mv = lm[i].mv;
                ok = true;
            }
        }
    } else if (mv.kind == MK_CHECK || mv.kind == MK_CALL) {
        for (int i = 0; i < n; ++i) {
            if (lm[i].slot == kSlotCheckCall) {
                mv = lm[i].mv;
                ok = true;
            }
        }
    } else {
        if (canRaiseTo(s, int(mv.to))) {
            mv = makeRaise(s, int(mv.to));
            ok = (mv.kind == MK_RAISE || mv.kind == MK_ALLIN);
        } else {
            for (int i = 0; i < n; ++i) {
                if (lm[i].slot >= kSlotSize0 && lm[i].slot != kSlotAllIn &&
                    int(mv.to) >= lm[i].mv.to && int(mv.to) <= lm[i].toHi) {
                    mv = makeRaise(s, std::min<int>(int(mv.to), lm[i].toHi));
                    ok = (mv.kind == MK_RAISE || mv.kind == MK_ALLIN);
                    break;
                }
            }
        }
    }
    if (!ok) return false;
    out = mv;
    if (detail != nullptr) *detail = std::move(r);
    return true;
}

}  // namespace texas
