#include "texas/rivercfr.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace texas {
namespace {

constexpr int kMaxActs = 8;
constexpr int kMaxDepth = 12;

struct Node {
    bool terminal = false;
    int actor = -1;         // 非终局：行动者座位
    bool showdown = false;  // 终局：摊牌
    int winner = -1;        // 终局弃牌：赢家座位
    int pot = 0;
    int bet[2] = {0, 0};    // 本街投入额（绝对值）
    int stack[2] = {0, 0};  // 身后筹码
    int extra[2] = {0, 0};  // bet − 根节点 bet（从根起的追加投入）
    int aggr = 0;           // 本街已发生的激进动作数（下注/加注）
    int nActs = 0;
    Move mv[kMaxActs];
    int child[kMaxActs] = {};
};

struct ComboAux {
    int c[2] = {0, 0};
    int ltG = 0, leG = 0;  // 在 order_ 里的位置：rank < / <= 本组合
    int ltC[2] = {0, 0};   // 在对应牌的小表里的位置
    int leC[2] = {0, 0};
};

class RiverSolver {
public:
    RiverSolver(const HUState& s, int seat, const std::vector<float>& oppRange,
                const std::vector<float>& ownRange, const RiverCfrConfig& cfg)
        : s_(s), seat_(seat), cfg_(cfg) {
        combos_ = riverCombos(s);
        n_ = static_cast<int>(combos_.size());
        rootPot_ = s.pot();
        rootBet_[0] = int(s.contrib[0]);
        rootBet_[1] = int(s.contrib[1]);
        buildAux();
        initRanges(oppRange, ownRange);
        nodes_.reserve(512);
        root_ = build(seat_, rootPot_, rootBet_[0], rootBet_[1], int(s.stack[0]),
                      int(s.stack[1]), 0, toCall(s, seat) > 0 ? 1 : 0,
                      std::max<int>(int(s.lastRaiseSize), kBigBlind));
        for (const Node& nd : nodes_) {
            if (nd.terminal) continue;
            regret_.emplace_back(static_cast<size_t>(n_) * static_cast<size_t>(nd.nActs), 0.0f);
            avg_.emplace_back(static_cast<size_t>(n_) * static_cast<size_t>(nd.nActs), 0.0f);
        }
        // 对齐：regret_/avg_ 只给非终局节点建了条目，这里建立一个节点 -> 槽位映射。
        slotOf_.assign(nodes_.size(), -1);
        int next = 0;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (!nodes_[i].terminal) slotOf_[i] = next++;
        }
        // 缓冲
        dBuf_.assign(static_cast<size_t>(kMaxDepth) * kMaxActs * n_, 0.0f);
        dReach_.assign(static_cast<size_t>(kMaxDepth) * n_, 0.0f);
        stratBuf_.assign(static_cast<size_t>(kMaxDepth) * kMaxActs * n_, 0.0f);
        prefixG_.assign(static_cast<size_t>(n_) + 1, 0.0f);
        wSorted_.assign(static_cast<size_t>(n_), 0.0f);
        prefixC_.resize(kNumCards);
        cardList_.resize(kNumCards);
        cardTotal_.assign(kNumCards, 0.0f);
        rootOut_.assign(static_cast<size_t>(n_), 0.0f);
        v0_.assign(static_cast<size_t>(n_), 0.0f);
        v1_.assign(static_cast<size_t>(n_), 0.0f);
        // 我们的实际手牌在组合表里的下标
        heroIdx_ = -1;
        const Card h0 = static_cast<Card>(s.hole[seat_][0]);
        const Card h1 = static_cast<Card>(s.hole[seat_][1]);
        for (int i = 0; i < n_; ++i) {
            const int a = combos_[i].c0, b = combos_[i].c1;
            if ((a == h0 && b == h1) || (a == h1 && b == h0)) {
                heroIdx_ = i;
                break;
            }
        }
        // 根节点上"我方拿到组合 i 时对手的可达质量"。求解内部一律用反事实值
        // （Σ_j reach·payoff），只有根聚合要把它还原成条件 EV，所以这里把
        // 每张牌的质量预先算好（与树的深度无关，后续 reach 只是被对手策略缩放）。
        for (int p = 0; p < 2; ++p) {
            const std::vector<float>& w = range_[1 - p];
            preparePrefix(w.data());
            compat_[p].assign(static_cast<size_t>(n_), 0.0f);
            for (int i = 0; i < n_; ++i) {
                const ComboAux& a = aux_[static_cast<size_t>(i)];
                float tot = total_;
                if (cfg_.blockers) {
                    tot -= cardTotal_[static_cast<size_t>(a.c[0])] +
                           cardTotal_[static_cast<size_t>(a.c[1])];
                    tot += w[static_cast<size_t>(i)];
                }
                compat_[p][static_cast<size_t>(i)] = std::max(tot, 0.0f);
            }
        }
    }

    // 逐组合反事实值 -> 给定我方这张牌的条件 EV
    double condValue(int hero, int i, const std::vector<float>& v) const {
        const float m = compat_[hero][static_cast<size_t>(i)];
        if (m <= 1e-6f) return 0.0;
        return double(v[static_cast<size_t>(i)]) / double(m);
    }

    RiverCfrResult run() {
        RiverCfrResult res;
        res.combos = combos_;
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

        // 根节点平均策略
        std::vector<float> rootStrat(static_cast<size_t>(n_) * kMaxActs, 0.0f);
        avgStrategy(root_, rootStrat.data());

        // 每个根动作的 EV：从子节点起双方都按平均策略续演
        double value = 0.0;
        for (int a = 0; a < na; ++a) {
            evalAvg(root.child[a], seat_, range_[1 - seat_].data(), false, 1, v0_.data());
            RiverRootAction ra;
            ra.slot = -1;
            ra.mv = root.mv[a];
            ra.ev = condValue(seat_, heroIdx_, v0_);
            ra.prob = double(rootStrat[static_cast<size_t>(heroIdx_) * kMaxActs + a]);
            value += ra.ev * ra.prob;
            res.root.push_back(ra);
        }
        res.rootValue = value;

        // 可利用度上界：双方各打 BR，零和（常和 = 根底池）时 BR0 + BR1 − 底池 → 0。
        // 逐组合值已经是反事实值（未归一化），这里除以"根节点上对手可达质量"，
        // 得到给定我方这张牌的**条件**期望收益，再按我方范围加权。
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
    void buildAux() {
        order_.resize(static_cast<size_t>(n_));
        for (int i = 0; i < n_; ++i) order_[static_cast<size_t>(i)] = i;
        auto byRank = [&](int a, int b) {
            if (combos_[a].rank != combos_[b].rank) return combos_[a].rank < combos_[b].rank;
            return a < b;
        };
        std::sort(order_.begin(), order_.end(), byRank);
        cardList_.assign(kNumCards, {});
        for (int i = 0; i < n_; ++i) {
            cardList_[static_cast<size_t>(combos_[i].c0)].push_back(i);
            cardList_[static_cast<size_t>(combos_[i].c1)].push_back(i);
        }
        for (auto& v : cardList_) std::sort(v.begin(), v.end(), byRank);
        aux_.resize(static_cast<size_t>(n_));
        for (int i = 0; i < n_; ++i) {
            ComboAux& a = aux_[static_cast<size_t>(i)];
            a.c[0] = combos_[i].c0;
            a.c[1] = combos_[i].c1;
            a.ltG = lowerRank(order_, combos_[i].rank);
            a.leG = upperRank(order_, combos_[i].rank);
            for (int k = 0; k < 2; ++k) {
                const std::vector<int>& v = cardList_[static_cast<size_t>(a.c[k])];
                a.ltC[k] = lowerRank(v, combos_[i].rank);
                a.leC[k] = upperRank(v, combos_[i].rank);
            }
        }
    }

    int lowerRank(const std::vector<int>& v, int rank) const {
        int lo = 0, hi = static_cast<int>(v.size());
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (combos_[static_cast<size_t>(v[static_cast<size_t>(mid)])].rank < rank) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    int upperRank(const std::vector<int>& v, int rank) const {
        int lo = 0, hi = static_cast<int>(v.size());
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (combos_[static_cast<size_t>(v[static_cast<size_t>(mid)])].rank <= rank) {
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
        const std::vector<float>& srcOwn = own;
        if (static_cast<int>(srcOwn.size()) == n_) {
            for (int i = 0; i < n_; ++i) range_[seat_][static_cast<size_t>(i)] = std::max(0.0f, srcOwn[static_cast<size_t>(i)]);
        }
        if (static_cast<int>(opp.size()) == n_) {
            for (int i = 0; i < n_; ++i) {
                range_[1 - seat_][static_cast<size_t>(i)] = std::max(0.0f, opp[static_cast<size_t>(i)]);
            }
        }
        normalize(range_[seat_]);
        // 对手不可能拿着我们手里的两张牌
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

    int addTerminal(bool showdown, int winner, int pot, int b0, int b1) {
        Node nd;
        nd.terminal = true;
        nd.showdown = showdown;
        nd.winner = winner;
        nd.pot = pot;
        nd.bet[0] = b0;
        nd.bet[1] = b1;
        nd.extra[0] = b0 - rootBet_[0];
        nd.extra[1] = b1 - rootBet_[1];
        nodes_.push_back(nd);
        return static_cast<int>(nodes_.size()) - 1;
    }

    // 建树。返回节点下标。aggr = 本街已发生的下注/加注次数。
    int build(int actor, int pot, int b0, int b1, int st0, int st1, int checks, int aggr,
              int lastRaiseSize) {
        const int idx = static_cast<int>(nodes_.size());
        {
            Node nd;
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
                child = addTerminal(true, -1, pot, b0, b1);
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
            addAct(fmv, addTerminal(false, op, pot, b0, b1));

            const int callAmt = std::min(need, stMe);
            const int nbMe = betMe + callAmt;
            const int nb0 = (me == 0) ? nbMe : b0;
            const int nb1 = (me == 1) ? nbMe : b1;
            Move cmv{};
            cmv.kind = MK_CALL;
            cmv.to = static_cast<int16_t>(nbMe);
            addAct(cmv, addTerminal(true, -1, pot + callAmt, nb0, nb1));

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

    void preparePrefix(const float* w) {
        float total = 0.0f;
        for (int k = 0; k < n_; ++k) {
            const float x = w[static_cast<size_t>(order_[static_cast<size_t>(k)])];
            wSorted_[static_cast<size_t>(k)] = x;
            total += x;
        }
        prefixG_[0] = 0.0f;
        for (int k = 0; k < n_; ++k) {
            prefixG_[static_cast<size_t>(k) + 1] =
                prefixG_[static_cast<size_t>(k)] + wSorted_[static_cast<size_t>(k)];
        }
        total_ = total;
        for (int c = 0; c < kNumCards; ++c) {
            const std::vector<int>& lst = cardList_[static_cast<size_t>(c)];
            std::vector<float>& p = prefixC_[static_cast<size_t>(c)];
            p.resize(lst.size() + 1);
            p[0] = 0.0f;
            float s = 0.0f;
            for (size_t k = 0; k < lst.size(); ++k) {
                s += w[static_cast<size_t>(lst[k])];
                p[k + 1] = s;
            }
            cardTotal_[static_cast<size_t>(c)] = s;
        }
    }

    // 终局的逐组合反事实值 = Σ_j ro[j] · u(i, j)，对手 reach 为 0 的组合不参与。
    //
    // 注意这里**不做**按可达质量归一化：CFR 的 regret 与 BR 的 argmax 都要求
    // 值保持"原始单位"，否则不同层的值缩放不一致（子节点被除以该点的对手
    // 可达质量，回到父节点时没有同样的回缩），BR 会被系统性放大。
    void terminalValue(const Node& nd, int hero, const float* w, float* out) {
        preparePrefix(w);
        const float potF = float(nd.pot);
        const float e = float(nd.extra[hero]);
        if (!nd.showdown) {
            const float c = (hero == nd.winner) ? (potF - e) : (-e);
            for (int i = 0; i < n_; ++i) {
                const ComboAux& a = aux_[static_cast<size_t>(i)];
                float tot = total_;
                if (cfg_.blockers) {
                    // 对手不可能拿着我们的两张牌：把含这两张牌的 reach 扣掉，
                    // 组合自己会被两张牌各扣一次，补回一次。
                    tot -= cardTotal_[static_cast<size_t>(a.c[0])] +
                           cardTotal_[static_cast<size_t>(a.c[1])];
                    tot += w[i];
                }
                out[i] = c * tot;
            }
            return;
        }
        const float cWin = potF - e;
        const float cLose = -e;
        const float cTie = 0.5f * potF - e;
        for (int i = 0; i < n_; ++i) {
            const ComboAux& a = aux_[static_cast<size_t>(i)];
            float sLess = prefixG_[static_cast<size_t>(a.ltG)];
            float sLe = prefixG_[static_cast<size_t>(a.leG)];
            float tot = total_;
            if (cfg_.blockers) {
                sLess -= prefixC_[static_cast<size_t>(a.c[0])][static_cast<size_t>(a.ltC[0])] +
                         prefixC_[static_cast<size_t>(a.c[1])][static_cast<size_t>(a.ltC[1])];
                sLe -= prefixC_[static_cast<size_t>(a.c[0])][static_cast<size_t>(a.leC[0])] +
                       prefixC_[static_cast<size_t>(a.c[1])][static_cast<size_t>(a.leC[1])];
                sLe += w[i];
                tot -= cardTotal_[static_cast<size_t>(a.c[0])] +
                       cardTotal_[static_cast<size_t>(a.c[1])];
                tot += w[i];
            }
            const float sGt = tot - sLe;   // 对手更强（我们输）
            const float sTie = sLe - sLess;
            out[i] = cWin * sLess + cLose * sGt + cTie * sTie;
        }
    }

    void cfr(int idx, int hero, const float* rh, const float* ro, int t, int depth, float* out) {
        const Node& nd = nodes_[static_cast<size_t>(idx)];
        if (nd.terminal) {
            terminalValue(nd, hero, ro, out);
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

    // 双方都按平均策略续演；br=true 时 hero 自己的节点取 max（best response）。
    void evalAvg(int idx, int hero, const float* ro, bool br, int depth, float* out) {
        const Node& nd = nodes_[static_cast<size_t>(idx)];
        if (nd.terminal) {
            terminalValue(nd, hero, ro, out);
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
    RiverCfrConfig cfg_;
    std::vector<RiverCombo> combos_;
    int n_ = 0;
    int rootBet_[2] = {0, 0};
    int rootPot_ = 0;
    int heroIdx_ = -1;

    std::vector<Node> nodes_;
    int root_ = 0;
    std::vector<int> slotOf_;
    std::vector<std::vector<float>> regret_;
    std::vector<std::vector<float>> avg_;

    std::vector<int> order_;
    std::vector<ComboAux> aux_;
    std::vector<std::vector<int>> cardList_;
    std::vector<float> cardTotal_;

    std::vector<float> range_[2];
    std::vector<float> compat_[2];  // 根节点上给定我方组合 i 时对手的可达质量
    std::vector<float> dBuf_;
    std::vector<float> dReach_;
    std::vector<float> stratBuf_;
    std::vector<float> prefixG_;
    std::vector<float> wSorted_;
    std::vector<std::vector<float>> prefixC_;
    std::vector<float> rootOut_;
    std::vector<float> v0_;
    std::vector<float> v1_;
    float total_ = 0.0f;
};

}  // namespace

std::vector<RiverCombo> riverCombos(const HUState& s) {
    bool used[kNumCards] = {};
    Card board[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 5; ++i) {
        board[i] = static_cast<Card>(s.board[i]);
        used[static_cast<size_t>(board[i])] = true;
    }
    Card rest[kNumCards];
    int nr = 0;
    for (int c = 0; c < kNumCards; ++c) {
        if (!used[static_cast<size_t>(c)]) rest[nr++] = static_cast<Card>(c);
    }
    std::vector<RiverCombo> out;
    out.reserve(static_cast<size_t>(kRiverComboCount));
    for (int i = 0; i < nr; ++i) {
        for (int j = i + 1; j < nr; ++j) {
            RiverCombo rc;
            rc.c0 = rest[i];
            rc.c1 = rest[j];
            Card h[7] = {rc.c0, rc.c1, board[0], board[1], board[2], board[3], board[4]};
            rc.rank = evaluate7(h);
            out.push_back(rc);
        }
    }
    return out;
}

RiverCfrResult riverCfrSolve(const HUState& s, int seat, const std::vector<float>& oppRange,
                             const std::vector<float>& ownRange, const RiverCfrConfig& cfg) {
    RiverCfrResult bad;
    // 必须是真正的河牌决策点：公共牌已发满 5 张，且没人在更早的街上就结束
    // 了这一手（提前全下时 boardN 也会被推到 5，但那时对局已经 finished）。
    if (s.boardN != 5 || s.street != ST_RIVER || s.finished || seat < 0 || seat > 1) return bad;
    if (s.stack[seat] <= 0) return bad;
    RiverSolver solver(s, seat, oppRange, ownRange, cfg);
    return solver.run();
}

bool riverCfrMove(const HUState& s, const RiverCfrConfig& cfg, Move& out,
                  const std::vector<float>& oppRange, const std::vector<float>& ownRange,
                  RiverCfrResult* detail) {
    if (s.boardN != 5 || s.street != ST_RIVER || s.finished) return false;
    if (s.stack[s.toAct] <= 0) return false;
    RiverCfrResult r = riverCfrSolve(s, s.toAct, oppRange, ownRange, cfg);
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
            // 求解器的尺寸算术若与规则引擎有出入，退到最接近的合法带端点
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
