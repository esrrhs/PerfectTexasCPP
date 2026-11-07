#include "texas/blueprint.hpp"

#include <algorithm>
#include <cmath>

namespace texas {

// ---------------------------------------------------------------- 抽象辅助

int potBucketOf(int pot) {
    if (pot < 150) pot = 150;
    int b = static_cast<int>(std::floor(std::log2(static_cast<double>(pot) / 150.0) * 4.0));
    if (b < 0) b = 0;
    if (b > 31) b = 31;
    return b;
}

int cardBucketOf(const HUState& s, int buckets) {
    StrengthModel& sm = StrengthModel::instance();
    const int p = s.toAct;
    if (s.street == ST_PREFLOP) {
        return StrengthModel::preflopClass(Card(s.hole[p][0]), Card(s.hole[p][1]));
    }
    const Card hole[2] = {Card(s.hole[p][0]), Card(s.hole[p][1])};
    Card board[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < s.boardN && i < 5; ++i) board[i] = Card(s.board[i]);
    return sm.postflopBucket(hole, board, s.boardN, buckets);
}

uint64_t packInfoset(int street, int pot, int cardBucket, const uint8_t* seq, int seqLen,
                     int seat) {
    if (seqLen < 0) seqLen = 0;
    if (seqLen > kMaxStreetSeq) seqLen = kMaxStreetSeq;
    uint64_t packed = 0;
    for (int i = 0; i < seqLen; ++i) packed |= static_cast<uint64_t>(seq[i] & 7) << (3 * i);
    const uint64_t cb = static_cast<uint64_t>(cardBucket) & 0xFF;       // bit 0..7
    const uint64_t se = seat > 0 ? 1ULL : 0ULL;                          // bit 8
    const uint64_t pb = static_cast<uint64_t>(potBucketOf(pot)) & 0x1F;  // bit 9..13
    const uint64_t st = static_cast<uint64_t>(street) & 3;                // bit 14..15
    const uint64_t ln = static_cast<uint64_t>(seqLen) & 0xF;              // bit 16..19
    return cb | (se << 8) | (pb << 9) | (st << 14) | (ln << 16) | (packed << 20);
}

int legalAbstract(const HUState& s, int raiseCap, AbstractMove* out) {
    if (s.finished) return 0;
    const int p = s.toAct;
    const int o = 1 - p;
    const int gap = toCall(s, p);
    const int myStack = s.stack[p];
    const int callTo = s.contrib[p] + std::min(gap, myStack);
    int n = 0;

    if (gap > 0) {
        out[n].id = AB_FOLD;
        out[n].mv.kind = MK_FOLD;
        out[n].mv.to = s.contrib[p];
        ++n;
    }
    out[n].id = AB_CHECK_CALL;
    out[n].mv.kind = (gap > 0) ? MK_CALL : MK_CHECK;
    out[n].mv.to = static_cast<int16_t>(callTo);
    ++n;

    const bool canRaise = (s.numRaises < raiseCap) && !s.allin[o] && myStack > gap &&
                          isRaisePossible(s, p);
    if (canRaise) {
        const int maxTo = maxRaiseTo(s, p);
        const int potAfterCall = s.pot() + std::min(gap, myStack);
        // 两个比例尺寸；clamp 后若退化成全下则与 all-in 去重。
        const int targets[2] = {
            s.contrib[p] + gap + static_cast<int>(std::lround(0.5 * potAfterCall)),
            s.contrib[p] + gap + static_cast<int>(std::lround(1.0 * potAfterCall)),
        };
        const int ids[2] = {AB_RAISE_05, AB_RAISE_10};
        Move allinMv = makeRaise(s, maxTo, p);
        for (int i = 0; i < 2; ++i) {
            Move mv = makeRaise(s, targets[i], p);
            if (mv.kind == MK_ALLIN) continue;  // 与 all-in 重合
            bool dup = false;
            for (int j = 0; j < n; ++j) {
                if (out[j].mv.kind == mv.kind && out[j].mv.to == mv.to) { dup = true; break; }
            }
            if (dup) continue;
            out[n].id = ids[i];
            out[n].mv = mv;
            ++n;
        }
        if (maxTo > callTo) {
            out[n].id = AB_ALLIN;
            out[n].mv = allinMv;
            ++n;
        }
    }
    return n;
}

// 从 hist 重建当前街抽象序列：用真实 HUState replay，每步用 legalAbstract 匹配，
// 与训练侧完全一致（避免 clamp 碰撞时的分类歧义）。
int seqFromState(const HUState& s, int raiseCap, uint8_t* out) {
    const Card h0[2] = {s.hole[0][0], s.hole[0][1]};
    const Card h1[2] = {s.hole[1][0], s.hole[1][1]};
    Card board[5];
    for (int i = 0; i < 5; ++i) board[i] = s.board[i];
    HUState shadow = makeState(h0, h1, board, 0);
    Rng rng(1);
    int seqLen = 0;
    uint8_t seq[kMaxStreetSeq] = {};

    for (int i = 0; i < s.histLen; ++i) {
        const HistEntry& e = s.hist[i];
        if (shadow.finished) break;
        AbstractMove am[kAbstractStride];
        const int n = legalAbstract(shadow, raiseCap, am);
        int id = AB_CHECK_CALL;
        bool exact = false;
        for (int j = 0; j < n; ++j) {
            if (am[j].mv.kind == e.kind && am[j].mv.to == e.to) { id = am[j].id; exact = true; break; }
        }
        if (!exact && n > 0) {
            // 超树动作翻译（计划书 §4.3）：真实对手的下注额是任意整数，不在抽象
            // 金额表里。按「同类型 + 金额最近」落回抽象动作，而不是粗暴归成
            // check-call（那会把 seq 打歪，查表落到完全不相干的信息集）。
            double bestD = 1e18;
            for (int j = 0; j < n; ++j) {
                if (am[j].mv.kind != e.kind &&
                    !(e.kind == MK_RAISE && am[j].mv.kind == MK_ALLIN)) continue;
                const double d = std::fabs(double(am[j].mv.to) - double(e.to));
                if (d < bestD) { bestD = d; id = am[j].id; }
            }
        }
        if (shadow.street == s.street && seqLen < kMaxStreetSeq) seq[seqLen++] = static_cast<uint8_t>(id);
        Move mv;
        mv.kind = e.kind;
        mv.to = e.to;
        applyMove(shadow, mv, rng);
    }
    for (int i = 0; i < seqLen; ++i) out[i] = seq[i];
    return seqLen;
}

bool queryBlueprint(const HUState& s, const RegretTable& tab, int buckets, int raiseCap,
                    float* out) {
    uint8_t seq[kMaxStreetSeq] = {};
    const int seqLen = seqFromState(s, raiseCap, seq);
    const int cb = cardBucketOf(s, buckets);
    const uint64_t key = packInfoset(s.street, s.pot(), cb, seq, seqLen, s.toAct);
    bool legal[kAbstractStride] = {};
    AbstractMove am[kAbstractStride];
    const int n = legalAbstract(s, raiseCap, am);
    for (int i = 0; i < n; ++i) legal[am[i].id] = true;
    const int64_t idx = tab.find(key);
    if (idx < 0) {
        int cnt = 0;
        for (int a = 0; a < kAbstractStride; ++a) if (legal[a]) ++cnt;
        for (int a = 0; a < kAbstractStride; ++a)
            out[a] = legal[a] && cnt > 0 ? 1.0f / cnt : 0.0f;
        return false;
    }
    tab.averageStrategy(static_cast<uint32_t>(idx), legal, out);
    return true;
}

// 把抽象金额落到 19 槽里的某一个尺寸带；找不到覆盖带时取带中点最近的。
static int slotForAbstractRaise(const HUState& s, int amount) {
    LegalMove lm[kNumActions];
    const int m = legalMoves(s, lm);
    for (int j = 0; j < m; ++j) {
        if (lm[j].slot < kSlotSize0 || lm[j].slot >= kSlotAllIn) continue;
        if (amount >= lm[j].mv.to && amount <= lm[j].toHi) return lm[j].slot;
    }
    int bestSlot = -1;
    double bestDist = 1e18;
    for (int j = 0; j < m; ++j) {
        if (lm[j].slot < kSlotSize0 || lm[j].slot >= kSlotAllIn) continue;
        const double mid = 0.5 * (double(lm[j].mv.to) + double(lm[j].toHi));
        const double d = std::fabs(double(amount) - mid);
        if (d < bestDist) { bestDist = d; bestSlot = lm[j].slot; }
    }
    return bestSlot;
}

bool abstractToSlots(const HUState& s, const RegretTable& tab, int buckets, int raiseCap,
                     float* target) {
    for (int i = 0; i < kNumActions; ++i) target[i] = 0.0f;
    float abs[kAbstractStride] = {};
    const bool trained = queryBlueprint(s, tab, buckets, raiseCap, abs);
    AbstractMove am[kAbstractStride];
    const int n = legalAbstract(s, raiseCap, am);
    for (int i = 0; i < n; ++i) {
        const int id = am[i].id;
        int slot = -1;
        switch (id) {
            case AB_FOLD: slot = kSlotFold; break;
            case AB_CHECK_CALL: slot = kSlotCheckCall; break;
            case AB_ALLIN: slot = kSlotAllIn; break;
            default: slot = slotForAbstractRaise(s, am[i].mv.to); break;
        }
        if (slot >= 0 && slot < kNumActions) target[slot] += abs[id];
    }
    double sum = 0.0;
    for (int i = 0; i < kNumActions; ++i) sum += target[i];
    if (sum > 1e-9) {
        for (int i = 0; i < kNumActions; ++i) target[i] = float(target[i] / sum);
    } else {
        // 兜底：把质量均匀放到真实合法槽上。
        LegalMove lm[kNumActions];
        const int m = legalMoves(s, lm);
        for (int j = 0; j < m; ++j) target[lm[j].slot] = 1.0f / float(m);
    }
    return trained;
}

Move sampleBlueprintMove(const HUState& s, const RegretTable& tab, int buckets, int raiseCap,
                         Rng& rng, float temp) {
    float abs[kAbstractStride] = {};
    queryBlueprint(s, tab, buckets, raiseCap, abs);
    AbstractMove am[kAbstractStride];
    const int n = legalAbstract(s, raiseCap, am);
    if (n <= 0) {
        Move mv;
        mv.kind = MK_CHECK;
        mv.to = s.contrib[s.toAct];
        return mv;
    }
    double w[kAbstractStride] = {};
    double sum = 0.0;
    const float t = std::max(temp, 0.05f);
    for (int i = 0; i < n; ++i) {
        const double p = std::max(0.0, double(abs[am[i].id]));
        w[i] = std::pow(p, 1.0 / double(t));
        sum += w[i];
    }
    if (sum <= 1e-12) {
        for (int i = 0; i < n; ++i) w[i] = 1.0;
        sum = double(n);
    }
    const double u = std::generate_canonical<double, 32>(rng) * sum;
    double cum = 0.0;
    int pick = n - 1;
    for (int i = 0; i < n; ++i) {
        cum += w[i];
        if (u < cum) { pick = i; break; }
    }
    return am[pick].mv;
}

// ---------------------------------------------------------------- RegretTable

namespace {
constexpr uint64_t kEmptyKey = UINT64_MAX;
constexpr uint64_t kSaveMagic = 0x424C554550524931ULL;  // "BLUEPRI1"

uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
}  // namespace

RegretTable::RegretTable(int stride) : stride_(stride) {
    capacity_ = 1024;
    mask_ = capacity_ - 1;
    keys_.assign(capacity_, kEmptyKey);
    vals_.assign(capacity_, 0);
    regret_.assign(static_cast<size_t>(capacity_) * stride_, 0.0f);
    strat_.assign(static_cast<size_t>(capacity_) * stride_, 0.0f);
}

uint64_t RegretTable::hashKey(uint64_t k) const { return splitmix64(k); }

void RegretTable::grow() {
    const uint32_t newCap = capacity_ * 2;
    const uint32_t newMask = newCap - 1;
    std::vector<uint64_t> newKeys(newCap, kEmptyKey);
    std::vector<uint32_t> newVals(newCap, 0);
    for (uint32_t i = 0; i < capacity_; ++i) {
        const uint64_t k = keys_[i];
        if (k == kEmptyKey) continue;
        uint32_t h = static_cast<uint32_t>(hashKey(k)) & newMask;
        while (newKeys[h] != kEmptyKey) h = (h + 1) & newMask;
        newKeys[h] = k;
        newVals[h] = vals_[i];
    }
    keys_.swap(newKeys);
    vals_.swap(newVals);
    capacity_ = newCap;
    mask_ = newMask;
    regret_.resize(static_cast<size_t>(capacity_) * stride_, 0.0f);
    strat_.resize(static_cast<size_t>(capacity_) * stride_, 0.0f);
}

uint32_t RegretTable::intern(uint64_t key) {
    if (static_cast<uint64_t>(ninfosets_ + 1) > static_cast<uint64_t>(capacity_) * 7 / 10) grow();
    uint32_t h = static_cast<uint32_t>(hashKey(key)) & mask_;
    while (keys_[h] != kEmptyKey) {
        if (keys_[h] == key) return vals_[h];
        h = (h + 1) & mask_;
    }
    keys_[h] = key;
    vals_[h] = ninfosets_;
    return ninfosets_++;
}

int64_t RegretTable::find(uint64_t key) const {
    uint32_t h = static_cast<uint32_t>(hashKey(key)) & mask_;
    while (keys_[h] != kEmptyKey) {
        if (keys_[h] == key) return vals_[h];
        h = (h + 1) & mask_;
    }
    return -1;
}

void RegretTable::strategy(uint32_t idx, const bool* legal, float* out) const {
    const size_t base = static_cast<size_t>(idx) * stride_;
    float sum = 0.0f;
    int cnt = 0;
    for (int a = 0; a < stride_; ++a) {
        if (!legal[a]) { out[a] = 0.0f; continue; }
        ++cnt;
        const float r = regret_[base + a];
        if (r > 0.0f) sum += r;
    }
    if (sum > 0.0f) {
        for (int a = 0; a < stride_; ++a) {
            if (!legal[a]) { out[a] = 0.0f; continue; }
            const float r = regret_[base + a];
            out[a] = r > 0.0f ? r / sum : 0.0f;
        }
    } else {
        const float u = cnt > 0 ? 1.0f / cnt : 0.0f;
        for (int a = 0; a < stride_; ++a) out[a] = legal[a] ? u : 0.0f;
    }
}

void RegretTable::averageStrategy(uint32_t idx, const bool* legal, float* out) const {
    const size_t base = static_cast<size_t>(idx) * stride_;
    float sum = 0.0f;
    int cnt = 0;
    for (int a = 0; a < stride_; ++a) {
        if (!legal[a]) { out[a] = 0.0f; continue; }
        ++cnt;
        const float w = strat_[base + a];
        if (w > 0.0f) sum += w;
    }
    if (sum > 0.0f) {
        for (int a = 0; a < stride_; ++a) {
            if (!legal[a]) { out[a] = 0.0f; continue; }
            const float w = strat_[base + a];
            out[a] = w > 0.0f ? w / sum : 0.0f;
        }
    } else {
        const float u = cnt > 0 ? 1.0f / cnt : 0.0f;
        for (int a = 0; a < stride_; ++a) out[a] = legal[a] ? u : 0.0f;
    }
}

void RegretTable::addRegretClamped(uint32_t idx, int a, double delta) {
    const size_t pos = static_cast<size_t>(idx) * stride_ + a;
    const double r = static_cast<double>(regret_[pos]) + delta;
    regret_[pos] = r > 0.0 ? static_cast<float>(r) : 0.0f;
}

void RegretTable::addStrategy(uint32_t idx, int a, double weight) {
    strat_[static_cast<size_t>(idx) * stride_ + a] += static_cast<float>(weight);
}

bool RegretTable::save(const std::string& path) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint64_t magic = kSaveMagic;
    const uint32_t stride = static_cast<uint32_t>(stride_);
    bool ok = true;
    auto w = [&](const void* p, size_t n) { ok = ok && std::fwrite(p, 1, n, f) == n; };
    w(&magic, sizeof(magic));
    w(&stride, sizeof(stride));
    w(&ninfosets_, sizeof(ninfosets_));
    w(&capacity_, sizeof(capacity_));
    w(keys_.data(), keys_.size() * sizeof(uint64_t));
    w(vals_.data(), vals_.size() * sizeof(uint32_t));
    w(regret_.data(), regret_.size() * sizeof(float));
    w(strat_.data(), strat_.size() * sizeof(float));
    std::fclose(f);
    return ok;
}

bool RegretTable::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint64_t magic = 0;
    uint32_t stride = 0, n = 0, cap = 0;
    bool ok = true;
    auto r = [&](void* p, size_t sz) { ok = ok && std::fread(p, 1, sz, f) == sz; };
    r(&magic, sizeof(magic));
    r(&stride, sizeof(stride));
    r(&n, sizeof(n));
    r(&cap, sizeof(cap));
    if (!ok || magic != kSaveMagic || stride != static_cast<uint32_t>(stride_) || cap == 0 ||
        (cap & (cap - 1)) != 0) {
        std::fclose(f);
        return false;
    }
    ninfosets_ = n;
    capacity_ = cap;
    mask_ = cap - 1;
    keys_.resize(cap);
    vals_.resize(cap);
    regret_.resize(static_cast<size_t>(cap) * stride_);
    strat_.resize(static_cast<size_t>(cap) * stride_);
    r(keys_.data(), keys_.size() * sizeof(uint64_t));
    r(vals_.data(), vals_.size() * sizeof(uint32_t));
    r(regret_.data(), regret_.size() * sizeof(float));
    r(strat_.data(), strat_.size() * sizeof(float));
    std::fclose(f);
    return ok;
}

}  // namespace texas