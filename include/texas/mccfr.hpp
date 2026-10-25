#pragma once

// 抽象无限注德州的 CFR 求解器。
//
// 翻牌有 C(48,3)=17296 种组合，全树遍历不可能，所以改用 external sampling MCCFR：
// 轮到 traverser 时展开全部动作，对手和机会节点各采样一个。
//
// 信息集 = (街, 牌力桶, 下注历史)。牌力桶由 strength.hpp 提供，
// 这才让"具体手牌"塌缩成有限个状态。
//
// 性能上有两处不能想当然：
//   * 信息集表用平坦开放寻址，不用 std::unordered_map。遍历里每个决策节点都要
//     查一次（实测每轮 74 次），unordered_map 的指针追逐能吃掉四分之一的时间。
//   * 平坦表扩容会让引用失效，而 traverse 会跨递归持有 InfoSet&。所以扩容只在
//     两轮之间做（prepare），保证一轮遍历内部引用稳定。

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "holdem.hpp"
#include "strength.hpp"

namespace texas {

// 遗憾累计方式。两者都是"遗憾匹配 + 加权平均"，区别只在折扣：
//   CFRPlus  遗憾裁到非负、平均权重线性增长（iter）。简单、稳定。
//   DCFR     额外按 t 给正/负遗憾和平均策略打折（Brown & Sandholm 2019）。
//            正遗憾打 t^1.5/(t^1.5+1) 的折，负遗憾打 t^0.5/(t^0.5+1) 的折，
//            平均策略打 (t/(t+1))^2 的折。直觉是"越早的迭代越不可信"：
//            负遗憾打折比正遗憾快，等于允许策略较快地放弃坏动作；
//            平均策略打折则让后期迭代占主导。
enum class CfrVariant { CFRPlus, DCFR };

struct HoldemAbstraction {
    int buckets = 20;

    // 应对方能否看见对手的下注额。true 用带尺寸的历史哈希，false 用抹掉尺寸的那个。
    // 这看起来是个无害的粒度选择，其实是信息划分里最容易出错的一维：设成 false 时
    // "加注到 5"和"加注到 200"对应同一个信息集，应对方被迫用同一套动作分布应付
    // 小额加注和全下两种完全不同的局面。默认必须是 true。
    bool sizedHistory = true;

    // 每街给几档加注尺寸、各是多大。默认值等于历史行为（2.5x/4x、半池/一池、含 all-in）。
    ActionSpace actionSpace;

    int bucketOf(const HoldemGame::State& s, int p) const {
        const StrengthModel& m = StrengthModel::instance();
        if (s.boardCount < 3) {
            // 翻牌前只有 169 种可区分的手牌，全分辨率使用。
            // 这里绝不能套 buckets：10 个桶会把 AA 和 KK、AKs 和 AKo
            // 压成同一个信息集，等于把翻牌前的手牌结构整个抹掉。
            return StrengthModel::preflopClass(s.hole[p][0], s.hole[p][1]);
        }
        return m.postflopBucket(s.hole[p], s.board, s.boardCount, buckets);
    }

    std::uint64_t infosetIdWithBucket(const HoldemGame::State& s, int bucket) const {
        const int p = s.player;
        // histHash / histSizedHash 在 next() 里增量维护，这里 O(1) 取用，
        // 不用重新哈希整个历史
        std::uint64_t h = sizedHistory ? s.histSizedHash : s.histHash;
        h ^= static_cast<std::uint64_t>(bucket + 1) * 0x9E3779B97F4A7C15ULL;
        h ^= static_cast<std::uint64_t>(s.street + 1) * 0xC2B2AE3D27D4EB4FULL;
        h ^= static_cast<std::uint64_t>(p + 1) * 0x165667B19E3779F9ULL;
        return h;
    }

    std::uint64_t infosetId(const HoldemGame::State& s) const {
        return infosetIdWithBucket(s, bucketOf(s, s.player));
    }
};

// 单个信息集的累计量
struct InfoSetData {
    std::array<double, HoldemGame::kMaxActions> regret{};
    std::array<double, HoldemGame::kMaxActions> avg{};
    int n = 0;
};

// 老存盘格式（v4）的两个长度常量：那时 kMaxActions = 6、kMaxSizes = 3。
// 这两个数**不能**跟着现在的常量走，否则旧文件会被错位解读——v4 的布局把编译期常量
// 写死进了文件，所以解释旧文件时必须用当时的值。
constexpr int kV4Actions = 6;
constexpr int kV4Sizes = 3;
// v5 的记录长度已经是自描述的，但文件头里的动作数组还是按当时的 kMaxSizes = 4 定长。
// 于是"记录能跨 build 读、文件头不能"——查这个的时候差点被一个 +7.7 筹码/手的假结论骗过去。
constexpr int kV5Sizes = 4;

// 信息集键 -> InfoSetData 的平坦哈希表（线性探测）
class InfoSetTable {
public:
    InfoSetData& get(std::uint64_t key) {
        // 插入前先保证有余量。以前靠调用方"每轮预留够多"这条约定（prepare(256)），
        // 一旦某轮新增的信息集超过预留量，表会被填满，下面的线性探测就**永远**
        // 找不到空槽——直接死循环，而且采样剖析会把它算进调用者身上，很难定位。
        // 3 档动作集一轮能新增 330 个，正好踩中这个坑。
        if ((size_ + 1) * 10 >= slots_.size() * 7) {
            rehash(slots_.empty() ? 1024 : slots_.size() * 2);
        }
        std::size_t i = mix(key) & mask_;
        for (;;) {
            Slot& sl = slots_[i];
            if (!sl.used) {
                sl.used = true;
                sl.key = key;
                sl.val = InfoSetData{};
                ++size_;
                return sl.val;
            }
            if (sl.key == key) return sl.val;
            i = (i + 1) & mask_;
        }
    }

    const InfoSetData* find(std::uint64_t key) const {
        if (slots_.empty()) return nullptr;
        std::size_t i = mix(key) & mask_;
        for (;;) {
            const Slot& sl = slots_[i];
            if (!sl.used) return nullptr;
            if (sl.key == key) return &sl.val;
            i = (i + 1) & mask_;
        }
    }

    // 在两轮遍历之间腾出 extra 个空位。只是性能优化：有了 get() 里的自动扩容，
    // 即使预留不足也不会死循环，只是那一轮会多一次扩容 + 一次重新取引用。
    void prepare(std::size_t extra) {
        while (slots_.empty() || (size_ + extra) * 10 >= slots_.size() * 7) {
            rehash(slots_.empty() ? 1024 : slots_.size() * 2);
        }
    }

    std::size_t size() const { return size_; }

    // 扩容代数。traverse 在递归前后比一次：只有真的扩容过，它手里的引用才失效。
    // 直接每次都重新查表会让大表上每个 traverser 节点多一次 cache miss，
    // 实测 100 万轮慢 15% 左右；比代数几乎不要钱。
    std::size_t generation() const { return generation_; }

    // 所有信息集的正遗憾之和。除以迭代轮数就是平均外部遗憾的上界，
    // 也是"当前策略"离均衡多远的廉价指标（收敛时应按 1/sqrt(T) 下降）。
    double positiveRegretSum() const {
        double s = 0.0;
        for (const Slot& sl : slots_) {
            if (!sl.used) continue;
            const int n = std::min(sl.val.n, HoldemGame::kMaxActions);
            for (int i = 0; i < n; ++i) {
                if (sl.val.regret[i] > 0.0) s += sl.val.regret[i];
            }
        }
        return s;
    }

    // 键值对遍历，供存盘使用
    template <class Fn>
    void forEach(Fn&& fn) const {
        for (const Slot& sl : slots_) {
            if (sl.used) fn(sl.key, sl.val);
        }
    }

    void clear() {
        slots_.clear();
        mask_ = 0;
        size_ = 0;
        ++generation_;
    }

private:
    struct Slot {
        std::uint64_t key = 0;
        InfoSetData val{};
        bool used = false;
    };

    static std::uint64_t mix(std::uint64_t k) {
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccdULL;
        k ^= k >> 33;
        k *= 0xc4ceb9fe1a85ec53ULL;
        k ^= k >> 33;
        return k;
    }

    void rehash(std::size_t cap) {
        ++generation_;
        std::vector<Slot> old = std::move(slots_);
        slots_.assign(cap, Slot{});
        mask_ = cap - 1;
        size_ = 0;
        for (Slot& sl : old) {
            if (!sl.used) continue;
            std::size_t i = mix(sl.key) & mask_;
            while (slots_[i].used) i = (i + 1) & mask_;
            slots_[i] = std::move(sl);
            ++size_;
        }
    }

    std::vector<Slot> slots_;
    std::size_t mask_ = 0;
    std::size_t size_ = 0;
    std::size_t generation_ = 0;
};

// 给两位玩家各发两张底牌
template <class Rng>
inline void dealHoles(HoldemGame::State& s, Rng& rng) {
    Card deck[kNumCards];
    const int dn = HoldemGame::buildDeck(s, deck);
    const int need = s.numPlayers * 2;
    for (int i = 0; i < need; ++i) {
        const int j = i + static_cast<int>(rng() % static_cast<std::uint64_t>(dn - i));
        std::swap(deck[i], deck[j]);
    }
    int k = 0;
    for (int p = 0; p < s.numPlayers; ++p) {
        s.hole[p][0] = deck[k++];
        s.hole[p][1] = deck[k++];
    }
}

// 按概率向量采样一个下标（线性扫描，动作数最多 6，不值得二分）
template <class Rng>
inline int sampleIndex(const double* p, int n, Rng& rng) {
    if (n <= 0) return 0;
    const double u = static_cast<double>(rng() >> 11) * (1.0 / 9007199254740992.0);
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += p[i];
        if (u <= acc) return i;
    }
    return n - 1;
}

class HoldemSolver {
public:
    using InfoSet = InfoSetData;

    // sizedHistory 只能在造对象时定：训练中途改它会瞬间把所有键对到另一套表上，
    // 已有的遗憾和平均策略全部失效。
    explicit HoldemSolver(int buckets = 20, std::uint64_t seed = 20240518ULL,
                          CfrVariant variant = CfrVariant::DCFR, bool sizedHistory = true)
        : rng_(seed), variant_(variant) {
        abstraction_.buckets = buckets;
        abstraction_.sizedHistory = sizedHistory;
    }

    // 换动作集。**必须在第一次 iterate() 之前调用**：动作集决定合法动作个数，
    // 而信息集里存的就是"每个动作一个遗憾值"，中途换会让 n 对不上、旧数据全部作废。
    void setActionSpace(const ActionSpace& space) { abstraction_.actionSpace = space; }

    void iterate(int rounds) {
        for (int i = 0; i < rounds; ++i) {
            ++iter_;
            // 折扣因子只跟轮数有关，每轮算一次就够：一轮遍历里一个信息集最多被
            // 更新一次（external sampling 只展开 traverser 的动作）
            if (variant_ == CfrVariant::DCFR) {
                const double t = static_cast<double>(iter_);
                const double tp = t * std::sqrt(t);          // t^1.5
                posDiscount_ = tp / (tp + 1.0);
                const double sq = std::sqrt(t);              // t^0.5
                negDiscount_ = sq / (sq + 1.0);
                const double q = t / (t + 1.0);
                avgDiscount_ = q * q;
            }
            // 预留量按上一轮实际新增量自适应，别写死。写死 256 时，3 档动作集
            // 一轮能新增 330 个信息集，直接把表填满（见 get() 里的注释）。
            isets_.prepare(perRoundReserve_);
            HoldemGame::State root = HoldemGame::init(2);
            dealHoles(root, rng_);
            // 整副公共牌在根节点一次抽定：与逐街抽牌的联合分布相同，但整棵树共用
            // 一副公共牌，"牌力桶"就可以按 (玩家, 街) 在本轮内缓存
            sampleFullBoard(root);
            // 交替更新，每轮只让一个玩家做 traverser
            const int traverser = iter_ & 1;
            const std::size_t before = isets_.size();
            traverse(root, traverser, 1.0);
            const std::size_t added = isets_.size() - before;
            if (added > perRoundReserve_) perRoundReserve_ = added * 2;
        }
    }

    // 写入 out（至少 HoldemGame::kMaxActions 个元素），返回动作个数
    int fillAverageStrategy(const HoldemGame::State& s, double* out, int maxOut) const {
        Move moves[HoldemGame::kMaxActions];
        const int na =
            HoldemGame::actions(s, moves, HoldemGame::kMaxActions, abstraction_.actionSpace);
        const int n = std::min(na, maxOut);
        if (n <= 0) return 0;
        const InfoSetData* is = isets_.find(abstraction_.infosetId(s));
        if (is == nullptr || is->n != na) {
            for (int i = 0; i < n; ++i) out[i] = 1.0 / n;
            return n;
        }
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            out[i] = std::max(0.0, is->avg[i]);
            sum += out[i];
        }
        if (sum > 0.0) {
            for (int i = 0; i < n; ++i) out[i] /= sum;
        } else {
            for (int i = 0; i < n; ++i) out[i] = 1.0 / n;
        }
        return n;
    }

    std::vector<double> averageStrategy(const HoldemGame::State& s) const {
        double buf[HoldemGame::kMaxActions] = {};
        const int n = fillAverageStrategy(s, buf, HoldemGame::kMaxActions);
        return std::vector<double>(buf, buf + n);
    }

    // 按平均策略采样一个动作索引
    int sampleAction(const HoldemGame::State& s) {
        double buf[HoldemGame::kMaxActions] = {};
        const int n = fillAverageStrategy(s, buf, HoldemGame::kMaxActions);
        return sampleIndex(buf, n, rng_);
    }

    std::size_t numInfosets() const { return isets_.size(); }
    int iterations() const { return iter_; }
    CfrVariant variant() const { return variant_; }
    const HoldemAbstraction& abstraction() const { return abstraction_; }

    // 平均外部遗憾上界：正遗憾总和 / 迭代轮数。
    // 这是"当前策略"距均衡的合法上界（不是平均策略的），廉价且单调下降即可信；
    // 两个配置比强度时，同样轮数下这个数越小说明收敛得越好。
    double regretBound() const {
        return (iter_ > 0) ? isets_.positiveRegretSum() / static_cast<double>(iter_) : 0.0;
    }

    // 策略质量的粗略标尺：用当前平均策略打一个"在合法动作上均匀随机"的对手，
    // 返回我方每手牌的平均净收益（筹码）。不是可利用度，只做横向比较。
    double evVsUniform(int deals, int heroSeat = 0) {
        if (deals <= 0) return 0.0;
        double sum = 0.0;
        for (int i = 0; i < deals; ++i) {
            HoldemGame::State s = HoldemGame::init(2);
            dealHoles(s, rng_);
            double buf[HoldemGame::kMaxActions] = {};
            for (;;) {
                if (HoldemGame::isTerminal(s)) break;
                if (HoldemGame::needsChance(s)) {
                    HoldemGame::deal(s, rng_);
                    continue;
                }
                Move moves[HoldemGame::kMaxActions];
                const int na =
            HoldemGame::actions(s, moves, HoldemGame::kMaxActions, abstraction_.actionSpace);
                if (na <= 0) break;
                int pick;
                if (s.player == heroSeat) {
                    const int n = fillAverageStrategy(s, buf, HoldemGame::kMaxActions);
                    pick = sampleIndex(buf, n, rng_);
                } else {
                    pick = static_cast<int>(rng_() % static_cast<std::uint64_t>(na));
                }
                if (pick >= na) pick = na - 1;
                s = HoldemGame::next(s, moves[pick]);
            }
            sum += HoldemGame::utilityFrom(s, heroSeat);
        }
        return sum / static_cast<double>(deals);
    }

    // ---------------- 存盘 / 读盘 ----------------
    //
    // 用途是把训练好的策略冻成"基准对手"：以后每次改求解器，都拿新策略跟这份
    // 快照对打若干手，用带标准误的净收益判断到底变强了没有。没有固定基准，
    // 就只能靠自对局，而自对局在零和对称游戏里恒为 0，什么也测不出来。
    bool save(const char* path) const {
        std::FILE* f = std::fopen(path, "wb");
        if (f == nullptr) return false;
        const std::uint32_t magic = 0x54434652u;  // "TCFR"
        // 版本 4 起把抽象参数也写进去。必须存：信息集键是从"当前抽象"算出来的，
        // 读盘时若桶数/尺寸可见性/动作集与存盘时不同，所有键都对不上，会静默退化成
        // 均匀策略。这种错误比读不出来更难查。
        //
        // 版本 5 把每条记录的长度从 kMaxActions 改成 n。v4 写的是固定 6 个数，
        // 于是 kMaxActions 一改，所有旧文件都会被**错位**解读（不是读不出来，是读成
        // 别的信息集的数据）——这正是"静默比报错更糟"的那种改动。
        //
        // 版本 6 把文件头也改成自描述的：先写档数，再写那么多尺寸。v5 只修了记录、
        // 漏了文件头，于是换个编译期上限读同一个文件，尺寸数组会多读/少读几个 int，
        // 后面全部错位——和 v4 的毛病一模一样，只是换了个位置。实测中它表现为
        // "换一个 build 对打能赢 7.7 筹码/手"，看起来像重大发现，其实是读串了。
        const std::uint32_t ver = 6;
        const std::uint64_t cnt = static_cast<std::uint64_t>(isets_.size());
        const std::uint64_t iters = static_cast<std::uint64_t>(iter_);
        const std::int32_t buckets = abstraction_.buckets;
        const std::int32_t sized = abstraction_.sizedHistory ? 1 : 0;
        const std::int32_t allin = abstraction_.actionSpace.includeAllIn ? 1 : 0;
        std::fwrite(&magic, sizeof(magic), 1, f);
        std::fwrite(&ver, sizeof(ver), 1, f);
        std::fwrite(&cnt, sizeof(cnt), 1, f);
        // 轮数必须存下来：DCFR 的折扣因子依赖 t，断点续训要接着原来的 t 走；
        // 另外读盘后遗憾上界 = 正遗憾和 / t，没有 t 就没法显示
        std::fwrite(&iters, sizeof(iters), 1, f);
        std::fwrite(&buckets, sizeof(buckets), 1, f);
        std::fwrite(&sized, sizeof(sized), 1, f);
        // v6 起文件头自描述：档数在前，尺寸数组只写实际用到的那么多
        std::fwrite(&abstraction_.actionSpace.preflopCount, sizeof(int), 1, f);
        std::fwrite(abstraction_.actionSpace.preflopMul1000, sizeof(int),
                    static_cast<std::size_t>(abstraction_.actionSpace.preflopCount), f);
        std::fwrite(&abstraction_.actionSpace.postflopCount, sizeof(int), 1, f);
        std::fwrite(abstraction_.actionSpace.postflopFrac1000, sizeof(int),
                    static_cast<std::size_t>(abstraction_.actionSpace.postflopCount), f);
        std::fwrite(&allin, sizeof(allin), 1, f);
        isets_.forEach([f](std::uint64_t key, const InfoSetData& is) {
            const std::uint32_t n = static_cast<std::uint32_t>(is.n);
            std::fwrite(&key, sizeof(key), 1, f);
            std::fwrite(&n, sizeof(n), 1, f);
            std::fwrite(is.regret.data(), sizeof(double), n, f);
            std::fwrite(is.avg.data(), sizeof(double), n, f);
        });
        const bool ok = (std::ferror(f) == 0);
        std::fclose(f);
        return ok;
    }

    bool load(const char* path) {
        std::FILE* f = std::fopen(path, "rb");
        if (f == nullptr) return false;
        std::uint32_t magic = 0;
        std::uint32_t ver = 0;
        std::uint64_t cnt = 0;
        std::uint64_t iters = 0;
        if (std::fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0x54434652u) {
            std::fclose(f);
            return false;
        }
        if (std::fread(&ver, sizeof(ver), 1, f) != 1 || ver < 4 || ver > 6) {
            std::fclose(f);
            return false;
        }
        if (std::fread(&cnt, sizeof(cnt), 1, f) != 1) {
            std::fclose(f);
            return false;
        }
        if (std::fread(&iters, sizeof(iters), 1, f) != 1) {
            std::fclose(f);
            return false;
        }
        // 抽象参数跟着模型走：读盘后这个求解器代表的就是存盘时那个模型
        std::int32_t buckets = 20;
        std::int32_t sized = 1;
        std::int32_t allin = 1;
        ActionSpace space;
        if (std::fread(&buckets, sizeof(buckets), 1, f) != 1) {
            std::fclose(f);
            return false;
        }
        if (std::fread(&sized, sizeof(sized), 1, f) != 1) {
            std::fclose(f);
            return false;
        }
        // 三种文件头布局：
        //   v4/v5  动作数组在前、档数在后，数组长度是**当时**的 kMaxSizes
        //   v6     档数在前、数组长度 = 档数（自描述）
        if (ver <= 5) {
            const int fixed = (ver == 4) ? kV4Sizes : kV5Sizes;
            if (std::fread(space.preflopMul1000, sizeof(int), static_cast<std::size_t>(fixed), f) !=
                static_cast<std::size_t>(fixed)) {
                std::fclose(f);
                return false;
            }
            if (std::fread(&space.preflopCount, sizeof(int), 1, f) != 1) {
                std::fclose(f);
                return false;
            }
            if (std::fread(space.postflopFrac1000, sizeof(int), static_cast<std::size_t>(fixed),
                           f) != static_cast<std::size_t>(fixed)) {
                std::fclose(f);
                return false;
            }
            if (std::fread(&space.postflopCount, sizeof(int), 1, f) != 1) {
                std::fclose(f);
                return false;
            }
        } else {
            if (std::fread(&space.preflopCount, sizeof(int), 1, f) != 1) {
                std::fclose(f);
                return false;
            }
            if (space.preflopCount < 0 || space.preflopCount > ActionSpace::kMaxSizes) {
                std::fclose(f);
                return false;
            }
            if (std::fread(space.preflopMul1000, sizeof(int),
                           static_cast<std::size_t>(space.preflopCount), f) !=
                static_cast<std::size_t>(space.preflopCount)) {
                std::fclose(f);
                return false;
            }
            if (std::fread(&space.postflopCount, sizeof(int), 1, f) != 1) {
                std::fclose(f);
                return false;
            }
            if (space.postflopCount < 0 || space.postflopCount > ActionSpace::kMaxSizes) {
                std::fclose(f);
                return false;
            }
            if (std::fread(space.postflopFrac1000, sizeof(int),
                           static_cast<std::size_t>(space.postflopCount), f) !=
                static_cast<std::size_t>(space.postflopCount)) {
                std::fclose(f);
                return false;
            }
        }
        if (std::fread(&allin, sizeof(allin), 1, f) != 1) {
            std::fclose(f);
            return false;
        }
        abstraction_.buckets = static_cast<int>(buckets);
        abstraction_.sizedHistory = (sized != 0);
        abstraction_.actionSpace = space;
        abstraction_.actionSpace.includeAllIn = (allin != 0);
        isets_.clear();
        isets_.prepare(static_cast<std::size_t>(cnt) + 1024);
        for (std::uint64_t i = 0; i < cnt; ++i) {
            std::uint64_t key = 0;
            std::uint32_t n = 0;
            InfoSetData is{};
            if (std::fread(&key, sizeof(key), 1, f) != 1) break;
            if (std::fread(&n, sizeof(n), 1, f) != 1) break;
            // v4 的记录固定 kV4Actions 个动作，v5 起是 n 个
            const std::size_t width =
                (ver == 4) ? static_cast<std::size_t>(kV4Actions) : static_cast<std::size_t>(n);
            if (width > static_cast<std::size_t>(HoldemGame::kMaxActions)) break;
            if (std::fread(is.regret.data(), sizeof(double), width, f) != width) break;
            if (std::fread(is.avg.data(), sizeof(double), width, f) != width) break;
            is.n = static_cast<int>(n);
            isets_.get(key) = is;
        }
        std::fclose(f);
        iter_ = static_cast<int>(iters);
        return true;
    }

private:
    // 整轮内公共牌固定，同一个玩家同一条街的牌力桶只算一次。
    // 实测这是最大的单点收益：原本每轮要算 74 次，现在约 8 次。
    int cachedBucket(const HoldemGame::State& s, int p) {
        const int bc = s.boardCount;
        if (bucketStamp_[p][bc] == iter_) return bucketVal_[p][bc];
        const int b = abstraction_.bucketOf(s, p);
        bucketStamp_[p][bc] = iter_;
        bucketVal_[p][bc] = b;
        return b;
    }

    void sampleFullBoard(const HoldemGame::State& s) {
        Card deck[kNumCards];
        const int dn = HoldemGame::buildDeck(s, deck);
        for (int i = 0; i < 5; ++i) {
            const int j = i + static_cast<int>(rng_() % static_cast<std::uint64_t>(dn - i));
            std::swap(deck[i], deck[j]);
            board_[i] = deck[i];
        }
    }

    double traverse(const HoldemGame::State& s, int traverser, double reachT) {
        if (HoldemGame::isTerminal(s)) {
            // 带上牌力缓存：整轮公共牌固定，同一摊牌的牌力只算一次而不是几十次
            return HoldemGame::utilityFrom(s, traverser, &scores_, iter_);
        }
        if (HoldemGame::needsChance(s)) {
            HoldemGame::State t = s;
            HoldemGame::reveal(t, board_);
            return traverse(t, traverser, reachT);
        }

        const int p = s.player;
        if (p < 0) return 0.0;

        Move moves[HoldemGame::kMaxActions];
        const int na =
            HoldemGame::actions(s, moves, HoldemGame::kMaxActions, abstraction_.actionSpace);
        if (na <= 0) return HoldemGame::utilityFrom(s, traverser, &scores_, iter_);

        // 这个引用只在**递归之前**用。递归期间 get() 可能扩容，`is` 会失效，
        // 所以更新遗憾之前必须按 key 重新取一次（见下面的 is2）。
        const std::uint64_t key = abstraction_.infosetIdWithBucket(s, cachedBucket(s, p));
        InfoSetData& is = isets_.get(key);
        if (is.n != na) is.n = na;

        double sigma[HoldemGame::kMaxActions] = {};
        double sum = 0.0;
        for (int i = 0; i < na; ++i) {
            sigma[i] = std::max(0.0, is.regret[i]);
            sum += sigma[i];
        }
        if (sum > 0.0) {
            for (int i = 0; i < na; ++i) sigma[i] /= sum;
        } else {
            for (int i = 0; i < na; ++i) sigma[i] = 1.0 / na;
        }

        if (p == traverser) {
            const std::size_t gen = isets_.generation();
            double vals[HoldemGame::kMaxActions] = {};
            double nodeVal = 0.0;
            for (int i = 0; i < na; ++i) {
                vals[i] = traverse(HoldemGame::next(s, moves[i]), traverser, reachT * sigma[i]);
                nodeVal += sigma[i] * vals[i];
            }
            // 递归里 get() 可能扩容。只有扩容过（代数变了）才需要按 key 重新取，
            // 否则复用上面的引用——大表上每次重取都是一次 cache miss。
            InfoSetData* p2 = (isets_.generation() == gen) ? &is : &isets_.get(key);
            InfoSetData& is2 = *p2;
            // CFR+：遗憾裁到非负；DCFR：先按正负分别打折，不裁剪
            for (int i = 0; i < na; ++i) {
                double r = is2.regret[i] *
                           (is2.regret[i] > 0.0 ? posDiscount_ : negDiscount_);
                r += vals[i] - nodeVal;
                is2.regret[i] = (variant_ == CfrVariant::CFRPlus) ? std::max(0.0, r) : r;
            }
            // 平均策略的加权方式两个变体不同，这里混用会出问题：
            //   CFR+ 靠线性权重 w = t 实现"越晚的迭代越重要"。
            //   DCFR 的 (t/(t+1))^2 折扣本身就是加权方式（等价于权重 ∝ k^2），
            //   再乘一个 t 会让权重变成 k^3，平均窗口被压到只剩最后 16%。
            // 实测后果是训练越久平均策略越弱：跨种子两两对打时 100 万/200 万轮
            // 相对 50 万轮分别 -0.225±0.163 / -0.269±0.165（显著更差）；
            // 去掉重复的线性权重后变成 -0.091 / -0.064（不显著）。
            const double w = (variant_ == CfrVariant::DCFR) ? 1.0 : static_cast<double>(iter_);
            for (int i = 0; i < na; ++i) {
                is2.avg[i] = is2.avg[i] * avgDiscount_ + w * reachT * sigma[i];
            }
            return nodeVal;
        }

        // 对手节点：按当前策略采样一个动作
        const int pick = std::min(sampleIndex(sigma, na, rng_), na - 1);
        return traverse(HoldemGame::next(s, moves[pick]), traverser, reachT);
    }

    HoldemAbstraction abstraction_;
    InfoSetTable isets_;
    std::mt19937_64 rng_;
    HoldemGame::ScoreCache scores_;
    Card board_[5] = {};
    int bucketVal_[2][6] = {};
    int bucketStamp_[2][6] = {{-1, -1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1, -1}};
    CfrVariant variant_ = CfrVariant::CFRPlus;
    double posDiscount_ = 1.0;
    double negDiscount_ = 1.0;
    double avgDiscount_ = 1.0;
    int iter_ = 0;
    // 下一轮要预留多少空位。按实测的每轮新增量自适应，避免写死导致表被填满。
    std::size_t perRoundReserve_ = 256;
};

// ---------------- 跨策略对打 ----------------

struct MatchResult {
    int hands = 0;
    double heroPerHand = 0.0;  // hero 每手净收益（筹码）
    double stderr = 0.0;       // 上面这个均值自身的标准误
};

// 用 hero 的平均策略对抗 villain 的平均策略，返回 hero 的每手净收益。
// heroSeat 决定谁坐小盲；统计量带标准误，方便判断两个版本的差异是不是真的。
inline MatchResult headToHead(const HoldemSolver& hero, const HoldemSolver& villain, int heroSeat,
                              int hands, std::uint64_t seed = 1ULL) {
    MatchResult r;
    if (hands <= 0) return r;
    r.hands = hands;
    std::mt19937_64 rng(seed);
    double sum = 0.0;
    double sumsq = 0.0;

    for (int i = 0; i < hands; ++i) {
        HoldemGame::State s = HoldemGame::init(2);
        dealHoles(s, rng);
        double buf[HoldemGame::kMaxActions] = {};
        for (;;) {
            if (HoldemGame::isTerminal(s)) break;
            if (HoldemGame::needsChance(s)) {
                HoldemGame::deal(s, rng);
                continue;
            }
            Move moves[HoldemGame::kMaxActions];
            // 轮到谁就用谁的动作集。两边动作集不同时这在博弈论上是合法的：
            // 每个玩家只能从自己的抽象里选招，这恰好就是"受限抽象互相打"的定义。
            const bool isHero = (s.player == heroSeat);
            const ActionSpace& space = isHero ? hero.abstraction().actionSpace
                                              : villain.abstraction().actionSpace;
            const int na = HoldemGame::actions(s, moves, HoldemGame::kMaxActions, space);
            if (na <= 0) break;
            const int n = isHero ? hero.fillAverageStrategy(s, buf, HoldemGame::kMaxActions)
                                 : villain.fillAverageStrategy(s, buf, HoldemGame::kMaxActions);
            int pick = sampleIndex(buf, n, rng);
            if (pick >= na) pick = na - 1;
            s = HoldemGame::next(s, moves[pick]);
        }
        const double u = HoldemGame::utilityFrom(s, heroSeat);
        sum += u;
        sumsq += u * u;
    }

    r.heroPerHand = sum / hands;
    const double var = std::max(0.0, sumsq / hands - r.heroPerHand * r.heroPerHand);
    r.stderr = std::sqrt(var / hands);
    return r;
}

// 两个座位各打一遍，合成一个"座位对称"的结果（消除小盲位置优势）
inline MatchResult headToHeadBothSeats(const HoldemSolver& a, const HoldemSolver& b, int hands,
                                       std::uint64_t seed = 1ULL) {
    const MatchResult sb = headToHead(a, b, 0, hands, seed);
    const MatchResult bb = headToHead(a, b, 1, hands, seed + 1000003ULL);
    MatchResult r;
    r.hands = hands * 2;
    r.heroPerHand = 0.5 * (sb.heroPerHand + bb.heroPerHand);
    r.stderr = 0.5 * std::sqrt(sb.stderr * sb.stderr + bb.stderr * bb.stderr);
    return r;
}

}  // namespace texas
