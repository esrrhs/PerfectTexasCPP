# robopoker 的 CFR 到底怎么做（读码笔记）

> 对象：`~/tools/robopoker`，commit `b6bf9e2`（2026-10-24，MIT）。**只读，未编译、未训练**，
> 不占训练核。
>
> 要回答的问题：它 156K infoset 能打到对 Slumbot −13.1 bb/100，我们 1.39M infoset 的
> 表格 MCCFR 可利用度还有 22.7 BB/手。为什么它能成、我们不成？

**一句话结论**：差距不在算法变体，也不在“它的表更大/更细”。它做对的是三件事——
**更小的信息集 key（把底池和跨街历史移出表）+ 每角多 2~3 个数量级的访问量 + 用 argmax
而不是从平均策略采样**；翻后再用 5 秒重解补回被 key 丢掉的底池/范围信息。而且它**从未
公开过可利用度**，“−13.1 bb/100”和“ε ≤ 2 BB/手”不是同一个量，我们当初的判据用错了对象。

> **2026-11-08 R2 实测修正**：只照搬“小 key + 高访问密度”不够。我们把 `potBucket` 与
> 跨街 `hist` 从 key 里去掉（40K 角），访问密度从 115 提到 **746→1492 次/角**，
> ε 反而从 25.70 升到 **47.82→45.41**。访问密度不是充分条件；robopoker 的小 key 能成立，
> 还依赖它的牌力/动作抽象与翻后重解。**单纯瘦 key 只会让抽象更粗、更易被剥削。**

---

## 0. 事实清单

| 项 | robopoker | 我们（M0 最优表） |
|---|---|---|
| 算法 | 外部采样 MCCFR + Linear CFR + Pluribus 剪枝 | 外部采样 MCCFR + **CFR+/RM+** + 线性加权 |
| 牌力抽象 | 翻前 169 全分辨率；翻牌/转牌各 **256**（对下一街桶分布的 EMD k-means）；河牌 **101** 档胜率 | 每街 B 档“当前强度分位 + 听牌修正”（B=8…256） |
| 信息集 key | 街 + **当前街**动作序列 + 牌桶；HU 下 `Field` 零尺寸，**底池/筹码/跨街历史都不在 key 里** | 街 + 牌桶 + **底池桶(32)** + 街内序列 + 座位 + **压缩跨街加注史(8 bit)** |
| 表大小 | **156K** infoset | 16.6K / 260K / **1.39M** |
| 遍历量 | **1.94×10¹⁰ 棵树**（151.7M epoch × 128 树/批） | 60M 次遍历 |
| 每角访问 | **~1.2×10⁵ 次** | **43 次** |
| 部署 | 翻前查表；翻后每决策重解 5 s + 按访问数混合；**argmax 抽取** | 查表 + 温度采样（`--temp`） |
| 可利用度 | **没有公开**（见 §5） | 22.7 BB/手（抽象内抽样 BR 下界） |
| 对 Slumbot | −13.1 ± 14.0 bb/100（86.0K 手，95% CI） | ≈−21 ± 49 bb/100（95% CI，5927 手） |

最后一行很关键：两个点估计差 8 bb/100，而各自的误差都比这个大。**“它能成、我们不成”
这个前提本身并没有被数据支持**，详见 §5.4。

---

## 1. 算法变体：它用的是 Linear CFR（DCFR(1,1,1)），不是 CFR+

旗舰类型的定义就一行：

```rust
// crates/nlhe/src/cfr/flagship.rs:10
pub type Flagship = mccfr::Tabular<NlheEncoder, mccfr::LinearRegret, mccfr::LinearWeight, mccfr::PluribusSampling, 128>;
```

- **采样**：外部采样（external sampling）MCCFR。遍历者节点所有动作全展开；对手节点按当前
  策略采一个动作；机会节点（发牌/发公共牌）直接采样一个分支。**不是 outcome sampling**。
  见 `crates/mccfr/src/sample/schemes/external.rs`、`crates/mccfr/src/sample/mod.rs`
  里的 `Sampling::randomly`（chance 分支）。
- **regret**：`LinearRegret`，即每轮 `R ← R·t/(t+1) + Δ`。这就是 Linear CFR，等价于
  DCFR(1,1,1)，也是 Pluribus 的配置。`crates/forge/src/workers/worker.rs:16-17` 的注释
  写得很直白：“Uses the Pluribus (Brown & Sandholm, Science 2019) configuration:
  LinearRegret (Linear CFR = DCFR(1,1,1)) + LinearWeight.”
- **平均策略**：`LinearWeight`，按 `t·π_i·σ` 累加（`crates/mccfr/src/policy/accumulators/linear.rs`）。
- **CFR+ / RM+**：代码里有 `FlooredRegret`（`regret/accumulators/floored.rs`），但
  Flagship 没用；`DiscountedRegret`（DCFR 1.5/0.5）也有，也没用。所以**没有 CFR+**。
- **regret matching**：普通 RM，负 regret 截 0，带一个 `Probability::MIN_POSITIVE`
  的下限；不是 RM+。见 `crates/mccfr/src/strategy/math/flow.rs:109-110`。
- **剪枝**：`PluribusSampling` 在遍历者节点按 regret 剪枝——阈值 `-3e5`、5% 概率照走、
  前 16384 epoch 不剪（`crates/mccfr/src/hyperparams/pruning.rs`）。
- **批量**：`Tabular<..., B=128>`，一个 `step()` = 一批 128 棵树（128 个采样牌局），
  每个 step 让 epoch +1；遍历者在 epoch 之间交替（`crates/mccfr/src/solver/solver.rs`、
  `crates/forge/src/workers/worker.rs:95-97`）。

**对照我们**：`include/texas/blueprint.hpp:127-189` 的 `mcfrTraverse`/`mcfrTrain` 是
同样的外部采样 + 遍历者交替 + `t·π_i·σ` 线性平均，而且我们的 `addRegretClamped`
（`src/blueprint.cpp:372-376`）把负 regret 截 0，就是 **CFR+/RM+**。所以：

> **算法变体这项我们不落后，甚至理论上收敛更快。别在这上面改，改这里不会解决问题。**

（一个可忽略的差别：它有 regret-based 剪枝，我们没有；剪枝只影响速度，不影响收敛到的点。）

---

## 2. 抽象：重点不是“几个桶”，而是“桶用什么特征”和“key 里放什么”

### 2.1 牌力桶：分层、potential-aware 的 EMD k-means

`crates/pokerkit/src/consts.rs` 和 `crates/lloyd/src/cluster/layer.rs` 给出了全貌：

| 街 | 桶数 | 怎么来 |
|---|---|---|
| 翻前 | **169** | 不聚类，直接用 169 个同构类（`N == K`，identity） |
| 翻牌 | **256** | 每个翻牌同构点表示成“**转牌 256 个桶上的直方图**”，用 **EMD** 距离做 k-means |
| 转牌 | **256** | 每个转牌同构点表示成“**河牌 101 档胜率上的直方图**”，同样 EMD k-means |
| 河牌 | **101** | 不聚类，直接按精确胜率分位切 101 档 |

具体实现：

- 特征不是“当前牌力百分位”，而是**下一街抽象桶上的分布**：`layer.rs` 里
  `Lookup::from_street(street.next()).projections()`，即“这手牌到下一街会变成哪些桶、
  各以多大概率”。这就是 potential-aware abstraction 的核心。
- 距离是 **Earth Mover's Distance**，用 Sinkhorn（熵正则 OT）近似；
  `crates/lloyd/src/hyperparams/sinkhorn.rs` 里温度 0.025、128 次迭代，双向取平均保证对称。
- k-means 用 k-means++ 初始化 + Elkan 加速，32 轮 Lloyd（`crates/lloyd/src/hyperparams/kmeans.rs`），
  随机种子由街号派生，**可复现**。
- 河牌不聚类：`Lookup` 把河牌同构点直接映射到 101 个胜率档（README 说的是四条街合计
  **3.1 T（3.1×10¹²）个同构局面**，河牌只是其中一部分）。这也是为什么河牌抽象表有
  **3.02 GB**（README “Resources”：翻前 4 KB、翻牌 32 MB、转牌 347 MB、河牌 3.02 GB）。

**我们的做法**（`include/texas/strength.hpp:8-25`）：翻前 169 类（这点一致），翻后是
“当前最佳五张的精确百分位 + 听牌 outs 的期望修正”，再按分位切成 B 个桶。差别是：

- 我们看的是**这一街的强度标量**；它看的是**下一街桶的分布**。
- 我们的桶是分位数切分；它的桶是 EMD 空间里的 k-means 质心。
- 结果：同样是 256 个桶，它的桶在“听牌/后续街可玩性”上更有分辨率；我们的桶在
  “当前摊牌价值”上更有分辨率。

这不是“抽象质量应提升”这种空话——它是两种不同的特征空间，翻后河牌的漏洞形状会完全不同。

### 2.2 下注抽象：pot-relative 的固定菜单，每街最多 3 次加注

默认 `Regime::Pluribus`（`crates/kicker/src/betting/regime/mod.rs`、`size.rs`）：

- 翻前开池（唯一一个 BB 相对格）：**2 / 3 / 4 / 5 BB**；
- 其余所有加注都是**底池相对**的固定池（`RAISES` 的 10 个比例里选子集）：
  翻前 3-bet `[1:1, 2:1, 3:1]`、4-bet+ `[1:1, 2:1]`；
  翻牌开池 `[1/4, 1/2, 1:1, 2:1]`、再加注 `[1/2, 1:1]`、之后 `[1:1]`；
  转牌/河牌开池 `[1/3, 1/2, 1:1, 2:1]`、再加注 `[1:1, 2:1]`、之后 `[1:1]`；
- 全下始终在菜单里；`MAX_RAISE_REPEATS = 3`（每街最多 3 次加注，第 4 次不再有菜单）；
- 另有 `Regime::Slumbot`（每个格都是 1/2、1 倍池）做对照，专门最小化对 Slumbot 的平移误差。

**对照我们**：`fold / check-call / 0.5 池 / 1.0 池 / all-in`，每街最多 3 次加注
（`include/texas/blueprint.hpp:30-42`）。我们在首个下注点的尺寸只有 2 个，它有 4 个。
更宽的动作菜单会增加 infoset，但也会显著降低“尺寸抽象误差”。这是一个可以单独做 A/B 的轴。

### 2.3 信息集 key：这才是 156K 的真正来源

`crates/nlhe/src/state/info.rs` + `public.rs` + `crates/nlhe/src/cfr/encoder.rs`：

```
NlheInfo = (street, choices, field, bucket)
           choices = 当前街的抽象动作路径（每次 Draw 重置）
           field   = 按按钮锚定的存活集合 + 行动位置
```

`public.rs:6-9` 明确写着：**“field carries no information at N=2”**，而且
`crates/kicker/src/info/field.rs` 里 HU 的 `Field` 是零尺寸类型（`size_of::<Field>() == 0`）。
也就是说，单挑下它的信息集里：

- **没有底池 / 没有筹码深度 / 没有 SPR**；
- **没有跨街动作历史**（翻牌一开，翻前的下注序列就被丢掉了）；
- 只有：街、这条街的动作序列、牌桶。

所以它的表大小约等于
`169×翻前线数 + 256×翻牌线数 + 256×转牌线数 + 101×河牌线数`，
在几百条线/街的量级下正好是 **10⁵**。

**我们的 key**（`include/texas/blueprint.hpp:55-68`）：

```
street + cardBucket + potBucket(32 档) + 当前街序列 + seat(2) + 压缩跨街加注史(8 bit, r1)
```

`potBucketOf(pot) = floor(log2(pot/150)*4)`，clamp 到 0..31——**理论上最多把表乘 32 倍**；
`hist` 理论上再乘 256 倍。M0 的实测拆分是：桶数 8→256 乘 **15.7×**（16,628 → 260,560），
recall 0→1 再乘 **3.5–5.3×**（取决于桶数；58,010 → 1,387,344 是 5.3×），
底池桶的单独贡献没有消融过（上限 32×）。所以我们的表大，一半来自“桶更细”，
一半来自“key 上挂了底池和召回”。

这不是说我们做错了：底池桶让策略能区分 SPR，压缩召回让策略能记住对手前面街的加注档。
但代价是同样的 6000 万次遍历被摊到 83 倍多的角上。robopoker 的选择是**把这些信息从
表里拿掉，交给翻后重解**（§3）。

一个副作用值得记一笔，而且它有源码注释背书（`crates/spar/src/translate.rs:9-14`）：

> “Our chip stack when playing Slumbot. Hardcoded to 400 (200 BB at `B_BLIND=2`) to match
> Slumbot's 200-BB game. **The blueprint is trained at `STACK=200` (100 BB)**; this override
> only affects the chip math during a Slumbot session … **Strategy lookup degrades gracefully
> via the pot-relative bet abstraction.**”

也就是说：**那张 156K 的表是在 100BB 上训的，直接拿去打 200BB 的 Slumbot**，
靠“所有尺寸相对化 + 不在 key 里放筹码”硬扛。这是一个很激进的抽象，但它就是能跑。
反过来说，我们也不该把“200BB 专属 key”当成必须保留的设计前提。

### 2.4 动作平移

它默认用 `Translation::Snap`（把对手的任意尺寸映射到最近的抽象格），另有
Harmonic/Exact 可选；对完全 off-tree 的尺寸，`Nest` 包装器会做嵌套重解。
我们的动作空间在牌桌上本来就是“任意整数”，部署期不需要平移；但如果要用蓝图直接打
任意尺寸的对手，这一层是必须的。

---

## 3. 翻后不是只查蓝图：每个决策重解 5 秒，再按访问数与蓝图混合

`crates/parlor/src/players/bots/brain.rs` 是完整的决策流程：

1. **翻前**：直接查蓝图，不重解。
2. **翻牌后**：
   - `base`（纯蓝图）：也查表；
   - `depth` / `world`：调用 `solve(...)` 做子博弈重解，**预算 `SubgameHyperParams::timeout_ms`
     = 5000 ms**（`crates/subgame/src/hyperparams.rs`），然后
     `Solved::extract` 把“重解策略”和“蓝图策略”混合。
3. `dirac` 是**抽取方式**，不是搜索：对最终策略取 argmax。

混合公式（`crates/parlor/src/players/bots/resolve/solved.rs`）：

```text
w     = visits / (visits + V),   V = 2^18 = 262144
最终策略 = (1 - w) · 蓝图策略 + w · 重解策略
```

含义：**重解没跑够就几乎完全信蓝图**，跑够了才逐步接管。这是“安全重解”的工程化：
不会因为 5 秒内没收敛就把一个噪声解直接拿去打。

`world` 与 `depth` 的差别（`resolve/world.rs`、`resolve/depth.rs`）：

- `world`（`adapt_safe`）：**安全重解、不设深度上限**，把对手范围按
  `N_WORLDS = 4` 个世界分区，每个世界一棵树；
- `depth`（`adapt_full`）：**安全 + 深度受限**，叶子用**有偏 continuation**
  （`FrontierHyperParams`：bias 5.0、`FRONTIER_LEAVES = 4` 条候选后续策略、
  每对 16 次 rollout），比蓝图先验更接近真实后续；
- `nest`：对 off-tree 尺寸直接做嵌套重解并采纳，用于“对手下了一个菜单外的尺寸”。

注意 `opponent_range`（`crates/nlhe/src/cfr/adapt.rs:120-130`）：它用 **`Witness` 的完整
观测线**去算对手范围的后验——**这正是蓝图 key 里丢掉的跨街信息**。也就是说，
“表里没有跨街历史”并不等于“决策时不知道跨街历史”：重解时用的是真实牌谱，
蓝图只作为先验和 continuation。

**强度怎么拆**（README Table 1）：

| 配置 | 手数 | bb/100 | 95% CI |
|---|---:|---:|---:|
| `world+dirac` | 86.0K | **−13.1** | ±14.0 |
| `depth+dirac` | 86.2K | −25.3 | ±14.3 |
| `dirac`（纯蓝图 argmax） | 480K | −28.4 | ±6.0 |

注意 Table 1 的脚注：**“Every dirac variant beats every non-dirac one: sampling
temperature, not search, is the dominant loss.”** 也就是说，从平均策略“按概率采样”
这个动作本身就输掉了很大一块；搜索/重解是第二位的。这对我们是直接可测的（§6 R3）。

所以回答“它是不是只查蓝图”：**不是，但也不是靠搜索赢的**。纯蓝图 + argmax ≈ −28.4 bb/100，
加上 5 秒重解 ≈ −13.1 bb/100。156K 那张表提供的是一个“足够好的先验 + 剥削基线”，
最终强度是表 + argmax + 重解三者叠加。

---

## 4. 训练规模：1.94×10¹⁰ 棵树，16 vCPU / 120 GB

### 4.1 遍历量怎么换算

- `Tabular<..., 128>`：一个 step = 128 棵树；
- `step()` 里 `profile.advance()` 让 epoch +1（`crates/mccfr/src/solver/solver.rs`）；
- 多进程模式下，`Pool::epoch()` 是**各 worker epoch 求和**（`crates/forge/src/workers/pool.rs:58-59`）；
- README Table 2 的旗舰表：**Epoch 151.7 M、156 K infosets、sum regret 32.1**。

于是总遍历量 ≈ `151.7e6 × 128 ≈ 1.94e10` 棵树。即使把 156K 当作“去重后的角数上限”，
平均每个 infoset 也拿到 **~1.2×10⁵ 次访问**（slow 口径的 infos 是各 worker 求和，真实去重数
只会更小、访问密度只会更高）。

### 4.2 资源与产物

- 训练：**16 vCPU / 120 GB**；PostgreSQL：8 vCPU / 64 GB（README “Resources”）。
- 抽象产物：翻前 4 KB、翻牌 32 MB、转牌 347 MB、**河牌 3.02 GB**。
- **训练时长没有公开**。README 只写了“一次 winrate 评测要几周”（那是 8 万手 Slumbot 的
  外部评测，不是训练）。机时可以直接换算：计划书 §9.3 的实测口径是
  **10 核 ≈ 8.6×10⁹ 迭代/天**，所以 1.94e10 棵树 ≈ **2.3 天 / 10 核**；
  它是 16 vCPU，若每核吞吐接近，约 **1.5 天**。这不是“它算力多”，而是
  “它的表小、训得久”——同一台 M1 Pro 也付得起，前提是表压到 10⁵。

### 4.3 我们差在哪

- 我们：60M 次遍历 / 1.39M 角 = **43 次/角**。
- 它：1.94e10 次遍历 / 156K 角 = **~1.2×10⁵ 次/角**。
- 比值 **~2900×**，不是 2 倍、不是 10 倍。M0 的 30M→60M 只把 ε 从 22.69 降到 22.24，
  在这个访问密度下是完全正常的——CFR 的收敛是 O(1/√T)，43 次访问本来就不该指望收敛。

**结论：M0 证伪的是“我们的 key + 60M 遍历”，不是“表格 CFR”。**
**R2 进一步定死：即使把访问密度提到 746–1492 次/角，瘦 key 的 ε 仍是对照的 1.8–1.9 倍——
瓶颈在 key 的信息量，不在遍历量。表格 CFR 路线已于 2026-11-08 关闭（见迭代账本 R2）。**

---

## 5. 可利用度：正面回答

### 5.1 它公开了吗？**没有。**

三条证据：

1. **训练日志里叫 “exploitability” 的字段，其实是 sum regret。**
   `crates/forge/src/fast.rs:129-131`：

   ```rust
   let e = sum_regret();
   *exploit = e;
   tracing::info!(exploit = e, "exploitability")
   ```

   README Table 2 里的 “sum regret 32.1” 就是它。**sum regret 不是可利用度**，
   量纲是“regret 质量”，不能和 BB/手 或 bb/100 比。

2. **精确 BR 只在小游戏上跑。** `mccfr::Grower::exploitability()` 会从 `exploitability_root()`
   建整棵树做精确 BR；只有 Kuhn / Leduc / RPS 这类玩具游戏 override 了
   `exploitability_root`。NLHE 没有 override，所以 `mxploitability()`（Monte Carlo 版）
   也没有接到 NLHE 上——`mc_exploitability_samples = 1024` 这个配置项实际是**死代码**
   （全仓库只有定义，没有使用）。

3. **最接近的公开量是“局部下界”，不是全局 ε。** `crates/phh` 有一个 `Ledger`/`duel` 工具，
   能在 Pluribus 公开牌谱的探针点上给“可利用度的下界”，但只覆盖那些点，
   而且没有公开旗舰蓝图的具体 δ 值（只给了“always check/call ≈ +0.6188 bb/decision”的标尺）。

所以对“robopoker 的可利用度是多少”这个问题，正确答案是：**不知道**。

### 5.2 我们的 22.7 是什么

`include/texas/blueprint_br.hpp` 写得很清楚：`blueprint-br` 是**抽样最优应对**，
用的是同一套抽象、同一套桶观测、同一套抽象动作。因此它给的是**抽象游戏内**的
可利用度，而抽象游戏内的 BR 也是真实游戏里的合法策略，所以它是**真实可利用度的下界**。

即：我们的蓝图真实 ε **至少** 22.7 BB/手。

### 5.3 判据错在哪：三处对象错配

我们当初在 `docs/CFR_BLUEPRINT_PLAN.md:361` 写的是：

> ε ≤ 2 BB/手，且 30M→60M 仍在下降；否则关闭 CFR 路线。

问题不在数字大小，而在**用“接近均衡”去判“能不能赢一个固定对手”**：

1. **目标错配。** ε 是“对最坏对手的保证”；而我们的目标是打赢 Slumbot——一个固定的、
   非 BR 的 2018 年机器人。一个策略可以 ε 很大、同时把 Slumbot 剥削得很惨
   （反之亦然）。这两件事没有单调关系。
2. **对象错配。** 我们的 22.7 是“在我们的抽象里”的下界；robopoker 的表如果按同样口径
   测，大概率**也远高于 2**——它连底池都不在 key 里、河牌只有 101 档、跨街历史全丢。
   作为参照，我们 `B=256 r0`（260K infoset，只有桶、没有跨街召回）的 ε 是
   **25.70**。所以“ε ≤ 2”这条门**同样会把 robopoker 的 156K 表拒掉**，而那张表实际
   打到了 −13.1 bb/100。这就是“误杀”的直接证据。
3. **不对称执行。** 这条门只对表格 CFR 生效，对我们**实际部署**的 PPO 路线从没测过。
   `stn2` 是一个 PPO + 脚本教师 + 联赛的模型，没有任何理由认为它的 ε ≤ 2
   （它甚至没有抽象动作限制，BR 只会更容易）。**我们用一条自己都没达标的门，杀掉了
   另一条路线。**

补充两条旁证：

- robopoker 自己的 litmus 有 6 项失败，`README:120` 把原因写得明明白白：
  “flop k-means merges `AQo` through `A5o` into one bucket while `AK` gets its own,
  so CFR routes kicker strength through the jam.” 一个**已知有抽象缺陷**的表，
  照样能打。
- 它的 `phh` 文档（`crates/phh/src/doc/agree.md:4-5`）直接写：
  **“Pluribus is not correctness — Pluribus is not a Nash equilibrium.”**
  它自己就不追求 Nash；我们却拿 Nash 的距离当准入门槛。

### 5.4 但也不要过度解读

“它能成、我们不成”在数字上并不成立：

| | 点估计 | 95% CI | 样本 |
|---|---:|---:|---:|
| robopoker `world+dirac` | −13.1 bb/100 | ±14.0 | 86.0K 手 |
| robopoker `dirac`（纯蓝图） | −28.4 bb/100 | ±6.0 | 480K 手 |
| 我们 `stn2` + T=0.5 | ≈ −21 bb/100 | ±49 | 5.9K 手 |

（我们原始读数是 −0.21 ± 0.25 **BB/手（SE）**，换算成 bb/100 的 95% CI 是 ±49。）

- 我们的点估计落在它的“纯蓝图”和“蓝图+重解”之间；
- 三者的误差都大于彼此的差值（7–8 bb/100）；
- 它真正的优势是**样本量**（86K vs 5.9K）和**管线完整度**（抽象→训练→重解→argmax 一条龙），
  不是“156K 的表比 1.39M 的表强”。

### 5.5 裁决

- `ε ≤ 2 BB/手` 作为**采纳门**是定错了：它既不是目标函数，也不是同一个对象，
  而且只对一条路线执行。M0 的结论应该改写为：
  **“我们的 key + 60M 遍历 + 温度采样，在 M1 Pro 上不收敛到可用”**，
  而不是“表格 CFR 这条路线不行”。
- 但“重启”不等于把 60M 加到 600M。要按它验证过的配方改 key、改抽象、改抽取方式，
  再决定要不要付 1.94e10 棵树的钱。

---

## 6. 可借鉴的具体做法（按“能动手”排序）

### R1. 把 ε 从“门”降级为“趋势指标”，并给 stn2 补一次同口径测量（今天就能做，0 算力）

- 改 `docs/CFR_BLUEPRINT_PLAN.md` §9.1：删掉“ε ≤ 2 否则关闭”，改成两级：
  - **诊断**：看 ε 随“每 infoset 访问数”的斜率（这是 M0 真正该看的轴，而不是“桶数”）；
  - **采纳**：外部 12k 手 ≥ 基线 + 2σ（现成的慢速环口径）。
- 代码：`cmdBlueprintTrain` 的日志每 5M 迭代加一行 `iters / tab.size()`；
  `blueprint-br` 输出同一个口径。这样下一次能直接回答“是访问量不够，还是抽象限制”。
- 预期收益：不再用一条“我们自己都没达标的门”杀掉路线；也让 M0 的结论可被正确复述。

### R2. 做一次“瘦 key” A/B：去掉底池桶 + 跨街 hist，把表压到 10⁵（半天，1 组对照）

- 改 `include/texas/blueprint.hpp:67` 的 `packInfoset`：新增 `--key-slim` 开关，
  打包时**跳过 `potBucket`（32 档）和 `hist`（8 bit）**，只保留
  `street + cardBucket + 当前街 seq`（座位可由 seq 长度奇偶隐含，先保留 1 bit 更稳）。
- 用 `B=256`、30M/60M 各跑一遍，对照现有 `b256_r0`（260K 角、ε=25.70）。
  预期：同 60M 下 infoset 降到 10⁴–10⁵，每角访问从 43 升到 10³ 量级；
  如果 ε 随访问密度明显下降（而不是像 M0 那样几乎平），就说明“key 太胖”是主因。
- 风险与判读：底池桶承载 SPR 信息，拿掉后如果 ε 不降反升，说明**必须配 R3 的重解**
  来补底池/范围，而不是把 pot 塞回 key。这正是 robopoker 的取舍。

### R3. 抽取改 argmax；给河牌重解加“访问数混合”（1 天，两个独立开关）

- **argmax**：`cmdBlueprintEval` 现在把温度写死在 `src/main.cpp:4988`
  （`sampleBlueprintMove(..., 1.0f, recall)`）。把它提成 `--temp` 参数，先跑
  `--temp 0.05`（`sampleBlueprintMove` 内部 `max(temp, 0.05)`，等于概率的 20 次幂，
  近似 argmax）；要严格 argmax 就在函数里加一个 `temp <= 0 → argmax` 分支，一行。
  robopoker 的证据是“所有 dirac 变体都优于非 dirac”，这是**最大的一跳**（−28.4 vs 更差）；
  我们现在的查表是温度采样，值得先花 1 小时测这个。
- **重解混合**：`riverCfr` 现在是“硬接管”（`--river-cfr` 直接换策略），这也是它当年被
  否决的原因之一（接管 12% 时 −0.69）。改成 robopoker 的公式：
  `新策略 = (1-w)·蓝图 + w·解`，`w = visits/(visits + 2^18)`。
  先在内部 H2H 上验证，再决定要不要接外部。转牌同理。
- 预期：如果 argmax 在我们这也有一跳，说明“部署期采样”一直是一个被忽略的损失项；
  如果重解混合能救回河牌，就补上了我们 §10.3 里“河牌需要完整跨街范围推断”的那个洞
  （它的解法不是更细的召回，而是用真实 history 做重解）。

---

## 7. 读码索引（证据位置）

| 主题 | 文件 |
|---|---|
| 旗舰类型 / 算法配置 | `crates/nlhe/src/cfr/flagship.rs:10`；`crates/forge/src/workers/worker.rs:16-17` |
| LinearRegret / FlooredRegret / DiscountedRegret | `crates/mccfr/src/regret/accumulators/{linear,floored,discounted}.rs` |
| 线性平均策略 | `crates/mccfr/src/policy/accumulators/linear.rs` |
| RM（负 regret 截 0） | `crates/mccfr/src/strategy/math/flow.rs:109-110` |
| 剪枝参数 | `crates/mccfr/src/hyperparams/pruning.rs` |
| 外部采样 / chance 采样 | `crates/mccfr/src/sample/schemes/external.rs`、`crates/mccfr/src/sample/mod.rs` |
| batch=128 / epoch / worker 交替 | `crates/mccfr/src/solver/{solver,tabular}.rs`、`crates/forge/src/workers/worker.rs:95-97` |
| 桶数与尺寸 | `crates/pokerkit/src/consts.rs` |
| 分层 EMD k-means | `crates/lloyd/src/cluster/layer.rs`、`crates/lloyd/src/distance/*`、`crates/lloyd/src/hyperparams/{kmeans,sinkhorn}.rs` |
| 信息集 key / 无底池 | `crates/nlhe/src/state/{info,public}.rs`、`crates/kicker/src/info/field.rs` |
| 下注菜单 | `crates/kicker/src/betting/{regime,size}.rs` |
| 翻后重解 / 混合 | `crates/parlor/src/players/bots/brain.rs`、`resolve/{solved,world,depth,nest}.rs` |
| 重解预算 | `crates/subgame/src/hyperparams.rs`、`crates/subgame/src/depth/frontier/hyperparams.rs` |
| “exploitability”=sum regret | `crates/forge/src/fast.rs:129-131` |
| 只在小游戏算精确 ε | `crates/mccfr/src/strategy/math/equilibrium.rs`、`crates/mccfr/src/solver/grower.rs` |
| 评测数字 / 资源 / 桶大小 | `README.md` Table 1 / Table 2 / Resources |
| 不追求 Nash | `crates/phh/src/doc/agree.md:4-5` |

---

## 8. 边界（这份笔记没证明什么）

1. 没有编译、没有训练、没有复现它的 −13.1。所有强度数字都来自它的 README。
2. 它的 156K / 151.7M 是 README 口径；slow 模式下 `infos` 是各 worker 求和，
   真实去重角数可能更低（访问密度更高），但结论方向不变。
3. 它的可利用度是**未知**，不是“很大”。§5.3 的“大概率 > 2”是从抽象结构
   （无底池、101 档河牌、无跨街历史）和我们的同类测量推出来的，不是直接读数。
4. 我们的 22.7 是**下界**，不是精确值；而且它是在我们的抽象游戏里测的。
