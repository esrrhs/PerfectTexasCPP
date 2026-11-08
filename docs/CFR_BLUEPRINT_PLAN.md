# CFR Blueprint 设计方案计划书

> 2026-11-07 立项。用户已确认串行路线：**表格 MCCFR blueprint → 监督蒸馏进 actor → 验收**。
> 本文件是该管线的唯一执行口径。决策依据（表格 MCCFR vs Deep CFR 的对比）见 README 第七节。
> 纪律沿用 README 第四节：外部只看全下 EV 调整、12000 手、2σ、单会话串行 delay≥150ms。
>
> **2026-11-07 二次复核见 §9**：把"追上主流"重定义为可裁决判据，收紧 G1（0.5 → 0.15 BB/手），
> 并给出 4~6 小时止损实验（M0）。§1–§8 保留为第一次立项的原始设计与实测记录。
>
> **2026-11-08 M0 已执行：未通过，表格 CFR blueprint 路线关闭，见 §10。**
> 现役部署不变（stn2 + T=0.5）；备选方向（§9.5）需用户另行决策。

---

## 0. 目标与验收门

造出 200BB 单挑无限注德州的离线 CFR blueprint，蒸馏进现有 actor，在外部基准上追平/超过现役。

| 门 | 内容 | 判据 |
|---|---|---|
| **G1 蓝图质量** | 抽样最优应对（BR）测蓝图可利用度 | 双座位 BR 之和 ≤ **0.5 BB/手**，且随迭代单调下降 |
| **G2 蒸馏保真** | 蒸馏后 actor 内部强度 | 留出状态 top-1 一致率 ≥ 70%；内部锚点 rule/random/pro/search/solid/mix **全正** |
| **G3 外部裁决** | Slumbot 12000 手、全下 EV 调整 | 点估计 ≥ **−0.2 BB/手**，且相对本代平台基线（−0.5）改善 ≥ **2σ** |

stretch 目标：G3 点估计 ≥ 0（真正追平 Slumbot）。任一内部门不过则不进下一门；外部不达标只记录、不强行采纳。

---

## 1. 抽象设计

### 1.1 牌面抽象（复用 `StrengthModel`，已实现已测试）

- **翻前**：169 类全分辨率（`preflopClass`），不分桶。
- **翻后**：`postflopBucket(hole, board, boardCount, B)`，默认 **B = 8**/街。
  该分桶按牌型分配桶位 + 听牌潜力修正（花听/顺听不被压在底层），正是 CFR 需要的牌力抽象。
- **召回**：当前街桶 + 当前街完整下注序列 + 量化底池（见 1.3）。**跨街 imperfect recall**
  （忘记前街确切动作，用底池桶代理历史投入）。这是标准近似；若 G1 不达标，升级路径是
  把前街桶序列并入 key（perfect recall），内存翻 ~8×/街。

### 1.2 动作抽象

每街抽象动作集（最多 5 个）：

| id | 动作 | 金额 |
|---|---|---|
| 0 | fold | 仅面对下注时合法 |
| 1 | check / call | 补齐缺口 |
| 2 | raise 0.5×pot | 跟注后底池的 0.5 倍 |
| 3 | raise 1.0×pot | 跟注后底池的 1.0 倍 |
| 4 | all-in | 身后全押 |

- 加注额定义：设缺口 `gap = toCall`，跟注后底池 `P' = pot + gap`；
  `raiseTo = 本街已投入 + gap + frac × P'`，clamp 到 `[minRaiseTo, maxRaiseTo]`；
  顶到上限退化为 all-in；若尺寸 < 最小加注则并入最小加注。
- **每街加注上限 R = 3**（raise 与 all-in 都计数）；达到上限后只剩 fold/check-call。
  真实游戏无上限 → 部署时把超树动作翻译回最近抽象动作（见 4.3）。
- 翻前用同一套 pot-frac 机制（pot=150），自然落在 ~2–3×BB  open，不另设档位。

### 1.3 信息集索引（精确打包，无哈希碰撞）

key（uint64，唯一）= `cardBucket | potBucket<<4 | street<<9 | seqLen<<11 | seqPacked<<15`：

- `cardBucket`：翻前 0–168（占 8 bit，实际 key 低位留 8 bit），翻后 0–7。
  统一留 **8 bit**（翻前 169 需要）。
- `potBucket`：`clamp(floor(log2(pot/150) × 4), 0, 31)`，**5 bit**（四分之一倍频程量化底池）。
- `street`：2 bit。
- 当前街抽象动作序列：每动作 3 bit，长度 ≤ 8 → `seqLen` 4 bit + `seqPacked` 24 bit。

key → 稠密 idx 用**开放寻址哈希表**（存 full key 校验，负载因子 0.7）。
regret / strategy-sum 按 idx 稠密存 `float[idx × A]`。合法抽象动作只依赖下注字段
（pot/contrib/stack，与牌无关），所以每个局面的动作集可由 `HUState` 直接算。

### 1.4 内存预算

infoset 量级 ≈ 4 街 × 32 底池桶 × ~200 街内序列 × 169/8 牌桶 ≈ **10⁵–10⁶**。
每 infoset `A=5 × (regret 4B + stratSum 4B) = 40B` → **4–40 MB**；哈希表 ~2× → 总 **< 200 MB**。
32GB 统一内存极充裕。若 perfect-recall 升级或 B 加大，仍 < 2GB。

---

## 2. MCCFR 算法

- **External-sampling MCCFR**：机会节点采样（发底牌/公共牌），对手节点按当前策略采样一个动作，
  己方（traverser）节点递归全部动作。每次迭代只更新 traverser 的 regret。
- **Regret Matching+**：regret 截断 ≥0；策略 = 正遗憾归一，全零则均匀。
- **线性加权平均策略**：在采样到的对手节点累加 `stratSum[a] += t × σ[a]`
  （采样频率自带到达概率权重，external-sampling 的标准简化）。Pluribus 同款 linear weighting。
- **traverser 选择**：每次迭代交替 0/1。
- **正确性金标准**：MCCFR 核心做成游戏无关模板 `mcfrTrain<Game>`；先用 **Kuhn poker**
  （3 张牌，已知纳什家族）验证可利用度 → 0，再跑 HUNL。
- **并行**：先单线程验证正确；性能不足时线程池共享 regret 表（atomic add）。
- **检查点**：每 N 迭代存 regret + stratSum + 哈希表 + 迭代数到
  `/Users/mingming/models/blueprint_v1/`（禁止 `/tmp`）。

---

## 3. BR 可利用度评估（G1 工具）

- 抽样公共状态：随机发（BR 座位底牌 + 完整 5 张公共牌 runout）；对手底牌 = 1326 维向量
  （与公共牌/我方底牌冲突的 combo 清零）。
- 每个抽样在抽象下注树上**全宽遍历**：BR 节点取 `max_a`；对手节点对每个 combo 查 blueprint
  策略（bucket 由对手 hole+board 算）按概率加权；机会已固定。复用 `rivercfr` 的 reach 向量
  + O(1) blocker 查询技术。
- `BR value` = 该抽样下 BR 座位的精确最优 EV。`exploitability ≈ E[BR₀] + E[BR₁]`
  （两人零和纳什值和 = 0），筹码/手 ÷100 = BB/手。N=2000 抽样报均值 ± SE。
- 这是抽样 local-best-response 估计（非全局精确 BR），足以把 G1 门。

---

## 4. 蒸馏管线（G2）

### 4.1 数据生成
blueprint 自对弈 + 随机发牌混合采样 `HUState` 决策点；每点查 blueprint 抽象动作分布。

### 4.2 槽映射（abstract → 19 槽）
- fold → `kSlotFold`；check/call → `kSlotCheckCall`；all-in → `kSlotAllIn`。
- raise 0.5/1.0 pot → 在 `legalMoves` 里找覆盖该金额的尺寸带槽，概率质量放到该槽。

### 4.3 训练与部署翻译
- 新子命令 `blueprint-distill`：复用 `features`/`buildBatch`/`learner`，loss = `KL(target || actor)`
  （纯监督，默认不混 PPO）。从 `stn2` 初始化。
- 部署翻译（blueprint 直接查表路径 / 联赛锚点用）：对手真实下注额 → 最近抽象动作
  （按金额落带）；blueprint 输出抽象动作 → 真实金额（带内代表值）。超树加注按最近尺寸归类。

### 4.4 保真验收
留出状态 top-1 一致率 ≥ 70%、KL ≤ 0.1（实测调）；内部锚点全正（硬约束）。

---

## 5. 实现拆解（文件级）

| 文件 | 内容 |
|---|---|
| `include/texas/blueprint.hpp` / `src/blueprint.cpp` | 抽象动作枚举/金额翻译、infoset key 打包、开放寻址索引表、regret/stratSum、`mcfrTrain<Game>` 核心、HUNL adapter、检查点 save/load、`queryStrategy(HUState)` |
| `include/texas/blueprint_br.hpp` / `src/blueprint_br.cpp` | 抽样 BR 评估 |
| `src/main.cpp` | 子命令 `blueprint-train` / `blueprint-br` / `blueprint-distill` / `blueprint-eval` |
| `tests/test_main.cpp` | 抽象映射合法性 + 翻译往返、key 唯一性、**Kuhn 收敛到已知纳什**、BR 在已知策略上的值、蒸馏槽映射 |
| `CMakeLists.txt` | 加入新 src |

---

## 6. 里程碑

| # | 交付 | 验收 |
|---|---|---|
| M1 | 抽象层 + 索引 + 游戏无关 MCCFR 核心 + Kuhn/HUNL adapter | Kuhn 可利用度 → 0；单测全绿 |
| M2 | HUNL 训练器 + 检查点 + 短跑 | regret/BR 随迭代下降 |
| M3 | BR 工具 + 长训（后台机时） | **G1** ≤ 0.5 BB/手 |
| M4 | 蒸馏 + 内部锚点 | **G2** 全正 |
| M5 | 外部 12000 手 | **G3** 裁决 |

---

## 7. 风险与回退

- **抽象过粗**（G1 > 0.5）：升 B/R 或 perfect recall，重训。
- **蒸馏损耗大**（G2 锚点负）：改槽映射 / 加带对照的 PPO 微调臂。
- **内部好外部差**：记录不采纳（内外脱钩纪律）。
- **机时超预算**：先降抽象出粗蓝图跑通全管线，再细化。
- **imperfect recall 偏差**：底池桶代理历史可能丢攻击性信息；G1 不达标时优先升级召回。

---

## 8. 执行结果（2026-11-07）

M1–M5 全部实现并跑通；**G1 未过、G2 的 KL 门未过、G3 未裁决，按纪律不采纳，部署配方不变**。
本节是实测口径，所有数字可复现。

### 8.1 交付清单

| 文件 | 内容 |
|---|---|
| `include/texas/blueprint.hpp` / `src/blueprint.cpp` | 抽象动作与金额翻译、info-set key（含座位）、开放寻址 RegretTable + 存盘、游戏无关 `mcfrTrain<Game>`、HUNL adapter、`queryBlueprint` / `abstractToSlots` / `sampleBlueprintMove`、超树动作翻译 |
| `include/texas/blueprint_br.hpp` / `src/blueprint_br.cpp` | 抽样 BR：1326-combo reach、按牌力桶分组查表、街切换重映射、抽象树全宽遍历、`--self-play-check` 自洽校验 |
| `src/main.cpp` | `blueprint-train` / `blueprint-br` / `blueprint-distill`（留出集早停 `--eval-every`/`--patience`、同口径复测 `--eval-only`）/ `blueprint-eval` |
| `src/learner.cpp` | `Learner::distill(..., qIsTarget)`：精确 `KL(target‖actor)` 监督分支 |
| `tests/test_main.cpp` | Kuhn 纳什收敛、key 唯一性、槽映射、BR 自洽、检查点往返等；**45236 条断言 0 失败** |

检查点目录格式：`bp.bin`（regret/平均策略）+ `meta.txt`（迭代数/B/R/seed）+ `rng.txt`
（mt19937_64 全状态，续训逐位可复现）+ `snapshots/bp_<iters>.bin`。
全部写在 `/Users/mingming/models/`，未使用 `/tmp`。

### 8.2 验收门读数

| 门 | 判据 | 实测 | 结论 |
|---|---|---|---|
| **G1 蓝图质量** | BR0+BR1 ≤ 0.5 BB/手，且随迭代下降 | `blueprint_v3`（B=8 R=3, 6M iters, 16558 infosets）：**30.320 ± 1.876 BB/手**（N=200 抽样, 74.8M 节点）；快照 6M→26M 为 28.2→28.7 BB/手（N=20, 同 seed），**无下降** | ❌ **FAIL** |
| **G2 保真** | top-1 ≥ 0.70、KL ≤ 0.1 | 四个版本同留出集（1009 点 @seed 20261107；v4 为 1884 点 @seed 20261109）对照：v1 0.661/0.530、v2 0.593/0.865、v3 0.679/**0.439**、**v4 0.717/0.373**（并列容忍 0.844、H(target) 0.952、KL 占目标熵 39.2%） | ⚠️ **top-1 PASS / KL FAIL** |
| **G2 锚点** | rule/random/pro/search/solid/mix 全正 | v4 @ 4000~8000 手：**+847 / +574 / +367 / +35 / +221 / +321** 筹码/手（v1 为 +622 / +788 / +224 / +257 / +129 / +173），**六项全正** | ✅ PASS |
| **G3 外部** | 12000 手、全下 EV 调整、≥ −0.2 且 2σ | **未裁决（机时受限）**：`bp_distill_v1` 两段共 **958 手**（s1 358 + s2 600），全下 EV 调整后 **−0.688 ± 0.384 BB/手**（95% CI −1.44 ~ +0.07）；点估计明显低于 −0.2，但 CI 覆盖判据、样本量不足，且差于 `stn2` 的 −0.21 ± 0.25（差距 −0.478，z=−1.04）。单会话实测 ~0.3 手/秒，12000 手需 ~11 小时 | ⏳ 未裁决 |

补充读数：`blueprint-eval` 查表策略对 `rule` 2000 手 **+8.13 ± 1.87 BB/手**，查表命中 95.4%
（蓝图不是「坏到不能用」，但离均衡级可利用度差两个数量级）。

### 8.3 G1 未过的根因（已排除的假设）

BR 工具先通过了自洽校验，排除了工具本身的问题：

- `--self-play-check`（BR 座位也按蓝图策略走，双方都用同一张表）：
  **−2.2 ± 1.2 BB/手**，与零和一致；
- 空表（均匀策略）BR = **85.1 BB/手**，6M 训练表 = **28.3 BB/手** ⇒ 训练确实大幅改善策略；
- Kuhn 金标准：正确 reach 加权 BR 下可利用度 **0.0043**、BR0 = **−0.0527**（纳什 −1/18）。
  原单测的 Kuhn BR 对每对具体底牌单独取 max，等于让 BR 偷看对手底牌，已修正。

逐一排除的实现级假设：

| 假设 | 实验 | 结果 |
|---|---|---|
| 平均策略累接口径不标准 | 改为标准 external-sampling（遍历者节点累加 `t·π_i·σ`） | BR 28.2 → 28.3，无变化 |
| infoset 未区分座位 | key 加 `toAct` bit | BR **逐位相同**（座位已由 street+seq 隐含） |
| 训练量不足 | 6M → 10M → 16M → 20M → 26M 迭代快照 | 28.2 → 28.7，**平台期** |

**结论**：这是抽象表达能力的上限，不是收敛问题。蓝图只有 **16558 个 infoset**
（计划书预估 10⁵–10⁶），且跨街 imperfect recall：策略查表时看不到完整下注历史，
也不维护对手范围。BR 则持有 1326 维 combo reach，随每个动作做贝叶斯范围更新，
系统性剥削「同一 key 下被合并的不同历史」。计划书的回退路径（升 B/R、
perfect recall 把前街桶序列并入 key）依然正确，但要把可利用度从 30 BB/手 压到
0.5 BB/手，需要 10⁷–10⁸ infosets 与完整跨街下注序列，属更大规模工程，本轮未投入。

### 8.4 蒸馏踩坑（有复用价值）

1. **`stn2` 的 actor logits 实测高达 +500 / −280**（PPO 训出的极尖策略）。
   直接做 `KL(target‖actor)` 时 softmax 完全饱和，CE 达 37 / 284、梯度几乎推不动。
   解决：只把 `head` / `auxAct2` / `oppAct2` / `rngAct2` 四个**输出末层**乘 0.01
   （`--logit-scale`），保留 LSTM、l1–l4、信念头的全部表示；CE 降到 1.4 / 1.1，
   top-1 从 0.30 提到 0.66，KL 从 1.95 降到 0.53。
2. `seqFromState` 增加**超树动作翻译**：真实对手的任意整数下注额按「同类型 +
   金额最近」落回抽象动作，否则 `seq` 会错位、查表落到不相干的信息集。
3. `Learner::distill` 新分支用 `CE(target, actor) = KL(target‖actor) + H(target)`，
   `H(target)` 与参数无关，因此最小化 CE 就是精确最小化 KL。
4. **`distill` 的一个 step 等于一个 epoch（不是一条 minibatch）**。它内部先 `randperm(B)`
   再遍历全部 batch，所以 `--steps 1500` 是把 3500 条样本过 1500 遍。v2 的
   「数据更多（2000 局）、步数更大（1500）、lr 更高（5e-4）」组合直接过拟合：
   top-1 从 0.661 掉到 0.593、KL 从 0.530 升到 0.865。修法是**按留出集分块选优**
   （`--eval-every 50 --patience 3/4`，取 KL 最小的快照并保留），v3 在同样 1200 局上
   从 0.661/0.530 提到 0.679/0.439，v4 再加数据到 2400 局得到 0.717/0.373。
   经验：**低 lr（2e-4）+ 早停** 优于「大 lr + 长训」，有效步数落在 100–350 个 epoch。
5. **`--eval-only` 必须在 logit 缩放之前返回**。第一版把提前返回写在缩放之后，
   重测时对已训练好的头又乘了一次 0.01，导致 top-1 完全一致而 KL 从 0.530 虚高到
   1.292（softmax 被压平、argmax 不变）。多版本对照表必须同口径，这个 bug 靠
   「同一检查点应复现同一读数」才暴露出来。

### 8.5 复现命令

```bash
# M2/M3：训练 + BR（检查点写 ~/models，禁止 /tmp）
./build_rl/perfecttexas blueprint-train --iters 6000000 --save-every 3000000 \
  --log-every 3000000 --out /Users/mingming/models/blueprint_v3 --seed 20261107
./build_rl/perfecttexas blueprint-br --bp /Users/mingming/models/blueprint_v3 \
  --samples 200 --seed 424242
./build_rl/perfecttexas blueprint-br --bp /Users/mingming/models/blueprint_v3 \
  --samples 20 --seed 99001 --self-play-check     # 自洽校验

# M4：蒸馏 + 保真（低 lr + 留出集早停；v4 为验收口径）
./build_rl/perfecttexas blueprint-distill --bp /Users/mingming/models/blueprint_v3 \
  --resume /Users/mingming/models/stn2 --out /Users/mingming/models/bp_distill_v4 \
  --games 2400 --holdout 600 --steps 800 --lr 2e-4 --eval-every 50 --patience 3 \
  --seed 20261109
# 同口径复测任一检查点（只测不训）
./build_rl/perfecttexas blueprint-distill --bp /Users/mingming/models/blueprint_v3 \
  --resume /Users/mingming/models/bp_distill_v4 --out /Users/mingming/models/bp_distill_v4 \
  --games 2400 --holdout 600 --seed 20261109 --eval-only

# 内部锚点 / 蓝图查表策略
for opp in rule random pro search solid mix; do
  ./build_rl/perfecttexas eval --resume /Users/mingming/models/bp_distill_v4 \
    --vs $opp --hands 4000 --seed 90001 --temp 0.5
done
./build_rl/perfecttexas blueprint-eval --bp /Users/mingming/models/blueprint_v3 \
  --vs rule --hands 2000 --seed 90001

# M5：外部分段（单会话串行、delay ≥150ms，禁止并发）
./build_rl/perfecttexas slumbot --resume /Users/mingming/models/bp_distill_v1 \
  --hands 1500 --temp 0.5 --delay-ms 150 --seed 96001 --log /tmp/bp_ext_s1.csv
```

### 8.6 纪律结论

- **G1 未过 ⇒ 按「任一内部门不过则不进下一门」不进入 G3 裁决**；外部 958 手
  （−0.688 ± 0.384 BB/手，95% CI −1.44 ~ +0.07）只是机制性读数，样本量不足以
  裁决 0.2BB 级改动，更不构成采纳依据。
- 内部锚点全正再次印证 README 的「内外脱钩」：锚点只能防机制性崩坏，不能用来采纳。
- G2 的 top-1 门靠「2400 局 + 低 lr + 早停」已过（0.717），**KL 门仍差 3.7 倍**
  （0.373 vs 0.100）。KL 是分布级指标，target 是桶级蓝图分布（H=0.952 nats，
  并列容忍 top-1 0.844），把 19 槽的完整分布拟合到 0.1 以内需要更大网络/更多数据；
  在 G1 未过时继续投入不改变结论，故停在这里并如实记为未过。
- 蒸馏模型锚点仍低于 `stn2`（v4 rule +847 / solid +221 vs stn2 +1702 / +564），
  说明把粗蓝图灌进 actor 会损失现有强度，**不替换部署配方**。
- 现役部署仍为 **`stn2` + T=0.5**（`--range-eq 0`），外部 −0.21 ± 0.25 BB/手（5927 手截断样本）。
- 下一步若继续该方向：按 §7 升级 perfect recall 与牌力桶，把 infoset 提到 10⁷ 量级后重训，
  再跑 G1；在此之前不要重新蒸馏或占用外部通道。

---

## 9. 二次立项（2026-11-07 复核）：判据、可行性、预算

用户新目标：**继续迭代，赢面追上主流算法**。本节只回答三件事——目标怎么量、这条路能不能到、
要花多少机时。**结论先行：按原方案再投 1~2 周不划算；先做一个 4~6 小时的"抽象斜率"实验（M0），
不通过就关闭这条线。**

### 9.1 把目标变成可裁决的判据

**"主流"在这里的操作化定义**：Slumbot —— ACPC 2017/2018 冠军级、纯蓝图无实时搜索的公开机器人，
也是本机唯一可复现的外部锚点。ACPC 只公布两两对局结果，**没有公开的"中位名次对应 BB/手"可直接引用**，
所以不拿编造的联赛中位数当目标，只认"对 Slumbot 的调整后收益"。

- **达标定义**：对 Slumbot 全下 EV 调整后 **≥ 0 BB/手**（统计上可宣称）。
- 绝对口径（可利用度）无法与 Slumbot 对比（其策略未公开），只作内部门。

**分辨率是硬约束**。由 12000 手 SE 0.19 反推单手噪声 σ ≈ 20 BB/手：

| 样本 | SE (BB/手) | 2σ 能裁决的差距 | 串行外部机时（0.3 手/秒） |
|---|---|---|---|
| 12 000 | 0.19 | 0.38 | ≈ 11 h |
| 24 000 | 0.13 | 0.26 | ≈ 22 h |
| **36 000** | **0.11** | **0.22** | **≈ 33 h** |

| 门 | 原判据 | 新判据（§9 起生效） |
|---|---|---|
| **G1 蓝图质量** | 可利用度 ≤ 0.5 BB/手 | **≤ 0.15 BB/手**（见下） |
| **G2 蒸馏保真** | top-1 ≥ 0.70、KL ≤ 0.10、锚点全正 | 不变 |
| **P1 外部裁决** | 12 000 手、≥ −0.2 且 2σ | **36 000 手，点估计 ≥ +0.22 BB/手**（单侧 95% 下界 > 0） |
| **P0 方向门** | — | 12 000 手，点估计 **≥ +0.10**（1σ 正向），只决定是否扩样 |

**为什么 G1 必须从 0.5 收紧到 0.15**：可利用度 ε 是最坏情况下每手的损失上界。若 ε = 0.5，
对接近均衡的 Slumbot 最多只能打到 −0.5 量级；要外部 +0.22，ε 必须显著小于 0.2。原门 0.5
在数学上就不可能支撑新目标，这是第一次立项的判据漏洞。

### 9.2 可行性判定：现有粗抽象到不了，而且不是收敛问题

§8 已把实现层嫌疑全部排除（Kuhn 金标准、自洽校验、空表对照、座位分离、平均策略口径），
剩下的是表达力上限。本机实测补三组数据：

| 事实 | 读数 |
|---|---|
| 当前蓝图规模 | **16 558 info-set**（B=8、R=3、5 动作、跨街 imperfect recall） |
| 可利用度 | **30.320 ± 1.876 BB/手**；6M→26M 迭代 28.2→28.7，**平台** |
| 桶数扩展（实测） | B=64 → **118 549** info-set，32.7k iter/s（单核）；B=256 → **393 328**，25.0k iter/s |
| 训练吞吐（实测） | B=8：**47k iter/s 单核**（6M ≈ 2.1 分钟，26M ≈ 10 分钟） |
| 10 核外推 | ≈ 10⁵ iter/s ≈ **8.6×10⁹ 迭代/天** |
| 表格 perfect recall | 按街历史组合爆炸；B=64 时约 10⁸–10⁹ info-set（32 GB 放不下或训不完） |

把 ε 从 30 压到 0.15 需要 **200×** 改善，可用的自由度只有三个，且都撑不住：

1. **牌桶**：实测增长是线性的（B 每 ×8 只带来 ~3.3× info-set）。就算上到 10⁷ info-set，
   经验斜率也只能给出约 3~10×，落在 3~10 BB/手。
2. **召回**：真 perfect recall 在 200BB 四街下是 10⁸–10⁹ 级，内存与收敛时间都不可行；
   压缩历史（每街 2 bit）能到 10⁶–10⁷，是可试的最大档。
3. **动作抽象**：5 动作 / R=3 已经够用，再细化只会让 info-set 继续膨胀，且 BR 用的是同一套
   抽象动作，细化它**不直接降低**可利用度。

**判断**：粗抽象的强度上限本身就低于目标。16.5k info-set 给出的 30 BB/手不是"没训够"，
而是"表里没有信息可学"。乐观情形（B=256 + 压缩召回，10⁶–10⁷ info-set）我的先验是
**1~5 BB/手**——仍差目标 5~30×。所以**不直接投 1~2 周**，用 M0 把这个先验量出来再决定。

### 9.3 路线与逐步验证

| 阶段 | 做什么 | 产出物 | 机时 | 通过条件 |
|---|---|---|---|---|
| **M-1 基线量准** | `stn2` 外部补到 12 000 手（单会话串行、1500 手/段、delay ≥150ms） | 基线 −0.21→±0.19 的定论读数 | 11 h 外部（零代码） | 无门，只定义"差距" |
| **M0 斜率实验** | ① info-set key 加"每街 2 bit 压缩历史"（`--recall 0\|1`）；② BR 增加按街分解（`--by-street`）；③ 单测；④ 6 组配置（B=8/64/256 × recall 0/1）× 30M 迭代 + BR 1000 抽样 | ε 随桶数/召回的斜率表 | ~6 h（训练 <1 h，BR 2–3 h，编码 2–3 h） | **ε ≤ 2 BB/手，且 30M→60M 仍在下降**；否则关闭 CFR 路线 |
| **M1 细化** | 取 M0 最优配置扩到 10⁶–10⁷ info-set，多线程 atomic regret，训到平台 | `blueprint_v5` + ε 曲线 | 1–2 天 | **ε ≤ 0.15 BB/手** |
| **M2 蒸馏** | `blueprint-distill` → `bp_distill_v5`，留出集早停 | 蒸馏模型 + 保真读数 | 1 天 | G2：top-1 ≥ 0.70、KL ≤ 0.10、六项锚点全正 |
| **M3 外部** | 12 000 手（P0），过了再 36 000 手（P1） | 外部裁决读数 | 11 h + 33 h 外部 | P0 → P1 |

**预算口径**：算力不是瓶颈（10 核 ≈ 8.6×10⁹ 迭代/天，10⁷ info-set 收敛只需数小时），
**瓶颈是表达力与外部通道**。全部通过时墙钟约 **1.5 周**，其中 44 小时是外部串行采样；
M0 不过时总投入 **≤ 1 天**即止损。

### 9.4 止损与纪律

- **M0 不过就关闭路线**：不蒸馏、不占外部通道、不写 README 进度条。
- 外部通道只给 M-1（量准基线）和 M3；其余阶段全部内部评估。
- 所有检查点写 `/Users/mingming/models/`，禁止 `/tmp`；M0 的临时表用 `bp_bench_*` 并在实验后删除。
- 一次只跑一条外部会话；分段 1500 手、delay ≥150 ms、禁并发。

### 9.5 如果 M0 失败（备选，需另行确认）

1. **接受现状**：把 `stn2` 外部读数补到 12 000 手（11 h），把 −0.21 ± 0.25 变成 ±0.19 的定论。
2. **换算法就是换算力**：Deep CFR / ReBeL 需要云端 GPU；本机 MPS 端到端反而更慢（README 第一节实测），
   继续在 M1 Pro 上做没有意义。预算与周期另议。
3. **翻前专用 blueprint**（探索项，不作主线）：翻前 169 类无损，翻后用现网络当叶节点 EV，
   1 天出原型。它不替换翻后决策，但对手范围仍只能用自身策略当先验——与此前被否决的
   部署期求解同源，必须先过内部对照。

### 9.6 需要用户确认

1. 接受 P1 = 36 000 手 / **+0.22 BB/手**（单侧 95% 下界 > 0）作为"追上主流"的裁决口径？
2. 批准 **M-1（11 h 外部）+ M0（4~6 h）**？M0 不需要多天机时，可在一次会话内跑完。
3. M0 若失败，选哪条备选：关闭 CFR 路线 / 租 GPU 做 Deep CFR / 翻前 blueprint 探索？

---

## 10. M0 执行结果（2026-11-08）：斜率表与裁决

用户指令"继续迭代"后按 §9.3 执行 M0（纯内部，不占外部通道）。**结论：M0 未通过，
按 §9.4 关闭表格 CFR blueprint 路线。**

### 10.1 交付与事故记录

- 代码：`--recall 0|1`（每街 2 bit 压缩加注历史入 key，bit 44..51）、BR `--by-street`
  （BR 值与同抽样自对弈值按终局街分解）、meta 持久化 recall、resume 口径跟随。
  单测 **54873 断言 0 失败**（含 recall key 训练/查表一致性、压缩历史独立入 key）。
- **事故**：首轮 6 组训练因 shell 引号 bug 全部 `B=0` 静默退化（目录名带空格
  `"b8_r0 8 0"`），产物无效，已隔离到 `~/models/m0_invalid_b0/`。
  修复：`blueprint-train` 增加桶数门禁 **[1,256]**（key 牌桶字段仅 8 bit，
  B=0 退化 / B>256 截断碰撞都拒跑，放在 resume 跟随之后，无效检查点也拒）。
- **resume bug 修复**：`--resume` 时 `--buckets`/`--raise-cap` 未显式传参现在跟随检查点
  （原先拿默认值 B=8 与检查点 B=256 误报"抽象不一致"）；显式传参且不一致仍拒跑。
  两个探针（显式冲突 / B=0 检查点）均正确拒绝。

### 10.2 斜率表（6 组 × 30M 迭代，seed 777；BR N=1000，seed 424242，双座位+按街分解）

| 配置 | infosets | 训练耗时 | ε = BR0+BR1（BB/手） | 翻前 | 翻牌 | 转牌 | 河牌 |
|---|---|---|---|---|---|---|---|
| B=8 r0 | 16 628 | 713 s | **29.084 ± 0.704** | +4.85 | +4.80 | +5.83 | +13.24 |
| B=8 r1 | 58 010 | 860 s | **27.586 ± 0.654** | +5.38 | +1.86 | +5.01 | +15.13 |
| B=64 r0 | 72 804 | 844 s | **28.450 ± 0.739** | +4.62 | +4.67 | +6.18 | +12.75 |
| B=64 r1 | 366 648 | 1063 s | **24.825 ± 0.650** | +4.83 | +2.42 | +5.19 | +12.22 |
| B=256 r0 | 260 560 | 1017 s | **25.702 ± 0.716** | +4.16 | +4.35 | +5.48 | +11.43 |
| B=256 r1 | 1 387 344 | 1241 s | **22.691 ± 0.657** | +4.46 | +2.61 | +4.89 | +10.58 |

（按街分解 = BR 相对同抽样自对弈的增益，四个分量之和 ≈ ε − 自对弈和；自对弈和≈0。）

### 10.3 斜率解读

- **牌桶**：B 8→256（32×，infoset 16×）只降 3.4 BB/手（29.1→25.7，r0 列）。
- **召回**：r0→r1 在同桶数降 1.5~3.6 BB/手；与牌桶叠加最优 22.7。
- **合计**：infoset 16.6k→1.39M（**83×**）只降 **22%**（29.1→22.7）。
  对数线性斜率 ≈ **−1.45 BB/手 / e-fold infoset**。外推到 ε=2 需 ~14 个 e-fold，
  即 **~10¹² infosets**——内存与收敛时间都远超可行域；§9.2 的乐观先验（10⁶ infoset
  给 1~5 BB/手）实测为 22.7，比先验还差 4~20×。
- **按街分解**：所有配置的漏洞都集中在**河牌**（+10.6~15.1，占 ε 约一半），
  召回主要修翻牌（+4.8→+1.9~2.6），修不动河牌——河牌需要完整跨街范围推断，
  压缩历史（每街仅加注次数档）表达不了。
- **迭代下降检查**：最优配置 b256_r1 续训 30M→60M（逐位续跑，recall=1 跟随检查点），
  读数见 §10.4。

### 10.4 裁决

**M0 FAIL，按 §9.4 关闭表格 CFR blueprint 路线。**

- 最优配置 b256_r1：30M → 60M 迭代，ε = **22.691 ± 0.657 → 22.242 ± 0.601 BB/手**。
  点估计仍在缓降（−0.45 / 30M），但即便线性外推也要 **~13 亿迭代**才到 2，且增速递减；
  门值 ≤2 差 **11 倍**，G1 终值 0.15 差 **148 倍**。两个条件联合不可能满足。
- 关闭范围：不再投入牌桶/召回细化、不再蒸馏、不占外部通道。§9.5 的备选
  （接受现状补基线 / 云端 GPU Deep CFR / 翻前 blueprint 探索）**需用户另行决策**。
- 现役部署不变：**`stn2` + T=0.5**（`--range-eq 0`）。
- 产物：7 张表与全部日志在 `~/models/m0/`（b256_r1_60m 为最优终表）；
  无效首轮产物隔离在 `~/models/m0_invalid_b0/`。复现容易，磁盘充裕前保留。

### 10.5 复现命令

```bash
# 训练矩阵（6 组并行，各 ~12~21 分钟）
for B in 8 64 256; do for R in 0 1; do
  ./build_rl/perfecttexas blueprint-train --iters 30000000 --save-every 10000000 \
    --log-every 10000000 --out /Users/mingming/models/m0/b${B}_r${R} \
    --buckets $B --recall $R --seed 777
done; done
# BR（N=1000，各 ~5~9 分钟）
./build_rl/perfecttexas blueprint-br --bp /Users/mingming/models/m0/b256_r1 \
  --samples 1000 --seed 424242 --by-street
# 最优配置续 60M（recall/B/R 自动跟随检查点）
./build_rl/perfecttexas blueprint-train --resume /Users/mingming/models/m0/b256_r1 \
  --iters 30000000 --save-every 10000000 --log-every 10000000 \
  --out /Users/mingming/models/m0/b256_r1_60m --seed 777
./build_rl/perfecttexas blueprint-br --bp /Users/mingming/models/m0/b256_r1_60m \
  --samples 1000 --seed 424242 --by-street
```