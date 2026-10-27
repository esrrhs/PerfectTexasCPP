# Spec：从“河牌大注剥削”转向“全下纪律”训练（第 89 条）

## 1. 背景与证据（当前工作树实测）

外部基准是唯一尺子：Slumbot 公开 API（50/100，200BB，每手重置）。

| 臂 | 手数 | 调整后 EV | 结论 |
|---|---|---|---|
| solid2 @ T0.5（历史基线） | 5013 | −0.72 ± 0.33 BB/手 | 负 |
| solid2 + 河牌 1.5x override（本臂，seeds 20261191–97） | 1360 | **−0.88 ± 0.66 BB/手** | 未转正 |

河牌臂的**机制被证伪**：诊断预测 Slumbot 对 >1.3x 河牌下注弃 76.8%（n=69），
本臂实测只弃 **47.5%（n=59）**。1.5x 下注的保本弃牌率约 64–76%，47.5% 下
纯诈唬 EV 为负。按写死的裁决规则（机制不成立即停），本线终止。

把本臂逐手拆开（`/tmp/ens_eval/slumbot_ro_*.csv`，`allin_adjusted` 列）：

- **非全下手：n=1340，平均 0.0 筹码/手**（打平）。
- **全下手：n=20，平均 −5972 筹码/手**，合计约 −119k。
- 即：全部赤字集中在 1.5% 的全下决策上，其余对局已经打平。

全下决策的具体错误（调整后 EV，来自明细）：

| 类型 | 例子 | 调整后 EV |
|---|---|---|
| 翻前接 200BB 全下，弱牌 | K3s vs 99、A7o vs TT、A8s vs 66、K9o vs AK | −6.4k / −8.2k / −2.5k / −10.9k |
| 翻前 5-bet 后接全下 | K9o（b200b1002b3000b7296b20000c） | −10.9k |
| 转牌用中等成牌全下 | 88 在 7 7 5 Q 面推 16948 | −18.2k |
| 河牌用一对跟大注 | Q8 在 K Q J A 6 面跟 14533；K2 在 9 9 A 3 2 面跟 7424 | −18.6k / −20.0k |

一个“完美全下纪律”规则（只用顶范围接/推全下）在本样本可回收约
**+0.3 ~ +0.45 BB/手**，把 −0.7 推到约 −0.3。方向明确，但仍不足以转正。

另：翻前开牌尺寸不构成剥削。按尺寸分桶测 Slumbot 的弃牌率
（n=1230，我方 SB 先动）：<1.5x pot 弃 10.3%、1.5–1.8x 弃 15.8%、
1.8–2.2x 弃 33.8%、2.2–2.8x 弃 43.1%、2.8–3.2x 弃 73.3%、>6x 弃 80.0%。
每个尺寸的弃牌率都恰好落在该尺寸的保本线附近 → Slumbot 翻前是平衡的，
“开大一点”不是免费钱，放弃该假设。

## 2. 决策

1. **停止河牌 override 线**：杀掉 7 条正在跑的外部会话（保留已写入的 CSV 作记录），
   不再投入训练侧实现。
2. **转向全下纪律**：在联赛里新增“紧全下”对手 `tight`（只用一个顶范围
   推/接全下，其余过牌/弃牌），让策略学到“全下=强范围”。
3. **新增 `allin-audit` 诊断**：从外部 CSV 直接量化“全下决策的调整后 EV”
   按（街、主动推/被动接、座位）分组，作为该改动的外部验证指标。
4. 从 solid2 独立续训出新候选 `tight1`（不要链式续训），内部全基准通过后
   再跑 3000 手外部裁决。

## 3. 实现细节（精确到文件/函数）

### 3.1 新增紧全下机器人

- `include/texas/rollout.hpp`
  - `RolloutConfig` 增加 `float tightProb = 0.0f;`（放在 `solidProb` 之后）。
  - 声明 `Move tightAgentMove(const HUState& s, Rng& rng);`（放在 `solidAgentMove` 旁）。
- `src/rollout.cpp`
  - 新增实现 `tightAgentMove`（放在 `solidAgentMove` 之后），规则：
    - 翻前：`eq = equityVsRandom(s, me, 32)`；面对全下（`need >= 0.5*stack` 或
      `lastRaiseTo` 已到全下）时只有 `eq > 0.68` 才跟，否则弃；主动加注只用
      `eq > 0.80`，尺寸 `3 * kBigBlind`（面对开牌时 `3 * lastRaiseTo`）；
      `eq < 0.45` 且需要跟注时弃牌；其余过牌/跟注。
    - 翻后：`eq > 0.85` 才下注/加注（0.75x pot）；面对下注时要求
      `eq > max(0.55, odds + 0.18)` 才跟，否则弃；面对全下要求 `eq > 0.72`。
    - 目标不是强，而是**提供“全下=紧范围”的对抗方向**（与 solid 的“宽跟注”互补）。
  - `ROLE_TIGHT = 10` 加入枚举（7 已被 sharp 占用，8/9 是 solid/folder）。
  - 权重：`const float wTight = std::max(0.0f, cfg.tightProb);`，并入 `wSum`
    与 `1 - (...)` 的自身权重；轮盘选择里加 `oppRole = ROLE_TIGHT` 分支
    （在 solid 之后、folder 之前）。
  - 加 `runTight` lambda（复制 `runSolid` 结构，调用 `tightAgentMove`）。
  - 联赛分发：`std::vector<int> tightIdx;` + `case ROLE_TIGHT: tightIdx.push_back(i);`
    + `runTight(seat, tightIdx);`。
- `include/texas/ppo.hpp`：`Config` 增加 `float tightProb = 0.0f;`。
- `src/main.cpp`：
  - `usage()` 增加 `[--tight-prob F]` 与说明行；
  - train 解析 `--tight-prob`，赋给 rollout 配置（与 `--solid-prob` 同一处）。

### 3.2 新增 `allin-audit` 子命令

- `src/main.cpp` 新增 `int cmdAllinAudit(int argc, char** argv)`：
  - `--csv FILE`（可重复）；
  - 复用 `slumbot-analyze` 的重放逻辑（`buildSlumbotState` + 逐 token 重放），
    只在 `allin_adjusted==1` 的手上统计；
  - 输出按（座位 × 街 × 主动推/被动接）分组的：手数、平均调整后 EV、
    平均投入筹码、我方手牌百分位中位数（`handStrength(hole, board, 5)`）；
  - 汇总一行：`全下合计 EV / 手`，用于和外部总 EV 对齐（应能解释全部赤字）。
  - 注册到 `main()` 的 `cmd == "allin-audit"`。

### 3.3 测试（`tests/test_main.cpp`）

新增 `testTightAgent()` 并注册进 `main`：

1. AA 翻前面对开牌 → 返回加注（`MK_RAISE`/`MK_ALLIN`）；
2. 72o 翻前面对 3-bet → 弃牌；
3. 河牌坚果（用已知公共牌构造）无人下注 → 下注；
4. 河牌一对弱牌面对全下 → 弃牌；
5. 同一状态跑 200 次，动作集合稳定（不出现随机弃强牌）。

## 4. 验证与门槛

1. 构建：`cd build_rl && cmake --build . -j 8`。
2. 单测：`./build_rl/perfecttexas_tests` 全绿（当前 475 条，新增 ≥5 条）。
3. 训练：从 `solid2` 续训 300 轮：
   `--resume /tmp/solid2 --updates 300 --games 384 --lr 5e-4 --seed 10021`
   `--solid-prob 0.25 --tight-prob 0.20 --league 0.25 --rule-prob 0.10 --random-prob 0.05`
   `--entropy 0.02 --oracle-coef 0.3 --out /tmp/tight1`
4. 内部硬约束（缺一不可，全为正）：`eval --vs rule|random|pro|search|mix|solid` 与
   `match --b /tmp/solid2`；任一为负即淘汰，不进入外部。
5. 外部裁决（写死）：7 会话 × 4000 手，`--temp 0.5`，seeds 20261201–07，
   `allin-audit` 同步统计；到 3000 手：
   - 点估计 > −0.3 → 采纳为新的主臂，扩到 12000 手；
   - −0.7 ~ −0.3 → 只保留“全下纪律”的部署 guard 继续验证；
   - ≤ −0.7 → 该对手分布无效，回到训练配方层（不叠加河牌线）。

## 5. 不做 / 已排除

- 不做河牌 override 的训练侧实现（机制已证伪）。
- 不做“开牌尺寸放大”剥削（实测弃牌率与保底赔率一致，Slumbot 翻前平衡）。
- 不用内部锚点/克隆的胜负作最终裁决。
- 不把目标标记完成：外部仍未转正。

## 6. 需要用户确认

1. 允许杀掉当前 7 条河牌 override 外部会话（进程 45259–45265），保留 CSV。
2. 允许按上述改动实现 `tight` 对手 + `allin-audit`，并启动 `tight1` 训练与外部臂。
