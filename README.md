# PerfectTexasCPP

单挑无限注德州（Heads-Up No-Limit Hold'em）的自对局强化学习，C++17 + LibTorch，
训练与评估循环全部在 C++ 里。规则口径对齐 ACPC（50/100 盲注、20000 筹码 = 200BB、
每街无加注上限、任意整数下注额），详见 [docs/RULES.md](docs/RULES.md)。

当前目标：在 200BB 单挑上追平成熟 AI 基准（Slumbot）。现水平约
**−0.4 ~ −0.6 BB/手**，尚未追平。

> **本文件只保留仍然有效的架构、纪律、结论和复现方法。** 上百条按日期追加的
> 实验流水已移除；需要历史数字时以 git 历史为准（`git log -p README.md`）。

---

## 快速开始

```bash
# 依赖：LibTorch（CPU 版即可）、CURL
curl -L -o /tmp/libtorch.zip \
  https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.5.1.zip
unzip -q /tmp/libtorch.zip -d ~

cmake -S . -B build -DCMAKE_PREFIX_PATH=$HOME/libtorch -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure      # 或直接跑 ./build/perfecttexas_tests
```

当前测试规模：**927 条断言，0 失败**。

```bash
# 内部强度（对内置基准；anchor 用 rangeEq16、T=0.5）
./build/perfecttexas eval --resume /Users/mingming/models/stn2 --vs solid \
  --hands 4000 --seed 90001 --temp 0.5 --range-eq 16

# 两只模型对打
./build/perfecttexas match --a /Users/mingming/models/stn2 --b /Users/mingming/models/stn3 \
  --hands 4000 --seed 90001 --a-temp 0.5 --b-temp 0.5 --range-search 16

# 外部基准（Slumbot 公开 API，单会话）
./build/perfecttexas slumbot --resume /Users/mingming/models/stn2 \
  --hands 1500 --temp 0.5 --delay-ms 150 --log /tmp/ext_s1.csv

# 设备基准（决定 --device 与 minibatch；本机结论见第一节末）
./build/perfecttexas bench-device --iters 12 --batch 512 --batch 1024 --batch 2048
```

---

## 一、算法架构

### 游戏与规则

- 单挑 NLHE，盲注 50/100，起始 20000（200BB），每手重置（Doyle's Game）。
- 座位固定：0 = 小盲（按钮，翻前先动、翻后后动），1 = 大盲。
- 每街加注无上限；合法下注额是任意整数（只要求 ≥ 最小加注、≤ 全下）。
- 状态是纯 POD（`HUState`），一手牌的全部信息可拷贝、可重放。

### 网络与特征

| 部分 | 内容 |
|---|---|
| Actor（**分座位独立**） | 256 hidden + LSTM(128)；SB / BB 各一套，不共享 |
| Critic（**分座位独立**） | 256 hidden；训练时看开天眼信息 |
| 节点特征 | 388 = 我方底牌 104 + 公共牌 260 + 24 标量（底池/筹码/跟注额/位置/胜率百分位/听牌…） |
| 行为序列 | 12 步 × 12 维（谁、街、动作类型、金额、动作后底池），喂 LSTM |
| 开天眼特征（仅 critic） | 108 = 对手底牌 104 + 完美信息胜率/底池赔率/边际 |
| 动作特征 | 每槽 16 维 |
| 动作空间 | 19 槽 = fold / check-call / **16 个尺寸带** / all-in |

尺寸带不是预设金额：把 `[最小加注, 全下)` 按常用锚点切成连续区间（翻前按“上次加注额的
倍数” 2.2 / 2.5 / 3.0 / 4.0 / 5.5 / 8.0 / 12 / 20 / 35 / 60，翻后按底池比例 1/4…5），
带内按对数均匀落点。因此金额本身没有档位，策略只选“量级”。

### 训练目标

- **PPO + GAE**：不完信息 Actor 采样，完美信息 Critic 估值；奖励 = 终局净筹码/起始筹码
  加上开天眼势能稠密项 `Δ(胜率 × 底池 − 已投入)/起始筹码 × coef`（默认 coef 0.05，
  当前配方用 0.30）。
- **脚本教师**（可选）：对同一副牌枚举合法动作、用规则/胜率机器人续演得到反事实 Q，
  再以 `softmax(Q/T)` 蒸馏回策略。当前外部最强的世代就是靠它学会看牌。
- **rangeEq**（可选）：每个决策按对手最近动作的策略似然对采样底牌重加权，算
  vs-range 摊牌胜率，作为动作残差支路的输入（训练用 16，部署默认关闭）。

### 联赛与历史池

每局随机指定一个“对手座位”，按概率分区：

| 对手 | 默认概率 | 作用 |
|---|---|---|
| 当前最新策略（自对弈） | 0.30 | 主线进化 |
| 历史快照 | 0.30（抽中 archive 占 30%） | 打断自对弈循环 |
| 规则机器人 | 0.15 | 人类风格锚点、防漂移 |
| 均匀随机 | 0.05 | 防止对强牌过度弃牌 |
| 稳健 / 强价值 / 紧全下 / 陷阱 / 按尺寸弃牌 / 小注骚扰 / 过度下注 / 读范围 / 河牌 CFR | 0（按需开启） | 各补一种风格漏洞 |

历史池两级：`recent`（8，滚动丢最老）与 `archive`（16，满了隔一个丢一个），
归档层保证早期快照永不消失。

### 评估

- **内部锚点**：`rule / random / pro / search / solid / mix`，必须**全为正**。
- **外部基准**：Slumbot 公开 API（规则完全一致），用终局 `bot_hole_cards` 做
  全下 EV 调整，剔除 runout 运气；只看调整后的数。

### 训练设备与后端（本机实测，2026-11-03）

| 项 | 事实 |
|---|---|
| 机器 | Apple **M1 Pro**，arm64，10 核 CPU、**16 核 GPU**、32 GB 统一内存 |
| LibTorch | **2.5.1**（macOS arm64 官方包；`libtorch_cpu.dylib` 内含 `torch::mps::*` 符号） |
| 运行时探测 | `torch::cuda::is_available() = 0`；`torch::mps::is_available() = 1` |
| 目前训练 | 采样恒在 CPU；**设备只跑 PPO 更新**（模型副本 + 每轮权重回拷，~0.03s） |

`bench-device` 实测（256×128，`--iters 12`，吞吐口径；多轮重复的区间，机器有
±30% 抖动——睿频/后台负载，采纳前应在本机重跑）：

| 设备 | batch 512 fwd / train | batch 1024 | batch 2048 |
|---|---|---|---|
| CPU（8 线程） | 14~21 / 38~56 ms | 35~39 / 82~96 ms | 57~75 / 105~177 ms |
| MPS | 5~7 / 36~74 ms | 8~9 / 45~47 ms | 9~12 / 40~53 ms |

**kernel 层面 MPS 快 2~4×（前向 3.8~8×），但端到端训练反而慢**（256 局/轮、3 采样线程）：

| 配置 | 更新/秒 |
|---|---|
| CPU + minibatch 128（默认） | **1.00** |
| CPU + minibatch 512 | 1.32 |
| MPS + minibatch 512 | 0.22 |
| MPS + minibatch 1024（设备默认） | 0.63 |

拆分耗时：rollout 两边都是 0.10~0.14s，差距全在 PPO 更新（CPU 0.9~1.1s，MPS 1.5~4.5s）——
小网络的算子启动/同步开销远超算力收益。所以：

- **`--device auto` → CUDA（若有），否则 CPU**，**不自动选 MPS**；
- 想用 GPU 必须显式 `--device mps`（或 `--mps`），并建议 `--minibatch 1024`；
- 大网络（512×256）时 MPS 在 batch 2048 的训练吞吐约为 CPU 的 2.5×，但当前模型用不到。

---

## 二、当前水平

| 项 | 读数 |
|---|---|
| 本代外部水平（对 Slumbot，12000 手级） | **−0.4 ~ −0.6 BB/手** |
| 现役最强可用检查点 | **`stn2`**：−0.21 ± 0.25 BB/手（5927 手，被限流截断） |
| 历史最好单臂读数 | `stn1@16` −0.42 ± 0.13（权重已丢失，仅存档记录） |
| 部署配方 | **`stn2` + T = 0.5**（`--temp 0.5`，默认 `--range-eq 0`） |
| 内部锚点 | 全部大幅为正（rule +1200~1700、solid +400~560、mix +600~900 筹码/手量级） |

**内部与外部排序长期脱钩**：内部锚点只能防“机制性崩坏”，不能用来采纳。
一个在 `solid` 上 +585 的模型，外部仍可能 −0.5 BB/手。

---

## 三、已证伪 / 已关闭的方向

每条都做过对照实验；量化结论按“调整后筹码/手”或“BB/手”给出。

| 方向 | 结论 | 关键量化 |
|---|---|---|
| **转牌 CFR 子博弈当部署算子** | 关闭 | solid −219.9 ± 69.0（3.2σ）、pro −223.5 ± 80.1（2.8σ）；4.2 秒/手（~100× 成本） |
| **河牌 CFR ��博弈当部署算子** | 关闭 | min-pot 4000（接管 2.7%）≈ 0（−0.01 ± 0.35，1500 手）；min-pot 1500（接管 12.0%）**−0.69 ± 0.29**（4500 手，比 0 低 2.4σ）；内部 vs solid −236.9 ± 138.2；H2H vs 自身 −1.2 ± 10.9 |
| **河牌大池 1.5x 剥削** | 关闭 | 外部 −0.88（2763 手）；机制证伪：Slumbot 面对 >1.3x 河牌下注只弃 47.5%（诊断预测 76.8%） |
| **大池教师蒸馏**（`--teacher-min-pot 4000`） | 关闭 | 外部 −1.22 ± 0.46（2092 手），基座 −0.72 ± 0.33；内部 H2H 仅打平 |
| **单步前瞻搜索当部署算子** | 关闭 | v12 自对打 8000 手：search-4 **−667 ± 85**、search-8 −394 ± 85；Q 噪声 ~0.3BB 远大于动作间真实差 0.02~0.05BB |
| **搜索当教师（专家迭代）** | 关闭 | 匹配对照 H2H **−30.5 ± 7.6（4.0σ，64k 手）**；solid −293（2.2σ）；标签方差直接写进策略 |
| **纯自对局 / 纯强对手联赛** | 关闭 | 纯自对局打规则机器人 **−91.0 ± 79.7**（违反“基准全为正”硬约束）；脚本锚点同时是防漂移装置 |
| **rangeEq 类额外胜率标量** | 无一致收益 | 部署 @16 早期 +0.26，24k 复核只剩 **+0.16（0.8σ）**，`stn2` 反向 **−0.53**；`rangeEq32/96` −0.85（2988 手）；Slumbot 日志拟合的经验范围模型 −0.60（5698 手） |
| **往联赛加手写对手**（9 种：solid/trapper/shark/ranger/tight/folder/spammer/cfr/maniac） | 关闭 | 16000 手锚点上没有一个站得住；`spm1` 相对同配方对照 rule −3.4σ、random −3.8σ |
| **BB 3-bet guard / 全下 guard / 翻前·翻牌限攻** | 关闭 | 12k 手重算后增量 ≈ 0；与基线统计不可分辨 |
| **部署温度扫描** | 噪声级 | T=0.1 / 0.35 / 0.5 全部 −0.5 ~ −0.7，差值 <1σ；固定 T=0.5 |
| **行为克隆 Slumbot 当对手/代理** | 关闭 | 数据量 ×10（1 万手留出）后平衡 acc 仍 ≈ 0.30，raise 预测 10.6% vs 真实 16.4% |
| **链式微调（在冠军上继续训）** | 无外部收益 | `stn2` 再续 1500 轮外部 −0.74；`stn3`/`big1` 不优于；同配方纯续训本身锚点就掉 ~1BB（必须带对照） |
| **模型 soup 滥用** | 仅限少量同架构 | 2-seed soup 曾 +52 ± 14，但 4/6 种子组合都输 v12，且会把某个座位平均成“永远弃牌”（BB −376） |
| **更大网络** | 关闭 | 512/256 同预算、同 seed 输给 256/128 **−215.6 ± 53.1**（容量不是瓶颈） |
| **动作级集成** | 关闭 | mix 掉 226 |
| **部署期脚本阈值/启发式 override** | 纪律禁止 | 河牌/全下/限攻等 override 全部与基线不可分辨；改进必须来自训练，不是部署开关 |

**共同机制**：凡是用“模拟/求解/搜索”的方向，对手模型都只能用我们自己的策略当先验；
对脚本对手会丢掉剥削力，对均衡型对手又不赚。接管覆盖面越大，损失越大
（河牌 CFR：2.7% ≈ 0，12% −0.69）。

---

## 四、训练与评估纪律

1. **内置基准必须全为正**（rule / random / pro / search / solid / mix）。任一为负即淘汰，
   不看其它指标。
2. **联赛保留混合强弱配比**，不用纯强对手：纯自对局/纯强对手会让策略漂移到
   “高激进、极少摊牌”，直接输给规则机器人。
3. **外部请求串行 + 指数退避**；并发 ≥2 会把单会话拖到 ~12 秒/手并卡死。
   长跑按“1 条会话、delay ≥150ms、每 1500 手停一段”。
4. **外部只看全下 EV 调整后的数**；原始收益的 runout 运气可占一半。
5. **手数与分辨率**：3000 手 SE ≈ **0.38 BB/手**；12000 手 ≈ **0.19**。
   **0.2BB 级改动必须 ~12000 手**才能裁决，3000 手只用于发现机制性错误。
6. **内部锚点手数**：4000 手单项 SE ≈ 0.5 BB（两臂差 ≈0.7），16000 手 ≈0.25（差 ≈0.35）。
   零成本筛选可用 4000，采纳判断一律 16000。
7. **每条臂都要有同配方对照**：同配方纯续训本身就会让锚点掉约 1BB，没有对照的
   “改动效果”大多是续训漂移。
8. **接管覆盖率与伤害放大**：任何“替换策略某类决策”的算子，先报接管比例，再看 EV；
   比例越大越危险，不能只看点估计。
9. **只采纳可复现的改进**：外部 12000 手、点估计优于基线且 ≥2σ；否则记为“不可分辨”。
10. **新训练输出写到 `/Users/mingming/models/`，禁止写 `/tmp`**（系统会清理，
    `stn1` 的权重就是这样丢的）。
11. **GPU 不是默认项**：先跑 `bench-device`；本机 256×128 模型上 MPS 端到端比 CPU
    慢 1.6~4.5×，`--device auto` 因此回退 CPU。GPU 只在 `--device cuda`（真 NVIDIA）
    或显式 `--device mps --minibatch 1024` 时使用，且必须重新验证更新吞吐。

---

## 五、检查点与产物

| 路径 | 说明 |
|---|---|
| `/Users/mingming/models/stn2` | **现役最强**（外部 −0.21 ± 0.25，截断样本） |
| `/Users/mingming/models/stn3` | 同血统更长训练，外部未测 |
| `/Users/mingming/models/big1` | 384×160 大网，锚点 search 已转负，不推荐 |
| `ckpt_v12/` 等仓库内目录 | 教师蒸馏早期世代（历史存档，弱于 stn2）|
| `docs/RULES.md` | 规则口径与 ACPC 逐条对账 |

---

## 六、复现命令

### 训练（从现有检查点续训）

```bash
./build/perfecttexas train --resume /Users/mingming/models/stn2 \
  --out /Users/mingming/models/NEW --updates 1500 --games 512 --workers 3 --lr 3e-4 \
  --device auto --seed 10101 --oracle-coef 0.30 --range-eq 16 --range-eq-iters 48 \
  --league 0.15 --rule-prob 0.08 --random-prob 0.05 --solid-prob 0.35 \
  --shark-prob 0.15 --folder-prob 0.07 --trapper-prob 0.05 --tight-prob 0.05 \
  --eval-every 250
```

### 内部配对评估（必须跑全部基准）

```bash
for opp in rule random pro search solid; do
  ./build/perfecttexas eval --resume DIR --vs $opp --hands 8000 \
    --seed 90001 --temp 0.5 --range-eq 16
done
./build/perfecttexas eval --resume DIR --vs mix --hands 4000 --seed 90001 --temp 0.5 --range-eq 16
```

### 外部裁决（单会话、分段）

```bash
./build/perfecttexas slumbot --resume /Users/mingming/models/stn2 \
  --hands 1500 --temp 0.5 --delay-ms 150 --seed 96001 --log /tmp/ext_s1.csv
# 每段看“中止/错误数”和调整后 EV；正常后换 seed 续跑，合并 12000 手再裁决
```

### 诊断工具

```bash
./build/perfecttexas commit-audit --csv FILE.csv...     # 弃牌前投入/分座位/翻前层级
./build/perfecttexas allin-audit --csv FILE.csv...      # 全下决策审计
./build/perfecttexas slumbot-analyze --csv FILE.csv     # 逐决策事后胜率、尺寸-弃牌率
./build/perfecttexas slumbot-decisions --csv FILE.csv... # 导出对手逐决策样本
./build/perfecttexas showdown --resume DIR --vs solid --hands 4000   # 摊牌拆分
./build/perfecttexas policy-stats --resume DIR --hands 1000          # 动作分布
./build/perfecttexas soup --a DIR --b DIR --out DIR --alpha 0.5      # 少量同架构平均
```

---

## 七、待决策：离线 CFR blueprint

部署期求解已经关闭（见第三节）。要再往前，唯一有量级意义的方向是**离线 CFR blueprint**。
关键前提：**GPU 帮不帮得上，取决于用哪一种 CFR**——这不是一台“有 GPU 就能快”的问题。

### 两条路线

| | A. 表格 MCCFR（抽象 + regret 表） | B. Deep CFR（优势网络 + regret 网络） |
|---|---|---|
| 计算形态 | 纯 C++ 树遍历 + regret 表更新 | 遍历采样 + 批量前向/反向训练网络 |
| 主要瓶颈 | **CPU 单核性能 + 内存带宽/缓存** | 遍历（CPU）+ 网络训练（GPU） |
| GPU 加速 | **≈0**（表更新是随机写内存，GPU 反而更差） | 只加速网络训练那一半 |
| 本机实测依据 | 同构 rollout：~0.10~0.14 秒 / 256 局（3 线程，约 2.5k 局/秒、~25k 决策/秒） | `bench-device`：256×128 网 B=2048 时 MPS 40ms vs CPU 177ms（4.4×）；512×256 网 B=2048 时 90ms vs 229ms（2.5×） |
| 需要的抽象 | 必须更粗（翻前 169；翻后每街 5~8 桶；每街 2~3 尺寸） | 可以更细（桶数翻倍、尺寸多一档） |
| 表/显存需求 | infoset ~10^7~10^8，regret 表 **1~4 GB**（32GB 统一内存可容纳） | 网络 <100 MB；采样/回放缓冲 1~4 GB；MPS 与 CPU 共享内存 |
| 可实现强度上限 | 受抽象精度限制，粗抽象通常仍 **> 0.2 BB/手** 可利用度 | 上限更高，但遍历吞吐是硬约束：当前 ~25k 决策/秒，离 Deep CFR 需要的 10^6~10^7/秒差两个量级 |
| 预计工期（本机） | 抽象设计 1~2 天 + MCCFR 实现 2~3 天 + 训练 **1~3 天** + 蒸馏/外部验收 2~3 天 ⇒ **约 1~1.5 周** | 实现 3~5 天 + 遍历向量化/并行 3~5 天 + 训练数天~数周 ⇒ **≥2~3 周**，且大概率卡在遍历 |
| 风险 | 抽象过粗 ⇒ 追不上 Slumbot；过细则内存与收敛时间爆炸 | 遍历吞吐不足 ⇒ 训练规模上不去；蒸馏再遇“对手模型不匹配” |

**结论：推荐 A（表格 MCCFR）先做，B 暂不投入。** GPU 对本项目当前唯一的训练循环
（PPO 更新）都帮不上（见第四节实测），对表格 MCCFR 更是无效；Deep CFR 的理论优势
被遍历吞吐吃掉。若未来把游戏缩小（例如只做 turn/river 子博弈、或降到 100BB），
B 才值得重新评估。

### 交付与验收

- **接回网络**：把 blueprint 的逐决策动作分布蒸馏进 actor，或作为联赛里的固定对手
  （必须与脚本锚点混用，纯强对手配比已被证伪）。
- **验收**：内部 BR / `commit-audit` 先验证可利用度，再走 12000 手外部、2σ 判据。
- **风险点**：① 粗抽象蓝图自身可利用度可能 >0.2 BB/手；② 蒸馏阶段会再次遇到
  “对手模型不匹配”；③ 1~1.5 周内不能并行其它实验。

**当前状态：未启动，等确认。**

---

## 八、目录结构

```
include/texas/
  card.hpp      52 张牌、O(n) 牌力评估
  strength.hpp  169 类翻前胜率表、翻后百分位、听牌出路
  hu.hpp        单挑无限注规则引擎（纯 POD）
  features.hpp  节点/开天眼/序列/动作特征编码
  ppo.hpp       GAE、掩码 softmax
  nn.hpp        Actor / Critic 网络
  rollout.hpp   自对局采样、联赛、历史池、部署策略路径
  learner.hpp   PPO 学习器
  eval.hpp      对固定基准的强度评估
  rivercfr.hpp  河牌范围均衡求解器（离线工具）
  turncfr.hpp   转牌子博弈求解器（离线工具）
  slumbot.hpp   Slumbot API 客户端、全下 EV 调整、承诺审计
src/            对应实现
tests/          test_main.cpp（927 条断言）
docs/           RULES.md（规则与 ACPC 对账）
```

## 九、许可

MIT。
