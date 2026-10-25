# texas_cfr

德州扑克不完全信息博弈求解器，C++17 实现，无第三方依赖。

核心是 **CFR（Counterfactual Regret Minimization，反事实遗憾最小化）** 家族算法。这是
Libratus、Pluribus、DeepStack 击败人类职业选手时用的方法，而**不是** PPO / DQN 这类
标准强化学习。

## 为什么不用普通强化学习

德州扑克是**不完全信息博弈**：

| | 完全信息（围棋/象棋） | 不完全信息（德州扑克） |
|---|---|---|
| 决策点 | 局面唯一确定 | 信息集内无法分辨（对手底牌有多种可能） |
| 数学框架 | MDP | 扩展式博弈 + 信念分布 |
| 最优解 | 确定性策略（选价值最大的动作） | **混合策略**，必须按概率随机出招 |
| 求解目标 | 最大化收益 | 逼近**纳什均衡**（任何固定打法都会被剥削） |

所以这里求的不是"最优动作"，而是"让对手无法利用的策略分布"。

## 算法

CFR 每一轮迭代做四件事：

1. 用当前策略遍历博弈树，算出每个信息集里每个动作的反事实价值 `v(I,a)`
2. 累积遗憾 `R(I,a) += v(I,a) − v(I,σ)`
3. 后悔匹配更新策略：`σ(I,a) ∝ max(R(I,a), 0)`，越后悔的动作做得越多
4. 按到达概率加权累积**平均策略** `σ̄`

输出的是 `σ̄`，不是最后一轮的 `σ`。理论保证平均策略的可利用度以 `O(1/√T)` 收敛到 0。

本仓库实现了两个变体：

- **vanilla CFR**：标准形式
- **CFR+**（默认）：遗憾裁剪到非负 + 交替更新 + 线性加权平均，实测快 1.4 倍以上

几个实现要点：

- **反事实加权**：累积遗憾时只乘对手的到达概率，不乘自己的——这正是"反事实"的含义
- **整数信息集键**：避免每次递归都构造 `std::string`，Leduc 上快一个数量级
- **模板化的游戏接口**：求解器与具体游戏零耦合，新增游戏只需提供静态接口

## 目录结构

```
include/texas/
  card.hpp      52 张牌表示、7 选 5 牌力评估、牌型解析
  equity.hpp    蒙特卡洛胜率
  kuhn.hpp      Kuhn Poker（3 张牌，有解析解，用于验证正确性）
  leduc.hpp     Leduc Hold'em（6 张牌 + 公共牌 + 两轮下注，标准 benchmark）
  cfr.hpp       通用 CFR / CFR+ 求解器（模板）
  br.hpp        最优反应与可利用度（exploitability）
  holdem.hpp    无限注德州规则引擎（四街、all-in、边池、摊牌）
  strength.hpp  牌力抽象（169 类翻牌前 + 翻牌后分桶，可接 texas_algorithm 查表）
  mccfr.hpp     无限注德州的 external sampling MCCFR 求解器
src/main.cpp    CLI 入口
tests/          自测（用解析解做硬断言）
```

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Windows + MSYS2：

```bash
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
mingw32-make -C build -j
```

## 使用

```bash
# 训练 Kuhn Poker，报告可利用度下降曲线
./build/texas_cli solve kuhn --iters 20000

# 训练 Leduc，写出 CSV 方便画收敛曲线
./build/texas_cli solve leduc --iters 10000 --report 1000 --csv leduc.csv

# 无限注德州（完整规则 + MCCFR）
./build/texas_cli holdem --iters 50000

# 对比 vanilla CFR
./build/texas_cli solve kuhn --iters 10000 --vanilla

# 手牌胜率（蒙特卡洛）
./build/texas_cli equity --hero AhKh --board QhJh2c --opp 2 --iters 100000

# 一键演示
./build/texas_cli demo
```

`solve` 输出示例：

```
        iter  exploitability     value(P0)    infosets
        2500        0.003033     -0.055816          12
        5000        0.002157     -0.055738          12

  平均策略（前 12 / 12 个信息集）:
    P0 card=J hist=""                  p=0.763 b=0.237
    P0 card=K hist=""                  p=0.304 b=0.696
    P0 card=Q hist=""                  p=0.997 b=0.003
    ...
```

## 正确性验证

自测不靠"看起来能跑"，而是和**解析解**逐项对照：

**Kuhn Poker**（理论值 `-1/18 = -0.055556`）

```
value = -0.055646   误差 9e-5
exploitability = 0.001080   共 12 个信息集
```

均衡策略结构也完全对上（`card: 0=J 1=Q 2=K`，`p`=过牌/弃牌，`b`=下注/跟注）：

| 信息集 | 实测 | 理论 |
|---|---|---|
| P0 拿 J 主动下注 α | 0.235 | α ∈ [0, 1/3] |
| P0 拿 K 主动下注 | 0.696 | 3α = 0.705 |
| P0 拿 Q 主动下注 | 0.003 | 0（永不） |
| P0 拿 Q 面对下注跟注 | 0.571 | α + 1/3 = 0.570 |
| P1 过牌后拿 J 诈唬 | 0.338 | 1/3 |
| P1 面对下注拿 Q 跟注 | 0.339 | 1/3 |
| P1 拿 J 面对下注 | 0.000 | 必弃 |
| P1 拿 K 面对下注 | 1.000 | 必跟 |

**Leduc Hold'em**（文献标准值 `-0.0856`）

```
15000 轮  : value = -0.085152   exploitability = 0.0179
200000 轮 : value = -0.085446   exploitability = 0.0052
共 288 个信息集（精确值）
```

信息集按 rank 合并：两张 J 在博弈上完全等价（对手看不到花色、发牌分布相同），
所以 `P0 拿 J` 只算一个信息集。这样策略表更紧凑、打印也不重复；代价是收敛略慢于
按具体牌 id 存储的 936 个信息集版本。

**可利用度基准**：均匀随机策略的 exploitability 是 0.4583，训练 20000 轮后降到 0.0011，
下降 424 倍。

**胜率对照**（40000 次采样 vs 随机手牌）：AA 85.72%（理论 85.2%）、KK 82.57%（82.4%）、
72o 34.58%（34.6%）。

## Leduc 规则

- 6 张牌：J,J,Q,Q,K,K，两人各发 1 张私有牌，各放 1 注 ante
- 两轮下注。第一轮下注额 2，**第二轮翻倍为 4**（Southey et al. 2005 的标准定义）
- 每轮最多 2 次加注（首次下注记作第 1 次），第二轮开始前翻 1 张公共牌
- 成对（手牌 rank == 公共牌 rank）胜过非成对，都非成对时比 rank

## 无限注德州（完整规则）

`holdem.hpp` 是完整的无限注德州规则引擎：

- 52 张牌、四条街（翻牌前 / 翻牌 / 转牌 / 河牌）
- 无限注下注，每街最多 2 次加注，下注尺寸抽象为 半个底池 / 一个底池 / all-in
- all-in 后一次发完剩余公共牌、**边池分层结算**、摊牌比牌，支持 2～6 人
- 盲注、行动顺序、无人下注时不可弃牌等边界都有单测覆盖

### 为什么必须抽象

HUNL 原始决策点约 10^160，这里做两层压缩：

1. **牌力抽象**（`strength.hpp`）
   - 翻牌前：169 类手牌，强度取对随机对手的蒙特卡洛胜率，分 10 桶
   - 翻牌后：优先调用 `texas_algorithm` 的查表排名（`GetWinProbability`，约 280ns/次，
     给出该手牌在所有同张数组合中的精确百分位）；未启用时回退到牌型 + 踢脚启发式
2. **动作抽象**（`holdem.hpp`）：每个决策点最多 5 个动作

### 求解器

`mccfr.hpp` 用 external sampling MCCFR：轮到 traverser 展开全部动作，对手和机会节点各采样一个。
全树遍历不可行——光翻牌就有 C(48,3)=17296 种组合。

信息集 =（街，牌力桶，下注历史），这才让"具体手牌"塌缩成有限状态。

### 用法

```bash
# 零依赖构建（内置启发式牌力）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/texas_cli holdem --iters 50000

# 启用 texas_algorithm 查表后端（翻牌后约 11000 轮/秒）
cmake -S . -B build_ta -DCMAKE_BUILD_TYPE=Release \
      -DTEXAS_CFR_WITH_ALGORITHM=ON \
      -DTEXAS_ALGORITHM_DIR=../texas_algorithm/cpp
cmake --build build_ta -j
./build_ta/texas_cli holdem --iters 50000 --ta-data /path/to/texas_data
```

输出：

```
  牌力桶 10 | 牌力后端: texas_algorithm 查表
        iter    infosets       轮/s
       20000        9269       11353

  翻牌前（小盲/按钮面对大盲）平均策略:
    AA      fold=0.01  call=0.68  r4=0.14  r5=0.08  r200=0.09
    AKs     fold=0.01  call=0.41  r4=0.08  r5=0.32  r200=0.17
    72o     fold=0.79  call=0.04  r4=0.06  r5=0.07  r200=0.04
```

### 已验证

- 盲注、行动顺序、翻牌发牌、无人下注不可弃牌
- 摊牌比牌、弃牌结算、两人零和
- 三人边池：主池 150 由三人争、边池 100 由 P1/P2 争，两种赢家情形下净收益和均为 0
- all-in 后一次发完 5 张公共牌并摊牌
- 牌力抽象单调：AA(桶 8) > AKs(6) > 22(3) > 72o(3)
- 训练后 AA 加注率 0.66、72o 0.13，方向与扑克直觉一致

### 还没做的

- 这里的 MCCFR 只到"趋势正确"，没到可利用度可测的水平。真正对抗职业选手还需要
  蓝图策略 + 实时子博弈求解（DeepStack / Libratus 路线）和 GPU 级算力。
- 牌力抽象用的是简单胜率分位，更细的做法是 potential-aware 或 OSS 分桶。
