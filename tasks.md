# Tasks：第 89 条（全下纪律）

> 状态：等待用户批准 spec.md 后执行。所有外部裁决按写死规则，不做中途改口。

## T1 停止河牌 override 外部臂（记录失败）

- [x] 杀掉 7 条 `slumbot --river-overbet 1.5` 会话（PID 45259–45265）。
- [x] 保留 `/tmp/ens_eval/slumbot_ro_*.csv`，合并副本 `/tmp/ens_eval/ro_merged.csv` 保留。
- [x] 记录最终数字：n=2763，EV −0.88 ± 0.66 BB/手；>1.3x 河牌弃牌率 47.5%（n=59）。

验证：`ps aux | grep perfecttexas` 无 slumbot 进程；CSV 仍在。

## T2 实现 `tight` 联赛对手

- [x] `include/texas/rollout.hpp`：`tightProb` 字段 + `tightAgentMove` 声明。
- [x] `src/rollout.cpp`：`tightAgentMove` 实现、`ROLE_TIGHT=10`、权重/轮盘选择、
      `runTight`、`tightIdx` 分发。
- [x] `include/texas/ppo.hpp`：`Config.tightProb`。
- [x] `src/main.cpp`：`--tight-prob` 解析 + usage 文案。

验证：`cd build_rl && cmake --build . -j 8` 通过。

## T3 实现 `allin-audit`

- [x] `src/main.cpp` 新增 `cmdAllinAudit`：按（座位 × 街 × 推/接）分组统计
      调整后 EV、投入、手牌百分位中位数，并输出“全下合计 EV/手”。
- [x] 注册子命令；`usage()` 增加一行。

验证：对 `/tmp/ens_eval/ro_merged.csv` 运行，合计应接近 −0.88 BB/手（对齐总 EV）。

## T4 单元测试

- [x] `tests/test_main.cpp` 新增 tight 测试（AA 加注、72o 弃 3-bet、
      坚果下注、弱牌弃全下、动作稳定性），注册进 main。
- [x] `./build_rl/perfecttexas_tests` 全绿（484 条）。

## T5 训练 `tight1`

- [x] 从 `/tmp/solid2` 续训 300 轮（seed 10021，lr 5e-4，solid 25%，tight 20%）。
- [x] 内部基准：rule +1532 / random +1349 / pro +741 / search +189 / mix +936 / solid +472 / H2H +45，全为正。
- [x] 硬约束通过（全为正）→ 进入 T6。

## T6 外部裁决

- [x] 7 会话 × 4000 手，T=0.5，seeds 20261201–07，日志 `/tmp/ens_eval/tight1_{a..g}.csv`。
- [x] 3000 手时 `allin-audit` + 合并 EV：3141 手 −0.68 ± 0.35，
      落在 −0.7 ~ −0.3 → 只验证部署 guard。
- [x] 结论写进 README 第 89 条（tight1 −0.68，机制翻前有效、翻后无效）。

## T6b 翻前全下 guard（中间带的后续验证）

- [x] 实现 `slumbot --allin-guard-pre/--allin-guard-post-cat`，单测 490 条全绿。
- [x] 用 `allin-audit --detail` 估算：翻前 guard 在 solid2 上值 +0.22 BB/手；
      翻后 guard 不赚（到全下时筹码已投入）。
- [x] 启动 solid2 + 翻前 guard 外部臂（7 会话，seeds 20261211–17）。
- [x] 3000 手裁决：3169 手 −0.80 ± 0.38；机制成立（翻前接全下 11→1 手）
      但总 EV 无可测改善；三条臂统计上不可区分。结论见 README 第 90 条。
- [x] `tight2`（tight 35%、600 轮、seed 10023）内部全基准为正（rule 1530 /
      random 1116 / pro 784 / solid 475 / mix 833 / search 184 / H2H +25）。

## T7 文档

- [x] README：第 87 条补充“外部机制证伪（47.5%）”，新增第 89 条记录本 spec 的
      决策、实现、内外结果；顶部摘要与“下一步”同步。

## T12 trap1 + guard 外部臂（当前主线）

- [x] 训练 `trapper` 对手、`trap1`（trapper 25%、600 轮、seed 10031），内部全基准为正。
- [x] 启动并恢复会话（seeds 20261301–10），目标 12000 手。
- [x] 3490 手读：+0.12 ± 0.33（BB −0.48 / SB +0.73；全下 −0.22、非全下 +0.40）。
- [ ] 12000 手裁决（SE≈0.19）：点估计 > 0 且 > 2σ 才算"领先"。
