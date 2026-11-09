#!/bin/sh
# 快速环：分钟级本地面板。
#
# **这是烟雾测试，不是采纳依据。** 内部指标与外部排序长期脱钩（README 第二节），
# 且 2026-11-08 的证伪实验显示：唯一基于外部状态的候选指标（事后 EV 损失）
# 连"温度从 0.1 拉到 10"都测不出来（1.11 / 1.10 / 1.17，非单调），无效。
# 因此本脚本只能用来发现机制性崩坏（锚点转负、动作分布塌缩），
# 采纳仍必须走外部 12000 手、≥2σ（README 第四节第 9 条）。
#
# 用法：tools/fastloop.sh <候选检查点> [对照检查点，默认 stn2]
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN="$ROOT/build_rl/perfecttexas"
CAND=${1:?用法: tools/fastloop.sh <候选检查点> [对照检查点]}
BASE=${2:-/Users/mingming/models/stn2}
CSV1=/Users/mingming/models/ext_stn2/s1.csv
CSV2=/Users/mingming/models/ext_stn2/s2.csv
TMP="${TMPDIR:-/tmp}/fastloop_$$.csv"

echo "[1/4] H2H vs ${BASE}（8000 手 ×2 座位 = 16k 手，固定 seed 424242）"
"$BIN" match --a "$CAND" --b "$BASE" --hands 8000 --seed 424242 --log "$TMP" | head -2
awk -F, 'NR>1 {n++; s+=$3; if ($2>=4000) {bn++; bs+=$3}}
  END {if (n>0) printf "  终局底池 >4000：%d 手 %+0.1f 筹码/手；其余 %d 手 %+0.1f 筹码/手\n",
  bn, (bn?bs/bn:0), n-bn, ((n-bn)?(s-bs)/(n-bn):0)}' "$TMP"
rm -f "$TMP"

echo "[2/4] 锚点（机制崩坏检查）"
"$BIN" eval --resume "$CAND" --vs solid --hands 4000 --seed 31337 | head -1
"$BIN" eval --resume "$CAND" --vs rule --hands 2000 --seed 31337 | head -1

echo "[3/4] 策略动作分布（漂移检查，T=0.5）"
"$BIN" policy-stats --resume "$CAND" --hands 2000 --temp 0.5 | grep -E "SB 翻前开牌|BB 面对开牌|河牌"

echo "[4/4] L1 离线闸门：12k 手外部状态上的「坏弃金额」（策略期望口径，见 tools/badfold_diag.py）"
POL="${TMPDIR:-/tmp}/fastloop_pol_$$.csv"
"$BIN" slumbot-policy-eval --resume "$CAND" --csv "$CSV1" --csv "$CSV2" --temp 0.5 \
  --dump-policy "$POL" | sed -n '2p'
python3 "$ROOT/tools/badfold_diag.py" --hands 3000 --label "$(basename "$CAND")" "$POL" \
  | grep -E "0\.5-1x|1-1\.5x|合计"
rm -f "$POL"
echo "  基线：坏弃 0.492 BB/手（stn2，s1+s2 3,000 手；全 12k 为 0.398）"
