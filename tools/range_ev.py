#!/usr/bin/env python3
"""R33 的「钱」口径尺子：读 slumbot-policy-eval --dump-policy 的 CSV（需含 rq 列）。

对每个翻后面临下注的决策，用
    EV(call) - EV(fold) = eq*(pot_before + need) - need
（翻后不再下注的近似；河牌精确）算四种反事实的总 EV（BB/手）：

  cur   —— 当前策略（p_call 加权）
  mdf   —— 按 MDF 频率随机跟注
  rq    —— 按经验范围胜率 `rq > odds + m` 跟注（部署可得信息）
  oracle—— 开天眼 `eq > odds + m` 跟注（不可达上界）

用法：
  python3 tools/range_ev.py --hands 12000 --margin 0.10 /tmp/r33_stn2_emp.csv
"""
import argparse
import csv
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--hands", type=int, default=12000, help="该 dump 对应的外部手数（分母）")
    ap.add_argument("--margin", type=float, default=0.10, help="赔率外要求的边际")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    if not rows or "rq" not in rows[0]:
        raise SystemExit("CSV 缺少 rq 列：用 --range-eq N [--range-model empirical] 重新 dump")

    def delta(r):
        pot = float(r["pot_before"])
        need = float(r["need"])
        return float(r["eq"]) * (pot + need) - need

    per_street = defaultdict(list)
    for r in rows:
        per_street[int(r["street"])].append(r)

    names = {1: "flop", 2: "turn", 3: "river"}
    print(f"{'street':7s} {'n':>5s} {'cur':>7s} {'mdf':>7s} {'rq':>7s} {'oracle':>7s}  mean_eq mean_odds mean_pcall mean_rq")
    tot = defaultdict(float)
    for st in (1, 2, 3):
        sub = per_street.get(st, [])
        if not sub:
            continue
        n = len(sub)
        d = args.hands * 100.0
        cur = sum(float(r["p_call"]) * delta(r) for r in sub) / d
        mdf = sum(float(r["pot_before"]) / (float(r["pot_before"]) + float(r["need"])) * delta(r)
                  for r in sub) / d
        rq = sum(delta(r) for r in sub if float(r["rq"]) > float(r["odds"]) + args.margin) / d
        orc = sum(delta(r) for r in sub if float(r["eq"]) > float(r["odds"]) + args.margin) / d
        for k, v in (("cur", cur), ("mdf", mdf), ("rq", rq), ("oracle", orc)):
            tot[k] += v
        print(f"{names[st]:7s} {n:5d} {cur:+7.3f} {mdf:+7.3f} {rq:+7.3f} {orc:+7.3f}  "
              f"{sum(float(r['eq']) for r in sub)/n:7.3f} {sum(float(r['odds']) for r in sub)/n:9.3f} "
              f"{sum(float(r['p_call']) for r in sub)/n:10.3f} {sum(float(r['rq']) for r in sub)/n:7.3f}")
    print(f"{'total':7s} {len(rows):5d} {tot['cur']:+7.3f} {tot['mdf']:+7.3f} {tot['rq']:+7.3f} {tot['oracle']:+7.3f}")

    # L1 回归指标（只算 eq > odds+0.10 的弃牌成本）
    bad = sum(float(r["p_fold"]) * delta(r) for r in rows if delta(r) > 0 and float(r["p_fold"]) > 0)
    badcall = sum(float(r["p_call"]) * -delta(r) for r in rows if delta(r) < 0)
    print(f"坏弃（p_fold*delta+）={bad/ (args.hands*100.0):+.3f} BB/手；坏跟={badcall/(args.hands*100.0):+.3f} BB/手")


if __name__ == "__main__":
    main()
