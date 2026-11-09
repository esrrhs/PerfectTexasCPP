#!/usr/bin/env python3
"""L1 验收尺子：外部日志里的「坏弃」金额与「继续率 vs MDF」。

两种输入，同一套判据：
  1) `slumbot-analyze --dump-facing` 的**实际动作** CSV：
       hand,street,seat,pot_before,need,ratio,eq,odds,action,cat,hole,board
  2) `slumbot-policy-eval --dump-policy` 的**候选策略概率** CSV（离线闸门，
     不必重跑外部对局，12k 手约 30 秒）：
       hand,street,pot_before,need,ratio,eq,odds,p_fold,p_call,p_raise,ev_fold,ev_call
     此时「坏弃」按期望计：sum p_fold（落在 eq > 赔率+0.10 的决策上）。

坏弃判据（与 slumbot-analyze 的 FacingAgg 一致）：
  弃牌 且 eq > 赔率 + 0.10，成本 = eq*(pot+need) - need（正数 = 跟注本可多赢的筹码）。
「继续率」= 1 - 弃牌率，经验目标是 MDF = pot_before/pot。

只统计**翻后**（street>=1）：翻前 SB 补盲（need=50、ratio=0.5）会被当成
「面对 0.5x 下注」，会把表整体污染（那是 R11 数字偏大的原因）。

用法：
  python3 tools/badfold_diag.py --hands 12000 facing/*.csv
  python3 tools/badfold_diag.py --hands 12000 --policy pol.csv
"""
import argparse
import csv

SIZES = [(0.0, 0.5, "<0.5x"), (0.5, 1.0, "0.5-1x"), (1.0, 1.5, "1-1.5x"),
         (1.5, 3.0, "1.5-3x"), (3.0, 1e9, ">3x")]


def sz_name(r):
    for lo, hi, name in SIZES:
        if lo <= r < hi:
            return name
    return ">3x"


def load(paths, street_min=1):
    rows = []
    policy = False
    for p in paths:
        with open(p) as f:
            rd = csv.DictReader(f)
            if rd.fieldnames and "p_fold" in rd.fieldnames:
                policy = True
            for r in rd:
                r["street"] = int(r["street"])
                if r["street"] < street_min:
                    continue
                r["need"] = int(r["need"])
                r["pot_before"] = int(r["pot_before"])
                r["pot"] = r["pot_before"] + r["need"]
                r["ratio"] = float(r["ratio"])
                r["eq"] = float(r["eq"])
                r["odds"] = float(r["odds"])
                if policy:
                    r["fold"] = float(r["p_fold"])
                    r["call"] = float(r["p_call"])
                else:
                    r["fold"] = 1.0 if r["action"] == "f" else 0.0
                    r["call"] = 1.0 if r["action"] == "c" else 0.0
                rows.append(r)
    return rows, policy


def bad_folds(g, margin=0.10):
    folds = [r for r in g if r["fold"] > 0.0 and r["eq"] > r["odds"] + margin]
    return folds, sum(r["fold"] * (r["eq"] * r["pot"] - r["need"]) for r in folds)


def bad_calls(g, margin=0.10):
    """反向风险：eq 明显低于赔率却跟注（坏跟），成本 = (赔率-eq)*最终底池。"""
    calls = [r for r in g if r["call"] > 0.0 and r["eq"] < r["odds"] - margin]
    return calls, sum(r["call"] * (r["odds"] - r["eq"]) * r["pot"] for r in calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, required=True, help="这些日志对应的手数")
    ap.add_argument("--street-min", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    rows, policy = load(a.paths, a.street_min)
    h = float(a.hands)
    tag = f"[{a.label}] " if a.label else ""
    print(f"{tag}尺度={'策略期望(离线)' if policy else '实际动作'} | "
          f"{len(rows)} 个翻后面对下注决策（{len(a.paths)} 段日志，{a.hands} 手）")
    print(f"{'尺寸':8} {'决策':>5} {'弃%':>6} {'继续%':>7} {'MDF':>6} {'坏弃':>5} {'坏弃BB/手':>10}")
    tot_bad, tot_loss = 0.0, 0.0
    for _, _, name in SIZES:
        g = [r for r in rows if sz_name(r["ratio"]) == name]
        if not g:
            continue
        n = len(g)
        nf = sum(r["fold"] for r in g)
        bad, loss = bad_folds(g)
        mdf = sum(r["pot_before"] / r["pot"] for r in g) / n
        tot_bad += len(bad) if not policy else nf
        tot_loss += loss
        print(f"{name:8} {n:5d} {100*nf/n:5.1f}% {100*(n-nf)/n:6.1f}% "
              f"{mdf:6.3f} {len(bad):5d} {loss/100/h:10.3f}")
    print(f"{'合计':8} {len(rows):5d} {'':6} {'':7} {'':6} {tot_bad:5.0f} {tot_loss/100/h:10.3f}")

    print("\n坏弃损失按 (eq-赔率) 分解（只算 >0.05 的弃牌）:")
    total = sum(r["fold"] * (r["eq"] * r["pot"] - r["need"]) for r in rows
                if r["fold"] > 0.0 and r["eq"] > r["odds"] + 0.05)
    for lo, hi in [(0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 10.0)]:
        g = [r for r in rows if lo < r["eq"] - r["odds"] <= hi]
        loss = sum(r["fold"] * (r["eq"] * r["pot"] - r["need"]) for r in g)
        print(f"  {lo:.2f}-{hi:.2f}: {len(g):4d} 个 {loss:10.0f} 筹码 "
              f"{loss/100/h:7.3f} BB/手 ({100*loss/total if total else 0:5.1f}%)")

    print("\n按街:")
    for st, name in [(1, "翻牌"), (2, "转牌"), (3, "河牌")]:
        g = [r for r in rows if r["street"] == st]
        if not g:
            continue
        n = len(g)
        nf = sum(r["fold"] for r in g)
        bad, loss = bad_folds(g)
        print(f"  {name}: {n:5d} 决策 弃 {100*nf/n:5.1f}% 坏弃 {len(bad):4d} "
              f"{loss:9.0f} 筹码 {loss/100/h:7.3f} BB/手")

    # 反向风险：把「跟得更多」学成「跟得更烂」时，这里会抬升。必须与坏弃一起看。
    bc, bcloss = bad_calls(rows)
    print(f"\n坏跟（eq < 赔率−0.10 仍跟）：{len(bc)} 个 / {bcloss:.0f} 筹码 "
          f"= {bcloss/100/h:.3f} BB/手")


if __name__ == "__main__":
    main()
