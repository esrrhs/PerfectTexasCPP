#!/usr/bin/env python3
"""R11 误差结构诊断：面对下注时，实际动作 vs「只按牌力+赔率」的规则。

输入是 slumbot-analyze --dump-facing 导出的 CSV：
  hand,street,seat,pot_before,need,ratio,eq,odds,action,cat,hole,board
eq 是开天眼胜率（知道对手底牌），因此下面的规则是「完美读牌、但不用范围」的
上界；它回答的是：如果只看自己的牌和赔率，会有多少弃牌/跟注是错的。
"""
import csv
import glob
import sys
from collections import defaultdict

CAT = ["高牌", "一对", "两对", "三条", "顺子", "同花", "葫芦", "四条", "同花顺"]
STREET = ["翻前", "翻牌", "转牌", "河牌"]
SIZES = [(0.0, 0.5, "<0.5x"), (0.5, 1.0, "0.5-1x"), (1.0, 1.5, "1-1.5x"),
         (1.5, 3.0, "1.5-3x"), (3.0, 1e9, ">3x")]


def sz_name(r):
    for lo, hi, name in SIZES:
        if lo <= r < hi:
            return name
    return ">3x"


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                r["street"] = int(r["street"])
                r["seat"] = int(r["seat"])
                r["pot_before"] = int(r["pot_before"])
                r["need"] = int(r["need"])
                r["ratio"] = float(r["ratio"])
                r["eq"] = float(r["eq"])
                r["odds"] = float(r["odds"])
                r["cat"] = int(r["cat"])
                r["pot"] = r["pot_before"] + r["need"]
                r["fold"] = r["action"] == "f"
                r["call"] = r["action"] == "c"
                rows.append(r)
    return rows


def call_ev(r):
    """跟注的（事后）EV：eq*(pot+need) - need。"""
    return r["eq"] * r["pot"] - r["need"]


def mdf(r):
    return r["pot_before"] / r["pot"] if r["pot"] > 0 else 0.0


def table_fold_structure(rows):
    print("\n== A. 弃牌结构（按街 × 尺寸） ==")
    print(f"{'街':4} {'尺寸':8} {'决策':>5} {'弃%':>6} {'跟%':>6} {'加%':>6} {'MDF':>6} "
          f"{'弃时eq':>7} {'弃时赔率':>7} {'坏弃':>5} {'坏弃损失':>11}")
    for st in range(4):
        for _, _, name in SIZES:
            g = [r for r in rows if r["street"] == st and sz_name(r["ratio"]) == name]
            if not g:
                continue
            n = len(g)
            f = sum(r["fold"] for r in g)
            c = sum(r["call"] for r in g)
            ra = n - f - c
            folds = [r for r in g if r["fold"]]
            bad = [r for r in folds if r["eq"] > r["odds"] + 0.10]
            loss = sum(call_ev(r) for r in bad)
            eqf = sum(r["eq"] for r in folds) / len(folds) if folds else 0.0
            odf = sum(r["odds"] for r in folds) / len(folds) if folds else 0.0
            print(f"{STREET[st]:4} {name:8} {n:5d} {100*f/n:5.1f}% {100*c/n:5.1f}% "
                  f"{100*ra/n:5.1f}% {sum(mdf(r) for r in g)/n:6.3f} {eqf:7.3f} {odf:7.3f} "
                  f"{len(bad):5d} {loss:11.0f}")


def table_fold_by_strength(rows):
    print("\n== B. 强牌弃牌（翻后，尺寸 0.5–1.5×） ==")
    print(f"{'牌型':8} {'决策':>5} {'弃%':>6} {'弃时eq':>7} {'eq≥0.7弃':>9} {'其中坏弃':>8} {'坏弃损失':>10}")
    for cat in range(9):
        g = [r for r in rows if r["street"] >= 1 and 0.5 <= r["ratio"] < 1.5
             and r["cat"] == cat]
        if not g:
            continue
        folds = [r for r in g if r["fold"]]
        strong = [r for r in folds if r["eq"] >= 0.70]
        bad = [r for r in strong if r["eq"] > r["odds"] + 0.10]
        loss = sum(call_ev(r) for r in bad)
        eqf = sum(r["eq"] for r in folds) / len(folds) if folds else 0.0
        print(f"{CAT[cat]:8} {len(g):5d} {100*len(folds)/len(g):5.1f}% {eqf:7.3f} "
              f"{len(strong):9d} {len(bad):8d} {loss:10.0f}")

    print("\n  强牌弃牌明细（eq ≥ 0.80，翻后，尺寸 ≥ 0.5×，按 eq 降序取前 12）:")
    ex = [r for r in rows if r["street"] >= 1 and r["ratio"] >= 0.5 and r["fold"]
          and r["eq"] >= 0.80]
    ex.sort(key=lambda r: -r["eq"])
    for r in ex[:12]:
        print(f"    {STREET[r['street']]} {sz_name(r['ratio']):8} pot={r['pot_before']:5d} "
              f"need={r['need']:5d} eq={r['eq']:.3f} odds={r['odds']:.3f} "
              f"cat={CAT[r['cat']]:3} {r['hole']:4} board={r['board']}")
    print(f"    eq≥0.80 的弃牌合计 {len(ex)} 个，"
          f"事后 EV 损失 {sum(call_ev(r) for r in ex):+.0f} 筹码")


def table_eq_vs_size(rows):
    print("\n== C. 同一 eq 区间内，弃牌率是否随尺寸上升（翻后） ==")
    bins = [(0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 1.01)]
    print(f"{'eq 区间':10} " + " ".join(f"{name:>12}" for _, _, name in SIZES[:4]))
    for lo, hi in bins:
        cells = []
        for _, _, name in SIZES[:4]:
            g = [r for r in rows if r["street"] >= 1 and lo <= r["eq"] < hi
                 and sz_name(r["ratio"]) == name]
            if not g:
                cells.append(f"{'-':>12}")
            else:
                cells.append(f"{100*sum(r['fold'] for r in g)/len(g):5.1f}% (n={len(g):4d})")
        print(f"{lo:.2f}-{hi:.2f}   " + " ".join(f"{c:>12}" for c in cells))


def table_rule(rows, margin=0.0):
    print(f"\n== D. 规则「eq > 赔率 + {margin:.2f} 就跟注」vs 实际 ==")
    kept = [r for r in rows if r["fold"] and r["eq"] > r["odds"] + margin]
    avoided = [r for r in rows if r["call"] and r["eq"] <= r["odds"] - 0.10]
    gain = sum(call_ev(r) for r in kept)
    save = sum((r["odds"] - r["eq"]) * r["pot"] for r in avoided)
    print(f"  实际弃牌、规则会跟：{len(kept):5d} 个，事后 EV 差 {gain:+11.0f} 筹码")
    print(f"  实际跟注、规则会弃：{len(avoided):5d} 个，事后 EV 差 {save:+11.0f} 筹码")
    print(f"  合计（相对实际动作的事后 EV 上界）：{gain + save:+11.0f} 筹码 "
          f"= {(gain + save) / 100 / 6000:+.3f} BB/手（按 6,000 手折算）")

    print(f"\n  按尺寸：规则继续率 vs MDF")
    print(f"{'尺寸':8} {'决策':>5} {'MDF':>6} {'实际继续%':>9} {'规则继续%':>9} "
          f"{'规则=跟%':>8}")
    for _, _, name in SIZES:
        g = [r for r in rows if sz_name(r["ratio"]) == name]
        if not g:
            continue
        n = len(g)
        act = sum(1 for r in g if not r["fold"])
        rule = sum(1 for r in g if r["eq"] > r["odds"] + margin)
        print(f"{name:8} {n:5d} {sum(mdf(r) for r in g)/n:6.3f} {100*act/n:8.1f}% "
              f"{100*rule/n:8.1f}% {100*rule/n:7.1f}%")


def table_loss_decomp(rows):
    print("\n== E. 坏弃损失按「eq 高出赔率多少」分解 ==")
    folds = [r for r in rows if r["fold"]]
    buckets = [(0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 1.0)]
    print(f"{'eq-赔率':10} {'个数':>5} {'损失':>10} {'占全部坏弃%':>10}")
    total = sum(call_ev(r) for r in folds if r["eq"] > r["odds"] + 0.05)
    for lo, hi in buckets:
        g = [r for r in folds if lo < r["eq"] - r["odds"] <= hi]
        loss = sum(call_ev(r) for r in g)
        print(f"{lo:.2f}-{hi:.2f}     {len(g):5d} {loss:10.0f} "
              f"{100*loss/total if total else 0:9.1f}%")
    print(f"合计（eq-赔率 > 0.05）：{len([r for r in folds if r['eq'] > r['odds'] + 0.05])} 个，"
          f"{total:.0f} 筹码")


def main():
    paths = sys.argv[1:]
    if not paths:
        paths = glob.glob("/Users/mingming/models/range_r10/facing/*.csv")
    rows = load(sorted(paths))
    print(f"样本：{len(rows)} 个面对下注决策（{len(paths)} 段日志）")
    table_fold_structure(rows)
    table_fold_by_strength(rows)
    table_eq_vs_size(rows)
    table_rule(rows, 0.0)
    table_rule(rows, 0.05)
    table_loss_decomp(rows)


if __name__ == "__main__":
    main()
