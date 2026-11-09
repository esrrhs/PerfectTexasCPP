#!/usr/bin/env python3
"""外部评测汇总：一个或多个 perfecttexas slumbot CSV → 判据所需的全部数字。

用法：python3 tools/ext_report.py FILE.csv [FILE2.csv ...]
口径：只认全下 EV 调整后的 adjusted 列；BB = 100 筹码；主指标 = BB/手。
"""
import csv
import math
import sys


def load(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if not r.get("winnings"):
                    continue
                r["_src"] = p.rsplit("/", 1)[-1]
                rows.append(r)
    return rows


def bb(rows):
    return sum(float(r["adjusted"]) for r in rows) / 100.0


def stat(name, rows, denom):
    n = len(rows)
    if n == 0:
        print(f"  {name}: n=0")
        return
    vals = [float(r["adjusted"]) / 100.0 for r in rows]
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = float("nan")
    per_hand = bb(rows) / denom
    print(f"  {name}: n={n:5d} 每手 {mean:+7.3f} BB (SE {se:5.3f}, 95%CI ±{1.96*se:5.3f})"
          f" | 贡献 {per_hand:+7.3f} BB/手")


def main(paths):
    rows = load(paths)
    if not rows:
        print("没有有效行")
        return
    n = len(rows)
    total = bb(rows)
    print(f"合并 {n} 手（{len(paths)} 个文件）")
    print(f"点估计 {total/n:+7.4f} BB/手，合计 {total:+10.1f} BB")
    print("\n分段（每 250 手，看漂移）：")
    for i in range(0, n, 250):
        seg = rows[i:i + 250]
        v = bb(seg) / len(seg)
        print(f"  {i+1:5d}-{i+len(seg):5d}: {v:+7.3f} BB/手")

    print("\n按座位（client_pos=0 ⇒ BB，1 ⇒ SB）：")
    for pos, nm in (("0", "BB"), ("1", "SB")):
        stat(nm, [r for r in rows if r["client_pos"] == pos], n)

    print("\n底池规模（按 |adjusted| 近似分桶）：")
    for lo, hi, nm in ((0, 500, "小池 <5BB"), (500, 2000, "中池 5-20BB"),
                       (2000, 4000, "大池 20-40BB"), (4000, 10**9, "超大池 >40BB")):
        sel = [r for r in rows if lo <= abs(float(r["adjusted"])) < hi]
        stat(nm, sel, n)

    print("\n翻前是否面对 3-bet（翻前段出现 ≥2 个加注）：")
    def raises(r):
        pre = r["action"].split("/")[0]
        return sum(1 for c in pre if c == "b")
    stat("被 3-bet", [r for r in rows if raises(r) >= 2], n)
    stat("未被 3-bet", [r for r in rows if raises(r) < 2], n)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
