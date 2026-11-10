#!/usr/bin/env python3
"""R60：外部采纳线的统计功效分析（0 算力，只读已有 CSV）。

背景：GOAL.md 的 L0 采纳线 = 「候选点估计 ≥ 基线 + 2·SE_diff」。这是一个**固定样本量下**
的判据（12k vs 15k → 阈值 −0.218/−0.232）。R43 已实测 −0.2495（12,497 手，1.91σ），差 0.03 不过。
本脚本回答：**再花多少外部机时，R43 这类「真效应 ~0.38 BB/手」的候选才有把握过线？**

口径：
  - 逐手全下调整 EV（CSV 的 `adjusted` 列 / 100 = BB），与 tools/ext_report.py 一致。
  - 功效 = P(观测点估计 ≥ 基线均值 + 2·SE_diff | 真效应 = 观测差值)。
  - 两侧都可扩样：SE_diff = sqrt(SD_c/N_c + SD_b/N_b)。

用法：python3 tools/ext_power.py
"""
import csv
import math
import os

B = os.path.expanduser("~/models/ext_stn2/")

R43_FILES = [
    "r43_12k_s96033_trunc498.csv", "r43_12k_s96033.csv", "r43_12k_s96034.csv",
    "r43_12k_s96035.csv", "r43_12k_s96036.csv", "r43_12k_s96037.csv",
    "r43_12k_s96038.csv", "r43_12k_s96039.csv", "r43_s1500_s96032.csv",
]
BASE_FILES = [
    "s1.csv", "s2.csv", "s3.csv", "s96004.csv", "s96005.csv", "s96006.csv",
    "s96007.csv", "s96008.csv", "s96009.csv", "s96013_base.csv",
]
SEC_PER_HAND = 3.1  # 实测：1,500 手段 ≈78 分钟（--delay-ms 150）


def load(files):
    v = []
    for f in files:
        p = B + f if not f.startswith("/") else f
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r.get("winnings") and r.get("adjusted"):
                    v.append(float(r["adjusted"]) / 100.0)
    return v


def stat(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    return n, m, sd, sd / math.sqrt(n)


def phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def power(sd_c, n_c, sd_b, n_b, effect):
    """P(观测差值 ≥ 2·SE_diff)，观测差值 ~ N(effect, SE_diff²)。effect>0 = 候选更好。"""
    se = math.sqrt(sd_c ** 2 / n_c + sd_b ** 2 / n_b)
    return phi(effect / se - 2.0), se


def main():
    nc, mc, sdc, sec_ = stat(load(R43_FILES))
    nb, mb, sdb, seb = stat(load(BASE_FILES))
    effect = mc - mb  # >0 = 候选比基线好（EV 更高）
    se_diff = math.sqrt(sec_ ** 2 + seb ** 2)
    print(f"R43  : n={nc:>6}  均值 {mc:+.4f} BB/手  SD {sdc:.2f}  SE {sec_:.4f}")
    print(f"基线 : n={nb:>6}  均值 {mb:+.4f} BB/手  SD {sdb:.2f}  SE {seb:.4f}")
    print(f"差值 : {effect:+.4f} BB/手（候选更好）  SE_diff {se_diff:.4f}  → {effect/se_diff:.2f}σ")
    print(f"单手 SD {sdc:.2f} BB → 1,500 手段的 SE = {sdc/math.sqrt(1500):.3f} BB/手")
    print()

    print("A) 只扩候选侧（基线固定 %d 手）" % nb)
    print(f"{'候选手数':>9} {'需再打':>8} {'SE_diff':>8} {'2σ 采纳线':>10} {'功效':>7} {'机时':>8}")
    for n in (12497, 18000, 24588, 37000, 60000):
        pw, se = power(sdc, n, sdb, nb, effect)
        line = mb + 2 * se
        hrs = max(0, n - nc) * SEC_PER_HAND / 3600
        print(f"{n:>9} {n-nc:>8} {se:>8.4f} {line:>10.4f} {pw*100:>6.1f}% {hrs:>7.1f}h")
    print()

    print("B) 两侧同时扩样（基线也要补，SE_diff 才真正下降）")
    print(f"{'每侧手数':>9} {'候选再打':>9} {'基线再打':>9} {'SE_diff':>8} {'功效':>7} {'总机时':>8}")
    for n in (12497, 18000, 24588, 37000, 50000):
        pw, se = power(sdc, n, sdb, n, effect)
        hrs = (max(0, n - nc) + max(0, n - nb)) * SEC_PER_HAND / 3600
        print(f"{n:>9} {n-nc:>9} {n-nb:>9} {se:>8.4f} {pw*100:>6.1f}% {hrs:>7.1f}h")
    print()

    print(f"C) 反解：功效 ≥80% 需要每侧多少手（真效应 = 观测差值 {effect:.4f}）")
    lo, hi = 1000, 2000000
    for _ in range(60):
        mid = (lo + hi) / 2
        pw, _se = power(sdc, mid, sdb, mid, effect)
        if pw < 0.80:
            lo = mid
        else:
            hi = mid
    n80 = int(hi)
    hrs = (max(0, n80 - nc) + max(0, n80 - nb)) * SEC_PER_HAND / 3600
    print(f"   每侧 {n80} 手 → 合计再打 {(n80-nc)+(n80-nb)} 手 ≈ {hrs:.1f}h 外部机时（≈{hrs/24:.1f} 天）")


if __name__ == "__main__":
    main()
