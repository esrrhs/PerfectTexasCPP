#!/usr/bin/env python3
"""R10 第 1 步：离线范围预测的精度评估（numpy，无 torch 依赖）。

读 tools/range_dataset.cpp 写出的二进制，训练一个低秩双线性 softmax 回归：
    u      = ctx @ U                    (B x r)
    logits = u @ V^T + ws * strength + b (B x 169)
标签是「对手动作似然的运行后验」在 169 个翻前类上的分布。

指标（按街 / 底池 / 下注尺寸切片）：
  - KL(真范围 || 预测范围)，单位 nats
  - 真范围落在预测 top 20% 类里的质量（覆盖率）
  - 真实底牌类是否在预测 top 20%（随机 = 20%）
  - H(真范围)：不可约不确定性
  - -log p(真实底牌类)：开天眼看牌（点质量）口径的损失
"""

import argparse
import struct

import numpy as np

CTX_DIM = 532
NCLASS = 169

DTYPE = np.dtype([
    ("street", "u1"), ("toAct", "u1"), ("targetSeat", "u1"), ("slot", "u1"),
    ("trueClass", "u1"), ("boardN", "u1"), ("board", "u1", 5), ("oppHole", "u1", 2),
    ("pot", "<i4"), ("toCall", "<i4"), ("effStack", "<i4"), ("betRatio", "<f4"),
    ("gameSeed", "<u8"), ("gameIdx", "<i4"), ("decisionIdx", "<i4"),
    ("ctx", "<f4", CTX_DIM), ("strength", "<f4", NCLASS), ("target", "<f4", NCLASS),
])

STREET_NAMES = ["preflop", "flop", "turn", "river"]


def load(paths):
    rows = []
    for p in paths:
        with open(p, "rb") as f:
            magic = f.read(8)
            if magic[:6] != b"PTXRG1":
                raise SystemExit(f"{p}: 不是 PTXRG1")
            ver, ctx_dim, nclass, count = struct.unpack("<4i", f.read(16))
            if ctx_dim != CTX_DIM or nclass != NCLASS:
                raise SystemExit(f"{p}: 维度不符 {ctx_dim}/{nclass}")
            # count 只在生成器正常收尾时回填；生成中的文件按剩余字节全读。
            data = np.fromfile(f, dtype=DTYPE, count=(count if count > 0 else -1))
            rows.append(data)
    return np.concatenate(rows)


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def kl(t, p, eps=1e-12):
    return float(np.sum(t * (np.log(np.maximum(t, eps)) - np.log(np.maximum(p, eps))),
                         axis=1).mean())


def entropy(t):
    return float(-np.sum(t * np.log(np.maximum(t, 1e-12)), axis=1).mean())


def topk_mass_and_hit(t, p, truth, frac=0.2):
    k = max(1, int(round(NCLASS * frac)))
    order = np.argsort(-p, axis=1)[:, :k]
    rows = np.arange(p.shape[0])[:, None]
    mass = t[rows, order].sum(axis=1).mean()
    hit = (order == truth[:, None]).any(axis=1).mean()
    return float(mass), float(hit)


def rank_of_truth(p, truth):
    order = np.argsort(-p, axis=1)
    pos = np.empty_like(order)
    rows = np.arange(p.shape[0])[:, None]
    pos[rows, order] = np.arange(NCLASS)[None, :]
    return pos[np.arange(p.shape[0]), truth]


def range_width90(t):
    """覆盖 90% 质量所需的最少类数（真范围的集中度）。"""
    s = np.sort(t, axis=1)[:, ::-1]
    c = np.cumsum(s, axis=1)
    return int(np.argmax(c >= 0.9, axis=1).mean() + 1)


class Bilinear:
    def __init__(self, ctx_dim, nclass, rank=16, seed=0):
        rng = np.random.default_rng(seed)
        self.U = (rng.standard_normal((ctx_dim, rank)) * 0.02).astype(np.float32)
        self.V = (rng.standard_normal((nclass, rank)) * 0.02).astype(np.float32)
        self.ws = np.float32(0.0)
        self.b = np.zeros(nclass, dtype=np.float32)

    def logits(self, ctx, strength):
        return (ctx @ self.U) @ self.V.T + self.ws * strength + self.b

    def get_state(self):
        return {k: getattr(self, k).copy() for k in ("U", "V", "ws", "b")}

    def set_state(self, st):
        for k, v in st.items():
            setattr(self, k, v.copy())

    def fit(self, ctx, strength, target, val, epochs=60, batch=128, lr=3e-4, l2=3e-3,
            patience=8, verbose=True):
        n = ctx.shape[0]
        rng = np.random.default_rng(123)
        params = {"U": self.U, "V": self.V, "b": self.b}
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}
        ms = np.float32(0.0)
        vs = np.float32(0.0)
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0
        best = (1e9, self.get_state())
        bad = 0
        for ep in range(epochs):
            idx = rng.permutation(n)
            tot = 0.0
            for st in range(0, n, batch):
                bi = idx[st:st + batch]
                c, s, t = ctx[bi], strength[bi], target[bi]
                u = c @ self.U
                z = u @ self.V.T + self.ws * s + self.b
                lsm = log_softmax(z)
                p = np.exp(lsm)
                tot += -float(np.sum(t * lsm))
                g = (p - t) / len(bi)
                gU = c.T @ (g @ self.V) + l2 * self.U
                gV = g.T @ u + l2 * self.V
                gb = g.sum(axis=0)
                gs = np.float32(np.sum(g * s))
                # 全局梯度裁剪：标准化后的 ctx 容易把初学者的 logits 推爆。
                gn = float(np.sqrt(np.sum(gU * gU) + np.sum(gV * gV) + np.sum(gb * gb) +
                                   float(gs) ** 2))
                if gn > 1.0:
                    sc = np.float32(1.0 / gn)
                    gU *= sc; gV *= sc; gb *= sc; gs = np.float32(gs * sc)
                step += 1
                for k, grad in (("U", gU), ("V", gV), ("b", gb)):
                    m[k] *= b1
                    m[k] += (1 - b1) * grad
                    v[k] *= b2
                    v[k] += (1 - b2) * grad * grad
                    params[k] -= lr * (m[k] / (1 - b1 ** step)) / (
                        np.sqrt(v[k] / (1 - b2 ** step)) + eps)
                ms = b1 * ms + (1 - b1) * gs
                vs = b2 * vs + (1 - b2) * gs * gs
                self.ws -= np.float32(lr * (ms / (1 - b1 ** step)) /
                                      (np.sqrt(vs / (1 - b2 ** step)) + eps))
            vc, vstr, vs_ = val
            vl = float(-np.sum(vs_ * log_softmax(self.logits(vc, vstr))) / len(vc))
            if vl < best[0] - 1e-4:
                best = (vl, self.get_state())
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"  early stop @ epoch {ep+1}（val CE {best[0]:.4f}）")
                    break
            if verbose and (ep + 1) % 10 == 0:
                print(f"  epoch {ep+1:3d}  train CE {tot/n:.4f}  val CE {vl:.4f}")
        self.set_state(best[1])
        return best[0]


def bet_bucket(br):
    if br <= 0:
        return 0
    if br < 0.5:
        return 1
    if br < 0.75:
        return 2
    if br < 1.25:
        return 3
    if br < 2.0:
        return 4
    return 5


def table_predict(d, tr, te, alpha=10.0):
    """非参数基线：按 (街, 动作槽, 下注尺寸桶) 取训练集的平均真范围，向先验收缩。"""
    prior = d["target"][tr].mean(axis=0)
    key = (d["street"].astype(np.int64) * 100 +
           d["slot"].astype(np.int64) * 6 +
           np.array([bet_bucket(x) for x in d["betRatio"]]))
    out = np.tile(prior, (int(te.sum()), 1)).astype(np.float32)
    ktr, kte = key[tr], key[te]
    for k in np.unique(ktr):
        mtr = ktr == k
        mte = kte == k
        if not mte.any():
            continue
        w = d["target"][tr][mtr].sum(axis=0)
        out[mte] = (w + alpha * prior) / (mtr.sum() + alpha)
    return out


def hit_vec(p, truth, frac=0.2):
    k = max(1, int(round(NCLASS * frac)))
    order = np.argsort(-p, axis=1)[:, :k]
    return (order == truth[:, None]).any(axis=1).astype(np.float64)


def kl_vec(t, p, eps=1e-12):
    return np.sum(t * (np.log(np.maximum(t, eps)) - np.log(np.maximum(p, eps))), axis=1)


def air_share_table(tag, t, p, strength, br):
    """防诈唬：空气（牌力后 30%）在真范围/预测范围里的权重，按下注尺寸分桶。"""
    print(f"\n== 空气（strength≤0.30）权重 · {tag} ==")
    print(f"{'尺寸桶':<12}{'n':>6}{'真范围空气':>12}{'预测空气':>10}")
    buckets = [(0.0, 0.0, "过牌/跟注"), (1e-9, 0.5, "<0.5x"), (0.5, 1.0, "0.5-1x"),
               (1.0, 1.5, "1-1.5x"), (1.5, 3.0, "1.5-3x"), (3.0, 1e9, ">3x")]
    air = strength <= 0.30
    for lo, hi, name in buckets:
        m = (br >= lo) & (br <= hi) if lo == hi else (br > lo) & (br <= hi)
        if m.sum() == 0:
            continue
        ta = float((t[m] * air[m]).sum(axis=1).mean())
        pa = float((p[m] * air[m]).sum(axis=1).mean())
        print(f"{name:<12}{int(m.sum()):>6}{ta*100:>11.1f}%{pa*100:>9.1f}%")


def paired(tag, a, b):
    d = a - b
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    print(f"  {tag}: 差值 {d.mean():+.4f} ± {se:.4f}（{len(d)} 条，配对）")


def evaluate(tag, t, p, truth):
    unif = np.full_like(p, 1.0 / NCLASS)
    mass, hit = topk_mass_and_hit(t, p, truth)
    umass, _ = topk_mass_and_hit(t, unif, truth)
    ent = entropy(t)
    nll_true = float(-np.log(np.maximum(p[np.arange(len(truth)), truth], 1e-12)).mean())
    se = float(np.sqrt(max(hit * (1 - hit), 1e-9) / len(t)))
    print(f"{tag:<24} n={len(t):5d}  KL={kl(t,p):6.3f}（均匀 {kl(t,unif):6.3f}）  "
          f"H={ent:5.3f}  覆盖@20%={mass*100:5.1f}%（均匀 {umass*100:5.1f}%）  "
          f"真牌@20%={hit*100:5.1f}±{se*100:4.1f}%（随机 20.0%）  -logp_true={nll_true:5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = load(args.files)
    print(f"读入 {len(d)} 条样本；target 和 = {d['target'].sum(axis=1).mean():.6f}")
    print("街分布: " + "  ".join(
        f"{STREET_NAMES[s]}={int((d['street']==s).sum())}" for s in range(4)))
    print(f"真范围集中度：覆盖 90% 质量需 {range_width90(d['target'])}/169 类"
          f"（均匀需 {int(0.9*NCLASS)+1}）")

    # 按局分 train/val/test = 60/20/20，同局决策不跨集。
    gid = {g: i for i, g in enumerate(np.unique(d["gameSeed"]))}
    gidx = np.array([gid[g] for g in d["gameSeed"]])
    tr = (gidx % 5) < 3
    va = (gidx % 5) == 3
    te = (gidx % 5) == 4
    if te.sum() == 0:
        va, te = va | te, va
    print(f"train {int(tr.sum())} / val {int(va.sum())} / test {int(te.sum())}（按局）")

    ctx = d["ctx"].astype(np.float32)
    strength = d["strength"].astype(np.float32)
    target = d["target"].astype(np.float32)
    mu = ctx[tr].mean(axis=0)
    sd = ctx[tr].std(axis=0) + 1e-6
    ctx = (ctx - mu) / sd
    # 稀疏 one-hot（某张牌在训练局里很少出现）会让标准化值爆到上千，直接把
    # 线性模型推到极端置信。截断到 ±8。
    ctx = np.clip(ctx, -8.0, 8.0)

    # 基线 1：训练集平均 target（先验）。
    prior = target[tr].mean(axis=0)
    pt = np.tile(prior, (int(te.sum()), 1))
    # 基线 2：只用 strength 的 softmax（拟合一个温度 + 类偏置）。
    sm = Bilinear(CTX_DIM, NCLASS, rank=1, seed=1)
    sm.U[:] = 0.0
    sm.fit(np.zeros_like(ctx[tr]), strength[tr], target[tr],
           (np.zeros_like(ctx[va]), strength[va], target[va]), epochs=30, l2=1e-2,
           verbose=False)
    ps = softmax(sm.logits(np.zeros_like(ctx[te]), strength[te]))

    model = Bilinear(CTX_DIM, NCLASS, rank=args.rank, seed=args.seed)
    print(f"\n训练低秩双线性模型（rank={args.rank}）")
    model.fit(ctx[tr], strength[tr], target[tr],
              (ctx[va], strength[va], target[va]), epochs=args.epochs)

    p = softmax(model.logits(ctx[te], strength[te]))
    ptbl = table_predict(d, tr, te)
    t = target[te]
    truth = d["trueClass"][te]
    street = d["street"][te]
    pot = d["pot"][te]
    br = d["betRatio"][te]

    print("\n== 测试集总体 ==")
    omass, ohit = topk_mass_and_hit(t, t, truth)
    print(f"上限（用真范围本身预测）: 覆盖@20%={omass*100:.1f}%  真牌@20%={ohit*100:.1f}%")
    evaluate("模型", t, p, truth)
    print("\n== 基线 ==")
    evaluate("训练集先验(平均)", t, pt, truth)
    evaluate("街槽尺寸查表", t, ptbl, truth)
    evaluate("仅 strength", t, ps, truth)
    order = np.argsort(-prior)[: int(round(NCLASS * 0.2))]
    print(f"先验 top-20% 类命中率 {np.isin(truth, order).mean()*100:.1f}%（随机 20.0%）")

    print("\n== 配对检验（模型 - 基线，同一批测试样本）==")
    hm, hp, ht = hit_vec(p, truth), hit_vec(pt, truth), hit_vec(ptbl, truth)
    paired("真牌@20% 模型-先验", hm, hp)
    paired("真牌@20% 模型-查表", hm, ht)
    paired("KL 模型-先验（负=更好）", kl_vec(t, p), kl_vec(t, pt))
    paired("KL 模型-查表（负=更好）", kl_vec(t, p), kl_vec(t, ptbl))

    print("\n== 分街 ==")
    for s in range(4):
        m = street == s
        if m.sum() > 0:
            evaluate(STREET_NAMES[s], t[m], p[m], truth[m])
            _, oh = topk_mass_and_hit(t[m], t[m], truth[m])
            print(f"{'  ^ 上限':<24} 真牌@20%={oh*100:.1f}%")

    print("\n== 大池 / 大注 ==")
    for name, m in (("pot>=1500", pot >= 1500), ("pot>=4000", pot >= 4000),
                    ("bet>=0.75pot", br >= 0.75), ("bet>=1.5pot", br >= 1.5),
                    ("大池&大注", (pot >= 1500) & (br >= 0.75))):
        if m.sum() > 0:
            evaluate(name, t[m], p[m], truth[m])
            _, oh = topk_mass_and_hit(t[m], t[m], truth[m])
            print(f"{'  ^ 上限':<24} 真牌@20%={oh*100:.1f}%")
        else:
            print(f"{name:<24} n=0")

    air_share_table("测试集（模型 vs 真范围）", t, p, d["strength"][te], br)

    print("\n== 反事实：范围预测 vs 开天眼看牌 ==")
    ent = entropy(t)
    ce = float(-np.sum(t * np.log(np.maximum(p, 1e-12)), axis=1).mean())
    nll_true = float(-np.log(np.maximum(p[np.arange(len(truth)), truth], 1e-12)).mean())
    print(f"H(真范围)={ent:.3f} nats | 范围标签 CE={ce:.3f}（H={ent:.3f}, KL={ce-ent:.3f}）")
    print(f"点质量（已知底牌）-logp_true={nll_true:.3f}；均匀={np.log(NCLASS):.3f}")
    print(f"真牌类平均排名={rank_of_truth(p, truth).mean():.1f}/169（随机 85）")


if __name__ == "__main__":
    main()
