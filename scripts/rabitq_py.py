#!/usr/bin/env python3
"""RaBitQ / Extended-RaBitQ 的 Python 参考实现（单位向量感知 + 无偏估计）。
协议（对齐论文 2409.09913）:
- 输入已 L2 归一化向量 o；随机正交 P；o' = P⁻¹o
- B-bit 网格: y = round(t·o')，扫掠最优缩放 t* 使 cos²(ȳ,o') 最大
- 估计量: ⟨o,q⟩ ≈ ⟨ō,q⟩/⟨ō,o⟩，⟨ō,o⟩ = ⟨y,o'⟩/‖y‖（精确可算，无偏）
"""
import numpy as np

def random_orthogonal(d, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q.astype(np.float32)

def quantize_Bbit(rot, B):
    """rot: (N,d) 旋转后单位向量。返回 codes(N,d)∈[0,2^B), factors(N)=⟨ō,o⟩"""
    N, d = rot.shape
    L = 2 ** B
    half = (L - 1) / 2.0
    absr = np.abs(rot)
    codes = np.zeros((N, d), np.int32)
    factors = np.zeros(N, np.float32)
    if B == 1:
        y = np.ones((N, d), np.int32)
        ys = y * np.sign(rot)
        dot = np.sum(ys * rot, axis=1)
        norm = np.sqrt(d)
        factors[:] = (dot / norm).astype(np.float32)
        codes[:] = np.where(rot > 0, 1, 0).astype(np.int32)
        return codes, factors
    # 半整数网格（对齐论文 even-L 码本 {-1.5,-0.5,0.5,1.5} 等）:
    # |y| ∈ {0.5, 1.5, ..., L/2-0.5}; 穿越临界 t = k/|o'_j|, k=1..L/2-1
    ks = np.arange(1, L // 2, dtype=np.float64)          # k=1..L/2-1
    for i in range(N):
        a = absr[i].astype(np.float64)
        a = np.where(a == 0, 1e-12, a)                   # 避免除零 -> t 无 inf -> cos2 无 nan
        t = (ks[None, :] / a[:, None]).ravel()           # (M,) 临界值
        t = np.sort(t)
        best_cos2, best_y = -1.0, None
        for c0 in range(0, len(t), 512):
            tc = t[c0:c0+512]
            c = np.clip(np.floor(tc[:, None] * a[None, :]), 0, L // 2 - 1)  # 穿越次数
            y = c + 0.5                                   # |y| 半整数
            dott = y @ a
            n2 = np.sum(y * y, axis=1)
            cos2 = (dott * dott) / (n2 + 1e-30)
            j = int(np.argmax(cos2))
            if cos2[j] > best_cos2:
                best_cos2 = cos2[j]; best_y = y[j]
        ys = best_y * np.sign(rot[i])
        dot = np.sum(ys * rot[i]); norm = np.sqrt(np.sum(ys ** 2))
        factors[i] = (dot / norm).astype(np.float32)
        half = (L - 1) / 2.0
        codes[i] = (ys + half).astype(np.int32)
    return codes, factors

def reconstruct(P, codes, B):
    N, d = codes.shape
    half = ((2 ** B) - 1) / 2.0
    y = codes.astype(np.float32) - half
    o = (P @ y.T).T
    o /= (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
    return o.astype(np.float32)

def estimate_scores(qrot, P, codes, B, factors):
    """分数: ⟨ō,q⟩/⟨ō,o⟩（无偏估计）"""
    o = reconstruct(P, codes, B)
    return (qrot @ o.T) / factors[None, :]
