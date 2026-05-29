"""ε 正則化 と Q-format 量子化の効果を切り分ける診断スクリプト.

3 種類の d² を計算して比較する:
  A. d2_ref         : 元の Σ (非正則化, ε=0) で計算した numpy d² ※既存
  B. d2_reg_float   : 正則化 Σ_ε で計算した numpy float d² (= "PIC が理想的に出すべき値")
  C. d2_pic_q       : 正則化 + Q-format 量子化 d² (= PIC 実装)

二つの差を分けて測る:
  正則化効果   : B - A
  量子化効果   : C - B (これだけが PIC 実装由来)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from mahalanobis_q import QModel, design_q_model, shrink

RESULTS = Path(__file__).resolve().parent.parent / "results"


def d2_with_model(rms_x, rms_y, mu_x, mu_y, s_xx, s_xy, s_yy, det):
    dx = rms_x - mu_x
    dy = rms_y - mu_y
    return (s_yy * dx * dx - 2.0 * s_xy * dx * dy + s_xx * dy * dy) / det


def first_anomaly(d2: np.ndarray, T: float) -> int:
    above = d2 > T
    return int(np.argmax(above)) if above.any() else -1


def main():
    npz = RESULTS / "bearing3_features.npz"
    js = RESULTS / "bearing3_model.json"
    data = np.load(npz, allow_pickle=True)
    m = json.load(open(js))
    rms_x = data["rms_x"]
    rms_y = data["rms_y"]
    d2_ref = data["d2"]  # = A

    print(f"ρ_xy healthy = {m['s_xy']/(m['s_xx']*m['s_yy'])**0.5:.4f}")

    # ε を 16-bit に収まる最小値に
    eps = 2e-5
    sx, syx, sy, det = shrink(m["s_xx"], m["s_xy"], m["s_yy"], eps)
    print(f"\nε = {eps:.1e} で正則化:")
    print(f"  det: {m['det']:.3e} -> {det:.3e}  ({det/m['det']:.1f}× 拡大)")

    # B: 正則化 float d²
    d2_reg_float = d2_with_model(rms_x, rms_y, m["mu_x"], m["mu_y"], sx, syx, sy, det)

    # C: 正則化 + Q-format 量子化 d²
    qm, info = design_q_model(
        mu_x=m["mu_x"], mu_y=m["mu_y"],
        s_xx=m["s_xx"], s_xy=m["s_xy"], s_yy=m["s_yy"],
        det=m["det"],
        rms_x_max=float(rms_x.max()), rms_y_max=float(rms_y.max()),
        eps=eps,
    )
    print(f"  Q model: frac_x={qm.frac_x}, frac_k={qm.frac_k}, "
          f"k=({qm.k_a},{qm.k_b},{qm.k_c}), μ_q=({qm.mu_x_q},{qm.mu_y_q})")
    d2_q = qm.vectorize_d2(rms_x, rms_y)
    d2_pic = np.array([qm.d2_float_equiv(int(v)) for v in d2_q])

    # 効果切り分け
    reg_effect = d2_reg_float - d2_ref
    quant_effect = d2_pic - d2_reg_float

    print("\n=== 3 系列の値域 ===")
    for name, arr in [("A: ε=0 ref", d2_ref),
                      ("B: ε=2e-5 ref (=PIC理想)", d2_reg_float),
                      ("C: ε=2e-5 + Q量子化 (PIC)", d2_pic)]:
        print(f"  {name:32s} min={arr.min():9.3f}  mean={arr.mean():9.3f}  max={arr.max():9.3f}")

    print("\n=== 量子化単独の誤差 (C - B) ===")
    print(f"  max |Δ|   = {np.max(np.abs(quant_effect)):.4f}")
    print(f"  mean |Δ|  = {np.mean(np.abs(quant_effect)):.4f}")
    print(f"  max rel%  = {np.max(np.abs(quant_effect)/np.where(np.abs(d2_reg_float)>1e-9, np.abs(d2_reg_float),1.0)) * 100:.3f} %")

    print("\n=== 正則化効果単独 (B - A) ===")
    print(f"  max Δ     = {reg_effect.max():.3f}")
    print(f"  min Δ     = {reg_effect.min():.3f}")
    print(f"  mean Δ    = {reg_effect.mean():.3f}")

    print("\n=== しきい値感度 (B = PIC理想を使って T を選ぶ) ===")
    healthy_d2 = d2_reg_float[:215]
    T_99 = float(np.percentile(healthy_d2, 99))
    T_999 = float(np.percentile(healthy_d2, 99.9))
    print(f"  healthy(=最初215窓) 99%ile = {T_99:.3f}, 99.9%ile = {T_999:.3f}")
    for T_name, T in [("99%ile", T_99), ("99.9%ile", T_999), ("ref T=50", 50.0)]:
        fa_a = first_anomaly(d2_ref, T)
        fa_b = first_anomaly(d2_reg_float, T)
        fa_c = first_anomaly(d2_pic, T)
        agree_bc = ((d2_reg_float > T) == (d2_pic > T)).mean() * 100
        print(f"  T={T_name:10s}={T:8.3f}  first(A,B,C)=({fa_a},{fa_b},{fa_c})  B↔C 一致={agree_bc:.2f}%")

    # 推奨 T = 健常 99.9%ile (B 系列, PIC 理想)
    T_pic = T_999
    fa_b = first_anomaly(d2_reg_float, T_pic)
    fa_c = first_anomaly(d2_pic, T_pic)
    print(f"\n=== 採用しきい値 T (PIC 系列 B の健常 99.9%ile) = {T_pic:.3f} ===")
    print(f"  Q値での T_q = {int(round(T_pic * (1 << (2*qm.frac_x + qm.frac_k - qm.output_shift))))}")
    print(f"  first anomaly: B(=理想)={fa_b}, C(=PIC)={fa_c}  → 量子化由来の遅延 = {fa_c-fa_b} 窓")

    # 保存
    out = RESULTS / "bearing3_diagnose.json"
    with open(out, "w") as f:
        json.dump({
            "eps": eps,
            "ranges": {
                "A": [float(d2_ref.min()), float(d2_ref.max())],
                "B": [float(d2_reg_float.min()), float(d2_reg_float.max())],
                "C": [float(d2_pic.min()), float(d2_pic.max())],
            },
            "quantization": {
                "max_abs": float(np.max(np.abs(quant_effect))),
                "mean_abs": float(np.mean(np.abs(quant_effect))),
            },
            "regularization": {
                "max": float(reg_effect.max()),
                "min": float(reg_effect.min()),
                "mean": float(reg_effect.mean()),
            },
            "thresholds": {
                "p99": T_99, "p999": T_999, "T50": 50.0, "T_used": T_pic,
            },
            "first_anomaly": {
                "ref_A_p999": first_anomaly(d2_ref, T_999),
                "reg_B_p999": fa_b,
                "pic_C_p999": fa_c,
                "quantization_delay_windows": fa_c - fa_b,
            },
            "model": qm.to_dict(),
        }, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
