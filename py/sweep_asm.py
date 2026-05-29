"""asm の bit-exact 演算を全 2156 窓に走らせて、numpy regularized ref と比較.

asm strategy:
  acc = k_c * (dx²>>16) + k_a * (dy²>>16) - k_b * ((2*dx*dy)>>16)   (32-bit signed)
  anomaly := acc >= T_q (= 11898)   ; asm の subwf+btfsc(C) と一致
  → これは d²_physical * 2^12 のスケール。

比較対象: regularized float d² (ε=2e-5)、と元データ d²_ref。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"

MU_X_Q = 2276
MU_Y_Q = 2320
K_A    = 26188
K_B    = 23062
K_C    = 26081
T_Q    = 11898
FRAC_X = 14


def s16(v):
    v &= 0xFFFF
    return v if v < 0x8000 else v - 0x10000


def s32(v):
    v &= 0xFFFFFFFF
    return v if v < 0x80000000 else v - 0x100000000


def asm_d2(in_x_q, in_y_q):
    """asm と完全同じ semantics で d² 計算 → 32-bit signed acc."""
    dx = s16(in_x_q - MU_X_Q)
    dy = s16(in_y_q - MU_Y_Q)

    # term_c = k_c * (dx² >> 16)
    p = s32(dx * dx)
    p_hi = s16((p >> 16) & 0xFFFF)
    term_c = s32(K_C * p_hi)

    # term_a = k_a * (dy² >> 16)
    p = s32(dy * dy)
    p_hi = s16((p >> 16) & 0xFFFF)
    term_a = s32(K_A * p_hi)

    # term_b = k_b * ((2*dx*dy) >> 16)
    p = s32(dx * dy)
    p = s32(p << 1)   # *2 in 32-bit
    p_hi = s16((p >> 16) & 0xFFFF)
    term_b = s32(K_B * p_hi)

    return s32(term_c + term_a - term_b)


def main():
    data = np.load(RESULTS / "bearing3_features.npz", allow_pickle=True)
    rms_x = data["rms_x"]
    rms_y = data["rms_y"]
    d2_ref = data["d2"]                # 元 ε=0 ref (非正則化)

    # Q14 入力に量子化
    in_x_q = np.round(rms_x * (1 << FRAC_X)).astype(np.int64)
    in_y_q = np.round(rms_y * (1 << FRAC_X)).astype(np.int64)
    # 16-bit unsigned 範囲確認
    print(f"in_x_q range: [{in_x_q.min()}, {in_x_q.max()}]  16-bit OK: {in_x_q.max() < (1<<16)}")
    print(f"in_y_q range: [{in_y_q.min()}, {in_y_q.max()}]")

    # asm semantics を 2156 窓に走らせる
    acc = np.array([asm_d2(int(x), int(y)) for x, y in zip(in_x_q, in_y_q)])
    print(f"asm acc range: [{acc.min()}, {acc.max()}]")

    # 異常判定 (asm: subwf+btfsc(C) は acc >= T_Q を見る)
    anomaly_asm = acc >= T_Q

    # 正則化リファレンスの d² と比較 (diagnose.py で出した B 系列に相当する判定)
    # B 系列はそこには npz 保存されていない。再計算する。
    with open(RESULTS / "bearing3_model.json") as f:
        m = json.load(f)
    eps = 2e-5
    sx = m["s_xx"] + eps
    sy = m["s_yy"] + eps
    sxy = m["s_xy"]
    det = sx * sy - sxy * sxy
    dx_f = rms_x - m["mu_x"]
    dy_f = rms_y - m["mu_y"]
    d2_B = (sy * dx_f**2 - 2*sxy * dx_f*dy_f + sx * dy_f**2) / det
    print(f"\n正則化 ref (B) range: [{d2_B.min():.3f}, {d2_B.max():.3f}]")

    # 同等しきい値: T_q / 2^12 (asm scale → physical scale)
    T_phys = T_Q / (1 << 12)
    print(f"T_phys = T_Q / 2^12 = {T_phys:.4f}  (期待: ~2.905)")
    anomaly_B = d2_B > T_phys

    agree = (anomaly_asm == anomaly_B).mean() * 100
    print(f"\nasm vs B (正則化 ref) 一致率: {agree:.3f}%")
    miss = ((anomaly_B) & (~anomaly_asm)).sum()
    fp = ((~anomaly_B) & (anomaly_asm)).sum()
    print(f"  miss = {miss}, false positive = {fp}")

    fa_B = int(np.argmax(anomaly_B)) if anomaly_B.any() else -1
    fa_asm = int(np.argmax(anomaly_asm)) if anomaly_asm.any() else -1
    print(f"  first anomaly: B(正則化ref) = {fa_B}, asm = {fa_asm}")

    # スケール変換: asm acc を physical d² に戻す
    d2_asm_phys = acc / (1 << 12)
    err = d2_asm_phys - d2_B
    rel = err / np.where(np.abs(d2_B) > 1e-9, np.abs(d2_B), 1.0)
    print(f"\nasm 由来 d²_phys vs B (正則化 ref):")
    print(f"  max |Δ|   = {np.max(np.abs(err)):.4f}")
    print(f"  mean |Δ|  = {np.mean(np.abs(err)):.4f}")
    print(f"  max rel%  = {np.max(np.abs(rel))*100:.2f}%")

    # 保存
    out = RESULTS / "bearing3_asm_sweep.json"
    with open(out, "w") as f:
        json.dump({
            "T_q": T_Q,
            "T_phys": T_phys,
            "windows": int(len(rms_x)),
            "acc_range": [int(acc.min()), int(acc.max())],
            "agree_pct": float(agree),
            "miss": int(miss),
            "false_pos": int(fp),
            "first_anomaly_B": fa_B,
            "first_anomaly_asm": fa_asm,
            "err_vs_B": {
                "max_abs": float(np.max(np.abs(err))),
                "mean_abs": float(np.mean(np.abs(err))),
                "max_rel_pct": float(np.max(np.abs(rel)) * 100),
            },
        }, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
