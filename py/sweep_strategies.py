"""asm シフト戦略を変えて agreement を測定."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"

MU_X_Q, MU_Y_Q = 2276, 2320
K_A, K_B, K_C = 26188, 23062, 26081
T_Q_BASE = 11898   # for >>16 strategy
FRAC_X = 14


def s16(v):
    v &= 0xFFFF
    return v if v < 0x8000 else v - 0x10000

def s32(v):
    v &= 0xFFFFFFFF
    return v if v < 0x80000000 else v - 0x100000000

def s48(v):
    v &= 0xFFFFFFFFFFFF
    return v if v < 0x800000000000 else v - 0x1000000000000


def strat_truncate(in_x_q, in_y_q):
    """>>16 truncation (現行 asm)."""
    dx = s16(in_x_q - MU_X_Q)
    dy = s16(in_y_q - MU_Y_Q)
    p = s32(dx*dx); term_c = s32(K_C * s16((p >> 16) & 0xFFFF))
    p = s32(dy*dy); term_a = s32(K_A * s16((p >> 16) & 0xFFFF))
    p = s32(s32(dx*dy) << 1); term_b = s32(K_B * s16((p >> 16) & 0xFFFF))
    return s32(term_c + term_a - term_b)


def strat_round(in_x_q, in_y_q):
    """>>16 with rounding (+0x8000 before shift)."""
    dx = s16(in_x_q - MU_X_Q)
    dy = s16(in_y_q - MU_Y_Q)
    def rshift_round(v):
        return s16(((s32(v) + 0x8000) >> 16) & 0xFFFF)
    p = s32(dx*dx); term_c = s32(K_C * rshift_round(p))
    p = s32(dy*dy); term_a = s32(K_A * rshift_round(p))
    p = s32(s32(dx*dy) << 1); term_b = s32(K_B * rshift_round(p))
    return s32(term_c + term_a - term_b)


def strat_full48(in_x_q, in_y_q):
    """48-bit acc, no shift (frac precision 維持). 6-byte arithmetic on PIC."""
    dx = s16(in_x_q - MU_X_Q)
    dy = s16(in_y_q - MU_Y_Q)
    # k * dx² for full 32-bit dx² is 48-bit result
    p_xx = s32(dx*dx)  # 32-bit
    p_yy = s32(dy*dy)
    p_xy = s32(dx*dy)
    # 16-bit × 32-bit → 48-bit
    term_c = s48(K_C * p_xx)
    term_a = s48(K_A * p_yy)
    term_b = s48(2 * K_B * p_xy)
    return s48(term_c + term_a - term_b)


def eval_strategy(name, fn, T_q, scale_to_phys, label):
    data = np.load(RESULTS / "bearing3_features.npz", allow_pickle=True)
    rms_x = data["rms_x"]
    rms_y = data["rms_y"]
    in_x_q = np.round(rms_x * (1 << FRAC_X)).astype(np.int64)
    in_y_q = np.round(rms_y * (1 << FRAC_X)).astype(np.int64)

    acc = np.array([fn(int(x), int(y)) for x, y in zip(in_x_q, in_y_q)])

    # B 系列
    with open(RESULTS / "bearing3_model.json") as f:
        m = json.load(f)
    eps = 2e-5
    sx = m["s_xx"] + eps
    sy = m["s_yy"] + eps
    sxy = m["s_xy"]
    det = sx * sy - sxy * sxy
    dxf = rms_x - m["mu_x"]
    dyf = rms_y - m["mu_y"]
    d2_B = (sy * dxf**2 - 2*sxy * dxf*dyf + sx * dyf**2) / det

    # T_q in physical
    T_phys = T_q / scale_to_phys
    anomaly_asm = acc > T_q
    anomaly_B = d2_B > T_phys
    agree = (anomaly_asm == anomaly_B).mean() * 100
    miss = ((anomaly_B) & (~anomaly_asm)).sum()
    fp = ((~anomaly_B) & (anomaly_asm)).sum()
    fa_B = int(np.argmax(anomaly_B)) if anomaly_B.any() else -1
    fa_asm = int(np.argmax(anomaly_asm)) if anomaly_asm.any() else -1
    d2_asm_phys = acc / scale_to_phys
    err = d2_asm_phys - d2_B
    rel = np.max(np.abs(err) / np.where(np.abs(d2_B) > 1e-9, np.abs(d2_B), 1.0)) * 100

    print(f"{name} ({label}):")
    print(f"  acc range: [{acc.min()}, {acc.max()}]")
    print(f"  agree = {agree:.2f}%  miss = {miss}  fp = {fp}")
    print(f"  first anomaly: B = {fa_B}, asm = {fa_asm}  delay = {fa_asm - fa_B}")
    print(f"  err vs B: max |Δ| = {np.max(np.abs(err)):.4f}, mean = {np.mean(np.abs(err)):.4f}, max rel = {rel:.2f}%")
    print()


def strat_shift12(in_x_q, in_y_q):
    """>>12 truncation. (dx²)>>12 を 16-bit signed として扱う.
    dx²_max = 7451² ≈ 5.5e7 → >>12 = 13500 (< 16384). 16-bit signed OK."""
    dx = s16(in_x_q - MU_X_Q)
    dy = s16(in_y_q - MU_Y_Q)
    def rsh12(v):
        # >>12 with signed semantics
        v32 = s32(v)
        return s16((v32 >> 12) & 0xFFFF)
    p = s32(dx*dx); term_c = s32(K_C * rsh12(p))
    p = s32(dy*dy); term_a = s32(K_A * rsh12(p))
    p = s32(s32(dx*dy) << 1); term_b = s32(K_B * rsh12(p))
    return s32(term_c + term_a - term_b)


if __name__ == "__main__":
    print("=== asm シフト戦略比較 ===\n")
    eval_strategy(">>16 truncate (現行)", strat_truncate,
                  T_Q_BASE, 1 << 12, "32-bit acc, x16 shift")
    eval_strategy(">>16 round (+0x8000)", strat_round,
                  T_Q_BASE, 1 << 12, "32-bit acc, x16 round")
    eval_strategy(">>12 truncate", strat_shift12,
                  int(2.905 * (1 << 16)), 1 << 16, "32-bit acc, x12 shift")
    # 48-bit acc, T_q = 11898 * 2^16 = 779_739,000 程度
    eval_strategy("full 48-bit, no shift", strat_full48,
                  T_Q_BASE * (1 << 16), 1 << 28, "48-bit acc, exact")
