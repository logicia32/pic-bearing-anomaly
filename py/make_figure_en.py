"""Bearing 3 の RMS 推移 と PIC 等価 d² / しきい値 / 検出点を 1 枚に."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# default font (English labels)
plt.rcParams["axes.unicode_minus"] = False

MU_X_Q, MU_Y_Q = 2276, 2320
K_A, K_B, K_C = 26188, 23062, 26081
T_Q = 11898
FRAC_X = 14


def s16(v):
    v &= 0xFFFF
    return v if v < 0x8000 else v - 0x10000


def s32(v):
    v &= 0xFFFFFFFF
    return v if v < 0x80000000 else v - 0x100000000


def asm_d2_acc(in_x_q, in_y_q):
    """asm semantics で acc (32-bit signed) を返す."""
    dx = s16(in_x_q - MU_X_Q)
    dy = s16(in_y_q - MU_Y_Q)
    p = s32(dx*dx); term_c = s32(K_C * s16((p >> 16) & 0xFFFF))
    p = s32(dy*dy); term_a = s32(K_A * s16((p >> 16) & 0xFFFF))
    p = s32(s32(dx*dy) << 1); term_b = s32(K_B * s16((p >> 16) & 0xFFFF))
    return s32(term_c + term_a - term_b)


def main():
    data = np.load(RESULTS / "bearing3_features.npz", allow_pickle=True)
    rms_x = data["rms_x"]
    rms_y = data["rms_y"]
    ts_str = data["ts"]
    ts = np.array([np.datetime64(s) for s in ts_str])

    in_x_q = np.round(rms_x * (1 << FRAC_X)).astype(np.int64)
    in_y_q = np.round(rms_y * (1 << FRAC_X)).astype(np.int64)
    acc = np.array([asm_d2_acc(int(x), int(y)) for x, y in zip(in_x_q, in_y_q)])

    anomaly_asm = acc > T_Q
    first_asm = int(np.argmax(anomaly_asm)) if anomaly_asm.any() else -1

    # acc を物理スケールに戻す
    d2_phys_asm = acc / (1 << 12)
    T_phys = T_Q / (1 << 12)

    print(f"first anomaly window = {first_asm}, time = {ts[first_asm]}")
    print(f"final window time = {ts[-1]}, elapsed = {(ts[-1]-ts[first_asm]) / np.timedelta64(1,'h'):.1f} h")
    print(f"acc range = [{acc.min()}, {acc.max()}]")

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.0],
                                          "hspace": 0.12})
    ax1, ax2 = axes

    # Top: RMS_x, RMS_y
    ax1.plot(ts, rms_x, color="#1f77b4", lw=0.8, label="RMS_x  (Ch5)")
    ax1.plot(ts, rms_y, color="#ff7f0e", lw=0.8, label="RMS_y  (Ch6)", alpha=0.75)
    ax1.axvspan(ts[0], ts[214], alpha=0.10, color="#2ca02c")
    ax1.text(ts[107], 0.50,
             "healthy interval\n(training, 215 windows)", fontsize=8.5, color="#2ca02c",
             ha="center", va="top")
    ax1.set_ylabel("RMS  [g]")
    ax1.set_title("NASA IMS Bearing 3 — 35 days of vibration and PIC-equivalent d² detection")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax1.grid(alpha=0.25)
    ax1.set_ylim(0, max(rms_x.max(), rms_y.max()) * 1.10)

    # Bottom: d² (asm) + threshold + detection
    # クリッピング: 負値 (量子化ノイズ) は 0 にして見やすく
    d2_clip = np.clip(d2_phys_asm, 0.01, None)
    ax2.plot(ts, d2_clip, color="#444", lw=0.8, label="d² (PIC 32-bit acc)")
    ax2.axhline(T_phys, color="#d62728", lw=1.2, ls="--",
                label=f"threshold T = {T_phys:.2f}")
    if first_asm > 0:
        for ax in (ax1, ax2):
            ax.axvline(ts[first_asm], color="#d62728", lw=1.5, alpha=0.85)
        # warning window
        ax1.axvspan(ts[first_asm], ts[-1], alpha=0.07, color="#d62728")
        warn_days = (ts[-1] - ts[first_asm]) / np.timedelta64(1, "D")
        ax1.text(ts[first_asm] + np.timedelta64(1,"D"), 0.50,
                 f"← first window over the threshold\n  ({warn_days:.0f} days before the end of the test)",
                 fontsize=9.5, color="#d62728", va="top")
    ax2.set_yscale("log")
    ax2.set_ylim(0.05, 1500)
    ax2.set_ylabel("d²  (converted to physical scale, log)")
    ax2.set_xlabel("test time (UTC, 2003)")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax2.grid(alpha=0.25, which="both")

    # x-axis 日付フォーマット
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    fig.autofmt_xdate(rotation=0, ha="center")

    fig.tight_layout()
    out = FIG_DIR / "bearing3_detection_en.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
