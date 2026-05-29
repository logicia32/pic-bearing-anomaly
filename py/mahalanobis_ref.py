"""2D マハラノビス距離の numpy リファレンス実装.

Bearing 3 (Ch 5,6 = x,y) について、各ファイル (20480 サンプル × 1 秒) ごとに
(RMS_x, RMS_y) を 1 点とみなし、run 全体で時系列を作る。
最初の "healthy" 区間から μ と Σ を学習し、全 run に d² を流して劣化過程を見る。

Σ は対称 2x2:
    Σ = [[s_xx, s_xy],
         [s_xy, s_yy]]
det(Σ) = s_xx * s_yy - s_xy^2
Σ^-1 = (1/det) * [[ s_yy, -s_xy],
                  [-s_xy,  s_xx]]
d^2 = (x-μ)^T Σ^-1 (x-μ)
    = (1/det) * ( s_yy * dx^2  -  2*s_xy * dx*dy  +  s_xx * dy^2 )

PIC 側ではこの「3 係数 / det」を事前計算して定数化、d^2 を 5 回の整数乗算 +
加算で計算する → 除算ゼロ、sqrt なし、しきい値比較で完結。
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from load_ims import list_files, load_one, BEARING_CHANNELS, DATA_DIR

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


@dataclass
class MahalanobisModel:
    mu_x: float
    mu_y: float
    s_xx: float       # Σ[0,0] (var x)
    s_xy: float       # Σ[0,1] (cov xy)
    s_yy: float       # Σ[1,1] (var y)
    det: float
    n_train: int      # 学習に使った窓数

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> "MahalanobisModel":
        mu = np.array([x.mean(), y.mean()])
        # bias を取った 2x2 共分散 (np.cov の bias=True と同じ)
        dx = x - mu[0]
        dy = y - mu[1]
        s_xx = float((dx * dx).mean())
        s_yy = float((dy * dy).mean())
        s_xy = float((dx * dy).mean())
        det = s_xx * s_yy - s_xy * s_xy
        return cls(
            mu_x=float(mu[0]), mu_y=float(mu[1]),
            s_xx=s_xx, s_xy=s_xy, s_yy=s_yy, det=det,
            n_train=int(len(x)),
        )

    def d2(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dx = x - self.mu_x
        dy = y - self.mu_y
        return (self.s_yy * dx * dx
                - 2.0 * self.s_xy * dx * dy
                + self.s_xx * dy * dy) / self.det


def compute_features(bearing: int = 3, max_files: int | None = None,
                     progress_every: int = 200) -> tuple[np.ndarray, np.ndarray, list]:
    """Set 1 全 2156 ファイルから Bearing N の (RMS_x, RMS_y) を取る."""
    entries = list_files()
    print(f"set 1 files: {len(entries)}")
    if max_files:
        entries = entries[:max_files]

    cx, cy = BEARING_CHANNELS[bearing]
    rms_x = np.zeros(len(entries))
    rms_y = np.zeros(len(entries))
    timestamps = [e.ts for e in entries]

    t0 = time.time()
    for i, e in enumerate(entries):
        arr = load_one(e.path)
        rms_x[i] = float(np.sqrt(np.mean(arr[:, cx] ** 2)))
        rms_y[i] = float(np.sqrt(np.mean(arr[:, cy] ** 2)))
        if (i + 1) % progress_every == 0 or i + 1 == len(entries):
            dt = time.time() - t0
            print(f"  {i+1}/{len(entries)}  ({dt:.1f}s, {(i+1)/dt:.1f} files/s)")
    return rms_x, rms_y, timestamps


def main(bearing: int = 3, train_frac: float = 0.10, max_files: int | None = None):
    rms_x, rms_y, ts = compute_features(bearing=bearing, max_files=max_files)
    n = len(rms_x)
    n_train = max(20, int(n * train_frac))
    print(f"\nbearing {bearing}: total windows = {n}, train (healthy head) = {n_train}")
    model = MahalanobisModel.fit(rms_x[:n_train], rms_y[:n_train])
    print(f"  mu = ({model.mu_x:.4f}, {model.mu_y:.4f})")
    print(f"  Σ  = [[{model.s_xx:.4e}, {model.s_xy:.4e}],")
    print(f"        [{model.s_xy:.4e}, {model.s_yy:.4e}]]  det = {model.det:.4e}")
    d2 = model.d2(rms_x, rms_y)
    print(f"  d^2 stats: min={d2.min():.3f}  mean={d2.mean():.3f}  max={d2.max():.3f}")
    print(f"  d^2 last 10 windows: {d2[-10:]}")

    # 保存
    out = RESULTS / f"bearing{bearing}_features.npz"
    np.savez(out, rms_x=rms_x, rms_y=rms_y, d2=d2,
             ts=np.array([t.isoformat() for t in ts]))
    print(f"saved {out}")
    with open(RESULTS / f"bearing{bearing}_model.json", "w") as f:
        json.dump(asdict(model), f, indent=2)
    print(f"saved {RESULTS / f'bearing{bearing}_model.json'}")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    bearing = int(args[0]) if args else 3
    max_files = int(args[1]) if len(args) > 1 else None
    main(bearing=bearing, max_files=max_files)
