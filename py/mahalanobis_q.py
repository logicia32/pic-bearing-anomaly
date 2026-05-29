"""Q-format 固定小数で 2D マハラノビス距離を計算する PIC 等価ロジック.

PIC16F1xxx クラスは 8-bit ALU + 8x8→16-bit 単一サイクル乗算。
RAM とサイクルの予算内で d² = c·dx² − 2b·dx·dy + a·dy² を組むため、以下の
スケーリングを採用する（記事の「shrink」の核）。

設計判断
--------
(1) PIC へ渡す入力は (RMS_x, RMS_y) の "ウィンドウ特徴量" のみ（生波形ではない）。
    1 窓 = 1 ファイル = 20480 点。20 kHz × 1 秒 の RMS は MCU 側で逐次計算するが、
    本スクリプトでは「RMS_x, RMS_y が手元にある」前提から d² の計算に集中する。

(2) RMS_x, RMS_y は ~0.07-0.30 g（健常時）〜 0.8 g（劣化末期）程度の浮動小数で得られる。
    これを 16-bit 符号無し整数 q_x = round(RMS_x * 2^FRAC_X) に丸める。FRAC_X は
    実データで決める。dx = q_x - mu_x_q も 16-bit signed に収まるよう mu_q を選ぶ。

(3) Σ⁻¹ は対称 2x2 で 3 つの定数。スケーリング統一のため
       k_a, k_b, k_c  =  round( {s_yy, s_xy, s_xx} * 2^FRAC_K / det )
    として「除算込みで定数化」。これで d² 計算側は除算ゼロ。

(4) PIC 上の演算
       p1 = dx * dx       (16x16 → 32, ただし dx は 16-bit signed)
       p2 = dx * dy       (16x16 → 32, signed)
       p3 = dy * dy       (16x16 → 32)
       d2_q = k_c*p1 - 2*k_b*p2 + k_a*p3   (符号付き加減算)
    最終的に d2_q を右シフトして物理スケール（FRAC_X * 2 + FRAC_K - shift）に揃える。

(5) しきい値比較は threshold_q との比較のみ。sqrt は不要。

本スクリプトは PIC アセンブリに落とす前段として、ビット幅・丸め・オーバーフローを
すべて Python で実演する。numpy リファレンスと値が一致することが PIC 実装の真値。
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _saturate(x: int, bits: int, signed: bool) -> int:
    if signed:
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    else:
        lo, hi = 0, (1 << bits) - 1
    return max(lo, min(hi, x))


@dataclass
class QModel:
    # 入力スケーリング
    frac_x: int        # q_x = RMS * (1 << frac_x), q_x は 16-bit unsigned
    mu_x_q: int        # 16-bit signed 範囲に dx が入るよう μ を量子化
    mu_y_q: int
    # 係数スケーリング
    frac_k: int        # k = round( Σ⁻¹_ij * 2^frac_k )
    k_a: int           # corresponds to s_xx / det
    k_b: int           # corresponds to s_xy / det
    k_c: int           # corresponds to s_yy / det
    # 最終出力のスケール
    output_shift: int  # d2_q >> output_shift = (近似的に) 物理 d²

    @classmethod
    def from_float_model(cls, mu_x: float, mu_y: float,
                         s_xx: float, s_xy: float, s_yy: float,
                         det: float,
                         frac_x: int = 12,
                         frac_k: int = 14,
                         output_shift: int = 0) -> "QModel":
        mu_x_q = int(round(mu_x * (1 << frac_x)))
        mu_y_q = int(round(mu_y * (1 << frac_x)))
        k_a = int(round(s_xx / det * (1 << frac_k)))
        k_b = int(round(s_xy / det * (1 << frac_k)))
        k_c = int(round(s_yy / det * (1 << frac_k)))
        return cls(frac_x=frac_x, mu_x_q=mu_x_q, mu_y_q=mu_y_q,
                   frac_k=frac_k, k_a=k_a, k_b=k_b, k_c=k_c,
                   output_shift=output_shift)

    def to_dict(self) -> dict:
        return asdict(self)

    def d2_q(self, rms_x: float, rms_y: float) -> int:
        """1 窓ぶんの d² を Q-format 整数で返す（PIC 等価）.
        オーバーフローは検出してエラーにする（PIC 側の幅で動くことを保証）."""
        q_x = int(round(rms_x * (1 << self.frac_x)))
        q_y = int(round(rms_y * (1 << self.frac_x)))
        # PIC では q_x, q_y は 16-bit unsigned。実データの値域チェック
        if not (0 <= q_x < (1 << 16)) or not (0 <= q_y < (1 << 16)):
            raise ValueError(f"q_x/q_y が 16-bit unsigned に入らない: q_x={q_x}, q_y={q_y}")
        # dx, dy は 16-bit signed
        dx = q_x - self.mu_x_q
        dy = q_y - self.mu_y_q
        if not (-(1 << 15) <= dx < (1 << 15)) or not (-(1 << 15) <= dy < (1 << 15)):
            raise ValueError(f"dx/dy が 16-bit signed を逸脱: dx={dx}, dy={dy}")
        # 16x16 -> 32-bit signed
        p1 = dx * dx     # always non-negative
        p2 = dx * dy
        p3 = dy * dy
        # 32-bit signed の積に係数 (16-bit signed) を掛ける = 48-bit signed
        # PIC では 32-bit までを基本とし、必要に応じ係数側を右シフトして 32-bit に収める。
        # ここでは Python 任意精度で 48-bit のまま計算し、最終 d2_q を 32-bit にスケール。
        acc = self.k_c * p1 - 2 * self.k_b * p2 + self.k_a * p3
        # 物理 d^2 = acc / 2^(2*frac_x + frac_k)
        # output_shift で粗くスケールダウン
        d2_q = acc >> self.output_shift
        return d2_q

    def d2_float_equiv(self, d2_q: int) -> float:
        """Q-format の d2_q を浮動小数に戻す."""
        scale = 1 << (2 * self.frac_x + self.frac_k - self.output_shift)
        return d2_q / scale

    def vectorize_d2(self, rms_x: np.ndarray, rms_y: np.ndarray) -> np.ndarray:
        out = np.zeros(len(rms_x), dtype=np.int64)
        for i in range(len(rms_x)):
            out[i] = self.d2_q(float(rms_x[i]), float(rms_y[i]))
        return out


def shrink(s_xx: float, s_xy: float, s_yy: float, eps: float):
    """Σ <- Σ + εI（対角正則化／Tikhonov / shrinkage）.

    軸間相関で Σ が特異に近いとき (det≈0)、Σ⁻¹ の係数が爆発する。
    PIC16F1xxx の 16-bit signed に収めるため、診断的に最小限の ε を足して
    det を持ち上げる。これは "拘束を作る" のではなく、健常軸方向の
    過剰な感度を抑える物理的にも妥当な操作（健常時に強相関＝ほぼ同じ情報を
    冗長に見ているので、その方向の過反応を緩和する）.
    """
    s_xx_r = s_xx + eps
    s_yy_r = s_yy + eps
    s_xy_r = s_xy  # 対角のみ補正
    det_r = s_xx_r * s_yy_r - s_xy_r * s_xy_r
    return s_xx_r, s_xy_r, s_yy_r, det_r


def design_q_model(mu_x: float, mu_y: float, s_xx: float, s_xy: float, s_yy: float,
                   det: float, rms_x_max: float, rms_y_max: float,
                   eps: float = 0.0) -> tuple[QModel, dict]:
    """データの値域から frac_x / frac_k を選ぶ.

    要件:
      - q_x = rms_x * 2^frac_x が 16-bit unsigned (< 65536) に収まる
      - dx = q_x - mu_x_q が 16-bit signed (-32768..32767) に収まる
      - k_{a,b,c} が 16-bit signed に収まる（>0 を想定するが s_xy は負もあり）
    """
    info = {"eps": eps, "regularized": eps > 0.0}
    if eps > 0.0:
        s_xx, s_xy, s_yy, det = shrink(s_xx, s_xy, s_yy, eps)
    info.update(dict(s_xx_used=s_xx, s_xy_used=s_xy, s_yy_used=s_yy, det_used=det))

    # 入力スケール: RMS が ~1.0 g にも備える
    frac_x = 14
    while frac_x > 0:
        q_max = max(rms_x_max, rms_y_max) * (1 << frac_x)
        if q_max < (1 << 15) - 8:  # 16-bit signed に収まる余裕を見る
            break
        frac_x -= 1
    # 係数スケール: |k| < 32767 を満たす最大の frac_k
    inv_max = max(abs(s_xx), abs(s_xy), abs(s_yy)) / det
    frac_k = 14
    while frac_k > 0:
        if inv_max * (1 << frac_k) < (1 << 15) - 8:
            break
        frac_k -= 1
    info["frac_x"] = frac_x
    info["frac_k"] = frac_k
    info["k_a_unscaled"] = s_xx / det
    info["k_b_unscaled"] = s_xy / det
    info["k_c_unscaled"] = s_yy / det
    qm = QModel.from_float_model(mu_x, mu_y, s_xx, s_xy, s_yy, det,
                                 frac_x=frac_x, frac_k=frac_k, output_shift=0)
    return qm, info


def compare_with_reference(rms_x: np.ndarray, rms_y: np.ndarray,
                           model: QModel, d2_ref: np.ndarray) -> dict:
    d2_q = model.vectorize_d2(rms_x, rms_y)
    d2_pic_float = np.array([model.d2_float_equiv(int(v)) for v in d2_q])
    err = d2_pic_float - d2_ref
    rel = err / np.where(np.abs(d2_ref) > 1e-12, d2_ref, 1.0)
    return {
        "max_abs_err": float(np.max(np.abs(err))),
        "max_rel_err": float(np.max(np.abs(rel))),
        "mean_abs_err": float(np.mean(np.abs(err))),
        "n_samples": int(len(rms_x)),
    }


def evaluate_for_eps(rms_x, rms_y, d2_ref, m, eps):
    """eps を変えて Q-format の係数サイズと誤差を測る."""
    qm, info = design_q_model(
        mu_x=m["mu_x"], mu_y=m["mu_y"],
        s_xx=m["s_xx"], s_xy=m["s_xy"], s_yy=m["s_yy"],
        det=m["det"],
        rms_x_max=float(rms_x.max()), rms_y_max=float(rms_y.max()),
        eps=eps,
    )
    k_max = max(abs(qm.k_a), abs(qm.k_b), abs(qm.k_c))
    fits_16 = k_max < (1 << 15)
    cmp_rate = compare_with_reference(rms_x, rms_y, qm, d2_ref) if fits_16 else None
    return qm, info, k_max, fits_16, cmp_rate


def main():
    """numpy リファレンスの結果を読み込んで Q-format の誤差を測る.

    Σ が x/y 強相関で det≈0 のため、ε (Tikhonov 正則化) を変えながら
    16-bit signed に係数が入る最小の ε を探す.
    """
    npz = RESULTS / "bearing3_features.npz"
    js = RESULTS / "bearing3_model.json"
    if not npz.exists() or not js.exists():
        print(f"reference not found: run mahalanobis_ref.py first")
        return
    data = np.load(npz, allow_pickle=True)
    with open(js) as f:
        m_dict = json.load(f)
    rms_x = data["rms_x"]
    rms_y = data["rms_y"]
    d2_ref = data["d2"]

    print(f"loaded {len(rms_x)} windows; ref d^2 in [{d2_ref.min():.3f}, {d2_ref.max():.3f}]")
    print(f"RMS x range: [{rms_x.min():.4f}, {rms_x.max():.4f}]  y: [{rms_y.min():.4f}, {rms_y.max():.4f}]")
    print(f"Σ healthy: s_xx={m_dict['s_xx']:.3e}  s_yy={m_dict['s_yy']:.3e}  s_xy={m_dict['s_xy']:.3e}")
    print(f"           det={m_dict['det']:.3e}, ρ_xy={m_dict['s_xy']/(m_dict['s_xx']*m_dict['s_yy'])**0.5:.4f}")

    print("\n=== ε sweep: 16-bit signed に係数が入るか + 誤差 ===")
    eps_candidates = [0.0, 1e-7, 1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 5e-4]
    results = []
    chosen = None
    for eps in eps_candidates:
        qm, info, k_max, fits_16, cmp = evaluate_for_eps(rms_x, rms_y, d2_ref, m_dict, eps)
        flag = "OK" if fits_16 else "OVF"
        line = f"  eps={eps:8.1e}  frac_k={qm.frac_k:2d}  k_max={k_max:8d}  [{flag}]"
        if cmp:
            line += f"  max_abs_err={cmp['max_abs_err']:8.3f}  max_rel_err={cmp['max_rel_err']*100:6.2f}%"
            if chosen is None and fits_16:
                chosen = (eps, qm, info, cmp)
        results.append(line)
        print(line)

    if chosen is None:
        print("\n!! ε candidates すべてオーバーフロー — 32-bit 係数に切り替えるか range を広げる必要")
        return
    eps, qm, info, cmp = chosen
    print(f"\n=== 採用: ε = {eps:.1e} ===")
    print(f"  mu_x_q={qm.mu_x_q}, mu_y_q={qm.mu_y_q}")
    print(f"  k_a={qm.k_a}, k_b={qm.k_b}, k_c={qm.k_c}")
    print(f"  frac_x={qm.frac_x}, frac_k={qm.frac_k}")
    print(f"  Q vs ref: max_abs_err={cmp['max_abs_err']:.3f}, "
          f"max_rel_err={cmp['max_rel_err']*100:.3f}%, "
          f"mean_abs_err={cmp['mean_abs_err']:.3f}")

    # しきい値設計: ref ベースで anomaly = d2_ref > T を採用、Q 側の d2 が同じ判定を出すか確認
    T_ref = 50.0  # ベースライン σ から十分離れた点
    d2_q = qm.vectorize_d2(rms_x, rms_y)
    d2_pic_float = np.array([qm.d2_float_equiv(int(v)) for v in d2_q])
    anomaly_ref = d2_ref > T_ref
    anomaly_pic = d2_pic_float > T_ref
    agree = (anomaly_ref == anomaly_pic).mean() * 100
    miss = ((anomaly_ref) & (~anomaly_pic)).sum()
    false_pos = ((~anomaly_ref) & (anomaly_pic)).sum()
    print(f"\n  threshold T=50 で anomaly 判定一致率: {agree:.2f}%  (miss={miss}, fp={false_pos})")
    first_ref = np.argmax(anomaly_ref) if anomaly_ref.any() else -1
    first_pic = np.argmax(anomaly_pic) if anomaly_pic.any() else -1
    print(f"  最初に anomaly になった窓: ref={first_ref}, pic={first_pic}  (差={first_pic-first_ref})")

    out = RESULTS / "bearing3_qmodel.json"
    with open(out, "w") as f:
        json.dump({"eps": eps, "qmodel": qm.to_dict(), "info": info, "error": cmp,
                   "threshold_ref": T_ref, "agree_pct": agree,
                   "miss": int(miss), "false_pos": int(false_pos),
                   "first_anomaly_ref": int(first_ref), "first_anomaly_pic": int(first_pic)},
                  f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
