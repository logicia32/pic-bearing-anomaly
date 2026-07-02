# pic-bearing-anomaly — 2D Mahalanobis-distance bearing anomaly detection on a dollar 8-bit PIC

**English** | [日本語](#日本語)

> An attempt to run **2D Mahalanobis-distance** anomaly detection against NASA IMS Bearing run-to-failure data on a **PIC16F1503** (14-pin DIP / 8-bit enhanced mid-range / 128 B SRAM / 2K words flash / **no hardware multiplier**). Not deployed on real silicon — verified for behavior and cycle count on gpsim.

Write-ups:
**English** → https://logicia32.hashnode.dev/mahalanobis-bearing-anomaly-8bit-pic ·
**日本語** → https://zenn.dev/logicia32/articles/2026-05-29-pic-mahalanobis-bearing

> Note: the source-code comments are in Japanese. The code itself is plain Python / PIC assembly, so it reads fine either way; only the prose comments are untranslated.

---

## Structure

```
asm/
  mahal.asm           d² evaluation routine for the PIC16F1503 (for gpasm)
  run_gpsim.stc       gpsim startup script
py/
  extract.py          stage the NASA zip (whose payload is a 7z)
  extract_set1.py     extract 1st_test.rar from inside the 7z
  load_ims.py         scan and read the Set 1 file series
  mahalanobis_ref.py  numpy reference for the 2D Mahalanobis distance
  mahalanobis_q.py    Q-format fixed-point version + ε-regularization search
  diagnose.py         separate the regularization effect from the quantization error
  verify_asm.py       confirm the gpsim output is bit-exact with Python
  sweep_asm.py        run the asm semantics over all 2156 windows, compare against the B reference
  sweep_strategies.py compare the >>16 / rounding / >>12 / 48-bit-acc strategies
  make_figure.py      plot RMS / d² / detection point
results/
  bearing3_*.json     model coefficients / diagnostic summaries
  bearing3_features.npz  Bearing 3 (RMS_x, RMS_y) and d² (2156 windows)
figures/
  bearing3_detection.png
```

## Getting the source data

This uses the **IMS Bearing Data** (provided by the University of Cincinnati), distributed by the [NASA Prognostics Center of Excellence — Bearings](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/). The data itself is **not** included in this repo (for both license and size reasons).

Direct link (NASA PCoE mirror / S3):
- https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip  (~1.0 GB)

The zip's payload is `IMS.7z` (which nests three sets of RAR). `py/extract.py` and `py/extract_set1.py` peel it, in order, from zip → 7z → RAR → `data/1st_test/`. Extracting the RAR needs [unrar](https://www.rarlab.com/).

## Running (rough steps)

```bash
# 1. Build the numeric reference (assuming the NASA data is in place)
python py/extract.py          # zip -> IMS.7z
python py/extract_set1.py     # 7z  -> data/1st_test/
python py/mahalanobis_ref.py  # 2156-window features -> numpy training -> results/

# 2. Work out the Q-format / quantization
python py/mahalanobis_q.py    # epsilon sweep, fit the coefficients into 16-bit
python py/diagnose.py         # separate regularization effect from quantization error
python py/sweep_strategies.py # compare >>16 / rounding / >>12 / 48-bit

# 3. Assemble the PIC asm
gpasm -p 16F1503 -I <header_path> -o asm/mahal.hex asm/mahal.asm

# 4. Verify behavior and cycle count in gpsim
gpsim -p p16f1503 < asm/run_gpsim.stc

# 5. Run the asm semantics over all 2156 windows
python py/sweep_asm.py

# 6. Plot the figure
python py/make_figure.py
```

## Key numbers (measured)

- Per d² evaluation: **2,793 cycles** (measured in gpsim) ≈ 349 μs @ 8 MIPS
- Program Memory: **230 / 2,048 words (11%)**
- SRAM: **~27 / 128 bytes (21%)**
- Anomaly decision over all 2,156 windows: agreement between the PIC equivalent (32-bit acc / >>16) and the regularized reference is **77.46%**, with FP **0**
- Extending the 32-bit acc to 48-bit raises the agreement to **99.72%**, with a detection delay of **0 windows**

## Dependencies

- Python 3.10+ / numpy, scipy, matplotlib, py7zr
- gpasm / gpsim (verified with gputils 1.4.0, gpsim 0.32.1)
- unrar (to peel the RAR inside NASA's `IMS.7z`)

## License

[MIT](LICENSE). Copyright K. NISHIMURA.

---

## 日本語

# pic-bearing-anomaly

NASA IMS Bearing run-to-failure データに対して、**2D マハラノビス距離**で異常検知する処理を、
**PIC16F1503**（14-pin DIP / 8-bit enhanced mid-range / SRAM 128 B / Flash 2K words /
ハードウェア乗算器なし）の上に乗せる試みです。実機には搭載していません — gpsim 上で
動作と cycle 数を確認するところまで。記事本体は

> 「ρ=0.998 の壁 — 100円の 8-bit マイコンに 2D マハラノビス距離を乗せて、ベアリング劣化を聴く」
> https://zenn.dev/logicia32/articles/2026-05-29-pic-mahalanobis-bearing

## 構成

```
asm/
  mahal.asm          PIC16F1503 用 d² 評価ルーチン (gpasm 用)
  run_gpsim.stc      gpsim 起動スクリプト
py/
  extract.py         NASA zip (中身は 7z) のステージング
  extract_set1.py    7z 内 1st_test.rar を展開
  load_ims.py        Set 1 ファイル列を走査・読み込み
  mahalanobis_ref.py 2D マハラノビス距離の numpy リファレンス
  mahalanobis_q.py   Q-format 固定小数版 + ε 正則化探索
  diagnose.py        正則化効果と量子化誤差の切り分け
  verify_asm.py      gpsim の出力と Python の bit-exact 一致確認
  sweep_asm.py       全 2156 窓に asm semantics を走らせて B 参照と比較
  sweep_strategies.py >>16 / 丸め / >>12 / 48-bit acc の戦略比較
  make_figure.py     RMS / d² / 検出点の図を描く
results/
  bearing3_*.json    モデル係数・診断結果のサマリ
  bearing3_features.npz  Bearing 3 の (RMS_x, RMS_y) と d² (2156 窓)
figures/
  bearing3_detection.png
```

## 元データの入手

[NASA Prognostics Center of Excellence — Bearings](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
で配布されている **IMS Bearing Data**（University of Cincinnati 提供）を使います。
本リポジトリにデータ本体は含まれていません（ライセンス・サイズの両面から）。

直リンク（ホストは NASA PCoE のミラー / S3）:
- https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip  (約 1.0 GB)

zip の中身は `IMS.7z`（3 セットの RAR を内包）です。`py/extract.py` と
`py/extract_set1.py` が、zip → 7z → RAR → `data/1st_test/` までを順に剥がします。
RAR の展開には [unrar](https://www.rarlab.com/) が必要です。

## 動かす（おおまかな手順）

```bash
# 1. 数値リファレンスを作る (NASA データを置いた前提)
python py/extract.py          # zip → IMS.7z
python py/extract_set1.py     # 7z → data/1st_test/
python py/mahalanobis_ref.py  # 2156 窓の特徴量 → numpy 学習 → results/

# 2. Q-format / 量子化を検討する
python py/mahalanobis_q.py    # ε sweep, 16-bit に係数を収める
python py/diagnose.py         # 正則化効果 と 量子化誤差 を切り分け
python py/sweep_strategies.py # >>16 / 丸め / >>12 / 48-bit 比較

# 3. PIC asm を組む
gpasm -p 16F1503 -I <header_path> -o asm/mahal.hex asm/mahal.asm

# 4. gpsim で動作と cycle 数を確認
gpsim -p p16f1503 < asm/run_gpsim.stc

# 5. 全 2156 窓に asm semantics を流す
python py/sweep_asm.py

# 6. 図を描く
python py/make_figure.py
```

## 主要な数字（実測ベース）

- 1 d² 評価あたり: **2,793 サイクル** (gpsim 計測) ≒ 349 μs @ 8 MIPS
- Program Memory: **230 / 2,048 words (11%)**
- SRAM: **約 27 / 128 bytes (21%)**
- 全 2,156 窓で異常判定: PIC 等価 (32-bit acc / >>16) と正則化参照の一致率 **77.46%**, FP **0**
- 32-bit acc を 48-bit に拡張すると一致率 **99.72%**, 検出遅延 **0 窓**

## 依存

- Python 3.10+ / numpy, scipy, matplotlib, py7zr
- gpasm / gpsim (gputils 1.4.0、gpsim 0.32.1 で確認)
- unrar (NASA 配布物 IMS.7z 内の RAR を剥がすため)

## ライセンス

[MIT](LICENSE). 著作権は K. NISHIMURA。
