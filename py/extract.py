"""ims-bearings.zip -> IMS.7z を覗いて、Set 1 だけ展開する.

7z 内構造:
  IMS.7z
    1st_test/   <- Set 1: 8ch, Bearing3 inner-race / Bearing4 roller defect
    2nd_test/   <- Set 2: 4ch
    3rd_test/   <- Set 3: 4ch
ものによっては階層が違うかもしれないので、まず list する。
"""
from __future__ import annotations
import sys
import zipfile
import time
from pathlib import Path

import py7zr

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "ims-bearings.zip"
DATA_DIR = ROOT / "data"
INNER_7Z = "4. Bearings/IMS.7z"
INNER_LOCAL = ROOT / "IMS.7z"


def stage_inner_7z():
    """zip から IMS.7z をローカルに取り出す（ストリーミング展開）."""
    if INNER_LOCAL.exists():
        print(f"already staged: {INNER_LOCAL} ({INNER_LOCAL.stat().st_size/1e6:.1f} MB)")
        return
    print(f"extracting {INNER_7Z} from {ZIP}")
    with zipfile.ZipFile(ZIP) as zf:
        with zf.open(INNER_7Z) as src, open(INNER_LOCAL, "wb") as dst:
            t0 = time.time()
            n = 0
            while True:
                buf = src.read(1 << 20)
                if not buf:
                    break
                dst.write(buf)
                n += len(buf)
            print(f"wrote {n/1e6:.1f} MB in {time.time()-t0:.1f}s")


def list_7z():
    print(f"listing {INNER_LOCAL}")
    with py7zr.SevenZipFile(INNER_LOCAL, "r") as z:
        infos = z.list()
        # 階層ごとに集計
        from collections import Counter
        counts = Counter()
        for info in infos:
            parts = info.filename.split("/")
            top = parts[0]
            counts[top] += 1
        for top, cnt in sorted(counts.items()):
            print(f"  {top}: {cnt}")
        # 最初の数件
        print("\nsamples:")
        for info in infos[:6]:
            print(f"  {info.filename}  (size={info.uncompressed:,})")
    return infos


def extract_set1(target_top: str = "1st_test"):
    """Set 1 だけ展開."""
    DATA_DIR.mkdir(exist_ok=True)
    with py7zr.SevenZipFile(INNER_LOCAL, "r") as z:
        names = [info.filename for info in z.list()
                 if info.filename.startswith(target_top + "/")]
        print(f"extracting {len(names)} files of {target_top!r} to {DATA_DIR}")
        t0 = time.time()
        z.extract(path=DATA_DIR, targets=names)
        print(f"done in {time.time()-t0:.1f}s")


def main():
    stage_inner_7z()
    infos = list_7z()
    # set 1 の top dir を探す
    top_candidates = sorted({info.filename.split("/")[0]
                             for info in infos
                             if "/" in info.filename})
    print(f"\ntop-level dirs: {top_candidates}")
    # 1st_test / 1st test / Set1 等のバリエーションを許容
    target = None
    for t in top_candidates:
        low = t.lower().replace(" ", "").replace("_", "")
        if "1sttest" in low or "set1" in low or low == "1":
            target = t
            break
    if target is None:
        print("could not auto-detect Set 1 top dir; pass it explicitly.")
        sys.exit(2)
    print(f"target Set 1 top dir = {target!r}")
    if "--list-only" in sys.argv:
        return
    extract_set1(target)


if __name__ == "__main__":
    main()
