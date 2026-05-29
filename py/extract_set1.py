"""IMS.7z から 1st_test.rar を取り出し、unrar で data/1st_test/ に展開."""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path
import py7zr

ROOT = Path(__file__).resolve().parent.parent
INNER_7Z = ROOT / "IMS.7z"
RAR_LOCAL = ROOT / "1st_test.rar"
DATA = ROOT / "data"
UNRAR = Path.home() / ".local/bin/unrar"


def extract_rar_from_7z():
    if RAR_LOCAL.exists():
        print(f"already extracted: {RAR_LOCAL} ({RAR_LOCAL.stat().st_size/1e6:.1f} MB)")
        return
    print(f"extracting 1st_test.rar from {INNER_7Z}")
    t0 = time.time()
    with py7zr.SevenZipFile(INNER_7Z, "r") as z:
        z.extract(path=ROOT, targets=["1st_test.rar"])
    print(f"  done in {time.time()-t0:.1f}s; size = {RAR_LOCAL.stat().st_size/1e6:.1f} MB")


def extract_files_from_rar():
    DATA.mkdir(exist_ok=True)
    print(f"unrar -> {DATA}")
    t0 = time.time()
    r = subprocess.run(
        [str(UNRAR), "x", "-o+", str(RAR_LOCAL), str(DATA) + "/"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("STDOUT:", r.stdout[-2000:])
        print("STDERR:", r.stderr[-2000:])
        sys.exit(r.returncode)
    # 末尾の summary だけ表示
    print(r.stdout.splitlines()[-3:])
    print(f"  done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    extract_rar_from_7z()
    extract_files_from_rar()
    # 簡易確認
    subdirs = sorted(DATA.iterdir())
    print("\ndata/ subdirs:", subdirs)
    for d in subdirs:
        if d.is_dir():
            n = sum(1 for _ in d.iterdir())
            print(f"  {d.name}: {n} entries")
