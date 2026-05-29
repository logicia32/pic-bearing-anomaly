"""IMS Bearing dataset loader (Set 1 = 1st_test).

data/1st_test/ 以下の 2156 ファイルを走査して、Bearing 3 (Ch 5,6) などの
2 軸特徴量を取り出す。各ファイル = 20480 行 × 8 列 ASCII、1 秒 @ 20 kHz。
ファイル名は "2003.10.22.12.06.24" 形式の timestamp。
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "1st_test"

# Set 1 内のチャネル割り当て (1-indexed -> 0-indexed)
BEARING_CHANNELS = {
    1: (0, 1),  # Ch 1, 2
    2: (2, 3),  # Ch 3, 4
    3: (4, 5),  # Ch 5, 6  <-- inner race defect 起きるベアリング
    4: (6, 7),  # Ch 7, 8
}

TS_RE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})$")


@dataclass
class FileEntry:
    path: Path
    ts: datetime


def list_files(data_dir: Path = DATA_DIR) -> list[FileEntry]:
    """data/1st_test/ 配下を走査し、timestamp ソート済みの一覧を返す."""
    out: list[FileEntry] = []
    for p in data_dir.iterdir():
        if not p.is_file():
            continue
        m = TS_RE.match(p.name)
        if not m:
            continue
        ts = datetime(*map(int, m.groups()))
        out.append(FileEntry(path=p, ts=ts))
    out.sort(key=lambda e: e.ts)
    return out


def load_one(path: Path) -> np.ndarray:
    """1 ファイル -> ndarray (20480, 8)."""
    return np.loadtxt(path)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def features_for_file(path: Path, bearing: int) -> tuple[float, float]:
    arr = load_one(path)
    cx, cy = BEARING_CHANNELS[bearing]
    return rms(arr[:, cx]), rms(arr[:, cy])


if __name__ == "__main__":
    entries = list_files()
    print(f"files: {len(entries)}")
    if entries:
        print(f"first: {entries[0].path.name}  {entries[0].ts}")
        print(f"last : {entries[-1].path.name}  {entries[-1].ts}")
        # 1 ファイルだけロードして形状確認
        arr = load_one(entries[0].path)
        print(f"sample shape: {arr.shape}  dtype={arr.dtype}")
        print(f"sample channels min/max:")
        for ch in range(arr.shape[1]):
            print(f"  ch{ch+1}: [{arr[:,ch].min():+.4f}, {arr[:,ch].max():+.4f}]  "
                  f"rms={rms(arr[:,ch]):.4f}")
