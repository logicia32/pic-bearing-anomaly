"""mahal.asm の動作を Python で bit-exact に再現して、gpsim 出力と一致するか検証.

asm のアルゴリズム:
  dx = in_x - mu_x_q
  dy = in_y - mu_y_q
  acc = 0
  prod = signed16x16 -> 32: dx*dx          ; full 32-bit
  prod = signed16x16: (prod>>16) * k_c     ; 結果も 32-bit
  acc += prod
  prod = dx*dx 同様で k_a * (dy²>>16)
  acc += prod
  prod = signed16x16: dx*dy
  prod <<= 1     ; *2 (32-bit shift left)
  prod = (prod>>16) * k_b
  acc -= prod
  if acc > T_q (compared as 16-bit unsigned since acc fits)  → anomaly
"""
from __future__ import annotations
import struct


MU_X_Q = 2276
MU_Y_Q = 2320
K_A    = 26188
K_B    = 23062
K_C    = 26081
T_Q    = 11898


def s16(v):
    """16-bit signed truncation."""
    v &= 0xFFFF
    return v if v < 0x8000 else v - 0x10000


def s32(v):
    """32-bit signed truncation."""
    v &= 0xFFFFFFFF
    return v if v < 0x80000000 else v - 0x100000000


def mul16x16s(a, b):
    """signed 16 * signed 16 -> signed 32 (truncated)."""
    return s32(a * b)


def evaluate(in_x, in_y):
    dx = s16(in_x - MU_X_Q)
    dy = s16(in_y - MU_Y_Q)
    print(f"dx = {dx}  (0x{dx & 0xFFFF:04X})")
    print(f"dy = {dy}  (0x{dy & 0xFFFF:04X})")

    acc = 0

    # c * dx²
    prod = mul16x16s(dx, dx)
    print(f"dx² = {prod}  (0x{prod & 0xFFFFFFFF:08X})")
    hi = (prod >> 16) & 0xFFFF
    hi_s = hi if hi < 0x8000 else hi - 0x10000   # signed 16
    print(f"  (dx²)>>16 = {hi_s}  (0x{hi_s & 0xFFFF:04X})")
    prod = mul16x16s(K_C, hi_s)
    print(f"  k_c * ((dx²)>>16) = {prod}  (0x{prod & 0xFFFFFFFF:08X})")
    acc = s32(acc + prod)
    print(f"  acc after += : {acc}  (0x{acc & 0xFFFFFFFF:08X})")

    # a * dy²
    prod = mul16x16s(dy, dy)
    print(f"dy² = {prod}")
    hi = (prod >> 16) & 0xFFFF
    hi_s = hi if hi < 0x8000 else hi - 0x10000
    prod = mul16x16s(K_A, hi_s)
    acc = s32(acc + prod)
    print(f"  acc after += k_a*(dy²>>16) : {acc}  (0x{acc & 0xFFFFFFFF:08X})")

    # 2*b * dx * dy
    prod = mul16x16s(dx, dy)
    prod = s32(prod * 2)    # left shift 1 in 32-bit
    print(f"2*dx*dy = {prod}  (0x{prod & 0xFFFFFFFF:08X})")
    hi = (prod >> 16) & 0xFFFF
    hi_s = hi if hi < 0x8000 else hi - 0x10000
    print(f"  (2*dx*dy)>>16 = {hi_s}")
    prod = mul16x16s(K_B, hi_s)
    acc = s32(acc - prod)
    print(f"  acc after -= k_b*((2dxdy)>>16) : {acc}  (0x{acc & 0xFFFFFFFF:08X})")

    # 閾値
    is_anomaly = acc >= T_Q
    print(f"\nfinal acc = {acc}  (0x{acc & 0xFFFFFFFF:08X})")
    print(f"T_q = {T_Q}")
    print(f"result = {'anomaly' if is_anomaly else 'healthy'}")
    return acc, is_anomaly


if __name__ == "__main__":
    # gpsim と同じ入力
    # in_x = 0x25FF = 9727
    # in_y = 0x1FBF = 8127
    acc, anom = evaluate(9727, 8127)
    print()
    print(f"gpsim では acc = 0x004DF697 = {0x004DF697}")
    print(f"Python  では acc = 0x{acc & 0xFFFFFFFF:08X} = {acc}")
    print(f"一致: {acc == 0x004DF697}")
