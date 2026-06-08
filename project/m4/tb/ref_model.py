"""
ref_model.py
ECE 410/510 — Python reference model for 32x32 systolic array
Provides bit-exact BF16 reference computations used by all array testbenches.
"""

import struct


# ── BF16 / FP32 helpers ───────────────────────────────────────────────────────

def float_to_bf16(f: float) -> int:
    b = struct.pack('>f', float(f))
    return (int.from_bytes(b, 'big') >> 16) & 0xFFFF


def bf16_to_float(bf16: int) -> float:
    return struct.unpack('>f', ((bf16 & 0xFFFF) << 16).to_bytes(4, 'big'))[0]


def fp32_bits(f: float) -> int:
    return int.from_bytes(struct.pack('>f', float(f)), 'big')


def fp32_to_float(bits: int) -> float:
    return struct.unpack('>f', bits.to_bytes(4, 'big'))[0]


# ── Single-tile reference (reused from existing tests) ────────────────────────

def ref_dot_bf16(weights: list, ifms: list) -> int:
    """BF16-quantized dot product of two 9-element lists. Returns FP32 bit pattern."""
    acc = 0.0
    for w, x in zip(weights, ifms):
        acc += bf16_to_float(float_to_bf16(w)) * bf16_to_float(float_to_bf16(x))
    return fp32_bits(acc)


# ── 32x32 array reference ─────────────────────────────────────────────────────

def ref_conv2d_32x32(weights: list, ifm_34x34: list) -> list:
    """
    Reference model for the 32x32 systolic array forward pass.

    weights   : 9 floats, row-major 3x3 kernel  [w0..w8]
    ifm_34x34 : 34x34 floats, input activation patch (list of 34 rows, each 34 values)
                Row 0 is top; col 0 is left.

    Returns   : 1024 FP32 bit patterns (int), tile t = row*32 + col,
                where row in [0..31], col in [0..31].
                result[t] = BF16-quantized dot(weights, ifm_34x34[row:row+3][col:col+3])
    """
    results = []
    for t in range(1024):
        row = t // 32
        col = t % 32
        window = [ifm_34x34[row + i][col + j] for i in range(3) for j in range(3)]
        results.append(ref_dot_bf16(weights, window))
    return results


def ref_window(ifm_34x34: list, tile: int) -> list:
    """Return the 9-element 3x3 window for a given tile index (0..1023)."""
    row = tile // 32
    col = tile % 32
    return [ifm_34x34[row + i][col + j] for i in range(3) for j in range(3)]


# ── AXI-Stream packing helpers ────────────────────────────────────────────────

def pack_bf16_beat(values: list, beat_idx: int) -> int:
    """Pack 32 already-BF16 int values from a flat list into one 512-bit integer."""
    word = 0
    for i in range(32):
        idx = beat_idx * 32 + i
        v = values[idx] if idx < len(values) else 0
        word |= v << (i * 16)
    return word


def build_weight_beat(weights: list) -> int:
    """Pack 9 BF16 weights into a 512-bit stream beat (bits [143:0], upper bits 0)."""
    word = 0
    for i, w in enumerate(weights):
        word |= float_to_bf16(w) << (i * 16)
    return word


def build_ifm_tile_beats(ifm_34x34: list) -> list:
    """
    Pack a 34x34 IFM tile into 37 x 512-bit AXI-Stream beats.
    Flat order: row 0 left-to-right, then row 1, ..., row 33.
    Beat k carries values flat[k*32 : k*32+32] (BF16, element i at bits [i*16+15:i*16]).
    Beat 36 carries the remaining 1156 - 36*32 = 4 values; upper bits are zero.
    """
    flat = [float_to_bf16(v) for row in ifm_34x34 for v in row]  # 1156 values
    return [pack_bf16_beat(flat, b) for b in range(37)]


# ── Verification against numpy ────────────────────────────────────────────────

def _verify_against_numpy():
    try:
        import numpy as np
    except ImportError:
        print("numpy not available, skipping cross-check")
        return

    import random
    random.seed(42)

    weights = [float(random.randint(-4, 4)) for _ in range(9)]
    ifm_34x34 = [[float(random.randint(0, 8)) for _ in range(34)] for _ in range(34)]

    ref = ref_conv2d_32x32(weights, ifm_34x34)

    w_bf16 = [bf16_to_float(float_to_bf16(w)) for w in weights]
    x_bf16 = [[bf16_to_float(float_to_bf16(v)) for v in row] for row in ifm_34x34]

    w_np = np.array(w_bf16, dtype=np.float32).reshape(1, 1, 3, 3)
    x_np = np.array(x_bf16, dtype=np.float32).reshape(1, 1, 34, 34)

    out_np = np.zeros((32, 32), dtype=np.float32)
    for r in range(32):
        for c in range(32):
            out_np[r, c] = np.sum(w_np[0, 0] * x_np[0, 0, r:r+3, c:c+3])

    mismatches = 0
    for t in range(1024):
        got = fp32_to_float(ref[t])
        exp = float(out_np[t // 32, t % 32])
        if abs(got - exp) > abs(exp) * 1e-4 + 1e-6:
            print(f"  MISMATCH tile {t}: got={got} expected={exp}")
            mismatches += 1

    if mismatches == 0:
        print(f"ref_conv2d_32x32: all 1024 tiles match numpy (weights={weights[:3]}...)")
    else:
        print(f"ref_conv2d_32x32: {mismatches} mismatches vs numpy")


if __name__ == "__main__":
    _verify_against_numpy()
