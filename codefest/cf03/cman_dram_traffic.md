# CF03 CMAN — DRAM Traffic Analysis: Naive vs. Tiled Matrix Multiply

**Given:** Two square FP32 matrices of size N×N with N = 32, stored and accessed in row-major order.

## 1. Naive Triple Loop (ijk order)

For computing one output element:

C[i][j] = Σ A[i][k] × B[k][j]

The inner k-loop runs N = 32 times, so for one C[i][j]:
- Each element of B in column j (B[0][j], B[1][j], ... B[31][j]) is accessed **1 time**
- Total B accesses for one output = N = 32
- Total A accesses for one output = N = 32

**Across the full N×N output:**

Number of output elements = N² = 1,024

Each of these needs N accesses to A and N accesses to B, so:

Total A accesses = N² × N = N³ = 32,768

Total B accesses = N² × N = N³ = 32,768

**Total A + B accesses = 2N³ = 65,536**

So each individual element of B ends up being accessed N = 32 times overall (once per row of C that uses its column).

**DRAM traffic (no reuse, FP32 = 4 bytes):**

**Traffic = 2N³ × 4 = 262,144 bytes = 256 KiB**

## 2. Tiled Loop Analysis (T = 8)

With tiling, data loaded from DRAM is reused on-chip. In the idealized tiled model, instead of counting repeated fetches during blocked execution, we count only the **unique matrix data** that must be brought from DRAM — each element is loaded once and reused from on-chip memory for all subsequent multiply-accumulates.

Each matrix has N² = 1,024 elements.

**Total DRAM loads (unique elements):**

A elements loaded = N² = 1,024

B elements loaded = N² = 1,024

Total elements loaded = 2 × N² = 2,048

**DRAM traffic:**

**Traffic = 2N² × 4 = 2,048 × 4 = 8,192 bytes = 8 KiB**

## 3. Ratio of Naive to Tiled DRAM Traffic

Ratio = (2 × N³ × 4) / (2 × N² × 4) = N³ / N² = N

**Plugging in: 262,144 / 8,192 = 32**

**Note:** \
One-sentence explanation: The ratio equals N because the naive loop reads each matrix element from DRAM N times (once per output element that uses it, giving O(N³) traffic), while the ideal tiled version loads each element from DRAM exactly once and reuses it on-chip for all N of its multiply-accumulates, reducing traffic to O(N²).

## 4. Execution Time: Naive vs. Tiled

**Setup:**
- Compute: 10 TFLOPS = 10 × 10¹² FLOPs/s
- DRAM bandwidth: 320 GB/s = 320 × 10⁹ bytes/s
- Machine balance (ridge point): (10 × 10¹²) / (320 × 10⁹) = **31.25 FLOPs/byte**

Below 31.25 → memory-bound. Above → compute-bound.

**Total FLOPs (same for both):**

2N³ = 2 × 32³ = 65,536 FLOPs

Compute-only time = 65,536 / (10 × 10¹²) = **6.55 ns** (the floor if compute-bound)

### Naive Case

Arithmetic intensity = 65,536 / 262,144 = **0.25 FLOPs/byte**

0.25 ≪ 31.25 → **memory-bound**

Memory time = 262,144 / (320 × 10⁹) = **819 ns**

**Execution time ≈ 819 ns. Compute sits idle ~99% of the time.
**
### Tiled Case

Arithmetic intensity = 65,536 / 8,192 = **8 FLOPs/byte**

8 < 31.25 → **still memory-bound**

Memory time = 8,192 / (320 × 10⁹) = **25.6 ns**

**Execution time ≈ 25.6 ns.**

### Summary

| Case | Traffic | AI (FLOPs/B) | Time | Bound |
|------|---------|--------------|------|-------|
| Naive | 256 KiB | 0.25 | 819 ns | Memory |
| Tiled | 8 KiB | 8.0 | 25.6 ns | Memory |
| Compute roof | — | ≥ 31.25 | 6.55 ns | Compute |

Tiled is **32× faster** than naive — matching the traffic reduction factor N, since both are memory-bound. Even with ideal tiling, arithmetic intensity (8) still falls below the machine balance (31.25), so the kernel remains memory-bound at this problem size.
