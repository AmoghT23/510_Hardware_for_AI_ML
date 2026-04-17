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

Total A + B accesses = 2N³ = 65,536

So each individual element of B ends up being accessed N = 32 times overall (once per row of C that uses its column).

**DRAM traffic (no reuse, FP32 = 4 bytes):**

Traffic = 2N³ × 4 = 262,144 bytes = 256 KiB

## 2. Tiled Loop Analysis (T = 8)

With N = 32 and T = 8, each matrix splits into (N/T)² = 4 × 4 = 16 tiles. Each tile holds T² = 64 elements = 256 bytes.

To compute one T×T tile of C, we iterate through N/T = 4 pairs of A and B tiles (load A-tile, load B-tile, multiply-accumulate into C-tile, repeat).

**Total tile loads across all output tiles:**

A tile loads = (N/T)² × (N/T) = (N/T)³ = 64 tiles

B tile loads = (N/T)³ = 64 tiles

**DRAM traffic:**

Each tile = 256 bytes, and we load 64 A-tiles + 64 B-tiles:

Total traffic = 2 × 64 × 256 = 32,768 bytes = 32 KiB

## 3. Ratio of Naive to Tiled DRAM Traffic

Ratio = (2 × N³ × 4) / (2 × (N/T)³ × T² × 4) = N³ / ((N/T)³ × T²) = T

Plugging in: 262,144 / 32,768 = 8

**Note:** The ratio equals **T = 8** (not N = 32) because each element loaded into a tile is reused T times before eviction, amortizing one DRAM load over T multiply-accumulates; it would equal N only in the idealized case where the whole matrix fits in one tile (T = N), giving 2N³ / 2N² = N.

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

Execution time ≈ 819 ns. Compute sits idle ~99% of the time.

### Tiled Case (T = 8)

Arithmetic intensity = 65,536 / 32,768 = **2 FLOPs/byte**

2 < 31.25 → **still memory-bound**

Memory time = 32,768 / (320 × 10⁹) = **102 ns**

Execution time ≈ 102 ns.

### Summary

| Naive | 256 KiB | 0.25 | 819 ns | Memory |
| Tiled (T=8) | 32 KiB | 2.0 | 102 ns | Memory |
| Compute roof | — | ≥ 31.25 | 6.55 ns | Compute |

Tiled is **8× faster** than naive — matching the traffic reduction factor T, since both are memory-bound. To hit the compute roof here, we'd need T ≥ 32, i.e., treat the whole matrix as one tile.
