# CF03 CMAN — DRAM Traffic Analysis: Naive vs. Tiled Matrix Multiply

**Given:** Two square FP32 matrices of size N×N with N = 32, stored and accessed in row-major order.

---

## 1. Naive Triple Loop (ijk order)

For computing one output element:

$$C[i][j] = \sum_k A[i][k] \times B[k][j]$$

The inner k-loop runs N = 32 times, so for one C[i][j]:
- Each element of B in column j (B[0][j], B[1][j], ... B[31][j]) is accessed **1 time**
- Total B accesses for one output = N = 32
- Total A accesses for one output = N = 32

**Across the full N×N output:**

Number of output elements = $N^2$ = 1,024

Each of these needs N accesses to A and N accesses to B, so:

$$\text{Total A accesses} = N^2 \times N = N^3 = 32{,}768$$

$$\text{Total B accesses} = N^2 \times N = N^3 = 32{,}768$$

$$\text{Total A + B accesses} = 2N^3 = 65{,}536$$

So each individual element of B ends up being accessed N = 32 times overall (once per row of C that uses its column).

**DRAM traffic (no reuse, FP32 = 4 bytes):**

$$\text{Traffic} = 2N^3 \times 4 = 262{,}144 \text{ bytes} = 256 \text{ KiB}$$

---

## 2. Tiled Loop Analysis (T = 8)

With N = 32 and T = 8, each matrix splits into $(N/T)^2$ = 4 × 4 = 16 tiles. Each tile holds $T^2$ = 64 elements = 256 bytes.

To compute one T×T tile of C, we iterate through N/T = 4 pairs of A and B tiles (load A-tile, load B-tile, multiply-accumulate into C-tile, repeat).

**Total tile loads across all output tiles:**

$$\text{A tile loads} = (N/T)^2 \times (N/T) = (N/T)^3 = 64 \text{ tiles}$$

$$\text{B tile loads} = (N/T)^3 = 64 \text{ tiles}$$

**DRAM traffic:**

Each tile = 256 bytes, and we load 64 A-tiles + 64 B-tiles:

$$\text{Total traffic} = 2 \times 64 \times 256 = 32{,}768 \text{ bytes} = 32 \text{ KiB}$$

---

## 3. Ratio of Naive to Tiled DRAM Traffic

$$\frac{\text{Naive}}{\text{Tiled}} = \frac{2N^3 \times 4}{2(N/T)^3 \times T^2 \times 4} = \frac{N^3}{(N/T)^3 \cdot T^2} = T$$

Plugging in:

$$\frac{262{,}144}{32{,}768} = 8$$

**Note:** The ratio equals T = 8 (not N = 32) because each element loaded into a tile is reused T times before eviction, amortizing one DRAM load over T multiply-accumulates; it would equal N only in the idealized case where the whole matrix fits in one tile (T=N), giving $$\frac{2 times N^{3}}/{2 \times N{2}}=N$$.

---

## 4. Execution Time: Naive vs. Tiled

**Setup:**
- Compute: 10 TFLOPS = $10 \times 10^{12}$ FLOPs/s
- DRAM bandwidth: 320 GB/s = $320 \times 10^9$ bytes/s
- Machine balance (ridge point): $\frac{10 \times 10^{12}}{320 \times 10^9}$ = **31.25 FLOPs/byte**

Below 31.25 → memory-bound. Above → compute-bound.

**Total FLOPs (same for both):**

$$2N^3 = 2 \times 32^3 = 65{,}536 \text{ FLOPs}$$

Compute-only time = $\frac{65{,}536}{10 \times 10^{12}}$ = **6.55 ns** (the floor if compute-bound)

### Naive Case

Arithmetic intensity = $\frac{65{,}536}{262{,}144}$ = **0.25 FLOPs/byte**

0.25 ≪ 31.25 → **memory-bound**

Memory time = $\frac{262{,}144}{320 \times 10^9}$ = **819 ns**

Execution time ≈ 819 ns. Compute sits idle ~99% of the time.

### Tiled Case (T = 8)

Arithmetic intensity = $\frac{65{,}536}{32{,}768}$ = **2 FLOPs/byte**

2 < 31.25 → **still memory-bound**

Memory time = $\frac{32{,}768}{320 \times 10^9}$ = **102 ns**

Execution time ≈ 102 ns.

### Summary

| Case | Traffic | AI (FLOPs/B) | Time | Bound |
|------|---------|--------------|------|-------|
| Naive | 256 KiB | 0.25 | 819 ns | Memory |
| Tiled (T=8) | 32 KiB | 2.0 | 102 ns | Memory |
| Compute roof | — | ≥ 31.25 | 6.55 ns | Compute |

Tiled is **8× faster** than naive — matching the traffic reduction factor T, since both are memory-bound. To hit the compute roof here, we'd need T ≥ 32, i.e., treat the whole matrix as one tile.
