# CF03 CMAN — DRAM Traffic Analysis: Naive vs. Tiled Matrix Multiply

**Given:** Two square FP32 matrices of size $N \times N$ with $N = 32$, stored and accessed in row-major order.

---

## 1. Naive Triple Loop (ijk order)

**Number of output elements in C:**

$$N^2 = 32^2 = 1{,}024$$

**Accesses per output element** $C[i][j] = \sum_k A[i][k] \times B[k][j]$: the inner loop runs $N = 32$ times, so each output element requires $N$ accesses to $A$ and $N$ accesses to $B$. Each element of $B$ (and $A$) is touched **once per output element**.

**Total accesses across the full computation:**

$$\text{A accesses} = N^2 \times N = N^3 = 32{,}768$$

$$\text{B accesses} = N^2 \times N = N^3 = 32{,}768$$

$$\text{Total A + B accesses} = 2N^3 = 65{,}536$$

**DRAM traffic (no reuse, FP32 = 4 bytes per access):**

$$\text{Traffic} = 2N^3 \times 4 = 65{,}536 \times 4 = 262{,}144 \text{ bytes} = 256 \text{ KiB}$$

---

## 2. Tiled Loop Analysis (T = 8)

With $N = 32$ and $T = 8$, each matrix is divided into $(N/T)^2 = 4 \times 4 = 16$ tiles, where each tile holds $T^2 = 64$ elements ($64 \times 4 = 256$ bytes).

**Tile loads per output tile:** To compute one $T \times T$ tile of $C$, we iterate through $N/T = 4$ pairs of $A$ and $B$ tiles.

**Total tile loads across all output tiles:**

$$\text{A tile loads} = (N/T)^2 \times (N/T) = (N/T)^3 = 4^3 = 64 \text{ tiles}$$

$$\text{B tile loads} = (N/T)^3 = 64 \text{ tiles}$$

**DRAM traffic:**

$$\text{Traffic per tile} = T^2 \times 4 = 64 \times 4 = 256 \text{ bytes}$$

$$\text{Total traffic} = 2 \times 64 \times 256 = 32{,}768 \text{ bytes} = 32 \text{ KiB}$$

---

## 3. Ratio of Naive to Tiled DRAM Traffic

$$\frac{\text{Naive}}{\text{Tiled}} = \frac{2N^3 \times 4}{2(N/T)^3 \times T^2 \times 4} = \frac{N^3}{(N/T)^3 \cdot T^2} = \frac{N^3 \cdot T^3}{N^3 \cdot T^2} = T$$

For $N = 32$ and $T = 8$:

$$\frac{262{,}144}{32{,}768} = 8 = T$$

**Conclusion:** The ratio equals the tile size $T$, not the matrix dimension $N$. Each element of $A$ or $B$, once loaded into a tile, is reused exactly $T$ times before eviction, so tiling amortizes a single DRAM load over $T$ multiply-accumulate operations.

**Note:** The ratio would equal $N$ only in the case of **perfect reuse** (ideal cache large enough to hold all of $A$, $B$, and $C$ simultaneously), where each matrix is loaded exactly once giving $3N^2$ traffic. Then the ratio becomes $2N^3 / 2N^2 = N$. This corresponds to the extreme case $T = N$.

---

## 4. Execution Time Analysis: Naive vs. Tiled

### Setup

- **Compute throughput:** $10 \text{ TFLOPS} = 10 \times 10^{12}$ FLOPs/s
- **DRAM bandwidth:** $320 \text{ GB/s} = 320 \times 10^9$ bytes/s
- **Machine balance (ridge point):**

$$\frac{10 \times 10^{12}}{320 \times 10^9} = 31.25 \text{ FLOPs/byte}$$

A kernel with arithmetic intensity below 31.25 FLOPs/byte is memory-bound; above it, compute-bound.

### Total Work (Same for Both Cases)

Matrix multiply performs $2N^3$ FLOPs (one multiply + one add per inner-loop iteration):

$$2 \times 32^3 = 65{,}536 \text{ FLOPs}$$

**Compute-only time (lower bound):**

$$\frac{65{,}536}{10 \times 10^{12}} = 6.55 \text{ ns}$$

### Naive Case

- **DRAM traffic (A + B reads):** 262,144 bytes
- **Arithmetic intensity:**

$$\frac{65{,}536}{262{,}144} = 0.25 \text{ FLOPs/byte}$$

Since $0.25 \ll 31.25$, this is **memory-bound**.

- **Memory time:**

$$\frac{262{,}144}{320 \times 10^9} = 819 \text{ ns}$$

**Execution time ≈ 819 ns** (memory-bound; compute sits idle ~99% of the time).

### Tiled Case (T = 8)

- **DRAM traffic (A + B reads):** 32,768 bytes
- **Arithmetic intensity:**

$$\frac{65{,}536}{32{,}768} = 2 \text{ FLOPs/byte}$$

Since $2 < 31.25$, this is **still memory-bound**.

- **Memory time:**

$$\frac{32{,}768}{320 \times 10^9} = 102 \text{ ns}$$

**Execution time ≈ 102 ns**.

### Summary Table

| Case | Traffic (bytes) | AI (FLOPs/byte) | Memory time | Bottleneck |
|------|-----------------|-----------------|-------------|------------|
| Naive | 262,144 | 0.25 | 819 ns | Memory |
| Tiled ($T = 8$) | 32,768 | 2.0 | 102 ns | Memory |
| Compute-bound limit | — | ≥ 31.25 | 6.55 ns | Compute |

**Conclusion:** The tiled version is approximately **8× faster** than naive — exactly matching the traffic reduction factor $T$, since both versions are memory-bound. To become compute-bound, the arithmetic intensity must exceed 31.25 FLOPs/byte, which would require $T \geq 32$ (i.e., tiling the entire matrix as a single block).
