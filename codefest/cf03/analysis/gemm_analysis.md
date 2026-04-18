# GEMM Roofline Analysis — 1024×1024 FP32, NVIDIA T4

## (a) Why the naive kernel is memory-bound

The naive kernel has an arithmetic intensity of **10.0 FLOP/Byte**, well below the T4's
ridge point of **27 FLOP/Byte** (8,100 GFLOP/s ÷ 300 GB/s). Every thread independently
loads a full row of A and a full column of B from DRAM for each output element, with no
coordination between threads in the same block. This causes massive redundant DRAM traffic:
threads sharing the same row of A reload identical data independently rather than
collaborating. The result is that the kernel stalls on memory latency and achieves only
**233.5 GFLOP/s** — 2.9 % of FP32 peak — despite the compute pipelines being largely idle
(SM busy: 21 %, issue slots busy: 18 %).

## (b) How tiling reduces DRAM traffic

Tiling loads a T×T sub-block of A and B into shared memory once, then all T² threads in
the block reuse those values for T multiply-accumulate operations before the tile is
evicted. This amortises one DRAM load over T MACs, so arithmetic intensity scales as
**AI ∝ T**. At T = 8 the measured AI rises to **12.1 FLOP/Byte** (+21 % over naive),
and the L2 hit rate jumps from 81 % to **93.5 %**, confirming that repeated accesses now
hit shared memory or L2 instead of DRAM.

## (c) Did tiling achieve the expected improvement?

No — not meaningfully. Despite the higher AI, performance improved by only **0.8 %**
(235.3 vs 233.5 GFLOP/s). The tile size T = 8 is too small: AI = 12.1 still sits far
below the ridge point, so the kernel remains **memory-bound**. The remaining bottleneck
is two-fold: (1) the 8×8 block uses only 64 threads vs. 256 for the naive kernel,
reducing warp-level parallelism and memory-latency hiding; (2) the L1 hit rate collapsed
from 87 % to 22 %, indicating that the narrow tiles cause poor spatial reuse at the L1
level. Increasing T to 32 would project AI to ~48.5 FLOP/Byte, crossing the ridge and
shifting the bottleneck to compute — the necessary next step.
