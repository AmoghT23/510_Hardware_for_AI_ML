# Benchmark Results — CF09 CLLM
**ECE 410/510 — Codefest 9**
**Project:** Anemia Detection — HybridModel (ResNet18 + AttentionFusion)

---

## Platform Summary

| Platform | Description | Clock | Notes |
|---|---|---|---|
| SW Baseline | Intel Core i5-10210U (4c/8t, boost 4.2 GHz), DDR4-2667 | 1.6 GHz (boost 4.2 GHz) | PyTorch 2.11, MKL backend |
| HW — Single Tile (M4 Rev4) | SAED 14nm, `compute_core_ti` Rev4 — 9 BF16 MACs, 4-cycle FSM | 500 MHz | Cadence Genus 17.14; 1-cycle parallel LOAD replaces 9-cycle SRAM sequential read |
| HW — Design Target | SAED 14nm, 444-tile BF16 systolic array | 500 MHz | Extrapolated from single-tile synthesis × 444 tiles |

---

## Benchmark Results Table

> **Note:** All HW figures are **PROJECTED** from synthesis results (Cadence Genus 17.14, SAED 14nm).
> Single-tile numbers derived from M4 Rev4 FSM cycle count and synthesis clock.
> Design-target numbers extrapolate by scaling to 444 tiles.
> Projected numbers are labeled **(P)** throughout.

| Metric | SW Baseline (measured) | HW Single-Tile — sequential (M) | HW Single-Tile — FSM-only (M) | HW Design Target — 1 TFLOP/s (P) |
|---|---|---|---|---|
| Cycles per tile | — | **6 cycles** (4 FSM + 2 poll overhead) | **4 cycles** (LOAD+MUL+ACC+DONE) | 4 cycles/tile (streaming) |
| Conv2d throughput (GFLOP/s) | **142.7 GFLOP/s** | **1.5 GFLOP/s** | **2.25 GFLOP/s** | **999 GFLOP/s ≈ 1 TFLOP/s** |
| Zero-skip cycle count | — | **4 cycles** (55% of tiles) | 2 cycles saved | — |
| Conv2d kernel time per batch | **2,442 ms** | 232,267 ms | 154,844 ms | **348 ms** |
| Training throughput (samples/sec) | **7.58 samples/sec** | 0.14 samples/sec | 0.21 samples/sec | **20.9 samples/sec** |
| Kernel speedup vs SW baseline | — | 0.010× (99× slower) | 0.016× (63× slower) | **7.0×** |
| System speedup (Amdahl, 74.4%) | — | — | — | **2.76×** |
| Peak RSS memory | **2,165.5 MB** | N/A (host-managed) | N/A (host-managed) | N/A (host-managed) |

> **(M) = MEASURED** from cocotb simulation of `compute_core_ti` Rev4 (TESTS=2 PASS=2 FAIL=0).
> **(P) = PROJECTED** — extrapolated from single-tile measurements × 444 tiles.
> Sequential: one tile driven at a time with polling (testbench overhead).
> FSM-only: pure RTL cycle count (streaming systolic array amortizes overhead).

---

## Projection Assumptions

All projections are from **M4 Rev4 synthesis** (Cadence Genus 17.14, SAED 14nm, 500 MHz, WNS = +2.4 ps).

### Single-Tile Measured Results (M4 Rev4 cocotb simulation)

Simulation: `compute_core_ti` Rev4, Icarus Verilog 11.0, cocotb 2.0.1.
Run: `python sim/run_core_rev4.py` from `project/m4/`. TESTS=2 PASS=2 FAIL=0.

```
Functional tests (5/5 PASS):
  T1  ramp weights × unit IFMs  → 45.0     ✓  6 cycles, 12 ns  (normal path)
  T2  alternating-sign × 2.0    → 2.0      ✓  6 cycles, 12 ns  (normal path)
  T3  max BF16 × unit IFMs      → 1143.0   ✓  6 cycles, 12 ns  (normal path)
  T4  Laplacian × ramp IFMs     → 0.0      ✓  6 cycles, 12 ns  (normal path)
  T5  all-zero IFMs             → 0.0      ✓  4 cycles,  8 ns  (zero-skip)

Throughput test (200 tiles back-to-back):
  Avg cycles per tile : 6.00
  Measured throughput : 1.5000 GFLOP/s
  FSM peak (4 cycles) : 2.2500 GFLOP/s
  Utilization         : 66.7%
```

The 6-cycle measured count (vs 4-cycle FSM) reflects 2 cycles of sequential testbench
overhead: one alignment edge before `start` and one extra poll cycle due to cocotb NBA
timing. In a streaming systolic array (444 tiles, shared weight broadcast), the
between-tile gap is eliminated and the FSM-native 4 cycles is the effective tile rate.

### Single-Tile Derivation (M4 Rev4)

M4 Rev4 redesigns `compute_core_ti` with two architectural changes that eliminate
the M3-TI bottlenecks:

**Fix 1 — 1-cycle parallel LOAD** (eliminates 9-cycle SRAM sequential read):

All 9 weight and IFM values arrive simultaneously on a `weights_packed[143:0]` and
`ifms_packed[143:0]` bus. The LOAD state completes in 1 cycle instead of 9.

**Fix 2 — INT32 block-float accumulator** (replaces FP32 binary tree):

Finds the max exponent across all 9 products, aligns mantissas as integers, sums via
a 4-level integer adder tree, and normalizes once. Equal or better accuracy than the
FP32 chain for all practical NN inputs (same approach as Google TPU v1/v2).

**New FSM: IDLE → LOAD(1) → PIPE_MUL(1) → PIPE_ACC(1) → DONE_ST(1)**

```
FSM cycles per tile : 4  (was 12 in M3-TI Rev3 — 3× reduction)
Time per tile       : 4 cycles / 500 MHz = 8 ns
FLOPs per tile      : 9 MACs × 2 = 18 FLOP
Throughput          : 18 FLOP / 8 ns = 2.25 GFLOP/s

Conv2d time per batch = 348.4 GFLOPs / 2.25 GFLOP/s = 154.8 s = 154,844 ms
Training throughput   = 32 / 154.8 = 0.21 samples/sec
Speedup vs CPU        = 2.25 / 142.7 = 0.016×  (63× slower — proof tile, not array)
```

### Single-Tile + Sparsity Derivation

M4 Rev4 adds `ifm_all_zero` detection: if all 9 IFM values are BF16 zero, the FSM
jumps LOAD → DONE_ST directly, skipping PIPE_MUL and PIPE_ACC (saves 2 cycles per tile).
ResNet-18 post-ReLU activation sparsity is approximately **55%**.

```
Average tile latency = 0.55 × 2 cycles + 0.45 × 4 cycles = 2.9 cycles
Average throughput   = 18 FLOP / 2.9 cycles × 500 MHz = 3.1 GFLOP/s

Conv2d time per batch (with sparsity) = 348.4 / 3.1 = 112.4 s = 112,387 ms
Training throughput (with sparsity)   = 32 / 112.4 = 0.28 samples/sec
```

### Design-Target Derivation (444-tile Systolic Array)

```
Tiles required for 1 TFLOP/s:
  999 GFLOP/s / 2.25 GFLOP/s per tile = 444 tiles  (was 111 tiles based on wrong 9 GFLOP/s)

Peak throughput (design target):
  444 tiles × 2.25 GFLOP/s/tile = 999 GFLOP/s ≈ 1 TFLOP/s

With sparsity (55%):
  444 tiles × 3.1 GFLOP/s/tile = 1,376 GFLOP/s ≈ 1.38 TFLOP/s

Conv2d time per batch:
  348.4 GFLOPs / 999 GFLOP/s = 0.348 s = 348 ms per batch

Kernel speedup:
  2,442 ms / 348 ms = 7.0×   (matches M1/cf02 roofline target)

Interface bandwidth check (BF16, AI = 93.8 FLOP/byte):
  Required BW = 999 GFLOP/s / 93.8 = 10.7 GB/s < 32 GB/s AXI → not bottleneck

System speedup (Amdahl, Conv2d fraction = 74.4%):
  1 / (0.256 + 0.744/7) = 2.76×

Training throughput: 7.58 × 2.76 = 20.9 samples/sec
```

---

## Speedup Analysis

| Comparison | Kernel Speedup | Notes |
|---|---|---|
| Single-tile (M4 Rev4) vs SW baseline | **0.016×** (63× slower) | Proof tile — 4-cycle FSM, 2.25 GFLOP/s |
| Single-tile + 55% sparsity vs SW baseline | **0.022×** (46× slower) | Sparsity shortcut, 3.1 GFLOP/s |
| Design target (1 TFLOP/s, 444 tiles) vs SW baseline | **7.0×** | Matches M1 roofline projection |
| System speedup (444 tiles, Amdahl) | **2.76×** | 74.4% accelerated fraction |

The single-tile is slower than the CPU because 1 tile has 9 MACs while the i5-10210U
deploys ~32–64 effective SIMD MAC lanes via AVX2+MKL. The 7× system speedup requires
scaling to 444 tiles — the primary M4 remaining task (see `project/remaining_tasks.md`).

**Note on tile count correction:** The previous CF09 estimate of 111 tiles was based on
an incorrect theoretical peak of 9 GFLOP/s per tile (assumed 1 MAC/cycle). M4 Rev4
measures 2.25 GFLOP/s per tile from the 4-cycle FSM, requiring 444 tiles for 1 TFLOP/s.
The 1 TFLOP/s target and 7× speedup are unchanged.

---

## Energy Efficiency

From `project/m3_ti/cadence_syn/power.rpt` (TT corner, 0.8V, 25°C):

| Module | Dynamic Power | Leakage |
|---|---|---|
| top_ti | 3.064 mW | 2.625 µW |
| compute_core_ti | 1.202 mW | 1.168 µW |
| conv_interface_ti | 1.732 mW | 1.456 µW |
| grad_core | 1.069 mW | 0.959 µW |

**Per-FLOP energy (single tile):**
```
Energy/FLOP = Power / Throughput = 3.064 mW / 2.25 GFLOP/s = 1.36 pJ/FLOP
CPU:          ~15 W / 142.7 GFLOP/s = 105 pJ/FLOP
Efficiency gain: 105 / 1.36 = 77× more energy efficient per FLOP
```

**Design-target energy (444-tile array, linear power scaling):**
```
Power  = 444 × 3.064 mW = 1.36 W
SW baseline energy per batch: 15 W × 2.442 s = 36.6 J
HW design target energy:       1.36 W × 0.348 s = 0.473 J
Energy reduction: 36.6 / 0.473 = 77× per batch
```
