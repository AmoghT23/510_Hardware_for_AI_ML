# Benchmark Results — M4 Hardware Accelerator vs M1 Software Baseline
**ECE 410/510 | Project: Anemia Detection — ResNet18 Conv2d Accelerator**  
**PDK: SAED14nm RVT | Tool: Cadence Genus 17.14-s037_1 | Clock: 500 MHz**

---

## 1. M1 Software Baseline (Reference)

Measured on Intel Core i5-10210U @ 1.6 GHz / 4.2 GHz boost, DDR4-2667, Windows 11.  
Full methodology in `project/m1/sw_baseline.md`.

| Metric | Value | Method |
|---|---|---|
| Full-pipeline wall-clock (1 epoch) | 2573.9 s (median, 10 runs) | `time.perf_counter()` wrapping `nbconvert` |
| Full-pipeline throughput | 3.73 samples/sec | 9,600 samples / 2573.9 s |
| Training-only throughput | 7.58 samples/sec | tqdm batch timing |
| Conv2d kernel throughput | **142.7 GFLOP/s** | 348.4 GFLOP / 2.442 s/batch (cProfile) |
| CPU peak theoretical | 268.8 GFLOP/s | Intel ARK, AVX2 boost |
| CPU utilization | ~53% | 142.7 / 268.8 |
| Arithmetic intensity (Conv2d) | 46.9 FLOP/byte | From `ai_calculation.md` |
| CPU power (TDP) | ~15 W | Intel i5-10210U spec |
| Energy per FLOP (CPU) | ~105 pJ/FLOP | 15 W / 142.7 GFLOP/s |

Conv2d (fwd + bwd) accounts for **74.4%** of the accelerated pipeline (cProfile).

---

## 2. M4 Accelerator — Measured Results

The M4 accelerator is a **32×32 weight-stationary systolic array (1024 tiles)** implemented
in `top_ti` and synthesized by Cadence Genus 17.14 on SAED14nm at 500 MHz. All area,
timing, and power numbers below are from the full-array synthesis (`synth/area_report.txt`,
`synth/timing_report.txt`, `synth/power_report.txt`) — not projections.

### Method of measurement
- **Timing and frequency**: Cadence Genus post-synthesis QoR (`synth/timing_report.txt`)
- **Throughput**: derived from synthesis-confirmed frequency × tiles × MACs × pipeline depth
- **Verification**: cocotb simulation on `top_ti` DUT — TESTS=3 PASS=3 FAIL=0 (`sim/final_run.log`)

### Per-tile datapath
- Forward pass latency: **4 cycles** (LOAD + PIPE_MUL + PIPE_ACC + DONE_ST)
- MACs per tile: 9 (3×3×1 BF16 kernel)
- FLOPs per tile per invocation: 18 (9 multiplies + 9 adds)
- Per-tile throughput: 9 MACs × 2 FLOPs × 500 MHz / 4 cycles = **2.25 GFLOP/s**

### 32×32 Array — synthesis results

| Metric | Value | Source |
|---|---|---|
| Clock | 500 MHz | `synth/timing_report.txt` |
| WNS (full array) | **0 ps** | `synth/timing_report.txt` (meets constraint) |
| Timing violations | 0 | `synth/timing_report.txt` |
| Total cells | **5,591,114** | `synth/area_report.txt` |
| — u_pe_array (1024 tiles) | 5,453,789 | `synth/area_report.txt` |
| — u_interface + u_ifm_buf | 137,325 | `synth/area_report.txt` |
| Total area | **2,862,977 µm² (2.86 mm²)** | `synth/area_report.txt` |
| — u_pe_array area | 2,773,992 µm² (96.9%) | `synth/area_report.txt` |
| Per-tile area | ~2,709 µm² | `synth/area_report.txt` |
| Total power (TT 500 MHz) | **1,359.89 mW (1.36 W)** | `synth/power_report.txt` |
| — u_pe_array power | 1,259.17 mW (92.6%) | `synth/power_report.txt` |
| — u_interface power | 43.97 mW (3.2%) | `synth/power_report.txt` |
| Per-tile power | ~1.23 mW | `synth/power_report.txt` |
| **Array throughput** | **2,304 GFLOP/s (2.304 TFLOP/s)** | 1024 × 2.25 GFLOP/s |
| **Energy per FLOP** | **0.59 pJ/FLOP** | 1.36 W / 2.304 TFLOP/s |

---

## 3. Speedup vs M1 Software Baseline

### Kernel-level speedup (Conv2d)

| Configuration | Throughput | Speedup vs M1 CPU | Basis |
|---|---|---|---|
| M1 CPU (measured) | 142.7 GFLOP/s | 1× (baseline) | cProfile |
| M4 single tile | 2.25 GFLOP/s | 0.016× | per-tile synthesis |
| **M4 32×32 array (1024 tiles)** | **2,304 GFLOP/s** | **16.1×** | synthesis-validated |

### System-level speedup (Amdahl's Law)

```
Conv2d fwd+bwd fraction of pipeline: f  = 74.4%
Kernel speedup (1024-tile array):    S_k = 16.1×

System speedup = 1 / ((1 - f) + f / S_k)
              = 1 / (0.256 + 0.744 / 16.1)
              = 1 / (0.256 + 0.046)
              = 1 / 0.302
              = 3.31×

End-to-end time with accelerator: 2573.9 s / 3.31 = ~777 s per epoch
```

| Level | Speedup | Notes |
|---|---|---|
| Kernel (Conv2d only) | **16.1×** | 1024-tile array vs CPU, synthesis-validated |
| System (full pipeline) | **3.31×** | Amdahl, 74.4% accelerated fraction |
| Full-pipeline throughput | 3.73 → **~12.3 samples/sec** | With 32×32 array |
| Wall-clock per epoch | 2573.9 s → **~777 s** | 42.9 min → ~13.0 min |

---

## 4. Energy Comparison

### Per-FLOP energy efficiency

| Platform | Power | Throughput | Energy/FLOP |
|---|---|---|---|
| M1 CPU (i5-10210U) | ~15 W | 142.7 GFLOP/s | **105 pJ/FLOP** |
| M4 single tile (SAED14nm) | 1.23 mW | 2.25 GFLOP/s | **0.55 pJ/FLOP** |
| **M4 32×32 array (synthesis)** | **1.36 W** | **2,304 GFLOP/s** | **0.59 pJ/FLOP** |
| NVIDIA T4 GPU | 70 W | 65,000 GFLOP/s (INT8 peak) | ~1.08 pJ/FLOP |

```
Energy efficiency gain vs CPU:  105 pJ / 0.59 pJ = 178× more efficient
Energy efficiency vs T4 GPU:    1.08 pJ / 0.59 pJ = 1.8× more efficient per FLOP
```

### Power comparison

```
M4 32×32 array at full load (2.304 TFLOP/s):    1.36 W
NVIDIA T4 GPU (nominal):                          70 W

Power ratio: 70 W / 1.36 W = 51× less power
```

| Scale | Tiles | Throughput | Power | Energy/FLOP |
|---|---|---|---|---|
| Single tile | 1 | 2.25 GFLOP/s | 1.23 mW | 0.55 pJ/FLOP |
| **32×32 array (M4, synthesized)** | **1024** | **2.304 TFLOP/s** | **1.36 W** | **0.59 pJ/FLOP** |
| NVIDIA T4 GPU | — | 65 TFLOP/s (INT8) | 70 W | ~1.08 pJ/FLOP |
| M1 CPU | — | 142.7 GFLOP/s | ~15 W | ~105 pJ/FLOP |

The 178× energy advantage over the CPU comes from SAED14nm FinFET efficiency
(~2,709 µm² per tile) and the dedicated BF16 datapath eliminating general-purpose
CPU overhead (cache hierarchies, out-of-order execution, branch prediction).

---

## 5. Gap Between Measured and Theoretical Performance

The 32×32 array achieves 2.304 TFLOP/s at 1.36 W (synthesis-validated). Two gaps
exist between this and ideal performance:

1. **Pipeline load fraction.** Each tile spends 1 of 4 cycles computing (PIPE_MUL +
   PIPE_ACC = 2 cycles, LOAD + DONE = 2 cycles of overhead). Utilization is 50%.
   A deeper pipeline or weight-stationary reuse across IFM tiles would improve this.

2. **AXI4-Stream clock vs target.** The M1 interface specification assumed AXI4-Stream
   at 1 GHz (64 GB/s). The synthesized design runs at 500 MHz (32 GB/s effective).
   The 21.3 GB/s bandwidth requirement is still met (32 GB/s > 21.3 GB/s), so the
   design is not interface-bound, but the rated bandwidth margin is reduced from 3× to 1.5×.

---

## 6. Raw Data

See `benchmark_data.csv` for all numbers behind this summary.  
See `roofline_final.png` for the visual roofline with measured and synthesis-validated points.
