# CMAN — Arithmetic Intensity Analysis
**ECE 410/510 — Codefest 9**
**Project:** Anemia Detection — HybridModel (ResNet18 + AttentionFusion)
**Target platform:** SAED 14nm RVT, Cadence Genus 17.14 (`compute_core_ti`)
**Note on data type:** The hardware transfers **BF16 (2 bytes/element)** across the DRAM interface.
All byte counts in Tasks 3–5 use BF16. The M1 software baseline used FP32 (4 bytes);
its measured AI (46.9 FLOP/byte) is plotted separately on the sketch at the FP32 operating point.

---

## Task 1 — Dominant Kernel

The dominant kernel is **`torch.conv2d`** — the convolutional forward and backward passes of
the ResNet18 backbone in HybridModel. From cProfile (cf01/cf02), `torch.conv2d` accounts
for 23.1% of total runtime directly and drives 57.8% through backpropagation.

The hardware implementation is the **BF16 9-element dot-product array** in `compute_core_ti`
(M3-TI design), which computes one 3×3×1 convolutional tile per invocation.

| Field | Value |
|---|---|
| Kernel type | Conv2d (GEMM-style, im2col + matrix multiply) |
| Hardware module | `compute_core_ti` — BF16 dot-product MAC array |
| Kernel size | 3×3×1 (9 elements per tile) |
| Layer count | 20 Conv2d layers (C_in: 3→512, C_out: 64→512) |
| Input spatial | 224 × 224 × 3 RGB per image |
| Batch size | 32 (operating point) |
| Weight precision | BF16 (16-bit brain float) |
| Activation precision | BF16 (16-bit) |
| Accumulator / output | FP32 (32-bit IEEE 754) |

---

## Task 2 — FLOPs

**Formula per Conv2d layer (one invocation):**

```
FLOPs = 2 × C_in × K × K × C_out × H_out × W_out
```

Factor of 2: each MAC (multiply-accumulate) = 1 multiply + 1 add = 2 FLOPs.

**Total Mult-Adds for full ResNet18 backbone (torchinfo, 1 image, forward):**

| Layer group | Representative layer | Mult-Adds |
|---|---|---|
| Conv1 (stem) | 3 → 64, K=7, 112×112 | 118,013,952 |
| Layer1 (4 convs) | 64 → 64, K=3, 56×56 | 4 × 115,605,504 |
| Layer2 (4 convs + downsample) | 64/128 → 128, K=3/1 | ~295,436,288 |
| Layer3 (4 convs + downsample) | 128/256 → 256, K=3/1 | ~295,436,288 |
| Layer4 (4 convs + downsample) | 256/512 → 512, K=3/1 | ~295,436,288 |
| **Total** | | **1,814,511,616 MACs** |

**torchinfo confirmed: Total Mult-Adds = 1.81 GMACs**

**FLOPs per image, forward pass:**
```
FLOPs_fwd = 2 × 1,814,511,616 = 3,629,023,232 FLOPs ≈ 3.63 GFLOPs
```

**FLOPs per image, forward + backward (backprop ≈ 2× forward):**
```
FLOPs_fwd_bwd = 3 × FLOPs_fwd = 3 × 3,629,023,232 = 10,887,069,696 FLOPs ≈ 10.89 GFLOPs
```

**FLOPs per batch (batch = 32, forward + backward):**
```
FLOPs_batch = 32 × 10,887,069,696 = 348,386,230,272 FLOPs ≈ 348.4 GFLOPs = 0.348 TFLOPs
```

---

## Task 3 — Bytes Transferred

**Data type:** **BF16** (16-bit brain float, **2 bytes** per element). Weights and activations
are stored and transferred in BF16 at the DRAM interface; the FP32 accumulator stays
in on-chip MAC registers and does not transit the memory bus.

**Byte formula per Conv2d layer:**
```
Bytes = (C_out × C_in × K × K × 2)        ← weights (BF16)
      + (C_in  × H_in × W_in × 2)          ← input activations (BF16)
      + (C_out × H_out × W_out × 2)         ← output activations (BF16)
```

**Per image, forward + backward (no reuse), from torchinfo element counts × 2 B:**
```
Bytes_fwd_bwd = Weights × 3 + Acts_in × 2 + Acts_out × 2

Weights (20 Conv2d layers)   = 22,824,448 B  (22.8 MB)   [11,412,224 elements × 2 B]
Input activations (all)      = 11,993,344 B  (12.0 MB)   [ 5,996,672 elements × 2 B]
Output activations (all)     = 11,796,480 B  (11.8 MB)   [ 5,898,240 elements × 2 B]

× 3 weights (fwd read + bwd gradient read + gradient write)
× 2 activations (fwd + bwd)

= 3 × 22,824,448 + 2 × 11,993,344 + 2 × 11,796,480
= 68,473,344 + 23,986,688 + 23,592,960
= 116,052,992 B per image
```

---

### Lower Bound — No Data Reuse

All weights and activations reload from DRAM for every operation.

```
Bytes_no_reuse = 32 × 116,052,992 B
               = 3,713,695,744 B
               ≈ 3.71 GB per batch
```

---

### Upper Bound — Perfect On-Chip Weight Reuse (GEMM-style)

**Reuse pattern: GEMM-style weight reuse** — weight matrices are loaded once into on-chip
SRAM at the start of each batch and held resident. Only activation tensors stream
through the memory interface for each image. This is exactly the dataflow implemented
in `sram_sp` (persistent weight SRAM in `conv_interface_ti`): weights are loaded once
via AXI4-Stream and reused across all IFM tiles without reloading.

```
Bytes_weight_reuse = N_images × (Acts_in × 2 + Acts_out × 2)

Acts_in × 2  per image = 2 × 11,993,344 = 23,986,688 B  (fwd + bwd, BF16)
Acts_out × 2 per image = 2 × 11,796,480 = 23,592,960 B  (fwd + bwd, BF16)
Activation bytes/image = 47,579,648 B  (47.6 MB)

Bytes_weight_reuse = 32 × 47,579,648 B
                   = 1,522,548,736 B
                   ≈ 1.52 GB per batch
```

---

## Task 4 — Arithmetic Intensity

```
AI = FLOPs / Bytes
```

| Bound | FLOPs (batch) | Bytes (batch, BF16) | AI (FLOP/byte) |
|---|---|---|---|
| **Lower (no reuse)** | 348,386,230,272 | 3,713,695,744 | **93.8** |
| **Upper (full weight reuse)** | 348,386,230,272 | 1,522,548,736 | **228.8** |

---

### Roofline — Target Platform: SAED 14nm RVT (Cadence Genus 17.14)

From M3-TI synthesis results (`cadence_syn/qor.rpt`, `area.rpt`, `power.rpt`):

| Spec | Value | Derivation |
|---|---|---|
| Clock | 2 ns | CLOCK_PERIOD = 2,000 ps, WNS = +2 ps |
| Frequency | **500 MHz** | 1 / 2 ns |
| Peak compute | **9 GFLOP/s** | 9 MACs × 2 FLOP/MAC × 500 MHz |
| AXI4-Stream BW | **32 GB/s** | 512-bit bus × 500 MHz |
| **Ridge point** | **0.28 FLOP/byte** | 9 GFLOP/s ÷ 32 GB/s |

**Attainable performance (roofline):**

Both AI bounds (93.8 and 228.8 FLOP/byte) are far above the ridge point (0.28 FLOP/byte),
placing the kernel in the **compute-bound** region on the SAED 14nm accelerator.

```
Attainable at AI_low  = min(9 GFLOP/s, 32 × 93.8)  = min(9, 3002)  = 9 GFLOP/s
Attainable at AI_high = min(9 GFLOP/s, 32 × 228.8) = min(9, 7322)  = 9 GFLOP/s
```

---

### Roofline — M1 Software Baseline: Intel Core i5-10210U

| Spec | Value |
|---|---|
| Peak compute | 268.8 GFLOP/s (AVX2, boost 4.2 GHz, FP32) |
| Peak memory BW | 45.8 GB/s (DDR4-2667, dual-channel) |
| **Ridge point** | **5.9 FLOP/byte** |
| Measured (Conv2d) | 142.7 GFLOP/s at AI = 46.9 (53% of peak) |

Both HW AI bounds (93.8 and 228.8) exceed the CPU ridge (5.9) → also compute-bound on CPU.
The SW measured point at AI = 46.9 FLOP/byte (FP32) also exceeds the CPU ridge, consistent
with the measured throughput of 142.7 GFLOP/s (53% of 268.8 GFLOP/s peak).

**Roofline sketch saved as:** `codefest/cf09/cman_roofline_sketch.png`

*Sketch axes: X = AI (FLOP/byte), log scale 10⁻¹ to 10³; Y = Attainable (GFLOP/s), log scale 10⁻² to 10³.*
*Both platform rooflines drawn; ridge points, compute ceilings, BW slopes, and both BF16 AI bounds marked.*
*SW measured point (FP32, AI = 46.9) plotted separately; HW projected bounds at BF16 AI = 93.8 and 228.8.*

---

## Task 5 — Bottleneck and Highest-Leverage Improvement

**Current bottleneck: Compute units (MAC array)**

Both AI bounds (93.8–228.8 FLOP/byte) are 335–817× above the SAED 14nm ridge point
(0.28 FLOP/byte). The AXI4-Stream interface (32 GB/s) can supply data far faster than
the 9-MAC array can consume it. The MAC array's arithmetic throughput is the binding resource.

```
Required BW to saturate compute = Peak compute / AI  (single tile, M4 Rev4)
  At AI_low:  2.25 GFLOP/s / 93.8  = 0.024 GB/s   (0.075% of 32 GB/s — interface near-idle)
  At AI_high: 2.25 GFLOP/s / 228.8 = 0.010 GB/s   (0.031% of 32 GB/s)
```

**Single highest-leverage improvement:**

Scale `compute_core_ti` from a single 9-MAC tile to a **systolic array of parallel
MAC tiles** sharing a weight-broadcast bus. Per-tile throughput is **2.25 GFLOP/s**
(M4 Rev4, 4-cycle FSM: LOAD + PIPE_MUL + PIPE_ACC + DONE, each 1 cycle). To match
the CPU baseline (142.7 GFLOP/s) at the same 500 MHz clock requires **64 tiles**;
to reach the 1 TFLOP/s design target requires **444 tiles**:

```
64 tiles  × 2.25 GFLOP/s = 144 GFLOP/s   (matches CPU baseline)
444 tiles × 2.25 GFLOP/s = 999 GFLOP/s ≈ 1 TFLOP/s  (7× kernel speedup)
```

With M4 Rev4 activation sparsity (55% post-ReLU zero-skip, avg 2.9 cycles/tile):
```
444 tiles × 3.1 GFLOP/s  = 1,376 GFLOP/s ≈ 1.38 TFLOP/s
```

The 32 GB/s AXI4-Stream bus remains sufficient at all scales — required bandwidth
at AI = 93.8 FLOP/byte is 999 / 93.8 = 10.7 GB/s, well within the 32 GB/s limit.
The ridge point at 444 tiles shifts to 999 / 32 = 31.2 FLOP/byte — still below both
AI bounds (93.8 and 228.8), keeping the design firmly compute-bound.
