# ECE-510 Hardware for AI/ML — Amogh Thakur

This repository contains the codefests (weekly challenges), profiling work, and project deliverables for **ECE-510 (Hardware for AI/ML, Spring 2026)**, taught by Prof. Christof Teuscher at Portland State University.

**Languages:** SystemVerilog, Python

---

## M4 Submission — Final Deliverable

> **Milestone 4 (final submission, due June 7 2026)** is in [`project/m4/`](project/m4/).

- **M4 deliverables index:** [`project/m4/README.md`](project/m4/README.md)
- **Design justification report (PDF):** [`project/m4/report/design_justification.pdf`](project/m4/report/design_justification.pdf)
- **Final synthesis results:** [`project/m4/synth/`](project/m4/synth/) — Cadence Genus 17.14-s037_1, SAED14nm RVT, 500 MHz, WNS = 0 ps
- **Benchmark comparison:** [`project/m4/bench/benchmark.md`](project/m4/bench/benchmark.md) — **16.1× kernel speedup, 3.31× system speedup** (Amdahl)

---

## Project — BF16 Conv2d Accelerator for Anemia Detection

**Summary**
Design a hardware co-processor for a Hybrid CNN anemia-detection model, grounded in profiling data, roofline analysis, and a measured software baseline.

**Project Topic**
AI Accelerator for *"Hybrid CNN model for Anemia Detection in Blood Smear Images"*

---

### 1. Module to Accelerate: 2D Convolution (Conv2d)

The accelerator targets the dominant arithmetic kernel of a HybridModel (ResNet18 backbone + AttentionFusion MLP) classifying peripheral blood smear images as healthy or anemic. Profiling on an Intel Core i5-10210U (PyTorch 2.11.0 CPU-only, AneRBC dataset, 9,600 train images) identifies `torch.conv2d` as the dominant kernel, accounting for **23.1% of direct runtime and 57.8% including backpropagation** (292.9 s of 1,266.6 s over 300 batches).

The co-processor maps Conv2d operations onto a **32×32 weight-stationary systolic array** of BF16 MAC tiles. Roofline analysis (AI = 46.9 FLOP/byte, compute-bound at both CPU ridge 5.9 and co-processor ridge 21.8 FLOP/byte) drove the weight-stationary dataflow choice. The M4 1,024-tile array achieves **2.304 TFLOP/s** — **16.1× kernel speedup** — and applying Amdahl's Law over the 74.4% accelerated fraction gives a **3.31× end-to-end system speedup** (wall-clock: 2,573.9 s → ~777 s per epoch). AttentionFusion, BatchNorm, pooling, and the Adam optimizer remain on the host CPU.

---

### 2. Precision: BF16 with INT32 Block-Float Accumulator

The precision choice evolved across milestones as the design matured:

| Milestone | Format | Rationale |
|---|---|---|
| M1 (baseline) | Q16.16 fixed-point | Software baseline; silent overflow risk, 2× bandwidth vs FP16 |
| M2 (first RTL) | FP16 (IEEE 754) | 2× bandwidth gain over FP32, floating exponent |
| M3 / M4 (final) | **BF16** | 8-bit exponent = FP32 dynamic range; no quantization rescaling; TPU/A100 alignment |

The M4 design uses BF16 for weights and activations with a block-float INT32 accumulator per tile:

| Signal | Format | Width |
|---|---|---|
| Weights / activations | BF16 | 16 bits |
| Multiplier output | FP32 (sign/exp/mantissa) | 32 bits |
| MAC accumulator (block-float) | INT32 | 32 bits |
| Final result | FP32 | 32 bits |

BF16 shares the 8-bit exponent field with FP32 (bias=127), so BF16→FP32 conversion requires only zero-extending the mantissa — no exponent rebasing adder on the critical path. BF16 SQNR ≈ 44 dB (above the 40 dB minimum for CNN classification). Worst-case accumulated error across 9 MACs ≈ 5.5×10⁻⁴ — three orders of magnitude below classification output tolerance.

---

### 3. Interface: AXI4-Stream (data) + AXI4-Lite (control)

The chiplet exposes:

- **AXI4-Stream** (512-bit, 500 MHz) for input feature maps and weight tiles — **32 GB/s**.
- **AXI4-Lite** (13-bit address, 500 MHz) for control — start/done handshake and 1,024 FP32 result registers at 0x1000–0x1FFC.

#### Why AXI4-Stream

| Interface | Bandwidth | Required (21.3 GB/s at 1 TFLOP/s) | Verdict |
|---|---|---|---|
| SPI (50 MHz) | ~0.006 GB/s | 21.3 GB/s | No — 3,500× short |
| I²C (1 MHz) | ~0.0001 GB/s | 21.3 GB/s | No |
| AXI4-Lite alone | ~1–4 GB/s | 21.3 GB/s | No — register overhead, no bursts |
| **AXI4-Stream (512-bit, 500 MHz)** | **32 GB/s** | 21.3 GB/s | **Yes — 1.5× margin** |
| PCIe 4.0 x16 | ~32 GB/s | 21.3 GB/s | Over-specified, adds complexity |

#### Required bandwidth from arithmetic intensity

```
Arithmetic Intensity (Conv2d fwd+bwd) = 46.9 FLOP/byte
Required BW at M1 target (1 TFLOP/s) = 1,000 GFLOP/s / 46.9 FLOP/byte = 21.3 GB/s
Available BW (AXI4-Stream, 512-bit, 500 MHz)  = 32 GB/s  →  1.5× margin
```

The 34×34 BF16 IFM scratchpad (`ifm_buffer.sv`, 18,496 bits on-chip) absorbs burst load: the host fills it once per Conv2d invocation via 37 AXI4-Stream beats, then all 1,024 PEs read their 3×3 windows combinationally with no further AXI traffic during compute. The design is **not interface-bound** during the compute phase.

---

### 4. Architecture: 32×32 Weight-Stationary Systolic Array

```
top_ti (rtl/top.sv)
├── conv_interface_ti (rtl/interface_ti.sv)    AXI4-Lite 13-bit / AXI4-Stream 512-bit
│   └── grad_core (rtl/grad_core.sv)           18 gradients (dL/dW + dL/dX) in 1 cycle
├── ifm_buffer (rtl/ifm_buffer.sv)             34×34 BF16 scratchpad; combinational 3×3 extractor
└── pe_array (rtl/pe_array.sv)                 generate loop: 1024 × compute_core_ti
    └── compute_core_ti × 1024 (rtl/compute_core_ti.sv)
        4-state FSM per tile:
        IDLE → LOAD (latch 9 BF16 weight+IFM pairs, 1 cycle)
             → PIPE_MUL (9 × bf16_mul → FP32, 1 cycle)
             → PIPE_ACC (block-float INT32 adder tree → normalize → FP32, 1 cycle)
             → DONE_ST (assert done, hold result, 1 cycle)
        Total: 4 cycles per Conv2d tile  (down from 11 cycles in M3)
```

**Weight-stationary dataflow:** 9 BF16 weights (144 bits) are broadcast to all 1,024 PEs simultaneously via `weights_packed[143:0]`. Each PE receives its own 3×3 IFM window from the on-chip scratchpad. All 1,024 tiles compute in parallel — one output feature map tile per invocation.

---

## Key M4 Results

| Metric | Value |
|---|---|
| Technology | SAED14nm RVT 14 nm FinFET |
| Synthesis tool | Cadence Genus 17.14-s037_1 |
| Clock | 500 MHz (2 ns period) |
| Timing WNS (32×32 array) | **0 ps — MEETS CONSTRAINT**, 0 violations |
| Cell count (32×32 array) | 5,591,114 |
| Die area (32×32 array) | **2,862,977 µm² (2.86 mm²)** |
| Dominant area contributor | pe_array — 96.9% |
| Total power (TT, 0.8 V, 25 °C) | **1,359.89 mW (1.36 W)** |
| Dominant power contributor | pe_array — 92.6% |
| Per-tile area | ~2,709 µm² |
| Per-tile power | ~1.23 mW |
| Throughput (1 tile) | 2.25 GFLOP/s |
| Throughput (32×32 array, 1,024 tiles) | **2,304 GFLOP/s (2.304 TFLOP/s)** |
| M1 CPU baseline (Conv2d) | 142.7 GFLOP/s |
| **Kernel speedup vs CPU** | **16.1×** |
| **System speedup (Amdahl, f=74.4%)** | **3.31×** |
| Energy per FLOP (array) | **0.59 pJ/FLOP** |
| Energy per FLOP (CPU) | 105 pJ/FLOP |
| **Energy efficiency gain** | **178× vs CPU** |
| Power vs NVIDIA T4 GPU (70 W) | **51× less power** |
| Simulation | TESTS=3 PASS=3 FAIL=0 at 342,140 ns |

---

## Milestone Journey: M1 → M4

| Milestone | Deliverable | Key Result |
|---|---|---|
| **M1** | Software baseline + roofline analysis | Conv2d = 74.4% of pipeline; AI = 46.9 FLOP/byte; 142.7 GFLOP/s CPU; AXI4-Stream selected |
| **M2** | FP16 MAC core + AXI4-Stream RTL | 2/2 unit tests + 4/4 integration tests PASS (cocotb + Icarus) |
| **M3** | BF16 fwd+bwd single tile; synthesis pivot | TESTS=3 PASS=3; OpenLane/sky130A failed (130 ns best vs 2 ns target); pivoted to Cadence SAED14nm |
| **M4** | 32×32 systolic array; full synthesis + benchmark | WNS=0 ps @ 500 MHz; 2.86 mm²; 1.36 W; **16.1× kernel / 3.31× system speedup** |

### Design evolution highlights

- **Precision:** Q16.16 (M1) → FP16 RTL (M2) → **BF16 final (M3/M4)**. BF16 eliminates the exponent rebasing adder on the critical path and matches PyTorch's native training dtype.
- **Scale:** Single-tile proof (M2/M3) → **32×32 = 1,024-tile array (M4)**. `pe_array.sv` generate-loop scales from 1 to any N without RTL changes.
- **Synthesis flow:** OpenLane 2 / sky130A (M3, failed timing) → **Cadence Genus 17.14-s037_1 / SAED14nm RVT (M4, WNS = 0 ps at 500 MHz)**.
- **AXI4-Lite address:** 6-bit (M3) → **13-bit (M4)** to cover 1,024 result registers at 0x1000–0x1FFC.
- **FSM latency per tile:** 11 cycles (M3) → **4 cycles (M4)** via parallel LOAD and pipelined MUL/ACC stages.

---

## Repository Structure

```
510_Hardware_for_AI_ML/
├── README.md                ← you are here; top-level pointer to M4
├── codefest/
│   ├── cf01/ – cf09/        Weekly challenge notebooks
└── project/
    ├── heilmeier.md         Project motivation and goals (Heilmeier questions)
    ├── m1/                  Software baseline: profiling, roofline, interface selection
    ├── m2/                  FP16 MAC core + AXI4-Stream RTL, module tests
    ├── m3/                  BF16 tile + backward pass; OpenLane/sky130A attempt
    ├── m3_ti/               BF16 training+inference co-processor; Cadence SAED14nm single-tile synthesis
    └── m4/                  ← Final submission (M4)
        ├── README.md        File catalog — every file with checklist cross-references
        ├── rtl/             32×32 systolic array RTL (top.sv, pe_array.sv, compute_core_ti.sv, …)
        ├── tb/              Testbenches (tb_top.sv + cocotb Python suites)
        ├── sim/             Simulation outputs (final_run.log PASS, final_waveform.png)
        ├── synth/           Cadence Genus synthesis results (timing, area, power)
        ├── bench/           Benchmark vs M1 baseline (benchmark.md, CSV, roofline plot)
        └── report/          Design justification report (PDF + figures)
```
