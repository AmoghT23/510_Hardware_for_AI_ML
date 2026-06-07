# ECE-510 Hardware for AI/ML — Amogh Thakur

This repository contains the codefests (weekly challenges), profiling work, and project deliverables for **ECE-510 (Hardware for AI/ML, Spring 2026)**, taught by Prof. Christof Teuscher at Portland State University.

**Languages:** SystemVerilog, Verilog, Python

---

## M4 Submission — Final Deliverable

> **Milestone 4 (final submission, due June 7 2026)** is in [`project/m4/`](project/m4/).

- **M4 deliverables index:** [`project/m4/README.md`](project/m4/README.md)
- **Design justification report (PDF):** [`project/m4/report/design_justification.pdf`](project/m4/report/design_justification.pdf)
- **Final synthesis results:** [`project/m4/synth/`](project/m4/synth/) — Cadence Genus 17.14-s037_1, SAED14nm RVT, 500 MHz, WNS = 0 ps
- **Benchmark comparison:** [`project/m4/bench/benchmark.md`](project/m4/bench/benchmark.md) — 3.31× system speedup (Amdahl)

---

## Hybrid CNN Project

**Summary**
Design a hardware co-processor chiplet for a Hybrid CNN, grounded in profiling data, roofline analysis, and a measured software baseline.

**Project Topic**
AI Accelerator for *"Hybrid CNN model for Anemia Detection in Blood Smear Images"*

**Full implementation repository (joint with teammate)**
https://github.com/nehalshivane04/Accelerator---Anemia-Dataset

---

## Project — Anemia Detection Co-processor Chiplet

### 1. Module to accelerate: 2D Convolution (Conv2d)

The accelerator targets the dominant arithmetic kernel of a HybridModel (ResNet18 backbone + AttentionFusion MLP) used to classify peripheral blood smear images as healthy or anemic. Profiling on an Intel Core i5-10210U (PyTorch 2.11.0 CPU build, AneRBC dataset, 9,600 train images) identifies `torch.conv2d` as the dominant kernel, accounting for **23.1 % of direct runtime and 57.8 % including backpropagation** (292.9 s out of 1266.6 s over 300 batches).

The co-processor chiplet maps these Conv2d operations onto a weight-stationary systolic array of multiply-accumulate (MAC) units. Roofline analysis projects a **7× kernel-level speedup** (142.7 GFLOP/s baseline → 1 TFLOP/s target) and, applying Amdahl's Law over the 74.4 % accelerated fraction, a **3.31× end-to-end system speedup**. AttentionFusion, BatchNorm, pooling, handcrafted feature extraction (GLCM, morphology, color), and optimizer steps remain on the host CPU.

### 2. Precision: BF16 with INT32 accumulator

The precision choice evolved across milestones as the design matured:

| Milestone | Format | Rationale |
|---|---|---|
| M1 (target) | INT8 | Maximum MAC density, minimum bandwidth |
| M2 (first RTL) | FP16 | Preserve accuracy during initial validation |
| M3 / M4 (final) | **BF16** | 8-bit exponent retains FP32 dynamic range; 7-bit mantissa reduces bandwidth; no quantization rescaling overhead |

The final M4 design uses BF16 for weights and activations, with a block-float INT32 accumulator per tile:

| | Format | Width |
|---|---|---|
| Weights / activations | BF16 | 16 bits |
| Multiplier output | INT32 block-float | 32 bits |
| MAC accumulator | INT32 | 32 bits |

BF16 matches PyTorch's native training dtype, requires no rescaling pipeline, and fits SAED14nm adder/multiplier primitives cleanly. It reduces memory bandwidth by 50% versus FP32 while preserving the numerical dynamic range needed for Conv2d partial-sum accumulation.

### 3. Interface: AXI4-Stream (data) + AXI4-Lite (control)

The chiplet exposes:

- **AXI4-Stream** for input feature maps and weight tiles — 512-bit data width, **64 GB/s rated** at 500 MHz.
- **AXI4-Lite** (13-bit address) for the control plane — start/done handshake and 1,024 result registers at addresses 0x1000–0x1FFC.

#### Why AXI4-Stream and not SPI / I²C / AXI4-Lite alone

| Interface | Rated BW | Required (21.3 GB/s) | Verdict |
|---|---|---|---|
| SPI (50 MHz) | ~0.006 GB/s | 21.3 GB/s | No — 3,500× short |
| I²C (1 MHz) | ~0.0001 GB/s | 21.3 GB/s | No |
| AXI4-Lite | ~1–4 GB/s | 21.3 GB/s | No — register overhead, no bursts |
| **AXI4-Stream (512-bit, 500 MHz)** | **64 GB/s** | 21.3 GB/s | **Yes — 3× margin** |
| PCIe 4.0 x16 | ~32 GB/s | 21.3 GB/s | Over-specified, adds complexity |
| UCIe (die-to-die) | ~460 GB/s | 21.3 GB/s | Far exceeds requirement |

#### Required bandwidth from arithmetic intensity

The Conv2d kernel has measured **arithmetic intensity AI = 46.9 FLOP/byte** (fwd+bwd, no DRAM reuse). At the 1 TFLOP/s co-processor target:

Required BW = Peak compute / Arithmetic Intensity
= 1,000 GFLOP/s / 46.9 FLOP/byte
= 21.3 GB/s



AXI4-Stream at 512-bit / 500 MHz provides **64 GB/s**, giving a **3× margin** over the requirement. Cross-checking the interface bottleneck:

Interface-limited perf = 64 GB/s × 46.9 FLOP/byte = ~3.0 TFLOP/s
Compute-limited perf   = 1.0 TFLOP/s  ← binding constraint



The compute array is the bottleneck — not the interface — which is the correct design outcome.

#### Roofline position on the co-processor

The co-processor shares the host DDR4-2667 bus (45.8 GB/s peak), giving a co-processor ridge point of **21.8 FLOP/byte**. Since the kernel's AI (46.9) exceeds this ridge, the kernel remains **compute-bound** on the co-processor as well.

---

## Milestone Journey: M1 → M4

| Milestone | Deliverable | Key Result |
|---|---|---|
| **M1** | Software baseline + roofline analysis | Conv2d = 23.1% runtime; AI = 46.9 FLOP/byte; 142.7 GFLOP/s; AXI4-Stream selected |
| **M2** | FP16 MAC core + AXI4-Stream RTL | 2/2 unit tests + 4/4 integration tests passing (Icarus Verilog) |
| **M3** | BF16 fwd+bwd tile (top_ti); synthesis pivot | TESTS=3 PASS=3; OpenLane/sky130A failed (21.3 ns critical path vs 1.67 ns required); moved to Cadence SAED14nm |
| **M4** | 32×32 systolic array; full synthesis + benchmark | WNS=0 ps @ 500 MHz; 2.86 mm²; 1,359.89 mW; TESTS=3 PASS=3; **3.31× system speedup** |

### Design evolution highlights

- **Precision:** INT8 target (M1) → FP16 RTL (M2) → BF16 final (M3/M4). BF16 was chosen because it matched PyTorch's training dtype, required no quantization rescaling, and fit the SAED14nm standard-cell multiplier/adder primitives more cleanly.
- **Scale:** Single-tile proof (M2/M3) → 32×32 = 1,024-tile array (M4). The generate-loop in `pe_array.sv` scales from 1 to any N×N without RTL changes.
- **Synthesis flow:** OpenLane 2 / sky130A (M3, failed to close timing) → Cadence Genus 17.14-s037_1 / SAED14nm RVT (M4, closed at 500 MHz with WNS = 0 ps).
- **AXI4-Lite address space:** 6-bit (M3) → 13-bit (M4) to cover 1,024 result registers at 0x1000–0x1FFC.
- **FSM latency per tile:** 11 cycles (M3) → 4 cycles (M4) via parallel LOAD and pipelined MUL/ACC stages.

---

## Key M4 Results

| Metric | Value |
|---|---|
| Technology | SAED14nm RVT |
| Synthesis tool | Cadence Genus 17.14-s037_1 |
| Clock | 500 MHz (2 ns period) |
| Timing slack (WNS) | 0 ps — timing closed, 0 violations |
| Cell count | 5,591,114 |
| Total die area | 2,862,977 µm² (2.86 mm²) |
| Dominant area contributor | pe_array (96.9 %) |
| Total power (TT, 0.8 V, 25 °C) | 1,359.89 mW |
| Dominant power contributor | pe_array (92.6 %) |
| Simulation | TESTS=3 PASS=3 FAIL=0 at 342,140 ns |
| System speedup (Amdahl) | **3.31×** |

---

## Repository Structure

510_Hardware_for_AI_ML
├── README.md                ← you are here; top-level pointer to M4
├── codefest/
│   ├── cf01/
│   ├── cf02/
│   ├── cf03/
│   ├── cf04/
│   ├── cf05/
│   ├── cf06/
│   ├── cf07/
│   ├── cf08/
│   └── cf09/
└── project/
    ├── heilmeier.md          ← Project motivation and goals (Heilmeier questions)
    ├── m1/                   ← Software baseline: profiling, roofline, interface selection
    ├── m2/                   ← FP16 MAC core + AXI4-Stream RTL, module tests
    ├── m3/                   ← BF16 tile + backward pass; OpenLane/sky130A attempt
    └── m4/                   ← BF16 training+inference co-processor; Cadence SAED14nm single-tile synthesis
        ├── README.md           ← File catalog — every M4 file with checklist cross-references
        ├── rtl/                ← 32×32 systolic array RTL (top.sv, pe_array.sv, compute_core_ti.sv, …)
        ├── tb/                 ← Testbenches (tb_top.sv + cocotb Python suites)
        ├── sim/                ← Simulation outputs (final_run.log PASS, final_waveform.png)
        ├── synth/              ← Cadence Genus synthesis results (timing, area, power)
        ├── bench/              ← Benchmark vs M1 baseline (benchmark.md, CSV, roofline plot)
        └── report/             ← Design justification report (PDF + figures)

