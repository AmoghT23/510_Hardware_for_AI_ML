# Anemia Detection — ResNet18 BF16 Conv2d Hardware Accelerator
**ECE 410/510 · Hardware for AI/ML · Spring 2026**

This repository contains the complete M4 submission for a custom hardware accelerator targeting the Conv2d bottleneck of an anemia detection pipeline (HybridModel: ResNet18 + AttentionFusion MLP on the AneRBC peripheral blood smear dataset). The accelerator is a 32×32 weight-stationary systolic array performing BF16 forward and backward convolution, synthesized with Cadence Genus 17.14-s037_1 on SAED14nm RVT at 500 MHz with WNS = 0 ps. The 1024-tile array achieves 2.304 TFLOP/s at 1.36 W — 16.1× kernel speedup and 3.31× system-level speedup over the CPU baseline (Amdahl, 74.4% accelerated fraction), and 178× more energy-efficient than the CPU at 0.59 pJ/FLOP.

## M4 Submission

- **M4 deliverables index:** [`project/m4/README.md`](project/m4/README.md)
- **Design justification report (PDF):** [`project/m4/report/design_justification.pdf`](project/m4/report/design_justification.pdf)
- **Final synthesis results:** [`project/m4/synth/`](project/m4/synth/) — Cadence Genus 17.14-s037_1, SAED14nm RVT, 500 MHz
- **Benchmark comparison:** [`project/m4/bench/benchmark.md`](project/m4/bench/benchmark.md)

## Repository Structure

```
project/
├── heilmeier.md          Project motivation and goals (Heilmeier questions)
├── m1/                   Software baseline: profiling, roofline, interface selection
├── m2/                   RTL: FP16 compute core + AXI4-Stream interface, module tests
├── m3/                   INT8 compute core, OpenLane synthesis attempt (sky130A)
├── m3_ti/                BF16 training+inference co-processor, Cadence SAED14nm single-tile synthesis
└── m4/                   ← Final submission (M4)
    ├── README.md         File catalog (this folder)
    ├── rtl/              Final synthesized RTL (top.sv, compute_core_ti.sv, interface_ti.sv, ...)
    ├── tb/               Testbenches (tb_top.sv SystemVerilog + cocotb Python suites)
    ├── sim/              Simulation outputs (final_run.log PASS, final_waveform.png)
    ├── synth/            Cadence Genus synthesis results (timing, area, power, critical path)
    ├── bench/            Benchmark vs M1 baseline (benchmark.md, CSV, roofline plot)
    └── report/           Design justification report (PDF + figures)
```

## Key Results

| Metric | Value |
|---|---|
| Technology | SAED14nm RVT 14 nm FinFET |
| Tool | Cadence Genus 17.14-s037_1 |
| Clock | 500 MHz (2 ns period) |
| Timing slack WNS (32×32 array) | **0 ps — MEETS CONSTRAINT**, 0 violations |
| Cell count (32×32 array) | 5,591,114 |
| Die area (32×32 array) | 2,862,977 µm² **(2.86 mm²)** |
| Power (32×32 array, 500 MHz TT) | **1,359.89 mW (1.36 W)** |
| Per-tile area | ~2,709 µm² |
| Per-tile power | ~1.23 mW |
| Throughput (1 tile) | 2.25 GFLOP/s |
| Throughput (32×32 array, 1024 tiles) | **2,304 GFLOP/s (2.304 TFLOP/s)** |
| Kernel speedup vs CPU (Conv2d) | **16.1×** |
| System speedup vs CPU (Amdahl) | **3.31×** |
| Energy efficiency | **0.59 pJ/FLOP (178× better than CPU)** |
| vs T4 GPU (70 W) | **51× less power** |
| Simulation | TESTS=3 PASS=3 FAIL=0 at 342,140 ns |

## M1 → M4 Evolution

| Milestone | Precision | Synthesis | Tiles | Throughput | Speedup |
|---|---|---|---|---|---|
| M1 | Q16.16 fixed | — | — | 142.7 GFLOP/s (CPU baseline) | 1× |
| M2 | FP16 | — | 1 (proof of concept) | — | — |
| M3 | BF16 | OpenLane/sky130A (failed timing) | 1 | 2.25 GFLOP/s | 0.016× |
| M3_ti | BF16 | Cadence Genus/SAED14nm | 1 | 2.25 GFLOP/s | 0.016× |
| **M4** | **BF16** | **Cadence Genus 17.14/SAED14nm** | **1024 (32×32)** | **2,304 GFLOP/s** | **16.1× kernel / 3.31× system** |
