# Anemia Detection — ResNet18 Conv2d Hardware Accelerator
**ECE 410/510 · Hardware for AI/ML · Spring 2026**

This repository contains the complete M4 submission for a custom hardware accelerator targeting the Conv2d bottleneck of an anemia detection pipeline (HybridModel: ResNet18 + AttentionFusion MLP on the AneRBC peripheral blood smear dataset). The accelerator is a BF16/INT8 forward+backward tile synthesized with Cadence Genus 21.1 on SAED14nm, achieving 500 MHz with +2.4 ps timing slack, 6,155 µm² die area, and 3.07 mW power per tile. A 444-tile array projects to 1 TFLOP/s at 1.36 W — 51× more power-efficient than an NVIDIA T4 GPU at the same workload.

## M4 Submission

- **M4 deliverables index:** [`project/m4/README.md`](project/m4/README.md)
- **Design justification report (PDF):** [`project/m4/report/design_justification.pdf`](project/m4/report/design_justification.pdf)
- **Final synthesis results:** [`project/m4/synth/`](project/m4/synth/) — Cadence Genus 21.1, SAED14nm, 500 MHz
- **Benchmark comparison:** [`project/m4/bench/benchmark.md`](project/m4/bench/benchmark.md)

## Repository Structure

```
project/
├── heilmeier.md          Project motivation and goals (Heilmeier questions)
├── m1/                   Software baseline: profiling, roofline, interface selection
├── m2/                   RTL: FP16 compute core + AXI4-Stream interface, module tests
├── m3/                   INT8 compute core, OpenLane synthesis attempt (sky130A)
├── m3_ti/                BF16 training+inference co-processor, Cadence SAED14nm synthesis
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
| Technology | SAED14nm 14 nm FinFET |
| Tool | Cadence Genus 21.1 |
| Clock | 500 MHz (2 ns period) |
| Timing slack (WNS) | +2.4 ps — **PASS**, 0 violations |
| Cell count | 11,340 |
| Die area (1 tile) | 6,155 µm² (~78 × 78 µm) |
| Power (1 tile, 500 MHz TT) | 3.07 mW |
| Throughput (1 tile) | 2.25 GFLOP/s |
| Throughput (444-tile array) | ~1 TFLOP/s |
| Array power | 1.36 W |
| vs T4 GPU (70 W) | **51× less power** |
| vs CPU baseline (142.7 GFLOP/s) | **7× kernel, 2.76× system** |
| Simulation | TESTS=3 PASS=3 FAIL=0 |
