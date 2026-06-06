# ECE 410/510 — BF16 Conv2d Accelerator (32×32 Systolic Array)

**Course:** Hardware for AI/ML — Portland State University, Spring 2026  
**Milestone:** M4 (Final Submission)  
**PDK:** SAED14nm RVT | **Tool:** Cadence Genus 17.14-s037_1 | **Clock:** 500 MHz  
**Report:** [`project/m4/report/design_justification.pdf`](project/m4/report/design_justification.pdf)

---

## Project Overview

This repository contains the complete M4 hardware deliverable for an anemia-detection ResNet-18 training accelerator. The motivation comes from an M1 software profiling result showing that Conv2d forward and backward passes account for **74.4%** of a 2,573.9-second per-epoch wall-clock time on an Intel i5-10210U (142.7 GFLOP/s, 53% of AVX2 peak). The hardware target was to reach at least 1 TFLOP/s and achieve a ≥3× system-level speedup via Amdahl's Law.

The design implements a **32×32 weight-stationary systolic array** (1,024 processing elements) that computes a full 3×3 BF16 Conv2d across all output tiles simultaneously. Every PE shares a broadcast weight bus (9 × BF16 = 144 bits) loaded once per kernel invocation; per-tile input feature map (IFM) windows are extracted combinationally from a 34×34 BF16 scratchpad. The synthesized 32×32 array meets timing at 500 MHz with WNS = 0 ps, delivers **2.304 TFLOP/s**, and consumes **1.36 W** — yielding **0.59 pJ/FLOP** (178× better than the CPU baseline) and a **3.31× system speedup** (Amdahl, f = 74.4%).

The data format evolved across milestones: Q16.16 fixed-point (M1) → FP16 (M2) → **BF16** (M3/M4). BF16 was selected because it preserves the 8-bit FP32 exponent field, avoiding exponent re-scaling at the BF16→FP32 boundary. Multiply outputs are accumulated in a block-float INT32 tree (find max exponent, align mantissas, 4-level binary adder, single normalize to FP32) to prevent overflow without a wide floating-point adder chain.

---

## Architecture at a Glance

| Parameter | Value |
|---|---|
| Array size | 32×32 = 1,024 tiles (`pe_array.sv`) |
| Kernel size | 3×3 BF16 (9 MACs / tile / invocation) |
| Dataflow | Weight-stationary — weights broadcast, IFM windows distributed |
| Forward FSM | 4 cycles: IDLE → LOAD → PIPE_MUL → PIPE_ACC → DONE_ST |
| Zero-skip | Activation zero-skip saves 2 cycles when all 9 IFM values are zero |
| Weight zero-skip | `conv_interface_ti` suppresses `core_start` when all 9 weights are zero |
| Accumulator | INT32 block-float, 4-level adder tree, single FP32 normalize |
| Primary interface | AXI4-Stream 512-bit (IFM/weight load) + AXI4-Lite 13-bit addr (control/result) |
| Result storage | 1,024 FP32 results via register map 0x1000–0x1FFC |
| WNS (32×32) | 0 ps @ 500 MHz (timing_32x32.rpt) |
| Total area | 2.86 mm² (5,591,114 cells) |
| Total power | 1.36 W (1.23 mW/tile average) |
| Throughput | **2.304 TFLOP/s** |

---

## Folder Structure

```
systolicArray/
│
├── README.md                   ← this file
│
├── rtl/                        ← synthesizable RTL (all 32×32, DUT = top_ti)
│   ├── top.sv                  ← top-level: instantiates conv_interface_ti + pe_array
│   ├── pe_array.sv             ← 1024-tile generate loop; shared weights_packed (144-bit bus)
│   ├── compute_core_ti.sv      ← single PE: 4-cycle FSM, bf16_mul, INT32 accumulator
│   ├── interface_ti.sv         ← AXI4-Lite/Stream controller; 1024 result registers
│   ├── ifm_buffer.sv           ← 34×34 BF16 scratchpad; combinational 3×3 window extractor
│   ├── grad_core.sv            ← backward-pass gradient core (dL/dW, dL/dX)
│   └── sram_sp.sv              ← single-port SRAM macro model
│
├── tb/                         ← cocotb v2.0.1 testbenches (Icarus Verilog 11.0)
│   ├── tb_top.sv               ← SystemVerilog top-level harness for full 32×32 DUT
│   ├── test_forward.py         ← forward-pass test: 32×32 BF16 random kernel
│   ├── test_backward.py        ← backward-pass tests: grad_w and grad_x checks
│   ├── test_top.py             ← end-to-end AXI-Stream load → compute → AXI-Lite readback
│   ├── test_ifm_buffer.py      ← IFM buffer window extraction correctness
│   ├── test_pe_array.py        ← pe_array parallel tile result verification
│   ├── test_interface.py       ← AXI4-Lite register map and handshake tests
│   ├── test_compute_core.py    ← single-tile BF16 multiply-accumulate unit tests
│   ├── test_core_rev4.py       ← Rev4 FSM timing regression tests
│   └── ref_model.py            ← pure-Python BF16 reference model (golden results)
│
├── sim/                        ← simulation drivers and outputs
│   ├── run_ti.py               ← MAIN runner: compiles + runs all 3 M4 tests on top_ti
│   ├── final_run.log           ← last clean run: TESTS=3 PASS=3 FAIL=0, 342,140 ns
│   ├── final_waveform.png      ← GTKWave screenshot of forward-pass waveform
│   ├── dump.vcd / dump.gtkw    ← VCD waveform and GTKWave save file
│   ├── run_array.py            ← pe_array standalone runner
│   ├── run_ifm_buffer.py       ← ifm_buffer standalone runner
│   ├── run_interface.py        ← interface standalone runner
│   ├── run_pe_array.py         ← pe_array runner (alt entry point)
│   ├── run_core_rev4.py        ← compute_core_ti Rev4 runner
│   ├── run_top.py              ← full top_ti runner (alt entry point)
│   └── *_build/                ← per-test Icarus compilation artifacts (sim.vvp, results.xml)
│
├── bench/                      ← benchmarking and roofline analysis
│   ├── benchmark.md            ← narrative comparison: M1 CPU vs M4 single-tile vs M4 array
│   ├── benchmark_data.csv      ← all raw numbers (M1_baseline, M4_single_tile, M4_array_32x32)
│   ├── build_roofline.py       ← generates roofline_final.png (matplotlib, Agg backend)
│   └── roofline_final.png      ← roofline plot: CPU / co-proc curves + 3 operating points
│
├── synthesis/                  ← RTL snapshot sent to Cadence Genus for 32×32 run
│   ├── rtl/                    ← copy of rtl/ used during synthesis job
│   ├── run_genus.tcl           ← Genus legacy_ui script (constraints, effort, HDL limits)
│   └── run_innovus.tcl         ← Innovus P&R script (not fully closed; used for area estimate)
│
├── timing_32x32.rpt            ← Genus post-syn timing: WNS=0ps, 10 paths, 500 MHz
├── area_32x32.rpt              ← Genus post-syn area: 5,591,114 cells, 2,862,977 µm²
├── power_32x32.rpt             ← Genus post-syn power: 1,359.89 mW total (TT corner)
├── gates_32x32.rpt             ← gate-level netlist summary
├── genus.log                   ← full Cadence Genus stdout log for the 32×32 synthesis run
├── run_genus.tcl               ← root-level Genus script (mirrors synthesis/run_genus.tcl)
├── run_innovus.tcl             ← root-level Innovus script
│
└── project/m4/                 ← M4 submission package (organized deliverables)
    ├── README.md               ← per-file catalog with report section cross-references
    ├── rtl/                    ← (pending copy) RTL files for submission
    ├── tb/                     ← (pending copy) testbench files for submission
    ├── sim/                    ← (pending copy) final_run.log + waveform
    ├── bench/                  ← (pending copy) benchmark.md, CSV, roofline PNG
    ├── synth/
    │   └── config.json         ← synthesis configuration record (tool, PDK, clock, results)
    └── report/
        ├── design_justification.pdf  ← 9-section design report (~2,720 words)
        ├── generate_report.py        ← ReportLab 4.5.1 script that produced the PDF
        └── figures/                  ← figure assets for the report
```

---

## Running Simulations

All tests run against the full 32×32 DUT (`top_ti`). Requires cocotb v2.0.1 and Icarus Verilog 11.0.

```bash
# Run all three M4 tests (forward, backward, top-level AXI)
python sim/run_ti.py

# Run individual sub-tests
python sim/run_array.py
python sim/run_ifm_buffer.py
python sim/run_interface.py
```

Expected output: `TESTS=3 PASS=3 FAIL=0` in approximately 342,140 ns simulation time.  
Full log: [`sim/final_run.log`](sim/final_run.log)

---

## Regenerating Benchmark Artifacts

```bash
# Regenerate roofline plot (saves bench/roofline_final.png)
python bench/build_roofline.py

# Regenerate design justification PDF (saves project/m4/report/design_justification.pdf)
python project/m4/report/generate_report.py
```

---

## Key Results Summary

| Metric | M1 CPU Baseline | M4 32×32 Array | Improvement |
|---|---|---|---|
| Throughput (Conv2d) | 142.7 GFLOP/s | **2,304 GFLOP/s** | **16.1×** |
| System speedup (Amdahl) | 1× | **3.31×** | — |
| Wall-clock per epoch | 2,573.9 s | ~777 s | −70% |
| Energy per FLOP | 105 pJ/FLOP | **0.59 pJ/FLOP** | **178×** |
| Power | ~15 W (TDP) | 1.36 W | 51× less |
| Timing closure | — | WNS = 0 ps @ 500 MHz | ✓ |

---

## RTL Module Hierarchy

```
top_ti  (rtl/top.sv)
├── conv_interface_ti  (rtl/interface_ti.sv)   — AXI4-Lite/Stream, weight broadcast, result mux
│   └── ifm_buffer     (rtl/ifm_buffer.sv)     — 34×34 scratchpad, combinational window extract
└── pe_array           (rtl/pe_array.sv)        — 1024-tile generate loop
    └── compute_core_ti × 1024  (rtl/compute_core_ti.sv)
        └── bf16_mul (function)                  — BF16×BF16→FP32, mantissa concat corrected
```

The `grad_core.sv` and `sram_sp.sv` modules support the backward pass and SRAM modelling respectively; they are instantiated within `compute_core_ti` for the gradient computation path.

---

## Milestones in Other Branches

M1 (software baseline), M2 (FP16 unit), and M3 (single-tile RTL) deliverables are tracked separately in the project git repository and are not duplicated here. This folder contains only M4 material.
