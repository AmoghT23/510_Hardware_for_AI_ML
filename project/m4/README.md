# M4 Deliverables — BF16 Conv2d Accelerator (32×32 Systolic Array)

**ECE 410/510 | Portland State University | Spring 2026**  
**PDK:** SAED14nm RVT | **Tool:** Cadence Genus 17.14-s037_1 | **Clock:** 500 MHz

---

This directory contains all Milestone 4 deliverables for the **BF16 Conv2d Accelerator** project (ECE 510, Spring 2026, Portland State University). The accelerator implements a 32×32 systolic array that performs 2D convolution in BrainFloat16 (BF16), synthesized to the SAED14nm RVT standard-cell library at a 500 MHz target clock using Cadence Genus. The complete design justification — covering roofline analysis, precision choices, dataflow architecture, verification, synthesis results, benchmarks, and lessons learned — is in [report/design_justification.pdf](report/design_justification.pdf). The file catalog below maps every committed file to its checklist item and report section.

## File Catalog

```
project/m4/  —  32×32 BF16 Conv2d accelerator  |  ECE 410/510 Spring 2026  |  SAED14nm 500 MHz
│
├── README.md  —  this file; per-file catalog with report-section cross-references
│
├── rtl/  —  synthesizable RTL (7 modules, DUT = top_ti, 32×32 = 1024 tiles)               [§4, §5]
│   ├── top.sv              —  top_ti: instantiates conv_interface_ti + pe_array; AXI4-Lite 13-bit / AXI4-Stream 512-bit
│   ├── pe_array.sv         —  1024-tile generate loop; shared weights_packed[143:0] broadcast; done_all = &tile_done
│   ├── compute_core_ti.sv  —  single PE; 4-cycle FSM (IDLE→LOAD→PIPE_MUL→PIPE_ACC→DONE_ST); INT32 block-float acc
│   ├── interface_ti.sv     —  AXI controller; weight broadcast; 1024 FP32 result regs at 0x1000–0x1FFC; weight zero-skip
│   ├── ifm_buffer.sv       —  34×34 BF16 scratchpad; 37-beat AXI4-Stream fill; combinational 3×3 window extractor
│   ├── grad_core.sv        —  18 gradients (dL/dW + dL/dX) computed combinatorially in 1 clock cycle
│   └── sram_sp.sv          —  9×16-bit single-port SRAM; async read; weights persist between kernel invocations
│
├── tb/  —  cocotb v2.0.1 testbenches (Icarus Verilog 11.0)                                [§6]
│   ├── tb_top.sv           —  SystemVerilog top-level harness for full 32×32 DUT (top_ti)
│   ├── test_forward.py     —  test_bf16_forward (T1–T4): ramp/alternating/max/Laplacian kernels; all 1024 results checked
│   │                          test_weight_reuse: weights=[1..9] loaded once, 4 different IFM tiles verified
│   ├── test_backward.py    —  test_grad_w (B1–B4): dL/dW[i] = dL/dy × BF16(x[i]) for all 9 weight gradients
│   ├── test_top.py         —  end-to-end: AXI-Stream load → compute → AXI-Lite readback of all 1024 result regs
│   ├── test_ifm_buffer.py  —  37-beat buffer fill correctness + 3×3 window extraction for all 1024 tiles
│   ├── test_pe_array.py    —  parallel tile result verification across all N_TILES=1024 compute_core_ti instances
│   ├── test_interface.py   —  AXI4-Lite register map, handshake protocol, and weight zero-skip logic tests
│   ├── test_compute_core.py—  single-tile BF16 multiply-accumulate unit tests
│   ├── test_core_rev4.py   —  Rev4 FSM timing regression: verifies 4-cycle latency for compute_core_ti
│   └── ref_model.py        —  pure-Python BF16 reference model; generates golden results for all test vectors
│
├── sim/  —  simulation drivers and outputs                                                 [§6]
│   ├── run_ti.py           —  MAIN runner: compiles + runs all 3 M4 tests on top_ti; produces final_run.log
│   ├── final_run.log       —  TESTS=3 PASS=3 FAIL=0 SKIP=0 at 342,140 ns (cocotb + Icarus)
│   ├── final_waveform.png  —  GTKWave screenshot: AXI-Stream load → forward compute → result readback
│   ├── dump.vcd            —  VCD waveform file from the final simulation run
│   ├── dump.gtkw           —  GTKWave save file (signal selection and waveform annotations)
│   ├── run_array.py        —  pe_array standalone simulation runner
│   ├── run_ifm_buffer.py   —  IFM buffer standalone simulation runner
│   ├── run_interface.py    —  AXI interface standalone simulation runner
│   ├── run_pe_array.py     —  pe_array runner (alternate entry point)
│   ├── run_core_rev4.py    —  compute_core_ti Rev4 FSM standalone runner
│   └── run_top.py          —  full top_ti runner (alternate entry point)
│
├── synth/  —  synthesis results (Cadence Genus 17.14-s037_1, SAED14nm RVT, 500 MHz)      [§7]
│   ├── config.json         —  synthesis config: tool v17.14-s037_1, PDK, clock 2 ns, all 7 RTL sources│   ├── openlane_run.log    —  Cadence Genus full stdout; 151,145 lines; Jun 01 2026; auto.ece.pdx.edu │   ├── timing_report.txt   —  WNS=0 ps @ 500 MHz; 10 paths; critical path gen_tile[163] │   ├── area_report.txt     —  5,591,114 cells; 2,862,977 µm² (2.86 mm²); pe_array = 96.9% of area    │   ├── power_report.txt    —  1,359.89 mW total (TT corner, 0.8 V, 25 °C); pe_array = 92.6% of power│   ├── run_genus.tcl       —  Genus legacy_ui TCL script; hdl_max_loop_limit=4096; all timing constraints
│   ├── run_innovus.tcl     —  Innovus P&R script; attempted but timing could not close at 500 MHz
│   └── gates_report.rpt    —  gate-level netlist summary by module
│
├── bench/  —  benchmarking and roofline analysis                                           [§2, §8]
│   ├── benchmark.md        —  throughput, energy, speedup vs M1 CPU; Amdahl 3.31× system speedup │   ├── benchmark_data.csv  —  91 raw rows: M1_baseline, M4_single_tile, M4_array_32x32, comparison│   ├── build_roofline.py   —  script that generates roofline_final.png (matplotlib, Agg backend)
│   └── roofline_final.png  —  CPU + co-proc ridges + 3 operating points; log-log axes (Fig 2.1)  │
└── report/  —  design justification report                                                 [all §]
    ├── design_justification.pdf  —  9-section report, ~2,900 words, 18 pages (PDF)          ├── design_justification.md   —  Markdown source for the report
    └── figures/  —  figure assets committed for the report                                       ├── roofline_final.png                   —  Fig 2.1: roofline plot; referenced in §2.6
        ├── final_waveform.png                   —  Fig 6.1: forward-pass waveform; referenced in §6
        │
        │   [extra kernel result figures — additional outputs from model/kernel profiling]
        ├── confusion_matrix_anemia_core_kernel.png  —  EXTRA: confusion matrix of the anemia-detection CNN on the test set
        ├── hotspot_analysis.png                     —  EXTRA: conv2d kernel compute hotspot heatmap (profiling output)
        ├── post_training_inference.png              —  EXTRA: sample model inference results after training
        ├── sample_anemic_image.png                  —  EXTRA: representative anemic RBC image from AneRBC dataset
        └── sample_healthy_image.png                 —  EXTRA: representative healthy RBC image from AneRBC dataset
```

---

## RTL Differences from M3

The M4 RTL differs from M3_ti in the following ways (as required by the checklist):

| Change | M3_ti | M4 |
|---|---|---|
| Array size | Single tile (N_TILES=1) | **1024 tiles (32×32, N_TILES=1024)** |
| pe_array.sv | Not present | New: generate loop instantiating 1024 × compute_core_ti |
| ifm_buffer.sv | Not present | New: 34×34 BF16 scratchpad with combinational window extractor |
| top.sv | Single tile top | **top_ti with pe_array + conv_interface_ti** |
| AXI4-Lite address | 6-bit (M3) / 13-bit (M3_ti) | **13-bit** (covers 0x1000–0x1FFC result map) |
| Forward FSM latency | 11 cycles | **4 cycles** (parallel LOAD, pipelined MUL/ACC) |
| Synthesis scope | Single tile | **Full 32×32 array** (actual silicon area, power, timing) |

---

## How to Reproduce

```bash
# Run all M4 simulations
python sim/run_ti.py
# Expected: TESTS=3 PASS=3 FAIL=0 in ~342,140 ns

# Regenerate roofline plot
python bench/build_roofline.py
```
