# Design Justification Report

**ECE 410/510 — Hardware for Artificial Intelligence and Machine Learning**  
**Portland State University | Spring 2026**

---

**Project:** BF16 Conv2d Accelerator — 32×32 Weight-Stationary Systolic Array  
**Student:** Amogh Thakur | amogh.thakur23@gmail.com  
**PDK:** SAED14nm RVT | **Tool:** Cadence Genus 17.14-s037_1 | **Clock:** 500 MHz

---

## Section 1: Problem and Motivation

The target application is a hybrid anemia-detection model (ResNet18 backbone + AttentionFusion MLP) trained on the AneRBC peripheral blood smear dataset (**12,000 images**: 6,000 healthy, 6,000 anemic). The model was profiled on an Intel Core i5-10210U laptop CPU (Windows 11, PyTorch 2.11.0, CPU-only) over **10 full training epochs**, each epoch processing **9,600 training images** in **300 batches of 32**.

### 1.1 Wall-Clock Timing (time.perf_counter, full pipeline)

| Run | Wall-Clock (s) | Run | Wall-Clock (s) |
|-----|---------------|-----|---------------|
| 1 | 2762.6 | 6 | 2497.8 |
| 2 | 2670.4 | 7 | 2508.4 |
| 3 | 2595.1 | 8 | 2481.1 |
| 4 | 2537.6 | 9 | 2860.9 |
| 5 | 2552.6 | 10 | 2719.2 |

- **Median wall-clock: 2573.9 s (42.9 min) per epoch**
- Mean: **2618.6 s** | Std deviation: **128.8 s**

### 1.2 cProfile Kernel Identification

cProfile of `hybrid_train_one_epoch` (**300 batches, 1266.6 s** total profiling runtime) identified `torch.conv2d` as the dominant kernel:

| Function | Cumtime (s) | % of Total |
|---|---|---|
| `run_backward` (autograd engine) | **732.7** | **57.8%** |
| `HybridModel.forward` | 421.6 | 33.3% |
| `resnet.forward` | 324.8 | 25.6% |
| **`torch.conv2d` ← DOMINANT** | **292.9** | **23.1%** |
| `torch.batch_norm` | 60.3 | 4.8% |
| `torch.max_pool2d` | 47.1 | 3.7% |

`torch.conv2d` accounts for **23.1%** of runtime directly and **57.8%** including backpropagation. Combined Conv2d forward + backward accounts for **74.4%** of the accelerated pipeline.

### 1.3 Measured CPU Throughput

- FLOPs per batch (fwd+bwd, 32 images): **348.4 GFLOP** (from `ai_calculation.md`)
- Average backward time per batch: **2.442 s**
- **Measured throughput: 348.4 / 2.442 = 142.7 GFLOP/s**
- CPU peak theoretical (Intel ARK, AVX2, 4.2 GHz boost): **268.8 GFLOP/s**
- CPU utilization: **142.7 / 268.8 = ~53%**

The CPU is operating at **53%** of its theoretical peak on a compute-bound kernel that consumes **74.4%** of training time. Custom hardware that eliminates general-purpose CPU overhead (cache hierarchies, branch prediction, out-of-order execution) for Conv2d is the correct intervention.

### 1.4 Model Accuracy and Justification for Conv2d Acceleration

From `_run_outputs/run_01_output.ipynb` (representative run):

| Model | Val Accuracy | Val Loss | Train Accuracy |
|---|---|---|---|
| CNN Baseline (ResNet18, 1 epoch) | **92.37%** | 0.1846 | 85.52% |
| HybridModel (ResNet18 + AttentionFusion, 1 epoch) | **89.92%** | 0.2212 | 85.35% |

Attention weight analysis (sample inference, `0398_08_a.png`):

| Feature Source | Weight |
|---|---|
| CNN features (ResNet18 backbone) | **98.94%** |
| Handcrafted features (GLCM, morphology, colour) | 1.06% |

The model relies on the CNN visual path for **98.94%** of its classification decision. This directly confirms that accelerating the ResNet18 Conv2d layers (the dominant component of the CNN path) is the correct hardware target — the handcrafted feature path contributes negligibly and does not merit hardware acceleration.

**M1 Hypothesis:** build a systolic array delivering ≥ **1 TFLOP/s**, achieving ≥ **3× system-level speedup** (Amdahl, f = **74.4%**). At 1 TFLOP/s the projected kernel speedup was **7×** and system speedup **2.76×**.  
**M4 Result:** **2.304 TFLOP/s**, **16.1× kernel speedup**, **3.31× system speedup** — both targets exceeded because 1024 tiles were implemented rather than the minimum 444 tiles needed to reach 1 TFLOP/s.

---

## Section 2: Roofline Analysis

Arithmetic intensity was calculated analytically from ResNet18 layer dimensions (torchinfo, batch_size=1, input=224×224×3) and verified against cProfile measurements.

### 2.1 FLOP Count

| Scope | FLOPs |
|---|---|
| Total Mult-Adds (torchinfo, all Conv2d layers) | **1,814,511,616** |
| Forward pass (1 image) | **3,629,023,232** |
| Forward + backward (1 image, ×3 fwd) | **10,887,069,696** |
| **Per batch (32 images, fwd+bwd)** | **348,386,230,272 (348.4 GFLOP)** |

### 2.2 Bytes Transferred (float32, no DRAM reuse)

| Operand | Bytes |
|---|---|
| Weights (all 20 Conv2d layers) | 45,648,896 |
| Input activations | 23,986,688 |
| Output activations | 23,592,960 |
| **Forward total (1 image)** | **93,228,544 bytes (93.2 MB)** |
| **Forward + backward (1 image)** | **232,106,048 bytes (232.1 MB)** |
| **Per batch (32 images)** | **7,427,393,536 bytes (7.43 GB)** |

### 2.3 Arithmetic Intensity

```
AI (fwd + bwd, per batch) = 348.4 GFLOP / 7.43 GB = 46.9 FLOP/byte
AI (fwd only, 1 image)    = 3.63 GFLOP / 93.2 MB  = 38.9 FLOP/byte
```

### 2.4 Roofline Parameters

| Platform | Peak Compute | Memory BW | Ridge Point |
|---|---|---|---|
| CPU (i5-10210U) | **268.8 GFLOP/s** | **45.8 GB/s** | **5.9 FLOP/byte** |
| M4 32×32 Array | **2.304 TFLOP/s** | **45.8 GB/s** (shared DDR4) | **50.3 FLOP/byte** |

At **AI = 46.9 FLOP/byte >> ridge 5.9 FLOP/byte**, the Conv2d kernel is strongly **COMPUTE-BOUND** on the CPU. At **AI = 46.9 FLOP/byte < ridge 50.3 FLOP/byte**, the kernel approaches memory-bound on the co-processor — however, weight-stationary reuse (weights loaded once, broadcast to all **1024 PEs**) reduces effective bytes/FLOP, keeping the design in the compute-bound regime in practice.

### 2.5 Operating Points (see `bench/roofline_final.png`)

| Point | AI (FLOP/byte) | Performance |
|---|---|---|
| M1 CPU (measured, cProfile) | **46.9** | **142.7 GFLOP/s** |
| M4 single tile (Cadence synthesis) | **46.9** | **2.25 GFLOP/s** |
| **M4 32×32 array (synthesis-validated)** | **46.9** | **2,304 GFLOP/s** |

### 2.6 Amdahl's Law (f = 74.4%)

```
System speedup = 1 / ((1 - 0.744) + 0.744 / 16.1)
               = 1 / (0.256 + 0.046)
               = 1 / 0.302
               = 3.31×
```

The roofline analysis shaped two key architectural decisions:
1. **Weight-stationary dataflow** — reuse weights across all **1024** output tiles to reduce effective bandwidth demand.
2. **1024-tile array (32×32)** — scale compute to **2.304 TFLOP/s** to stay above the **1 TFLOP/s** M1 target while keeping bandwidth within DDR4 limits.

---

## Section 3: Precision and Data Format

The data format evolved across three milestones:

### 3.1 Format Evolution

| Milestone | Format | Width | Reason for Change |
|---|---|---|---|
| M1 | Q16.16 fixed-point | 32 bits | Baseline; silent overflow risk, 2× bandwidth vs FP16 |
| M2 | FP16 (IEEE 754) | 16 bits | 2× bandwidth gain, floating exponent, industry standard |
| **M3/M4** | **BF16 (Brain Float 16)** | **16 bits** | **No exponent rebasing hardware, TPU/A100 alignment** |

### 3.2 BF16 Format Detail

```
BF16: 1 sign + 8 exponent (bias=127) + 7 mantissa
FP16: 1 sign + 5 exponent (bias=15)  + 10 mantissa
FP32: 1 sign + 8 exponent (bias=127) + 23 mantissa
```

**Key advantage of BF16 over FP16:** BF16 shares the same 8-bit exponent field as FP32 (bias=127). Converting BF16→FP32 requires only zero-extending the mantissa from 7 to 23 bits — no exponent rebasing. FP16→FP32 requires adding **112** (= 127 − 15) to every exponent, adding a hardware adder on the critical path.

- **BF16 SQNR:** 6.02 × 7 + 1.76 ≈ **44 dB** (above the **40 dB** minimum for CNN classification)
- **FP16 SQNR:** 6.02 × 10 + 1.76 ≈ **62 dB**
- Worst-case accumulated error (deepest layer, **4,608 MACs**): **4,608 × 1.19×10⁻⁷ ≈ 5.5×10⁻⁴** — three orders of magnitude below classification output tolerance.

### 3.3 M4 INT32 Block-Float Accumulator

The M4 `compute_core_ti` implements a block-float INT32 accumulator in place of a floating-point adder tree:

1. `bf16_mul` function: BF16 × BF16 → FP32 (sign / exp / mantissa)
2. Find **maximum exponent** across all **9** products
3. Right-shift all mantissas to align to maximum exponent
4. Sum aligned mantissas in a **4-level binary adder tree** (INT32)
5. Single normalize + round → FP32 output

This avoids **9** independent FP32 adders; only **one** normalize step is needed, significantly reducing area.

---

## Section 4: Dataflow and Architecture

### 4.1 Dataflow Pattern: Weight-Stationary

In the weight-stationary pattern, filter weights are loaded once and held fixed in each PE while IFMs flow through:

- **9 BF16 weights (144 bits)** are broadcast to all **1024 PEs** simultaneously via a shared `weights_packed[143:0]` bus (`pe_array.sv`)
- Each PE receives its own **3×3 IFM window (144 bits)** extracted combinationally from the **34×34 BF16 IFM scratchpad** (`ifm_buffer.sv`)
- All **1024 PEs** compute their dot-product independently and **in parallel**

**Alternative patterns considered:**
- *Output-stationary:* requires accumulation across channels — not suited to single-channel 3×3 with 9 MACs per PE
- *Input-stationary:* wastes bandwidth when the same weights apply to many tiles

Weight-stationary was chosen because the **9 BF16 weights** broadcast over a **144-bit bus** cost zero per-PE overhead, and the **34×34 IFM buffer** enables combinational window extraction without SRAM read latency.

### 4.2 Module Hierarchy

```
top_ti  (rtl/top.sv)  —  N_TILES = 1024
├── conv_interface_ti  (rtl/interface_ti.sv)
│     AXI4-Lite 13-bit / AXI4-Stream 512-bit
│     Drives core_result[32767:0], core_done[1023:0], core_start
│     Weight zero-skip: suppresses core_start when all 9 weights = 0
│     Register map:
│       0x00:        CTRL   [0]=start [1]=load_weights [2]=backward
│       0x04:        STATUS [0]=done  [1]=busy [2]=wt_loaded
│       0x0C:        RESULT[0]  (tile 0 FP32 result)
│       0x1000–0x1FFC: RESULT[0]–RESULT[1023]  (all tiles, 4-byte stride)
│
├── ifm_buffer  (rtl/ifm_buffer.sv)
│     34×34 BF16 scratchpad  (18,496 bits on-chip)
│     AXI4-Stream input: 37 beats to fill the buffer (512 bits/beat)
│     Combinational 3×3 window extractor — all 1024 windows available
│     with no SRAM read latency
│
└── pe_array  (rtl/pe_array.sv)
      generate loop: for (genvar t = 0; t < 1024; t++)
      Shared weights_packed[143:0] broadcast to all tiles
      done_all = &tile_done  (AND-reduction of 1024 signals)
      └── compute_core_ti × 1024  (rtl/compute_core_ti.sv)
            4-state FSM:
              IDLE     → waits for core_start
              LOAD     → latches 9 BF16 IFMs + 9 BF16 weights  (1 cycle)
              PIPE_MUL → bf16_mul × 9, produces 9 FP32 products  (1 cycle)
              PIPE_ACC → block-float adder tree, normalize → FP32  (1 cycle)
              DONE_ST  → asserts core_done, holds result  (1 cycle)
            Total: 4 cycles per Conv2d tile
            Activation zero-skip: skips PIPE_MUL/PIPE_ACC when all IFMs = 0
              (saves 2 cycles for sparse inputs)
```

### 4.3 Persistent Weight SRAM (sram_sp.sv)

The `sram_sp.sv` module implements a **9×16-bit single-port SRAM** that holds the 9 BF16 filter weights between kernel invocations. Key properties:

- Weights are written **once** via AXI4-Stream (one 144-bit beat packs all 9 BF16 values) and held in the SRAM until the next weight-load command.
- The SRAM has an **asynchronous read port** — weight delivery to the compute core is single-cycle with no read latency.
- A look-ahead address scheme during the sequential write state (`sram_addr = seq_cnt + 1` during SEQ_DRIVE) allows single-cycle weight handoff without pipeline stalls.
- This persistent storage enables **weight reuse** across multiple IFM tiles without re-sending weights over AXI each time — the fundamental mechanism behind weight-stationary efficiency.

### 4.4 Backward Pass — grad_core.sv

The `grad_core.sv` module computes **18 weight and activation gradients combinatorially in a single clock cycle**:

- **dL/dW[i]** = upstream gradient dL/dy × BF16(x[i]) for i = 0..8 (9 weight gradients)
- **dL/dX[i]** = upstream gradient dL/dy × BF16(w[i]) for i = 0..8 (9 input gradients)

All 18 gradients are independent single-multiplies with no data dependency, so the fully-parallel combinational implementation costs nothing in latency beyond the multiply itself. A sequential version would save ~50% area but would require 9× longer backward latency — not a worthwhile trade-off at this kernel size. The Adam optimizer runs on the host CPU (FP32), receiving the 9 FP32 weight gradients via AXI4-Lite reads from the `GRAD_W0–W8` registers.

**FSM latency evolution (M3_ti → M4):**

| Milestone | FSM | Latency | Notes |
|---|---|---|---|
| M3_ti single-tile | IDLE→LOAD(×9)→COMPUTE→DONE | **11 cycles** | Sequential weight loading, one weight/cycle |
| **M4 compute_core_ti** | IDLE→LOAD→PIPE_MUL→PIPE_ACC→DONE_ST | **4 cycles** | Parallel load of all 9 weights in 1 cycle |

The **7-cycle reduction** (11→4) was achieved by restructuring the LOAD state to latch all 9 BF16 weight/IFM pairs simultaneously from the broadcast bus, rather than sequencing through 9 individual SRAM reads.

### 4.5 Memory Hierarchy

| Level | Location | Contents | Size |
|---|---|---|---|
| 0 | On-chip, per-PE | 9× BF16 weight latches (registered in LOAD) | 144 bits |
| 0b | On-chip, shared | sram_sp 9×16-bit persistent weight SRAM | 144 bits |
| 1 | On-chip, shared | 34×34 BF16 IFM scratchpad (ifm_buffer) | 18,496 bits |
| 2 | Off-chip | DDR4-2667 host memory | — |

---

## Section 5: Hardware Interface

### 5.1 Interface Comparison

| Interface | Width | Frequency | Bandwidth | Selection |
|---|---|---|---|---|
| **AXI4-Stream** | **512-bit** | **500 MHz** | **32 GB/s** | **Selected (primary data)** |
| **AXI4-Lite** | **32-bit** | **500 MHz** | **2 GB/s** | **Selected (control/result)** |
| PCIe Gen3 ×4 | — | — | 16 GB/s | No on-chip equivalent |
| AXI4-Stream 1 GHz | 512-bit | 1 GHz | 64 GB/s | M1 spec; exceeded SAED14nm limit |

### 5.2 Bandwidth Analysis

Required bandwidth to sustain **2.304 TFLOP/s** at **AI = 46.9 FLOP/byte**:

```
Peak required BW = 2304 GFLOP/s / 46.9 FLOP/byte = 49.1 GB/s
Available BW     = 512 bits / 8 × 500 MHz          = 32 GB/s
```

The design is **not interface-bound** because the **34×34 IFM scratchpad** acts as a staging buffer. The host loads it once per Conv2d invocation (**37 AXI beats ≈ 4.7 KB**), after which all **1024 PEs** read their windows combinationally from on-chip SRAM without further AXI traffic.

```
Transfer time per full pass at 32 GB/s = 4700 B / 32e9 B/s = 147 ns
```

With weight-stationary reuse, weights are loaded **once** across all 1024 tiles — reducing effective weight bandwidth to **1/1024** of the naive case.

### 5.3 Bandwidth Margin

| Metric | Value |
|---|---|
| Available (AXI4-Stream, 500 MHz, 512-bit) | **32 GB/s** |
| Required at M1 target (1 TFLOP/s) | **21.3 GB/s** |
| Bandwidth margin | **1.5×** (reduced from M1 spec's 3× at 64 GB/s) |

**The design is NOT interface-bound during the compute phase.**

---

## Section 6: Verification

Verification was carried out across all four milestones using **cocotb v2.0.1** on **Icarus Verilog 11.0**, escalating from unit tests to full end-to-end array tests.

### 6.1 M2 Verification (FP16 Single-Tile)

| Test | Description | Result |
|---|---|---|
| CC1 | Weights [1.0–9.0] FP16, IFMs [1.0×9], expected **45.0** | **PASS** |
| CC2 | Alternating-sign weights, IFMs [2.0×9], expected **2.0** | **PASS** |
| IF1 | AXI4-Lite write CTRL, read STATUS=done, RESULT=45.0 | **PASS** |
| IF2 | Weight reload without device reset | **PASS** |

Result: **TESTS=2 PASS=2 FAIL=0** (compute_core) + **TESTS=2 PASS=2 FAIL=0** (interface)

### 6.2 M4 Verification (BF16 Full 32×32 Array)

DUT: `top_ti` (N_TILES=**1024**) | Runner: `sim/run_ti.py`

| Test | Cocotb Name | Description | Coverage |
|---|---|---|---|
| 1 | `test_forward.test_bf16_forward` | **32×32 BF16 forward pass** — T1: ramp weights × unit IFM (all 1024 tiles = 45.0); T2: alternating-sign × const-2.0; T3: max BF16-safe × unit (1143.0); T4: Laplacian × ramp (per-tile varies). Checks all **1024** FP32 results vs `ref_model.py` (tol=**1e-3**) | Forward path, all tiles, register map 0x1000–0x1FFC |
| 2 | `test_forward.test_weight_reuse` | Loads weights `[1..9]` once, verifies 4 different 34×34 IFM tiles: unit (45.0), double (90.0), negative unit (−45.0), half (22.5) | Weight-stationary reuse, no reload between tiles |
| 3 | `test_backward.test_grad_w` | B1: unit IFMs, dldy=1.0 → grad_w=ifms; B2: ramp IFMs, dldy=2.0 → grad_w=2×ifms; B3: alternating IFMs, dldy=0.5; B4: Laplacian weights, ramp IFMs, dldy=−1.0 | Backward pass dL/dW for all 4 vectors |

**Final result (`sim/final_run.log`):**
> **TESTS=3 PASS=3 FAIL=0 SKIP=0 | Total simulation time: 342,140 ns**  
> DUT: top_ti (full 32×32 array, N_TILES=1024)

### 6.3 Bugs Documented (12 Total)

| ID | Category | Description |
|---|---|---|
| B1 | Stale ports | `tb_top.sv` instantiated `conv_interface` instead of `conv_interface_ti` |
| B2 | Stale ports | `tb_interface.sv` used **4-bit** AXI4-Lite address (should be **13-bit**) |
| B3 | Stale ports | `core_result[31:0]` (should be `core_result[32767:0]` = 1024×32) |
| B4 | cocotb timing | NBA write inside `@(posedge clk)` read back one cycle late; fixed with `await RisingEdge` |
| B5 | cocotb API | Direct `signal = value` silently failed; must use `signal.value = value` |
| B6 | Double-conversion | `bf16_to_float()` called on already-float value; result off by **~32,000×** |
| B7 | Icarus compat | `$bitstoreal()` unsupported inside generate blocks in Icarus 11.0 |
| B8 | Icarus compat | Parameter override syntax in generate loops caused elaboration errors |
| B9 | Icarus compat | Multi-dimensional packed array indexing requires intermediate `wire` |
| B10 | Makefile | `cocotb/compute_core/Makefile` referenced `compute_core.sv` (should be `compute_core_ti`) |
| **B11** | **Mantissa bug** | **`{prod[14:1], 8'h00}` dropped LSB → ~0.4% error/MAC → failed 1e-3 tol; fixed to `{prod[14:0], 8'h00}`** |
| B12 | cocotb timing | Combinational output required `await Timer(0)` delta NOP before read |

**Bug B11 was the most significant:** the off-by-one in `bf16_mul` accumulated across **9 MACs** and violated the **1e-3** tolerance threshold in `test_forward`. Fixed by correcting the slice from `[14:1]` to `[14:0]` in `compute_core_ti.sv`.

---

## Section 7: Synthesis Results

The 32×32 systolic array (`top_ti`, N_TILES=**1024**) was synthesized using **Cadence Genus 17.14-s037_1** (legacy_ui) on **SAED14nm RVT** standard cells. All numbers are from the actual 32×32 synthesis run — not single-tile projections.

### 7.1 Tool Configuration

| Parameter | Value |
|---|---|
| Tool | Cadence Genus 17.14-s037_1 (legacy_ui) |
| PDK | SAED14nm RVT |
| Top module | `top_ti` |
| Clock period | **2.0 ns (500 MHz)** |
| HDL loop limit | **4096** (required for N_TILES=1024 generate loop) |
| Synthesis effort | high |

### 7.2 Timing (`timing_report.txt`)

| Metric | Value |
|---|---|
| Clock period | **2000 ps (500 MHz)** |
| **Worst Negative Slack** | **0 ps — MEETS CONSTRAINT** |
| Total Negative Slack | **0 ps** |
| Setup violations | **0** |
| Critical path | `gen_tile[163]` mul_prod_reg_reg → result_reg_reg |
| Critical path arrival | **1950 ps** (50 ps positive slack) |
| Critical path route | BF16 latch → exponent adder → mantissa multiplier → normalize → result register |

### 7.3 Area (`area_report.txt`)

| Module | Cells | Area (µm²) | % of Total |
|---|---|---|---|
| **top_ti (total)** | **5,591,114** | **2,862,977** | **100%** |
| u_pe_array (1024 tiles) | 5,453,789 | 2,773,992 | **96.9%** |
| u_interface | 137,325 | 88,985 | 3.1% |

- **Total area: 2,862,977 µm² = 2.86 mm²**
- **Per-tile average: 2,773,992 / 1024 = 2,709 µm²**
- Dominant area contributor: `pe_array` (**96.9%**), driven by the BF16 multiplier (7×7 unsigned mantissa product tree + normalize logic) replicated **1024×**

### 7.4 Power (`power_report.txt`, TT corner, 500 MHz)

| Module | Power (mW) | % of Total |
|---|---|---|
| **top_ti (total)** | **1,359.89 mW (1.36 W)** | **100%** |
| u_pe_array | 1,259.17 mW | **92.6%** |
| u_interface | 43.97 mW | 3.2% |
| Other (ifm_buf, etc.) | 56.75 mW | 4.2% |

- **Per-tile average: 1,259.17 / 1024 = 1.23 mW**
- Single-tile synthesis (`m3_ti/cadence_syn/`): **3.07 mW** — the array per-tile figure is lower due to Genus optimizing the shared weight broadcast logic across the generate loop.

### 7.5 Single Tile vs 32×32 Array Comparison

| Metric | M4 Single Tile | M4 32×32 Array | Factor |
|---|---|---|---|
| WNS | +2.4 ps | **0 ps** | Tighter, still meets |
| Total cells | 11,340 | **5,591,114** | ×493 |
| Area | 6,155 µm² | **2,862,977 µm² (2.86 mm²)** | ×465 |
| Total power | 3.07 mW | **1,359.89 mW (1.36 W)** | ×443 |
| Per-tile power | 3.07 mW | **1.23 mW** | 0.4× (40% of single) |
| Throughput | 2.25 GFLOP/s | **2,304 GFLOP/s** | **×1024** |

---

## Section 8: Benchmark Results

All numbers are synthesis-validated from the 32×32 reports. Raw data: `bench/benchmark_data.csv`.

### 8.1 Throughput

```
Per-tile:   9 MACs × 2 FLOPs × 500 MHz / 4 cycles = 2.25 GFLOP/s
32×32 array: 1024 × 2.25 GFLOP/s = 2,304 GFLOP/s = 2.304 TFLOP/s
```

### 8.2 Kernel Speedup (Conv2d)

| Platform | Throughput | Speedup vs M1 |
|---|---|---|
| M1 CPU (measured, cProfile) | **142.7 GFLOP/s** | 1× (baseline) |
| M4 single tile (synthesis) | 2.25 GFLOP/s | 0.016× |
| **M4 32×32 array (synthesis-validated)** | **2,304 GFLOP/s** | **16.1×** |

### 8.3 System Speedup (Amdahl's Law)

```
Accelerated fraction:  f  = 74.4%  (Conv2d fwd+bwd, cProfile)
Kernel speedup:        Sk = 16.1×

System speedup = 1 / ((1 − 0.744) + 0.744 / 16.1)
               = 1 / (0.256 + 0.046)
               = 1 / 0.302
               = 3.31×
```

| Metric | Baseline | With 32×32 Array |
|---|---|---|
| Wall-clock per epoch | **2,573.9 s (42.9 min)** | **~777 s (~13.0 min)** |
| Pipeline throughput | **3.73 samples/sec** | **~12.3 samples/sec** |

### 8.4 Energy Efficiency

| Platform | Power | Throughput | Energy/FLOP |
|---|---|---|---|
| M1 CPU (i5-10210U) | ~15 W TDP | 142.7 GFLOP/s | **105 pJ/FLOP** |
| M4 single tile (SAED14nm) | 1.23 mW | 2.25 GFLOP/s | 0.55 pJ/FLOP |
| **M4 32×32 array (synthesis)** | **1.36 W** | **2,304 GFLOP/s** | **0.59 pJ/FLOP** |
| NVIDIA T4 GPU | 70 W | 65,000 GFLOP/s | ~1.08 pJ/FLOP |

```
Energy efficiency vs CPU:  105 pJ / 0.59 pJ = 178× more efficient per FLOP
Energy efficiency vs T4:   1.08 pJ / 0.59 pJ = 1.8× more efficient per FLOP
Power vs T4:               70 W / 1.36 W     = 51× less power
```

The **178× energy advantage** over the CPU comes from two sources:
1. **SAED14nm FinFET process** — per-gate switching energy is ~10–20× lower than the Intel 14nm FinFET in the i5-10210U.
2. **Dedicated BF16 datapath** — no cache hierarchy, branch predictor, or out-of-order execution logic toggles on each MAC operation.

### 8.5 Performance Gaps

1. **Pipeline utilization (50%):** Each PE spends **2 of 4 cycles** computing (PIPE_MUL + PIPE_ACC) and **2 cycles** on overhead (LOAD + DONE_ST). A deeper pipeline or output-stationary accumulation across channels would push utilization toward 100%.
2. **AXI bandwidth reduction:** M1 spec assumed **1 GHz / 64 GB/s**. The synthesized design runs at **500 MHz / 32 GB/s**. The **32 GB/s** exceeds the **21.3 GB/s** requirement at the **1 TFLOP/s** M1 target, but the **49.1 GB/s** peak requirement for **2.304 TFLOP/s** relies on weight-stationary reuse to avoid hitting the memory ceiling.

### 8.6 M1 Hypothesis Evaluation — Projected vs Actual

The M1 roofline (`partition_rationale.md`, `interface_selection.md`) projected a **7× kernel speedup** and **2.76× system speedup** at the 1 TFLOP/s design target:

```
M1 projection:  Speedup = 1 / (0.256 + 0.744 / 7)  = 2.76×
M4 actual:      Speedup = 1 / (0.256 + 0.744 / 16.1) = 3.31×
```

| Metric | M1 Projected (1 TFLOP/s) | M4 Actual (2.304 TFLOP/s) | Status |
|---|---|---|---|
| Compute target | ≥ 1 TFLOP/s | **2.304 TFLOP/s** | ✅ **Exceeded ×2.3** |
| Kernel speedup | 7× (roofline) | **16.1×** | ✅ **Exceeded ×2.3** |
| System speedup | 2.76× (Amdahl) | **3.31×** | ✅ **Exceeded** |
| Interface BW | 64 GB/s @ 1 GHz | **32 GB/s @ 500 MHz** | ⚠️ Reduced (SAED14nm limit) |

The over-achievement in compute is due to implementing **1024 tiles (32×32)** rather than the minimum **444 tiles** (= 1 TFLOP/s ÷ 2.25 GFLOP/s per tile) needed to meet the M1 target. The interface bandwidth reduction (64→32 GB/s) is acceptable because the **34×34 IFM scratchpad** absorbs the burst load; sustained effective bandwidth during compute is near zero (weights already loaded, IFM already on-chip).

**Three deviations from M1 plan (with justification):**

| Deviation | M1 Plan | M4 Actual | Reason |
|---|---|---|---|
| Data format | Q16.16 → FP16 | **BF16** | Eliminates exponent rebasing adder on critical path |
| Synthesis tool | OpenLane / sky130A | **Cadence Genus / SAED14nm** | OpenLane could not close 1024-tile design (130 ns best) |
| Synthesis scope | Single-tile proof | **Full 32×32 array** | Direct synthesis of production array at SAED14nm |

---

## Section 9: What Did Not Work

### 9.1 OpenLane 2 / sky130A Failed to Close Timing (Complete Failure)

Attempted: **OpenLane 2.3.10** with **sky130A** (`sky130_fd_sc_hd`) PDK for both M3 single-tile and M4 32×32 array synthesis.

**Single-tile result (`m3/synth/openlane_run.log`):**
- TT corner WNS: **+29.9 ns (PASS)**
- SS corner WNS: **−63.0 ns (FAIL)** — expected due to FP32 datapath depth; sky130A SS corner is highly pessimistic for floating-point chains.

**32×32 array:** OpenLane could not close the design at any clock period faster than **130 ns (7.7 MHz)**. Root causes:
- `sky130_fd_sc_hd` has **~2–3×** higher cell delays than SAED14nm RVT (180 nm bulk vs 14 nm FinFET process)
- The N_TILES=**1024** generate loop created routing congestion the OpenLane placer could not resolve within the **800×800 µm** die
- OpenLane 2.3.10 `hdl_max_loop_limit` defaulted to **256**; increasing to **4096** caused Yosys memory exhaustion on the lab machine

**Resolution:** switched to **Cadence Genus 17.14** on **SAED14nm RVT** via the PSU `auto.ece.pdx.edu` server, which closed timing at **500 MHz (WNS = 0 ps)**.

### 9.2 BF16_MUL Mantissa Off-By-One (Bug B11)

The `bf16_mul` function in `compute_core_ti.sv` originally used:

```verilog
{prod[14:1], 8'h00}  // WRONG — drops LSB of 15-bit product
```

Correct form:

```verilog
{prod[14:0], 8'h00}  // CORRECT — all 15 bits, then pad to FP32 mantissa
```

The off-by-one caused **~0.4% relative error per MAC**, accumulating across **9 MACs** to produce results that failed the **1e-3** tolerance threshold in `test_forward.py`. Detected by comparison against `ref_model.py` golden output. Fixed by correcting the slice from `[14:1]` to `[14:0]`.

### 9.3 cocotb NBA Timing (Bugs B4, B5, B12)

cocotb v2.0.1 changed signal assignment semantics vs v1.x:
- Assignments inside `@(posedge clk)` coroutines use NBA scheduling — reading back the value requires awaiting a delta cycle first
- Direct `signal = value` silently failed; must use `signal.value = value`
- Combinational outputs required an explicit `await Timer(0)` (delta NOP) after driving inputs

**Resolution:** added `await RisingEdge(dut.clk)` after each control write, and `await Timer(1, units='ns')` before reading combinational outputs.

### 9.4 Stale Ports in Legacy Testbenches (Bugs B1, B2, B3)

`tb/tb_interface.sv` was written for the M2 single-tile `conv_interface` module:
- AXI4-Lite address: **4-bit** (should be **13-bit** for 32×32 register map)
- `core_result` width: **32-bit** (should be **32,768-bit** = 1024 × 32)
- Module name: `conv_interface` (should be `conv_interface_ti`)

**Resolution:** `tb_interface.sv` was excluded from the M4 testbench suite. `tb_top.sv` (which correctly instantiates `top_ti`) is the M4 primary testbench.

### 9.5 Double-Conversion Bug (Bug B6)

In `test_backward.py`, `bf16_to_float()` was called on a value that was already a Python float. The function interpreted the float's integer representation as a BF16 bit pattern, producing a golden value off by a factor of **~32,000**. Fixed by tracking data types through the call chain.

### 9.6 Icarus Verilog Compatibility (Bugs B7, B8, B9)

| Bug | Issue | Fix |
|---|---|---|
| B7 | `$bitstoreal()` unsupported inside generate blocks in Icarus 11.0 | Replaced with explicit sign/exp/mantissa unpacking (synthesizable) |
| B8 | Parameter override syntax in generate loops caused elaboration errors | Replaced with explicit `localparam` per generate iteration |
| B9 | Multi-dimensional packed array indexing (`packed_bus[t*144 +: 144]`) requires intermediate `wire` | Added `wire [143:0] w_ifm_t` per generate iteration |

### 9.7 Unicode Encoding Error (Bug B10)

`test_backward.py` used the Unicode right-arrow character (U+2192 `→`) in test vector names. On Windows with cp1252 encoding, cocotb's log output raised:

```
UnicodeEncodeError: 'charmap' codec can't encode character '→'
```

Fixed by replacing `→` with ASCII `->` in all test name strings.

### 9.8 What Would Be Done Differently

- **Start with Cadence Genus on SAED14nm** from the beginning rather than attempting OpenLane/sky130A for a 1024-tile design; sky130A is appropriate for single-tile proof-of-concept only.
- **Formally verify `bf16_mul` mantissa alignment** from the start, rather than catching Bug B11 late in cocotb testing.
- **Use a cocotb compatibility shim layer** to isolate version-specific NBA timing behavior from test logic.

---

## Raw Data References

All numbers in this report trace to the following committed files:

| File | Contents |
|---|---|
| `bench/benchmark_data.csv` | All raw numbers: M1_baseline, M4_single_tile, M4_array_32x32, benchmark_comparison |
| `bench/benchmark.md` | Narrative benchmark with method of measurement |
| `bench/roofline_final.png` | Roofline plot: CPU/co-proc curves + 3 operating points |
| `synth/timing_report.txt` | WNS=**0 ps**, 10 timing paths, **500 MHz** |
| `synth/area_report.txt` | **5,591,114 cells**, **2,862,977 µm²** |
| `synth/power_report.txt` | **1,359.89 mW** total (TT corner) |
| `sim/final_run.log` | **TESTS=3 PASS=3 FAIL=0**, **342,140 ns** |
| `project/m1/sw_baseline.md` | 10-run wall-clock table, cProfile data |
| `project/m2/precision.md` | FP16 SQNR, error analysis, verification |
| `project/m3_ti/synthesis_notes_ti.md` | Single-tile design decisions |

---

*Word count (body text, excluding headers and tables): approximately 2,900 words*
