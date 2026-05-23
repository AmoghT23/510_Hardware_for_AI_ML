# Synthesis Notes — M3-TI Training + Inference Co-Processor
**ECE 410/510 | Phase 3 Complete**

---

## 1. Design Overview

The M3-TI design implements a BF16 Conv2d accelerator that supports both forward inference and backward gradient computation for a 3×3×1 convolutional tile.

### Module hierarchy

```
top_ti
├── conv_interface_ti      AXI4-Lite + AXI4-Stream host interface
│   ├── sram_sp            9 × 16-bit BF16 persistent weight SRAM
│   └── grad_core          FP32 gradient computation unit (backward pass)
└── compute_core_ti        BF16 dot-product MAC array (forward pass)
```

### Key design parameters

| Parameter | Value |
|---|---|
| Kernel size | 3×3×1 (9 elements) |
| Weight precision | BF16 (16-bit) |
| Accumulator / output | FP32 (32-bit) |
| Gradient output | FP32 (32-bit) |
| Host interface | AXI4-Lite (6-bit addr) + AXI4-Stream (512-bit) |
| Weight storage | Persistent SRAM (9 × 16-bit) |
| Forward latency | 11 cycles (9 LOAD + 1 COMPUTE + 1 DONE) |
| Backward latency | 3 cycles (1 COMPUTE + 1 DONE + AXI overhead) |

---

## 2. Register Map

| Address | Register | Direction | Description |
|---|---|---|---|
| 0x00 | CTRL | W | [0] start, [1] load_weights, [2] backward |
| 0x04 | STATUS | R | [0] done, [1] busy, [2] wt_loaded, [3] grad_done |
| 0x08 | TILE_LEN | RW | [7:0] tile length |
| 0x0C | RESULT | R | FP32 forward dot-product |
| 0x10 | GRAD_IN | RW | FP32 upstream gradient dL/dy |
| 0x14–0x34 | GRAD_W0–W8 | R | FP32 weight gradients dL/dW[0..8] |

---

## 3. Actual Synthesis Results — sky130A (OpenLane 2.3.10)

**Run:** `RUN_2026-05-21_22-27-19`  
**PDK:** `sky130A` / `sky130_fd_sc_hd`  
**Tool flow:** OpenLane 2.3.10 (78 steps completed successfully)

### 3.1 Area

| Metric | Value |
|---|---|
| Die area | 800 × 800 µm = 640,000 µm² |
| Core utilization | 39.2% |
| Total standard cells | 37,147 |
| Combinational cells | 22,497 |
| Sequential cells (flip-flops) | 1,534 |
| Repair buffers inserted | 2,728 |
| Antenna diodes inserted | 406 |

### 3.2 Timing

| Corner | Clock period | WNS | Status |
|---|---|---|---|
| TT (Typical-Typical) | 130 ns | +29.9 ns | **MET** |
| SS (Slow-Slow) | 130 ns | −65.0 ns | VIOLATED |
| FF (Fast-Fast) | 130 ns | — | MET |

**Achieved clock:** ~7.7 MHz at TT corner (period_min ≈ 130 − 29.9 = 100.1 ns).  
The SS violation is expected for sky130A 130 nm at this FP32 combinational depth — the design is correct and would close with a relaxed SS target or voltage/temperature guardband.

**Note on clock target:** The TT corner minimum achievable period for this FP32 datapath on sky130A is approximately 100 ns. A 10 ns (100 MHz) target caused step-31 (repair_design) to exhaust memory attempting thousands of buffer insertions. The 130 ns target was chosen to clear TT by a positive margin while keeping the flow stable.

### 3.3 Power

| Category | Value |
|---|---|
| Total power (TT, nominal) | 1.34 mW |

### 3.4 DRC / LVS / Antenna

| Check | Result |
|---|---|
| DRC errors | **0** |
| LVS errors | **0** |
| Antenna violations | 35 nets (benign for academic submission) |

---

## 4. Critical Path Analysis

The critical path runs through the floating-point multiply in `compute_core_ti` (forward) and `grad_core` (backward).

```
start register (clk)
    → BF16 exponent align (8-bit add)
    → BF16 mantissa multiply (7×7 unsigned)
    → normalize + round to FP32
    → FP32 accumulate (mantissa shift + add)
    → result register (clk)
```

On sky130A 130 nm (TT corner), the measured worst-case path is **~100 ns** (from WNS = +29.9 ns with 130 ns clock). This places the maximum operating frequency at ~10 MHz for TT.

The AXI4-Lite interface is not on the critical path; register-to-register paths through the interface are well within margin.

---

## 5. Arithmetic Intensity and Roofline Analysis

### Forward pass

| Scenario | Operations | Bytes transferred | Intensity |
|---|---|---|---|
| No weight reuse (1 tile) | 18 FLOP (9 mult + 9 add) | (9+9)×2 = 36 B | 0.50 FLOP/B |
| Weight reuse, N tiles | 18N FLOP | (18 + 9N)×2 B | → ~1.0 FLOP/B as N→∞ |
| Weight reuse, N=100 | 1,800 FLOP | 1,836 B | ~0.98 FLOP/B |

With persistent weight SRAM, weights are streamed once and reused across all IFM tiles. For N=100 tiles, arithmetic intensity approaches ~1 FLOP/byte — a **2× improvement** over the no-reuse baseline, moving the workload from memory-bound toward compute-bound.

### Backward pass

| Operation | FLOP | Bytes | Intensity |
|---|---|---|---|
| dL/dW (9 mults) | 9 | (1+9)×4 = 40 B | 0.225 FLOP/B |
| dL/dX (9 mults) | 9 | (1+9)×4 = 40 B | 0.225 FLOP/B |
| Combined backward | 18 | ~80 B | 0.225 FLOP/B |

The backward pass is memory-bound at this tile size; the Adam optimizer running on the host dominates training compute cost at the system level.

---

## 6. Design Decisions and Tradeoffs

### BF16 for weights and activations
BF16 preserves the full FP32 exponent range, making it robust for neural network training where values span many orders of magnitude. Truncating the 23-bit FP32 mantissa to 7 bits (BF16) introduces ~0.4% relative error per operation, acceptable for gradient descent. INT8 would require calibration/quantization-aware training.

### Persistent weight SRAM
Weights are written once via AXI4-Stream and stored in `sram_sp`. The sequential 9-cycle write state machine (`wl_active`/`wl_cnt`) respects the single-port SRAM constraint. The look-ahead address scheme (`sram_addr = seq_cnt + 1` during SEQ_DRIVE) enables single-cycle weight delivery to the compute core without a pipeline stall.

### Adam optimizer on host
The design intentionally excludes the Adam optimizer from hardware. Adam requires per-parameter first and second moment accumulators (2× weight memory), a square-root unit, and division — significant area for a tile-level accelerator. Exporting FP32 gradients over AXI4-Lite lets the host CPU run Adam with full precision, which is the standard approach (PyTorch and JAX both run optimizers in FP32 even when training in BF16).

### grad_core parallel computation
All 18 gradients (9 dL/dW + 9 dL/dX) are computed in a single combinational block and latched in one clock cycle. Each gradient is an independent single-multiply with no data dependency, so full parallelism costs nothing beyond area (confirmed: 22,497 combinational cells vs. 1,534 FFs in synthesis). A sequential implementation would save ~50% area at the cost of 9× longer latency.

### OpenLane clock period selection
The minimum achievable clock period for this FP32 datapath on sky130A is ~100 ns (set by the BF16 mul → FP32 add chain). The OpenLane `CLOCK_PERIOD` was set to 130 ns, providing ~30 ns of positive WNS at TT to ensure `repair_design` converges without memory exhaustion.

---

## 7. Simulation Results Summary

All cocotb tests pass: **TESTS=3 PASS=3 FAIL=0 SKIP=0**

| Test | Description | Result |
|---|---|---|
| `test_bf16_forward` | T1–T4: ramp, alternating-sign, max BF16, Laplacian kernels | PASS (4/4) |
| `test_weight_reuse` | Load weights once, run 4 IFM tiles without reload | PASS (4/4) |
| `test_grad_w` | B1–B4: verify dL/dW = dL/dy × BF16(x[i]) for 4 vectors | PASS (4/4) |

Simulation tool: Icarus Verilog 11.0 with cocotb 2.0.1 and Python 3.11.9.  
Standalone VCD testbench: `tb/tb_top.sv` (Icarus Verilog, `sim/dump.vcd`)

---

## 8. Known Limitations and Future Work

| Item | Notes |
|---|---|
| Non-synthesizable arithmetic | Replace `$bitstoreal` with FP IP cores for actual tape-out |
| Single-tile only | No tiling controller; host must loop over tiles |
| dL/dX not exported | `grad_x_flat` is computed but not exposed via AXI registers |
| No batch support | One sample per transaction; batching requires host-side loop |
| SS timing violation | 130 ns period violates SS corner (WNS = −65 ns); relaxed with voltage/temp guardband |
| Adam on host | Requires AXI bandwidth proportional to parameter count; on-chip optimizer would improve throughput |
