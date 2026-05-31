# Remaining Tasks Before M4
**ECE 410/510 — Project: Anemia Detection Accelerator**

---

## 1. Scale `compute_core_ti` to a 4×4 Systolic PE Array

Replace the single 3×3×1 tile in `rtl/compute_core_ti.sv` with a 4×4 mesh of 16 parallel
BF16 MAC units sharing a weight-broadcast bus from `sram_sp`. Each PE computes one dot
product; all 16 execute in parallel using different input activation windows but the same
weight tile. At 500 MHz, a 16-PE array delivers:

```
16 tiles × 9 MACs/tile × 2 FLOP/MAC × 500 MHz = 144 GFLOP/s
```

This matches the measured SW baseline (142.7 GFLOP/s) without changing the clock or
interface. Required changes: parameterize `KERNEL_H`, `KERNEL_W`, `N_PE` in
`compute_core_ti.sv`; add `pe_row` / `pe_col` generate loops; widen `ifm_data` AXI
stream to carry 16×9 BF16 values per beat (2,304 bits) or tile-stream over multiple beats.

---

## 2. Fix the Output MUX on the Critical Path

The Cadence Genus timing report (`cadence_syn/timing.rpt`, path 1) shows
`SAEDRVT14_MUX2_MM_0P5` at **1,940 ps arrival** with only **10 ps to the capture FF
setup check** — this MUX is the last gate on the 2 ns critical path with 2 ps slack.
Moving the output-select logic before the accumulate stage (early-select pattern)
removes the MUX from the timing path and adds ~50 ps of slack. This enables pushing
the synthesis clock target from 2 ns to ~1.9 ns, improving peak throughput to ~526 MHz
(+5.2%) at zero area cost. File: `rtl/compute_core_ti.sv`, PIPE_ACC output assignment
block (`result_reg <= ...`).

---

## 3. Add AXI4-Stream Burst DMA for Activation Tiles

The current `interface_ti.sv` loads 9 BF16 activation values per AXI4-Stream transaction
(144 of 512 bits used per beat). A burst DMA controller using the full 512-bit bus carries
32 BF16 values per beat, loading 3 complete input tiles per transaction instead of 1.
This reduces the ratio of AXI load beats to compute cycles from 9:1 to 3:1, improving
effective throughput by ~3× without scaling the MAC array. Implementation: add an
`axi4s_burst_rx` sub-module in `rtl/interface_ti.sv` with a 32-entry BF16 tile buffer;
decode tile boundaries from `TILE_LEN` and route to the appropriate PE input registers.

---

## 4. Add Structured Sparsity to `compute_core_ti` (M4 Extension)

The M3-TI design computes all 9 MACs unconditionally regardless of whether any weight or
activation value is zero. M4 adds **structured sparsity support** directly in the PE
datapath of `rtl/compute_core_ti.sv`. Each PE now contains a zero-check comparator on
its BF16 weight input: if `weight_buf[i] == 16'h0000` (BF16 zero), the multiply is gated
and the accumulator holds, consuming no switching power for that MAC slot.

At a sparsity ratio σ, the effective compute throughput scales as:

```
Effective throughput = peak_throughput × (1 − σ)
At σ = 0.50 (50% weight sparsity):
  Effective compute = 9 GFLOP/s × 0.50 = 4.5 GFLOP/s active MACs
  Power reduction   ≈ 50% of dynamic switching power in compute_core_ti
```

For ResNet18 Conv2d layers, post-training magnitude pruning at 50% sparsity introduces
less than 0.3% validation accuracy loss on ImageNet-class tasks (Han et al., 2016),
making this a low-risk throughput-for-energy trade-off. The zero-check adds one
comparator per PE (≈ 16 LUTs per PE, negligible vs. the BF16 multiplier at ~180 LUTs)
and does not lengthen the critical path.

---

## 5. Host-Side Weight Zeroing for Structured Sparsity

The hardware sparsity support (Task 4) requires that zero-valued weights are present in
the weight stream — the PE zero-check then gates them. The host CPU is responsible for
generating these sparse weight patterns before streaming to the accelerator. M4 adds a
host-side **magnitude pruning pass** that runs once at model initialisation:

```python
# Host-side: prune weights below threshold to exact BF16 zero before streaming
def prune_to_bf16_zero(weight_tensor, sparsity_ratio=0.50):
    threshold = torch.quantile(weight_tensor.abs(), sparsity_ratio)
    weight_tensor[weight_tensor.abs() < threshold] = 0.0
    return weight_tensor.to(torch.bfloat16)  # zero → BF16 0x0000 exactly
```

The pruned BF16 tensors are streamed over AXI4-Stream exactly as before — no change
to the RTL interface is required. Because `sram_sp` stores the weights persistently,
the pruning cost is paid once per model load, not once per inference. At 50% sparsity,
the host streams half as many non-zero values but the AXI packet length is unchanged
(zeros still occupy beats); the energy saving comes entirely from the gated MACs on the
hardware side, not from reduced bus traffic. A future compressed-sparse-row format in
`interface_ti.sv` could eliminate zero beats from the stream entirely and further
improve effective bandwidth, but this is left as a post-M4 optimisation.

---

## 6. Technology Node Migration: OpenLane sky130A (130 nm) → SAED 14 nm (Cadence Genus)

The M3 synthesis used **OpenLane 2.3.10 with the sky130A PDK** (130 nm planar CMOS).
The sky130A run produced a working design (TT WNS = +29.9 ns, DRC/LVS clean) but the
process node limits achievable frequency to ~10 MHz — far below the 500 MHz needed for
the 1 TFLOP/s target. The SS corner failed by −63 ns, inherent to 130 nm FP arithmetic
datapaths and impossible to close by RTL optimisation alone.

M4 migrates synthesis to **Cadence Genus 17.14 targeting the SAED 14 nm RVT library**
(`saed14rvt_tt0p8v25c`) on the university remote machine (`auto.ece.pdx.edu`). The
migration required:

1. Converting the OpenLane `config.json` flow to a Genus `run_genus.tcl` script
   (`cadence_syn/run_genus.tcl`).
2. Replacing sky130A-specific LIB/LEF references with SAED 14 nm equivalents.
3. Setting a 2 ns clock period (500 MHz target) in `cadence_syn/constraints.sdc`.
4. Resolving three RTL compatibility issues: removing `$bitstoreal`/`$realtobits`
   simulation primitives that Genus rejects, replacing with synthesisable FP unpack
   and multiply macros.

Results (`cadence_syn/qor.rpt`, synthesised May 27 2026):

| Metric | sky130A / OpenLane | SAED 14 nm / Cadence Genus |
|---|---|---|
| Clock period | 130 ns (7.7 MHz) | **2 ns (500 MHz)** |
| TT WNS | +29.9 ns (PASS) | **+2.4 ps (PASS)** |
| SS timing | −63.0 ns (FAIL — inherent) | **PASS (all corners)** |
| Total cells | 37,147 | **11,340** |
| Total power (TT) | 1.34 mW | **3.07 mW** (higher frequency) |
| Peak compute | ~0.07 GFLOP/s | **9 GFLOP/s** |

The 65× frequency improvement (7.7 MHz → 500 MHz) is the primary enabler for the
1 TFLOP/s design target and for closing timing at the SS process corner.
