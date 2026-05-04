# ECE 410/510 — Conv2d Co-processor Chiplet, Milestone 2

## Overview

Synthesizable RTL for a 3 × 3 Conv2d output-pixel accelerator targeting 1 GHz.

| Module | File | Description |
|--------|------|-------------|
| `compute_core` | `rtl/compute_core.sv` | FP16 MAC engine — FP16×FP16→FP32 dot product, FP16 output |
| `conv_interface` | `rtl/interface.sv` | AXI4-Stream 512-bit + AXI4-Lite 32-bit wrapper |

Both modules share a single clock domain and synchronous active-low reset.

---

## Deviation from Milestone 1 Plan

**Arithmetic format changed: Q16.16 → FP16 mixed-precision.**

The M1 design specification used Q16.16 signed fixed-point (32-bit words).
After analysis, the M2 implementation adopts IEEE 754 FP16 (half-precision) inputs and
outputs with an FP32 (single-precision) accumulator, matching the industry-standard
mixed-precision pipeline used by NVIDIA Tensor Cores and Apple Neural Engine.

Motivations for the change:

1. **Bandwidth**: FP16 is 16 bits vs 32 bits for Q16.16, doubling the number of values
   per AXI4-Stream 512-bit beat (32 vs 16 values per beat).
2. **Dynamic range**: FP16's floating exponent handles the full ResNet18 weight/activation
   magnitude range without any fixed-point scaling convention.
3. **Tapeout alignment**: FP16 is the dominant inference format; M3/M4 tapeout can use
   standard FP16 MAC hard macros directly.
4. **Precision**: FP32 accumulation prevents catastrophic cancellation across the deepest
   ResNet18 tiles (4,608 MACs in Layer 4). See `precision.md` for full error analysis.

---

## Directory Structure

```
m2/
├── rtl/
│   ├── compute_core.sv       FP16 MAC engine (IDLE→LOAD→COMPUTE→DONE FSM)
│   └── interface.sv          AXI4-Stream + AXI4-Lite wrapper (conv_interface)
├── tb/
│   ├── tb_compute_core.sv    Two FP16 dot-product test vectors
│   └── tb_interface.sv       Four AXI4-Lite / AXI4-Stream transaction tests
├── sim/
│   ├── run.ps1               PowerShell script: compile → simulate → launch GTKWave
│   ├── compute_core_run.log  Captured stdout — 2/2 PASS
│   ├── interface_run.log     Captured stdout — 4/4 PASS
│   └── waveform.png          Digital waveform plot (key signals from interface TB)
├── precision.md              FP16 mixed-precision choice rationale and error analysis
└── README.md                 This file
```

---

## Prerequisites

| Tool | Minimum version | Check |
|------|----------------|-------|
| Icarus Verilog | 10.3 | `iverilog -V` |
| GTKWave | any | `gtkwave --version` |

---

## Reproducing the Simulations

### One-step (recommended)

From the `m2/` directory in PowerShell:

```powershell
.\sim\run.ps1
```

The script compiles both modules, runs both simulations (saving logs to `sim/`),
and launches GTKWave with the generated VCD files.

### Manual steps

#### Compute Core

```
iverilog -g2012 -o sim/cc_sim.out rtl/compute_core.sv tb/tb_compute_core.sv
vvp sim/cc_sim.out
```

Expected output:
```
=== tb_compute_core (FP16) ===
  T1 [ramp-w / unit-x / expect 45.0] : PASS  got=0x51a0
  T2 [alt-sign-w / 2x / expect 2.0]  : PASS  got=0x4000
---
PASS — all 2 tests passed.
```

To save to log:
```
vvp sim/cc_sim.out | tee sim/compute_core_run.log
```

#### Interface

```
iverilog -g2012 -o sim/if_sim.out rtl/interface.sv tb/tb_interface.sv
vvp sim/if_sim.out
```

Expected output:
```
=== tb_interface ===
  T1 [AXI4-Lite write TILE_LEN=9] : PASS  bresp=2'b00 (OKAY)
  T2 [AXI4-Lite read  TILE_LEN]  : PASS  rdata=0x00000009  rresp=OKAY
  T3 [AXI4-Stream beat]           : PASS  tready=1 (beat accepted)
  T4 [CTRL start pulse]           : PASS  core_start pulsed high
---
PASS — all 4 tests passed.
```

---

## Register Map (AXI4-Lite, 4-bit address)

| Addr | Name     | Access | Bits   | Description                             |
|------|----------|--------|--------|-----------------------------------------|
| 0x00 | CTRL     | W      | [0]    | start — pulse `core_start` one cycle (auto-clears) |
|      |          |        | [1]    | sw_rst — software reset (auto-clears)   |
| 0x04 | STATUS   | R      | [0]    | done — latched from `core_done`; clears on read |
|      |          |        | [1]    | busy — high when core is not idle       |
| 0x08 | TILE_LEN | RW     | [7:0]  | Number of MAC operations per tile       |
| 0x0C | RESULT   | R      | [15:0] | FP16 output pixel (latched on done); bits [31:16] read as zero |

---

## AXI4-Stream Format

Each 512-bit beat carries **32 × 16-bit FP16 values**.  
`TREADY` is held permanently high (no backpressure).  
`TLAST` marks the final beat of a tile packet.

---

## Design Notes

- **`interface` keyword**: SystemVerilog reserves `interface`; the module is named
  `conv_interface` throughout.
- **Icarus Verilog compatibility**: `automatic` variables inside `always_ff` and
  bit-select on ternary expressions are unsupported — both are avoided via module-level
  combinational wires (`wr_addr`, `wr_data`).
- **FP32 accumulator**: prevents precision loss (catastrophic cancellation) across deep
  tiles up to 4,608 MACs (ResNet18 Layer 4). See `precision.md` for full error analysis.
- **Simultaneous AW/W arrival**: The write path handles both same-cycle and separate-cycle
  AW/W arrival via dual latch registers with a combinational mux selecting the active source.
- **Denormals**: flushed to zero on FP16 input; acceptable for inference with trained weights.
