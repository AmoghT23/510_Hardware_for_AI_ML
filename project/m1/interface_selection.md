# Interface Selection
**ECE 410/510 — M1 Deliverable**
**Project:** Anemia Detection — ResNet18 Conv2d Hardware Accelerator

---

## 1. Host Platform

The assumed host platform is an **FPGA SoC** (e.g., Xilinx Zynq UltraScale+ or
Intel Cyclone V SoC). The FPGA SoC provides:
- A hard ARM Cortex-A processor running the software stack (feature extraction,
  AttentionFusion MLP, optimizer, data loading)
- Programmable logic fabric that houses the Conv2d systolic-array accelerator chiplet
- On-chip AXI interconnect fabric natively supported in the SoC

This is the natural host for a chiplet-style accelerator in an academic prototyping
context — it avoids the full ASIC bring-up cycle while still exercising a realistic
hardware interface at the correct bandwidth scale.

---

## 2. Interface Chosen

**AXI4-Stream**

Selected from the project interface table: SPI, I²C, AXI4-Lite, **AXI4-Stream**,
PCIe, UCIe.

### Rationale

AXI4-Stream is a unidirectional, burst-capable, handshake-based interface (TVALID /
TREADY / TDATA) with no address overhead. It is the standard interface for
high-throughput datapath streaming in FPGA SoC designs — exactly the access pattern
required for feeding a systolic array with continuous tile streams of weights and
input feature maps. Unlike AXI4-Lite (which is register-mapped and low-bandwidth),
AXI4-Stream has no transaction overhead per beat, making it efficient for the
sustained, pipelined data delivery the Conv2d kernel demands.

---

## 3. Bandwidth Requirement Calculation

From the profiling and roofline analysis (`ai_calculation.md`, `partition_rationale.md`):

**Target operating point:** 8 TFLOP/s (proposed HW accelerator peak)

**Data transferred per batch (no DRAM reuse, float32):**

```
Weights (all Conv2d layers)     = 45,648,896 bytes
Input activations               = 23,986,688 bytes
Output activations              = 23,592,960 bytes
Total per image (fwd pass)      = 93,228,544 bytes

Per batch (batch_size = 32):
  Bytes_batch = 32 × 93,228,544 = 2,983,313,408 bytes  ≈ 2.98 GB
```

**Time per batch at target throughput:**

```
FLOPs per batch (fwd only) = 32 × 3,629,023,232 = 116,128,743,424 FLOP
                           ≈ 116.1 GFLOP

Time per batch = FLOPs / Peak throughput
               = 116.1 GFLOP / 8 TFLOP/s
               = 0.01451 s
```

**Required interface bandwidth:**

```
Required BW = Bytes per batch / Time per batch
            = 2,983,313,408 bytes / 0.01451 s
            = 205.6 GB/s
```

Using forward + backward (factor ×2.5 for gradient tensors):

```
Required BW (fwd+bwd) = 205.6 × 2.5 ≈ 514 GB/s
```

This is the **on-chip** bandwidth requirement (SRAM to systolic array), which is
met by the 512 GB/s on-chip SRAM design target.

**For the host-to-chiplet interface specifically** (transferring input feature maps
and weights from host DRAM to the accelerator), the requirement is the input-side
only:

```
Input BW = (Weights + Input activations) per batch / Time per batch
         = (45,648,896 + 32 × 23,986,688) bytes / 0.01451 s
         = (45,648,896 + 767,574,016) / 0.01451
         = 813,222,912 / 0.01451
         = 56.0 GB/s  required at the host interface
```

---

## 4. Interface Rated Bandwidth vs. Required Bandwidth

| Interface | Rated Bandwidth | Required (host→accel) | Sufficient? |
|---|---|---|---|
| SPI (50 MHz) | ~0.006 GB/s | 56.0 GB/s | No — 9,000× short |
| I²C (1 MHz) | ~0.0001 GB/s | 56.0 GB/s | No — way off |
| AXI4-Lite | ~1–4 GB/s | 56.0 GB/s | No — register overhead |
| **AXI4-Stream (128-bit, 500 MHz)** | **~8 GB/s** | 56.0 GB/s | **Partially** — see below |
| AXI4-Stream (512-bit, 1 GHz) | ~64 GB/s | 56.0 GB/s | Yes |
| PCIe 4.0 x16 | ~32 GB/s | 56.0 GB/s | No — 1.75× short |
| UCIe (die-to-die) | ~460 GB/s | 56.0 GB/s | Yes |

### Selected configuration: AXI4-Stream at 512-bit width, 1 GHz

```
Rated BW = (512 bits / 8) × 1 GHz
         = 64 bytes × 1×10⁹
         = 64 GB/s
```

**64 GB/s > 56.0 GB/s required → not interface-bound at the host interface.**

---

## 5. Bottleneck Status on the Roofline

The Conv2d kernel has arithmetic intensity **AI = 46.9 FLOP/byte**.

The proposed HW accelerator roofline:
- Peak compute: 8 TFLOP/s
- On-chip bandwidth: 512 GB/s
- Ridge point: 8,000 / 512 = **15.6 FLOP/byte**

Since AI = 46.9 > ridge = 15.6, the kernel is **compute-bound** on the
accelerator — the host interface at 64 GB/s feeds data faster than the systolic
array consumes it at full load, so the design is **not interface-bound**.

To verify: at AI = 46.9 FLOP/byte and interface BW = 64 GB/s:

```
Interface-limited perf = 64 GB/s × 46.9 FLOP/byte = 3,001.6 GFLOP/s ≈ 3.0 TFLOP/s
Compute-limited perf   = 8.0 TFLOP/s
```

Since compute-limited perf (8 TFLOP/s) > interface-limited perf (3.0 TFLOP/s),
the design **is interface-bound** at this host interface width. To eliminate the
interface bottleneck, either:

1. Use **UCIe** (460 GB/s): interface-limited perf = 460 × 46.9 = 21.6 TFLOP/s > 8 TFLOP/s ✓
2. Widen to **AXI4-Stream at 4096-bit, 1 GHz** (~512 GB/s effective) ✓

**Recommended:** UCIe die-to-die link for the chiplet context (eliminates the
interface bottleneck entirely and is consistent with the chiplet architecture
assumed in the system diagram).

---

## 6. Summary

| Item | Value |
|---|---|
| Host platform | FPGA SoC (Xilinx Zynq UltraScale+ class) |
| Interface selected | AXI4-Stream (primary), UCIe for chiplet boundary |
| Required host→accel BW | 56.0 GB/s (forward pass) |
| AXI4-Stream rated BW (512b, 1GHz) | 64 GB/s |
| UCIe rated BW | ~460 GB/s |
| Interface-bound with AXI4-Stream? | Yes — limits to 3.0 TFLOP/s effective |
| Interface-bound with UCIe? | No — 21.6 TFLOP/s headroom > 8 TFLOP/s target |
| Kernel bound type (on accelerator) | Compute-bound (AI 46.9 > ridge 15.6) |
