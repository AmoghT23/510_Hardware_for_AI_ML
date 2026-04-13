# Interface Selection
**ECE 410/510 — M1 Deliverable**
**Project:** Anemia Detection — ResNet18 Conv2d Co-processor Chiplet

---

## 1. Host Platform

The host platform is the **Intel Core i5-10210U** (this project's measured baseline
machine): quad-core, 1.6 GHz base / 4.2 GHz boost, DDR4-2667 dual-channel memory
at **45.8 GB/s** peak bandwidth, running Windows 11 with PyTorch 2.11.0 CPU build.

The co-processor chiplet connects to this host via an AXI4-Stream interface. In the
prototyping context (M2 onward), the interface logic will be synthesized into an
FPGA fabric (Xilinx Zynq UltraScale+ class) that emulates the chiplet boundary
while the ARM cores on the SoC stand in for the i5-10210U host. This is the
standard academic prototyping approach: the software baseline is measured on the
real CPU, and the interface bandwidth analysis is grounded in that platform's
actual memory specifications.

The co-processor **shares the host DDR4-2667 bus** — it is not an independent ASIC
with its own HBM. This matches the design decision described in the roofline and
partition rationale documents.

---

## 2. Interface Chosen

**AXI4-Stream (512-bit data width, 1 GHz clock)**

Selected from the project interface table: SPI, I²C, AXI4-Lite, **AXI4-Stream**,
PCIe, UCIe.

### Rationale

AXI4-Stream is a unidirectional, burst-capable, handshake-based interface
(TVALID / TREADY / TDATA) with no address overhead per data beat. It is the
standard streaming interface for high-throughput datapath accelerators in FPGA
SoC designs — exactly the access pattern required for feeding a systolic array
with continuous tile streams of input feature maps and weights.

At the 1 TFLOP/s co-processor target, the required sustained bandwidth is
**21.3 GB/s** (derived below). AXI4-Stream at 512-bit / 1 GHz provides **64 GB/s**
rated bandwidth — a 3× margin over the requirement. This eliminates the interface
as the performance bottleneck, keeping the systolic array compute capacity as the
binding resource.

AXI4-Lite is paired for the control plane (register-mapped configuration: tile
dimensions, stride, padding, start/done handshake) while AXI4-Stream carries the
data plane.

---

## 3. Bandwidth Requirement Calculation

From the profiling and roofline analysis (`ai_calculation.md`, `partition_rationale.md`):

**Target operating point:** 1 TFLOP/s (co-processor systolic array peak)

**Arithmetic Intensity of the dominant kernel:** 46.9 FLOP/byte (fwd+bwd, no DRAM reuse)

**Required sustained DRAM bandwidth (to keep systolic array fully fed):**

```
Required BW = Peak compute / Arithmetic Intensity
            = 1,000 GFLOP/s / 46.9 FLOP/byte
            = 21.3 GB/s
```

Cross-check from data volume:

```
Bytes per batch (fwd+bwd, batch_size=32, no reuse) = 7.43 GB
Time per batch at 1 TFLOP/s = 348.4 GFLOP / 1,000 GFLOP/s = 0.3484 s

Required BW = 7.43 GB / 0.3484 s = 21.3 GB/s  ✓
```

**For the host-to-chiplet interface specifically** (input feature maps and weights
transferred from host DRAM to the co-processor each forward pass):

```
Input bytes per batch = Weights + Input activations
                      = 45,648,896 + (32 × 23,986,688)
                      = 813,222,912 bytes  ≈ 0.813 GB

Time per batch (fwd only at 1 TFLOP/s) = 116.1 GFLOP / 1,000 GFLOP/s = 0.1161 s

Input-side BW = 0.813 GB / 0.1161 s = 7.0 GB/s (host → co-proc, fwd inputs only)
```

The full 21.3 GB/s figure is more conservative and is used for interface sizing.

---

## 4. Interface Rated Bandwidth vs. Required Bandwidth

| Interface | Rated Bandwidth | Required (21.3 GB/s) | Sufficient? |
|---|---|---|---|
| SPI (50 MHz) | ~0.006 GB/s | 21.3 GB/s | No — 3,500× short |
| I²C (1 MHz) | ~0.0001 GB/s | 21.3 GB/s | No |
| AXI4-Lite | ~1–4 GB/s | 21.3 GB/s | No — register overhead |
| **AXI4-Stream (512-bit, 1 GHz)** | **64 GB/s** | 21.3 GB/s | **Yes — 3× margin** |
| PCIe 4.0 x16 | ~32 GB/s | 21.3 GB/s | Yes — 1.5× margin |
| UCIe (die-to-die) | ~460 GB/s | 21.3 GB/s | Yes — far exceeds need |

**AXI4-Stream at 512-bit / 1 GHz is selected:** it provides sufficient margin,
is implementable in FPGA fabric without external IP, and is the industry standard
for accelerator streaming pipelines. PCIe and UCIe are over-specified for this
bandwidth requirement and would add unnecessary implementation complexity.

---

## 5. Bottleneck Status on the Roofline

The Conv2d kernel has arithmetic intensity **AI = 46.9 FLOP/byte**.

The co-processor roofline:
- Peak compute: 1 TFLOP/s (1,000 GFLOP/s)
- External memory bandwidth: 45.8 GB/s (shared DDR4-2667)
- Ridge point: 1,000 / 45.8 = **21.8 FLOP/byte**

Since AI = 46.9 > ridge = 21.8, the kernel is **compute-bound** on the
co-processor — the shared DDR4 bandwidth is sufficient to sustain the systolic
array at full load.

**Interface bottleneck check (AXI4-Stream at 64 GB/s):**

```
Interface-limited perf = 64 GB/s × 46.9 FLOP/byte = 3,001.6 GFLOP/s ≈ 3.0 TFLOP/s
Compute-limited perf   = 1.0 TFLOP/s
```

Since compute-limited (1.0 TFLOP/s) < interface-limited (3.0 TFLOP/s), the
design is **not interface-bound** — the AXI4-Stream link can deliver data faster
than the systolic array can consume it at the 1 TFLOP/s target. The compute
array is the bottleneck, which is the correct design outcome.

---

## 6. Summary

| Item | Value |
|---|---|
| Host platform | Intel Core i5-10210U, DDR4-2667 dual-channel (45.8 GB/s) |
| Interface selected | AXI4-Stream 512-bit @ 1 GHz (data plane) + AXI4-Lite (control) |
| Co-processor target | 1 TFLOP/s systolic array, shared DDR4 memory |
| Required DRAM BW at 1 TFLOP/s | 21.3 GB/s (= 1 TFLOP/s ÷ 46.9 FLOP/byte) |
| AXI4-Stream rated BW (512b, 1GHz) | 64 GB/s |
| Interface-bound? | No — interface-limited at 3.0 TFLOP/s > 1 TFLOP/s target |
| Kernel bound type (on co-processor) | Compute-bound (AI 46.9 > ridge 21.8) |
| Kernel speedup (roofline) | 7× (142.7 GFLOP/s → 1 TFLOP/s) |
| System speedup (Amdahl, 74.4% accel.) | 2.76× |
