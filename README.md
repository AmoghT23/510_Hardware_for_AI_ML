# ECE-510 Hardware for AI/ML — Amogh Thakur

This repository contains the codefests (weekly challenges), profiling work, and project deliverables for **ECE-510 (Hardware for AI/ML, Spring 2026)**, taught by Prof. Christof Teuscher at Portland State University.

**Languages:** SystemVerilog, Verilog, Python

---

## Hybrid CNN Project

**Summary**
Design a hardware co-processor chiplet for a Hybrid CNN, grounded in profiling data, roofline analysis, and a measured software baseline.

**Project Topic**
AI Accelerator for *"Hybrid CNN model for Anemia Detection in Blood Smear Images"*

**Full implementation repository (joint with teammate)**
https://github.com/nehalshivane04/Accelerator---Anemia-Dataset

---

## Project — Anemia Detection Co-processor Chiplet

### 1. Module to accelerate: 2D Convolution (Conv2d)

The accelerator targets the dominant arithmetic kernel of a HybridModel (ResNet18 backbone + AttentionFusion MLP) used to classify peripheral blood smear images as healthy or anemic. Profiling on an Intel Core i5-10210U (PyTorch 2.11.0 CPU build, AneRBC dataset, 9,600 train images) identifies `torch.conv2d` as the dominant kernel, accounting for **23.1 % of direct runtime and 57.8 % including backpropagation** (292.9 s out of 1266.6 s over 300 batches).

The co-processor chiplet maps these Conv2d operations onto a weight-stationary systolic array of multiply-accumulate (MAC) units. Roofline analysis projects a **7× kernel-level speedup** (142.7 GFLOP/s baseline → 1 TFLOP/s target) and, applying Amdahl's Law over the 74.4 % accelerated fraction, a **2.76× end-to-end system speedup**. AttentionFusion, BatchNorm, pooling, handcrafted feature extraction (GLCM, morphology, color), and optimizer steps remain on the host CPU.

### 2. Precision: INT8 with 32-bit accumulator

| | Format | Width |
|---|---|---|
| Weights / activations | INT8 (symmetric per-tensor quantization) | 8 bits |
| Multiplier output | signed | 16 bits |
| MAC accumulator | signed | 32 bits |

INT8 is chosen to maximize MAC density and minimize on-chip memory bandwidth versus the FP32 baseline. The 32-bit accumulator prevents overflow across the partial sums of a conv window. The CMAN deliverable in `codefest/cf04` walks through symmetric quantization (S = max(|W|) / 127), the round–clamp–dequantize cycle, and demonstrates the saturation failure mode when the scale factor is too small.

Compared to FP32:

| Metric | Improvement |
|---|---|
| Weight memory footprint | −75 % (8 vs 32 bits per element) |
| Memory bandwidth per inference | −75 % |
| MAC datapath width | −75 % (8-bit multipliers vs 32-bit) |

### 3. Interface: AXI4-Stream (data) + AXI4-Lite (control)

The chiplet exposes:

- **AXI4-Stream** for input feature maps and weight tiles — 512-bit data width, 1 GHz clock, **64 GB/s rated**.
- **AXI4-Lite** for the control plane — tile dimensions, stride, padding, and the start/done handshake.

#### Why AXI4-Stream and not SPI / I²C / AXI4-Lite alone

| Interface | Rated BW | Required (21.3 GB/s) | Verdict |
|---|---|---|---|
| SPI (50 MHz) | ~0.006 GB/s | 21.3 GB/s | No — 3,500× short |
| I²C (1 MHz) | ~0.0001 GB/s | 21.3 GB/s | No |
| AXI4-Lite | ~1–4 GB/s | 21.3 GB/s | No — register overhead, no bursts |
| **AXI4-Stream (512-bit, 1 GHz)** | **64 GB/s** | 21.3 GB/s | **Yes — 3× margin** |
| PCIe 4.0 x16 | ~32 GB/s | 21.3 GB/s | Over-specified, adds complexity |
| UCIe (die-to-die) | ~460 GB/s | 21.3 GB/s | Far exceeds requirement |

#### Required bandwidth from arithmetic intensity

The Conv2d kernel has measured **arithmetic intensity AI = 46.9 FLOP/byte** (fwd+bwd, no DRAM reuse). At the 1 TFLOP/s co-processor target:

```
Required BW = Peak compute / Arithmetic Intensity
            = 1,000 GFLOP/s / 46.9 FLOP/byte
            = 21.3 GB/s
```

AXI4-Stream at 512-bit / 1 GHz provides **64 GB/s**, giving a **3× margin** over the requirement. Cross-checking the interface bottleneck:

```
Interface-limited perf = 64 GB/s × 46.9 FLOP/byte = ~3.0 TFLOP/s
Compute-limited perf   = 1.0 TFLOP/s  ← binding constraint
```

The compute array is the bottleneck — not the interface — which is the correct design outcome.

#### Roofline position on the co-processor

The co-processor shares the host DDR4-2667 bus (45.8 GB/s peak), giving a co-processor ridge point of **21.8 FLOP/byte**. Since the kernel's AI (46.9) exceeds this ridge, the kernel remains **compute-bound** on the co-processor as well. The design is consistent across both the interface and host-memory boundaries.

---

## Repository Structure

```
510_Hardware_for_AI_ML/
├── codefest/
│   ├── cf01/
│   ├── cf02/
│   ├── cf03/
│   └── cf04/
└── project/
    ├── m1/
    ├── hdl/
    ├── algorithm_diagram.png
    └── heilmeier.md
```
