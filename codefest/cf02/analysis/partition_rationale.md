# HW/SW Partition Rationale
**ECE 410/510 — Codefest 2**
**Project:** Anemia Detection — HybridModel (ResNet18 + AttentionFusion)

---

## (a) Kernel Accelerated in Hardware and Roofline Justification

The kernel selected for hardware acceleration is the **Conv2d forward and backward
pass of the ResNet18 backbone** (`torch.conv2d`). The cProfile results show it
accounts for 23.1% of runtime directly and drives 57.8% of total runtime through
backpropagation. Across 300 batches per epoch it consumed 292.9 cumulative seconds
out of 1266.6 seconds total — the single largest contributor by a large margin.

The roofline analysis directly supports this choice. The kernel's arithmetic
intensity is **46.9 FLOP/byte**, placing it well to the right of the CPU ridge
point at 5.9 FLOP/byte. This confirms the kernel is **compute-bound** on the
i5-10210U: adding memory bandwidth would not improve performance — only more
compute throughput will. The proposed co-processor is a systolic-array chiplet
targeting **1 TFLOP/s** that shares the host's DDR4-2667 memory bus (45.8 GB/s).
Its ridge point is 1,000 / 45.8 = **21.8 FLOP/byte**. Since the kernel's AI
(46.9) exceeds this ridge, the kernel **remains compute-bound on the co-processor**
— the shared DDR4 bandwidth is sufficient to feed the systolic array at the
1 TFLOP/s target. The roofline projects a **7× kernel-level speedup**
(142.7 GFLOP/s → 1 TFLOP/s). Applying Amdahl's Law with Conv2d (fwd + bwd)
accounting for **74.4%** of the accelerated pipeline yields a projected
**system-level speedup of 2.76×**:

```
Speedup_system = 1 / ((1 - 0.744) + 0.744 / 7)
               = 1 / (0.256 + 0.106)
               = 2.76×
```

## (b) What Remains in Software

The host CPU continues to handle all non-Conv2d operations: dataset loading and
preprocessing (image decode, resize, normalize), handcrafted feature extraction
(GLCM, morphology, color statistics), the AttentionFusion MLP, the Adam optimizer
step, BatchNorm, pooling, loss computation, and all evaluation and logging logic.
These phases collectively account for less than 25% of total runtime and have
low arithmetic intensity — they are better suited to the general-purpose CPU than
to a fixed-function accelerator.

## (c) Interface Bandwidth Requirement

At the target operating point of 1 TFLOP/s and AI = 46.9 FLOP/byte, the
sustained DRAM bandwidth required to keep the systolic array fed is:

```
Required BW = Peak compute / Arithmetic Intensity
            = 1 TFLOP/s / 46.9 FLOP/byte
            = 21.3 GB/s
```

Equivalently from the data volume:

```
Required BW = Bytes per batch / Time per batch
            = 7.43 GB / (348.4 GFLOP / 1 TFLOP/s)
            = 7.43 GB / 0.3484 s
            = 21.3 GB/s
```

The host DDR4-2667 dual-channel bus provides **45.8 GB/s** — more than double the
requirement. The selected interface, **AXI4-Stream at 512-bit width / 1 GHz**,
delivers a rated 64 GB/s, which also exceeds the 21.3 GB/s requirement with 3×
margin. Neither the external DDR4 bus nor the AXI4-Stream host link is the
bottleneck at the 1 TFLOP/s target. The systolic array compute capacity is the
binding resource, which is the correct design outcome.

## (d) Bound Classification and Effect of Accelerator

On the current i5-10210U CPU, the Conv2d kernel is **compute-bound**
(AI = 46.9 FLOP/byte >> ridge = 5.9 FLOP/byte). The CPU is simply too slow
arithmetically; memory bandwidth is not the bottleneck.

The co-processor (1 TFLOP/s systolic array, sharing host DDR4-2667 at 45.8 GB/s)
has a ridge point of 1,000 / 45.8 = **21.8 FLOP/byte**. Since the kernel AI of
46.9 remains above this ridge, the kernel **stays compute-bound on the
co-processor** as well. This is the desirable outcome: the shared DDR4 bandwidth
(45.8 GB/s) is more than sufficient to feed the systolic array at 1 TFLOP/s
(requires only 21.3 GB/s), and performance scales directly with compute
throughput. The limiting resource is MAC array utilization, which should be
targeted for >85% efficiency through input tiling and double-buffering in the
co-processor's local SRAM scratchpad. Note that simultaneous CPU + co-processor
DDR4 access creates contention risk; scheduling the co-processor as an offload
engine (CPU idle during Conv2d tiles) eliminates this hazard.
