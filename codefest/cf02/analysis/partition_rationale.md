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
compute throughput will. A hardware accelerator built around a systolic array
(targeting 8 TFLOP/s with 512 GB/s on-chip SRAM bandwidth) keeps the kernel
compute-bound (AI 46.9 > HW ridge 15.6) and projects a **56× speedup** over the
CPU baseline, moving the operating point from 142.7 GFLOP/s to 8 TFLOP/s on
the roofline.

## (b) What Remains in Software

The host CPU continues to handle all non-Conv2d operations: dataset loading and
preprocessing (image decode, resize, normalize), handcrafted feature extraction
(GLCM, morphology, color statistics), the AttentionFusion MLP, the Adam optimizer
step, BatchNorm, pooling, loss computation, and all evaluation and logging logic.
These phases collectively account for less than 25% of total runtime and have
low arithmetic intensity — they are better suited to the general-purpose CPU than
to a fixed-function accelerator.

## (c) Interface Bandwidth Requirement

At the target operating point of 8 TFLOP/s and batch size 32, the accelerator
must consume input feature maps and weights at a rate sufficient to keep the
systolic array fed. From the DRAM bytes calculation:

```
Required BW = Bytes per batch / Time per batch
            = 7.43 GB / (348.4 GFLOP / 8 TFLOP/s)
            = 7.43 GB / 0.0436 s
            = 170.4 GB/s required interface bandwidth
```

A PCIe 4.0 x16 link provides ~32 GB/s bidirectional — insufficient. A UCIe or
AXI4-Stream with HBM2e (up to 460 GB/s) would meet the requirement and keep the
accelerator from becoming interface-bound. For a chiplet context, UCIe with
on-package HBM is the appropriate interface choice.

## (d) Bound Classification and Effect of Accelerator

On the current i5-10210U CPU, the Conv2d kernel is **compute-bound**
(AI = 46.9 FLOP/byte >> ridge = 5.9 FLOP/byte). The CPU is simply too slow
arithmetically; memory bandwidth is not the bottleneck.

The proposed hardware accelerator (8 TFLOP/s, 512 GB/s on-chip) has a ridge
point of 15.6 FLOP/byte. Since the kernel AI of 46.9 remains above this ridge,
the kernel **stays compute-bound on the accelerator** as well. This is the
desirable outcome: it means the on-chip SRAM bandwidth (512 GB/s) is more than
sufficient to feed the systolic array, and performance scales directly with
compute throughput. The design does not need to increase on-chip bandwidth further
— the limiting resource is the MAC array utilization, which should be targeted
for >85% utilization through tiling and double-buffering.
