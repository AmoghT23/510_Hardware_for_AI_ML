# Heilmeier Questions — Anemia Detection Hardware Accelerator
**ECE 410/510 | Revised after profiling (Codefest 2)**

---

## Q1 — What are you trying to do?

I am building a co-processor chiplet that analyzes microscope images of blood cells
to detect anemia. Currently this requires either a trained specialist or a powerful
computer. My chip performs the image pattern recognition step — specifically the 2D
convolution operations inside a ResNet18 neural network — on a small, low-power
device. This enables a portable, battery-powered blood analyzer that a community
health worker can use in a rural clinic with no internet, no lab, and no specialist,
delivering a screening result in under one second.

---

## Q2 — How is it done today, and what are the limits of current practice?

Manual examination by a hematologist is slow, subjective, and unavailable in most
resource-limited settings where anemia is most prevalent (affecting 2+ billion people
globally). Automated approaches use CNNs on laptops, achieving ~91–92% validation
accuracy with hybrid models. But these approaches have hard limits that profiling
now quantifies precisely:

**Software baseline measured on Intel Core i5-10210U (this project):**
- Median wall-clock time per run: **2573.9 seconds (~43 minutes)** for 1 epoch over
  9,600 images — far too slow for real-time screening.
- Peak memory usage: **2165 MB RSS** — far beyond any embedded device.
- cProfile identified `torch.conv2d` as the dominant kernel: **292.9 s out of 1266.6 s
  (23.1% of runtime directly; 57.8% including backpropagation through the same weights)**.
- Measured throughput: **142.7 GFLOP/s** on a 4-core laptop CPU consuming ~15W TDP.
- Roofline analysis confirms the kernel is **compute-bound** (arithmetic intensity =
  46.9 FLOP/byte >> CPU ridge point = 5.9 FLOP/byte), meaning adding memory bandwidth
  will not help — only more arithmetic throughput will.

The limit is therefore not algorithmic but architectural: a general-purpose CPU cannot
deliver the FLOP/s density needed for real-time inference at low power. Prior to
profiling, the dominant kernel was assumed to be the full forward pass; profiling
revised this to the Conv2d operations specifically, which are individually cheaper to
accelerate than the whole network.

---

## Q3 — What is new in your approach, and why do you think it will be successful?

Three things, all grounded in the profiling data:

**First, the HW/SW partition is data-driven.** The cProfile and roofline results
directly justify which kernel goes to hardware. Conv2d accounts for 23.1% of direct
runtime and is the bottleneck for the 57.8% backprop cost. The arithmetic intensity
of 46.9 FLOP/byte places it squarely in the compute-bound regime on the CPU — exactly
the workload a systolic-array accelerator is optimized for. The AttentionFusion MLP,
BatchNorm, pooling, handcrafted feature extraction (GLCM, morphology, color — 14
features, ~5% of runtime), and optimizer steps all remain on the host CPU, which
handles them adequately. This split is not an assumption; it is derived from measured
data.

**Second, the accelerator design is matched to the kernel's roofline position.**
The proposed co-processor chiplet targets **1 TFLOP/s** compute throughput and shares
the host's DDR4-2667 memory bus (45.8 GB/s), giving a co-processor ridge point of
21.8 FLOP/byte. Since the kernel's AI (46.9) exceeds this ridge, the kernel remains
**compute-bound on the co-processor** — the shared DDR4 bandwidth is sufficient to
feed the systolic array at the 1 TFLOP/s target (requires only 21.3 GB/s of the
45.8 GB/s available). Roofline projects a **7× kernel-level speedup**
(142.7 GFLOP/s → 1 TFLOP/s). Applying Amdahl's Law — Conv2d fwd+bwd accounts for
**74.4%** of the accelerated pipeline — gives a projected **2.76× system-level
end-to-end speedup**. This is a defensible, analytically grounded estimate rather
than an optimistic peak-throughput ratio. The required host interface bandwidth at
1 TFLOP/s is only 21.3 GB/s, which AXI4-Stream at 512-bit / 1 GHz (rated 64 GB/s)
satisfies with 3× margin — the design is not interface-bound.

**Third, the scope is deliberately narrow and verifiable.** Accelerating Conv2d alone —
the single function that cProfile pinpoints — keeps the design surface small enough
for full synthesis and benchmarking within 10 weeks, while delivering the largest
possible runtime reduction per unit design effort.

---

*Profiling performed with cProfile on HybridModel (ResNet18 + AttentionFusion),
AneRBC dataset, 9,600 training images, batch size 32, Intel Core i5-10210U,
Python 3.11.9, PyTorch 2.11.0. Full profiler output in
`codefest/cf02/profiling/project_profile.txt`.*
