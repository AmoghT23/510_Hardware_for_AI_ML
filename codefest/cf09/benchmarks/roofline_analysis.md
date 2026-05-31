# Roofline Analysis — CF09 CLLM
**ECE 410/510 — Codefest 9**

The plot shows three rooflines. The **M4 Rev4 single-tile** has two measured points:
1.5 GFLOP/s (6 cycles, sequential testbench) and 2.25 GFLOP/s (4-cycle FSM-only,
streaming rate), both confirmed by cocotb simulation (TESTS=2 PASS=2 FAIL=0). Both
land 63–99× below the CPU baseline (142.7 GFLOP/s) — 9 MACs cannot match the ~32–64
SIMD lanes AVX2+MKL deploys. The dominant uncertainty is the 2-cycle testbench polling
overhead not present in a real streaming array; end-to-end cocotb simulation of the
full 444-tile systolic array would convert the design-target point from projected to
measured. The **design-target (444-tile, 999 GFLOP/s)** remains projected at 7× above
the CPU. Both AI bounds (93.8–228.8 FLOP/byte) sit far above their ridge points
(0.28 single-tile, 31.2 design-target), confirming compute-bound operation at all scales.
