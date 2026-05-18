# M3 Plan — compute_core (INT8 QAT Pivot)

**Option A: own project core | ECE 410/510 Spring 2026**

CF07 synthesis produced a −19.653 ns setup violation (arrival 21.319 ns, required 1.667 ns) from the fully combinational FP16 MAC chain: `fp16_to_fp32` → 24×24-bit multiply → `clz24` → barrel-shift → `fp32_to_fp16`. Power: 367.9 mW, 97.8% combinational. SKY130A cannot close this path at 500 MHz.

**Changes for M3:**

1. **Pivot to INT8 QAT.** Remove all FP functions (~180 lines); replace COMPUTE with `accum <= accum + $signed(w) * $signed(x)` (INT8×INT8→INT32). Estimated critical path: 4.0–6.5 ns; 7-stage pipeline targets 800 MHz–1 GHz. Accuracy drop: ~0.1–0.2% for binary classification, within the 40–50 dB SQNR floor.

2. **Unroll 9 parallel MACs.** One 3×3 tile per cycle. Target: ~13,000–18,000 µm² and ~60–90 mW per unit (vs. 31,194 µm² / 367.9 mW).

3. **Fix WIDTHEXPAND warnings** (lines 82, 255) before the INT8 rewrite.

4. **Keep 2.0 ns clock target** to confirm positive setup slack after rewrite.
