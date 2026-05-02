# Q16.16 Fixed-Point Arithmetic — Choice Rationale and Error Analysis

## Format Definition

This design uses **Q16.16 signed fixed-point** arithmetic throughout the compute core and
interface pipeline. Every 32-bit data word encodes a rational value as:

```
  value = (signed 32-bit two's-complement integer) / 2^16
```

- Bit 31: sign (two's complement)
- Bits 30:16: integer part (15 usable magnitude bits)
- Bits 15:0: fractional part

**Range**: approximately −32 768 to +32 767.999 985  
**Unit in the last place (ULP)**: 1 / 65 536 ≈ 1.526 × 10⁻⁵  
**Effective decimal digits**: ≈ 4.8 (log₁₀(65 536))

The accumulator is widened to 64 bits (Q32.32) during the dot product so that no precision
is lost mid-computation. The final result is extracted as `accum[47:16]`, converting the
Q32.32 accumulator back to Q16.16.

---

## Why Fixed-Point over IEEE 754 Single-Precision (FP32)

FP32 provides higher dynamic range (≈ ±3.4 × 10³⁸) and finer precision (≈ 7 decimal
digits), but at significant hardware cost:

| Metric                    | FP32 multiplier | Q16.16 (32-bit fixed) |
|---------------------------|-----------------|----------------------|
| FPGA LUTs (typical)       | ~280–320        | ~70–90               |
| Clock cycles (pipelined)  | 3–5             | 1–2                  |
| Relative dynamic power    | 1×              | 0.3–0.5×             |
| Bit-exact reproducibility | No (round modes)| Yes                  |

For a 1 GHz chiplet constrained by area and power, fixed-point is the standard industry
choice. INT8 quantization — far less precise than Q16.16 — is routinely applied to
ResNet18 with less than 0.5 % accuracy degradation (MLPerf Inference v3.1, 2023).
Q16.16 exceeds INT8 precision by roughly 256×, providing a comfortable safety margin
while still being 3–4× cheaper in silicon than FP32.

---

## Why Q16.16 Specifically

ResNet18 Conv2d layers exhibit the following magnitude statistics after batch normalization:

- Weights: typically |w| < 4.0 (often < 2.0 after weight decay regularization)
- Activations: non-negative post-ReLU, typically |x| < 4.0

**Worst-case single MAC magnitude**: |w × x| < 4 × 4 = 16 → needs 5 integer bits.

**Worst-case tile accumulation** (3 × 3 kernel × 64 input channels = 576 MACs):  
576 × 16 = 9 216 ≈ 2¹³·² → needs 14 integer bits to avoid overflow.

Q16.16 provides **16 integer bits**, giving more than 3.5× headroom above worst-case
overflow (65 536 vs. 9 216). The remaining **16 fractional bits** deliver sub-15 μ
precision, well beyond what the downstream fully-connected classification head requires.

Formats considered and rejected:

| Format  | Int bits | Frac bits | Max value    | Precision    | Verdict        |
|---------|----------|-----------|--------------|--------------|----------------|
| Q8.8    | 8        | 8         | 127.996      | 3.9 × 10⁻³  | Overflows tile |
| **Q16.16** | **16**| **16**    | **32 767.999**| **1.5 × 10⁻⁵** | **Selected** |
| Q24.8   | 24       | 8         | 8 388 607    | 3.9 × 10⁻³  | Int range wasted|
| FP32    | dynamic  | dynamic   | 3.4 × 10³⁸  | 1.2 × 10⁻⁷  | Area/power cost|

Q8.8 overflows on a 576-MAC tile accumulation (max 9 216 >> 127.996).  
Q24.8 wastes integer range that is never needed and delivers the same fractional
precision as Q8.8.  
FP32 provides unnecessarily fine precision at 3–4× the hardware cost.

---

## Rounding Error Analysis

The 64-bit Q32.32 accumulator loses its lower 16 bits when the result is extracted as
`accum[47:16]`. This truncation introduces a per-operation rounding error bounded by:

```
  ε_per_MAC ≤ 2⁻¹⁷ ≈ 7.63 × 10⁻⁶
```

(half the ULP of the discarded portion, conservatively bounded as full ULP here)

**Worst-case accumulated error** for a 576-MAC tile (3 × 3 × 64 channels):

```
  ε_total ≤ 576 × 7.63 × 10⁻⁶ ≈ 4.4 × 10⁻³
```

In practice, rounding errors are zero-mean and partially cancel; the root-mean-square
error is closer to √576 × 7.63 × 10⁻⁶ ≈ 1.8 × 10⁻⁴.

**Comparison to FP32** (machine epsilon ε_mach ≈ 1.19 × 10⁻⁷):  
Q16.16 introduces roughly 15 000–37 000× more worst-case rounding error than FP32 over a
576-MAC tile. However, for binary classification (anemia vs. normal) the acceptable
per-pixel error budget is on the order of 10⁻² — Q16.16 sits comfortably inside that
budget. Prior work on ResNet18 INT8 quantization (Han et al., 2016; Krishnamoorthi,
2018) demonstrates that even 256× coarser precision causes less than 1 % accuracy loss,
confirming that Q16.16 is more than adequate.

---

## Verification

The simulation test vectors in `tb/tb_compute_core.sv` independently verify Q16.16
arithmetic against hand-computed reference values:

- **Test 1**: weights = [1.0 … 9.0], IFM = [1.0 × 9], expected 45.0 = `0x002D_0000`
- **Test 2**: alternating-sign fractional weights, IFM = 2.0 × 9, expected 2.0 = `0x0002_0000`

Both pass with exact bit-match, confirming the accumulator extraction formula
`accum[47:16]` is correct for Q16.16 inputs and Q32.32 intermediate arithmetic.
