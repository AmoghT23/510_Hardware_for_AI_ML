# M3 Synthesis Plan — compute_core

**Option A: synthesized own project core**

OpenROAD formal STA (nom_tt_025C_1v80) gives a worst negative setup slack of −19.653 ns against the 2.0 ns clock target (data arrival 21.319 ns, required 1.667 ns). The datapath is fully combinational between registers and must be pipelined before M3.

**Changes for M3:**

1. **Pipeline the FP16 datapath.** Insert registers every ~8–10 logic levels along the `ifm_buf$rdreg` → mux/AOI → output path; ~7 stages should bring each under 2 ns.

2. **Fix WIDTHEXPAND warnings.** Lines 82 and 255 of `synth_top.sv` have implicit width promotions that could mask truncation errors.

3. **Reduce mux count.** The 502 `mux2_1` cells (18.4% of total) suggest redundant mux trees; sharing structures in the normalization logic should reduce area.

4. **Keep the 2.0 ns clock target** for M3 to verify timing closure after pipelining.
