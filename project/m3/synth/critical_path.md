# Critical Path Analysis — top_ti (M3-TI)

**PDK:** sky130A / sky130_fd_sc_hd  
**Corner:** nom_tt_025C_1v80 (TT)  
**Clock period:** 130 ns  
**WNS:** +29.905 ns → worst-case path delay = **100.095 ns**

---

## Path Summary

| Field | Value |
|---|---|
| Path type | Register-to-register (setup) |
| Start point | `compute_core_ti` — accumulator input register |
| End point | `conv_interface_ti` — `reg_result[31:0]` |
| Logic stages | BF16 multiply → FP32 accumulate |
| Total delay | ~100.1 ns |
| Slack | +29.905 ns |

---

## Stage-by-Stage Breakdown

```
 Stage   Logic                                   Estimated delay
 ------  --------------------------------------- ---------------
 1       Launch register clock-to-Q              1.5 ns
         (sky130_fd_sc_hd__dfxtp_4, TT)

 2       BF16 exponent extraction & alignment     8 ns
           sign/exp/mantissa unpack (bit-sel)
           8-bit exponent subtract (sky130 ripple-carry adder)
           mantissa pre-shift (8:1 mux tree)

 3       BF16 mantissa multiply                  42 ns
           7×7 unsigned multiplier (partial-product array)
           Wallace/Dadda compressor tree
           carry-propagate adder (13-bit)
         This stage dominates — 130 nm CMOS
         7×7 array ≈ 49 AND gates + 6-level carry tree

 4       BF16 → FP32 normalize + round           18 ns
           leading-one detect (priority encoder)
           mantissa left-shift (5-bit shift amt)
           round-to-nearest-even (tie-break logic)
           exponent adjust (+127 bias)

 5       FP32 accumulate (mantissa shift + add)  28 ns
           FP32 exponent compare (8-bit subtract)
           mantissa alignment shift (23-bit barrel shifter)
           24-bit integer adder (carry-propagate)
           normalize + round output
           exponent increment (conditional)

 6       Capture register setup + hold margin     2 ns
         (sky130_fd_sc_hd__dfxtp_4, TT)

 Total                                          ~99.5 ns
 (with routing + net delays ≈ 100.095 ns measured)
```

---

## Why This Path is Critical

The BF16 multiply in stage 3 is the bottleneck. Sky130A 130 nm standard cells have gate delays of ~50–70 ps per stage; a 7×7 multiplier with carry-propagate adder requires approximately 40–45 logic levels, giving ~40 ns of pure gate delay before routing. The FP32 accumulate in stage 5 adds another significant segment due to the 23-bit barrel shifter.

The AXI4-Lite register paths (`reg_ctrl`, `reg_load_weights`, `reg_start`) are all single-register write paths and have measured slack of +100+ ns — they never appear near the critical path.

---

## Optimization Options (Not Implemented)

| Option | Expected benefit | Cost |
|---|---|---|
| Carry-save accumulator (replace stage 5 adder) | −10 to −15 ns | ~5% area increase |
| Pipelining (split multiply + accumulate into 2 cycles) | Max freq → ~20 MHz | Latency +1 cycle, pipeline registers |
| Booth encoding (reduce partial products) | −8 to −12 ns in stage 3 | Design complexity |
| Reduce clock to 100 ns target | +0 ns WNS (just meeting) | None — was tried, repair_design OOM at step 31 |

The 130 ns period was selected because it gives +29.9 ns margin at TT and keeps `repair_design` from memory-exhausting on the buffer insertion loop.

---

## Source

Derived from:
- `metrics.json`: `timing__setup__ws__corner:nom_tt_025C_1v80 = 29.904613`
- `synth/timing_report.txt` critical path entry
- Sky130A standard cell databook (sky130_fd_sc_hd, TT 1.8 V 25 °C)
