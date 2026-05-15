# Synthesis Interpretation — compute_core (sky130A, sky130_fd_sc_hd)

**Tool:** Yosys 0.49+3 + OpenROAD (OpenSTA) via OpenLane 2.1.11  
**PDK:** sky130A / sky130_fd_sc_hd  
**Clock period configured:** 2.0 ns (500 MHz target)

## Clock Period and Worst-Case Slack

OpenROAD/OpenSTA ran formal static timing analysis at the 2.0 ns clock period across three PVT corners. At the typical corner (nom_tt_025C_1v80, 25 °C, 1.80 V), the worst negative setup slack is **−19.653 ns** (data arrival 21.319 ns, required 1.667 ns). Across all corners: fast (ff, −40 °C, 1.95 V) WNS = −11.234 ns; slow (ss, 100 °C, 1.60 V) WNS = −41.050 ns. Hold timing meets at all corners (best hold slack +0.208 ns). TNS at typical corner: −1026.79 ns across 336 violating setup paths.

## Critical Path

Startpoint: _4951_ (dfxtp_2, fanout 72) → Endpoint: _5025_ (dfxtp_2). The path traverses mux4 → o21ai → nor4 → nand2 → a211o → or3 → a21o chains → xnor2/xor2 comparators → a41o → a31oi → or4 → wide AND-OR cells: 21.319 ns data arrival vs. 1.667 ns required, confirming the design is fully combinational between these registers and requires pipelining.

## Total Cell Area

Total cell area: **31,264.99 μm²**, of which 23.7% (7,423.37 μm²) is sequential. Top three contributors by instance count:
1. **mux2_1:** 502 cells — dominant, reflects heavy muxing in FP16 alignment and normalization
2. **dfxtp_2:** 349 cells — 349 registered state bits
3. **nand2_2:** 192 cells — standard NAND-heavy combinational fabric

## Warnings and Failures

Synthesis passed all checks (0 errors). Seven Verilator lint warnings: two WIDTHEXPAND (implicit width promotion in lines 82 and 255 of synth_top.sv) and five UNUSEDSIGNAL. STA reports 336 setup violations at typical corner (972 worst-case across all corners). No hold violations. Max-slew: 662 nets at typical corner. WIDTHEXPAND on lines 82/255 could mask unintended truncation and should be fixed before tape-out.
