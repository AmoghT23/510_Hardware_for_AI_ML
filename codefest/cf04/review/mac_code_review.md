# MAC Unit LLM Code Review

| File | LLM |
|------|-----|
| mac_llm_A.sv | Gemini 3 (Fast) |
| mac_llm_B.sv | Claude 4.7 Opus |

---

## mac_llm_B.sv — Simulation & Code Review

**DUT:** `mac_llm_B.sv` — INT8 MAC unit (AI accelerator building block)  
**Testbench:** `mac_tb.sv`

---

### 1. DUT Source Review

```systemverilog
module mac (
    input  logic        clk,
    input  logic        rst,           // active-high synchronous reset
    input  logic signed [7:0]  a,
    input  logic signed [7:0]  b,
    output logic signed [31:0] out
);
    always_ff @(posedge clk) begin
        if (rst)
            out <= 32'sd0;
        else
            out <= out + (a * b);
    end
endmodule
```

| # | Item | Status |
|---|------|--------|
| 1 | Uses `always_ff` — synthesisable flip-flop inference | ✅ Correct |
| 2 | Synchronous active-high reset clears accumulator to 0 | ✅ Correct |
| 3 | 8-bit × 8-bit signed → 16-bit product, sign-extended to 32-bit before addition | ✅ Correct |
| 4 | No `initial` blocks, no `$display`, no `#` delays in RTL | ✅ Meets constraints |
| 5 | 32-bit accumulator prevents overflow for reasonable burst lengths | ✅ Good practice |
| 6 | No overflow/saturation on `out` itself (wraps at ±2 147 483 647) | ⚠️ Acceptable for building block |
| 7 | `rst` is synchronous — reset only takes effect on a clock edge | ✅ Consistent with spec |

---

### 2. Testbench Stimulus Plan

```
Clock period : 10 ns (100 MHz)
Timescale    : 1ns / 1ps

Cycle  Phase                rst   a     b
-----  -------------------  ---   ---   ---
  0    Initial reset         1     0     0
  1    a=3, b=4  (1/3)       0     3     4
  2    a=3, b=4  (2/3)       0     3     4
  3    a=3, b=4  (3/3)       0     3     4
  4    Mid-stream reset      1     3     4   ← rst wins, a/b ignored
  5    a=-5, b=2 (1/2)       0    -5     2
  6    a=-5, b=2 (2/2)       0    -5     2
```

---

### 3. Hand-Simulated Transcript

> Values sampled 1 ns **after** the rising clock edge (post-NBA region).

```
============================================================
 mac_tb: INT8 MAC unit simulation
 DUT file: mac_llm_B.sv
============================================================
CYCLE | PHASE                  | INPUTS               | out    | expected  | STATUS
------------------------------------------------------------
CYCLE 0 | RESET (init)           | rst=1 a=   0 b=   0 | out=0   (expected 0)   | PASS
CYCLE 1 | a=3 b=4 (cyc 1/3)     | rst=0 a=   3 b=   4 | out=12  (expected 12)  | PASS
CYCLE 2 | a=3 b=4 (cyc 2/3)     | rst=0 a=   3 b=   4 | out=24  (expected 24)  | PASS
CYCLE 3 | a=3 b=4 (cyc 3/3)     | rst=0 a=   3 b=   4 | out=36  (expected 36)  | PASS
CYCLE 4 | RESET (mid-stream)     | rst=1 a=   3 b=   4 | out=0   (expected 0)   | PASS
CYCLE 5 | a=-5 b=2 (cyc 1/2)    | rst=0 a=  -5 b=   2 | out=-10 (expected -10) | PASS
CYCLE 6 | a=-5 b=2 (cyc 2/2)    | rst=0 a=  -5 b=   2 | out=-20 (expected -20) | PASS
============================================================
 Simulation complete — 7 cycles exercised
============================================================
```

---

### 4. Accumulator Trace

| Cycle | rst | a  | b  | a×b | out (after posedge) | Notes |
|-------|-----|----|----|-----|---------------------|-------|
| 0     |  1  |  0 |  0 |   0 | **0**               | Synchronous reset |
| 1     |  0  |  3 |  4 |  12 | **12**              | 0 + 12 |
| 2     |  0  |  3 |  4 |  12 | **24**              | 12 + 12 |
| 3     |  0  |  3 |  4 |  12 | **36**              | 24 + 12 |
| 4     |  1  |  3 |  4 |  12 | **0**               | Mid-stream reset — rst wins over multiply-accumulate |
| 5     |  0  | -5 |  2 | -10 | **-10**             | 0 + (−10) |
| 6     |  0  | -5 |  2 | -10 | **-20**             | −10 + (−10) |

---

### 5. How to Run

**VCS**
```bash
vcs -sverilog mac_llm_B.sv mac_tb.sv -o simv && ./simv
```

**Questasim / ModelSim**
```tcl
vlog -sv mac_llm_B.sv mac_tb.sv
vsim mac_tb -do "run -all; quit"
```

VCD output: `mac_tb.vcd` — open in GTKWave or DVE.

---

### 6. Summary

The LLM-generated RTL is **functionally correct** for all 7 test cycles:
- Synchronous reset zeroes the accumulator on the exact clock edge.
- Signed 8×8-bit multiplication accumulates correctly for positive inputs (3×4=12).
- Mid-stream reset overrides accumulation as required.
- Negative input (−5×2=−10) accumulates correctly in two's complement.
