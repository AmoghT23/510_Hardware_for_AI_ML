# MAC Unit — LLM Code Review

## File Attribution

| File | LLM |
|------|-----|
| `mac_llm_A.sv` | Gemini 3 (Fast) |
| `mac_llm_B.sv` | Claude 4.7 Opus |

---

## 1. Compile Check — iverilog

> **Note:** Verilator is not installed on this machine. Compile checks were performed
> with `iverilog 11.0 -g2012` instead.

**Command:**
```bash
iverilog -g2012 -o mac_llm_A_check.vvp mac_llm_A.sv
iverilog -g2012 -o mac_llm_B_check.vvp mac_llm_B.sv
iverilog -g2012 -o mac_correct_check.vvp mac_correct.sv
```

**Results:**

| File | Exit code | Errors / Warnings |
|------|-----------|-------------------|
| `mac_llm_A.sv` | 0 | None |
| `mac_llm_B.sv` | 0 | None |
| `mac_correct.sv` | 0 | None |

All three files compile cleanly under `iverilog -g2012`.

---

## 2. Code Comparison — mac_llm_A vs mac_llm_B

### mac_llm_A.sv (Gemini 3 Fast)
```systemverilog
module mac (
    input  logic              clk,
    input  logic              rst,
    input  logic signed [7:0] a,
    input  logic signed [7:0] b,
    output logic signed [31:0] out
);
    // Intermediate signal for the product.
    // 8x8 signed multiplication results in a 16-bit signed value.
    logic signed [15:0] product;

    // Explicitly casting or ensuring signed context for multiplication
    assign product = a * b;

    always_ff @(posedge clk) begin
        if (rst) begin
            out <= 32'sd0;
        end else begin
            out <= out + product;
        end
    end
endmodule
```

### mac_llm_B.sv (Claude 4.7 Opus)
```systemverilog
module mac (
    input  logic        clk,
    input  logic        rst,
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

### Differences

| Feature | mac_llm_A — Gemini | mac_llm_B — Claude | Explanation |
|---------|-------------------|-------------------|-------------|
| **Intermediate product** | `logic signed [15:0] product;`<br>`assign product = a * b;` | None — inlined | Gemini explicitly declares a 16-bit signed wire for the 8×8 product. Claude inlines it directly. Gemini's approach makes the intermediate width visible and easier to inspect in a waveform viewer. |
| **Accumulate expression** | `out <= out + product;` | `out <= out + (a * b);` | Gemini adds the named 16-bit wire to `out`; sign extension from 16→32 bits is implicit via `signed`. Claude relies on the 32-bit signed context of `out` to propagate into `(a * b)` — correct per IEEE 1800 but more subtle. |
| **Reset assignment** | `out <= 32'sd0;` | `out <= 32'sd0;` | Identical. Both explicitly assign a 32-bit signed zero. |
| **begin/end style** | Explicit `begin/end` in both branches | Single-statement, no `begin/end` | Style difference only — both synthesise identically. |
| **Code structure** | Two-stage: `assign` + `always_ff` | One-stage: everything inside `always_ff` | Gemini's two-stage approach is easier to trace in waveforms (`product` is a visible wire). Claude's is more concise. |

**Assessment:** Both are functionally correct. Gemini's intermediate `product` wire makes the design easier to debug in simulation (visible in GTKWave signal list). Claude's inline approach is more concise but relies on implicit signed context propagation.

---

## 3. Simulation — iverilog

**Testbench:** `mac_tb.sv` | **Tool:** `iverilog 11.0 -g2012`

**Commands:**
```bash
iverilog -g2012 -o sim.vvp <dut>.sv mac_tb.sv
vvp sim.vvp
```

### mac_llm_A.sv — Simulation Transcript

> Transcript header reads `DUT file: mac_llm_B.sv` — this is a hardcoded string
> in `mac_tb.sv` and does not reflect which DUT was compiled. All cycle values are correct.

```
VCD info: dumpfile mac_tb.vcd opened for output.
============================================================
 mac_tb: INT8 MAC unit simulation
 DUT file: mac_llm_A.sv  [actual DUT — header string hardcoded in mac_tb.sv]
============================================================
CYCLE 0 | RESET (init)       | rst=1 a=   0 b=   0 | out=0   (expected 0)   | PASS
CYCLE 1 | a=3 b=4 (cyc 1/3) | rst=0 a=   3 b=   4 | out=12  (expected 12)  | PASS
CYCLE 2 | a=3 b=4 (cyc 2/3) | rst=0 a=   3 b=   4 | out=24  (expected 24)  | PASS
CYCLE 3 | a=3 b=4 (cyc 3/3) | rst=0 a=   3 b=   4 | out=36  (expected 36)  | PASS
CYCLE 4 | RESET (mid-stream) | rst=1 a=   3 b=   4 | out=0   (expected 0)   | PASS
CYCLE 5 | a=-5 b=2 (cyc 1/2)| rst=0 a=  -5 b=   2 | out=-10 (expected -10) | PASS
CYCLE 6 | a=-5 b=2 (cyc 2/2)| rst=0 a=  -5 b=   2 | out=-20 (expected -20) | PASS
============================================================
 Simulation complete — 7 cycles exercised
============================================================
Errors: 0  Warnings: 0
```

**GTKWave waveform — mac_llm_A:**

<img width="1751" height="672" alt="llm_A" src="https://github.com/user-attachments/assets/7e412c27-b233-41a7-b4ca-9a9dc961a47a" />

> Note: `product[15:0]` is visible as a separate wire in the signal list (Gemini's  
> intermediate wire). Values: `000C` (12 decimal) → `0018` (24) → `0024` (36) → reset  
> → `FFF6` (−10 in 16-bit two's complement). `out[31:0]` shows the accumulated  
> hex values: `0000000C` → `00000018` → `00000024` → `00000000` → `FFFFFFF6`.

---

### mac_llm_B.sv — Simulation Transcript

```
VCD info: dumpfile mac_tb.vcd opened for output.
============================================================
 mac_tb: INT8 MAC unit simulation
 DUT file: mac_llm_B.sv
============================================================
CYCLE 0 | RESET (init)       | rst=1 a=   0 b=   0 | out=0   (expected 0)   | PASS
CYCLE 1 | a=3 b=4 (cyc 1/3) | rst=0 a=   3 b=   4 | out=12  (expected 12)  | PASS
CYCLE 2 | a=3 b=4 (cyc 2/3) | rst=0 a=   3 b=   4 | out=24  (expected 24)  | PASS
CYCLE 3 | a=3 b=4 (cyc 3/3) | rst=0 a=   3 b=   4 | out=36  (expected 36)  | PASS
CYCLE 4 | RESET (mid-stream) | rst=1 a=   3 b=   4 | out=0   (expected 0)   | PASS
CYCLE 5 | a=-5 b=2 (cyc 1/2)| rst=0 a=  -5 b=   2 | out=-10 (expected -10) | PASS
CYCLE 6 | a=-5 b=2 (cyc 2/2)| rst=0 a=  -5 b=   2 | out=-20 (expected -20) | PASS
============================================================
 Simulation complete — 7 cycles exercised
============================================================
Errors: 0  Warnings: 0
```

**GTKWave waveform — mac_llm_B:**

<img width="1751" height="662" alt="llm_B" src="https://github.com/user-attachments/assets/6faf01dd-68a6-4c89-a9e1-079927a704e9" />

> Note: No intermediate `product` wire — only `clk`, `rst`, `a[7:0]`, `b[7:0]`, `out[31:0]`  
> visible in the signal list. `out[31:0]` hex trace identical to mac_llm_A.

---

## 4. Identified Issues

### Issue 1 — `logic signed` in port declarations (both files)

**(a) Offending lines** — `mac_llm_A.sv` lines 17–19 / `mac_llm_B.sv` lines 15–17:
```systemverilog
input  logic signed [7:0] a,
input  logic signed [7:0] b,
output logic signed [31:0] out
```

**(b) Why it is wrong:** `logic signed` as a combined specifier is valid SystemVerilog  
but not supported by the **Yosys Verilog-2005 frontend**, which parses `logic` as a type  
keyword and then sees `signed` as an unexpected token. Confirmed verbatim (EDA Playground):
```
design.sv:17: ERROR: syntax error, unexpected TOK_ID, expecting ',' or '=' or ')'
```

**(c) Fix:** Remove `signed` from port declarations; use `$signed()` at the point of use:
```verilog
input  [7:0]  a,
input  [7:0]  b,
output reg [31:0] out
// ...
out <= $signed(out) + ($signed(a) * $signed(b));
```

---

### Issue 2 — 16-bit intermediate `product` width fragility (mac_llm_A only)

**(a) Offending lines** — `mac_llm_A.sv` lines 24–27:
```systemverilog
logic signed [15:0] product;
assign product = a * b;
```

**(b) Why it is wrong:** The 16-bit wire is sufficient for INT8×INT8 (max |product| = 16 384,  
fits in 15 bits + sign). However, if either input is widened, the wire silently truncates  
the MSB. The fix-and-forget hardcoded width is a maintainability hazard.

**(c) Fix:** Inline the product inside `always_ff` to let width be context-determined:
```systemverilog
out <= out + (a * b);
```

---

### Issue 3 — Implicit sign context in accumulate (mac_llm_B only)

**(a) Offending line** — `mac_llm_B.sv` line 24:
```systemverilog
out <= out + (a * b);
```

**(b) Why it is ambiguous:** Relies on 32-bit signed context from `out` propagating  
into `(a * b)` per IEEE 1800 rules. This is correct in compliant tools. But if `signed`  
is ever stripped from the `out` port (e.g. for Yosys compatibility), `out` becomes  
unsigned, the entire expression turns unsigned, and `0xFB × 2 = 502` instead of `−10`.  
This was **confirmed by iverilog** during this review (cycles 5–6 FAIL before fix).

**(c) Fix:** Explicit `$signed()` on all operands:
```systemverilog
out <= $signed(out) + ($signed(a) * $signed(b));
```

---

### Issue 4 — Unsigned `out` breaks sign extension after Yosys fix (mac_correct.sv)

**(a) Offending line** — `mac_correct.sv` before fix:
```verilog
out <= out + ($signed(a) * $signed(b));
```
where `out` is `output logic [31:0]` (unsigned).

**(b) Why it is wrong:** Verified by iverilog — cycles 5–6 produced `502` and `1004`  
instead of `−10` and `−20`. Because `out` is unsigned, Verilog treats the entire  
`+` expression as unsigned, zero-extending `0xFB → 251` before multiplying.

**(c) Fix applied in `mac_correct.sv`:**
```verilog
out <= $signed(out) + ($signed(a) * $signed(b));
```
Re-run confirmed all 7 cycles PASS.

---

## 5. Corrected File — mac_correct.sv

```systemverilog
module mac (
    input  logic        clk,
    input  logic        rst,
    input  logic [7:0]  a,
    input  logic [7:0]  b,
    output logic [31:0] out
);
    always_ff @(posedge clk) begin
        if (rst)
            out <= 32'sd0;
        else
            out <= $signed(out) + ($signed(a) * $signed(b));
    end
endmodule
```

**iverilog simulation transcript:**
```
VCD info: dumpfile mac_tb.vcd opened for output.
CYCLE 0 | RESET (init)       | rst=1 a=   0 b=   0 | out=0   (expected 0)   | PASS
CYCLE 1 | a=3 b=4 (cyc 1/3) | rst=0 a=   3 b=   4 | out=12  (expected 12)  | PASS
CYCLE 2 | a=3 b=4 (cyc 2/3) | rst=0 a=   3 b=   4 | out=24  (expected 24)  | PASS
CYCLE 3 | a=3 b=4 (cyc 3/3) | rst=0 a=   3 b=   4 | out=36  (expected 36)  | PASS
CYCLE 4 | RESET (mid-stream) | rst=1 a=   3 b=   4 | out=0   (expected 0)   | PASS
CYCLE 5 | a=-5 b=2 (cyc 1/2)| rst=0 a=  -5 b=   2 | out=-10 (expected -10) | PASS
CYCLE 6 | a=-5 b=2 (cyc 2/2)| rst=0 a=  -5 b=   2 | out=-20 (expected -20) | PASS
Errors: 0  Warnings: 0
```

**GTKWave waveform — mac_correct:**

<img width="1755" height="668" alt="correct" src="https://github.com/user-attachments/assets/9f0aa14e-6c5d-4de1-a811-2f0256f7320b" />

