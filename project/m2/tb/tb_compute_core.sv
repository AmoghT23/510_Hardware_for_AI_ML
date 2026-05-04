// ============================================================================
// tb_compute_core.sv
// ECE 410/510 — Testbench: compute_core  (FP16 revision)
//
// Drives two representative Conv2d tile vectors against the DUT and compares
// results to independently computed FP16 reference values.
// Prints PASS / FAIL for each test and a final verdict line.
//
// FP16 format: S[15] E[14:10] M[9:0], exponent bias = 15
//   Examples:
//     1.0  = 0x3C00  (exp=15, man=0)
//     2.0  = 0x4000  (exp=16, man=0)
//     45.0 = 0x51A0  (exp=20, man=0b0110100000 — verified: 1.01101×2^5 = 45)
//
// Test 1 — ramp weights, unit IFM:
//   w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]   (FP16)
//   x = [1.0 × 9]                                          (FP16)
//   dot = 1+2+…+9 = 45.0  →  FP16: 0x51A0
//
// Test 2 — alternating-sign fractional weights, 2× IFM:
//   w = [0.5, -0.5, 1.5, -1.5, 2.0, -2.0, 0.25, -0.25, 1.0]  (FP16)
//   x = [2.0 × 9]                                               (FP16)
//   dot = 2×(0.5-0.5+1.5-1.5+2-2+0.25-0.25+1) = 2.0
//       →  FP16: 0x4000
//
// Reference values derived from IEEE 754 half-precision bit patterns.
// See project/m2/precision.md for the FP16 format rationale.
//
// Simulator: Icarus Verilog
//   iverilog -g2012 -o sim/cc_sim.out rtl/compute_core.sv tb/tb_compute_core.sv
//   vvp sim/cc_sim.out
// ============================================================================

`timescale 1ns/1ps

module tb_compute_core;

    // ── DUT ports ─────────────────────────────────────────────────────────
    logic        clk, rst_n, start;
    logic [15:0] weight_data, ifm_data, result;   // FP16 (was 32-bit Q16.16)
    logic        weight_valid, ifm_valid, done, ready;

    // ── DUT instantiation ─────────────────────────────────────────────────
    compute_core #(
        .KERNEL_H(3),
        .KERNEL_W(3),
        .IN_CH   (1)
    ) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .weight_data (weight_data),
        .weight_valid(weight_valid),
        .ifm_data    (ifm_data),
        .ifm_valid   (ifm_valid),
        .result      (result),
        .done        (done),
        .ready       (ready)
    );

    // ── 1 GHz clock (period = 1 ns) ───────────────────────────────────────
    initial clk = 0;
    always  #0.5 clk = ~clk;

    // ── VCD dump ──────────────────────────────────────────────────────────
    initial begin
        $dumpfile("sim/compute_core.vcd");
        $dumpvars(0, tb_compute_core);
    end

    // ── FP16 test vectors ─────────────────────────────────────────────────
    // All constants verified against IEEE 754 half-precision encoding.
    //
    // Encoding: S[15] E[14:10] M[9:0], value = (-1)^S × 2^(E−15) × 1.M
    //   1.0  : exp=15 (01111), man=0          → 0_01111_0000000000 = 0x3C00
    //   2.0  : exp=16 (10000), man=0          → 0_10000_0000000000 = 0x4000
    //   3.0  : exp=16 (10000), man=1000000000 → 0_10000_1000000000 = 0x4200
    //   4.0  : exp=17 (10001), man=0          → 0_10001_0000000000 = 0x4400
    //   5.0  : exp=17 (10001), man=0100000000 → 0_10001_0100000000 = 0x4500
    //   6.0  : exp=17 (10001), man=1000000000 → 0_10001_1000000000 = 0x4600
    //   7.0  : exp=17 (10001), man=1100000000 → 0_10001_1100000000 = 0x4700
    //   8.0  : exp=18 (10010), man=0          → 0_10010_0000000000 = 0x4800
    //   9.0  : exp=18 (10010), man=0010000000 → 0_10010_0010000000 = 0x4880
    //   45.0 : exp=20 (10100), man=0110100000 → 0_10100_0110100000 = 0x51A0
    //   0.5  : exp=14 (01110), man=0          → 0_01110_0000000000 = 0x3800
    //   1.5  : exp=15 (01111), man=1000000000 → 0_01111_1000000000 = 0x3E00
    //   0.25 : exp=13 (01101), man=0          → 0_01101_0000000000 = 0x3400
    //   -x   : flip bit 15

    logic [15:0] w_t1 [0:8];   // Test 1 weights  (FP16)
    logic [15:0] x_t1 [0:8];   // Test 1 IFM      (FP16)
    logic [15:0] w_t2 [0:8];   // Test 2 weights  (FP16)
    logic [15:0] x_t2 [0:8];   // Test 2 IFM      (FP16)

    integer pass_count, fail_count, i;

    // ── main stimulus ─────────────────────────────────────────────────────
    initial begin : stim

        pass_count   = 0;
        fail_count   = 0;
        rst_n        = 0;
        start        = 0;
        weight_data  = 16'h0;  weight_valid = 0;
        ifm_data     = 16'h0;  ifm_valid    = 0;

        // synchronous reset: hold low for 4 cycles
        repeat (4) @(posedge clk);
        #0.1; rst_n = 1;

        $display("=== tb_compute_core (FP16) ===");

        // ════════════════════════════════════════════════════════════════
        // TEST 1: ramp weights [1..9], unit IFM [1.0×9], expected 45.0
        //         FP16(45.0) = 0x51A0
        // ════════════════════════════════════════════════════════════════
        w_t1[0] = 16'h3C00;  // 1.0
        w_t1[1] = 16'h4000;  // 2.0
        w_t1[2] = 16'h4200;  // 3.0
        w_t1[3] = 16'h4400;  // 4.0
        w_t1[4] = 16'h4500;  // 5.0
        w_t1[5] = 16'h4600;  // 6.0
        w_t1[6] = 16'h4700;  // 7.0
        w_t1[7] = 16'h4800;  // 8.0
        w_t1[8] = 16'h4880;  // 9.0
        for (i = 0; i < 9; i++) x_t1[i] = 16'h3C00;  // 1.0

        while (!ready) @(posedge clk);

        @(posedge clk); #0.1; start = 1;
        @(posedge clk); #0.1; start = 0;

        weight_valid = 1;  ifm_valid = 1;
        for (i = 0; i < 9; i++) begin
            weight_data = w_t1[i];
            ifm_data    = x_t1[i];
            @(posedge clk); #0.1;
        end
        weight_valid = 0;  ifm_valid = 0;

        @(posedge done);

        if (result === 16'h51a0) begin
            $display("  T1 [ramp-w / unit-x / expect 45.0] : PASS  got=0x%04h", result);
            pass_count = pass_count + 1;
        end else begin
            $display("  T1 [ramp-w / unit-x / expect 45.0] : FAIL  got=0x%04h  exp=0x51a0", result);
            fail_count = fail_count + 1;
        end

        // ════════════════════════════════════════════════════════════════
        // TEST 2: alternating-sign fractional weights, IFM=2.0, expected 2.0
        //         dot = 2×(0.5-0.5+1.5-1.5+2-2+0.25-0.25+1) = 2.0
        //         FP16(2.0) = 0x4000
        // ════════════════════════════════════════════════════════════════
        w_t2[0] = 16'h3800;  //  0.5
        w_t2[1] = 16'hB800;  // -0.5
        w_t2[2] = 16'h3E00;  //  1.5
        w_t2[3] = 16'hBE00;  // -1.5
        w_t2[4] = 16'h4000;  //  2.0
        w_t2[5] = 16'hC000;  // -2.0
        w_t2[6] = 16'h3400;  //  0.25
        w_t2[7] = 16'hB400;  // -0.25
        w_t2[8] = 16'h3C00;  //  1.0
        for (i = 0; i < 9; i++) x_t2[i] = 16'h4000;  // 2.0

        while (!ready) @(posedge clk);

        @(posedge clk); #0.1; start = 1;
        @(posedge clk); #0.1; start = 0;

        weight_valid = 1;  ifm_valid = 1;
        for (i = 0; i < 9; i++) begin
            weight_data = w_t2[i];
            ifm_data    = x_t2[i];
            @(posedge clk); #0.1;
        end
        weight_valid = 0;  ifm_valid = 0;

        @(posedge done);

        if (result === 16'h4000) begin
            $display("  T2 [alt-sign-w / 2x / expect 2.0]  : PASS  got=0x%04h", result);
            pass_count = pass_count + 1;
        end else begin
            $display("  T2 [alt-sign-w / 2x / expect 2.0]  : FAIL  got=0x%04h  exp=0x4000", result);
            fail_count = fail_count + 1;
        end

        // ── final verdict ─────────────────────────────────────────────
        $display("---");
        if (fail_count == 0)
            $display("PASS — all %0d tests passed.", pass_count);
        else
            $display("FAIL — %0d of %0d tests failed.", fail_count, pass_count + fail_count);

        $finish;
    end

    // ── watchdog ──────────────────────────────────────────────────────────
    initial begin
        #10000;
        $display("FAIL — simulation timeout (>10000 ns).");
        $finish;
    end

endmodule
