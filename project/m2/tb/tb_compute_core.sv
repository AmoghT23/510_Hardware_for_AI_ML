// ============================================================================
// tb_compute_core.sv
// ECE 410/510 — Testbench: compute_core
//
// Drives two representative Conv2d tile vectors against the DUT and compares
// results to independently computed Q16.16 reference values.
// Prints PASS / FAIL for each test and a final verdict line.
//
// Test 1 — ramp weights, unit IFM:
//   w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
//   x = [1.0 × 9]
//   dot = 1+2+…+9 = 45.0  →  Q16.16: 45×65536 = 0x002D_0000
//
// Test 2 — alternating-sign fractional weights, 2× IFM:
//   w = [0.5, -0.5, 1.5, -1.5, 2.0, -2.0, 0.25, -0.25, 1.0]
//   x = [2.0 × 9]
//   dot = 2×(0.5-0.5+1.5-1.5+2-2+0.25-0.25+1) = 2.0
//       →  Q16.16: 2×65536 = 0x0002_0000
//
// Reference values are derived by hand / Python arithmetic, NOT from a prior
// DUT run.  See project/m2/precision.md for the Q16.16 error analysis.
//
// Simulator: Icarus Verilog   iverilog -g2012 -o sim.out
//                             compute_core.sv tb_compute_core.sv && vvp sim.out
// ============================================================================

`timescale 1ns/1ps

module tb_compute_core;

    // ── DUT ports ─────────────────────────────────────────────────────────
    logic        clk, rst_n, start;
    logic [31:0] weight_data, ifm_data, result;
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

    // ── test vectors (module-level for iverilog compatibility) ─────────────
    // Q16.16 encoding: integer_value = real_value * 65536
    logic [31:0] w_t1 [0:8];   // Test 1 weights
    logic [31:0] x_t1 [0:8];   // Test 1 IFM
    logic [31:0] w_t2 [0:8];   // Test 2 weights
    logic [31:0] x_t2 [0:8];   // Test 2 IFM

    integer pass_count, fail_count, i;

    // ── main stimulus ─────────────────────────────────────────────────────
    initial begin : stim

        // initialise
        pass_count   = 0;
        fail_count   = 0;
        rst_n        = 0;
        start        = 0;
        weight_data  = 0;  weight_valid = 0;
        ifm_data     = 0;  ifm_valid    = 0;

        // synchronous reset: hold low for 4 cycles
        repeat (4) @(posedge clk);
        #0.1; rst_n = 1;

        $display("=== tb_compute_core ===");

        // ════════════════════════════════════════════════════════════════
        // TEST 1: ramp weights [1..9], unit IFM [1.0 x9], expected 45.0
        // ════════════════════════════════════════════════════════════════
        w_t1[0] = 32'h00010000;  // 1.0
        w_t1[1] = 32'h00020000;  // 2.0
        w_t1[2] = 32'h00030000;  // 3.0
        w_t1[3] = 32'h00040000;  // 4.0
        w_t1[4] = 32'h00050000;  // 5.0
        w_t1[5] = 32'h00060000;  // 6.0
        w_t1[6] = 32'h00070000;  // 7.0
        w_t1[7] = 32'h00080000;  // 8.0
        w_t1[8] = 32'h00090000;  // 9.0
        for (i = 0; i < 9; i++) x_t1[i] = 32'h00010000;  // 1.0

        // wait for IDLE
        while (!ready) @(posedge clk);

        // pulse start for one cycle
        @(posedge clk); #0.1; start = 1;
        @(posedge clk); #0.1; start = 0;

        // stream 9 weight+IFM pairs (one per cycle, both valid high)
        weight_valid = 1;  ifm_valid = 1;
        for (i = 0; i < 9; i++) begin
            weight_data = w_t1[i];
            ifm_data    = x_t1[i];
            @(posedge clk); #0.1;
        end
        weight_valid = 0;  ifm_valid = 0;

        // wait for done pulse
        @(posedge done);

        // compare against reference: 45 * 65536 = 0x002D0000
        if (result === 32'h002D0000) begin
            $display("  T1 [ramp-w / unit-x / expect 45.0] : PASS  got=0x%08h", result);
            pass_count = pass_count + 1;
        end else begin
            $display("  T1 [ramp-w / unit-x / expect 45.0] : FAIL  got=0x%08h  exp=0x002D0000", result);
            fail_count = fail_count + 1;
        end

        // ════════════════════════════════════════════════════════════════
        // TEST 2: alternating-sign fractional weights, IFM=2.0, expected 2.0
        // ════════════════════════════════════════════════════════════════
        w_t2[0] = 32'h00008000;  //  0.5   =  32768
        w_t2[1] = 32'hFFFF8000;  // -0.5   = -32768 (two's complement)
        w_t2[2] = 32'h00018000;  //  1.5   =  98304
        w_t2[3] = 32'hFFFE8000;  // -1.5   = -98304
        w_t2[4] = 32'h00020000;  //  2.0   = 131072
        w_t2[5] = 32'hFFFE0000;  // -2.0   = -131072
        w_t2[6] = 32'h00004000;  //  0.25  =  16384
        w_t2[7] = 32'hFFFFC000;  // -0.25  = -16384
        w_t2[8] = 32'h00010000;  //  1.0   =  65536
        for (i = 0; i < 9; i++) x_t2[i] = 32'h00020000;  // 2.0 = 131072

        // wait for IDLE
        while (!ready) @(posedge clk);

        // pulse start
        @(posedge clk); #0.1; start = 1;
        @(posedge clk); #0.1; start = 0;

        // stream 9 pairs
        weight_valid = 1;  ifm_valid = 1;
        for (i = 0; i < 9; i++) begin
            weight_data = w_t2[i];
            ifm_data    = x_t2[i];
            @(posedge clk); #0.1;
        end
        weight_valid = 0;  ifm_valid = 0;

        // wait for done pulse
        @(posedge done);

        // compare against reference: 2 * 65536 = 0x00020000
        if (result === 32'h00020000) begin
            $display("  T2 [alt-sign-w / 2x / expect 2.0]  : PASS  got=0x%08h", result);
            pass_count = pass_count + 1;
        end else begin
            $display("  T2 [alt-sign-w / 2x / expect 2.0]  : FAIL  got=0x%08h  exp=0x00020000", result);
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

    // ── watchdog: abort if simulation hangs ───────────────────────────────
    initial begin
        #10000;
        $display("FAIL — simulation timeout (>10000 ns).");
        $finish;
    end

endmodule
