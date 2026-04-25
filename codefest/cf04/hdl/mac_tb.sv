// Testbench for mac_llm_B.sv (INT8 MAC unit)
// Stimulus: [a=3,b=4] x3 cycles → assert rst → [a=-5,b=2] x2 cycles
// Compatible with VCS and Questasim; produces VCD + transcript

`timescale 1ns/1ps

module mac_tb;

    // ------------------------------------------------------------------ //
    //  DUT signals
    // ------------------------------------------------------------------ //
    logic              clk;
    logic              rst;
    logic signed [7:0] a;
    logic signed [7:0] b;
    logic signed [31:0] out;

    // ------------------------------------------------------------------ //
    //  DUT instantiation
    // ------------------------------------------------------------------ //
    mac dut (
        .clk (clk),
        .rst (rst),
        .a   (a),
        .b   (b),
        .out (out)
    );

    // ------------------------------------------------------------------ //
    //  Clock: 10 ns period (100 MHz)
    // ------------------------------------------------------------------ //
    initial clk = 1'b0;
    always  #5 clk = ~clk;

    // ------------------------------------------------------------------ //
    //  VCD dump (works on both VCS and Questasim)
    // ------------------------------------------------------------------ //
    initial begin
        $dumpfile("mac_tb.vcd");
        $dumpvars(0, mac_tb);
    end

    // ------------------------------------------------------------------ //
    //  Stimulus + checking
    // ------------------------------------------------------------------ //
    integer cycle_num;
    logic signed [31:0] expected;

    // Helper task: advance one clock, sample outputs 1 ns after posedge
    // (samples after NBA region so always_ff results are stable)
    task automatic tick(
        input integer     exp_val,
        input string      label
    );
        @(posedge clk); #1;
        $display("CYCLE %0d | %-22s | rst=%b a=%4d b=%4d | out=%0d  (expected %0d) | %s",
                 cycle_num, label, rst, $signed(a), $signed(b),
                 $signed(out), exp_val,
                 ($signed(out) === exp_val) ? "PASS" : "*** FAIL ***");
        if ($signed(out) !== exp_val)
            $error("Mismatch at cycle %0d: got %0d, expected %0d", cycle_num, $signed(out), exp_val);
        cycle_num++;
    endtask

    initial begin
        $display("============================================================");
        $display(" mac_tb: INT8 MAC unit simulation");
        $display(" DUT file: mac_llm_B.sv");
        $display("============================================================");
        $display("%-7s | %-22s | %-19s | %-6s | %-16s | %s",
                 "CYCLE", "PHASE", "INPUTS", "out", "expected", "STATUS");
        $display("------------------------------------------------------------");

        // ----- initialise ------------------------------------------------
        cycle_num = 0;
        rst = 1'b1;
        a   = 8'sd0;
        b   = 8'sd0;

        // ----- Cycle 0: initial synchronous reset ------------------------
        // out must become 0
        expected = 32'sd0;
        tick(expected, "RESET (init)");

        // ----- Cycles 1-3: a=3, b=4 -------------------------------------
        // After each posedge: out += 3*4 = 12
        rst = 1'b0;
        a   = 8'sd3;
        b   = 8'sd4;

        // Cycle 1: 0 + 12 = 12
        expected = 32'sd12;
        tick(expected, "a=3 b=4 (cyc 1/3)");

        // Cycle 2: 12 + 12 = 24
        expected = 32'sd24;
        tick(expected, "a=3 b=4 (cyc 2/3)");

        // Cycle 3: 24 + 12 = 36
        expected = 32'sd36;
        tick(expected, "a=3 b=4 (cyc 3/3)");

        // ----- Assert rst (synchronous reset mid-stream) -----------------
        // out must snap to 0 on this rising edge
        rst = 1'b1;
        // a,b intentionally left at 3,4 to confirm rst wins
        expected = 32'sd0;
        tick(expected, "RESET (mid-stream)");

        // ----- Cycles 5-6: a=-5, b=2 ------------------------------------
        // After each posedge: out += (-5)*2 = -10
        rst = 1'b0;
        a   = -8'sd5;   // 8'hFB in 2's complement
        b   =  8'sd2;

        // Cycle 5: 0 + (-10) = -10
        expected = -32'sd10;
        tick(expected, "a=-5 b=2 (cyc 1/2)");

        // Cycle 6: -10 + (-10) = -20
        expected = -32'sd20;
        tick(expected, "a=-5 b=2 (cyc 2/2)");

        // ----- Done ------------------------------------------------------
        $display("============================================================");
        $display(" Simulation complete — %0d cycles exercised", cycle_num);
        $display("============================================================");
        $finish;
    end

endmodule
