// ============================================================================
// tb_interface.sv
// ECE 410/510 — Testbench: conv_interface  (M3 revision)
//
// Copied from M2 and updated for M3:
//   Task 1: core_result stub widened [15:0] → [31:0] to match updated port.
//
// Exercises all four required transaction types and prints PASS / FAIL.
//
// Test 1 — AXI4-Lite write (complete transaction):
//   Write TILE_LEN = 9 to register 0x08.
//   AW + W channels presented simultaneously; B response accepted.
//   Pass condition: BRESP = 2'b00 (OKAY).
//
// Test 2 — AXI4-Lite read (complete transaction):
//   Read TILE_LEN back from register 0x08.
//   AR channel presented; R channel data captured.
//   Pass condition: RDATA[7:0] = 8'd9, RRESP = 2'b00.
//
// Test 3 — AXI4-Stream write beat:
//   Send one TDATA beat (512-bit, non-trivial pattern) with TLAST.
//   Pass condition: TREADY = 1 during the beat (transfer accepted).
//
// Test 4 — CTRL start pulse:
//   Write 0x1 to CTRL register (0x00); DUT must drive core_start = 1
//   for exactly one clock cycle.
//   Pass condition: core_start sampled high immediately after write posedge.
//
// Simulator: Icarus Verilog
//   iverilog -g2012 -o sim/if_sim.out rtl/interface.sv tb/tb_interface.sv && vvp sim/if_sim.out
// ============================================================================

`timescale 1ns/1ps

module tb_interface;

    // ── DUT port signals ──────────────────────────────────────────────────
    logic        clk, rst_n;

    // AXI4-Lite
    logic [3:0]  awaddr;  logic        awvalid; logic awready;
    logic [31:0] wdata;   logic [3:0]  wstrb;   logic wvalid;  logic wready;
    logic [1:0]  bresp;   logic        bvalid;  logic bready;
    logic [3:0]  araddr;  logic        arvalid; logic arready;
    logic [31:0] rdata;   logic [1:0]  rresp;   logic rvalid;  logic rready;

    // AXI4-Stream
    logic [511:0] tdata;
    logic          tvalid, tready, tlast;

    // Core stub signals
    logic        core_start;
    logic [7:0]  core_tile_len;
    logic [31:0] core_result;   // INT32 (widened from [15:0] FP16 in M2)
    logic        core_done, core_ready;

    // ── DUT instantiation ─────────────────────────────────────────────────
    conv_interface dut (
        .clk            (clk),
        .rst_n          (rst_n),
        // AXI4-Lite write address
        .s_axil_awaddr  (awaddr),  .s_axil_awvalid(awvalid), .s_axil_awready(awready),
        // AXI4-Lite write data
        .s_axil_wdata   (wdata),   .s_axil_wstrb  (wstrb),
        .s_axil_wvalid  (wvalid),  .s_axil_wready (wready),
        // AXI4-Lite write response
        .s_axil_bresp   (bresp),   .s_axil_bvalid (bvalid),  .s_axil_bready(bready),
        // AXI4-Lite read address
        .s_axil_araddr  (araddr),  .s_axil_arvalid(arvalid), .s_axil_arready(arready),
        // AXI4-Lite read data
        .s_axil_rdata   (rdata),   .s_axil_rresp  (rresp),
        .s_axil_rvalid  (rvalid),  .s_axil_rready (rready),
        // AXI4-Stream
        .s_axis_tdata   (tdata),   .s_axis_tvalid (tvalid),
        .s_axis_tready  (tready),  .s_axis_tlast  (tlast),
        // Core interface
        .core_start     (core_start),  .core_tile_len(core_tile_len),
        .core_result    (core_result), .core_done    (core_done),
        .core_ready     (core_ready)
    );

    // ── 1 GHz clock ───────────────────────────────────────────────────────
    initial clk = 0;
    always  #0.5 clk = ~clk;

    // ── VCD dump ──────────────────────────────────────────────────────────
    initial begin
        $dumpfile("sim/waveform.vcd");
        $dumpvars(0, tb_interface);
    end

    // ── core stubs: no compute_core attached for this testbench ───────────
    assign core_done   = 1'b0;
    assign core_ready  = 1'b1;
    assign core_result = 32'h0;   // INT32 stub

    // ── counters and shared variables ─────────────────────────────────────
    integer      pass_count, fail_count;
    logic [31:0] read_val;
    logic        got_start;

    // ── AXI4-Lite write task ──────────────────────────────────────────────
    task axil_write;
        input [3:0]  addr;
        input [31:0] data;
        @(posedge clk); #0.1;
        awaddr  = addr;  awvalid = 1'b1;
        wdata   = data;  wvalid  = 1'b1;  wstrb = 4'hF;
        while (!(awready && wready)) @(posedge clk);
        @(posedge clk); #0.1;
        awvalid = 1'b0;  wvalid = 1'b0;
        while (!bvalid) @(posedge clk);
        #0.1; bready = 1'b1;
        @(posedge clk); #0.1;
        bready = 1'b0;
    endtask

    // ── AXI4-Lite read task ───────────────────────────────────────────────
    task axil_read;
        input  [3:0]  addr;
        output [31:0] rd_data;
        @(posedge clk); #0.1;
        araddr  = addr;  arvalid = 1'b1;
        while (!arready) @(posedge clk);
        @(posedge clk); #0.1;
        arvalid = 1'b0;
        while (!rvalid) @(posedge clk);
        rd_data = rdata;
        #0.1; rready = 1'b1;
        @(posedge clk); #0.1;
        rready = 1'b0;
    endtask

    // ── main stimulus ─────────────────────────────────────────────────────
    initial begin : stim

        pass_count = 0;  fail_count = 0;
        rst_n   = 1'b0;
        awaddr  = 4'h0;   awvalid = 1'b0;
        wdata   = 32'd0;  wstrb   = 4'h0;  wvalid = 1'b0;
        bready  = 1'b0;
        araddr  = 4'h0;   arvalid = 1'b0;
        rready  = 1'b0;
        tdata   = 512'd0; tvalid  = 1'b0;  tlast  = 1'b0;
        got_start = 1'b0;

        repeat (4) @(posedge clk);
        #0.1; rst_n = 1'b1;

        $display("=== tb_interface (M3) ===");

        // ════════════════════════════════════════════════════════════════
        // TEST 1 — AXI4-Lite write: TILE_LEN = 9 to address 0x08
        // ════════════════════════════════════════════════════════════════
        axil_write(4'h8, 32'h00000009);

        if (bresp === 2'b00) begin
            $display("  T1 [AXI4-Lite write TILE_LEN=9] : PASS  bresp=2'b00 (OKAY)");
            pass_count = pass_count + 1;
        end else begin
            $display("  T1 [AXI4-Lite write TILE_LEN=9] : FAIL  bresp=0x%01h", bresp);
            fail_count = fail_count + 1;
        end

        // ════════════════════════════════════════════════════════════════
        // TEST 2 — AXI4-Lite read: read TILE_LEN back from address 0x08
        // ════════════════════════════════════════════════════════════════
        axil_read(4'h8, read_val);

        if (read_val[7:0] === 8'd9 && rresp === 2'b00) begin
            $display("  T2 [AXI4-Lite read  TILE_LEN]  : PASS  rdata=0x%08h  rresp=OKAY", read_val);
            pass_count = pass_count + 1;
        end else begin
            $display("  T2 [AXI4-Lite read  TILE_LEN]  : FAIL  rdata=0x%08h  rresp=0x%01h", read_val, rresp);
            fail_count = fail_count + 1;
        end

        // ════════════════════════════════════════════════════════════════
        // TEST 3 — AXI4-Stream beat
        // ════════════════════════════════════════════════════════════════
        @(posedge clk); #0.1;
        tdata  = {{480{1'b0}}, 32'hDEAD_BEEF};
        tvalid = 1'b1;
        tlast  = 1'b1;

        if (tready === 1'b1) begin
            $display("  T3 [AXI4-Stream beat]           : PASS  tready=1 (beat accepted)");
            pass_count = pass_count + 1;
        end else begin
            $display("  T3 [AXI4-Stream beat]           : FAIL  tready=0");
            fail_count = fail_count + 1;
        end

        @(posedge clk); #0.1;
        tvalid = 1'b0;  tlast = 1'b0;

        // ════════════════════════════════════════════════════════════════
        // TEST 4 — CTRL start pulse
        // ════════════════════════════════════════════════════════════════
        @(posedge clk); #0.1;
        awaddr  = 4'h0;  awvalid = 1'b1;
        wdata   = 32'h00000001;  wvalid = 1'b1;  wstrb = 4'hF;

        while (!(awready && wready)) @(posedge clk);
        @(posedge clk); #0.1;
        awvalid  = 1'b0;  wvalid = 1'b0;
        got_start = core_start;

        while (!bvalid) @(posedge clk);
        #0.1; bready = 1'b1;
        @(posedge clk); #0.1;
        bready = 1'b0;

        if (got_start === 1'b1) begin
            $display("  T4 [CTRL start pulse]           : PASS  core_start pulsed high");
            pass_count = pass_count + 1;
        end else begin
            $display("  T4 [CTRL start pulse]           : FAIL  core_start was not high");
            fail_count = fail_count + 1;
        end

        // ── final verdict ─────────────────────────────────────────────────
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
