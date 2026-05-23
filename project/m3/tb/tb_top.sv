`timescale 1ns/1ps
// ============================================================================
// tb_top.sv  —  SystemVerilog testbench for top_ti
// ECE 410/510  M3-TI
//
// Generates a clean VCD at sim/dump.vcd.
// Run from the m3_ti/ directory:
//
//   iverilog -g2012 -o sim/tb_top.vvp \
//       tb/tb_top.sv \
//       rtl/sram_sp.sv rtl/grad_core.sv rtl/interface_ti.sv \
//       rtl/compute_core_ti.sv rtl/top_ti.sv
//   vvp sim/tb_top.vvp
//   gtkwave sim/dump.vcd sim/dump.gtkw
//
// Three annotated waveform regions visible in GTKWave:
//   Region 1: Host Write  — CTRL[1] + AXI-Stream weight beat
//   Region 2: Compute     — CTRL[0] start, wait STATUS[0] done
//   Region 3: Host Read   — AXI-Lite RESULT read-back (FP32 45.0)
// ============================================================================

module tb_top;

    // ── clock / reset ─────────────────────────────────────────────────────────
    logic clk = 1'b0;
    always #5 clk = ~clk;          // 10 ns → 100 MHz

    logic rst_n;

    // ── AXI-Lite ──────────────────────────────────────────────────────────────
    logic [5:0]   s_axil_awaddr;
    logic         s_axil_awvalid;
    logic         s_axil_awready;
    logic [31:0]  s_axil_wdata;
    logic [3:0]   s_axil_wstrb;
    logic         s_axil_wvalid;
    logic         s_axil_wready;
    logic [1:0]   s_axil_bresp;
    logic         s_axil_bvalid;
    logic         s_axil_bready;
    logic [5:0]   s_axil_araddr;
    logic         s_axil_arvalid;
    logic         s_axil_arready;
    logic [31:0]  s_axil_rdata;
    logic [1:0]   s_axil_rresp;
    logic         s_axil_rvalid;
    logic         s_axil_rready;

    // ── AXI-Stream ────────────────────────────────────────────────────────────
    logic [511:0] s_axis_tdata;
    logic         s_axis_tvalid;
    logic         s_axis_tready;
    logic         s_axis_tlast;

    // ── DUT ───────────────────────────────────────────────────────────────────
    top_ti dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .s_axil_awaddr  (s_axil_awaddr),
        .s_axil_awvalid (s_axil_awvalid),
        .s_axil_awready (s_axil_awready),
        .s_axil_wdata   (s_axil_wdata),
        .s_axil_wstrb   (s_axil_wstrb),
        .s_axil_wvalid  (s_axil_wvalid),
        .s_axil_wready  (s_axil_wready),
        .s_axil_bresp   (s_axil_bresp),
        .s_axil_bvalid  (s_axil_bvalid),
        .s_axil_bready  (s_axil_bready),
        .s_axil_araddr  (s_axil_araddr),
        .s_axil_arvalid (s_axil_arvalid),
        .s_axil_arready (s_axil_arready),
        .s_axil_rdata   (s_axil_rdata),
        .s_axil_rresp   (s_axil_rresp),
        .s_axil_rvalid  (s_axil_rvalid),
        .s_axil_rready  (s_axil_rready),
        .s_axis_tdata   (s_axis_tdata),
        .s_axis_tvalid  (s_axis_tvalid),
        .s_axis_tready  (s_axis_tready),
        .s_axis_tlast   (s_axis_tlast)
    );

    // ── VCD dump ──────────────────────────────────────────────────────────────
    initial begin
        $dumpfile("sim/dump.vcd");
        $dumpvars(0, tb_top);
    end

    // ── BF16 stream beat constants ────────────────────────────────────────────
    // weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] in BF16
    // tdata[16*i+15:16*i] = BF16(weights[i])
    localparam [511:0] WEIGHT_BEAT = {368'h0,
        16'h4110, 16'h4100, 16'h40e0, 16'h40c0,
        16'h40a0, 16'h4080, 16'h4040, 16'h4000, 16'h3f80};

    // IFM = [1.0]*9 in BF16 (0x3f80 each)
    localparam [511:0] IFM_BEAT = {368'h0,
        16'h3f80, 16'h3f80, 16'h3f80, 16'h3f80,
        16'h3f80, 16'h3f80, 16'h3f80, 16'h3f80, 16'h3f80};

    // ── AXI-Lite write (4 cycles) ────────────────────────────────────────────
    // Signals are driven #1ps after @(posedge clk) to avoid the testbench/DUT
    // race: without #1, both the testbench assignment and DUT always_ff fire
    // in the same delta cycle so wr_data may be sampled before it is driven.
    task axil_write;
        input [5:0]  addr;
        input [31:0] data;
        begin
            @(posedge clk); #1;        // 1 ps after edge — safe driving window
            s_axil_awaddr  = addr;
            s_axil_awvalid = 1'b1;
            s_axil_wdata   = data;
            s_axil_wstrb   = 4'hF;
            s_axil_wvalid  = 1'b1;

            @(posedge clk); #1;        // DUT samples AW + W at this posedge
            s_axil_awvalid = 1'b0;
            s_axil_wvalid  = 1'b0;

            @(posedge clk); #1;        // b_pend=1 (bvalid=1)
            s_axil_bready  = 1'b1;

            @(posedge clk); #1;        // DUT clears b_pend
            s_axil_bready  = 1'b0;
        end
    endtask

    // ── AXI-Lite read (4 cycles) ─────────────────────────────────────────────
    task axil_read;
        input  [5:0]  addr;
        output [31:0] data;
        begin
            @(posedge clk); #1;
            s_axil_araddr  = addr;
            s_axil_arvalid = 1'b1;

            @(posedge clk); #1;        // DUT samples AR; r_pend queued, rdata latched
            s_axil_arvalid = 1'b0;

            @(posedge clk); #1;        // r_pend=1 → rvalid=1; rdata stable
            data           = s_axil_rdata;
            s_axil_rready  = 1'b1;

            @(posedge clk); #1;        // DUT clears r_pend
            s_axil_rready  = 1'b0;
        end
    endtask

    // ── AXI-Stream beat (2 cycles, tready always 1) ──────────────────────────
    task send_stream;
        input [511:0] beat;
        begin
            @(posedge clk); #1;
            s_axis_tdata  = beat;
            s_axis_tvalid = 1'b1;
            s_axis_tlast  = 1'b1;

            @(posedge clk); #1;        // DUT latches tdata into wl_capture / ifm_buf
            s_axis_tvalid = 1'b0;
            s_axis_tlast  = 1'b0;
        end
    endtask

    // ── stimulus ──────────────────────────────────────────────────────────────
    integer k;
    logic [31:0] status;
    logic [31:0] result;

    initial begin
        // initialise driven signals
        rst_n          = 1'b0;
        s_axil_awaddr  = 6'h00;
        s_axil_awvalid = 1'b0;
        s_axil_wdata   = 32'h0;
        s_axil_wstrb   = 4'h0;
        s_axil_wvalid  = 1'b0;
        s_axil_bready  = 1'b0;
        s_axil_araddr  = 6'h00;
        s_axil_arvalid = 1'b0;
        s_axil_rready  = 1'b0;
        s_axis_tdata   = 512'h0;
        s_axis_tvalid  = 1'b0;
        s_axis_tlast   = 1'b0;
        status         = 32'h0;
        result         = 32'h0;

        // reset (4 cycles low, then deassert 1ps after posedge)
        repeat (4) @(posedge clk); #1;
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // ====================================================================
        // REGION 1 : HOST WRITE — load weight tile
        //   CTRL[1]=1 sets load_weights; next stream beat captured as weights.
        //   SRAM write FSM takes 9 clock cycles → poll STATUS[2] = wt_loaded.
        // ====================================================================
        $display("[%0t ns] ===== REGION 1: Host Write (weight load) =====", $time);

        axil_write(6'h00, 32'h2);      // CTRL[1] = load_weights
        send_stream(WEIGHT_BEAT);       // BF16 weights [1..9] → wl_capture

        // poll STATUS[2] = wt_loaded
        status = 32'h0;
        for (k = 0; k < 30; k = k + 1) begin
            axil_read(6'h04, status);
            if (status[2]) k = 30;
        end
        $display("[%0t ns]   wt_loaded=%b  STATUS=0x%08h", $time, status[2], status);

        // ====================================================================
        // REGION 2 : COMPUTE — stream IFM tile, start, wait done
        //   CTRL[0]=1 starts sequencer; compute_core takes ~11 cycles.
        //   Poll STATUS[0] = done.
        // ====================================================================
        $display("[%0t ns] ===== REGION 2: Compute =====", $time);

        send_stream(IFM_BEAT);          // BF16 IFM [1.0]*9 → ifm_buf
        axil_write(6'h00, 32'h1);      // CTRL[0] = start

        // poll STATUS[0] = done
        status = 32'h0;
        for (k = 0; k < 50; k = k + 1) begin
            axil_read(6'h04, status);
            if (status[0]) k = 50;
        end
        $display("[%0t ns]   done=%b  STATUS=0x%08h", $time, status[0], status);

        // ====================================================================
        // REGION 3 : HOST READ — FP32 dot-product result
        //   T1 vector: dot([1..9],[1.0]*9) = 45.0 → FP32 0x42340000
        // ====================================================================
        $display("[%0t ns] ===== REGION 3: Host Read (result) =====", $time);

        axil_read(6'h0C, result);
        $display("[%0t ns]   RESULT=0x%08h", $time, result);

        if (result === 32'h42340000)
            $display("[%0t ns]   PASS — 45.0 (0x42340000) matches expected", $time);
        else
            $display("[%0t ns]   INFO — got 0x%08h, expected 0x42340000 (45.0)", $time, result);

        repeat (10) @(posedge clk);
        $display("[%0t ns] ===== SIMULATION COMPLETE =====", $time);
        $finish;
    end

endmodule
