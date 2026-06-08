`timescale 1ns/1ps
// ============================================================================
// tb_top.sv  —  SystemVerilog testbench for top_ti (32×32 BF16 Conv2d)
// ECE 410/510  M4
//
// Generates a clean VCD at sim/dump.vcd.
// Run from the project directory:
//
//   iverilog -g2012 -o sim/tb_top.vvp \
//       tb/tb_top.sv \
//       rtl/sram_sp.sv rtl/grad_core.sv rtl/ifm_buffer.sv rtl/pe_array.sv \
//       rtl/compute_core_ti.sv rtl/interface_ti.sv rtl/top.sv
//   vvp sim/tb_top.vvp
//   gtkwave sim/dump.vcd sim/dump.gtkw
//
// Test vector T1: ramp weights [1..9] × all-1.0 34×34 IFM
//   Expected: all 1024 results = 45.0 → FP32 0x42340000
//
// Four annotated waveform regions visible in GTKWave:
//   Region 1: Weight load  — CTRL[1] + 1 weight beat → poll STATUS[2]
//   Region 2: IFM load     — 37 stream beats (34×34 all-1.0) → poll STATUS[4]
//   Region 3: Compute      — CTRL[0] start → poll STATUS[0] done
//   Region 4: Result read  — 1024 × AXI-Lite reads from 0x1000–0x1FFC
// ============================================================================

module tb_top;

    // ── clock / reset ─────────────────────────────────────────────────────────
    logic clk = 1'b0;
    always #5 clk = ~clk;          // 10 ns → 100 MHz

    logic rst_n;

    // ── AXI-Lite (13-bit address: reaches 0x1FFC for result[1023]) ───────────
    logic [12:0]  s_axil_awaddr;
    logic         s_axil_awvalid;
    logic         s_axil_awready;
    logic [31:0]  s_axil_wdata;
    logic [3:0]   s_axil_wstrb;
    logic         s_axil_wvalid;
    logic         s_axil_wready;
    logic [1:0]   s_axil_bresp;
    logic         s_axil_bvalid;
    logic         s_axil_bready;
    logic [12:0]  s_axil_araddr;
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

    // ── BF16 stream beat constants (T1 vector) ────────────────────────────────
    // weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] in BF16
    // tdata[16*i+15:16*i] = BF16(weights[i]), i = 0..8; upper bits zero.
    localparam [511:0] WEIGHT_BEAT = {368'h0,
        16'h4110, 16'h4100, 16'h40e0, 16'h40c0,
        16'h40a0, 16'h4080, 16'h4040, 16'h4000, 16'h3f80};

    // IFM tile: 34×34 all-1.0 (BF16 0x3f80) packed flat row-major.
    // Beat k carries flat[k*32 : k*32+31].  1156 values → 37 beats.
    // Beats 0-35: 32 × BF16(1.0) each.  Beat 36: 4 × BF16(1.0), upper zero.
    localparam [511:0] IFM_FULL_BEAT = {32{16'h3f80}};   // beats 0-35
    localparam [511:0] IFM_LAST_BEAT = {448'h0,           // beat 36 (4 values)
        16'h3f80, 16'h3f80, 16'h3f80, 16'h3f80};

    // T1 expected: dot([1..9],[1.0]*9) = 45.0 → FP32 0x42340000 (all 1024 tiles)
    localparam [31:0] EXPECTED_RESULT = 32'h42340000;

    // ── AXI-Lite write (4 cycles) ────────────────────────────────────────────
    // Driven #1ps after posedge to avoid testbench/DUT delta-cycle race.
    task axil_write;
        input [12:0] addr;
        input [31:0] data;
        begin
            @(posedge clk); #1;
            s_axil_awaddr  = addr;
            s_axil_awvalid = 1'b1;
            s_axil_wdata   = data;
            s_axil_wstrb   = 4'hF;
            s_axil_wvalid  = 1'b1;

            @(posedge clk); #1;        // DUT samples AW + W
            s_axil_awvalid = 1'b0;
            s_axil_wvalid  = 1'b0;

            @(posedge clk); #1;        // b_pend=1 → bvalid=1
            s_axil_bready  = 1'b1;

            @(posedge clk); #1;        // DUT clears b_pend
            s_axil_bready  = 1'b0;
        end
    endtask

    // ── AXI-Lite read (4 cycles) ─────────────────────────────────────────────
    task axil_read;
        input  [12:0] addr;
        output [31:0] data;
        begin
            @(posedge clk); #1;
            s_axil_araddr  = addr;
            s_axil_arvalid = 1'b1;

            @(posedge clk); #1;        // DUT samples AR; rdata latched
            s_axil_arvalid = 1'b0;

            @(posedge clk); #1;        // r_pend=1 → rvalid=1; rdata stable
            data           = s_axil_rdata;
            s_axil_rready  = 1'b1;

            @(posedge clk); #1;        // DUT clears r_pend
            s_axil_rready  = 1'b0;
        end
    endtask

    // ── AXI-Stream single beat (2 cycles, tready always 1) ───────────────────
    task send_stream;
        input [511:0] beat;
        begin
            @(posedge clk); #1;
            s_axis_tdata  = beat;
            s_axis_tvalid = 1'b1;
            s_axis_tlast  = 1'b1;

            @(posedge clk); #1;
            s_axis_tvalid = 1'b0;
            s_axis_tlast  = 1'b0;
        end
    endtask

    // ── 37-beat IFM tile sender ───────────────────────────────────────────────
    // Sends a 34×34 all-1.0 IFM tile as 37 back-to-back 512-bit beats.
    // Beats 0-35: IFM_FULL_BEAT (32 × BF16(1.0)).
    // Beat 36: IFM_LAST_BEAT (4 × BF16(1.0), upper zero), tlast=1.
    task send_ifm_tile;
        integer j;
        begin
            for (j = 0; j < 36; j = j + 1) begin
                @(posedge clk); #1;
                s_axis_tdata  = IFM_FULL_BEAT;
                s_axis_tvalid = 1'b1;
                s_axis_tlast  = 1'b0;
                @(posedge clk); #1;
                s_axis_tvalid = 1'b0;
                s_axis_tlast  = 1'b0;
            end
            @(posedge clk); #1;        // beat 36: final 4 values + tlast
            s_axis_tdata  = IFM_LAST_BEAT;
            s_axis_tvalid = 1'b1;
            s_axis_tlast  = 1'b1;
            @(posedge clk); #1;
            s_axis_tvalid = 1'b0;
            s_axis_tlast  = 1'b0;
        end
    endtask

    // ── Read all 1024 FP32 results from 0x1000–0x1FFC ────────────────────────
    // Returns tile-0 result and count of tiles that deviate from EXPECTED_RESULT.
    task read_all_results;
        output [31:0]  tile0;
        output integer err_count;
        integer        t;
        logic   [31:0] r;
        begin
            tile0     = 32'h0;
            err_count = 0;
            for (t = 0; t < 1024; t = t + 1) begin
                axil_read(13'h1000 + t * 4, r);
                if (t == 0) tile0 = r;
                if (r !== EXPECTED_RESULT) err_count = err_count + 1;
            end
        end
    endtask

    // ── stimulus ──────────────────────────────────────────────────────────────
    integer k;
    logic [31:0] status;
    logic [31:0] tile0_result;
    integer      failed_tiles;

    initial begin
        // initialise driven signals
        rst_n          = 1'b0;
        s_axil_awaddr  = 13'h0;
        s_axil_awvalid = 1'b0;
        s_axil_wdata   = 32'h0;
        s_axil_wstrb   = 4'h0;
        s_axil_wvalid  = 1'b0;
        s_axil_bready  = 1'b0;
        s_axil_araddr  = 13'h0;
        s_axil_arvalid = 1'b0;
        s_axil_rready  = 1'b0;
        s_axis_tdata   = 512'h0;
        s_axis_tvalid  = 1'b0;
        s_axis_tlast   = 1'b0;
        status         = 32'h0;
        tile0_result   = 32'h0;
        failed_tiles   = 0;

        repeat (4) @(posedge clk); #1;
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // ====================================================================
        // REGION 1: WEIGHT LOAD
        //   CTRL[1]=1 sets load_weights; single stream beat → wl_capture.
        //   Weight SRAM FSM takes 9 cycles; poll STATUS[2] = wt_loaded.
        // ====================================================================
        $display("[%0t ns] ===== REGION 1: Weight Load =====", $time);

        axil_write(13'h000, 32'h2);     // CTRL[1] = load_weights
        send_stream(WEIGHT_BEAT);        // BF16 [1..9] → wl_capture

        status = 32'h0;
        for (k = 0; k < 30; k = k + 1) begin
            axil_read(13'h004, status);
            if (status[2]) k = 30;      // STATUS[2] = wt_loaded
        end
        $display("[%0t ns]   wt_loaded=%b  STATUS=0x%08h", $time, status[2], status);

        // ====================================================================
        // REGION 2: IFM LOAD  (37 stream beats)
        //   ifm_buffer accepts 37 × 512-bit beats and asserts load_done.
        //   core_start is gated on ifm_load_done; poll STATUS[4] before start.
        // ====================================================================
        $display("[%0t ns] ===== REGION 2: IFM Load (37 beats) =====", $time);

        send_ifm_tile();                 // 34×34 all-1.0 IFM, back-to-back

        status = 32'h0;
        for (k = 0; k < 100; k = k + 1) begin
            axil_read(13'h004, status);
            if (status[4]) k = 100;     // STATUS[4] = ifm_load_done
        end
        $display("[%0t ns]   ifm_loaded=%b  STATUS=0x%08h", $time, status[4], status);

        // ====================================================================
        // REGION 3: COMPUTE
        //   CTRL[0]=1 starts all 1024 pe_array tiles simultaneously.
        //   pe_array takes 4 cycles; done_all latches all 1024 results.
        //   Poll STATUS[0] = done.
        // ====================================================================
        $display("[%0t ns] ===== REGION 3: Compute (1024 tiles) =====", $time);

        axil_write(13'h000, 32'h1);     // CTRL[0] = start

        status = 32'h0;
        for (k = 0; k < 50; k = k + 1) begin
            axil_read(13'h004, status);
            if (status[0]) k = 50;      // STATUS[0] = done
        end
        $display("[%0t ns]   done=%b  STATUS=0x%08h", $time, status[0], status);

        // ====================================================================
        // REGION 4: READ ALL 1024 RESULTS  (0x1000 + t*4, t = 0..1023)
        //   T1: all tiles expected = 45.0 → FP32 0x42340000.
        // ====================================================================
        $display("[%0t ns] ===== REGION 4: Read All 1024 Results =====", $time);

        read_all_results(tile0_result, failed_tiles);

        $display("[%0t ns]   tile[0]=0x%08h  (%0d/1024 mismatches)",
                 $time, tile0_result, failed_tiles);

        if (failed_tiles == 0)
            $display("[%0t ns]   PASS — all 1024 tiles = 45.0 (0x42340000)", $time);
        else
            $display("[%0t ns]   FAIL — %0d tiles did not match 0x42340000",
                     $time, failed_tiles);

        repeat (10) @(posedge clk);
        $display("[%0t ns] ===== SIMULATION COMPLETE =====", $time);
        $finish;
    end

endmodule
