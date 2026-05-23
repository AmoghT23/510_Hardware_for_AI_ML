`timescale 1ns/1ps

// ============================================================================
// interface.sv  —  Conv2d Co-processor Chiplet Interface Module
// ECE 410/510 — Anemia Detection Accelerator, Milestone 3
//
// Copied from M2 and updated for M3 INT8 compute_core.
// Change log (M2 → M3):
//   Task 1: core_result widened from [15:0] FP16 → [31:0] INT32 to match
//           M3 compute_core result port.
//   Task 2: core_weight_data[7:0] and core_weight_valid output ports added.
//   Task 3: core_ifm_data[7:0] and core_ifm_valid output ports added.
//   Task 4: Data sequencer implemented. One 512-bit AXI4-Stream beat carries
//           9 weight+IFM INT8 pairs (tdata[16i+7:16i]=weight[i],
//           tdata[16i+15:16i+8]=ifm[i], i=0..8). On the CTRL start pulse the
//           sequencer drives one pair per cycle to the compute_core for 9
//           consecutive cycles, matching its LOAD-state handshake.
//
// Implements the M1-selected interface:
//   Data plane  — AXI4-Stream slave, 512-bit data width
//                 (AMBA AXI4-Stream Protocol Spec IHI0051, TVALID/TREADY contract)
//   Control plane — AXI4-Lite slave, 32-bit data, 4-bit address
//                 (AMBA AXI4-Lite Spec IHI0022, handshake on every channel)
//
// ── Register map (AXI4-Lite, 4-bit address space) ──────────────────────────
//   Addr  Name      Access  Bits     Description
//   0x00  CTRL      W       [0]      start    — write 1 to pulse core_start (auto-clears)
//                           [1]      sw_rst   — software reset request (auto-clears)
//   0x04  STATUS    R       [0]      done     — latched from core_done; clears on read
//                           [1]      busy     — asserted when core is not idle
//   0x08  TILE_LEN  RW      [7:0]   num_elem — number of MAC operations per tile
//   0x0C  RESULT    R       [31:0]  int32    — INT32 output pixel (latched on done)
//
// ── AXI4-Stream data packing (M3 INT8 format) ───────────────────────────────
//   One 512-bit beat carries 9 weight+IFM pairs for a 3x3x1 Conv2d tile:
//     tdata[16i+7  : 16i]   = weight[i]  (INT8 signed, i = 0..8)
//     tdata[16i+15 : 16i+8] = ifm[i]     (INT8 signed, i = 0..8)
//     tdata[511:144]         = unused (zero-padded by host)
//   TREADY is held high; the interface always accepts incoming beats.
//   TLAST must be asserted on the single tile beat.
//
// Ports:
//   clk              in    1    System clock (target 1 GHz)
//   rst_n            in    1    Synchronous active-low reset
//   s_axil_awaddr    in    4    AXI4-Lite write address
//   s_axil_awvalid   in    1    Write address valid
//   s_axil_awready   out   1    Write address ready
//   s_axil_wdata     in   32    Write data
//   s_axil_wstrb     in    4    Write byte strobes (unused; all bytes written)
//   s_axil_wvalid    in    1    Write data valid
//   s_axil_wready    out   1    Write data ready
//   s_axil_bresp     out   2    Write response (2'b00 = OKAY)
//   s_axil_bvalid    out   1    Write response valid
//   s_axil_bready    in    1    Write response ready
//   s_axil_araddr    in    4    AXI4-Lite read address
//   s_axil_arvalid   in    1    Read address valid
//   s_axil_arready   out   1    Read address ready
//   s_axil_rdata     out  32    Read data
//   s_axil_rresp     out   2    Read response (2'b00 = OKAY)
//   s_axil_rvalid    out   1    Read data valid
//   s_axil_rready    in    1    Read data ready
//   s_axis_tdata     in  512    AXI4-Stream packed data beat
//   s_axis_tvalid    in    1    Stream data valid
//   s_axis_tready    out   1    Stream ready (always 1)
//   s_axis_tlast     in    1    Last beat of packet
//   core_start       out   1    Forwarded start pulse to compute_core
//   core_tile_len    out   8    Tile length forwarded to compute_core
//   core_weight_data  out  8    INT8 weight[i] driven by sequencer (cycles 0..8)
//   core_weight_valid out  1    Asserted for 9 cycles during LOAD
//   core_ifm_data     out  8    INT8 ifm[i] driven by sequencer   (cycles 0..8)
//   core_ifm_valid    out  1    Asserted for 9 cycles during LOAD
//   core_result      in   32    INT32 result from compute_core (latched on done)
//   core_done        in    1    Done pulse from compute_core
//   core_ready       in    1    Idle flag from compute_core
//
// Clock domain : single (clk)
// Reset        : synchronous, active-low (rst_n = 0 clears all registers)
// ============================================================================

module conv_interface #(
    parameter int AXIS_DATA_W = 512,
    parameter int AXIL_ADDR_W = 4,
    parameter int AXIL_DATA_W = 32
) (
    input  logic clk,
    input  logic rst_n,

    // ── AXI4-Lite write address channel ─────────────────────────────────
    input  logic [AXIL_ADDR_W-1:0] s_axil_awaddr,
    input  logic                    s_axil_awvalid,
    output logic                    s_axil_awready,

    // ── AXI4-Lite write data channel ────────────────────────────────────
    input  logic [AXIL_DATA_W-1:0] s_axil_wdata,
    input  logic [3:0]              s_axil_wstrb,
    input  logic                    s_axil_wvalid,
    output logic                    s_axil_wready,

    // ── AXI4-Lite write response channel ────────────────────────────────
    output logic [1:0]              s_axil_bresp,
    output logic                    s_axil_bvalid,
    input  logic                    s_axil_bready,

    // ── AXI4-Lite read address channel ──────────────────────────────────
    input  logic [AXIL_ADDR_W-1:0] s_axil_araddr,
    input  logic                    s_axil_arvalid,
    output logic                    s_axil_arready,

    // ── AXI4-Lite read data channel ─────────────────────────────────────
    output logic [AXIL_DATA_W-1:0] s_axil_rdata,
    output logic [1:0]              s_axil_rresp,
    output logic                    s_axil_rvalid,
    input  logic                    s_axil_rready,

    // ── AXI4-Stream slave (data plane) ──────────────────────────────────
    input  logic [AXIS_DATA_W-1:0] s_axis_tdata,
    input  logic                    s_axis_tvalid,
    output logic                    s_axis_tready,
    input  logic                    s_axis_tlast,

    // ── Compute core interface ───────────────────────────────────────────
    output logic        core_start,
    output logic [7:0]  core_tile_len,
    output logic [7:0]  core_weight_data,   // INT8 weight to core  (task 4: sequencer)
    output logic        core_weight_valid,  // weight valid strobe  (task 4: sequencer)
    output logic [7:0]  core_ifm_data,      // INT8 IFM to core     (task 4: sequencer)
    output logic        core_ifm_valid,     // IFM valid strobe     (task 4: sequencer)
    input  logic [31:0] core_result,        // INT32 (widened from [15:0] FP16 in M2)
    input  logic        core_done,
    input  logic        core_ready
);

    // ── internal registers ────────────────────────────────────────────────
    logic        reg_start;        // auto-clears after one cycle
    logic        reg_done;         // latched from core_done, clears on STATUS read
    logic [7:0]  reg_tile_len;     // tile length register
    logic [31:0] reg_result;       // INT32 output pixel latch
    logic [15:0] stream_buf;       // first 16 bits of last stream beat
    logic [7:0]  reg_weight_data;
    logic        reg_weight_valid;
    logic [7:0]  reg_ifm_data;
    logic        reg_ifm_valid;

    // ── data sequencer ────────────────────────────────────────────────────
    // seq_buf[i][7:0]  = weight[i], seq_buf[i][15:8] = ifm[i]  (i = 0..8)
    localparam logic SEQ_IDLE  = 1'b0;
    localparam logic SEQ_DRIVE = 1'b1;
    logic        seq_state;
    logic [3:0]  seq_cnt;           // counts pairs driven: 0..8
    logic [15:0] seq_buf [0:8];     // 9 packed pairs from AXI4-Stream beat

    // ── AXI4-Lite write path state ────────────────────────────────────────
    logic                    aw_latch;  // write address pending
    logic [AXIL_ADDR_W-1:0] aw_addr;
    logic                    w_latch;   // write data pending
    logic [AXIL_DATA_W-1:0] w_data;
    logic                    b_pend;    // write response pending

    // ── AXI4-Lite read path state ─────────────────────────────────────────
    logic r_pend;  // read response pending

    // ── combinational output assignments ──────────────────────────────────
    assign s_axil_awready = !aw_latch && !b_pend;
    assign s_axil_wready  = !w_latch  && !b_pend;
    assign s_axil_bresp   = 2'b00;
    assign s_axil_bvalid  = b_pend;

    assign s_axil_arready = !r_pend;
    assign s_axil_rresp   = 2'b00;
    assign s_axil_rvalid  = r_pend;

    assign s_axis_tready  = 1'b1;   // always ready; no backpressure

    // ── write mux: prefer latched values; fall back to current-cycle bus ──
    logic [AXIL_ADDR_W-1:0] wr_addr;
    logic [AXIL_DATA_W-1:0] wr_data;
    assign wr_addr = aw_latch ? aw_addr : s_axil_awaddr;
    assign wr_data = w_latch  ? w_data  : s_axil_wdata;

    assign core_start        = reg_start;
    assign core_tile_len     = reg_tile_len;
    assign core_weight_data  = reg_weight_data;
    assign core_weight_valid = reg_weight_valid;
    assign core_ifm_data     = reg_ifm_data;
    assign core_ifm_valid    = reg_ifm_valid;

    // ── registered logic ──────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            aw_latch     <= 1'b0;  aw_addr  <= '0;
            w_latch      <= 1'b0;  w_data   <= '0;
            b_pend       <= 1'b0;
            r_pend       <= 1'b0;
            s_axil_rdata <= '0;
            reg_start        <= 1'b0;
            reg_done         <= 1'b0;
            reg_tile_len     <= 8'd0;
            reg_result       <= 32'd0;
            stream_buf       <= 16'd0;
            reg_weight_data  <= 8'h0;
            reg_weight_valid <= 1'b0;
            reg_ifm_data     <= 8'h0;
            reg_ifm_valid    <= 1'b0;
            seq_state        <= SEQ_IDLE;
            seq_cnt          <= 4'd0;
            for (int i = 0; i < 9; i++) seq_buf[i] <= 16'h0;
        end else begin

            // ── auto-clear start pulse ────────────────────────────────
            reg_start <= 1'b0;

            // ── latch full INT32 core result on done ──────────────────
            if (core_done) begin
                reg_done   <= 1'b1;
                reg_result <= core_result;  // full 32-bit INT32
            end

            // ── AXI4-Stream beat capture: unpack 9 INT8 weight+IFM pairs ─
            // tdata[16i+7:16i]=weight[i], tdata[16i+15:16i+8]=ifm[i]
            if (s_axis_tvalid) begin
                stream_buf <= s_axis_tdata[15:0];
                for (int i = 0; i < 9; i++)
                    seq_buf[i] <= s_axis_tdata[i*16 +: 16];
            end

            // ── data sequencer: drives weight/IFM to core after start ─────
            // SEQ_IDLE  → detects reg_start=1 (same cycle it was committed),
            //             pre-loads pair[0] and transitions to SEQ_DRIVE.
            // SEQ_DRIVE → drives pair[seq_cnt] this cycle; advances to next
            //             pair; deasserts valid after pair[8] is delivered.
            case (seq_state)
                SEQ_IDLE: begin
                    if (reg_start) begin
                        seq_state        <= SEQ_DRIVE;
                        seq_cnt          <= 4'd0;
                        reg_weight_data  <= seq_buf[0][7:0];
                        reg_ifm_data     <= seq_buf[0][15:8];
                        reg_weight_valid <= 1'b1;
                        reg_ifm_valid    <= 1'b1;
                    end
                end
                SEQ_DRIVE: begin
                    if (seq_cnt == 4'd8) begin
                        seq_state        <= SEQ_IDLE;
                        reg_weight_valid <= 1'b0;
                        reg_ifm_valid    <= 1'b0;
                        reg_weight_data  <= 8'h0;
                        reg_ifm_data     <= 8'h0;
                    end else begin
                        seq_cnt         <= seq_cnt + 1'b1;
                        reg_weight_data <= seq_buf[seq_cnt + 1][7:0];
                        reg_ifm_data    <= seq_buf[seq_cnt + 1][15:8];
                    end
                end
                default: seq_state <= SEQ_IDLE;
            endcase

            // ── AXI4-Lite write address latch ─────────────────────────
            if (s_axil_awvalid && s_axil_awready) begin
                aw_latch <= 1'b1;
                aw_addr  <= s_axil_awaddr;
            end

            // ── AXI4-Lite write data latch ────────────────────────────
            if (s_axil_wvalid && s_axil_wready) begin
                w_latch <= 1'b1;
                w_data  <= s_axil_wdata;
            end

            // ── perform write when both AW and W are present ──────────
            if ((aw_latch || (s_axil_awvalid && s_axil_awready)) &&
                (w_latch  || (s_axil_wvalid  && s_axil_wready ))) begin

                case (wr_addr[3:2])
                    2'b00: begin          // 0x00 CTRL
                        if (wr_data[0]) reg_start <= 1'b1;
                    end
                    2'b10: begin          // 0x08 TILE_LEN
                        reg_tile_len <= wr_data[7:0];
                    end
                    default: ;            // 0x04 STATUS, 0x0C RESULT are read-only
                endcase

                aw_latch <= 1'b0;
                w_latch  <= 1'b0;
                b_pend   <= 1'b1;
            end

            // ── clear write response after BREADY ─────────────────────
            if (b_pend && s_axil_bready) begin
                b_pend <= 1'b0;
            end

            // ── AXI4-Lite read: accept address, register response ─────
            if (s_axil_arvalid && s_axil_arready) begin
                r_pend <= 1'b1;
                case (s_axil_araddr[3:2])
                    2'b00: s_axil_rdata <= {30'd0, 1'b0, reg_start};
                    2'b01: begin
                        s_axil_rdata <= {30'd0, !core_ready, reg_done};
                        reg_done     <= 1'b0;  // clear done flag on STATUS read
                    end
                    2'b10: s_axil_rdata <= {24'd0, reg_tile_len};
                    2'b11: s_axil_rdata <= reg_result;  // full 32-bit INT32
                    default: s_axil_rdata <= 32'd0;
                endcase
            end

            // ── clear read response after RREADY ──────────────────────
            if (r_pend && s_axil_rready) begin
                r_pend <= 1'b0;
            end

        end
    end

endmodule
