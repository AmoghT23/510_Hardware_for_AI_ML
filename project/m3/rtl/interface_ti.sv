`timescale 1ns/1ps

// ============================================================================
// interface_ti.sv  —  Conv2d Interface  (Training + Inference, Phase 3)
// ECE 410/510
//
// Phase 3 additions over Phase 2:
//   Backward pass  : instantiates grad_core; triggered by CTRL[2]=1
//   GRAD_IN reg    : host writes FP32 dL/dy to 0x10 before asserting CTRL[2]
//   GRAD_W regs    : 9 FP32 weight-gradient results readable at 0x14-0x34
//   STATUS[3]      : grad_done — asserted when grad_core completes
//   AXIL_ADDR_W    : widened 4→6 bits to reach 0x34
//
// Register map (AXI4-Lite, 6-bit address):
//   0x00  CTRL      W   [0] start        — triggers sequencer + compute
//                       [1] load_weights — next stream beat captured as weights
//                       [2] backward     — triggers grad_core (needs GRAD_IN set)
//   0x04  STATUS    R   [0] done         — forward result ready (clears on read)
//                       [1] busy         — core not idle
//                       [2] wt_loaded    — weight SRAM valid
//                       [3] grad_done    — backward result ready (clears on read)
//   0x08  TILE_LEN  RW  [7:0]
//   0x0C  RESULT    R   [31:0] FP32 dot-product
//   0x10  GRAD_IN   RW  [31:0] FP32 dL/dy (upstream gradient)
//   0x14  GRAD_W0   R   [31:0] dL/dW[0]
//   0x18  GRAD_W1   R   [31:0] dL/dW[1]
//   0x1C  GRAD_W2   R   [31:0] dL/dW[2]
//   0x20  GRAD_W3   R   [31:0] dL/dW[3]
//   0x24  GRAD_W4   R   [31:0] dL/dW[4]
//   0x28  GRAD_W5   R   [31:0] dL/dW[5]
//   0x2C  GRAD_W6   R   [31:0] dL/dW[6]
//   0x30  GRAD_W7   R   [31:0] dL/dW[7]
//   0x34  GRAD_W8   R   [31:0] dL/dW[8]
//
// Host programming sequence (backward pass):
//   After a forward pass (weights + IFM loaded):
//   1. Write GRAD_IN (0x10) with FP32 dL/dy
//   2. Write CTRL[2]=1
//   3. Poll STATUS[3] until grad_done=1
//   4. Read GRAD_W0..GRAD_W8 (0x14..0x34)
// ============================================================================

module conv_interface_ti #(
    parameter int AXIS_DATA_W = 512,
    parameter int AXIL_ADDR_W = 6,
    parameter int AXIL_DATA_W = 32
) (
    input  logic clk,
    input  logic rst_n,

    input  logic [AXIL_ADDR_W-1:0] s_axil_awaddr,
    input  logic                    s_axil_awvalid,
    output logic                    s_axil_awready,

    input  logic [AXIL_DATA_W-1:0] s_axil_wdata,
    input  logic [3:0]              s_axil_wstrb,
    input  logic                    s_axil_wvalid,
    output logic                    s_axil_wready,

    output logic [1:0]              s_axil_bresp,
    output logic                    s_axil_bvalid,
    input  logic                    s_axil_bready,

    input  logic [AXIL_ADDR_W-1:0] s_axil_araddr,
    input  logic                    s_axil_arvalid,
    output logic                    s_axil_arready,

    output logic [AXIL_DATA_W-1:0] s_axil_rdata,
    output logic [1:0]              s_axil_rresp,
    output logic                    s_axil_rvalid,
    input  logic                    s_axil_rready,

    input  logic [AXIS_DATA_W-1:0] s_axis_tdata,
    input  logic                    s_axis_tvalid,
    output logic                    s_axis_tready,
    input  logic                    s_axis_tlast,

    output logic        core_start,
    output logic [7:0]  core_tile_len,
    output logic [15:0] core_weight_data,
    output logic        core_weight_valid,
    output logic [15:0] core_ifm_data,
    output logic        core_ifm_valid,
    input  logic [31:0] core_result,
    input  logic        core_done,
    input  logic        core_ready
);

    // ── internal registers ────────────────────────────────────────────────────
    logic        reg_start;
    logic        reg_load_weights;
    logic        reg_done;
    logic        reg_weights_loaded;
    logic [7:0]  reg_tile_len;
    logic [31:0] reg_result;
    logic [15:0] reg_weight_data;
    logic        reg_weight_valid;
    logic [15:0] reg_ifm_data;
    logic        reg_ifm_valid;

    // ── Phase 3: backward pass registers ─────────────────────────────────────
    logic        reg_backward;          // auto-clears after one cycle
    logic [31:0] reg_grad_in;           // dL/dy written by host
    logic [31:0] reg_grad_w [0:8];      // dL/dW results from grad_core
    logic        reg_grad_done;         // set when grad_core completes

    // ── weight SRAM (9 × 16-bit BF16, persistent across tiles) ───────────────
    logic        sram_we;
    logic [3:0]  sram_addr;
    logic [15:0] sram_wdata;
    logic [15:0] sram_rdata;

    sram_sp #(.DEPTH(9), .WIDTH(16)) u_weight_sram (
        .clk   (clk),
        .we    (sram_we),
        .addr  (sram_addr),
        .wdata (sram_wdata),
        .rdata (sram_rdata)
    );

    // Weight write state machine
    logic [3:0]  wl_cnt;
    logic        wl_active;
    logic [15:0] wl_capture [0:8];

    // ── IFM buffer ────────────────────────────────────────────────────────────
    logic [15:0] ifm_buf [0:8];

    // ── grad_core interface ───────────────────────────────────────────────────
    logic [143:0] weights_flat;   // packed 9×BF16 from wl_capture
    logic [143:0] ifms_flat;      // packed 9×BF16 from ifm_buf
    logic [287:0] grad_w_flat;    // packed 9×FP32 weight gradients
    logic [287:0] grad_x_flat;    // packed 9×FP32 activation gradients (unused by host)
    logic         grad_done_pulse;
    logic         grad_ready;

    // Pack wl_capture and ifm_buf into flat vectors for grad_core
    always_comb begin
        for (int gi = 0; gi < 9; gi++) begin
            weights_flat[gi*16 +: 16] = wl_capture[gi];
            ifms_flat   [gi*16 +: 16] = ifm_buf[gi];
        end
    end

    grad_core u_grad_core (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (reg_backward),
        .grad_out    (reg_grad_in),
        .weights_flat(weights_flat),
        .ifms_flat   (ifms_flat),
        .grad_w_flat (grad_w_flat),
        .grad_x_flat (grad_x_flat),
        .done        (grad_done_pulse),
        .ready       (grad_ready)
    );

    // ── sequencer ─────────────────────────────────────────────────────────────
    localparam logic SEQ_IDLE  = 1'b0;
    localparam logic SEQ_DRIVE = 1'b1;
    logic       seq_state;
    logic [3:0] seq_cnt;

    // ── AXI4-Lite write path ──────────────────────────────────────────────────
    logic                    aw_latch;
    logic [AXIL_ADDR_W-1:0] aw_addr;
    logic                    w_latch;
    logic [AXIL_DATA_W-1:0] w_data;
    logic                    b_pend;
    logic                    r_pend;

    assign s_axil_awready = !aw_latch && !b_pend;
    assign s_axil_wready  = !w_latch  && !b_pend;
    assign s_axil_bresp   = 2'b00;
    assign s_axil_bvalid  = b_pend;
    assign s_axil_arready = !r_pend;
    assign s_axil_rresp   = 2'b00;
    assign s_axil_rvalid  = r_pend;
    assign s_axis_tready  = 1'b1;

    logic [AXIL_ADDR_W-1:0] wr_addr;
    logic [AXIL_DATA_W-1:0] wr_data;
    assign wr_addr = aw_latch ? aw_addr : s_axil_awaddr;
    assign wr_data = w_latch  ? w_data  : s_axil_wdata;

    assign core_start        = reg_start & reg_weights_loaded;
    assign core_tile_len     = reg_tile_len;
    assign core_weight_data  = reg_weight_data;
    assign core_weight_valid = reg_weight_valid;
    assign core_ifm_data     = reg_ifm_data;
    assign core_ifm_valid    = reg_ifm_valid;

    // ── SRAM port mux ────────────────────────────────────────────────────────
    logic [3:0] sram_rd_addr;
    always_comb begin
        if (seq_state == SEQ_DRIVE)
            sram_rd_addr = (seq_cnt >= 4'd8) ? 4'd0 : (seq_cnt + 4'd1);
        else
            sram_rd_addr = 4'd0;
    end

    always_comb begin
        if (wl_active) begin
            sram_we    = 1'b1;
            sram_addr  = wl_cnt;
            sram_wdata = wl_capture[wl_cnt];
        end else begin
            sram_we    = 1'b0;
            sram_addr  = sram_rd_addr;
            sram_wdata = 16'h0;
        end
    end

    // ── registered logic ──────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            aw_latch          <= 1'b0;  aw_addr  <= '0;
            w_latch           <= 1'b0;  w_data   <= '0;
            b_pend            <= 1'b0;
            r_pend            <= 1'b0;
            s_axil_rdata      <= '0;
            reg_start         <= 1'b0;
            reg_load_weights  <= 1'b0;
            reg_done          <= 1'b0;
            reg_weights_loaded<= 1'b0;
            reg_tile_len      <= 8'd0;
            reg_result        <= 32'd0;
            reg_weight_data   <= 16'h0;
            reg_weight_valid  <= 1'b0;
            reg_ifm_data      <= 16'h0;
            reg_ifm_valid     <= 1'b0;
            reg_backward      <= 1'b0;
            reg_grad_in       <= 32'd0;
            reg_grad_done     <= 1'b0;
            seq_state         <= SEQ_IDLE;
            seq_cnt           <= 4'd0;
            wl_active         <= 1'b0;
            wl_cnt            <= 4'd0;
            for (int i = 0; i < 9; i++) begin
                ifm_buf[i]    <= 16'h0;
                wl_capture[i] <= 16'h0;
                reg_grad_w[i] <= 32'd0;
            end
        end else begin

            reg_start    <= 1'b0;   // auto-clears after one cycle
            reg_backward <= 1'b0;   // auto-clears after one cycle

            if (core_done) begin
                reg_done   <= 1'b1;
                reg_result <= core_result;
            end

            // Latch grad_core results when done pulse arrives
            if (grad_done_pulse) begin
                reg_grad_done <= 1'b1;
                for (int i = 0; i < 9; i++)
                    reg_grad_w[i] <= grad_w_flat[i*32 +: 32];
            end

            // ── AXI4-Stream beat ─────────────────────────────────────────
            if (s_axis_tvalid) begin
                if (reg_load_weights) begin
                    for (int i = 0; i < 9; i++)
                        wl_capture[i] <= s_axis_tdata[i*16 +: 16];
                    wl_active        <= 1'b1;
                    wl_cnt           <= 4'd0;
                    reg_load_weights <= 1'b0;
                end else begin
                    for (int i = 0; i < 9; i++)
                        ifm_buf[i] <= s_axis_tdata[i*16 +: 16];
                end
            end

            // ── Weight SRAM write state machine ───────────────────────────
            if (wl_active) begin
                if (wl_cnt == 4'd8) begin
                    wl_active          <= 1'b0;
                    reg_weights_loaded <= 1'b1;
                end else begin
                    wl_cnt <= wl_cnt + 1'b1;
                end
            end

            // ── Data sequencer ────────────────────────────────────────────
            case (seq_state)
                SEQ_IDLE: begin
                    if (reg_start && reg_weights_loaded) begin
                        seq_state        <= SEQ_DRIVE;
                        seq_cnt          <= 4'd0;
                        reg_weight_data  <= sram_rdata;
                        reg_ifm_data     <= ifm_buf[0];
                        reg_weight_valid <= 1'b1;
                        reg_ifm_valid    <= 1'b1;
                    end
                end
                SEQ_DRIVE: begin
                    if (seq_cnt == 4'd8) begin
                        seq_state        <= SEQ_IDLE;
                        seq_cnt          <= 4'd0;
                        reg_weight_valid <= 1'b0;
                        reg_ifm_valid    <= 1'b0;
                        reg_weight_data  <= 16'h0;
                        reg_ifm_data     <= 16'h0;
                    end else begin
                        seq_cnt         <= seq_cnt + 1'b1;
                        reg_weight_data <= sram_rdata;
                        reg_ifm_data    <= ifm_buf[seq_cnt + 1];
                    end
                end
                default: seq_state <= SEQ_IDLE;
            endcase

            // ── AXI4-Lite write address latch ─────────────────────────────
            if (s_axil_awvalid && s_axil_awready) begin
                aw_latch <= 1'b1;
                aw_addr  <= s_axil_awaddr;
            end

            if (s_axil_wvalid && s_axil_wready) begin
                w_latch <= 1'b1;
                w_data  <= s_axil_wdata;
            end

            if ((aw_latch || (s_axil_awvalid && s_axil_awready)) &&
                (w_latch  || (s_axil_wvalid  && s_axil_wready ))) begin
                case (wr_addr[5:2])
                    4'd0: begin   // 0x00 CTRL
                        if (wr_data[0]) reg_start         <= 1'b1;
                        if (wr_data[1]) begin
                            reg_load_weights   <= 1'b1;
                            reg_weights_loaded <= 1'b0;
                        end
                        if (wr_data[2]) reg_backward <= 1'b1;
                    end
                    4'd2: reg_tile_len <= wr_data[7:0];  // 0x08 TILE_LEN
                    4'd4: reg_grad_in  <= wr_data;        // 0x10 GRAD_IN
                    default: ;
                endcase
                aw_latch <= 1'b0;
                w_latch  <= 1'b0;
                b_pend   <= 1'b1;
            end

            if (b_pend && s_axil_bready)
                b_pend <= 1'b0;

            if (s_axil_arvalid && s_axil_arready) begin
                r_pend <= 1'b1;
                case (s_axil_araddr[5:2])
                    4'd0: s_axil_rdata <= {30'd0, 1'b0, reg_start};               // 0x00 CTRL
                    4'd1: begin                                                     // 0x04 STATUS
                        s_axil_rdata <= {28'd0, reg_grad_done, reg_weights_loaded,
                                         !core_ready, reg_done};
                        reg_done      <= 1'b0;
                        reg_grad_done <= 1'b0;
                    end
                    4'd2: s_axil_rdata <= {24'd0, reg_tile_len};                   // 0x08 TILE_LEN
                    4'd3: s_axil_rdata <= reg_result;                              // 0x0C RESULT
                    4'd4: s_axil_rdata <= reg_grad_in;                             // 0x10 GRAD_IN
                    4'd5: s_axil_rdata <= reg_grad_w[0];                           // 0x14 GRAD_W0
                    4'd6: s_axil_rdata <= reg_grad_w[1];                           // 0x18 GRAD_W1
                    4'd7: s_axil_rdata <= reg_grad_w[2];                           // 0x1C GRAD_W2
                    4'd8: s_axil_rdata <= reg_grad_w[3];                           // 0x20 GRAD_W3
                    4'd9: s_axil_rdata <= reg_grad_w[4];                           // 0x24 GRAD_W4
                    4'd10: s_axil_rdata <= reg_grad_w[5];                          // 0x28 GRAD_W5
                    4'd11: s_axil_rdata <= reg_grad_w[6];                          // 0x2C GRAD_W6
                    4'd12: s_axil_rdata <= reg_grad_w[7];                          // 0x30 GRAD_W7
                    4'd13: s_axil_rdata <= reg_grad_w[8];                          // 0x34 GRAD_W8
                    default: s_axil_rdata <= 32'd0;
                endcase
            end

            if (r_pend && s_axil_rready)
                r_pend <= 1'b0;

        end
    end

endmodule
