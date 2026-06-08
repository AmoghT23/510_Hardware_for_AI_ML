`timescale 1ns/1ps

// ============================================================================
// top.sv  —  Integrated Top Module  (M4: BF16 Conv2d Training + Inference)
// ECE 410/510  |  SAED14nm RVT  |  Cadence Genus 17.14-s037_1  |  500 MHz
//
// Instantiates conv_interface_ti (AXI host interface + grad_core) and
// compute_core_ti (BF16 dot-product MAC array) and connects all inter-module
// signals.  No glue logic is required: the interface drives core_* wires
// directly into the compute core.
//
// ── Port list ─────────────────────────────────────────────────────────────────
//  Name               Dir    Width  Role
//  -----------------  -----  -----  -----------------------------------------
//  clk                in      1     System clock (all registers)
//  rst_n              in      1     Active-low synchronous reset
//
//  s_axil_awaddr      in     13     AXI4-Lite write-address channel: address
//  s_axil_awvalid     in      1     AXI4-Lite write-address valid
//  s_axil_awready     out     1     AXI4-Lite write-address ready
//  s_axil_wdata       in     32     AXI4-Lite write-data channel: data
//  s_axil_wstrb       in      4     AXI4-Lite write-data byte enables
//  s_axil_wvalid      in      1     AXI4-Lite write-data valid
//  s_axil_wready      out     1     AXI4-Lite write-data ready
//  s_axil_bresp       out     2     AXI4-Lite write-response (always 2'b00)
//  s_axil_bvalid      out     1     AXI4-Lite write-response valid
//  s_axil_bready      in      1     AXI4-Lite write-response ready
//  s_axil_araddr      in     13     AXI4-Lite read-address channel: address
//  s_axil_arvalid     in      1     AXI4-Lite read-address valid
//  s_axil_arready     out     1     AXI4-Lite read-address ready
//  s_axil_rdata       out    32     AXI4-Lite read-data channel: data
//  s_axil_rresp       out     2     AXI4-Lite read-data response (always 2'b00)
//  s_axil_rvalid      out     1     AXI4-Lite read-data valid
//  s_axil_rready      in      1     AXI4-Lite read-data ready
//
//  s_axis_tdata       in    512     AXI4-Stream data: 9 × BF16 in [16i+15:16i]
//  s_axis_tvalid      in      1     AXI4-Stream valid
//  s_axis_tready      out     1     AXI4-Stream ready
//  s_axis_tlast       in      1     AXI4-Stream end-of-frame
//
// ── Inter-module signals (no glue logic) ──────────────────────────────────────
//  w_start            wire    1     interface → core: begin computation
//  w_tile_len         wire    8     interface → core: tile length
//  w_weight_data      wire   16     interface → core: BF16 weight word
//  w_weight_valid     wire    1     interface → core: weight word valid
//  w_ifm_data         wire   16     interface → core: BF16 IFM word
//  w_ifm_valid        wire    1     interface → core: IFM word valid
//  w_result           wire   32     core → interface: FP32 dot-product result
//  w_done             wire    1     core → interface: computation done
//  w_ready            wire    1     core → interface: core ready for next tile
// ============================================================================

module top_ti (
    input  logic        clk,
    input  logic        rst_n,

    input  logic [12:0] s_axil_awaddr,
    input  logic        s_axil_awvalid,
    output logic        s_axil_awready,

    input  logic [31:0] s_axil_wdata,
    input  logic [3:0]  s_axil_wstrb,
    input  logic        s_axil_wvalid,
    output logic        s_axil_wready,

    output logic [1:0]  s_axil_bresp,
    output logic        s_axil_bvalid,
    input  logic        s_axil_bready,

    input  logic [12:0] s_axil_araddr,
    input  logic        s_axil_arvalid,
    output logic        s_axil_arready,

    output logic [31:0] s_axil_rdata,
    output logic [1:0]  s_axil_rresp,
    output logic        s_axil_rvalid,
    input  logic        s_axil_rready,

    input  logic [511:0] s_axis_tdata,
    input  logic         s_axis_tvalid,
    output logic         s_axis_tready,
    input  logic         s_axis_tlast
);

    logic           w_start;
    logic [7:0]     w_tile_len;
    logic [143:0]   w_weights_packed;
    logic [147455:0] w_ifms_packed;      // 1024×144 bits
    logic [32767:0] w_result;            // 1024×32 bits
    logic [1023:0]  w_done;
    logic [1023:0]  w_ready;

    logic pe_done_all;
    logic pe_ready_all;
    assign w_done  = {1024{pe_done_all}};
    assign w_ready = {1024{pe_ready_all}};

    conv_interface_ti #(
        .AXIS_DATA_W (512),
        .AXIL_ADDR_W (13),
        .AXIL_DATA_W (32)
    ) u_interface (
        .clk                 (clk),
        .rst_n               (rst_n),
        .s_axil_awaddr       (s_axil_awaddr),
        .s_axil_awvalid      (s_axil_awvalid),
        .s_axil_awready      (s_axil_awready),
        .s_axil_wdata        (s_axil_wdata),
        .s_axil_wstrb        (s_axil_wstrb),
        .s_axil_wvalid       (s_axil_wvalid),
        .s_axil_wready       (s_axil_wready),
        .s_axil_bresp        (s_axil_bresp),
        .s_axil_bvalid       (s_axil_bvalid),
        .s_axil_bready       (s_axil_bready),
        .s_axil_araddr       (s_axil_araddr),
        .s_axil_arvalid      (s_axil_arvalid),
        .s_axil_arready      (s_axil_arready),
        .s_axil_rdata        (s_axil_rdata),
        .s_axil_rresp        (s_axil_rresp),
        .s_axil_rvalid       (s_axil_rvalid),
        .s_axil_rready       (s_axil_rready),
        .s_axis_tdata        (s_axis_tdata),
        .s_axis_tvalid       (s_axis_tvalid),
        .s_axis_tready       (s_axis_tready),
        .s_axis_tlast        (s_axis_tlast),
        .core_start          (w_start),
        .core_tile_len       (w_tile_len),
        .core_weights_packed (w_weights_packed),
        .core_ifms_packed    (w_ifms_packed),
        .core_result         (w_result),
        .core_done           (w_done),
        .core_ready          (w_ready)
    );

    pe_array #(
        .N_TILES  (1024),
        .KERNEL_H (3),
        .KERNEL_W (3),
        .IN_CH    (1)
    ) u_pe_array (
        .clk             (clk),
        .rst_n           (rst_n),
        .start           (w_start),
        .weights_packed  (w_weights_packed),
        .ifms_all_packed (w_ifms_packed),
        .results_all     (w_result),
        .done_all        (pe_done_all),
        .ready_all       (pe_ready_all)
    );

endmodule
