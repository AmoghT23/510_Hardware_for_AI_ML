`timescale 1ns/1ps

// ============================================================================
// top_ti.sv  —  Integrated Top Module  (Training + Inference, Phase 3)
// ECE 410/510
//
// Phase 3: AXIL address widened 4→6 bits to reach 0x34 (GRAD_W8).
// Instantiates conv_interface_ti, compute_core_ti.
// grad_core is instantiated inside conv_interface_ti.
// ============================================================================

module top_ti (
    input  logic        clk,
    input  logic        rst_n,

    input  logic [5:0]  s_axil_awaddr,
    input  logic        s_axil_awvalid,
    output logic        s_axil_awready,

    input  logic [31:0] s_axil_wdata,
    input  logic [3:0]  s_axil_wstrb,
    input  logic        s_axil_wvalid,
    output logic        s_axil_wready,

    output logic [1:0]  s_axil_bresp,
    output logic        s_axil_bvalid,
    input  logic        s_axil_bready,

    input  logic [5:0]  s_axil_araddr,
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

    logic        w_start;
    logic [7:0]  w_tile_len;
    logic [15:0] w_weight_data;
    logic        w_weight_valid;
    logic [15:0] w_ifm_data;
    logic        w_ifm_valid;
    logic [31:0] w_result;
    logic        w_done;
    logic        w_ready;

    conv_interface_ti #(
        .AXIS_DATA_W (512),
        .AXIL_ADDR_W (6),
        .AXIL_DATA_W (32)
    ) u_interface (
        .clk              (clk),
        .rst_n            (rst_n),
        .s_axil_awaddr    (s_axil_awaddr),
        .s_axil_awvalid   (s_axil_awvalid),
        .s_axil_awready   (s_axil_awready),
        .s_axil_wdata     (s_axil_wdata),
        .s_axil_wstrb     (s_axil_wstrb),
        .s_axil_wvalid    (s_axil_wvalid),
        .s_axil_wready    (s_axil_wready),
        .s_axil_bresp     (s_axil_bresp),
        .s_axil_bvalid    (s_axil_bvalid),
        .s_axil_bready    (s_axil_bready),
        .s_axil_araddr    (s_axil_araddr),
        .s_axil_arvalid   (s_axil_arvalid),
        .s_axil_arready   (s_axil_arready),
        .s_axil_rdata     (s_axil_rdata),
        .s_axil_rresp     (s_axil_rresp),
        .s_axil_rvalid    (s_axil_rvalid),
        .s_axil_rready    (s_axil_rready),
        .s_axis_tdata     (s_axis_tdata),
        .s_axis_tvalid    (s_axis_tvalid),
        .s_axis_tready    (s_axis_tready),
        .s_axis_tlast     (s_axis_tlast),
        .core_start       (w_start),
        .core_tile_len    (w_tile_len),
        .core_weight_data (w_weight_data),
        .core_weight_valid(w_weight_valid),
        .core_ifm_data    (w_ifm_data),
        .core_ifm_valid   (w_ifm_valid),
        .core_result      (w_result),
        .core_done        (w_done),
        .core_ready       (w_ready)
    );

    compute_core_ti #(
        .KERNEL_H (3),
        .KERNEL_W (3),
        .IN_CH    (1)
    ) u_core (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (w_start),
        .weight_data  (w_weight_data),
        .weight_valid (w_weight_valid),
        .ifm_data     (w_ifm_data),
        .ifm_valid    (w_ifm_valid),
        .result       (w_result),
        .done         (w_done),
        .ready        (w_ready)
    );

endmodule
