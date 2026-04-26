// project/hdl/conv2d_accel.sv
// Top-level Conv2D accelerator stub for CodeFest assignment.
//
// This file provides a synthesis-friendly top-level interface for a
// Conv2D accelerator. It is intentionally a placeholder stub and not
// a complete implementation. The design includes clock/reset,
// start/ready handshaking, input data/kernel ports, and a result output.

module conv2d_accel #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32,
    parameter KERNEL_SIZE = 3
)(
    input  logic clk,
    input  logic rst,
    input  logic start,
    input  logic [DATA_WIDTH-1:0] data_in,
    input  logic [DATA_WIDTH-1:0] kernel_in,
    input  logic data_valid,
    input  logic kernel_valid,
    output logic [ACC_WIDTH-1:0] result,
    output logic result_valid,
    output logic ready
);

    // Placeholder internal state for stub behavior.
    logic busy;

    always_ff @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            result <= '0;
            result_valid <= 1'b0;
            ready <= 1'b1;
        end else begin
            result_valid <= 1'b0;
            if (start && ready) begin
                busy <= 1'b1;
                ready <= 1'b0;
                // Stub behavior: one-cycle multiply of input and kernel.
                result <= $signed(data_in) * $signed(kernel_in);
                result_valid <= 1'b1;
            end else if (busy) begin
                busy <= 1'b0;
                ready <= 1'b1;
            end
        end
    end

endmodule
