`timescale 1ns/1ps

// ============================================================================
// pe_array.sv  --  32x32 weight-stationary systolic array
// ECE 410/510  |  32x32 systolic array
//
// N_TILES compute_core_ti tiles, all sharing weights_packed (weight-stationary).
// Each tile receives its own 144-bit IFM window; all started by one start pulse.
// done_all / ready_all are AND-reduces across all tiles.
// ============================================================================

module pe_array #(
    parameter N_TILES  = 1024,
    parameter KERNEL_H = 3,
    parameter KERNEL_W = 3,
    parameter IN_CH    = 1
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,

    input  wire [143:0] weights_packed,

    input  wire [N_TILES*144-1:0] ifms_all_packed,
    output wire [N_TILES*32-1:0]  results_all,

    output wire         done_all,
    output wire         ready_all
);

    wire [N_TILES-1:0] tile_done;
    wire [N_TILES-1:0] tile_ready;

    genvar t;
    generate
        for (t = 0; t < N_TILES; t = t + 1) begin : gen_tile
            compute_core_ti u_tile (
                .clk            (clk),
                .rst_n          (rst_n),
                .start          (start),
                .weights_packed (weights_packed),
                .ifms_packed    (ifms_all_packed[t*144 +: 144]),
                .result         (results_all[t*32 +: 32]),
                .done           (tile_done[t]),
                .ready          (tile_ready[t])
            );
        end
    endgenerate

    assign done_all  = &tile_done;
    assign ready_all = &tile_ready;

endmodule
