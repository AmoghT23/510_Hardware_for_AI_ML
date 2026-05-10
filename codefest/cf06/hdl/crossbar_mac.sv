// 4x4 Binary-Weight Crossbar MAC Unit
// Computes out[j] = sum_i( weight[i][j] * in[i] ) each clock cycle
// Weights stored as 1-bit values: 1 = +1, 0 = -1
//
// Timing: assert load_weights for one cycle to latch weight_in,
//         out_data is valid one cycle after inputs are applied.
//
// Implementation note: uses generate-based carry-chain MAC to avoid
// runtime wire-array indexing, which iverilog 11 does not support.

`timescale 1ns/1ps

module crossbar_mac #(
    parameter integer N         = 4,   // crossbar dimension
    parameter integer IN_WIDTH  = 8,   // input bit width (signed)
    parameter integer OUT_WIDTH = 10   // output/accumulator width
)(
    input  wire                           clk,
    input  wire                           rst_n,

    input  wire                           load_weights,
    input  wire [N-1:0][N-1:0]           weight_in,   // weight_in[i][j]: 1=>+1, 0=>-1

    // Packed 2D: in_data[i] = IN_WIDTH-bit signed input i
    input  wire [N-1:0][IN_WIDTH-1:0]    in_data,
    // Packed 2D: out_data[j] = OUT_WIDTH-bit signed accumulation j
    output reg  [N-1:0][OUT_WIDTH-1:0]   out_data
);

    // -----------------------------------------------------------------------
    // Weight bank: packed 2D reg, async reset, synchronous load
    // -----------------------------------------------------------------------
    reg [N-1:0][N-1:0] weight_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            weight_reg <= {(N*N){1'b0}};
        else if (load_weights)
            weight_reg <= weight_in;
    end

    // -----------------------------------------------------------------------
    // Sign-extend each IN_WIDTH input to OUT_WIDTH bits.
    // Packed 2D wire driven by generate; genvar k is a compile-time constant
    // so no runtime-index-into-wire issue arises.
    // -----------------------------------------------------------------------
    wire [N-1:0][OUT_WIDTH-1:0] in_ext;

    genvar k;
    generate
        for (k = 0; k < N; k = k + 1) begin : gen_ext
            assign in_ext[k] = {{(OUT_WIDTH-IN_WIDTH){in_data[k][IN_WIDTH-1]}},
                                  in_data[k]};
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Carry-chain MAC using generate loops.
    //
    // psum[j][0]    = 0
    // psum[j][i+1]  = psum[j][i] + (weight[i][j] ? +in_ext[i] : -in_ext[i])
    // mac_comb[j]   = psum[j][N]
    //
    // All array accesses use genvar (compile-time) indices only.
    // -----------------------------------------------------------------------
    wire signed [OUT_WIDTH-1:0] psum [0:N-1][0:N];

    genvar gi, gj;
    generate
        for (gj = 0; gj < N; gj = gj + 1) begin : gen_col
            assign psum[gj][0] = {OUT_WIDTH{1'b0}};
            for (gi = 0; gi < N; gi = gi + 1) begin : gen_row
                assign psum[gj][gi+1] = psum[gj][gi] +
                    (weight_reg[gi][gj]
                        ?  $signed(in_ext[gi])
                        : -$signed(in_ext[gi]));
            end
        end
    endgenerate

    // Collect final partial sums into a single packed 2D wire
    wire [N-1:0][OUT_WIDTH-1:0] mac_comb;

    genvar gfin;
    generate
        for (gfin = 0; gfin < N; gfin = gfin + 1) begin : gen_fin
            assign mac_comb[gfin] = psum[gfin][N];
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Pipeline register: single bulk assignment, no loop or variable index
    // -----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            out_data <= {(N*OUT_WIDTH){1'b0}};
        else
            out_data <= mac_comb;
    end

endmodule