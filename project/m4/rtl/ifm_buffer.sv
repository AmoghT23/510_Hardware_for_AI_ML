`timescale 1ns/1ps

// ============================================================================
// ifm_buffer.sv  --  IFM activation tile buffer + window extractor
// ECE 410/510  |  32x32 systolic array
//
// Stores a 34x34 BF16 input activation patch loaded over 37 AXI4-Stream beats.
// Combinationally extracts 1024 overlapping 3x3 windows (one per output pixel).
//
// Loading:
//   Flat order: row 0 left-to-right, row 1, ..., row 33  (1156 BF16 values).
//   512-bit beat carries 32 BF16 values at bits [i*16+15 : i*16], i=0..31.
//   Beat 0..35 carry 32 values each; beat 36 carries the remaining 4 values.
//   load_done pulses high for one cycle after beat 36 is accepted.
//
// Window extraction (combinational):
//   Tile t = row*32 + col  (row in [0..31], col in [0..31])
//   windows_packed[t*144 +: 144] = 9 BF16 values from buf[row:row+2][col:col+2]
//   Element order within a window: (i,j) at bit [(i*3+j)*16 +: 16], i=row-offset,
//   j=col-offset, both in [0..2].  Matches compute_core_ti ifms_packed packing.
//
// Parameters:
//   N_OUT   output pixels per side (32 for 32x32 array)
//   N_IN    input pixels per side  (N_OUT + 2 = 34)
// ============================================================================

module ifm_buffer #(
    parameter int N_OUT = 32,
    parameter int N_IN  = N_OUT + 2   // 34
) (
    input  logic         clk,
    input  logic         rst_n,

    // AXI4-Stream slave (tready always 1)
    input  logic [511:0] s_axis_tdata,
    input  logic         s_axis_tvalid,
    output logic         s_axis_tready,

    // Status
    output logic         load_done,   // pulses 1 cycle after beat 36 accepted

    // 1024 x 144-bit window outputs (combinational)
    output logic [N_OUT*N_OUT*144-1:0] windows_packed
);

    localparam int N_TILES = N_OUT * N_OUT;              // 1024
    localparam int N_VALS  = N_IN  * N_IN;              // 1156
    localparam int N_BEATS = (N_VALS + 31) / 32;        // 37

    // ── flat BF16 buffer  [0..N_VALS-1] ──────────────────────────────────────
    logic [15:0] buf_flat [0:N_VALS-1];

    // ── beat counter ─────────────────────────────────────────────────────────
    logic [5:0] beat_cnt;   // counts 0..36 (6 bits cover up to 63)

    assign s_axis_tready = 1'b1;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            beat_cnt  <= 6'd0;
            load_done <= 1'b0;
            // buf_flat is not reset: contents are X until the first 37-beat
            // load completes (load_done), and core_start is gated on
            // ifm_load_done, so undefined initial state never reaches outputs.
        end else begin
            // load_done stays high after tile completes; cleared when next tile starts
            if (s_axis_tvalid && beat_cnt == 0 && load_done)
                load_done <= 1'b0;

            if (s_axis_tvalid && beat_cnt < N_BEATS) begin
                for (int i = 0; i < 32; i++) begin
                    if ((beat_cnt * 32 + i) < N_VALS)
                        buf_flat[beat_cnt * 32 + i] <= s_axis_tdata[i*16 +: 16];
                end

                if (beat_cnt == N_BEATS - 1) begin
                    load_done <= 1'b1;
                    beat_cnt  <= 6'd0;
                end else begin
                    beat_cnt <= beat_cnt + 1'b1;
                end
            end
        end
    end

    // ── combinational window extractor ────────────────────────────────────────
    // For tile t: row = t/N_OUT, col = t%N_OUT
    // window element (i,j) = buf_flat[(row+i)*N_IN + (col+j)], i,j in [0..2]
    // packed at bit [(i*3+j)*16 +: 16] within the tile's 144-bit slice
    always_comb begin
        for (int t = 0; t < N_TILES; t++) begin
            for (int i = 0; i < 3; i++) begin
                for (int j = 0; j < 3; j++) begin
                    windows_packed[t*144 + (i*3+j)*16 +: 16] =
                        buf_flat[((t/N_OUT)+i)*N_IN + ((t%N_OUT)+j)];
                end
            end
        end
    end

endmodule
