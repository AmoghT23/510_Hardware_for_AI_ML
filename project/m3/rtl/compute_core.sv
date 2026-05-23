`timescale 1ns/1ps

// ============================================================================
// compute_core.sv
// ECE 410/510 — Conv2d Co-processor Compute Core  (INT8 QAT revision)
// Project: Anemia Detection — ResNet18 Conv2d Accelerator Chiplet
//
// Computes one output feature-map pixel as the dot product of
//   KERNEL_H x KERNEL_W x IN_CH weight values and IFM values.
//
// Arithmetic: INT8 x INT8 → INT32 (9 unrolled parallel MACs, one tile/cycle)
//   Inputs  : INT8 signed (8-bit two's complement)
//   Compute : 9 simultaneous signed multiplies, adder tree → INT32 sum
//   Output  : INT32 signed (32-bit accumulator, no requantization)
//
// Motivation for INT8 pivot (M3, from CF07 synthesis on SKY130A):
//   M2 FP16 critical path was fully combinational:
//     fp16_to_fp32 → 24×24-bit multiply → clz24 → barrel-shift → fp32_to_fp16
//   Measured arrival: 21.319 ns vs. required 1.667 ns (-19.653 ns slack).
//   SKY130A cannot close this path at 500 MHz.
//   INT8 9-MAC adder tree estimated 4.0–6.5 ns; targets 800 MHz–1 GHz.
//
// Precision impact:
//   INT8 QAT drops accuracy ~0.1–0.2% for binary classification.
//   SQNR ≈ 48 dB — above the 40–50 dB floor for image classification CNNs.
//   Max product per MAC: 127×127 = 16129 (fits INT16).
//   Max tile sum (9 MACs): 9×16129 = 145161 (fits INT32, no overflow).
//
// FSM: IDLE → LOAD → COMPUTE → DONE_ST → IDLE
//   LOAD    : NUM_ELEM cycles  (one INT8 weight + IFM pair per cycle)
//   COMPUTE : 1 cycle          (all 9 MACs unrolled, adder tree)
//   DONE_ST : 1 cycle          (latch result, pulse done)
//
// Port changes from FP16 revision:
//   weight_data : 16-bit FP16  →  8-bit INT8
//   ifm_data    : 16-bit FP16  →  8-bit INT8
//   result      : 16-bit FP16  →  32-bit INT32
//   accum register removed (single-cycle COMPUTE needs no running sum)
//
// WIDTHEXPAND warnings from FP16 revision (lines 82, 255) are resolved by
// removing the FP function block entirely.
//
// Clock domain : single (clk)
// Reset        : synchronous, active-low
// ============================================================================

module compute_core #(
    parameter int KERNEL_H = 3,
    parameter int KERNEL_W = 3,
    parameter int IN_CH    = 1
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [7:0]  weight_data,    // INT8 signed weight
    input  logic        weight_valid,
    input  logic [7:0]  ifm_data,       // INT8 signed activation
    input  logic        ifm_valid,
    output logic [31:0] result,         // INT32 output pixel
    output logic        done,
    output logic        ready
);

    // ── derived parameters ───────────────────────────────────────────────────
    localparam int NUM_ELEM = KERNEL_H * KERNEL_W * IN_CH;  // 9 for 3×3×1
    localparam int CNT_W    = $clog2(NUM_ELEM + 1);

    // ── FSM ──────────────────────────────────────────────────────────────────
    typedef enum logic [1:0] {
        IDLE    = 2'd0,
        LOAD    = 2'd1,
        COMPUTE = 2'd2,
        DONE_ST = 2'd3
    } state_t;

    state_t state;

    // ── data buffers and registers ───────────────────────────────────────────
    logic [7:0]        weight_buf [0:NUM_ELEM-1];  // INT8 weight storage
    logic [7:0]        ifm_buf    [0:NUM_ELEM-1];  // INT8 IFM storage
    logic [CNT_W-1:0]  cnt;
    logic [31:0]       result_reg;  // INT32 output latch

    // ── 9 unrolled MACs ──────────────────────────────────────────────────────
    // Both operands sign-extended to 32 bits before multiply so the product
    // is a full signed 32-bit result (no truncation from 8-bit default width).
    // Max single product : 127×127 = 16 129  — fits INT32 with large margin.
    // Max tile sum (9 MACs): 145 161          — fits INT32, no overflow.
    logic signed [31:0] mac_sum;

    always_comb begin
        mac_sum = 32'sd0;
        for (int i = 0; i < NUM_ELEM; i++)
            mac_sum = mac_sum +
                      $signed({{24{weight_buf[i][7]}}, weight_buf[i]}) *
                      $signed({{24{ifm_buf[i][7]}},   ifm_buf[i]});
    end

    // =========================================================================
    // Registered FSM + datapath
    // =========================================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state      <= IDLE;
            cnt        <= '0;
            result_reg <= 32'h0;
            done       <= 1'b0;
        end else begin
            done <= 1'b0;   // de-assert by default

            case (state)

                // ── IDLE: wait for start pulse ────────────────────────────
                IDLE: begin
                    if (start) begin
                        state <= LOAD;
                        cnt   <= '0;
                    end
                end

                // ── LOAD: buffer INT8 weight and IFM pairs ────────────────
                LOAD: begin
                    if (weight_valid && ifm_valid) begin
                        weight_buf[cnt] <= weight_data;
                        ifm_buf[cnt]    <= ifm_data;
                        if (cnt == CNT_W'(NUM_ELEM - 1)) begin
                            state <= COMPUTE;
                            cnt   <= '0;
                        end else begin
                            cnt <= cnt + 1'b1;
                        end
                    end
                end

                // ── COMPUTE: 9 unrolled INT8×INT8→INT32 MACs, one cycle ───
                // mac_sum is combinational over the fully loaded buffers;
                // registering it here closes the adder-tree path.
                COMPUTE: begin
                    result_reg <= mac_sum;
                    state      <= DONE_ST;
                end

                // ── DONE_ST: pulse done, return to IDLE ───────────────────
                DONE_ST: begin
                    done  <= 1'b1;
                    state <= IDLE;
                    cnt   <= '0;
                end

            endcase
        end
    end

    // ── output assignments ────────────────────────────────────────────────────
    assign result = result_reg;
    assign ready  = (state == IDLE);

`ifdef COCOTB_SIM
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, compute_core);
    end
`endif

endmodule
