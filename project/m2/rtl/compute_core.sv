// ============================================================================
// compute_core.sv
// ECE 410/510 — Conv2d Co-processor Compute Core
// Project: Anemia Detection — ResNet18 Conv2d Accelerator Chiplet
//
// Computes one output feature-map pixel as the dot product of
//   KERNEL_H x KERNEL_W x IN_CH weight values and IFM values.
// Arithmetic: Q16.16 signed fixed-point (32-bit ports, 64-bit accumulator).
// Output-stationary dataflow: accumulate all MACs, then write result.
//
// Port list:
//   clk          in   1    System clock (target 1 GHz)
//   rst_n        in   1    Synchronous active-low reset
//   start        in   1    Pulse high one cycle to begin a new computation
//   weight_data  in  32    Q16.16 weight value; sampled when weight_valid=1
//   weight_valid in   1    weight_data is valid this cycle
//   ifm_data     in  32    Q16.16 IFM value; sampled when ifm_valid=1
//   ifm_valid    in   1    ifm_data is valid this cycle
//   result       out 32    Q16.16 output pixel; stable while done=1 and after
//   done         out  1    Pulses high for exactly one cycle when result is ready
//   ready        out  1    High when core is idle and will accept start
//
// Clock domain : single (clk)
// Reset        : synchronous, active-low — rst_n=0 clears all registers and
//                returns FSM to IDLE on the next rising clock edge.
//
// Data format  : Q16.16 signed fixed-point.
//   Bit 31 = sign, bits 30:16 = integer part, bits 15:0 = fractional part.
//   Value = register_contents / 2^16.
//   Accumulator is 64-bit (Q32.32) to prevent overflow across NUM_ELEM MACs.
//   Result extracted as accum[47:16], converting Q32.32 back to Q16.16.
// ============================================================================

module compute_core #(
    parameter int KERNEL_H = 3,
    parameter int KERNEL_W = 3,
    parameter int IN_CH    = 1
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [31:0] weight_data,
    input  logic        weight_valid,
    input  logic [31:0] ifm_data,
    input  logic        ifm_valid,
    output logic [31:0] result,
    output logic        done,
    output logic        ready
);

    // ── derived parameters ───────────────────────────────────────────────────
    localparam int NUM_ELEM = KERNEL_H * KERNEL_W * IN_CH;  // 9 for default
    localparam int CNT_W    = $clog2(NUM_ELEM + 1);          // counter width

    // ── FSM states ───────────────────────────────────────────────────────────
    typedef enum logic [1:0] {
        IDLE    = 2'd0,
        LOAD    = 2'd1,
        COMPUTE = 2'd2,
        DONE_ST = 2'd3
    } state_t;

    state_t state;

    // ── internal buffers and registers ───────────────────────────────────────
    logic [31:0]        weight_buf [0:NUM_ELEM-1];
    logic [31:0]        ifm_buf    [0:NUM_ELEM-1];
    logic [CNT_W-1:0]   cnt;
    logic signed [63:0] accum;
    logic [31:0]        result_reg;

    // ── combinational MAC: sign-extend to 64-bit then multiply ───────────────
    // weight_buf[cnt] and ifm_buf[cnt] are Q16.16; product is Q32.32 (64-bit).
    logic signed [63:0] w_ext, i_ext, product;

    always_comb begin
        w_ext   = {{32{weight_buf[cnt][31]}}, weight_buf[cnt]};
        i_ext   = {{32{ifm_buf[cnt][31]}},    ifm_buf[cnt]};
        product = w_ext * i_ext;
    end

    // ── registered FSM + datapath ─────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state      <= IDLE;
            cnt        <= '0;
            accum      <= '0;
            result_reg <= '0;
            done       <= 1'b0;
        end else begin
            done <= 1'b0;  // de-assert by default every cycle

            case (state)

                // ── IDLE: wait for start pulse ────────────────────────────
                IDLE: begin
                    if (start) begin
                        state <= LOAD;
                        cnt   <= '0;
                    end
                end

                // ── LOAD: buffer weight and IFM pairs, one per cycle ──────
                // Both weight_valid and ifm_valid must be high together.
                LOAD: begin
                    if (weight_valid && ifm_valid) begin
                        weight_buf[cnt] <= weight_data;
                        ifm_buf[cnt]    <= ifm_data;
                        if (cnt == (NUM_ELEM - 1)) begin
                            state <= COMPUTE;
                            cnt   <= '0;
                            accum <= '0;
                        end else begin
                            cnt <= cnt + 1'b1;
                        end
                    end
                end

                // ── COMPUTE: sequential MAC over buffered elements ────────
                // One multiply-accumulate per cycle; output-stationary.
                COMPUTE: begin
                    accum <= accum + product;
                    if (cnt == (NUM_ELEM - 1)) begin
                        state <= DONE_ST;
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end

                // ── DONE_ST: latch result, pulse done, return to IDLE ─────
                DONE_ST: begin
                    result_reg <= accum[47:16];  // Q32.32 → Q16.16
                    done       <= 1'b1;
                    state      <= IDLE;
                    cnt        <= '0;
                end

            endcase
        end
    end

    // ── output assignments ────────────────────────────────────────────────────
    assign result = result_reg;
    assign ready  = (state == IDLE);

endmodule
