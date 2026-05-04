// ============================================================================
// compute_core.sv
// ECE 410/510 — Conv2d Co-processor Compute Core  (FP16 revision)
// Project: Anemia Detection — ResNet18 Conv2d Accelerator Chiplet
//
// Computes one output feature-map pixel as the dot product of
//   KERNEL_H x KERNEL_W x IN_CH weight values and IFM values.
//
// Arithmetic: mixed-precision FP16 × FP16 → FP32 accumulate → FP16 output
//   (industry standard: NVIDIA Tensor Core / Apple ANE approach)
//   Inputs  : FP16 (IEEE 754 half-precision, 16-bit)
//   Accum   : FP32 (IEEE 754 single-precision, 32-bit) — prevents catastrophic
//             cancellation in deep tiles (4,608-MAC Layer4 of ResNet18)
//   Output  : FP16 (rounded from FP32 accumulator, truncation mode)
//
// FP16 format  : S[15] E[14:10] M[9:0],  bias = 15
// FP32 format  : S[31] E[30:23] M[22:0], bias = 127
// Exponent rebase (FP16→FP32): fp32_exp = fp16_exp + 112  (127 − 15 = 112)
//
// Limitations (acceptable for inference with well-trained ResNet18 weights):
//   - Denormals flushed to zero on input
//   - Result rounds by truncation (no round-to-nearest-even)
//   - NaN/Inf propagation is best-effort, not IEEE-compliant
//
// Port list  : same as Q16.16 revision except weight_data/ifm_data/result
//              widths reduced from 32 to 16 bits.
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
    input  logic [15:0] weight_data,    // FP16 weight
    input  logic        weight_valid,
    input  logic [15:0] ifm_data,       // FP16 activation
    input  logic        ifm_valid,
    output logic [15:0] result,         // FP16 output pixel
    output logic        done,
    output logic        ready
);

    // ── derived parameters ───────────────────────────────────────────────────
    localparam int NUM_ELEM = KERNEL_H * KERNEL_W * IN_CH;
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
    logic [15:0]        weight_buf [0:NUM_ELEM-1];  // FP16 weight storage
    logic [15:0]        ifm_buf    [0:NUM_ELEM-1];  // FP16 IFM storage
    logic [CNT_W-1:0]   cnt;
    logic [31:0]        accum;       // FP32 running sum
    logic [15:0]        result_reg;  // FP16 output latch

    // =========================================================================
    // Floating-point helper functions
    // All functions are automatic (stack-allocated, safe for simulation).
    // Inputs and outputs match the FP16 / FP32 IEEE 754 bit layouts.
    // =========================================================================

    // ── FP16 → FP32 zero/sign/exponent expansion ─────────────────────────────
    // Denormals (exp16==0, man16!=0) are flushed to signed zero.
    function automatic [31:0] fp16_to_fp32(input [15:0] f16);
        logic [7:0] e32;
        if (f16[14:10] == 5'h1F) begin
            // Inf or NaN — preserve sign and top mantissa bit
            fp16_to_fp32 = {f16[15], 8'hFF, f16[9], 13'h0};
        end else if (f16[14:10] == 5'h00) begin
            // Zero or denormal → flush to signed zero
            fp16_to_fp32 = {f16[15], 31'h0};
        end else begin
            // Normal: rebase exponent bias 15 → 127  (add 112)
            e32 = {3'b000, f16[14:10]} + 8'd112;
            fp16_to_fp32 = {f16[15], e32, f16[9:0], 13'h0};
        end
    endfunction

    // ── FP32 → FP16 contraction (truncation, no round-to-nearest) ────────────
    function automatic [15:0] fp32_to_fp16(input [31:0] f32);
        logic [7:0] e32;
        logic [7:0] e16w;
        e32 = f32[30:23];
        if (e32 == 8'hFF) begin
            // Inf or NaN
            fp32_to_fp16 = {f32[31], 5'h1F, f32[22:13]};
        end else if (e32 <= 8'd112) begin
            // Underflow — value too small for FP16 normal → flush to zero
            fp32_to_fp16 = {f32[31], 15'h0};
        end else if (e32 >= 8'd143) begin
            // Overflow — saturate to FP16 Inf
            fp32_to_fp16 = {f32[31], 5'h1F, 10'h0};
        end else begin
            // Normal: rebase exponent bias 127 → 15  (subtract 112)
            e16w = e32 - 8'd112;
            fp32_to_fp16 = {f32[31], e16w[4:0], f32[22:13]};
        end
    endfunction

    // ── Count leading zeros in a 24-bit value (priority encoder) ─────────────
    // Returns 24 if input is zero. Used for post-subtraction normalisation.
    function automatic [4:0] clz24(input [23:0] x);
        if      (x[23]) clz24 = 5'd0;
        else if (x[22]) clz24 = 5'd1;
        else if (x[21]) clz24 = 5'd2;
        else if (x[20]) clz24 = 5'd3;
        else if (x[19]) clz24 = 5'd4;
        else if (x[18]) clz24 = 5'd5;
        else if (x[17]) clz24 = 5'd6;
        else if (x[16]) clz24 = 5'd7;
        else if (x[15]) clz24 = 5'd8;
        else if (x[14]) clz24 = 5'd9;
        else if (x[13]) clz24 = 5'd10;
        else if (x[12]) clz24 = 5'd11;
        else if (x[11]) clz24 = 5'd12;
        else if (x[10]) clz24 = 5'd13;
        else if (x[9])  clz24 = 5'd14;
        else if (x[8])  clz24 = 5'd15;
        else if (x[7])  clz24 = 5'd16;
        else if (x[6])  clz24 = 5'd17;
        else if (x[5])  clz24 = 5'd18;
        else if (x[4])  clz24 = 5'd19;
        else if (x[3])  clz24 = 5'd20;
        else if (x[2])  clz24 = 5'd21;
        else if (x[1])  clz24 = 5'd22;
        else if (x[0])  clz24 = 5'd23;
        else            clz24 = 5'd24;
    endfunction

    // ── FP32 multiply ─────────────────────────────────────────────────────────
    // Handles: normal × normal, zero × anything = zero.
    // Ignores: Inf/NaN propagation (acceptable for inference).
    function automatic [31:0] fp32_mul(input [31:0] a, input [31:0] b);
        logic        sr;
        logic [7:0]  ea, eb;
        logic [23:0] ma, mb;
        logic [47:0] mp;    // 24×24 = 48-bit raw product
        logic [9:0]  es;    // 10-bit exponent sum (avoids 8-bit wrap on addition)
        logic [7:0]  er;
        logic [22:0] mr;

        sr = a[31] ^ b[31];
        ea = a[30:23];
        eb = b[30:23];

        if (ea == 8'h00 || eb == 8'h00) begin
            fp32_mul = 32'h0;
        end else begin
            ma = {1'b1, a[22:0]};   // restore hidden bit
            mb = {1'b1, b[22:0]};
            mp = ma * mb;           // 48-bit product

            // Biased exponent of product = ea + eb − 127.
            // Use 10-bit sum then 8-bit subtract; 8-bit modular arithmetic
            // produces correct result for the FP16-in-FP32 exponent range.
            es = {2'b00, ea} + {2'b00, eb};

            if (mp[47]) begin
                // Leading 1 at bit 47: 1.mp[46:24] × 2^(ea+eb−126)
                mr = mp[46:24];
                er = es[7:0] - 8'd126;
            end else begin
                // Leading 1 at bit 46: 1.mp[45:23] × 2^(ea+eb−127)
                mr = mp[45:23];
                er = es[7:0] - 8'd127;
            end

            fp32_mul = {sr, er, mr};
        end
    endfunction

    // ── FP32 add ──────────────────────────────────────────────────────────────
    // Handles: zero operands, same-sign addition, opposite-sign subtraction
    // with full post-subtraction normalisation via clz24 + barrel shift.
    function automatic [31:0] fp32_add(input [31:0] a, input [31:0] b);
        logic        sa, sb, sl, ss;
        logic [7:0]  ea, eb, el, ess, er;
        logic [23:0] ma, mb, ml, ms_m, ms_shifted;
        logic [7:0]  diff_e;
        logic [24:0] sum25;
        logic [23:0] diff24, diff_shifted;
        logic [4:0]  lz;
        logic [22:0] mr;

        sa = a[31];  sb = b[31];
        ea = a[30:23];  eb = b[30:23];
        ma = {1'b1, a[22:0]};
        mb = {1'b1, b[22:0]};

        // Zero operand fast-paths
        if (ea == 8'h00 && eb == 8'h00) begin
            fp32_add = 32'h0;
        end else if (ea == 8'h00) begin
            fp32_add = b;
        end else if (eb == 8'h00) begin
            fp32_add = a;
        end else begin

            // Sort so that |large| >= |small| (by exponent, then mantissa)
            if (ea > eb || (ea == eb && ma >= mb)) begin
                el = ea;  ml = ma;  sl = sa;
                ess = eb; ms_m = mb; ss = sb;
            end else begin
                el = eb;  ml = mb;  sl = sb;
                ess = ea; ms_m = ma; ss = sa;
            end

            // Alignment shift
            diff_e = el - ess;
            if (diff_e >= 8'd24)
                ms_shifted = 24'h0;
            else
                ms_shifted = ms_m >> diff_e;

            if (sl == ss) begin
                // ── Same sign: add mantissas ──────────────────────────────
                sum25 = {1'b0, ml} + {1'b0, ms_shifted};
                if (sum25[24]) begin
                    // Carry-out: shift right 1, increment exponent
                    mr = sum25[23:1];
                    er = el + 8'd1;
                end else begin
                    mr = sum25[22:0];
                    er = el;
                end
                fp32_add = {sl, er, mr};

            end else begin
                // ── Different signs: subtract; large >= small, so result >= 0 ──
                diff24 = ml - ms_shifted;

                if (diff24 == 24'h0) begin
                    fp32_add = 32'h0;
                end else begin
                    // Normalise: shift left until leading 1 is at bit 23
                    lz = clz24(diff24);
                    diff_shifted = diff24 << lz;  // barrel shift
                    mr = diff_shifted[22:0];       // drop implicit leading 1

                    // Decrement exponent by shift amount; guard underflow
                    if ({3'b000, lz} >= {3'b000, el}) begin
                        fp32_add = 32'h0;          // underflow → zero
                    end else begin
                        er = el - {3'b000, lz};
                        fp32_add = {sl, er, mr};
                    end
                end
            end
        end
    endfunction

    // =========================================================================
    // Registered FSM + datapath
    // =========================================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state      <= IDLE;
            cnt        <= '0;
            accum      <= 32'h0;    // FP32 +0.0
            result_reg <= 16'h0;
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

                // ── LOAD: buffer FP16 weight and IFM pairs ────────────────
                LOAD: begin
                    if (weight_valid && ifm_valid) begin
                        weight_buf[cnt] <= weight_data;
                        ifm_buf[cnt]    <= ifm_data;
                        if (cnt == CNT_W'(NUM_ELEM - 1)) begin
                            state <= COMPUTE;
                            cnt   <= '0;
                            accum <= 32'h0;
                        end else begin
                            cnt <= cnt + 1'b1;
                        end
                    end
                end

                // ── COMPUTE: FP16×FP16→FP32 MAC, one per cycle ───────────
                // fp32_mul converts each FP16 pair to FP32 and multiplies;
                // fp32_add accumulates into the FP32 running sum.
                COMPUTE: begin
                    accum <= fp32_add(
                                 accum,
                                 fp32_mul(fp16_to_fp32(weight_buf[cnt]),
                                          fp16_to_fp32(ifm_buf[cnt]))
                             );
                    if (cnt == CNT_W'(NUM_ELEM - 1)) begin
                        state <= DONE_ST;
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end

                // ── DONE_ST: convert FP32 accumulator → FP16, pulse done ──
                DONE_ST: begin
                    result_reg <= fp32_to_fp16(accum);
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
