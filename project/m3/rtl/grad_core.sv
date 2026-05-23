`timescale 1ns/1ps

// ============================================================================
// grad_core.sv  --  BF16 Gradient Computation Unit  (Synthesizable)
// ECE 410/510 -- Phase 3
//
// Computes two sets of FP32 gradients for one 3x3x1 Conv2d tile:
//   dL/dW[i] = dL/dy x x[i]   (weight gradient)
//   dL/dX[i] = dL/dy x w[i]   (activation gradient)
//
// Arithmetic: fp32_mul_bf16 -- FP32 x BF16 -> FP32 (synthesizable).
// No real, $bitstoreal, $realtobits.  Safe for Yosys / OpenLane.
//
// fp32_mul_bf16:
//   ma24 = {1, fp32[22:0]}   24-bit mantissa with implicit 1
//   mb8  = {1, bf16[6:0]}    8-bit mantissa with implicit 1
//   prod = ma24 x mb8        fits in 32 bits (max (2^24-1)x255 < 2^32)
//   prod[31]=1 -> 10.xxx     mant = prod[30:8], exp = a.exp + b.exp-127 + 1
//   prod[31]=0 -> 01.xxx     mant = prod[29:7], exp = a.exp + b.exp-127
//
// NOTE: all functions use "function_name = value" (Verilog-2001 style) --
//       Yosys does not support the "return" keyword inside function bodies.
//
// FSM: IDLE -> COMPUTE (1 cycle, latch results) -> DONE_ST (1 cycle) -> IDLE
// ============================================================================

module grad_core (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    input  logic [31:0]  grad_out,       // dL/dy  (FP32)
    input  logic [143:0] weights_flat,   // 9 x BF16: w[i] = bits [16i+15:16i]
    input  logic [143:0] ifms_flat,      // 9 x BF16: x[i] = bits [16i+15:16i]
    output logic [287:0] grad_w_flat,    // 9 x FP32: gw[i] = bits [32i+31:32i]
    output logic [287:0] grad_x_flat,    // 9 x FP32: gx[i] = bits [32i+31:32i]
    output logic         done,
    output logic         ready
);

    typedef enum logic [1:0] {
        IDLE    = 2'd0,
        COMPUTE = 2'd1,
        DONE_ST = 2'd2
    } state_t;

    state_t state;

    // -------------------------------------------------------------------------
    // fp32_mul_bf16 : FP32 x BF16 -> FP32  (synthesizable)
    //
    // a : FP32  [31]=sign  [30:23]=exp(bias 127)  [22:0]=23-bit mantissa
    // b : BF16  [15]=sign  [14:7]=exp(bias 127)   [6:0]=7-bit mantissa
    //
    // Both have an implicit leading 1, so the product implicit 1 lands at
    // bit 30 (01.xxx) or bit 31 (10.xxx) -- one-bit normalisation only.
    // -------------------------------------------------------------------------
    function automatic logic [31:0] fp32_mul_bf16(
        input logic [31:0] a,
        input logic [15:0] b
    );
        logic        sign_r;
        logic [9:0]  exp_sum;
        logic [7:0]  exp_r;
        logic [23:0] ma24;
        logic [7:0]  mb8;
        logic [31:0] prod;
        logic [22:0] mant_r;

        if (a[30:23] == 8'h00 || b[14:7] == 8'h00) begin
            fp32_mul_bf16 = 32'h0;                          // zero input
        end else if (a[30:23] == 8'hFF || b[14:7] == 8'hFF) begin
            fp32_mul_bf16 = {a[31] ^ b[15], 8'hFF, 23'h0}; // inf / NaN
        end else begin
            sign_r  = a[31] ^ b[15];
            exp_sum = {2'b00, a[30:23]} + {2'b00, b[14:7]} - 10'd127;
            ma24    = {1'b1, a[22:0]};
            mb8     = {1'b1, b[6:0]};
            // 24x8 product; both zero-extended to 32 bits.
            // max(ma24) x max(mb8) = (2^24-1) x 255 < 2^32, no overflow.
            prod    = {8'h00, ma24} * {24'h00, mb8};

            if (exp_sum[9] || exp_sum == 10'h0) begin
                fp32_mul_bf16 = {sign_r, 31'h0};            // underflow
            end else if (exp_sum >= 10'd255) begin
                fp32_mul_bf16 = {sign_r, 8'hFF, 23'h0};     // overflow
            end else if (prod[31]) begin
                // 10.xxx form: result = 1.xxx x 2^(exp_sum+1)
                mant_r = prod[30:8];
                if (exp_sum >= 10'd254)
                    fp32_mul_bf16 = {sign_r, 8'hFF, 23'h0};
                else begin
                    exp_r         = exp_sum[7:0] + 8'd1;
                    fp32_mul_bf16 = {sign_r, exp_r, mant_r};
                end
            end else begin
                // 01.xxx form: result = 1.xxx x 2^exp_sum
                mant_r = prod[29:7];
                if (&exp_sum[7:0])
                    fp32_mul_bf16 = {sign_r, 8'hFF, 23'h0};
                else begin
                    exp_r         = exp_sum[7:0];
                    fp32_mul_bf16 = {sign_r, exp_r, mant_r};
                end
            end
        end
    endfunction

    // -------------------------------------------------------------------------
    // Combinational gradient computation
    // -------------------------------------------------------------------------
    logic [287:0] comb_grad_w;
    logic [287:0] comb_grad_x;

    always @(*) begin
        for (int i = 0; i < 9; i++) begin
            comb_grad_w[i*32 +: 32] =
                fp32_mul_bf16(grad_out, ifms_flat[i*16 +: 16]);
            comb_grad_x[i*32 +: 32] =
                fp32_mul_bf16(grad_out, weights_flat[i*16 +: 16]);
        end
    end

    // -------------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state       <= IDLE;
            done        <= 1'b0;
            grad_w_flat <= '0;
            grad_x_flat <= '0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE:    if (start) state <= COMPUTE;
                COMPUTE: begin
                    grad_w_flat <= comb_grad_w;
                    grad_x_flat <= comb_grad_x;
                    state       <= DONE_ST;
                end
                DONE_ST: begin
                    done  <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

    assign ready = (state == IDLE);

endmodule
