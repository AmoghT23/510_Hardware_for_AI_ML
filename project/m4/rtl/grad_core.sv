`timescale 1ns/1ps

// ============================================================================
// grad_core.sv  --  BF16 Gradient Computation Unit  (Synthesizable)
// ECE 410/510 -- Phase 3
//
// Computes two sets of FP32 gradients for one 3×3×1 Conv2d tile:
//   dL/dW[i] = dL/dy × x[i]   (weight gradient)
//   dL/dX[i] = dL/dy × w[i]   (activation gradient)
//
// Sequential 2-cycle split (vs single-cycle original):
//   COMPUTE_W: compute and latch 9 dL/dW values (grad_out fanout = 9)
//   COMPUTE_X: compute and latch 9 dL/dX values (grad_out fanout = 9)
// Halves the combinational fanout of grad_out (was 18, now 9 per cycle).
// Reduces instantiated multiplier logic by ~half vs parallel implementation.
//
// FSM: IDLE → COMPUTE_W(1) → COMPUTE_X(1) → DONE_ST(1) → IDLE
// Backward latency: 4 cycles  (was 3 in single-cycle design)
//
// fp32_mul_bf16: FP32 × BF16 → FP32 (synthesizable, no real/bitstoreal)
// ============================================================================

module grad_core (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    input  logic [31:0]  grad_out,       // dL/dy  (FP32)
    input  logic [143:0] weights_flat,   // 9 × BF16: w[i] = bits [16i+15:16i]
    input  logic [143:0] ifms_flat,      // 9 × BF16: x[i] = bits [16i+15:16i]
    output logic [287:0] grad_w_flat,    // 9 × FP32: gw[i] = bits [32i+31:32i]
    output logic [287:0] grad_x_flat,    // 9 × FP32: gx[i] = bits [32i+31:32i]
    output logic         done,
    output logic         ready
);

    typedef enum logic [1:0] {
        IDLE      = 2'd0,
        COMPUTE_W = 2'd1,
        COMPUTE_X = 2'd2,
        DONE_ST   = 2'd3
    } state_t;

    state_t state;

    // -------------------------------------------------------------------------
    // fp32_mul_bf16 : FP32 × BF16 → FP32  (synthesizable)
    //
    // a : FP32  [31]=sign  [30:23]=exp(bias 127)  [22:0]=23-bit mantissa
    // b : BF16  [15]=sign  [14:7]=exp(bias 127)   [6:0]=7-bit mantissa
    //
    // 24×8 product; both zero-extended to 32 bits.
    // max(ma24) × max(mb8) = (2^24−1)×255 < 2^32 — no overflow.
    // prod[31]=1 → 10.xxx: mant = prod[30:8], exp = ea + eb − 127 + 1
    // prod[31]=0 → 01.xxx: mant = prod[29:7], exp = ea + eb − 127
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
            fp32_mul_bf16 = 32'h0;
        end else if (a[30:23] == 8'hFF || b[14:7] == 8'hFF) begin
            fp32_mul_bf16 = {a[31] ^ b[15], 8'hFF, 23'h0};
        end else begin
            sign_r  = a[31] ^ b[15];
            exp_sum = {2'b00, a[30:23]} + {2'b00, b[14:7]} - 10'd127;
            ma24    = {1'b1, a[22:0]};
            mb8     = {1'b1, b[6:0]};
            prod    = {8'h00, ma24} * {24'h00, mb8};

            if (exp_sum[9] || exp_sum == 10'h0) begin
                fp32_mul_bf16 = {sign_r, 31'h0};
            end else if (exp_sum >= 10'd255) begin
                fp32_mul_bf16 = {sign_r, 8'hFF, 23'h0};
            end else if (prod[31]) begin
                mant_r = prod[30:8];
                if (exp_sum >= 10'd254)
                    fp32_mul_bf16 = {sign_r, 8'hFF, 23'h0};
                else begin
                    exp_r         = exp_sum[7:0] + 8'd1;
                    fp32_mul_bf16 = {sign_r, exp_r, mant_r};
                end
            end else begin
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
    // Combinational weight-gradient stage (9 multiplies, grad_out fanout = 9)
    // -------------------------------------------------------------------------
    logic [287:0] comb_grad_w;

    always_comb begin
        for (int i = 0; i < 9; i++)
            comb_grad_w[i*32 +: 32] =
                fp32_mul_bf16(grad_out, ifms_flat[i*16 +: 16]);
    end

    // -------------------------------------------------------------------------
    // Combinational activation-gradient stage (9 multiplies, fanout = 9)
    // -------------------------------------------------------------------------
    logic [287:0] comb_grad_x;

    always_comb begin
        for (int i = 0; i < 9; i++)
            comb_grad_x[i*32 +: 32] =
                fp32_mul_bf16(grad_out, weights_flat[i*16 +: 16]);
    end

    // -------------------------------------------------------------------------
    // FSM — sequential 2-cycle split
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
                IDLE:      if (start) state <= COMPUTE_W;

                COMPUTE_W: begin
                    grad_w_flat <= comb_grad_w;   // latch dL/dW
                    state       <= COMPUTE_X;
                end

                COMPUTE_X: begin
                    grad_x_flat <= comb_grad_x;   // latch dL/dX
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
