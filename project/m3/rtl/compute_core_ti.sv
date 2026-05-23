`timescale 1ns/1ps

// ============================================================================
// compute_core_ti.sv  --  BF16 Compute Core  (Synthesizable)
// ECE 410/510 -- Phase 1
//
// All arithmetic is synthesizable bit-manipulation.
// No real, $bitstoreal, $realtobits.  Safe for Yosys / OpenLane.
//
// Three helper functions:
//   count_lz24 -- 24-bit leading-zero encoder (casez priority encoder)
//   bf16_mul   -- BF16 x BF16 -> FP32  (sign XOR, exp add-bias, 8x8 mantissa)
//   fp32_add   -- FP32 + FP32 -> FP32  (align, add/sub, normalize)
//
// NOTE: all functions use "function_name = value" (Verilog-2001 style) --
//       Yosys does not support the "return" keyword inside function bodies.
//
// FSM: IDLE -> LOAD (9 cycles) -> COMPUTE (1 cycle) -> DONE_ST (1 cycle) -> IDLE
// ============================================================================

module compute_core_ti #(
    parameter int KERNEL_H = 3,
    parameter int KERNEL_W = 3,
    parameter int IN_CH    = 1
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [15:0] weight_data,    // BF16
    input  logic        weight_valid,
    input  logic [15:0] ifm_data,       // BF16
    input  logic        ifm_valid,
    output logic [31:0] result,         // FP32
    output logic        done,
    output logic        ready
);

    localparam int NUM_ELEM = KERNEL_H * KERNEL_W * IN_CH;
    localparam int CNT_W    = $clog2(NUM_ELEM + 1);

    typedef enum logic [1:0] {
        IDLE    = 2'd0,
        LOAD    = 2'd1,
        COMPUTE = 2'd2,
        DONE_ST = 2'd3
    } state_t;

    state_t           state;
    logic [15:0]      weight_buf [0:NUM_ELEM-1];
    logic [15:0]      ifm_buf    [0:NUM_ELEM-1];
    logic [CNT_W-1:0] cnt;
    logic [31:0]      result_reg;

    // -------------------------------------------------------------------------
    // count_lz24 : 24-bit leading-zero count (casez priority encoder)
    // Returns the number of leading zeros in x[23:0].
    // -------------------------------------------------------------------------
    function automatic logic [4:0] count_lz24(input logic [23:0] x);
        casez (x)
            24'b1???????????????????????: count_lz24 = 5'd0;
            24'b01??????????????????????: count_lz24 = 5'd1;
            24'b001?????????????????????: count_lz24 = 5'd2;
            24'b0001????????????????????: count_lz24 = 5'd3;
            24'b00001???????????????????: count_lz24 = 5'd4;
            24'b000001??????????????????: count_lz24 = 5'd5;
            24'b0000001?????????????????: count_lz24 = 5'd6;
            24'b00000001????????????????: count_lz24 = 5'd7;
            24'b000000001???????????????: count_lz24 = 5'd8;
            24'b0000000001??????????????: count_lz24 = 5'd9;
            24'b00000000001?????????????: count_lz24 = 5'd10;
            24'b000000000001????????????: count_lz24 = 5'd11;
            24'b0000000000001???????????: count_lz24 = 5'd12;
            24'b00000000000001??????????: count_lz24 = 5'd13;
            24'b000000000000001?????????: count_lz24 = 5'd14;
            24'b0000000000000001????????: count_lz24 = 5'd15;
            24'b00000000000000001???????: count_lz24 = 5'd16;
            24'b000000000000000001??????: count_lz24 = 5'd17;
            24'b0000000000000000001?????: count_lz24 = 5'd18;
            24'b00000000000000000001????: count_lz24 = 5'd19;
            24'b000000000000000000001???: count_lz24 = 5'd20;
            24'b0000000000000000000001??: count_lz24 = 5'd21;
            24'b00000000000000000000001?: count_lz24 = 5'd22;
            24'b000000000000000000000001: count_lz24 = 5'd23;
            default:                      count_lz24 = 5'd24;
        endcase
    endfunction

    // -------------------------------------------------------------------------
    // bf16_mul : BF16 x BF16 -> FP32  (synthesizable)
    // BF16: [15]=sign  [14:7]=exp(bias 127)  [6:0]=7-bit mantissa
    // -------------------------------------------------------------------------
    function automatic logic [31:0] bf16_mul(
        input logic [15:0] a,
        input logic [15:0] b
    );
        logic        sign_r;
        logic [9:0]  exp_sum;
        logic [7:0]  exp_r;
        logic [7:0]  ma8, mb8;
        logic [15:0] prod;
        logic [22:0] mant_r;

        if (a[14:7] == 8'h00 || b[14:7] == 8'h00) begin
            bf16_mul = 32'h0;
        end else if (a[14:7] == 8'hFF || b[14:7] == 8'hFF) begin
            bf16_mul = {a[15] ^ b[15], 8'hFF, 23'h0};
        end else begin
            sign_r  = a[15] ^ b[15];
            exp_sum = {2'b00, a[14:7]} + {2'b00, b[14:7]} - 10'd127;
            ma8     = {1'b1, a[6:0]};
            mb8     = {1'b1, b[6:0]};
            prod    = {8'h00, ma8} * {8'h00, mb8};

            if (exp_sum[9] || exp_sum == 10'h0) begin
                bf16_mul = {sign_r, 31'h0};
            end else if (exp_sum >= 10'd255) begin
                bf16_mul = {sign_r, 8'hFF, 23'h0};
            end else if (prod[15]) begin
                // 10.xxx form: result = 1.xxx x 2^(exp_sum+1)
                mant_r = {prod[14:1], 8'h00};
                if (exp_sum >= 10'd254)
                    bf16_mul = {sign_r, 8'hFF, 23'h0};
                else begin
                    exp_r    = exp_sum[7:0] + 8'd1;
                    bf16_mul = {sign_r, exp_r, mant_r};
                end
            end else begin
                // 01.xxx form: result = 1.xxx x 2^exp_sum
                mant_r = {prod[13:0], 9'h000};
                if (&exp_sum[7:0])
                    bf16_mul = {sign_r, 8'hFF, 23'h0};
                else begin
                    exp_r    = exp_sum[7:0];
                    bf16_mul = {sign_r, exp_r, mant_r};
                end
            end
        end
    endfunction

    // -------------------------------------------------------------------------
    // fp32_add : FP32 + FP32 -> FP32  (synthesizable)
    // Align smaller operand, add/sub mantissas, normalize result.
    // -------------------------------------------------------------------------
    function automatic logic [31:0] fp32_add(
        input logic [31:0] a,
        input logic [31:0] b
    );
        logic        sa, sb, sign_r;
        logic [7:0]  ea, eb, exp_r;
        logic [23:0] ma, mb, mb_sh;
        logic [7:0]  ediff;
        logic [24:0] sum_raw;
        logic [23:0] norm24;
        logic [4:0]  lz;

        sa = a[31]; ea = a[30:23]; ma = {1'b1, a[22:0]};
        sb = b[31]; eb = b[30:23]; mb = {1'b1, b[22:0]};

        if (ea == 8'h00) begin
            fp32_add = b;
        end else if (eb == 8'h00) begin
            fp32_add = a;
        end else begin
            // Swap so ea >= eb
            if (eb > ea)
                {sa, ea, ma, sb, eb, mb} = {sb, eb, mb, sa, ea, ma};

            ediff = ea - eb;
            mb_sh = (ediff >= 8'd24) ? 24'h0 : (mb >> ediff);

            if (sa == sb) begin
                sum_raw = {1'b0, ma} + {1'b0, mb_sh};
                sign_r  = sa;
            end else begin
                sum_raw = {1'b0, ma} - {1'b0, mb_sh};
                sign_r  = sa;
            end

            if (sum_raw == 25'h0) begin
                fp32_add = 32'h0;
            end else if (sum_raw[24]) begin
                // Carry out from addition: shift right 1, bump exponent
                if (ea == 8'hFE)
                    fp32_add = {sign_r, 8'hFF, 23'h0};
                else
                    fp32_add = {sign_r, ea + 8'd1, sum_raw[23:1]};
            end else begin
                // Normalize: shift left until leading 1 at bit 23
                lz = count_lz24(sum_raw[23:0]);
                if ({3'b000, lz} >= {1'b0, ea}) begin
                    fp32_add = {sign_r, 31'h0};   // underflow
                end else begin
                    norm24   = sum_raw[23:0] << lz;
                    exp_r    = ea - {3'b000, lz};
                    fp32_add = {sign_r, exp_r, norm24[22:0]};
                end
            end
        end
    endfunction

    // -------------------------------------------------------------------------
    // Combinational dot product
    // -------------------------------------------------------------------------
    logic [31:0] mac_bits;

    always @(*) begin
        mac_bits = 32'h0;
        for (int i = 0; i < NUM_ELEM; i++)
            mac_bits = fp32_add(mac_bits, bf16_mul(weight_buf[i], ifm_buf[i]));
    end

    // -------------------------------------------------------------------------
    // FSM + datapath
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state      <= IDLE;
            cnt        <= '0;
            result_reg <= 32'h0;
            done       <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= LOAD;
                        cnt   <= '0;
                    end
                end
                LOAD: begin
                    if (weight_valid && ifm_valid) begin
                        weight_buf[cnt] <= weight_data;
                        ifm_buf[cnt]    <= ifm_data;
                        if (cnt == CNT_W'(NUM_ELEM - 1)) begin
                            state <= COMPUTE;
                            cnt   <= '0;
                        end else
                            cnt <= cnt + 1'b1;
                    end
                end
                COMPUTE: begin
                    result_reg <= mac_bits;
                    state      <= DONE_ST;
                end
                DONE_ST: begin
                    done  <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

    assign result = result_reg;
    assign ready  = (state == IDLE);

`ifdef COCOTB_SIM
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, compute_core_ti);
    end
`endif

endmodule
