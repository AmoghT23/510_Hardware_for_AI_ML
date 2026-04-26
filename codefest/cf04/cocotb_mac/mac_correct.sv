//LLM used - Claude 4.7 Opus
/* Prompt - I'm designing a AI accelerator chiplet for which i have to build a building block "INT8 MAC unit". Can you generate this block in SystemVerilog 
Specifications :
Module name: mac
Inputs: clk (1-bit), rst (1-bit, active-high synchronous reset), a (8-bit signed), b (8-bit
signed)
Output: out (32-bit signed accumulator)
Behavior: On each rising clock edge: if rst is high, set out to 0; else add a×b to out.
Constraints: Synthesizable SystemVerilog only. No initial blocks, no $display, no delays
(#). Use always_ff.
*/
module mac (
    input  logic        clk,
    input  logic        rst,
    input  logic [7:0]  a,
    input  logic [7:0]  b,
    output logic [31:0] out
);

    always_ff @(posedge clk) begin
        if (rst)
            out <= 32'sd0;
        else
            out <= $signed(out) + ($signed(a) * $signed(b));
    end

endmodule