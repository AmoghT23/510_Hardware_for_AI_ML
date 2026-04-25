
//LLM used - Gemini 3(Fast)
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
    input  logic              clk,  // 1-bit clock
    input  logic              rst,  // 1-bit active-high synchronous reset
    input  logic signed [7:0] a,    // 8-bit signed input A
    input  logic signed [7:0] b,    // 8-bit signed input B
    output logic signed [31:0] out  // 32-bit signed accumulator
);

    // Internal signal to hold the product of a * b
    // A signed 8-bit * 8-bit mult results in a 16-bit signed value
    logic signed [15:0] product;

    // Combinational multiplication
    assign product = a * b;

    // Sequential logic for the accumulator
    always_ff @(posedge clk) begin
        if (rst) begin
            out <= 32'sd0;
        end else begin
            // Sign-extension of the product to 32 bits is handled 
            // automatically by the compiler because 'product' and 'out' 
            // are both declared as 'signed'.
            out <= out + product;
        end
    end

endmodule