
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
    input  logic              clk,
    input  logic              rst,
    input  logic signed [7:0] a,
    input  logic signed [7:0] b,
    output logic signed [31:0] out
);

    // Intermediate signal for the product.
    // 8x8 signed multiplication results in a 16-bit signed value.
    logic signed [15:0] product;

    // Explicitly casting or ensuring signed context for multiplication
    assign product = a * b;

    // Synchronous Reset: Only clk in the sensitivity list.
    always_ff @(posedge clk) begin
        if (rst) begin
            out <= 32'sd0;
        end else begin
            // When adding 16-bit 'product' to 32-bit 'out', 
            // the signed property ensures the sign-bit (MSB) 
            // of product is replicated to fill bits [31:16].
            out <= out + product;
        end
    end

endmodule
