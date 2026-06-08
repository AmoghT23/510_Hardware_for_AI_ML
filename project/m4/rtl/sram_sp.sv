`timescale 1ns/1ps

// ============================================================================
// sram_sp.sv  —  Parameterized single-port behavioral SRAM
// ECE 410/510 — M3-TI Phase 2 infrastructure
//
// Parameters:
//   DEPTH  number of entries
//   WIDTH  bits per entry
//
// Port notes:
//   Synchronous write (registered on posedge clk when we=1).
//   Asynchronous read  (rdata = mem[addr] combinatorially, no clock edge).
//   No output register — caller registers rdata if needed.
//
// In Phase 2 this module is instantiated by interface_ti.sv to hold the
// 9-entry × 16-bit BF16 weight tile.  Phase 4 will reuse it for the
// larger 4 MB scratchpad by changing DEPTH/WIDTH parameters.
// ============================================================================

module sram_sp #(
    parameter int DEPTH = 16,
    parameter int WIDTH = 16
) (
    input  logic                         clk,
    input  logic                         we,
    input  logic [$clog2(DEPTH)-1:0]    addr,
    input  logic [WIDTH-1:0]            wdata,
    output logic [WIDTH-1:0]            rdata
);

    logic [WIDTH-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk)
        if (we) mem[addr] <= wdata;

    assign rdata = mem[addr];   // async read

endmodule
