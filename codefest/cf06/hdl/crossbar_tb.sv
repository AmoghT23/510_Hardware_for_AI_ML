`timescale 1ns/1ps

// Testbench for crossbar_mac (iverilog-11 compatible)
//
// Weight matrix (weight[i][j], i=input row, j=output col):
//   row 0: [+1, -1, +1, -1]
//   row 1: [+1, +1, -1, -1]
//   row 2: [-1, +1, +1, -1]
//   row 3: [-1, -1, -1, +1]
//
// Input vector: in = [10, 20, 30, 40]
//
// Hand-calculated expected outputs (out[j] = sum_i weight[i][j]*in[i]):
//   out[0] =  1*10 +  1*20 + (-1)*30 + (-1)*40 = -40
//   out[1] = -1*10 +  1*20 +  1*30  + (-1)*40 =   0
//   out[2] =  1*10 + (-1)*20 + 1*30 + (-1)*40 = -20
//   out[3] = -1*10 + (-1)*20 + (-1)*30 + 1*40 = -20

module crossbar_tb;

    // -------------------------------------------------------------------
    // Parameters (must match DUT)
    // -------------------------------------------------------------------
    localparam integer N         = 4;
    localparam integer IN_WIDTH  = 8;
    localparam integer OUT_WIDTH = 10;

    // -------------------------------------------------------------------
    // DUT signals – packed 2D to match DUT ports
    // -------------------------------------------------------------------
    reg                          clk;
    reg                          rst_n;
    reg                          load_weights;
    reg  [N-1:0][N-1:0]         weight_in;
    reg  [N-1:0][IN_WIDTH-1:0]  in_data;
    wire [N-1:0][OUT_WIDTH-1:0] out_data;

    // -------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------
    crossbar_mac #(
        .N        (N),
        .IN_WIDTH (IN_WIDTH),
        .OUT_WIDTH(OUT_WIDTH)
    ) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .load_weights(load_weights),
        .weight_in   (weight_in),
        .in_data     (in_data),
        .out_data    (out_data)
    );

    // -------------------------------------------------------------------
    // Clock: 10 ns period
    // -------------------------------------------------------------------
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // -------------------------------------------------------------------
    // Scoreboard counters
    // -------------------------------------------------------------------
    integer pass_count;
    integer fail_count;

    // -------------------------------------------------------------------
    // Check helper task (no 'automatic', no 'int', no '++')
    // -------------------------------------------------------------------
    task check_port;
        input integer          port_idx;
        input [OUT_WIDTH-1:0]  actual;
        input integer          expected;
        begin
            if ($signed(actual) === expected) begin
                $display("  [PASS] out[%0d] = %4d", port_idx, $signed(actual));
                pass_count = pass_count + 1;
            end else begin
                $display("  [FAIL] out[%0d] = %4d  (expected %4d)",
                         port_idx, $signed(actual), expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // -------------------------------------------------------------------
    // Stimulus
    // -------------------------------------------------------------------
    integer idx;

    initial begin
        $dumpfile("crossbar_tb.vcd");
        $dumpvars(0, crossbar_tb);

        pass_count   = 0;
        fail_count   = 0;

        // Initialise all signals
        rst_n        = 1'b0;
        load_weights = 1'b0;
        weight_in    = {(N*N){1'b0}};
        in_data      = {(N*IN_WIDTH){1'b0}};

        // Hold reset for 2 rising edges
        repeat (2) @(posedge clk);
        #1; rst_n = 1'b1;

        // =================================================================
        // TEST 1 – Required test vector
        //   weights = [[+1,-1,+1,-1],[+1,+1,-1,-1],[-1,+1,+1,-1],[-1,-1,-1,+1]]
        //   input   = [10, 20, 30, 40]
        //   expected= [-40, 0, -20, -20]
        //
        //   row 0: j0=+1, j1=-1, j2=+1, j3=-1  => 4'b0101
        //   row 1: j0=+1, j1=+1, j2=-1, j3=-1  => 4'b0011
        //   row 2: j0=-1, j1=+1, j2=+1, j3=-1  => 4'b0110
        //   row 3: j0=-1, j1=-1, j2=-1, j3=+1  => 4'b1000
        // =================================================================
        weight_in[0] = 4'b0101;
        weight_in[1] = 4'b0011;
        weight_in[2] = 4'b0110;
        weight_in[3] = 4'b1000;
        load_weights  = 1'b1;

        in_data[0] = 8'sd10;
        in_data[1] = 8'sd20;
        in_data[2] = 8'sd30;
        in_data[3] = 8'sd40;

        // Cycle 1: weight register latches weight_in
        @(posedge clk); #1;
        load_weights = 1'b0;

        // Cycle 2: output register captures MAC result
        @(posedge clk); #1;

        $display("\n=== TEST 1: in=[10,20,30,40], weights=[[+1,-1,+1,-1],[+1,+1,-1,-1],[-1,+1,+1,-1],[-1,-1,-1,+1]] ===");
        check_port(0, out_data[0], -40);
        check_port(1, out_data[1],   0);
        check_port(2, out_data[2], -20);
        check_port(3, out_data[3], -20);

        // =================================================================
        // TEST 2 – All-zero input => all outputs must be 0
        // =================================================================
        in_data = {(N*IN_WIDTH){1'b0}};
        @(posedge clk); #1;
        @(posedge clk); #1;

        $display("\n=== TEST 2: in=[0,0,0,0] (same weights) ===");
        check_port(0, out_data[0], 0);
        check_port(1, out_data[1], 0);
        check_port(2, out_data[2], 0);
        check_port(3, out_data[3], 0);

        // =================================================================
        // TEST 3 – Verify async reset clears outputs
        // =================================================================
        in_data[0] = 8'sd10; in_data[1] = 8'sd20;
        in_data[2] = 8'sd30; in_data[3] = 8'sd40;
        @(posedge clk); #1;
        rst_n = 1'b0;          // assert reset asynchronously
        #3;                    // hold 3 ns

        $display("\n=== TEST 3: async reset asserted while inputs active ===");
        check_port(0, out_data[0], 0);
        check_port(1, out_data[1], 0);
        check_port(2, out_data[2], 0);
        check_port(3, out_data[3], 0);

        // Release reset and reload weights to verify recovery
        @(posedge clk); #1; rst_n = 1'b1;
        weight_in[0] = 4'b0101; weight_in[1] = 4'b0011;
        weight_in[2] = 4'b0110; weight_in[3] = 4'b1000;
        load_weights = 1'b1;
        @(posedge clk); #1; load_weights = 1'b0;
        @(posedge clk); #1;

        $display("\n=== TEST 4: recovery after reset, in=[10,20,30,40] ===");
        check_port(0, out_data[0], -40);
        check_port(1, out_data[1],   0);
        check_port(2, out_data[2], -20);
        check_port(3, out_data[3], -20);

        // =================================================================
        // Summary
        // =================================================================
        $display("\n============================================================");
        $display("  Simulation complete: %0d PASS, %0d FAIL", pass_count, fail_count);
        if (fail_count == 0)
            $display("  >>> ALL TESTS PASSED <<<");
        else
            $display("  >>> SOME TESTS FAILED <<<");
        $display("============================================================\n");

        $finish;
    end

endmodule