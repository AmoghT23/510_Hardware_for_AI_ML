module sigtest; reg [7:0] a; initial begin a=8'hfb; $display("signed=%0d unsigned=%0d hex=%h", $signed(a), a, a); $finish; end endmodule
