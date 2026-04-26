module sigtest2; reg [7:0] a,b; initial begin a=8'hfb; b=2; $display("A=%0d B=%0d PROD=%0d HEX=%h", $signed(a), $signed(b), $signed(a)*$signed(b), $signed(a)*$signed(b)); $finish; end endmodule
