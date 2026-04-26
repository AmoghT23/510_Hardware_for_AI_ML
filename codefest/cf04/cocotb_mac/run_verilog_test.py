from pathlib import Path
import subprocess
base = Path('tmp_verilog_test')
base.mkdir(exist_ok=True)
(base/'mac_correct.v').write_text('''// test module
module mac(
 input clk, rst,
 input [7:0] a, b,
 output reg [31:0] out);
 always @(posedge clk) begin
   if (rst) out <= 0;
   else out <= out + ($signed(a) * $signed(b));
 end
endmodule
''')
(base/'tb.v').write_text('''module tb;
 reg clk=0, rst;
 reg [7:0] a,b;
 wire [31:0] out;
 mac dut(.clk(clk), .rst(rst), .a(a), .b(b), .out(out));
 initial begin
   $display("OUT=%0d OUTHEX=%h", out, out);
   rst = 1; a = 0; b = 0;
   #5 rst = 0; a = 8'hfb; b = 2;
   #10 $display("OUT2=%0d OUTHEX2=%h", out, out);
   #10 $finish;
 end
 always #5 clk = ~clk;
 endmodule
''')
p = subprocess.run(['C:/iverilog/bin/iverilog','-o',str(base/'sim.vvp'),str(base/'mac_correct.v'),str(base/'tb.v')], capture_output=True, text=True)
print('compile rc', p.returncode)
print(p.stdout)
print(p.stderr)
if p.returncode == 0:
    q = subprocess.run(['C:/iverilog/bin/vvp', str(base/'sim.vvp')], capture_output=True, text=True)
    print(q.stdout)
    print(q.stderr)
