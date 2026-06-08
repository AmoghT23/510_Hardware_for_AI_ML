"""
run_ti.py
ECE 410/510 — 32×32 BF16 Conv2d cocotb runner

Compiles:
  rtl/sram_sp.sv
  rtl/grad_core.sv
  rtl/ifm_buffer.sv
  rtl/pe_array.sv
  rtl/compute_core_ti.sv
  rtl/interface_ti.sv
  rtl/top.sv

Runs:
  tb/test_forward.py   — BF16 forward pass + weight reuse (37-beat IFM, 1024 results)
  tb/test_backward.py  — dL/dW backward pass (grad_core, tile-0 window)

Run from the m3_ti directory:
    python sim/run_ti.py
"""

import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT      = Path(__file__).resolve().parent.parent   # m3_ti/
RTL       = ROOT / "rtl"
TB        = ROOT / "tb"
BUILD_DIR = ROOT / "sim" / "ti_build"

def main():
    if str(TB) not in sys.path:
        sys.path.insert(0, str(TB))

    runner = get_runner("icarus")

    runner.build(
        sources=[
            str(RTL / "sram_sp.sv"),
            str(RTL / "grad_core.sv"),
            str(RTL / "ifm_buffer.sv"),
            str(RTL / "pe_array.sv"),
            str(RTL / "compute_core_ti.sv"),
            str(RTL / "interface_ti.sv"),
            str(RTL / "top.sv"),
        ],
        hdl_toplevel="top_ti",
        build_args=["-g2012", "-DCOCOTB_SIM"],
        build_dir=str(BUILD_DIR),
        always=True,
    )

    runner.test(
        test_module="test_forward,test_backward",
        hdl_toplevel="top_ti",
        build_dir=str(BUILD_DIR),
        waves=True,
    )

if __name__ == "__main__":
    main()
