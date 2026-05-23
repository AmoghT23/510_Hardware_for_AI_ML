"""
run_top.py
ECE 410/510 — M3-TI cocotb runner for integrated top module

Compiles rtl/interface.sv + rtl/compute_core.sv + rtl/top.sv and runs
the end-to-end co-simulation in tb/test_top.py.

Run from the m3_ti directory:
    cd d:\\PSU\\Q3\\HW_AI-ML\\project\\m3_ti
    python sim/run_top.py
"""

import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT      = Path(__file__).resolve().parent.parent   # m3_ti/
RTL       = ROOT / "rtl"
TB        = ROOT / "tb"
BUILD_DIR = ROOT / "sim" / "top_build"

def main():
    if str(TB) not in sys.path:
        sys.path.insert(0, str(TB))

    runner = get_runner("icarus")

    runner.build(
        sources=[
            str(RTL / "interface.sv"),
            str(RTL / "compute_core.sv"),
            str(RTL / "top.sv"),
        ],
        hdl_toplevel="top",
        build_args=["-g2012", "-DCOCOTB_SIM"],
        build_dir=str(BUILD_DIR),
        always=True,
    )

    runner.test(
        test_module="test_top",
        hdl_toplevel="top",
        build_dir=str(BUILD_DIR),
        waves=True,
    )

if __name__ == "__main__":
    main()
