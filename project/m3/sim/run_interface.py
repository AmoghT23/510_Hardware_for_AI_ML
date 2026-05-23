"""
run_interface.py
ECE 410/510 — M3-TI cocotb runner for conv_interface

Run from the m3_ti directory:
    cd d:\\PSU\\Q3\\HW_AI-ML\\project\\m3_ti
    python sim/run_interface.py
"""

import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT      = Path(__file__).resolve().parent.parent   # m3_ti/
RTL       = ROOT / "rtl"
TB        = ROOT / "tb"
BUILD_DIR = ROOT / "sim" / "interface_build"

def main():
    if str(TB) not in sys.path:
        sys.path.insert(0, str(TB))

    runner = get_runner("icarus")

    runner.build(
        sources=[str(RTL / "interface.sv")],
        hdl_toplevel="conv_interface",
        build_args=["-g2012"],
        build_dir=str(BUILD_DIR),
        always=True,
    )

    runner.test(
        test_module="test_interface",
        hdl_toplevel="conv_interface",
        build_dir=str(BUILD_DIR),
        waves=True,
    )

if __name__ == "__main__":
    main()

