"""
run_array.py
ECE 410/510 — combined cocotb runner for ifm_buffer and pe_array unit tests

Runs both sub-module unit tests in sequence:
  1. ifm_buffer unit test
       Sources : rtl/ifm_buffer.sv
       Toplevel: ifm_buffer
       Tests   : tb/test_ifm_buffer.py
  2. pe_array unit test
       Sources : rtl/compute_core_ti.sv  rtl/pe_array.sv
       Toplevel: pe_array  (N_TILES=16 to keep VPI bus widths manageable)
       Tests   : tb/test_pe_array.py

Full 1024-tile integration is verified by sim/run_ti.py (top_ti toplevel).

Run from the project directory:
    python sim/run_array.py
"""

import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parent.parent
RTL  = ROOT / "rtl"
TB   = ROOT / "tb"


def run_ifm_buffer():
    build_dir = str(ROOT / "sim" / "array_ifm_build")
    runner = get_runner("icarus")
    runner.build(
        sources=[str(RTL / "ifm_buffer.sv")],
        hdl_toplevel="ifm_buffer",
        build_args=["-g2012"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(
        test_module="test_ifm_buffer",
        hdl_toplevel="ifm_buffer",
        build_dir=build_dir,
    )


def run_pe_array():
    build_dir = str(ROOT / "sim" / "array_pe_build")
    runner = get_runner("icarus")
    runner.build(
        sources=[
            str(RTL / "compute_core_ti.sv"),
            str(RTL / "pe_array.sv"),
        ],
        hdl_toplevel="pe_array",
        build_args=["-g2012", "-DCOCOTB_SIM", "-P", "pe_array.N_TILES=16"],
        build_dir=build_dir,
        always=True,
    )
    runner.test(
        test_module="test_pe_array",
        hdl_toplevel="pe_array",
        build_dir=build_dir,
    )


def main():
    if str(TB) not in sys.path:
        sys.path.insert(0, str(TB))

    run_ifm_buffer()
    run_pe_array()


if __name__ == "__main__":
    main()
