"""
run_core_rev4.py
Runner for compute_core_ti Rev4 direct testbench.
Compiles compute_core_ti.sv alone (no interface wrapper) and runs test_core_rev4.py.

Run from the m4/ directory:
    python sim/run_core_rev4.py
"""

import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

ROOT      = Path(__file__).resolve().parent.parent   # m4/
RTL       = ROOT / "rtl"
TB        = ROOT / "tb"
BUILD_DIR = ROOT / "sim" / "rev4_build"

def main():
    if str(TB) not in sys.path:
        sys.path.insert(0, str(TB))

    runner = get_runner("icarus")

    runner.build(
        sources=[str(RTL / "compute_core_ti.sv")],
        hdl_toplevel="compute_core_ti",
        build_args=["-g2012", "-DCOCOTB_SIM"],
        build_dir=str(BUILD_DIR),
        always=True,
    )

    runner.test(
        test_module="test_core_rev4",
        hdl_toplevel="compute_core_ti",
        build_dir=str(BUILD_DIR),
        waves=True,
    )

if __name__ == "__main__":
    main()
