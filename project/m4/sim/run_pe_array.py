"""sim/run_pe_array.py -- cocotb runner for pe_array unit tests.
Builds with N_TILES=16 (4x4 grid) to keep bus widths VPI-manageable.
Full 1024-tile correctness is verified through the top-level integration test.
"""
import sys
from pathlib import Path
from cocotb_tools.runner import get_runner

ROOT      = Path(__file__).resolve().parent.parent
RTL       = ROOT / "rtl"
TB        = ROOT / "tb"
BUILD_DIR = ROOT / "sim" / "pe_array_build"

def main():
    if str(TB) not in sys.path:
        sys.path.insert(0, str(TB))
    runner = get_runner("icarus")
    runner.build(
        sources=[
            str(RTL / "compute_core_ti.sv"),
            str(RTL / "pe_array.sv"),
        ],
        hdl_toplevel="pe_array",
        build_args=["-g2012", "-DCOCOTB_SIM", "-P", "pe_array.N_TILES=16"],
        build_dir=str(BUILD_DIR),
        always=True,
    )
    runner.test(
        test_module="test_pe_array",
        hdl_toplevel="pe_array",
        build_dir=str(BUILD_DIR),
    )

if __name__ == "__main__":
    main()
