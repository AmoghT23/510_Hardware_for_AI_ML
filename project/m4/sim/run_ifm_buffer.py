"""sim/run_ifm_buffer.py -- cocotb runner for ifm_buffer unit tests."""
import sys
from pathlib import Path
from cocotb_tools.runner import get_runner

ROOT      = Path(__file__).resolve().parent.parent
RTL       = ROOT / "rtl"
TB        = ROOT / "tb"
BUILD_DIR = ROOT / "sim" / "ifm_buffer_build"

def main():
    if str(TB) not in sys.path:
        sys.path.insert(0, str(TB))
    runner = get_runner("icarus")
    runner.build(
        sources=[str(RTL / "ifm_buffer.sv")],
        hdl_toplevel="ifm_buffer",
        build_args=["-g2012"],
        build_dir=str(BUILD_DIR),
        always=True,
    )
    runner.test(
        test_module="test_ifm_buffer",
        hdl_toplevel="ifm_buffer",
        build_dir=str(BUILD_DIR),
    )

if __name__ == "__main__":
    main()
