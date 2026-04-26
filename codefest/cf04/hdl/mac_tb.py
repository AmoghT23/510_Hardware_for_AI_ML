# cocotb testbench for mac_correct.sv (INT8 MAC unit)
# Stimulus: [a=3, b=4] x3 cycles -> assert rst -> [a=-5, b=2] x2 cycles
#
# Run with Icarus:   make SIM=icarus TOPLEVEL=mac MODULE=mac_tb
# Run with Questasim: make SIM=questa TOPLEVEL=mac MODULE=mac_tb

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


# ------------------------------------------------------------------ #
#  Helpers: raw unsigned port values -> Python signed integers
# ------------------------------------------------------------------ #
def s8(val):
    """8-bit unsigned -> signed (-128..127)."""
    v = int(val)
    return v if v < 128 else v - 256


def s32(val):
    """32-bit unsigned -> signed."""
    v = int(val)
    return v if v < (1 << 31) else v - (1 << 32)


# ------------------------------------------------------------------ #
#  Test
# ------------------------------------------------------------------ #
@cocotb.test()
async def mac_accumulator_test(dut):
    """INT8 MAC: [a=3,b=4] x3, assert rst, [a=-5,b=2] x2."""

    # 10 ns clock (100 MHz)
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    # Initialise all inputs before any clock edge
    dut.rst.value = 1
    dut.a.value   = 0
    dut.b.value   = 0

    dut._log.info("=" * 60)
    dut._log.info("  mac_tb.py  —  INT8 MAC cocotb testbench")
    dut._log.info("  DUT: mac_correct.sv")
    dut._log.info("=" * 60)
    dut._log.info(
        f"{'CYCLE':>5}  {'PHASE':<22}  "
        f"{'rst':>3} {'a':>4} {'b':>4}  "
        f"{'out':>8}  {'exp':>8}  STATUS"
    )
    dut._log.info("-" * 60)

    cycle = 0

    async def tick(phase: str, expected: int):
        """Advance one clock, sample 1 ns after posedge, assert and log."""
        nonlocal cycle
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")          # settle after NBA update

        got    = s32(dut.out.value)
        a_disp = s8(dut.a.value)
        b_disp = int(dut.b.value)
        status = "PASS" if got == expected else f"*** FAIL *** (got {got})"

        dut._log.info(
            f"{cycle:>5}  {phase:<22}  "
            f"{int(dut.rst.value):>3} {a_disp:>4} {b_disp:>4}  "
            f"{got:>8}  {expected:>8}  {status}"
        )

        assert got == expected, (
            f"Cycle {cycle}: expected out={expected}, got {got}"
        )
        cycle += 1

    # ---- Cycle 0: initial synchronous reset -------------------------
    await tick("RESET (init)", 0)

    # ---- Cycles 1-3: a=3, b=4 --------------------------------------
    dut.rst.value = 0
    dut.a.value   = 3
    dut.b.value   = 4
    await tick("a=3 b=4 (cyc 1/3)", 12)
    await tick("a=3 b=4 (cyc 2/3)", 24)
    await tick("a=3 b=4 (cyc 3/3)", 36)

    # ---- Cycle 4: mid-stream reset (a,b left at 3,4; rst must win) --
    dut.rst.value = 1
    await tick("RESET (mid-stream)", 0)

    # ---- Cycles 5-6: a=-5, b=2 --------------------------------------
    dut.rst.value = 0
    dut.a.value   = (-5) & 0xFF     # 0xFB — two's complement encoding
    dut.b.value   = 2
    await tick("a=-5 b=2 (cyc 1/2)", -10)
    await tick("a=-5 b=2 (cyc 2/2)", -20)

    dut._log.info("=" * 60)
    dut._log.info("  7 cycles complete — all assertions PASS")
    dut._log.info("=" * 60)
