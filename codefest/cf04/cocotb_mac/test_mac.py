# test_mac.py — cocotb testbench for mac_correct.v
# Run: make SIM=icarus
#
# Tests
#   test_mac_basic    : [a=3,b=4] x3 cycles, rst, [a=-5,b=2] x2 cycles
#   test_mac_overflow : accumulator wraps (not saturates) past 2^31 - 1

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge, Timer


# ------------------------------------------------------------------ #
#  test 1 — basic accumulation + mid-stream reset
# ------------------------------------------------------------------ #
@cocotb.test()
async def test_mac_basic(dut):
    """[a=3,b=4] for 3 cycles, assert rst, [a=-5,b=2] for 2 cycles."""

    dut.clk.value = 0
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    # --- initial reset -------------------------------------------
    dut.rst.value = 1
    dut.a.value   = 0
    dut.b.value   = 0
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert dut.out.value.to_signed() == 0, "reset failed"
    await Timer(1, unit="ps")
    dut._log.info("CYCLE 0 | RESET (init)        | out = 0  PASS")

    # --- a=3, b=4 for 3 cycles -----------------------------------
    dut.rst.value = 0
    dut.a.value   = 3
    dut.b.value   = 4
    for cycle, expected in enumerate([12, 24, 36], start=1):
        await RisingEdge(dut.clk)
        await ReadOnly()
        await Timer(1, unit="ps")
        got = dut.out.value.to_signed()
        assert got == expected, f"cycle {cycle}: expected {expected}, got {got}"
        dut._log.info(
            f"CYCLE {cycle} | a=3 b=4 ({cycle}/3)       "
            f"| out = {got:>6d}  exp = {expected:>6d}  PASS"
        )

    # --- mid-stream reset ----------------------------------------
    dut.rst.value = 1
    dut.a.value   = 0
    dut.b.value   = 0
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert dut.out.value.to_signed() == 0, "mid-stream reset failed"
    await Timer(1, unit="ps")
    dut._log.info("CYCLE 4 | RESET (mid-stream)  | out =      0  PASS")

    # --- a=-5, b=2 for 2 cycles ----------------------------------
    dut.rst.value = 0
    await Timer(1, unit="ps")
    dut.a.value   = 0xFB
    dut.b.value   = 2
    await Timer(1, unit="ps")
    dut._log.info(
        f"NEGATIVE STIMULUS | a={dut.a.value.to_unsigned()} signed={dut.a.value.to_signed()} bin={str(dut.a.value)} b={dut.b.value.to_unsigned()}"
    )
    for cycle, expected in enumerate([-10, -20], start=5):
        await RisingEdge(dut.clk)
        await ReadOnly()
        await Timer(1, unit="ps")
        got = dut.out.value.to_signed()
        assert got == expected, f"cycle {cycle}: expected {expected}, got {got}"
        dut._log.info(
            f"CYCLE {cycle} | a=-5 b=2 ({cycle-4}/2)      "
            f"| out = {got:>6d}  exp = {expected:>6d}  PASS"
        )

    dut._log.info("test_mac_basic — all 7 cycles PASS")


# ------------------------------------------------------------------ #
#  test 2 — overflow behaviour
# ------------------------------------------------------------------ #
@cocotb.test()
async def test_mac_overflow(dut):
    """
    Accumulator WRAPS (not saturates) when it crosses 2^31 - 1.

    Maximum INT8 signed product: (-128) * (-128) = +16384.
    Reaching overflow from zero takes about 131072 accumulation cycles.
    The test runs long enough to observe the first wrap-around event.

    Expected behaviour
    ------------------
    Cycle N : out reaches the largest representable signed 32-bit value.
    Cycle N+1: out wraps into the negative range with two's complement rollover.
    Result  : WRAP — no saturation logic in the RTL.
    """

    PRODUCT  = 16384                    # (-128) * (-128)
    MAX_S32  = 2**31 - 1                # 2 147 483 647

    dut.clk.value = 0
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    # Reset
    dut.rst.value = 1
    dut.a.value = 0
    dut.b.value = 0
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert dut.out.value.to_signed() == 0
    await Timer(1, unit="ps")
    dut.rst.value = 0

    # Inputs: a = -128 (0x80), b = -128 (0x80) → product = +16384
    dut.a.value = 0x80
    dut.b.value = 0x80

    prev_out = None
    got_wrap = None
    for cycle in range(200000):
        await RisingEdge(dut.clk)
        await ReadOnly()
        await Timer(1, unit="ps")
        current_out = dut.out.value.to_signed()
        if current_out < 0:
            got_wrap = current_out
            break
        prev_out = current_out

    assert got_wrap is not None, "Overflow did not occur within 200000 cycles"
    assert prev_out is not None, "Accumulator never produced a positive value before overflow"

    expected_wrap = ((prev_out + PRODUCT) & 0xFFFFFFFF)
    if expected_wrap & (1 << 31):
        expected_wrap = expected_wrap - (1 << 32)

    dut._log.info(f"Pre-overflow  | out = {prev_out:>14d}")
    dut._log.info(f"Post-overflow | out = {got_wrap:>14d}  (expected {expected_wrap})")
    dut._log.info("RESULT: Design WRAPS — two's complement rollover, no saturation.")
    assert got_wrap == expected_wrap, f"wrap: expected {expected_wrap}, got {got_wrap}"

    dut._log.info("test_mac_overflow PASS")
