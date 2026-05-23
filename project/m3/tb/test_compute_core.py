"""
test_compute_core.py
ECE 410/510 — cocotb testbench for compute_core (INT8 QAT revision)

Drives the DUT ports directly from Python — no separate SystemVerilog
AXI interface module required.  cocotb acts as the AXI-side host:
it controls all input signals and checks all output signals.

DUT interface (M3 INT8):
    clk, rst_n, start
    weight_data [7:0] (INT8 signed), weight_valid
    ifm_data    [7:0] (INT8 signed), ifm_valid
    result      [31:0] (INT32 signed), done, ready

Test coverage:
    T1  ramp_weights      weights=[1..9],  ifm=[1]*9          -> 45
    T2  alternating_sign  weights=[1,-1,2,-2,3,-3,4,-4,1],
                          ifm=[2]*9                            -> 2
    T3  max_positive      weights=[127]*9, ifm=[1]*9           -> 1143
    T4  negative_result   weights=[-1]*9,  ifm=[1]*9           -> -9
    T5  zero_weights      weights=[0]*9,   ifm=[127]*9         -> 0
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


def to_uint8(v: int) -> int:
    """Return the 8-bit unsigned representation of a signed INT8 value."""
    return v & 0xFF


async def reset(dut):
    """Assert synchronous active-low reset for 4 cycles, then release."""
    dut.rst_n.value       = 0
    dut.start.value       = 0
    dut.weight_valid.value = 0
    dut.ifm_valid.value   = 0
    dut.weight_data.value = 0
    dut.ifm_data.value    = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def run_tile(dut, weights: list, ifms: list) -> int:
    """
    Load one 3×3 INT8 tile into the DUT and return the signed INT32 result.

    Protocol (matches compute_core FSM):
      1. Wait for ready (IDLE state).
      2. Pulse start for one cycle -> FSM enters LOAD.
      3. Drive 9 weight/IFM pairs with valid=1, one per cycle.
      4. Wait for done pulse -> read result.
    """
    # Wait until core is idle
    for _ in range(50):
        if dut.ready.value == 1:
            break
        await RisingEdge(dut.clk)

    # Pulse start
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Stream 9 INT8 pairs
    dut.weight_valid.value = 1
    dut.ifm_valid.value    = 1
    for w, x in zip(weights, ifms):
        dut.weight_data.value = to_uint8(w)
        dut.ifm_data.value    = to_uint8(x)
        await RisingEdge(dut.clk)
    dut.weight_valid.value = 0
    dut.ifm_valid.value    = 0

    # Wait for done (COMPUTE + DONE_ST = 2 cycles after last load edge)
    for _ in range(10):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            return dut.result.value.to_signed()
    raise cocotb.result.TestFailure("Timeout: done never asserted")


# ── Test 1 ────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_ramp_weights(dut):
    """weights=[1..9], IFM=[1]*9 -> dot product = 45"""
    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)

    weights  = list(range(1, 10))   # [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ifms     = [1] * 9
    expected = sum(w * x for w, x in zip(weights, ifms))  # 45

    result = await run_tile(dut, weights, ifms)
    assert result == expected, f"FAIL  got={result}  want={expected}"
    dut._log.info(f"PASS  T1  result={result}  (expected {expected})")


# ── Test 2 ────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_alternating_sign(dut):
    """alternating-sign weights, IFM=[2]*9 -> 2"""
    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)

    weights  = [1, -1, 2, -2, 3, -3, 4, -4, 1]
    ifms     = [2] * 9
    expected = sum(w * x for w, x in zip(weights, ifms))  # 2

    result = await run_tile(dut, weights, ifms)
    assert result == expected, f"FAIL  got={result}  want={expected}"
    dut._log.info(f"PASS  T2  result={result}  (expected {expected})")


# ── Test 3 ────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_max_positive(dut):
    """weights=[127]*9, IFM=[1]*9 -> 1143  (max INT8 weight, unit IFM)"""
    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)

    weights  = [127] * 9
    ifms     = [1]   * 9
    expected = sum(w * x for w, x in zip(weights, ifms))  # 1143

    result = await run_tile(dut, weights, ifms)
    assert result == expected, f"FAIL  got={result}  want={expected}"
    dut._log.info(f"PASS  T3  result={result}  (expected {expected})")


# ── Test 4 ────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_negative_result(dut):
    """weights=[-1]*9, IFM=[1]*9 -> -9  (verifies signed output path)"""
    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)

    weights  = [-1] * 9
    ifms     = [1]  * 9
    expected = sum(w * x for w, x in zip(weights, ifms))  # -9

    result = await run_tile(dut, weights, ifms)
    assert result == expected, f"FAIL  got={result}  want={expected}"
    dut._log.info(f"PASS  T4  result={result}  (expected {expected})")


# ── Test 5 ────────────────────────────────────────────────────────────────────
@cocotb.test()
async def test_zero_weights(dut):
    """weights=[0]*9, IFM=[127]*9 -> 0  (zero-product check)"""
    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)

    weights  = [0]   * 9
    ifms     = [127] * 9
    expected = 0

    result = await run_tile(dut, weights, ifms)
    assert result == expected, f"FAIL  got={result}  want={expected}"
    dut._log.info(f"PASS  T5  result={result}  (expected {expected})")
