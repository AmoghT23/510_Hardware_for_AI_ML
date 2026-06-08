"""
test_pe_array.py
ECE 410/510 -- cocotb unit test for rtl/pe_array.sv

Built with N_TILES=16 (4x4 grid, N_OUT=4) to keep the IFM bus width
VPI-manageable (16*144 = 2304 bits).  Functional correctness scales to
any N_TILES since every tile is an identical independent compute_core_ti.

DUT ports (N_TILES=16):
  weights_packed      [143:0]    broadcast to all 16 tiles
  ifms_all_packed  [2303:0]    per-tile windows (tile t at [t*144+143:t*144])
  results_all      [511:0]     per-tile FP32 (tile t at [t*32+31:t*32])
  done_all, ready_all

Tests:
  test_functional   -- ramp weights x unit IFMs, all 16 results correct
  test_zero_skip    -- all-zero IFMs, done in <=4 cycles
  test_weight_reuse -- same weights, two different IFM tiles
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from ref_model import float_to_bf16, bf16_to_float, fp32_to_float, ref_dot_bf16

CLOCK_NS = 2
N_TILES  = 16   # matches runner -PN_TILES=16
N_OUT    = 4    # sqrt(N_TILES)
N_IN     = N_OUT + 2   # 6


# ── helpers ───────────────────────────────────────────────────────────────────

def pack_weights(weights):
    word = 0
    for i, w in enumerate(weights):
        word |= float_to_bf16(w) << (i * 16)
    return word


def make_ifm_small(seed=0):
    """6x6 IFM for the 4x4 (N_OUT=4) test grid."""
    return [[float((r * N_IN + c + seed) % 64) for c in range(N_IN)]
            for r in range(N_IN)]


def pack_ifms_small(ifm_6x6):
    """Pack 16 windows from a 6x6 IFM into a 16*144=2304-bit integer."""
    word = 0
    for t in range(N_TILES):
        row, col = t // N_OUT, t % N_OUT
        for k in range(9):
            i, j = k // 3, k % 3
            bf16 = float_to_bf16(ifm_6x6[row + i][col + j])
            word |= bf16 << (t * 144 + k * 16)
    return word


def ref_results_small(weights, ifm_6x6):
    """Compute 16 expected FP32 bit patterns."""
    results = []
    for t in range(N_TILES):
        row, col = t // N_OUT, t % N_OUT
        window = [ifm_6x6[row + i][col + j]
                  for i in range(3) for j in range(3)]
        results.append(ref_dot_bf16(weights, window))
    return results


async def reset(dut):
    dut.rst_n.value           = 0
    dut.start.value           = 0
    dut.weights_packed.value  = 0
    dut.ifms_all_packed.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def run_array(dut, weights, ifm_6x6):
    """Load data, pulse start, poll done_all, return (results, cycles)."""
    for _ in range(20):
        if int(dut.ready_all.value) == 1:
            break
        await RisingEdge(dut.clk)

    dut.weights_packed.value  = pack_weights(weights)
    dut.ifms_all_packed.value = pack_ifms_small(ifm_6x6)

    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    cycles = 1
    for _ in range(20):
        await RisingEdge(dut.clk)
        cycles += 1
        if int(dut.done_all.value) == 1:
            break

    raw = int(dut.results_all.value)
    results = [(raw >> (t * 32)) & 0xFFFF_FFFF for t in range(N_TILES)]
    return results, cycles


# ── Test 1: functional ────────────────────────────────────────────────────────

@cocotb.test()
async def test_functional(dut):
    """Ramp weights x variable IFMs: all 16 results must match reference."""
    cocotb.start_soon(Clock(dut.clk, CLOCK_NS, unit="ns").start())
    await reset(dut)

    weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    ifm     = make_ifm_small(seed=0)
    expected = ref_results_small(weights, ifm)

    got, cycles = await run_array(dut, weights, ifm)

    mismatches = sum(1 for g, e in zip(got, expected) if g != e)
    assert mismatches == 0, (
        f"{mismatches}/16 wrong. "
        f"tile0: got={fp32_to_float(got[0])} exp={fp32_to_float(expected[0])}"
    )
    dut._log.info(f"PASS functional: all 16 results correct, cycles={cycles}")
    dut._log.info(f"  tile[0]={fp32_to_float(got[0])}  tile[15]={fp32_to_float(got[15])}")


# ── Test 2: zero-skip ─────────────────────────────────────────────────────────

@cocotb.test()
async def test_zero_skip(dut):
    """All-zero IFMs: all results=0, done in <=4 cycles (zero-skip path)."""
    cocotb.start_soon(Clock(dut.clk, CLOCK_NS, unit="ns").start())
    await reset(dut)

    weights  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    ifm_zero = [[0.0] * N_IN for _ in range(N_IN)]

    got, cycles = await run_array(dut, weights, ifm_zero)

    assert all(r == 0 for r in got), "Non-zero result with zero IFMs"
    assert cycles <= 4, f"Expected <=4 cycles on zero-skip, got {cycles}"
    dut._log.info(f"PASS zero-skip: all results=0, cycles={cycles}")


# ── Test 3: weight reuse ──────────────────────────────────────────────────────

@cocotb.test()
async def test_weight_reuse(dut):
    """Same weights, two different IFM tiles: both correct and distinct."""
    cocotb.start_soon(Clock(dut.clk, CLOCK_NS, unit="ns").start())
    await reset(dut)

    weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    ifm_a   = make_ifm_small(seed=0)
    ifm_b   = make_ifm_small(seed=17)

    exp_a = ref_results_small(weights, ifm_a)
    exp_b = ref_results_small(weights, ifm_b)

    got_a, _ = await run_array(dut, weights, ifm_a)
    got_b, _ = await run_array(dut, weights, ifm_b)

    assert sum(1 for g, e in zip(got_a, exp_a) if g != e) == 0, "Tile A results wrong"
    assert sum(1 for g, e in zip(got_b, exp_b) if g != e) == 0, "Tile B results wrong"
    assert got_a != got_b, "Results identical for different IFM tiles"

    dut._log.info(f"PASS weight_reuse")
    dut._log.info(f"  a[0]={fp32_to_float(got_a[0])}  b[0]={fp32_to_float(got_b[0])}")
