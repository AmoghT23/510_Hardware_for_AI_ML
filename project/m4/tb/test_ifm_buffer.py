"""
test_ifm_buffer.py
ECE 410/510 -- cocotb unit test for rtl/ifm_buffer.sv

Sends 37 AXI4-Stream beats carrying a 34x34 BF16 IFM tile and verifies
that all 1024 extracted 3x3 windows match the Python reference model.

Tests:
  test_load_and_corners  -- load tile, check t=0,31,992,1023 (corners)
  test_all_windows       -- load tile, verify all 1024 windows bit-exact
  test_reload            -- load a second tile, confirm buffer is overwritten
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from ref_model import float_to_bf16, bf16_to_float, build_ifm_tile_beats, ref_window


# ── helpers ───────────────────────────────────────────────────────────────────

def make_ifm(seed=1):
    """34x34 IFM with value = (row*34 + col + seed) % 256, as float."""
    return [[float((r * 34 + c + seed) % 256) for c in range(34)] for r in range(34)]


async def reset(dut):
    dut.rst_n.value         = 0
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tdata.value  = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def send_ifm_tile(dut, ifm_34x34):
    """Stream 37 beats, one per clock edge (two-await pattern, valid de-asserted
    between beats). load_done is now sticky — polls up to 5 cycles after last beat."""
    beats = build_ifm_tile_beats(ifm_34x34)
    for beat in beats:
        await RisingEdge(dut.clk)           # idle edge (valid=0)
        dut.s_axis_tdata.value  = beat
        dut.s_axis_tvalid.value = 1
        await RisingEdge(dut.clk)           # DUT samples beat here
        dut.s_axis_tvalid.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
        if int(dut.load_done.value) == 1:
            return True
    return False


def get_dut_window(wp_int, tile):
    """Extract 9 decoded BF16 floats for tile t from windows_packed integer."""
    window = []
    for k in range(9):
        bf16 = (wp_int >> (tile * 144 + k * 16)) & 0xFFFF
        window.append(bf16_to_float(bf16))
    return window


def get_ref_window_floats(ifm_34x34, tile):
    """9 BF16-quantized floats for tile t from the Python reference."""
    return [bf16_to_float(float_to_bf16(v)) for v in ref_window(ifm_34x34, tile)]


# ── Test 1: corners ───────────────────────────────────────────────────────────

@cocotb.test()
async def test_load_and_corners(dut):
    """Load one tile; verify load_done fires and windows at 5 spot tiles are correct."""
    cocotb.start_soon(Clock(dut.clk, 2, unit="ns").start())
    await reset(dut)

    ifm = make_ifm(seed=1)
    load_ok = await send_ifm_tile(dut, ifm)
    assert load_ok, "load_done did not assert after beat 36"

    await RisingEdge(dut.clk)
    wp = int(dut.windows_packed.value)

    SPOT_TILES = {
        "top-left  (t=0)":    0,
        "top-right (t=31)":   31,
        "bot-left  (t=992)":  992,
        "bot-right (t=1023)": 1023,
        "center    (t=527)":  527,
    }

    for label, t in SPOT_TILES.items():
        got = get_dut_window(wp, t)
        exp = get_ref_window_floats(ifm, t)
        assert got == exp, f"FAIL {label}: got={got[:3]} expected={exp[:3]}"
        dut._log.info(f"PASS {label}  window[0:3]={got[:3]}")


# ── Test 2: all 1024 windows ──────────────────────────────────────────────────

@cocotb.test()
async def test_all_windows(dut):
    """Load one tile; verify all 1024 windows are bit-exact."""
    cocotb.start_soon(Clock(dut.clk, 2, unit="ns").start())
    await reset(dut)

    ifm = make_ifm(seed=7)
    load_ok = await send_ifm_tile(dut, ifm)
    assert load_ok, "load_done did not assert"

    await RisingEdge(dut.clk)
    wp = int(dut.windows_packed.value)

    mismatches = 0
    for t in range(1024):
        got = get_dut_window(wp, t)
        exp = get_ref_window_floats(ifm, t)
        if got != exp:
            mismatches += 1
            if mismatches <= 5:
                dut._log.info(f"FAIL tile {t}: got={got[:3]} exp={exp[:3]}")

    assert mismatches == 0, f"{mismatches} window mismatches out of 1024"
    dut._log.info("PASS: all 1024 windows match reference")


# ── Test 3: reload overwrites buffer ─────────────────────────────────────────

@cocotb.test()
async def test_reload(dut):
    """Load two tiles back-to-back; confirm second overwrites first."""
    cocotb.start_soon(Clock(dut.clk, 2, unit="ns").start())
    await reset(dut)

    ifm_a = make_ifm(seed=3)
    ifm_b = make_ifm(seed=99)

    await send_ifm_tile(dut, ifm_a)
    await RisingEdge(dut.clk)

    load_ok = await send_ifm_tile(dut, ifm_b)
    assert load_ok, "load_done did not assert on second tile"
    await RisingEdge(dut.clk)

    wp = int(dut.windows_packed.value)
    got   = get_dut_window(wp, 0)
    exp_b = get_ref_window_floats(ifm_b, 0)
    exp_a = get_ref_window_floats(ifm_a, 0)

    assert got == exp_b, (
        f"FAIL reload: got={got} expected ifm_b={exp_b} (ifm_a={exp_a})"
    )
    dut._log.info(f"PASS reload: window[0] matches second tile  got={got[:3]}")
