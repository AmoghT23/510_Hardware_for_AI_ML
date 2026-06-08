"""
test_forward.py
ECE 410/510 — 32×32 BF16 Conv2d forward pass testbench
Top-level: conv_interface_ti + pe_array (rtl/top.sv)

Protocol (32×32 array):
  Weight load  (once per kernel):
    1. Write CTRL[1]=1  → sets load_weights flag
    2. Stream 1 weight beat: tdata[16i+15:16i] = weight[i] (BF16, i=0..8)
    3. Poll STATUS[2] until wt_loaded=1

  IFM tile load + compute:
    4. Stream 37 beats: 34×34 BF16 IFM tile, flat row-major, 32 values/beat
    5. Poll STATUS[4] until ifm_loaded=1
    6. Write CTRL[0]=1  → starts all 1024 compute cores simultaneously
    7. Poll STATUS[0] until done=1
    8. Read 1024 FP32 results from 0x1000 + t*4 (t=0..1023)

Tests:
  test_bf16_forward  — T1..T4: each loads weights + 34×34 IFM, checks all 1024 results
  test_weight_reuse  — load weights ONCE, run 4 different 34×34 IFM tiles
"""

import struct
import sys
import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

sys.path.insert(0, os.path.dirname(__file__))
from ref_model import ref_conv2d_32x32, build_ifm_tile_beats


# ── BF16 / FP32 helpers ───────────────────────────────────────────────────────

def float_to_bf16(f):
    b = struct.pack('>f', float(f))
    return (int.from_bytes(b, 'big') >> 16) & 0xFFFF


def bf16_to_float(bf16):
    fp32_bits_val = (bf16 & 0xFFFF) << 16
    return struct.unpack('>f', fp32_bits_val.to_bytes(4, 'big'))[0]


def fp32_bits(f):
    return int.from_bytes(struct.pack('>f', float(f)), 'big')


def build_bf16_beat(values):
    """Pack values into a 512-bit beat: element i at bits [i*16+15:i*16]."""
    tdata = 0
    for i, v in enumerate(values):
        tdata |= float_to_bf16(v) << (i * 16)
    return tdata


# ── IFM 37-beat packer ────────────────────────────────────────────────────────

def build_ifm_tile_34x34(ifm_34x34):
    """Pack a 34×34 IFM tile into 37 × 512-bit integers (delegates to ref_model)."""
    return build_ifm_tile_beats(ifm_34x34)


# ── AXI helpers ───────────────────────────────────────────────────────────────

async def reset(dut):
    dut.rst_n.value           = 0
    dut.s_axil_awvalid.value  = 0
    dut.s_axil_awaddr.value   = 0
    dut.s_axil_wvalid.value   = 0
    dut.s_axil_wdata.value    = 0
    dut.s_axil_wstrb.value    = 0
    dut.s_axil_bready.value   = 0
    dut.s_axil_arvalid.value  = 0
    dut.s_axil_araddr.value   = 0
    dut.s_axil_rready.value   = 0
    dut.s_axis_tdata.value    = 0
    dut.s_axis_tvalid.value   = 0
    dut.s_axis_tlast.value    = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def axil_write(dut, addr, data):
    await RisingEdge(dut.clk)
    dut.s_axil_awaddr.value  = addr
    dut.s_axil_awvalid.value = 1
    dut.s_axil_wdata.value   = data
    dut.s_axil_wstrb.value   = 0xF
    dut.s_axil_wvalid.value  = 1
    for _ in range(20):
        if dut.s_axil_awready.value and dut.s_axil_wready.value:
            break
        await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.s_axil_awvalid.value = 0
    dut.s_axil_wvalid.value  = 0
    for _ in range(20):
        if dut.s_axil_bvalid.value:
            break
        await RisingEdge(dut.clk)
    dut.s_axil_bready.value = 1
    await RisingEdge(dut.clk)
    dut.s_axil_bready.value = 0


async def axil_read(dut, addr):
    await RisingEdge(dut.clk)
    dut.s_axil_araddr.value  = addr
    dut.s_axil_arvalid.value = 1
    for _ in range(20):
        if dut.s_axil_arready.value:
            break
        await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.s_axil_arvalid.value = 0
    for _ in range(20):
        if dut.s_axil_rvalid.value:
            break
        await RisingEdge(dut.clk)
    data = int(dut.s_axil_rdata.value)
    dut.s_axil_rready.value = 1
    await RisingEdge(dut.clk)
    dut.s_axil_rready.value = 0
    return data


async def send_stream_beat(dut, values):
    """Send one 512-bit beat (used for weight load: 9 BF16 values)."""
    await RisingEdge(dut.clk)
    dut.s_axis_tdata.value  = build_bf16_beat(values)
    dut.s_axis_tvalid.value = 1
    dut.s_axis_tlast.value  = 1
    await RisingEdge(dut.clk)
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tlast.value  = 0


async def send_ifm_tile(dut, ifm_34x34):
    """Send 37 stream beats for a 34×34 IFM tile, back-to-back with 1-cycle gaps."""
    beats = build_ifm_tile_34x34(ifm_34x34)
    for idx, beat_val in enumerate(beats):
        await RisingEdge(dut.clk)
        dut.s_axis_tdata.value  = beat_val
        dut.s_axis_tvalid.value = 1
        dut.s_axis_tlast.value  = 1 if idx == 36 else 0
        await RisingEdge(dut.clk)
        dut.s_axis_tvalid.value = 0
        dut.s_axis_tlast.value  = 0


async def load_weights(dut, weights):
    """Load 9 BF16 weights: CTRL[1]=1, stream one beat, poll STATUS[2]."""
    await axil_write(dut, 0x0, 0x2)
    await send_stream_beat(dut, weights)
    for _ in range(30):
        status = await axil_read(dut, 0x4)
        if status & 0x4:                    # STATUS[2] = wt_loaded
            break
        await RisingEdge(dut.clk)


async def run_ifm_tile(dut, ifm_34x34):
    """Stream 37-beat IFM tile, poll ifm_loaded, start 1024 cores,
    poll done, return list of 1024 FP32 bit patterns.
    """
    await send_ifm_tile(dut, ifm_34x34)
    for _ in range(100):
        status = await axil_read(dut, 0x4)
        if status & 0x10:                   # STATUS[4] = ifm_load_done
            break
        await RisingEdge(dut.clk)
    await axil_write(dut, 0x0, 0x1)        # CTRL[0] = start
    for _ in range(50):
        status = await axil_read(dut, 0x4)
        if status & 0x1:                    # STATUS[0] = done
            break
        await RisingEdge(dut.clk)
    results = []
    for t in range(1024):
        results.append(await axil_read(dut, 0x1000 + t * 4))
    return results


# ── test vectors (34×34 IFM tiles) ───────────────────────────────────────────

def _uniform_ifm(val):
    return [[val] * 34 for _ in range(34)]


VECTORS = [
    {
        "name":    "T1: ramp weights × unit IFM (all 1024 tiles = 45.0)",
        "weights": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ifm":     _uniform_ifm(1.0),
    },
    {
        "name":    "T2: alternating-sign weights × constant-2.0 IFM",
        "weights": [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 1.0],
        "ifm":     _uniform_ifm(2.0),
    },
    {
        "name":    "T3: max BF16-safe weights × unit IFM (all tiles = 1143.0)",
        "weights": [127.0] * 9,
        "ifm":     _uniform_ifm(1.0),
    },
    {
        "name":    "T4: Laplacian kernel × ramp IFM (per-tile result varies)",
        "weights": [-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0],
        "ifm":     [[float(r + c) for c in range(34)] for r in range(34)],
    },
]


# ── test 1: BF16 forward pass (T1..T4) ───────────────────────────────────────

@cocotb.test()
async def test_bf16_forward(dut):
    """32×32 forward pass: load weights + 37-beat 34×34 IFM, verify all 1024 results."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    passed = 0
    failed = 0

    dut._log.info("=" * 60)
    dut._log.info("32×32 BF16 Forward Pass")
    dut._log.info("IFM: 37-beat 34×34 stream  |  Results: 1024×FP32 @ 0x1000-0x1FFC")
    dut._log.info("=" * 60)

    for vec in VECTORS:
        expected = ref_conv2d_32x32(vec["weights"], vec["ifm"])

        await load_weights(dut, vec["weights"])
        results = await run_ifm_tile(dut, vec["ifm"])

        tile_errors = 0
        first_err   = None
        for t in range(1024):
            if results[t] != expected[t]:
                tile_errors += 1
                if first_err is None:
                    first_err = t

        if tile_errors == 0:
            passed += 1
            dut._log.info(f"  PASS  {vec['name']}")
        else:
            failed += 1
            t = first_err
            dut._log.info(f"  FAIL  {vec['name']}  ({tile_errors}/1024 tiles wrong)")
            dut._log.info(
                f"        first mismatch tile {t}: "
                f"got=0x{results[t]:08X}  expected=0x{expected[t]:08X}"
            )

    dut._log.info("-" * 60)
    if failed == 0:
        dut._log.info(f"PASS -- all {passed} tests passed.")
    else:
        dut._log.info(f"FAIL -- {failed} of {passed + failed} tests failed.")
    dut._log.info("=" * 60)

    assert failed == 0, f"{failed} forward test(s) failed"


# ── test 2: weight reuse (load weights once, 4 IFM tiles) ────────────────────

@cocotb.test()
async def test_weight_reuse(dut):
    """32×32 weight reuse: load weights=[1..9] once, verify 4 different 34×34 IFM tiles."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    WEIGHTS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    IFM_TILES = [
        (_uniform_ifm(1.0),  "unit activations   (exp all tiles = 45.0)"),
        (_uniform_ifm(2.0),  "double activations (exp all tiles = 90.0)"),
        (_uniform_ifm(-1.0), "negative unit      (exp all tiles = -45.0)"),
        (_uniform_ifm(0.5),  "half activations   (exp all tiles = 22.5)"),
    ]

    dut._log.info("=" * 60)
    dut._log.info("32×32 Weight Reuse: weights=[1..9] loaded once, 4 IFM tiles")
    dut._log.info("=" * 60)

    await load_weights(dut, WEIGHTS)
    dut._log.info("  Weight SRAM loaded.")

    passed = 0
    failed = 0

    for ifm, desc in IFM_TILES:
        expected = ref_conv2d_32x32(WEIGHTS, ifm)
        results  = await run_ifm_tile(dut, ifm)

        tile_errors = 0
        first_err   = None
        for t in range(1024):
            if results[t] != expected[t]:
                tile_errors += 1
                if first_err is None:
                    first_err = t

        if tile_errors == 0:
            passed += 1
            dut._log.info(f"  PASS  {desc}")
        else:
            failed += 1
            t = first_err
            dut._log.info(f"  FAIL  {desc}  ({tile_errors}/1024 tiles wrong)")
            dut._log.info(
                f"        first mismatch tile {t}: "
                f"got=0x{results[t]:08X}  expected=0x{expected[t]:08X}"
            )

    dut._log.info("-" * 60)
    if failed == 0:
        dut._log.info(f"PASS -- all {passed} weight-reuse tiles correct.")
    else:
        dut._log.info(f"FAIL -- {failed} of {passed + failed} tiles failed.")
    dut._log.info("=" * 60)

    assert failed == 0, f"{failed} weight-reuse test(s) failed"
