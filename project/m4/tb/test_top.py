"""
test_top.py
ECE 410/510 — 32×32 BF16 Conv2d end-to-end co-simulation testbench
Top-level: top_ti (conv_interface_ti + pe_array, rtl/top.sv)

All stimulus is driven and all results are read exclusively through the
AXI4-Lite (13-bit address) and AXI4-Stream ports exposed by top.sv.
No sub-module signals are accessed directly.

Host transaction sequence for one 32×32 conv tile:
  Weight load (once per kernel):
    1. Write CTRL (0x00) bit[1]=1      → sets load_weights flag
    2. Send 1 × 512-bit stream beat    → 9 BF16 weights in [143:0]
    3. Poll STATUS (0x04) bit[2]       → wait wt_loaded=1

  IFM tile + compute:
    4. Send 37 × 512-bit stream beats  → 34×34 BF16 IFM, flat row-major
    5. Poll STATUS (0x04) bit[4]       → wait ifm_loaded=1
    6. Write CTRL (0x00) bit[0]=1      → start all 1024 compute cores
    7. Poll STATUS (0x04) bit[0]       → wait done=1
    8. Read 1024 × FP32 results        → 0x1000 + t*4, t=0..1023

Test vectors (3×3×1 BF16 Conv2d, same kernel concepts as original M1/M3):
  T1  ramp weights × unit IFM          → all 1024 tiles = 45.0
  T2  alternating-sign weights × 2.0   → all 1024 tiles = 2.0
  T3  max BF16-safe weights × unit IFM → all 1024 tiles = 1143.0
  T4  Laplacian kernel × ramp IFM      → per-tile result varies
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


def build_bf16_beat(values):
    """Pack values into a 512-bit beat: element i at bits [i*16+15:i*16]."""
    tdata = 0
    for i, v in enumerate(values):
        tdata |= float_to_bf16(v) << (i * 16)
    return tdata


# ── AXI helpers (13-bit address: valid range 0x0000–0x1FFC) ──────────────────

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


# ── Stream helpers ────────────────────────────────────────────────────────────

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
    beats = build_ifm_tile_beats(ifm_34x34)
    for idx, beat_val in enumerate(beats):
        await RisingEdge(dut.clk)
        dut.s_axis_tdata.value  = beat_val
        dut.s_axis_tvalid.value = 1
        dut.s_axis_tlast.value  = 1 if idx == 36 else 0
        await RisingEdge(dut.clk)
        dut.s_axis_tvalid.value = 0
        dut.s_axis_tlast.value  = 0


# ── Tile protocol ─────────────────────────────────────────────────────────────

async def load_weights(dut, weights):
    """CTRL[1]=1, stream one BF16 weight beat, poll STATUS[2] (wt_loaded)."""
    await axil_write(dut, 0x00, 0x2)
    await send_stream_beat(dut, weights)
    for _ in range(30):
        status = await axil_read(dut, 0x04)
        if status & 0x4:                    # STATUS[2] = wt_loaded
            break
        await RisingEdge(dut.clk)


async def run_tile(dut, ifm_34x34):
    """Stream 37-beat IFM tile, poll ifm_loaded, start 1024 cores,
    poll done, return list of 1024 FP32 bit patterns from 0x1000–0x1FFC.
    """
    await send_ifm_tile(dut, ifm_34x34)
    for _ in range(100):
        status = await axil_read(dut, 0x04)
        if status & 0x10:                   # STATUS[4] = ifm_load_done
            break
        await RisingEdge(dut.clk)
    await axil_write(dut, 0x00, 0x1)       # CTRL[0] = start
    for _ in range(50):
        status = await axil_read(dut, 0x04)
        if status & 0x1:                    # STATUS[0] = done
            break
        await RisingEdge(dut.clk)
    results = []
    for t in range(1024):
        results.append(await axil_read(dut, 0x1000 + t * 4))
    return results


# ── test vectors ──────────────────────────────────────────────────────────────

def _uniform_ifm(val):
    return [[val] * 34 for _ in range(34)]


VECTORS = [
    {
        "name":    "T1: ramp weights × unit IFM (all 1024 tiles = 45.0)",
        "weights": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ifm":     _uniform_ifm(1.0),
    },
    {
        "name":    "T2: alternating-sign weights × constant-2.0 IFM (all tiles = 2.0)",
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


# ── main test ─────────────────────────────────────────────────────────────────

@cocotb.test()
async def test_end_to_end_cosim(dut):
    """
    End-to-end co-simulation: host → AXI4-Stream/AXI4-Lite (13-bit) → top_ti
    → pe_array (1024 tiles) → 1024 × FP32 results → host.
    No direct sub-module port access.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    passed = 0
    failed = 0

    dut._log.info("=" * 60)
    dut._log.info("32×32 BF16 Conv2d End-to-End Co-Simulation")
    dut._log.info("Interface: AXI4-Stream (37-beat IFM) + AXI4-Lite 13-bit")
    dut._log.info("Results:   1024 × FP32 @ 0x1000–0x1FFC")
    dut._log.info("=" * 60)

    for vec in VECTORS:
        expected = ref_conv2d_32x32(vec["weights"], vec["ifm"])

        await load_weights(dut, vec["weights"])
        results = await run_tile(dut, vec["ifm"])

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

    assert failed == 0, f"{failed} test(s) failed in end-to-end co-simulation"
