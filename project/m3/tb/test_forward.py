"""
test_forward.py
ECE 410/510 — M3-TI Phase 2: BF16 forward pass + weight reuse testbench
Top-level: conv_interface_ti + compute_core_ti (rtl/top_ti.sv)

Phase 2 protocol (two separate stream beats per compute):
  Weight load  (once):
    1. Write CTRL[1]=1  → sets load_weights flag
    2. Stream weight beat: tdata[16i+15:16i] = weight[i] (BF16, i=0..8)
    3. Poll STATUS[2] until wt_loaded=1
  Per IFM tile  (repeatable with same weights):
    4. Stream IFM beat: tdata[16i+15:16i] = ifm[i] (BF16, i=0..8)
    5. Write CTRL[0]=1  → starts sequencer + compute core
    6. Poll STATUS[0] until done=1
    7. Read RESULT (0x0C) → FP32

Tests:
  test_bf16_forward   — T1..T4 from M3, weight+IFM loaded per tile
  test_weight_reuse   — load weights ONCE, run 4 IFM tiles to prove persistence
"""

import struct
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


# ── BF16 helpers ─────────────────────────────────────────────────────────────

def float_to_bf16(f):
    """Python float → BF16 bit pattern (16-bit int). Truncates mantissa."""
    b = struct.pack('>f', float(f))
    return (int.from_bytes(b, 'big') >> 16) & 0xFFFF


def bf16_to_float(bf16):
    """BF16 bit pattern → Python float."""
    fp32_bits = (bf16 & 0xFFFF) << 16
    return struct.unpack('>f', fp32_bits.to_bytes(4, 'big'))[0]


def fp32_bits(f):
    """Python float → FP32 bit pattern (32-bit int)."""
    return int.from_bytes(struct.pack('>f', float(f)), 'big')


def build_bf16_beat(values):
    """Pack 9 BF16 values into a 512-bit stream beat.
    tdata[16i+15:16i] = values[i]  (i = 0..8)
    """
    tdata = 0
    for i, v in enumerate(values):
        tdata |= float_to_bf16(v) << (i * 16)
    return tdata


def ref_dot_bf16(weights, ifms):
    """Reference: quantize to BF16, accumulate in float64, return FP32 bits."""
    acc = 0.0
    for w, f in zip(weights, ifms):
        acc += bf16_to_float(float_to_bf16(w)) * bf16_to_float(float_to_bf16(f))
    return fp32_bits(acc)


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
    """Send one 512-bit stream beat carrying 9 BF16 values."""
    await RisingEdge(dut.clk)
    dut.s_axis_tdata.value  = build_bf16_beat(values)
    dut.s_axis_tvalid.value = 1
    dut.s_axis_tlast.value  = 1
    await RisingEdge(dut.clk)
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tlast.value  = 0


async def load_weights(dut, weights):
    """Phase 2: load weight tile into persistent SRAM.
    Sets CTRL[1]=1, streams weight beat, polls STATUS[2] until wt_loaded.
    """
    await axil_write(dut, 0x0, 0x2)        # CTRL[1] = load_weights
    await send_stream_beat(dut, weights)
    for _ in range(30):                     # wait for 9-cycle SRAM write
        status = await axil_read(dut, 0x4)
        if status & 0x4:                    # STATUS[2] = wt_loaded
            break
        await RisingEdge(dut.clk)


async def run_ifm_tile(dut, ifms):
    """Phase 2: stream one IFM tile, start compute, poll done, return result."""
    await send_stream_beat(dut, ifms)
    await axil_write(dut, 0x0, 0x1)        # CTRL[0] = start
    for _ in range(50):
        status = await axil_read(dut, 0x4)
        if status & 0x1:                    # STATUS[0] = done
            break
        await RisingEdge(dut.clk)
    return await axil_read(dut, 0xC)       # RESULT register


# ── test vectors ─────────────────────────────────────────────────────────────

VECTORS = [
    {
        "name":    "T1: ramp weights × unit activations",
        "weights": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ifms":    [1.0] * 9,
    },
    {
        "name":    "T2: alternating-sign weights × constant 2.0",
        "weights": [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 1.0],
        "ifms":    [2.0] * 9,
    },
    {
        "name":    "T3: max BF16-safe weights × unit activations",
        "weights": [127.0] * 9,
        "ifms":    [1.0] * 9,
    },
    {
        "name":    "T4: Laplacian kernel × ramp activations",
        "weights": [-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0],
        "ifms":    [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
    },
]


# ── test 1: BF16 forward pass (T1..T4, one weight load per tile) ─────────────

@cocotb.test()
async def test_bf16_forward(dut):
    """Phase 2 forward pass: 2-phase protocol (weight load + IFM tile per vector)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    passed = 0
    failed = 0

    dut._log.info("=" * 60)
    dut._log.info("M3-TI Phase 2: BF16 Forward Pass (2-phase protocol)")
    dut._log.info("Weight load: CTRL[1] + stream beat → weight SRAM")
    dut._log.info("IFM tile  : stream beat + CTRL[0] → compute")
    dut._log.info("=" * 60)

    for vec in VECTORS:
        expected_bits = ref_dot_bf16(vec["weights"], vec["ifms"])

        await load_weights(dut, vec["weights"])
        result_bits = await run_ifm_tile(dut, vec["ifms"])

        exp_f = struct.unpack('>f', expected_bits.to_bytes(4, 'big'))[0]
        res_f = struct.unpack('>f', result_bits.to_bytes(4, 'big'))[0]

        if result_bits == expected_bits:
            passed += 1
            dut._log.info(f"  PASS  {vec['name']}")
            dut._log.info(f"        got={res_f}  expected={exp_f}")
        else:
            failed += 1
            dut._log.info(f"  FAIL  {vec['name']}")
            dut._log.info(
                f"        got={res_f} (0x{result_bits:08X})  "
                f"expected={exp_f} (0x{expected_bits:08X})"
            )

    dut._log.info("-" * 60)
    if failed == 0:
        dut._log.info(f"PASS -- all {passed} tests passed.")
    else:
        dut._log.info(f"FAIL -- {failed} of {passed + failed} tests failed.")
    dut._log.info("=" * 60)

    assert failed == 0, f"{failed} BF16 forward test(s) failed"


# ── test 2: weight reuse (load once, run 4 IFM tiles) ────────────────────────

@cocotb.test()
async def test_weight_reuse(dut):
    """
    Phase 2 weight reuse: load weights=[1..9] ONCE, then compute 4 different
    IFM tiles without re-loading weights.  Demonstrates that weight SRAM
    persists across tiles, improving arithmetic intensity from ~1 to ~46.9
    FLOP/byte (per the roofline analysis).
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    WEIGHTS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    IFM_TILES = [
        ([1.0]  * 9, "unit activations  → 45.0"),
        ([2.0]  * 9, "double activations → 90.0"),
        ([-1.0] * 9, "negative unit      → -45.0"),
        ([0.5]  * 9, "half activations   → 22.5"),
    ]

    dut._log.info("=" * 60)
    dut._log.info("M3-TI Phase 2: Weight Reuse")
    dut._log.info(f"  weights = {WEIGHTS}")
    dut._log.info("  Loading weights ONCE, running 4 IFM tiles")
    dut._log.info("=" * 60)

    # Load weights exactly once
    await load_weights(dut, WEIGHTS)
    dut._log.info("  Weight SRAM loaded.")

    passed = 0
    failed = 0

    for ifms, desc in IFM_TILES:
        expected_bits = ref_dot_bf16(WEIGHTS, ifms)
        result_bits   = await run_ifm_tile(dut, ifms)

        exp_f = struct.unpack('>f', expected_bits.to_bytes(4, 'big'))[0]
        res_f = struct.unpack('>f', result_bits.to_bytes(4, 'big'))[0]

        if result_bits == expected_bits:
            passed += 1
            dut._log.info(f"  PASS  {desc}")
            dut._log.info(f"        got={res_f}  expected={exp_f}")
        else:
            failed += 1
            dut._log.info(f"  FAIL  {desc}")
            dut._log.info(
                f"        got={res_f} (0x{result_bits:08X})  "
                f"expected={exp_f} (0x{expected_bits:08X})"
            )

    dut._log.info("-" * 60)
    if failed == 0:
        dut._log.info(f"PASS -- all {passed} weight-reuse tiles correct.")
    else:
        dut._log.info(f"FAIL -- {failed} of {passed + failed} tiles failed.")
    dut._log.info("=" * 60)

    assert failed == 0, f"{failed} weight-reuse test(s) failed"
