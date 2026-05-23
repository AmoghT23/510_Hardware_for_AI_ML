"""
test_top.py
ECE 410/510 — M3 end-to-end co-simulation testbench
Top-level: conv_interface + compute_core (rtl/top.sv)

All stimulus is driven and all results are read exclusively through the
AXI4-Lite and AXI4-Stream ports exposed by top.sv.  No sub-module signals
are accessed directly.  This matches the protocol a real host would use.

Host transaction sequence for one tile:
  1. Send one 512-bit AXI4-Stream beat carrying 9 INT8 weight+IFM pairs.
       tdata[16i+7  : 16i]   = weight[i]  (i = 0..8)
       tdata[16i+15 : 16i+8] = ifm[i]     (i = 0..8)
  2. Write CTRL register (0x00) with bit[0]=1 via AXI4-Lite -> triggers
       compute_core start; sequencer drives 9 pairs over 9 cycles.
  3. Poll STATUS register (0x04) until bit[0]=1 (done).
  4. Read RESULT register (0x0C) -> INT32 dot product.
  5. Compare against Python reference model; assert PASS/FAIL.

Test vectors (all use the dominant 3x3x1 Conv2d kernel from M1 profiling):
  T1  ramp weights x unit activations          -> 45
  T2  alternating-sign weights x constant 2    -> 2
  T3  max positive weights x unit activations  -> 1143
  T4  Laplacian kernel x ramp activations      -> 0
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


# ── helpers ──────────────────────────────────────────────────────────────────

def to_uint8(v):
    """Convert a Python int to an unsigned 8-bit value (handles negatives)."""
    return v & 0xFF


def build_stream_beat(weights, ifms):
    """Pack 9 (weight, ifm) INT8 pairs into one 512-bit AXI4-Stream beat.
    tdata[16i+7:16i] = weight[i], tdata[16i+15:16i+8] = ifm[i]
    """
    tdata = 0
    for i, (w, f) in enumerate(zip(weights, ifms)):
        tdata |= (to_uint8(w) << (i * 16))
        tdata |= (to_uint8(f) << (i * 16 + 8))
    return tdata


def ref_dot(weights, ifms):
    """Python reference model: INT8 x INT8 -> INT32 dot product.
    Sign-extends each 8-bit value before multiplying, matching compute_core.sv.
    Returns an unsigned 32-bit value for direct comparison with DUT output.
    """
    acc = 0
    for w, f in zip(weights, ifms):
        w_s = w if w <= 127 else w - 256
        f_s = f if f <= 127 else f - 256
        acc += w_s * f_s
    return acc & 0xFFFFFFFF


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


async def run_tile(dut, weights, ifms):
    """Send one tile and return the INT32 result read back through the interface."""
    # 1. Send AXI4-Stream beat with packed weight+IFM pairs
    await RisingEdge(dut.clk)
    dut.s_axis_tdata.value  = build_stream_beat(weights, ifms)
    dut.s_axis_tvalid.value = 1
    dut.s_axis_tlast.value  = 1
    await RisingEdge(dut.clk)
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tlast.value  = 0

    # 2. Write CTRL[0]=1 to start the compute_core
    await axil_write(dut, 0x0, 0x1)

    # 3. Poll STATUS until done bit (bit[0]) is set
    #    compute_core takes 11 cycles (9 LOAD + 1 COMPUTE + 1 DONE_ST).
    #    Allow up to 50 cycles to account for interface latency.
    for _ in range(50):
        status = await axil_read(dut, 0x4)
        if status & 0x1:
            break
        await RisingEdge(dut.clk)

    # 4. Read RESULT register (done bit cleared by STATUS read above)
    result = await axil_read(dut, 0xC)
    return result


# ── test vectors ─────────────────────────────────────────────────────────────

VECTORS = [
    {
        "name":    "T1: ramp weights x unit activations",
        "weights": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "ifms":    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    },
    {
        "name":    "T2: alternating-sign weights x constant 2",
        "weights": [1, -1, 2, -2, 3, -3, 4, -4, 1],
        "ifms":    [2,  2, 2,  2, 2,  2, 2,  2, 2],
    },
    {
        "name":    "T3: max positive INT8 weights x unit activations",
        "weights": [127, 127, 127, 127, 127, 127, 127, 127, 127],
        "ifms":    [1,   1,   1,   1,   1,   1,   1,   1,   1  ],
    },
    {
        "name":    "T4: Laplacian kernel x ramp activations (ResNet18 edge-like)",
        "weights": [-1, -1, -1, -1,  8, -1, -1, -1, -1],
        "ifms":    [10, 20, 30, 40, 50, 60, 70, 80, 90],
    },
]


# ── main test ────────────────────────────────────────────────────────────────

@cocotb.test()
async def test_end_to_end_cosim(dut):
    """
    End-to-end co-simulation: host -> AXI4-Stream/AXI4-Lite -> conv_interface
    -> compute_core -> AXI4-Lite -> host.  No direct sub-module port access.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    passed = 0
    failed = 0

    dut._log.info("=" * 60)
    dut._log.info("M3 End-to-End Co-Simulation: INT8 Conv2d Compute Core")
    dut._log.info("Interface: AXI4-Stream (data) + AXI4-Lite (control)")
    dut._log.info("Kernel: 3x3x1 (9 MACs, INT8 QAT, INT32 output)")
    dut._log.info("=" * 60)

    for vec in VECTORS:
        expected = ref_dot(vec["weights"], vec["ifms"])
        result   = await run_tile(dut, vec["weights"], vec["ifms"])

        # Display signed for readability; compare as unsigned 32-bit
        exp_s = expected if expected < 0x80000000 else expected - 0x100000000
        res_s = result   if result   < 0x80000000 else result   - 0x100000000

        if result == expected:
            passed += 1
            dut._log.info(f"  PASS  {vec['name']}")
            dut._log.info(f"        got={res_s}  expected={exp_s}")
        else:
            failed += 1
            dut._log.info(f"  FAIL  {vec['name']}")
            dut._log.info(f"        got={res_s}  expected={exp_s}")

    dut._log.info("-" * 60)
    if failed == 0:
        dut._log.info(f"PASS -- all {passed} tests passed.")
    else:
        dut._log.info(f"FAIL -- {failed} of {passed + failed} tests failed.")
    dut._log.info("=" * 60)

    assert failed == 0, f"{failed} test(s) failed in end-to-end co-simulation"
