"""
test_backward.py
ECE 410/510 — M3-TI Phase 3: BF16 backward pass testbench
Top-level: top_ti (conv_interface_ti + compute_core_ti + grad_core)

Protocol:
  1. Load weights via the Phase 2 weight-load sequence.
  2. Load IFMs via stream beat.
  3. Write GRAD_IN (0x10) with FP32 dL/dy.
  4. Write CTRL[2]=1 (backward).
  5. Poll STATUS[3] until grad_done=1.
  6. Read GRAD_W0..GRAD_W8 (0x14..0x34).

Reference: dL/dW[i] = dL/dy × BF16(x[i])
           dL/dX[i] = dL/dy × BF16(w[i])
"""

import struct
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


# ── BF16 / FP32 helpers ───────────────────────────────────────────────────────

def float_to_bf16(f):
    b = struct.pack('>f', float(f))
    return (int.from_bytes(b, 'big') >> 16) & 0xFFFF


def bf16_to_float(bf16):
    fp32_bits = (bf16 & 0xFFFF) << 16
    return struct.unpack('>f', fp32_bits.to_bytes(4, 'big'))[0]


def fp32_bits(f):
    return int.from_bytes(struct.pack('>f', float(f)), 'big')


def build_bf16_beat(values):
    tdata = 0
    for i, v in enumerate(values):
        tdata |= float_to_bf16(v) << (i * 16)
    return tdata


def ref_grad_w(ifms, dldy):
    """Expected dL/dW[i] = dldy × BF16(ifms[i]), returned as FP32 bit patterns."""
    return [fp32_bits(dldy * bf16_to_float(float_to_bf16(x))) for x in ifms]


def ref_grad_x(weights, dldy):
    """Expected dL/dX[i] = dldy × BF16(weights[i]), returned as FP32 bit patterns."""
    return [fp32_bits(dldy * bf16_to_float(float_to_bf16(w))) for w in weights]


# ── AXI helpers (duplicated from test_forward.py for self-contained file) ────

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
    await RisingEdge(dut.clk)
    dut.s_axis_tdata.value  = build_bf16_beat(values)
    dut.s_axis_tvalid.value = 1
    dut.s_axis_tlast.value  = 1
    await RisingEdge(dut.clk)
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tlast.value  = 0


async def load_weights(dut, weights):
    await axil_write(dut, 0x00, 0x2)        # CTRL[1] = load_weights
    await send_stream_beat(dut, weights)
    for _ in range(30):
        status = await axil_read(dut, 0x04)
        if status & 0x4:                    # STATUS[2] = wt_loaded
            break
        await RisingEdge(dut.clk)


async def run_backward(dut, ifms, dldy):
    """Load IFM beat, write GRAD_IN, trigger backward, poll grad_done, return grad_w list."""
    await send_stream_beat(dut, ifms)
    await axil_write(dut, 0x10, fp32_bits(dldy))   # GRAD_IN
    await axil_write(dut, 0x00, 0x4)               # CTRL[2] = backward
    for _ in range(30):
        status = await axil_read(dut, 0x04)
        if status & 0x8:                            # STATUS[3] = grad_done
            break
        await RisingEdge(dut.clk)
    grad_addrs = [0x14 + i * 4 for i in range(9)]
    return [await axil_read(dut, a) for a in grad_addrs]


# ── test vectors ─────────────────────────────────────────────────────────────

BVECTORS = [
    {
        "name":    "B1: unit IFMs, dldy=1.0 → grad_w = ifms (all 1.0)",
        "weights": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ifms":    [1.0] * 9,
        "dldy":    1.0,
    },
    {
        "name":    "B2: ramp IFMs, dldy=2.0 → grad_w = 2*ifms",
        "weights": [1.0] * 9,
        "ifms":    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "dldy":    2.0,
    },
    {
        "name":    "B3: alternating IFMs, dldy=0.5",
        "weights": [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 1.0],
        "ifms":    [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        "dldy":    0.5,
    },
    {
        "name":    "B4: Laplacian weights, ramp IFMs, dldy=-1.0",
        "weights": [-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0],
        "ifms":    [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        "dldy":    -1.0,
    },
]


# ── test: backward pass (B1..B4) ─────────────────────────────────────────────

@cocotb.test()
async def test_grad_w(dut):
    """Phase 3 backward pass: verify dL/dW[i] = dL/dy × BF16(x[i]) for B1..B4."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    passed = 0
    failed = 0

    dut._log.info("=" * 60)
    dut._log.info("M3-TI Phase 3: Backward Pass — dL/dW verification")
    dut._log.info("=" * 60)

    for vec in BVECTORS:
        expected = ref_grad_w(vec["ifms"], vec["dldy"])

        await load_weights(dut, vec["weights"])
        got = await run_backward(dut, vec["ifms"], vec["dldy"])

        vec_pass = True
        for i in range(9):
            if got[i] != expected[i]:
                vec_pass = False
                break

        if vec_pass:
            passed += 1
            dut._log.info(f"  PASS  {vec['name']}")
            # Print first three values for sanity
            exp_fs = [struct.unpack('>f', e.to_bytes(4, 'big'))[0] for e in expected[:3]]
            got_fs = [struct.unpack('>f', g.to_bytes(4, 'big'))[0] for g in got[:3]]
            dut._log.info(f"        grad_w[0:3] got={got_fs}  expected={exp_fs}")
        else:
            failed += 1
            dut._log.info(f"  FAIL  {vec['name']}")
            for i in range(9):
                exp_f = struct.unpack('>f', expected[i].to_bytes(4, 'big'))[0]
                got_f = struct.unpack('>f', got[i].to_bytes(4, 'big'))[0]
                if got[i] != expected[i]:
                    dut._log.info(
                        f"        grad_w[{i}]: got={got_f} (0x{got[i]:08X})  "
                        f"expected={exp_f} (0x{expected[i]:08X})"
                    )

    dut._log.info("-" * 60)
    if failed == 0:
        dut._log.info(f"PASS -- all {passed} backward tests passed.")
    else:
        dut._log.info(f"FAIL -- {failed} of {passed + failed} backward tests failed.")
    dut._log.info("=" * 60)

    assert failed == 0, f"{failed} backward test(s) failed"
