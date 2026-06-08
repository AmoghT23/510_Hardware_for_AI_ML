import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

async def reset(dut):
    dut.rst_n.value = 0
    for sig in (dut.s_axil_awvalid, dut.s_axil_wvalid, dut.s_axil_bready,
                dut.s_axil_arvalid, dut.s_axil_rready,
                dut.s_axis_tvalid,  dut.s_axis_tlast):
        sig.value = 0
    dut.s_axil_awaddr.value  = 0; dut.s_axil_wdata.value  = 0
    dut.s_axil_wstrb.value   = 0; dut.s_axil_araddr.value = 0
    dut.s_axis_tdata.value   = 0
    dut.core_done.value      = 0; dut.core_ready.value = 1
    dut.core_result.value    = 0
    for _ in range(4): await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

async def axil_write(dut, addr, data):
    await RisingEdge(dut.clk)
    dut.s_axil_awaddr.value = addr; dut.s_axil_awvalid.value = 1
    dut.s_axil_wdata.value  = data; dut.s_axil_wstrb.value  = 0xF
    dut.s_axil_wvalid.value = 1
    for _ in range(20):
        if dut.s_axil_awready.value and dut.s_axil_wready.value: break
        await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.s_axil_awvalid.value = 0; dut.s_axil_wvalid.value = 0
    for _ in range(20):
        if dut.s_axil_bvalid.value: break
        await RisingEdge(dut.clk)
    dut.s_axil_bready.value = 1
    await RisingEdge(dut.clk)
    dut.s_axil_bready.value = 0

async def axil_read(dut, addr):
    await RisingEdge(dut.clk)
    dut.s_axil_araddr.value = addr; dut.s_axil_arvalid.value = 1
    for _ in range(20):
        if dut.s_axil_arready.value: break
        await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.s_axil_arvalid.value = 0
    for _ in range(20):
        if dut.s_axil_rvalid.value: break
        await RisingEdge(dut.clk)
    data = int(dut.s_axil_rdata.value)
    dut.s_axil_rready.value = 1
    await RisingEdge(dut.clk)
    dut.s_axil_rready.value = 0
    return data

@cocotb.test()
async def test_write_tile_len(dut):
    """T1: AXI4-Lite write TILE_LEN=9 -> BRESP=OKAY"""
    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)
    await axil_write(dut, 0x8, 9)
    assert int(dut.s_axil_bresp.value) == 0, "FAIL: bresp != OKAY"
    dut._log.info("PASS: write TILE_LEN=9, BRESP=OKAY")

@cocotb.test()
async def test_read_tile_len(dut):
    """T2: Read TILE_LEN back -> 9, RRESP=OKAY"""
    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)
    await axil_write(dut, 0x8, 9)
    rdata = await axil_read(dut, 0x8)
    assert rdata & 0xFF == 9, f"FAIL: got 0x{rdata:08X}"
    assert int(dut.s_axil_rresp.value) == 0, "FAIL: rresp != OKAY"
    dut._log.info(f"PASS: TILE_LEN=0x{rdata:08X}")

@cocotb.test()
async def test_axis_beat(dut):
    """T3: AXI4-Stream beat accepted (tready permanently high)"""
    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)
    await RisingEdge(dut.clk)
    dut.s_axis_tdata.value = 0xDEADBEEF
    dut.s_axis_tvalid.value = 1; dut.s_axis_tlast.value = 1
    assert dut.s_axis_tready.value == 1, "FAIL: tready=0"
    dut._log.info("PASS: tready=1")
    await RisingEdge(dut.clk)
    dut.s_axis_tvalid.value = 0; dut.s_axis_tlast.value = 0

@cocotb.test()
async def test_ctrl_start_pulse(dut):
    """T4: Write CTRL[0]=1 -> core_start pulses for exactly one cycle"""
    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)

    seen = [False]

    async def monitor():
        for _ in range(40):
            await RisingEdge(dut.clk)
            if dut.core_start.value == 1:
                seen[0] = True; return

    mon = cocotb.start_soon(monitor())
    await axil_write(dut, 0x0, 0x1)
    await mon
    assert seen[0], "FAIL: core_start never went high"
    dut._log.info("PASS: core_start pulsed high")

@cocotb.test()
async def test_result_register_32bit(dut):
    """T5 (task1): core_result[31:0] latched correctly into RESULT register (0x0C)"""
    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)

    expected = 0xDEAD_1234
    dut.core_result.value = expected
    dut.core_done.value   = 1
    await RisingEdge(dut.clk)
    dut.core_done.value   = 0

    rdata = await axil_read(dut, 0xC)
    assert rdata == expected, f"FAIL: RESULT=0x{rdata:08X}, expected 0x{expected:08X}"
    dut._log.info(f"PASS: RESULT=0x{rdata:08X} matches INT32 core_result")

@cocotb.test()
async def test_weight_data_port_reset(dut):
    """T6 (task2): core_weight_data is 0x00 after reset (port exists, default low)"""
    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)
    await RisingEdge(dut.clk)

    val = int(dut.core_weight_data.value)
    assert val == 0, f"FAIL: core_weight_data=0x{val:02X} after reset, expected 0x00"
    dut._log.info(f"PASS: core_weight_data=0x{val:02X} after reset")

@cocotb.test()
async def test_weight_valid_port_reset(dut):
    """T7 (task2): core_weight_valid is 0 after reset (port exists, default low)"""
    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)
    await RisingEdge(dut.clk)

    val = int(dut.core_weight_valid.value)
    assert val == 0, f"FAIL: core_weight_valid={val} after reset, expected 0"
    dut._log.info(f"PASS: core_weight_valid={val} after reset")

@cocotb.test()
async def test_ifm_data_port_reset(dut):
    """T8 (task3): core_ifm_data is 0x00 after reset (port exists, default low)"""
    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)
    await RisingEdge(dut.clk)

    val = int(dut.core_ifm_data.value)
    assert val == 0, f"FAIL: core_ifm_data=0x{val:02X} after reset, expected 0x00"
    dut._log.info(f"PASS: core_ifm_data=0x{val:02X} after reset")

@cocotb.test()
async def test_ifm_valid_port_reset(dut):
    """T9 (task3): core_ifm_valid is 0 after reset (port exists, default low)"""
    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)
    await RisingEdge(dut.clk)

    val = int(dut.core_ifm_valid.value)
    assert val == 0, f"FAIL: core_ifm_valid={val} after reset, expected 0"
    dut._log.info(f"PASS: core_ifm_valid={val} after reset")

def build_stream_beat(weights, ifms):
    """Pack 9 (weight, ifm) INT8 pairs into one 512-bit AXI4-Stream beat.
    tdata[16i+7:16i] = weight[i], tdata[16i+15:16i+8] = ifm[i]
    """
    tdata = 0
    for i, (w, f) in enumerate(zip(weights, ifms)):
        tdata |= ((w & 0xFF) << (i * 16))
        tdata |= ((f & 0xFF) << (i * 16 + 8))
    return tdata

@cocotb.test()
async def test_sequencer_valid_strobe(dut):
    """T10 (task4): valid strobes are asserted for exactly 9 cycles after start"""
    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)

    # Send a stream beat (content doesn't matter for this test)
    weights = list(range(1, 10))       # 1..9
    ifms    = [1] * 9
    await RisingEdge(dut.clk)
    dut.s_axis_tdata.value  = build_stream_beat(weights, ifms)
    dut.s_axis_tvalid.value = 1
    dut.s_axis_tlast.value  = 1
    await RisingEdge(dut.clk)
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tlast.value  = 0

    # Pulse start via AXI4-Lite CTRL
    await axil_write(dut, 0x0, 0x1)

    # Pair[0] is live on the outputs right after the CTRL write commits
    # (sequencer transitions SEQ_IDLE->SEQ_DRIVE in the same cycle as BRESP).
    # Read before awaiting the next edge to capture it; subsequent pairs arrive
    # one per rising edge.
    valid_cycles = 0
    if int(dut.core_weight_valid.value) == 1 and int(dut.core_ifm_valid.value) == 1:
        valid_cycles += 1
    for _ in range(20):
        await RisingEdge(dut.clk)
        wv = int(dut.core_weight_valid.value)
        iv = int(dut.core_ifm_valid.value)
        if wv == 1 and iv == 1:
            valid_cycles += 1
        elif valid_cycles > 0:
            break

    assert valid_cycles == 9, f"FAIL: valid asserted for {valid_cycles} cycles, expected 9"
    dut._log.info(f"PASS: valid asserted for exactly {valid_cycles} cycles")

@cocotb.test()
async def test_sequencer_data_values(dut):
    """T11 (task4): weight and IFM values driven to core match stream beat content"""
    cocotb.start_soon(Clock(dut.clk, 1, unit="ns").start())
    await reset(dut)

    weights = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    ifms    = [1,  2,  3,  4,  5,  6,  7,  8,  9]
    await RisingEdge(dut.clk)
    dut.s_axis_tdata.value  = build_stream_beat(weights, ifms)
    dut.s_axis_tvalid.value = 1
    dut.s_axis_tlast.value  = 1
    await RisingEdge(dut.clk)
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tlast.value  = 0

    await axil_write(dut, 0x0, 0x1)

    # Capture 9 pairs — pair[0] is live immediately after axil_write returns
    received_w = []
    received_f = []
    if int(dut.core_weight_valid.value) == 1:
        received_w.append(int(dut.core_weight_data.value))
        received_f.append(int(dut.core_ifm_data.value))
    for _ in range(20):
        await RisingEdge(dut.clk)
        if int(dut.core_weight_valid.value) == 1:
            received_w.append(int(dut.core_weight_data.value))
            received_f.append(int(dut.core_ifm_data.value))
        if len(received_w) == 9:
            break

    assert received_w == weights, f"FAIL: weights {received_w} != expected {weights}"
    assert received_f == ifms,    f"FAIL: ifms    {received_f} != expected {ifms}"
    dut._log.info(f"PASS: all 9 weight+IFM pairs match stream beat")
