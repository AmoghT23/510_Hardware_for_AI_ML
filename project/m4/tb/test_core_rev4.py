"""
test_core_rev4.py
ECE 410/510 — Direct cocotb testbench for compute_core_ti Rev4

Drives weights_packed[143:0] and ifms_packed[143:0] directly (no interface wrapper).
Measures actual cycle count from start assertion to done assertion.

Clock: 2 ns / 500 MHz  (matches Cadence Genus synthesis clock)

FSM under test:
    IDLE → LOAD(1) → PIPE_MUL(1) → PIPE_ACC(1) → DONE_ST(1)
    done is registered: asserted on the posedge that processes DONE_ST.
    Cycle count from start posedge to done-visible = 4 cycles (normal path).
    Zero-skip path: LOAD detects ifm_all_zero → DONE_ST in 2 cycles.

Tests:
    T1  ramp weights × unit IFMs          → 45.0    (normal path, 4 cycles)
    T2  alternating-sign × constant 2.0   → 2.0     (normal path, 4 cycles)
    T3  max BF16 weights × unit IFMs      → 1143.0  (normal path, 4 cycles)
    T4  Laplacian × ramp IFMs             → 0.0     (normal path, 4 cycles)
    T5  all-zero IFMs × ramp weights      → 0.0     (zero-skip,   2 cycles)
    THROUGHPUT: N=200 back-to-back tiles  → measured GFLOP/s
"""

import struct
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import cocotb.utils

CLOCK_NS   = 2        # 500 MHz
CLOCK_MHZ  = 1000 / CLOCK_NS
FLOPS_TILE = 18       # 9 MACs × 2 FLOP/MAC
N_TILES    = 200      # throughput measurement run length


# ── BF16 helpers ──────────────────────────────────────────────────────────────

def float_to_bf16(f: float) -> int:
    """Python float → 16-bit BF16 pattern (truncate lower 16 mantissa bits)."""
    b = struct.pack('>f', float(f))
    return (int.from_bytes(b, 'big') >> 16) & 0xFFFF


def bf16_to_float(bf16: int) -> float:
    """16-bit BF16 pattern → Python float."""
    return struct.unpack('>f', (bf16 << 16).to_bytes(4, 'big'))[0]


def fp32_bits(f: float) -> int:
    return int.from_bytes(struct.pack('>f', float(f)), 'big')


def pack144(values: list) -> int:
    """Pack 9 floats → 144-bit int with element i at bits [i*16+15 : i*16]."""
    word = 0
    for i, v in enumerate(values):
        word |= float_to_bf16(v) << (i * 16)
    return word


def ref_dot(weights: list, ifms: list) -> int:
    """Reference dot-product: quantize both to BF16, accumulate in float64."""
    acc = 0.0
    for w, x in zip(weights, ifms):
        acc += bf16_to_float(float_to_bf16(w)) * bf16_to_float(float_to_bf16(x))
    return fp32_bits(acc)


# ── DUT helpers ───────────────────────────────────────────────────────────────

async def reset(dut):
    dut.rst_n.value         = 0
    dut.start.value         = 0
    dut.weights_packed.value = 0
    dut.ifms_packed.value   = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def run_tile_measured(dut, weights: list, ifms: list):
    """
    Drive one tile and return (fp32_result_bits, cycle_count).

    Protocol:
      1. Present weights_packed + ifms_packed (stable before start).
      2. Assert start for exactly 1 rising edge.
      3. Hold data stable through the LOAD edge (Edge 2 after start).
      4. Poll done after each rising edge; count edges until done=1.
    """
    # Wait until core is ready (IDLE state)
    for _ in range(50):
        if dut.ready.value == 1:
            break
        await RisingEdge(dut.clk)

    # Present data — stable before start
    dut.weights_packed.value = pack144(weights)
    dut.ifms_packed.value    = pack144(ifms)

    # Capture start time, then assert start for one cycle
    await RisingEdge(dut.clk)
    t_start = cocotb.utils.get_sim_time(units='ns')
    dut.start.value = 1
    await RisingEdge(dut.clk)   # Edge 1: IDLE → LOAD  (data latched here)
    dut.start.value = 0
    # Hold data for one more edge to ensure LOAD sees it (already latched, but safe)

    # Count edges until done=1 (max 8 to catch hangs)
    cycles = 1
    for _ in range(8):
        await RisingEdge(dut.clk)
        cycles += 1
        if dut.done.value == 1:
            break

    t_end = cocotb.utils.get_sim_time(units='ns')
    elapsed_ns = t_end - t_start
    result_bits = int(dut.result.value)
    return result_bits, cycles, elapsed_ns


# ── Functional tests ──────────────────────────────────────────────────────────

VECTORS = [
    {
        "name":    "T1: ramp weights × unit IFMs",
        "weights": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ifms":    [1.0] * 9,
        "expect":  45.0,
        "path":    "normal",
    },
    {
        "name":    "T2: alternating-sign × constant 2.0",
        "weights": [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 1.0],
        "ifms":    [2.0] * 9,
        "expect":  2.0,
        "path":    "normal",
    },
    {
        "name":    "T3: max BF16 weights × unit IFMs",
        "weights": [127.0] * 9,
        "ifms":    [1.0] * 9,
        "expect":  1143.0,
        "path":    "normal",
    },
    {
        "name":    "T4: Laplacian kernel × ramp IFMs",
        "weights": [-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0],
        "ifms":    [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        "expect":  0.0,
        "path":    "normal",
    },
    {
        "name":    "T5: all-zero IFMs (zero-skip path)",
        "weights": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "ifms":    [0.0] * 9,
        "expect":  0.0,
        "path":    "zero-skip",
    },
]


@cocotb.test()
async def test_functional_and_cycles(dut):
    """Run all functional vectors, verify results, report measured cycle counts."""
    cocotb.start_soon(Clock(dut.clk, CLOCK_NS, unit="ns").start())
    await reset(dut)

    dut._log.info("=" * 68)
    dut._log.info("compute_core_ti Rev4 — Direct BF16 functional + cycle measurement")
    dut._log.info(f"Clock: {CLOCK_MHZ:.0f} MHz ({CLOCK_NS} ns period)")
    dut._log.info("=" * 68)

    passed = 0
    failed = 0

    for vec in VECTORS:
        expected_bits = ref_dot(vec["weights"], vec["ifms"])
        result_bits, cycles, elapsed_ns = await run_tile_measured(
            dut, vec["weights"], vec["ifms"]
        )

        exp_f = struct.unpack('>f', expected_bits.to_bytes(4, 'big'))[0]
        res_f = struct.unpack('>f', result_bits.to_bytes(4, 'big'))[0]
        ok    = result_bits == expected_bits

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        dut._log.info(
            f"  {status}  {vec['name']}"
        )
        dut._log.info(
            f"        got={res_f}  expected={exp_f}"
            f"  |  cycles={cycles}  elapsed={elapsed_ns:.0f} ns"
            f"  |  path={vec['path']}"
        )

    dut._log.info("-" * 68)
    dut._log.info(f"PASS={passed}  FAIL={failed}")
    dut._log.info("=" * 68)
    assert failed == 0, f"{failed} functional test(s) failed"


# ── Throughput measurement ────────────────────────────────────────────────────

@cocotb.test()
async def test_throughput(dut):
    """
    Run N_TILES back-to-back tiles (weights loaded, different IFMs).
    Measure total sim time → compute actual GFLOP/s.
    """
    cocotb.start_soon(Clock(dut.clk, CLOCK_NS, unit="ns").start())
    await reset(dut)

    WEIGHTS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    IFM     = [1.0] * 9

    dut._log.info("=" * 68)
    dut._log.info(f"THROUGHPUT MEASUREMENT — {N_TILES} back-to-back tiles")
    dut._log.info("=" * 68)

    total_cycles = 0
    cycle_counts = []

    for _ in range(N_TILES):
        _, cycles, _ = await run_tile_measured(dut, WEIGHTS, IFM)
        total_cycles += cycles
        cycle_counts.append(cycles)

    avg_cycles  = total_cycles / N_TILES
    time_per_ns = avg_cycles * CLOCK_NS
    throughput  = FLOPS_TILE / (time_per_ns * 1e-9) / 1e9

    dut._log.info(f"  Tiles measured          : {N_TILES}")
    dut._log.info(f"  FLOPs per tile          : {FLOPS_TILE}  (9 MACs × 2)")
    dut._log.info(f"  Synthesis clock         : {CLOCK_MHZ:.0f} MHz")
    dut._log.info(f"  Avg cycles per tile     : {avg_cycles:.2f}")
    dut._log.info(f"  Avg time per tile       : {time_per_ns:.1f} ns")
    dut._log.info(f"  MEASURED throughput     : {throughput:.4f} GFLOP/s")
    dut._log.info(f"  Theoretical peak        : {FLOPS_TILE / (4 * CLOCK_NS * 1e-9) / 1e9:.4f} GFLOP/s  (4 cycles, no overhead)")
    dut._log.info(f"  Utilization vs 4-cycle  : {throughput / (FLOPS_TILE / (4 * CLOCK_NS * 1e-9) / 1e9) * 100:.1f}%")
    dut._log.info(f"  Unique cycle counts     : {sorted(set(cycle_counts))}")
    dut._log.info("=" * 68)
