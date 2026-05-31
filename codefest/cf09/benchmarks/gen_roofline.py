import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 8))

ai_range = np.logspace(-1, 3, 2000)

# ── Platform 1: SAED 14nm — Single tile (as synthesized) ───────────────────
peak_hw   = 9.0       # GFLOP/s  (9 MACs × 2 × 500 MHz)
bw_hw     = 32.0      # GB/s     (512-bit × 500 MHz)
ridge_hw  = peak_hw / bw_hw   # 0.28 FLOP/byte

roof_hw = np.minimum(peak_hw, bw_hw * ai_range)
ax.loglog(ai_range, roof_hw, color='steelblue', linewidth=2.2,
          label=f'SAED 14nm — Single tile (9 GFLOP/s, 32 GB/s, ridge={ridge_hw:.2f})')
ax.axhline(peak_hw, color='steelblue', linestyle='-', linewidth=0.7, alpha=0.35)
ax.axvline(ridge_hw, color='steelblue', linestyle=':', linewidth=1.0, alpha=0.5)
ax.text(ridge_hw * 1.12, peak_hw * 0.42,
        f'Ridge\n{ridge_hw:.2f} FLOP/B',
        color='steelblue', fontsize=8, va='center')
ax.text(950, peak_hw * 1.25, f'{peak_hw} GFLOP/s',
        color='steelblue', fontsize=8, ha='right')

# ── Platform 2: SAED 14nm — Design target (111-tile systolic array) ────────
peak_dt   = 999.0     # GFLOP/s  (111 tiles × 9 GFLOP/s ≈ 1 TFLOP/s)
bw_dt     = 32.0      # GB/s     (same shared AXI4-Stream bus)
ridge_dt  = peak_dt / bw_dt   # 31.2 FLOP/byte

roof_dt = np.minimum(peak_dt, bw_dt * ai_range)
ax.loglog(ai_range, roof_dt, color='seagreen', linewidth=2.2, linestyle='-',
          label=f'SAED 14nm — Design target, 111 tiles (999 GFLOP/s≈1TFLOP/s, ridge={ridge_dt:.1f})')
ax.axhline(peak_dt, color='seagreen', linestyle='-', linewidth=0.7, alpha=0.35)
ax.axvline(ridge_dt, color='seagreen', linestyle=':', linewidth=1.0, alpha=0.5)
ax.text(ridge_dt * 1.12, peak_dt * 0.42,
        f'Ridge\n{ridge_dt:.1f} FLOP/B',
        color='seagreen', fontsize=8, va='center')
ax.text(950, peak_dt * 1.06, f'{peak_dt:.0f} GFLOP/s ≈ 1 TFLOP/s',
        color='seagreen', fontsize=8, ha='right')

# ── Platform 3: i5-10210U SW baseline ──────────────────────────────────────
peak_cpu  = 268.8     # GFLOP/s
bw_cpu    = 45.8      # GB/s
ridge_cpu = peak_cpu / bw_cpu   # 5.87 FLOP/byte

roof_cpu = np.minimum(peak_cpu, bw_cpu * ai_range)
ax.loglog(ai_range, roof_cpu, color='darkorange', linewidth=2.2,
          label=f'i5-10210U SW baseline (268.8 GFLOP/s, 45.8 GB/s, ridge={ridge_cpu:.1f})')
ax.axhline(peak_cpu, color='darkorange', linestyle='-', linewidth=0.7, alpha=0.35)
ax.axvline(ridge_cpu, color='darkorange', linestyle=':', linewidth=1.0, alpha=0.5)
ax.text(ridge_cpu * 1.12, peak_cpu * 0.42,
        f'Ridge\n{ridge_cpu:.1f} FLOP/B',
        color='darkorange', fontsize=8, va='center')
ax.text(950, peak_cpu * 1.06, f'{peak_cpu} GFLOP/s',
        color='darkorange', fontsize=8, ha='right')

# ── Kernel AI bounds (BF16, 2 bytes/element) ────────────────────────────────
ai_low  = 93.8    # no reuse, BF16
ai_high = 228.8   # full weight reuse, BF16
sw_ai   = 46.9    # SW measured in FP32

for ai, label, col in [
    (ai_low,  'AI lower\n(no reuse, BF16)\n93.8', 'crimson'),
    (ai_high, 'AI upper\n(weight reuse, BF16)\n228.8', 'darkred'),
]:
    ax.axvline(ai, color=col, linestyle='--', linewidth=1.6, alpha=0.8)
    ax.text(ai * 1.06, 0.014, label, color=col, fontsize=8, va='bottom')

ax.axvline(sw_ai, color='darkorange', linestyle='-.', linewidth=1.2, alpha=0.6)

# Shade attainable range for design target
ax.fill_betweenx([peak_dt * 0.88, peak_dt * 1.12],
                 ai_low, ai_high, alpha=0.08, color='seagreen')

# ── SW measured point ───────────────────────────────────────────────────────
ax.scatter(sw_ai, 142.7, color='darkorange', s=160, zorder=6, marker='*',
           label='SW measured: 142.7 GFLOP/s @ AI=46.9 (FP32)')
ax.annotate('SW measured\n142.7 GFLOP/s\n(FP32, AI=46.9)',
            xy=(sw_ai, 142.7), xytext=(sw_ai * 0.22, 80),
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.2),
            color='darkorange', fontsize=8.5)

# ── HW single-tile points ───────────────────────────────────────────────────
hw_measured   = 1.50   # GFLOP/s  [MEASURED] sequential testbench, 6 cycles
hw_fsm        = 2.25   # GFLOP/s  [MEASURED] FSM-only 4 cycles (streaming target)
hw_zeroskip   = 2.25   # GFLOP/s  zero-skip path: 4 cycles (MEASURED)

# Sequential measured point
ax.scatter(ai_low,  hw_measured, color='steelblue', s=90, zorder=6, marker='D',
           label='HW single-tile (sequential): 1.5 GFLOP/s, 6 cycles [MEASURED]')
ax.scatter(ai_high, hw_measured, color='steelblue', s=90, zorder=6, marker='D')
ax.annotate('HW single-tile\n1.5 GFLOP/s\n6 cycles\n[MEASURED]',
            xy=(ai_low, hw_measured), xytext=(ai_low * 0.18, hw_measured * 0.25),
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2),
            color='steelblue', fontsize=8.5)

# FSM-only (streaming) measured point
ax.scatter(ai_low,  hw_fsm, color='cornflowerblue', s=90, zorder=6, marker='s',
           label='HW single-tile FSM-only: 2.25 GFLOP/s, 4 cycles [MEASURED]')
ax.scatter(ai_high, hw_fsm, color='cornflowerblue', s=90, zorder=6, marker='s')
ax.annotate('FSM-only\n2.25 GFLOP/s\n4 cycles\n[MEASURED]',
            xy=(ai_high, hw_fsm), xytext=(ai_high * 0.22, hw_fsm * 2.5),
            arrowprops=dict(arrowstyle='->', color='cornflowerblue', lw=1.2),
            color='cornflowerblue', fontsize=8.5)

# ── HW design-target projected points ──────────────────────────────────────
ax.scatter(ai_low,  peak_dt, color='seagreen', s=120, zorder=6, marker='^',
           label='HW design target (1 TFLOP/s, 111 tiles): 7× over CPU [PROJECTED]')
ax.scatter(ai_high, peak_dt, color='seagreen', s=120, zorder=6, marker='^')
ax.annotate('HW design target\n999 GFLOP/s ≈ 1 TFLOP/s\n7× over CPU\n[PROJECTED]',
            xy=(ai_high, peak_dt), xytext=(ai_high * 1.25, peak_dt * 0.35),
            arrowprops=dict(arrowstyle='->', color='seagreen', lw=1.2),
            color='seagreen', fontsize=8.5)

# ── Labels and formatting ────────────────────────────────────────────────────
ax.set_xlabel('Arithmetic Intensity (FLOP / byte)', fontsize=13)
ax.set_ylabel('Attainable Performance (GFLOP/s)', fontsize=13)
ax.set_title(
    'Roofline Model — Conv2d Kernel (BF16 hardware)\n'
    'Anemia Detection Accelerator  |  ECE 410/510 CF09',
    fontsize=12)
ax.set_xlim(0.1, 1000)
ax.set_ylim(0.01, 5000)
ax.grid(True, which='both', alpha=0.25)
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig('roofline_plot.png', dpi=150, bbox_inches='tight')
print("Saved roofline_plot.png")
