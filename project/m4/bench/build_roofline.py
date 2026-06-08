import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)
fig.patch.set_facecolor('#f8f9fa')

ai_range = np.logspace(-2, 4, 1000)

# Hardware parameters
CPU_FLOPS   = 268.8e9;  CPU_BW   = 45.8e9   # i5-10210U Intel ARK
COPRO_FLOPS = 2.304e12; COPRO_BW = 45.8e9   # 1024-tile array (synthesis), shared DDR4

CPU_RIDGE   = CPU_FLOPS  / CPU_BW            # 5.9 FLOP/byte
COPRO_RIDGE = COPRO_FLOPS / COPRO_BW         # 50.3 FLOP/byte

# Operating points
KERNEL_AI           = 46.9     # FLOP/byte — same kernel throughout
M1_CPU_PERF         = 142.7e9  # measured in M1 (cProfile)
M4_TILE_PERF        = 2.25e9   # measured single tile (Cadence synthesis)
M4_ARRAY_PERF       = 2304e9   # 32x32 array: synthesis-validated (timing_32x32.rpt)

def roofline(ai, peak, bw):
    return np.minimum(peak, ai * bw)

# Roofline curves
ax.plot(ai_range, roofline(ai_range, CPU_FLOPS, CPU_BW),
        'b-', lw=2.2, label='CPU Roofline (i5-10210U, 268.8 GFLOP/s peak, 45.8 GB/s)')
ax.plot(ai_range, roofline(ai_range, COPRO_FLOPS, COPRO_BW),
        color='#1a7a1a', lw=2.4,
        label='Co-processor Roofline (2.304 TFLOP/s, shared DDR4-2667)')

# CPU peak / ridge
ax.axhline(CPU_FLOPS, color='blue', ls='--', lw=0.9, alpha=0.45)
ax.text(0.013, CPU_FLOPS * 1.15,
        'CPU Peak: 268.8 GFLOP/s', fontsize=8, color='blue')
ax.axvline(CPU_RIDGE, color='blue', ls=':', lw=1.1, alpha=0.5)
ax.annotate('CPU Ridge\n5.9 FLOP/B',
            xy=(CPU_RIDGE, CPU_FLOPS * 0.35),
            xytext=(CPU_RIDGE * 2.8, CPU_FLOPS * 0.18),
            fontsize=8.5, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.1),
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#eaf0ff',
                      edgecolor='blue', alpha=0.88))

# Co-proc peak / ridge
ax.axhline(COPRO_FLOPS, color='#1a7a1a', ls='--', lw=1.0, alpha=0.55)
ax.text(0.013, COPRO_FLOPS * 1.15,
        'Co-proc Peak: 2.304 TFLOP/s (1024-tile, synthesis-validated)', fontsize=8, color='#1a7a1a')
ax.axvline(COPRO_RIDGE, color='#1a7a1a', ls=':', lw=1.1, alpha=0.5)
ax.annotate('Co-proc Ridge\n50.3 FLOP/B',
            xy=(COPRO_RIDGE, COPRO_FLOPS * 0.5),
            xytext=(COPRO_RIDGE * 2.5, COPRO_FLOPS * 0.28),
            fontsize=8.5, color='#1a7a1a',
            arrowprops=dict(arrowstyle='->', color='#1a7a1a', lw=1.1),
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#e8f5e9',
                      edgecolor='#1a7a1a', alpha=0.9))

# ── Operating points ──────────────────────────────────────────────────────────

# M1 CPU (measured)
ax.scatter([KERNEL_AI], [M1_CPU_PERF],
           color='blue', s=180, zorder=10, marker='D',
           edgecolors='navy', linewidths=1.5)
ax.annotate('M1: Conv2d on CPU\n(measured, cProfile)\n142.7 GFLOP/s\nAI = 46.9 FLOP/B',
            xy=(KERNEL_AI, M1_CPU_PERF),
            xytext=(KERNEL_AI * 0.06, M1_CPU_PERF * 4.0),
            fontsize=8.5, color='navy',
            arrowprops=dict(arrowstyle='->', color='navy', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#ddeeff',
                      edgecolor='navy', alpha=0.92))

# M4 single tile (measured via Cadence, for reference)
ax.scatter([KERNEL_AI], [M4_TILE_PERF],
           color='#e65c00', s=140, zorder=10, marker='s',
           edgecolors='#a33d00', linewidths=1.5)
ax.annotate('M4: Single tile\n(Cadence synthesis)\n2.25 GFLOP/s\n500 MHz SAED14nm\n1.23 mW',
            xy=(KERNEL_AI, M4_TILE_PERF),
            xytext=(KERNEL_AI * 5.0, M4_TILE_PERF * 0.08),
            fontsize=8.5, color='#a33d00',
            arrowprops=dict(arrowstyle='->', color='#a33d00', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#fff0e0',
                      edgecolor='#a33d00', alpha=0.92))

# M4 32x32 array (synthesis-validated)
ax.scatter([KERNEL_AI], [M4_ARRAY_PERF],
           color='#1a7a1a', s=240, zorder=10, marker='*',
           edgecolors='darkgreen', linewidths=1.5)
ax.annotate('M4: 32x32 array (1024 tiles)\n(synthesis-validated)\n2.304 TFLOP/s\n1.36 W | 0.59 pJ/FLOP\nWNS=0ps @ 500 MHz',
            xy=(KERNEL_AI, M4_ARRAY_PERF),
            xytext=(KERNEL_AI * 0.05, M4_ARRAY_PERF * 0.18),
            fontsize=8.5, color='darkgreen',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#c8f0cc',
                      edgecolor='darkgreen', alpha=0.92))

# Speedup arrow: CPU → 32x32 array
ax.annotate('', xy=(KERNEL_AI, M4_ARRAY_PERF),
            xytext=(KERNEL_AI, M1_CPU_PERF),
            arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2.0))
ax.text(KERNEL_AI * 1.3, np.sqrt(M4_ARRAY_PERF * M1_CPU_PERF),
        '16.1x kernel\n3.31x system\n(Amdahl)',
        fontsize=9, color='darkgreen', fontweight='bold', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='darkgreen', alpha=0.92))

# Region labels
ax.text(0.016, 2e8, 'Memory-\nbound', fontsize=9, color='gray', style='italic', alpha=0.6)
ax.text(300,   2e8, 'Compute-\nbound', fontsize=9, color='gray', style='italic', alpha=0.6)

# Axes and labels
ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12, fontweight='bold')
ax.set_ylabel('Performance (FLOP/s)', fontsize=12, fontweight='bold')
ax.set_title(
    'M4 Final Roofline — Conv2d Accelerator\n'
    'ECE 410/510 | SAED14nm Cadence Genus | i5-10210U Host + 2.304 TFLOP/s 32x32 Array',
    fontsize=11, fontweight='bold')

ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x, _: f'{x/1e12:.1f} TFLOP/s' if x >= 1e12
           else (f'{x/1e9:.0f} GFLOP/s' if x >= 1e9
           else  f'{x/1e6:.0f} MFLOP/s')))
ax.set_xlim(0.01, 3000)
ax.set_ylim(5e7, 5e12)
ax.grid(True, which='both', alpha=0.25, linestyle='--')

legend_elements = [
    plt.Line2D([0], [0], color='blue',    lw=2,   label='CPU Roofline (i5-10210U)'),
    plt.Line2D([0], [0], color='#1a7a1a', lw=2.4, label='Co-processor Roofline (2.304 TFLOP/s, synthesis-validated)'),
    plt.scatter([], [], color='blue',    marker='D', s=80,  label='M1 CPU: 142.7 GFLOP/s (measured, cProfile)'),
    plt.scatter([], [], color='#e65c00', marker='s', s=80,  label='M4 single tile: 2.25 GFLOP/s (Cadence SAED14nm)'),
    plt.scatter([], [], color='#1a7a1a', marker='*', s=120, label='M4 32x32 array: 2.304 TFLOP/s (synthesis-validated, WNS=0ps)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=8.5, framealpha=0.93)

fig.text(0.01, 0.01,
    'All points share AI = 46.9 FLOP/byte (same Conv2d kernel). '
    'Single tile: 2.25 GFLOP/s, 1.23 mW (Cadence Genus 500 MHz SAED14nm). '
    '32x32 array (1024 tiles): 2.304 TFLOP/s, 1.36 W, 0.59 pJ/FLOP (synthesis-validated, timing_32x32.rpt WNS=0ps). '
    'System speedup (Amdahl, 74.4% accel.): 3.31x. Kernel speedup: 16.1x.',
    fontsize=7, color='gray', va='bottom')

plt.tight_layout(rect=[0, 0.05, 1, 1])
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roofline_final.png')
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved: {out}')
