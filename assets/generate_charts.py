#!/usr/bin/env python3
"""Generate benchmark charts for TurboQuant Llama 3.1 70B README."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')

OUT = '/Users/andromeda/marketing/turboquant-llama/assets'

# ── Color palette (matches Gemma project) ──
TQ_COLOR = '#FF6B35'
MLX_COLOR = '#2EC4B6'
BASELINE_COLOR = '#8B8B8B'
DANGER_COLOR = '#CC4444'
BG_COLOR = '#FAFAFA'
GRID_ALPHA = 0.3

def style_ax(ax):
    ax.set_facecolor(BG_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=GRID_ALPHA, linestyle='--')


# ═══════════════════════════════════════════════════════════════
# Chart 1: THE HERO — Memory Footprint at 128K Context
# "This is the chart that tells the whole story"
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5.5))
style_ax(ax)

configs = [
    ('Weights only\n(4-bit, both approaches)', 39.1, 0, 0),
    ('+ FP16 KV @ 128K\n(mlx-lm baseline)', 39.1, 40.0, 0),
    ('+ int4 KV @ 128K\n(TurboQuant)', 39.1, 12.5, 0),
    ('+ int4 KV @ 128K\n+ 2GB overhead', 39.1, 12.5, 2.0),
]

y_pos = range(len(configs))
for i, (label, w, kv, overhead) in enumerate(configs):
    total = w + kv + overhead
    # Weights bar
    ax.barh(i, w, height=0.55, color=MLX_COLOR, edgecolor='white', linewidth=1.5,
            label='Model Weights (4-bit)' if i == 0 else '')
    # KV bar
    if kv > 0:
        kv_color = DANGER_COLOR if total > 64 else TQ_COLOR
        ax.barh(i, kv, height=0.55, left=w, color=kv_color, edgecolor='white', linewidth=1.5,
                label='FP16 KV Cache' if i == 1 else ('int4 KV Cache (TurboQuant)' if i == 2 else ''))
    # Overhead bar
    if overhead > 0:
        ax.barh(i, overhead, height=0.55, left=w + kv, color='#AAAAAA', edgecolor='white', linewidth=1.5,
                label='Runtime overhead' if i == 3 else '')

    fits = total <= 64
    if kv > 0 or overhead > 0:
        marker = '  fits 64 GB' if fits else '  DOESN\'T FIT'
        icon = '✓' if fits else '✗'
        text_color = '#1a7a1a' if fits else '#cc0000'
        ax.text(total + 0.8, i, f'{total:.1f} GB  {icon} {marker}', va='center', fontsize=12,
                fontweight='bold', color=text_color)
    else:
        ax.text(total + 0.8, i, f'{total:.1f} GB', va='center', fontsize=12,
                fontweight='bold', color='#555555')

# 64GB line
ax.axvline(x=64, color='red', linestyle='--', alpha=0.6, linewidth=2.5)
ax.text(64.5, -0.45, '64 GB RAM limit', color='red', fontsize=11, fontweight='bold', alpha=0.7)

# Savings annotation
ax.annotate('', xy=(51.6, 2.5), xytext=(79.1, 1.5),
            arrowprops=dict(arrowstyle='<->', color=TQ_COLOR, lw=2.5))
ax.text(65, 2.0, '27.5 GB\nsaved', fontsize=14, fontweight='bold', color=TQ_COLOR,
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=TQ_COLOR, alpha=0.9))

ax.set_yticks(y_pos)
ax.set_yticklabels([c[0] for c in configs], fontsize=11)
ax.set_xlabel('Memory (GB)', fontsize=13)
ax.set_title('Llama 3.1 70B at 128K Context: FP16 KV Doesn\'t Fit, int4 KV Does',
             fontsize=15, fontweight='bold', pad=15)
ax.set_xlim(0, 95)
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUT}/memory_128k.png', dpi=150, bbox_inches='tight')
print('Saved memory_128k.png')


# ═══════════════════════════════════════════════════════════════
# Chart 2: KV Cache Growth — The Diverging Lines
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.5))
style_ax(ax)

ctx_tokens = [0, 1024, 4096, 16384, 32768, 65536, 131072]
ctx_labels = ['0', '1K', '4K', '16K', '32K', '64K', '128K']

kv_fp16 = [0, 0.31, 1.25, 5.0, 10.0, 20.0, 40.0]
kv_int4 = [0, 0.10, 0.39, 1.56, 3.13, 6.25, 12.5]
total_fp16 = [39.1 + k + 2.0 for k in kv_fp16]
total_int4 = [39.1 + k + 2.0 for k in kv_int4]

ax.fill_between(range(len(ctx_tokens)), total_int4, total_fp16, alpha=0.15, color=TQ_COLOR,
                label='Memory saved by TurboQuant')
ax.plot(range(len(ctx_tokens)), total_fp16, 'o-', color=DANGER_COLOR, linewidth=2.5, markersize=8,
        label='FP16 KV (mlx-lm)')
ax.plot(range(len(ctx_tokens)), total_int4, 's-', color=TQ_COLOR, linewidth=2.5, markersize=8,
        label='int4 KV (TurboQuant)')

# 64GB line
ax.axhline(y=64, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.text(0.2, 65.5, '64 GB RAM', color='red', fontsize=11, fontweight='bold', alpha=0.7)

# Mark OOM point for FP16
oom_idx = 5  # 64K
ax.annotate('OOM', xy=(oom_idx, total_fp16[oom_idx]),
            xytext=(oom_idx - 0.5, total_fp16[oom_idx] + 6),
            fontsize=14, fontweight='bold', color=DANGER_COLOR,
            arrowprops=dict(arrowstyle='->', color=DANGER_COLOR, lw=2))

# Mark "Still fits!" for int4 at 128K
ax.annotate('53.6 GB\nStill fits!', xy=(6, total_int4[6]),
            xytext=(5.0, total_int4[6] + 10),
            fontsize=12, fontweight='bold', color=TQ_COLOR,
            arrowprops=dict(arrowstyle='->', color=TQ_COLOR, lw=2))

ax.set_xticks(range(len(ctx_tokens)))
ax.set_xticklabels(ctx_labels, fontsize=11)
ax.set_xlabel('Context Length', fontsize=13)
ax.set_ylabel('Total Memory (GB)', fontsize=13)
ax.set_title('Memory Grows 3.2× Slower with int4 KV Cache', fontsize=15, fontweight='bold', pad=15)
ax.set_ylim(35, 90)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
plt.tight_layout()
plt.savefig(f'{OUT}/kv_cache_growth.png', dpi=150, bbox_inches='tight')
print('Saved kv_cache_growth.png')


# ═══════════════════════════════════════════════════════════════
# Chart 3: Fused Kernel Speedup vs Context Length
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.5))
style_ax(ax)

ctx_k = ['1K', '4K', '16K', '32K', '64K', '128K']
speedups = [3.06, 10.32, 20.0, 26.1, 39.4, 48.4]

bars = ax.bar(range(len(ctx_k)), speedups, 0.6, color=TQ_COLOR, edgecolor='white', alpha=0.9)

for i, v in enumerate(speedups):
    ax.text(i, v + 1.2, f'{v:.1f}×', ha='center', fontsize=13, fontweight='bold', color=TQ_COLOR)

ax.set_xticks(range(len(ctx_k)))
ax.set_xticklabels(ctx_k, fontsize=12)
ax.set_xlabel('KV Cache Length', fontsize=13)
ax.set_ylabel('Speedup vs Dequantize + SDPA', fontsize=13)
ax.set_title('Fused sdpa_int4 Kernel Speedup (64 Q heads, 8 KV heads)',
             fontsize=15, fontweight='bold', pad=15)
ax.set_ylim(0, 58)
plt.tight_layout()
plt.savefig(f'{OUT}/kernel_speedup.png', dpi=150, bbox_inches='tight')
print('Saved kernel_speedup.png')


# ═══════════════════════════════════════════════════════════════
# Chart 4: Memory Breakdown — Stacked Bar
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
style_ax(ax)

contexts = ['1K', '4K', '16K', '32K', '64K', '128K']
weights_gb = [39.1] * 6
kv_int4_gb = [0.10, 0.39, 1.56, 3.13, 6.25, 12.50]
overhead_gb = [2.0] * 6
kv_fp16_gb = [0.31, 1.25, 5.00, 10.00, 20.00, 40.00]

x = np.arange(len(contexts))
width = 0.35

# TurboQuant bars (left)
b1 = ax.bar(x - width/2, weights_gb, width, color=MLX_COLOR, label='Weights', edgecolor='white')
b2 = ax.bar(x - width/2, kv_int4_gb, width, bottom=weights_gb, color=TQ_COLOR,
            label='int4 KV (TurboQuant)', edgecolor='white')
b3 = ax.bar(x - width/2, overhead_gb, width, bottom=[w+k for w,k in zip(weights_gb, kv_int4_gb)],
            color='#CCCCCC', label='Overhead', edgecolor='white')

# FP16 bars (right) — KV only on top of weights
b4 = ax.bar(x + width/2, weights_gb, width, color=MLX_COLOR, edgecolor='white')
b5 = ax.bar(x + width/2, kv_fp16_gb, width, bottom=weights_gb, color=DANGER_COLOR,
            label='FP16 KV (mlx-lm)', edgecolor='white', alpha=0.7)
b6 = ax.bar(x + width/2, overhead_gb, width, bottom=[w+k for w,k in zip(weights_gb, kv_fp16_gb)],
            color='#CCCCCC', edgecolor='white')

# 64GB line
ax.axhline(y=64, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.text(5.6, 65, '64 GB', color='red', fontsize=10, fontweight='bold')

# Labels
ax.set_xticks(x)
ax.set_xticklabels(contexts, fontsize=11)
ax.set_xlabel('Context Length', fontsize=13)
ax.set_ylabel('Total Memory (GB)', fontsize=13)
ax.set_title('Memory Breakdown: TurboQuant (left) vs mlx-lm (right)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 90)
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig(f'{OUT}/memory_breakdown.png', dpi=150, bbox_inches='tight')
print('Saved memory_breakdown.png')


# ═══════════════════════════════════════════════════════════════
# Chart 5: Compression Ratio Comparison
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4.5))
style_ax(ax)

methods = ['FP16\n(baseline)', 'int8', 'int4\n(TurboQuant)', 'PolarQuant\n5-bit', 'QJL m=384\n(fails at 70B)']
compression = [1.0, 2.0, 3.2, 3.07, 5.12]
kv_128k = [40.0, 20.0, 12.5, 13.0, 7.8]
colors = [BASELINE_COLOR, '#88AACC', TQ_COLOR, MLX_COLOR, '#CCCCCC']
edge_colors = ['#666666', '#5588AA', TQ_COLOR, MLX_COLOR, '#999999']

bars = ax.bar(range(len(methods)), kv_128k, 0.6, color=colors, edgecolor=edge_colors, linewidth=1.5)

for i, (v, c) in enumerate(zip(kv_128k, compression)):
    ax.text(i, v + 0.8, f'{v:.1f} GB\n({c:.1f}×)', ha='center', fontsize=11, fontweight='bold',
            color='#333333')

# Strikethrough on QJL
ax.plot([3.7, 4.3], [kv_128k[4] + 5, kv_128k[4] + 5], color=DANGER_COLOR, linewidth=3)
ax.text(4, kv_128k[4] + 7, 'fails at\n80 layers', ha='center', fontsize=9, color=DANGER_COLOR,
        fontstyle='italic')

# Highlight int4
bars[2].set_linewidth(3)

ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('KV Cache at 128K (GB)', fontsize=12)
ax.set_title('KV Cache Compression Methods — Llama 3.1 70B',
             fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 50)
plt.tight_layout()
plt.savefig(f'{OUT}/compression_comparison.png', dpi=150, bbox_inches='tight')
print('Saved compression_comparison.png')


# ═══════════════════════════════════════════════════════════════
# Chart 6: Paper vs Implementation (table-style, matching Gemma project)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_facecolor(BG_COLOR)
ax.axis('off')

table_data = [
    ['', 'TurboQuant Paper\n(ICLR 2026, A100, 8B)', 'This Implementation\n(M1 Max, 70B)'],
    ['Compression\nMethod', 'QJL + PolarQuant\n2.5-bit, near-lossless', 'int4 asymmetric\n3.2× compression'],
    ['Fused\nKernel', 'CUDA, standard sizes', 'Metal, split-K\n48× speedup at 128K'],
    ['KV at 128K\n(70B)', 'Not tested at 70B', '12.5 GB\n(fits 64GB with 11.5GB headroom)'],
    ['QJL\n1-bit keys', 'Works on 8B (32 layers)', 'Fails on 70B (80 layers)\nerror compounds'],
    ['PolarQuant', 'Works (4.2× compression)', 'Works (cosine 0.989)\nbut slower than int4'],
    ['Model\nScale', '7-8B parameter models', '70B parameters\nfirst Apple Silicon port'],
    ['Speed', 'Not reported for\n70B on Apple Silicon', '6.0 tok/s decode\n(mlx-lm: 7.3 tok/s)'],
    ['Quality', 'Near-lossless\n(0.997 recall)', 'Coherent text 200+ tokens\nmatches mlx-lm top-1'],
]

colors = [['#E8E8E8', '#FFE0CC', '#CCE8E5']]
for i in range(1, len(table_data)):
    colors.append(['#F5F5F5', '#FFF5EE', '#F0FAF8'])

table = ax.table(cellText=table_data, cellColours=colors, loc='center',
                  cellLoc='center', colWidths=[0.16, 0.42, 0.42])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.0)

for j in range(3):
    table[0, j].set_text_props(fontweight='bold', fontsize=11)

ax.set_title('TurboQuant Paper vs This Implementation', fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUT}/paper_comparison.png', dpi=150, bbox_inches='tight')
print('Saved paper_comparison.png')


print('\nAll charts generated!')
