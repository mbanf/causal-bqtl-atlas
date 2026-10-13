#!/usr/bin/env python3
"""
25b_rewiring_figures_only.py — Regenerate rewiring paper figures from existing results.
Loads rewiring_deep_analysis_728.csv and tf_family_transition_matrix.csv directly.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pipeline_config import config
RESULTS = str(config.results_dir)
FIG_DIR = str(config.figures_dir)

rw_df = pd.read_csv(f'{RESULTS}/rewiring_deep_analysis_728.csv')
trans_matrix = pd.read_csv(f'{RESULTS}/tf_family_transition_matrix.csv', index_col=0)
print(f"Loaded {len(rw_df)} bQTL, transition matrix {trans_matrix.shape}")

rewired = rw_df[rw_df['category'] == 'rewired'].copy()
clean = rewired[rewired['clean_switch'] == True] if 'clean_switch' in rewired.columns else rewired[rewired['both_sides_have_sole'] == True]

# ── FIGURE 1: Transition heatmap + category breakdown ──
min_trans = 3
row_sums = trans_matrix.sum(axis=1)
col_sums = trans_matrix.sum(axis=0)
keep_rows = row_sums[row_sums >= min_trans].index
keep_cols = col_sums[col_sums >= min_trans].index
plot_matrix = trans_matrix.loc[keep_rows, keep_cols]

fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [2, 1]})

# Panel A: Transition heatmap
ax = axes[0]
im = ax.imshow(plot_matrix.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(plot_matrix.columns)))
ax.set_xticklabels(plot_matrix.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(plot_matrix.index)))
ax.set_yticklabels(plot_matrix.index, fontsize=8)
ax.set_xlabel('Created (NAM allele)', fontsize=10)
ax.set_ylabel('Disrupted (B73 allele)', fontsize=10)
ax.set_title('A. TF family switching at rewired bQTL', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8, label='Number of bQTL')

for i in range(len(plot_matrix.index)):
    for j in range(len(plot_matrix.columns)):
        val = plot_matrix.values[i, j]
        if val > 0:
            color = 'white' if val > plot_matrix.values.max() * 0.6 else 'black'
            ax.text(j, i, str(int(val)), ha='center', va='center',
                    fontsize=6, color=color)

# Panel B: Category breakdown
ax = axes[1]
categories = ['Rewired\n(both)', 'Only\ndisrupted', 'Only\ncreated', 'Neither']
counts = [len(rw_df[rw_df['category'] == c]) for c in ['rewired', 'only_disrupted', 'only_created', 'neither']]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#95a5a6']

bars = ax.bar(range(4), counts, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(4))
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel('Number of bQTL', fontsize=10)
ax.set_title('B. Motif change categories', fontsize=11, fontweight='bold')

rewired_n = len(rewired)
clean_n = len(clean)
ax.annotate(f'{clean_n} clean\nsole-copy\nswitches\n({100*clean_n/rewired_n:.0f}%)',
            xy=(0, counts[0]), xytext=(0.5, counts[0] + 20),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', color='red'))

for i, (count, cat) in enumerate(zip(counts, categories)):
    ax.text(i, count + 3, str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig_motif_rewiring.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{FIG_DIR}/fig_motif_rewiring.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: fig_motif_rewiring.pdf")

# ── FIGURE 2: Effect sizes by category and sole-copy status ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Effect sizes by category
ax = axes[0]
cat_data = []
cat_labels = []
cat_colors = []
for cat, color, label in [('rewired', '#e74c3c', 'Rewired'),
                           ('only_disrupted', '#3498db', 'Only disrupted'),
                           ('only_created', '#2ecc71', 'Only created'),
                           ('neither', '#95a5a6', 'Neither')]:
    sub = rw_df[rw_df['category'] == cat]
    if len(sub) > 0:
        cat_data.append(sub['abs_d'].values)
        cat_labels.append(f'{label}\n(n={len(sub)})')
        cat_colors.append(color)

bp = ax.boxplot(cat_data, tick_labels=cat_labels, patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], cat_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel("|Cohen's d|", fontsize=10)
ax.set_title('A. Effect size by motif change category', fontsize=11, fontweight='bold')

# Panel B: Clean switch vs other rewired
ax = axes[1]
if len(clean) > 5:
    not_clean = rewired[~rewired.index.isin(clean.index)]
    data = [clean['abs_d'].values, not_clean['abs_d'].values]
    labels = [f'Clean switch\n(n={len(clean)})', f'Other rewired\n(n={len(not_clean)})']
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('#f39c12')
    bp['boxes'][1].set_alpha(0.6)
    ax.set_ylabel("|Cohen's d|", fontsize=10)
    ax.set_title('B. Clean vs other sole-copy switches', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig_rewiring_effects.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{FIG_DIR}/fig_rewiring_effects.png', bbox_inches='tight', dpi=150)
plt.close()
print("Saved: fig_rewiring_effects.pdf")

print("\nDone — rewiring figures regenerated.")
