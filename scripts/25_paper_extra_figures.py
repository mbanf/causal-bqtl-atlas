#!/usr/bin/env python3
"""
25_paper_extra_figures.py — Generate paper figures not covered by other scripts:
  1. fig_causal_bqtl_hybrid_distribution.pdf  (4-panel hybrid distribution)
  2. fig_tf_cascade_network.pdf               (TF cascade network)
"""

import pandas as pd
import numpy as np
from collections import Counter
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pipeline_config import config
RESULTS = str(config.results_dir)
FIGS = str(config.figures_dir)
os.makedirs(FIGS, exist_ok=True)

# ── Load data ──
causal = pd.read_csv(f'{RESULTS}/causal_bqtl_728.csv')
atlas = pd.read_csv(f'{RESULTS}/gene_atlas_728.csv')
N = len(causal)
print(f"Loaded {N} causal bQTL")

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Hybrid Distribution (4-panel)
# ═══════════════════════════════════════════════════════════════

# Parse variant hybrids
hybrid_counts = Counter()
carrier_counts = []
for _, row in causal.iterrows():
    vh = str(row.get('variant_hybrids', ''))
    if vh and vh != 'nan':
        hybrids = vh.split('|')
        carrier_counts.append(len(hybrids))
        for h in hybrids:
            hybrid_counts[h.strip()] += 1
    else:
        carrier_counts.append(int(row['n_variant']))

causal['n_carriers'] = carrier_counts

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel A: Histogram of carrier count
ax = axes[0, 0]
ax.hist(carrier_counts, bins=range(1, max(carrier_counts) + 2),
        color='#1565C0', alpha=0.8, edgecolor='white', linewidth=0.5)
ax.axvline(np.median(carrier_counts), color='red', linestyle='--', linewidth=1.5,
           label=f'Median = {np.median(carrier_counts):.0f}')
ax.set_xlabel('Number of hybrids carrying variant')
ax.set_ylabel('Number of bQTL')
ax.set_title('A. Variant frequency distribution', fontweight='bold')
ax.legend()

# Panel B: Effect size vs carrier count
ax = axes[0, 1]
abs_d = causal['abs_d'].values
n_carr = np.array(carrier_counts)
ax.scatter(n_carr, abs_d, alpha=0.3, s=8, c='#1565C0')
# Add trend line
try:
    from scipy.stats import spearmanr
    rho, p = spearmanr(n_carr, abs_d)
    z = np.polyfit(n_carr, abs_d, 1)
    poly = np.poly1d(z)
    x_line = np.linspace(n_carr.min(), n_carr.max(), 100)
    ax.plot(x_line, poly(x_line), 'r-', linewidth=1.5)
    ax.set_title(f'B. Effect size vs frequency\n(ρ = {rho:.2f}, p = {p:.3f})',
                 fontweight='bold')
except ImportError:
    ax.set_title('B. Effect size vs frequency', fontweight='bold')
ax.set_xlabel('Number of hybrids carrying variant')
ax.set_ylabel("|Cohen's d|")

# Panel C: Variants per hybrid
ax = axes[1, 0]
hybrid_names = sorted(hybrid_counts.keys(), key=lambda x: hybrid_counts[x])
hybrid_vals = [hybrid_counts[h] for h in hybrid_names]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(hybrid_names)))
bars = ax.barh(range(len(hybrid_names)), hybrid_vals, color=colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(hybrid_names)))
ax.set_yticklabels(hybrid_names, fontsize=7)
ax.set_xlabel('Number of causal variants')
ax.set_title(f'C. Causal variants per hybrid ({min(hybrid_vals)}–{max(hybrid_vals)})',
             fontweight='bold')

# Panel D: Cumulative distribution
ax = axes[1, 1]
sorted_carriers = np.sort(carrier_counts)
cumulative = np.arange(1, len(sorted_carriers) + 1) / len(sorted_carriers)
ax.plot(sorted_carriers, cumulative, color='#1565C0', linewidth=2)
ax.fill_between(sorted_carriers, 0, cumulative, alpha=0.1, color='#1565C0')
ax.set_xlabel('Number of hybrids carrying variant')
ax.set_ylabel('Cumulative fraction of bQTL')
ax.set_title('D. Cumulative variant distribution', fontweight='bold')
ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
ax.axvline(np.median(carrier_counts), color='red', linestyle='--', linewidth=1)

plt.suptitle(f'Distribution of {N} Causal bQTL Across {len(hybrid_counts)} NAM Hybrids',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{FIGS}/fig_causal_bqtl_hybrid_distribution.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{FIGS}/fig_causal_bqtl_hybrid_distribution.png', bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved: fig_causal_bqtl_hybrid_distribution.pdf")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: TF Cascade Network
# ═══════════════════════════════════════════════════════════════

# Identify TF genes among causal targets
tf_targets = atlas[atlas['signaling_layer'] == 'Transcription Factor'].copy()
n_tf = len(tf_targets)
print(f"\nTF genes among causal targets: {n_tf}")

# Get disrupted families for each TF target
tf_families = {}
for _, row in tf_targets.iterrows():
    fams = str(row.get('tf_families', ''))
    if fams and fams != 'nan':
        tf_families[row['gene_id']] = fams.split(',')

# Count disrupted families across TF targets
disrupted_at_tf = Counter()
for gid, fams in tf_families.items():
    for f in fams:
        disrupted_at_tf[f.strip()] += 1

# Count how many non-TF genes each TF family disrupts
all_families_count = Counter()
for _, row in atlas.iterrows():
    fams = str(row.get('tf_families', ''))
    if fams and fams != 'nan':
        for f in fams.split(','):
            all_families_count[f.strip()] += 1

fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# Panel A: TF targets by disrupted family
ax = axes[0]
top_disrupt = disrupted_at_tf.most_common(15)
if top_disrupt:
    families, counts = zip(*top_disrupt)
    y = range(len(families))
    ax.barh(y, counts, color='#E53935', alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(families, fontsize=8)
    ax.set_xlabel('TF target genes disrupted')
    ax.set_title('A. Disrupted motifs at TF gene bQTL', fontweight='bold')
    ax.invert_yaxis()

# Panel B: Downstream reach of TF targets
ax = axes[1]
# Each TF target: how many total genes share same disrupted family?
tf_reach = []
tf_labels = []
for _, row in tf_targets.iterrows():
    fams = str(row.get('tf_families', ''))
    if fams and fams != 'nan':
        total_reach = sum(all_families_count.get(f.strip(), 0) for f in fams.split(','))
        desc = str(row.get('description', ''))[:30]
        tf_reach.append(total_reach)
        tf_labels.append(f"{row['gene_id'][-6:]}")

if tf_reach:
    # Show top 20 by reach
    sorted_idx = np.argsort(tf_reach)[::-1][:20]
    y = range(len(sorted_idx))
    ax.barh(y, [tf_reach[i] for i in sorted_idx], color='#2196F3', alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([tf_labels[i] for i in sorted_idx], fontsize=7)
    ax.set_xlabel('Co-regulated genes (same TF family)')
    ax.set_title('B. Top TF cascade nodes by reach', fontweight='bold')
    ax.invert_yaxis()

# Panel C: Drought response of TF targets
ax = axes[2]
if 'log2fc_drought' in tf_targets.columns:
    drought_vals = tf_targets['log2fc_drought'].dropna()
    up = (drought_vals > 1).sum()
    down = (drought_vals < -1).sum()
    stable = len(drought_vals) - up - down

    colors_pie = ['#E53935', '#2196F3', '#BDBDBD']
    labels_pie = [f'Drought UP\n({up})', f'Drought DOWN\n({down})', f'Stable\n({stable})']
    sizes = [up, down, stable]
    ax.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.0f%%',
           startangle=90, textprops={'fontsize': 9})
    ax.set_title(f'C. Drought response of {n_tf} TF targets', fontweight='bold')

    # Add effect size annotation
    mean_d = tf_targets['abs_d'].mean()
    ax.text(0, -1.3, f'Mean |d| = {mean_d:.2f}', ha='center', fontsize=9,
            transform=ax.transAxes)

plt.suptitle(f'TF Cascade Network — {n_tf} TF Genes Among {N} Causal bQTL Targets',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f'{FIGS}/fig_tf_cascade_network.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{FIGS}/fig_tf_cascade_network.png', bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved: fig_tf_cascade_network.pdf")

print("\nDone — all paper extra figures generated.")
