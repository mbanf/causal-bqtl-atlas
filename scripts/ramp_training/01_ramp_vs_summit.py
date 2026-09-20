#!/usr/bin/env python3
"""
01_ramp_vs_summit.py — Test Thomas's ramp hypothesis for bQTL training.

Hypothesis (Thomas Hartwig):
  TF binding sites sit in the RAMP (slope) of MOA peaks, not at summits.
  MNase exonuclease only degrades one strand; the real resolution comes from
  250× more frequent AT-cutting. TFs occupy the ~20bp zone in the ramp
  where signal drops from peak to baseline.

Predictions:
  1. bQTLs should be enriched at peak edges (ramps), not centers
  2. Models trained on ramp-positioned bQTLs should perform better
     (cleaner signal: directly disrupting TF motifs)
  3. Summit-positioned bQTLs may be noisier (indirect effects)

Experiment:
  Phase 1: Map bQTLs to peaks, characterize positional distribution
  Phase 2: Split into ramp/summit/full training sets
  Phase 3: Train RF models (fast) and compare AUC
  Phase 4: Train CNN models and compare

Uses existing hard-negative balanced dataset + pan-cistrome peak coordinates.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent.parent  # causal_bqtl_atlas/
DATA = BASE / "data"
RESULTS = BASE / "results"
OUTDIR = Path(__file__).resolve().parent
OUTDIR.mkdir(exist_ok=True)
FIG_DIR = OUTDIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Phase 1: Map bQTLs to peaks ──────────────────────────────────────────────

print("=" * 70)
print("RAMP VS SUMMIT: Testing Thomas's MOA-seq hypothesis")
print("=" * 70)

# Load data
bqtl = pd.read_csv(DATA / "processed" / "bqtl_snp_ww.csv")
peaks = pd.read_csv(RESULTS / "pan_cistrome_peaks.csv")

print(f"\nLoaded {len(bqtl):,} bQTLs, {len(peaks):,} pan-cistrome peaks")
print(f"Peak width: median={peaks['width'].median():.0f}bp, "
      f"mean={peaks['width'].mean():.0f}bp, "
      f"Q25={peaks['width'].quantile(0.25):.0f}bp, "
      f"Q75={peaks['width'].quantile(0.75):.0f}bp")

# Build interval index per chromosome for fast lookup
print("\nBuilding peak interval index...")
peak_idx = defaultdict(list)
for _, p in peaks.iterrows():
    peak_idx[p['chr']].append((p['start'], p['end'], p['width']))

# Sort by start position for binary search
for chrom in peak_idx:
    peak_idx[chrom].sort()

# Map each bQTL to its containing peak
def find_peak(chrom, pos, idx):
    """Find the peak containing this position. Returns (start, end, width) or None."""
    candidates = idx.get(chrom, [])
    # Binary search for efficiency
    lo, hi = 0, len(candidates) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end, width = candidates[mid]
        if pos < start:
            hi = mid - 1
        elif pos > end:
            lo = mid + 1
        else:
            return start, end, width
    return None

print("Mapping bQTLs to peaks...")
in_peak = []
rel_positions = []  # 0 = start edge, 0.5 = center, 1.0 = end edge
dist_to_edge = []   # min distance to nearest edge (bp)
peak_widths = []

n_mapped = 0
for _, row in bqtl.iterrows():
    result = find_peak(row['chr'], row['pos'], peak_idx)
    if result is not None:
        start, end, width = result
        n_mapped += 1
        in_peak.append(True)
        # Relative position: 0 = start, 1 = end
        rel = (row['pos'] - start) / max(width, 1)
        rel_positions.append(rel)
        # Distance to nearest edge
        d_edge = min(row['pos'] - start, end - row['pos'])
        dist_to_edge.append(d_edge)
        peak_widths.append(width)
    else:
        in_peak.append(False)
        rel_positions.append(np.nan)
        dist_to_edge.append(np.nan)
        peak_widths.append(np.nan)

bqtl['in_peak'] = in_peak
bqtl['rel_position'] = rel_positions
bqtl['dist_to_edge'] = dist_to_edge
bqtl['peak_width'] = peak_widths

# Symmetrize relative position: 0 = edge, 0.5 = center
bqtl['rel_pos_sym'] = bqtl['rel_position'].apply(
    lambda x: min(x, 1 - x) if not np.isnan(x) else np.nan
)

n_total = len(bqtl)
n_in = bqtl['in_peak'].sum()
print(f"\nbQTLs in peaks: {n_in:,} / {n_total:,} ({100*n_in/n_total:.1f}%)")
print(f"bQTLs outside peaks: {n_total - n_in:,}")

mapped = bqtl[bqtl['in_peak']].copy()
print(f"\nPositional distribution (mapped bQTLs):")
print(f"  Distance to nearest edge: median={mapped['dist_to_edge'].median():.0f}bp, "
      f"mean={mapped['dist_to_edge'].mean():.0f}bp")
print(f"  Relative position (sym): median={mapped['rel_pos_sym'].median():.3f} "
      f"(0=edge, 0.5=center)")

# ── Ramp vs Summit classification ────────────────────────────────────────────

# Thomas's model: TFs bind ~20bp from cut sites in the ramp
# Define regions based on distance to peak edge
RAMP_BP = 30   # bp from edge = ramp zone (Thomas: ~20bp, we use 30 for margin)
# Also define by fractional position for peaks of varying width
RAMP_FRAC = 0.2  # outer 20% on each side = ramp

mapped['zone_abs'] = 'middle'
mapped.loc[mapped['dist_to_edge'] <= RAMP_BP, 'zone_abs'] = 'ramp'
mapped.loc[mapped['dist_to_edge'] > mapped['peak_width'] * 0.3, 'zone_abs'] = 'summit'

mapped['zone_frac'] = pd.cut(mapped['rel_pos_sym'],
                              bins=[0, RAMP_FRAC, 0.5 - RAMP_FRAC, 0.5],
                              labels=['ramp', 'middle', 'summit'],
                              include_lowest=True)

print(f"\n── Zone classification (absolute: ramp ≤{RAMP_BP}bp from edge) ──")
for zone in ['ramp', 'middle', 'summit']:
    n = (mapped['zone_abs'] == zone).sum()
    print(f"  {zone:8s}: {n:>7,} ({100*n/len(mapped):.1f}%)")

print(f"\n── Zone classification (fractional: ramp = outer {RAMP_FRAC*100:.0f}% each side) ──")
for zone in ['ramp', 'middle', 'summit']:
    n = (mapped['zone_frac'] == zone).sum()
    print(f"  {zone:8s}: {n:>7,} ({100*n/len(mapped):.1f}%)")

# ── Enrichment test: are bQTLs enriched at edges? ────────────────────────────

print("\n── Edge enrichment test ──")
# Under null (uniform), rel_pos_sym should be uniform on [0, 0.5]
# Expected fraction in ramp (0 to RAMP_FRAC): 2 * RAMP_FRAC
expected_ramp_frac = 2 * RAMP_FRAC
observed_ramp_frac = (mapped['zone_frac'] == 'ramp').sum() / len(mapped)
enrichment = observed_ramp_frac / expected_ramp_frac

print(f"  Expected ramp fraction (uniform): {expected_ramp_frac:.3f}")
print(f"  Observed ramp fraction:           {observed_ramp_frac:.3f}")
print(f"  Enrichment:                       {enrichment:.3f}x")

from scipy import stats
# KS test: is the distribution skewed toward edges?
ks_stat, ks_p = stats.kstest(mapped['rel_pos_sym'].dropna(), 'uniform',
                              args=(0, 0.5))
print(f"  KS test vs uniform [0, 0.5]:      D={ks_stat:.4f}, p={ks_p:.2e}")

# ── Visualization ─────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# A: Histogram of relative position (symmetric)
ax = axes[0, 0]
ax.hist(mapped['rel_pos_sym'].dropna(), bins=50, color='#2E7D32', alpha=0.8,
        edgecolor='white', linewidth=0.5)
ax.axvline(RAMP_FRAC, color='red', ls='--', label=f'Ramp boundary ({RAMP_FRAC})')
ax.axhline(len(mapped)/50, color='gray', ls=':', alpha=0.5, label='Uniform expectation')
ax.set_xlabel('Relative position (0 = edge, 0.5 = center)')
ax.set_ylabel('Count')
ax.set_title('A. bQTL position within MOA peaks')
ax.legend(fontsize=9)

# B: Histogram of absolute distance to edge
ax = axes[0, 1]
ax.hist(mapped['dist_to_edge'].dropna().clip(upper=200), bins=100,
        color='#1565C0', alpha=0.8, edgecolor='white', linewidth=0.3)
ax.axvline(RAMP_BP, color='red', ls='--', label=f'Ramp zone ({RAMP_BP}bp)')
ax.set_xlabel('Distance to nearest peak edge (bp)')
ax.set_ylabel('Count')
ax.set_title('B. Distance to peak edge')
ax.legend(fontsize=9)

# C: Distribution by peak width bins
ax = axes[1, 0]
width_bins = pd.qcut(mapped['peak_width'], q=4, labels=['Q1 (narrow)', 'Q2', 'Q3', 'Q4 (wide)'])
for i, label in enumerate(['Q1 (narrow)', 'Q2', 'Q3', 'Q4 (wide)']):
    subset = mapped[width_bins == label]['rel_pos_sym'].dropna()
    ax.hist(subset, bins=30, alpha=0.5, label=label, density=True)
ax.set_xlabel('Relative position (0 = edge, 0.5 = center)')
ax.set_ylabel('Density')
ax.set_title('C. Position by peak width quartile')
ax.legend(fontsize=8)

# D: Zone counts bar chart
ax = axes[1, 1]
zone_counts = mapped['zone_frac'].value_counts()
colors = {'ramp': '#E53935', 'middle': '#FFA726', 'summit': '#42A5F5'}
bars = ax.bar(zone_counts.index, zone_counts.values,
              color=[colors.get(z, 'gray') for z in zone_counts.index])
ax.set_ylabel('Count')
ax.set_title(f'D. Zone classification (ramp = outer {RAMP_FRAC*100:.0f}%)')
for bar, count in zip(bars, zone_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'{count:,}', ha='center', va='bottom', fontsize=10)

plt.suptitle("bQTL position within MOA-seq peaks\n"
             "Thomas's hypothesis: TF binding in ramps, not summits",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'bqtl_peak_position.pdf', dpi=150, bbox_inches='tight')
plt.savefig(FIG_DIR / 'bqtl_peak_position.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {FIG_DIR / 'bqtl_peak_position.pdf'}")

# ── Save mapped data for Phase 2 ─────────────────────────────────────────────

mapped.to_csv(OUTDIR / 'bqtl_peak_mapping.csv', index=False)
print(f"Saved: {OUTDIR / 'bqtl_peak_mapping.csv'} ({len(mapped):,} rows)")

# ── Summary statistics ────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PHASE 1 SUMMARY")
print("=" * 70)
print(f"  Total bQTLs:              {len(bqtl):>10,}")
print(f"  Mapped to peaks:          {len(mapped):>10,} ({100*len(mapped)/len(bqtl):.1f}%)")
print(f"  Ramp zone (edge):         {(mapped['zone_frac']=='ramp').sum():>10,}")
print(f"  Middle zone:              {(mapped['zone_frac']=='middle').sum():>10,}")
print(f"  Summit zone (center):     {(mapped['zone_frac']=='summit').sum():>10,}")
print(f"  Edge enrichment:          {enrichment:>10.3f}x")
print(f"  Median peak width:        {mapped['peak_width'].median():>10.0f} bp")
print(f"  Median dist to edge:      {mapped['dist_to_edge'].median():>10.0f} bp")
print()
if enrichment > 1.1:
    print("  → bQTLs are ENRICHED at peak edges — supports ramp hypothesis")
elif enrichment < 0.9:
    print("  → bQTLs are DEPLETED at peak edges — against ramp hypothesis")
else:
    print("  → bQTLs show no strong edge preference — inconclusive")
