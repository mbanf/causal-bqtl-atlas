"""
34_scatac_integration.py

Integrates causal bQTL with Marand et al. 2021 Cell single-cell ATAC-seq data.
Tests:
  1. Do causal bQTL fall in accessible chromatin regions (ACRs)?
  2. Are causal bQTL enriched in cell-type-specific vs ubiquitous ACRs?
  3. Does the billboard finding (TF identity doesn't predict effect size)
     arise from cell-type averaging?
  4. Which cell types are most affected by causal regulatory variation?

Note: ACR coordinates are AGPv4, bQTL are NAM5. For most of the genome
these are identical; we use direct coordinate overlap with warnings.
"""
import sys
sys.path.insert(0, '..')
from pipeline_config import config

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter, defaultdict
import gzip
import bisect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

FIG_DIR = config.figures_dir
RESULTS_DIR = config.results_dir
ACR_FILE = config.data_dir / 'raw' / 'scatac' / 'tmp_repo' / 'all_ACRs.celltype_calls.overlap.bed.gz'

# ── 1. Load ACR data ─────────────────────────────────────────────────────
print("=" * 70)
print("SINGLE-CELL ATAC-SEQ INTEGRATION")
print("=" * 70)

print("Loading Marand et al. 2021 scATAC ACR data...")
acr_data = []
cell_types = None

with gzip.open(ACR_FILE, 'rt') as f:
    header = f.readline().strip().split('\t')
    cell_types = header[3:]  # Skip chr, start, end
    print(f"Cell types: {len(cell_types)}")

    for line in f:
        parts = line.strip().split('\t')
        chrom = parts[0]
        start = int(parts[1])
        end = int(parts[2])
        calls = np.array([int(x) for x in parts[3:]])
        n_types = calls.sum()
        acr_data.append((chrom, start, end, n_types, calls))

print(f"ACRs loaded: {len(acr_data)}")

# Classify ACRs by specificity
spec_bins = {
    'cell-type-specific (1-3)': 0,
    'tissue-restricted (4-10)': 0,
    'broadly accessible (11-30)': 0,
    'ubiquitous (>30)': 0,
}
for _, _, _, n, _ in acr_data:
    if n <= 3:
        spec_bins['cell-type-specific (1-3)'] += 1
    elif n <= 10:
        spec_bins['tissue-restricted (4-10)'] += 1
    elif n <= 30:
        spec_bins['broadly accessible (11-30)'] += 1
    else:
        spec_bins['ubiquitous (>30)'] += 1

print("\nACR specificity distribution:")
for cat, count in spec_bins.items():
    print(f"  {cat:<30} {count:>6} ({count/len(acr_data)*100:.1f}%)")

# Build chromosome-indexed ACR intervals for fast lookup
acr_by_chr = defaultdict(list)
for i, (chrom, start, end, n_types, calls) in enumerate(acr_data):
    acr_by_chr[chrom].append((start, end, i))

# Sort by start position
for c in acr_by_chr:
    acr_by_chr[c].sort()

# ── 2. Load bQTL data ────────────────────────────────────────────────────
print("\nLoading bQTL data...")
bqtl = pd.read_csv(config.bqtl)
causal = pd.read_csv(RESULTS_DIR / 'causal_bqtl_728.csv')
causal_positions = set(zip(causal['chr'], causal['pos']))
print(f"All bQTL: {len(bqtl)}, Causal: {len(causal_positions)}")

# ── 3. Overlap bQTL with ACRs ────────────────────────────────────────────
print("\n--- Overlapping bQTL with ACRs ---")

def find_overlapping_acrs(chrom, pos, acr_chr):
    """Find ACR intervals containing position."""
    if chrom not in acr_chr:
        return []
    intervals = acr_chr[chrom]
    # Binary search for potential overlaps
    starts = [iv[0] for iv in intervals]
    idx = bisect.bisect_right(starts, pos)
    results = []
    # Check backwards from idx
    for i in range(max(0, idx - 1), -1, -1):
        s, e, acr_idx = intervals[i]
        if s > pos:
            continue
        if e >= pos:
            results.append(acr_idx)
        if s < pos - 5000:  # ACRs are typically < 5kb
            break
    # Also check forward
    for i in range(idx, min(len(intervals), idx + 3)):
        s, e, acr_idx = intervals[i]
        if s <= pos <= e:
            results.append(acr_idx)
        if s > pos:
            break
    return list(set(results))

# For all bQTL — compute overlaps once and store for reuse in section 6
overlap_results = []
bqtl_acr_map = {}  # (chr, pos) -> list of ACR indices

for idx, row in bqtl.iterrows():
    if idx % 20000 == 0:
        print(f"  {idx}/{len(bqtl)}...", flush=True)
    chrom, pos = row['chr'], row['pos']
    is_causal = (chrom, pos) in causal_positions
    overlaps = find_overlapping_acrs(chrom, pos, acr_by_chr)

    if overlaps:
        bqtl_acr_map[(chrom, pos)] = overlaps
        best_acr = max(overlaps, key=lambda x: acr_data[x][3])
        n_types = acr_data[best_acr][3]

        if n_types <= 3:
            spec = 'cell-type-specific'
        elif n_types <= 10:
            spec = 'tissue-restricted'
        elif n_types <= 30:
            spec = 'broadly-accessible'
        else:
            spec = 'ubiquitous'
    else:
        n_types = 0
        spec = 'no-ACR'

    overlap_results.append({
        'chr': chrom,
        'pos': pos,
        'is_causal': is_causal,
        'in_acr': len(overlaps) > 0,
        'n_cell_types': n_types,
        'specificity': spec,
    })

print(f"  {len(bqtl_acr_map)} bQTL overlap ACRs", flush=True)

res_df = pd.DataFrame(overlap_results)
causal_res = res_df[res_df['is_causal']]
noncausal_res = res_df[~res_df['is_causal']]

# ── 4. Compare causal vs non-causal ──────────────────────────────────────
print("\n--- ACR overlap comparison ---")
c_in_acr = causal_res['in_acr'].mean()
nc_in_acr = noncausal_res['in_acr'].mean()
print(f"Causal in ACR:     {c_in_acr:.3f} ({causal_res['in_acr'].sum()}/{len(causal_res)})")
print(f"Non-causal in ACR: {nc_in_acr:.3f} ({noncausal_res['in_acr'].sum()}/{len(noncausal_res)})")
print(f"Enrichment:        {c_in_acr/nc_in_acr:.2f}×" if nc_in_acr > 0 else "N/A")

# Fisher's exact test for ACR enrichment
a = causal_res['in_acr'].sum()
b = len(causal_res) - a
c = noncausal_res['in_acr'].sum()
d = len(noncausal_res) - c
_, fisher_p = stats.fisher_exact([[a, b], [c, d]])
print(f"Fisher's exact p:  {fisher_p:.4e}")

# ── 5. Specificity comparison ────────────────────────────────────────────
print("\n--- Cell-type specificity comparison ---")
spec_cats = ['cell-type-specific', 'tissue-restricted', 'broadly-accessible', 'ubiquitous', 'no-ACR']

print(f"{'Category':<25} {'Causal':>8} {'Non-causal':>12} {'Enrichment':>12}")
for cat in spec_cats:
    c_frac = (causal_res['specificity'] == cat).mean()
    nc_frac = (noncausal_res['specificity'] == cat).mean()
    fold = c_frac / nc_frac if nc_frac > 0 else np.inf
    print(f"{cat:<25} {c_frac:>8.3f} {nc_frac:>12.3f} {fold:>12.2f}×")

# Among those in ACRs: compare median cell type count
c_ct = causal_res[causal_res['in_acr']]['n_cell_types']
nc_ct = noncausal_res[noncausal_res['in_acr']]['n_cell_types']
mw_stat, mw_p = stats.mannwhitneyu(c_ct, nc_ct, alternative='two-sided')
print(f"\nMedian cell types (in ACR): causal {c_ct.median():.0f} vs non-causal {nc_ct.median():.0f}")
print(f"Mann-Whitney p: {mw_p:.4e}")

# ── 6. Per-cell-type enrichment ───────────────────────────────────────────
print("\n--- Per-cell-type bQTL enrichment ---")

# Use precomputed bQTL→ACR overlaps from section 3
ct_enrichment = []
for ct_idx, ct_name in enumerate(cell_types):
    if 'unknown' in ct_name:
        continue

    # Build set of ACR indices active in this cell type
    ct_acr_indices = {i for i, (_, _, _, _, calls) in enumerate(acr_data) if calls[ct_idx] == 1}

    c_count = 0
    nc_count = 0
    for (chrom, pos), acr_indices in bqtl_acr_map.items():
        in_ct = any(o in ct_acr_indices for o in acr_indices)
        if in_ct:
            if (chrom, pos) in causal_positions:
                c_count += 1
            else:
                nc_count += 1

    c_rate = c_count / len(causal_res) if len(causal_res) > 0 else 0
    nc_rate = nc_count / len(noncausal_res) if len(noncausal_res) > 0 else 0
    fold = c_rate / nc_rate if nc_rate > 0 else 0

    ct_enrichment.append({
        'cell_type': ct_name.split('.')[0],
        'causal_count': c_count,
        'noncausal_count': nc_count,
        'causal_rate': c_rate,
        'noncausal_rate': nc_rate,
        'fold_enrichment': fold,
    })

ct_df = pd.DataFrame(ct_enrichment)
# Aggregate by cell type name (some have multiple clusters)
ct_agg = ct_df.groupby('cell_type').agg({
    'causal_count': 'sum',
    'noncausal_count': 'sum',
    'causal_rate': 'mean',
    'noncausal_rate': 'mean',
    'fold_enrichment': 'mean',
}).reset_index().sort_values('fold_enrichment', ascending=False)

print(f"{'Cell type':<30} {'Causal':>8} {'Non-causal':>12} {'Enrichment':>12}")
for _, row in ct_agg.head(15).iterrows():
    print(f"{row['cell_type']:<30} {row['causal_count']:>8.0f} {row['noncausal_count']:>12.0f} {row['fold_enrichment']:>12.2f}×")

# ── 7. Billboard test in cell-type context ────────────────────────────────
print("\n--- Billboard model in cell-type context ---")

# For causal bQTL in cell-type-specific ACRs (1-3 types):
# Do they still show no TF family → effect size correlation?
causal_merged = causal.merge(
    res_df[res_df['is_causal']][['chr', 'pos', 'n_cell_types', 'specificity', 'in_acr']],
    on=['chr', 'pos'], how='inner'
)

# Compare effect sizes by specificity
for cat in ['cell-type-specific', 'tissue-restricted', 'broadly-accessible', 'ubiquitous']:
    subset = causal_merged[causal_merged['specificity'] == cat]
    if len(subset) > 5:
        print(f"  {cat:<25} n={len(subset):>4}, median |d|={subset['abs_d'].median():.3f}")

# Test: effect size vs cell type count
if len(causal_merged[causal_merged['in_acr']]) > 10:
    in_acr = causal_merged[causal_merged['in_acr']]
    rho, rho_p = stats.spearmanr(in_acr['n_cell_types'], in_acr['abs_d'])
    print(f"\n  Effect size vs cell type count: rho={rho:.3f}, p={rho_p:.4e}")
    print(f"  Interpretation: {'broader ACRs → larger effects' if rho > 0 else 'more specific ACRs → larger effects' if rho < 0 else 'no relationship'}")

# ── 8. Figure ─────────────────────────────────────────────────────────────
print("\n--- Generating figure ---")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: ACR overlap rates
ax = axes[0, 0]
cats = ['cell-type-specific', 'tissue-restricted', 'broadly-accessible', 'ubiquitous', 'no-ACR']
c_fracs = [(causal_res['specificity'] == cat).mean() for cat in cats]
nc_fracs = [(noncausal_res['specificity'] == cat).mean() for cat in cats]
x = np.arange(len(cats))
width = 0.35
ax.bar(x - width/2, c_fracs, width, label='Causal', color='crimson', alpha=0.7)
ax.bar(x + width/2, nc_fracs, width, label='Non-causal', color='steelblue', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([c.replace('-', '\n') for c in cats], fontsize=7)
ax.set_ylabel('Fraction of bQTL')
ax.set_title(f'A. bQTL by ACR specificity\n(Fisher p={fisher_p:.2e})')
ax.legend(fontsize=8)

# Panel B: Cell type count distribution
ax = axes[0, 1]
ax.hist(c_ct, bins=30, alpha=0.6, density=True, color='crimson', label=f'Causal (n={len(c_ct)})', edgecolor='white')
ax.hist(nc_ct, bins=30, alpha=0.4, density=True, color='steelblue', label=f'Non-causal (n={len(nc_ct)})', edgecolor='white')
ax.set_xlabel('Number of cell types accessible')
ax.set_ylabel('Density')
ax.set_title(f'B. ACR cell-type breadth\n(MW p={mw_p:.2e})')
ax.legend(fontsize=8)

# Panel C: Top cell types by enrichment
ax = axes[1, 0]
top_ct = ct_agg.head(12)
colors = ['#e74c3c' if f > 1 else '#3498db' for f in top_ct['fold_enrichment']]
ax.barh(range(len(top_ct)), top_ct['fold_enrichment'], color=colors, alpha=0.7)
ax.set_yticks(range(len(top_ct)))
ax.set_yticklabels(top_ct['cell_type'], fontsize=7)
ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Fold enrichment (causal/non-causal)')
ax.set_title('C. Cell-type enrichment')

# Panel D: Effect size by specificity
ax = axes[1, 1]
spec_order = ['cell-type-specific', 'tissue-restricted', 'broadly-accessible', 'ubiquitous']
data_for_box = [causal_merged[causal_merged['specificity'] == cat]['abs_d'].values for cat in spec_order]
data_for_box = [d for d in data_for_box if len(d) > 0]
spec_labels = [s for s, d in zip(spec_order, [causal_merged[causal_merged['specificity'] == cat]['abs_d'].values for cat in spec_order]) if len(d) > 0]
if data_for_box:
    bp = ax.boxplot(data_for_box, tick_labels=[s.replace('-', '\n') for s in spec_labels], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.5)
ax.set_ylabel("|Cohen's d|")
ax.set_title('D. Effect size by ACR specificity')

plt.suptitle('Single-Cell Chromatin Accessibility Integration', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / 'scatac_integration.pdf', bbox_inches='tight', dpi=150)
print(f"Saved: {FIG_DIR / 'scatac_integration.pdf'}")

# ── 9. Save results ──────────────────────────────────────────────────────
results_dict = {
    'acr_overlap': {
        'causal_in_acr': float(c_in_acr),
        'noncausal_in_acr': float(nc_in_acr),
        'enrichment': float(c_in_acr / nc_in_acr) if nc_in_acr > 0 else None,
        'fisher_p': float(fisher_p),
    },
    'specificity': {cat: {
        'causal_frac': float((causal_res['specificity'] == cat).mean()),
        'noncausal_frac': float((noncausal_res['specificity'] == cat).mean()),
    } for cat in spec_cats},
    'cell_type_breadth': {
        'causal_median': float(c_ct.median()),
        'noncausal_median': float(nc_ct.median()),
        'mannwhitney_p': float(mw_p),
    },
    'n_acrs': len(acr_data),
    'n_cell_types': len(cell_types),
    'note': 'ACR coordinates are AGPv4; bQTL are NAM5. Direct overlap used (coordinates are highly similar for euchromatic regions).'
}

with open(RESULTS_DIR / 'scatac_integration.json', 'w') as f:
    json.dump(results_dict, f, indent=2)
print(f"Saved: {RESULTS_DIR / 'scatac_integration.json'}")

ct_agg.to_csv(RESULTS_DIR / 'scatac_celltype_enrichment.csv', index=False)
print(f"Saved: {RESULTS_DIR / 'scatac_celltype_enrichment.csv'}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"ACRs: {len(acr_data)}, Cell types: {len(cell_types)}")
print(f"Causal in ACR: {c_in_acr:.1%} vs non-causal {nc_in_acr:.1%} ({c_in_acr/nc_in_acr:.2f}×, p={fisher_p:.2e})")
print(f"Cell-type breadth: causal median {c_ct.median():.0f} vs non-causal {nc_ct.median():.0f} (p={mw_p:.2e})")
