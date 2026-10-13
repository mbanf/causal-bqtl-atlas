"""
37_te_context.py

Tests whether causal bQTL are enriched in specific transposable element
(TE) contexts. Uses EDTA TE annotations for B73 NAM5 from MaizeGDB.

Key question: Are causal regulatory variants preferentially located in
TE-derived regulatory elements? Which TE superfamilies harbour more
binding variation?

Builds on the chr2 hotspot finding (nested LTR retrotransposons) and
tests whether this is a general pattern.
"""
import sys
sys.path.insert(0, '..')
from pipeline_config import config

import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import gzip
import bisect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

np.random.seed(42)

FIG_DIR = config.figures_dir
RESULTS_DIR = config.results_dir
TE_GFF = config.data_dir / 'raw' / 'B73_NAM5_TE.gff3.gz'

# ── 1. Load TE annotations ──────────────────────────────────────────────
print("=" * 70)
print("TRANSPOSABLE ELEMENT CONTEXT OF bQTL")
print("=" * 70)

print("Loading TE annotations...")
# Build interval index per chromosome for fast lookup
te_intervals = defaultdict(list)  # chr -> [(start, end, superfamily, subfamily)]
te_count = 0
with gzip.open(TE_GFF, 'rt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9:
            continue
        chrom = parts[0]
        if chrom not in config.valid_chroms:
            continue
        start = int(parts[3])
        end = int(parts[4])
        attrs = parts[8]
        # Parse Classification=LTR/Gypsy
        classif = ''
        for a in attrs.split(';'):
            if a.startswith('Classification='):
                classif = a.split('=')[1]
                break
        superfamily = classif.split('/')[0] if classif else 'unknown'
        subfamily = classif.split('/')[1] if '/' in classif else 'unknown'
        te_intervals[chrom].append((start, end, superfamily, subfamily))
        te_count += 1

# Sort by start position for binary search
for chrom in te_intervals:
    te_intervals[chrom].sort(key=lambda x: x[0])

print(f"TE annotations loaded: {te_count}")
for chrom in sorted(te_intervals.keys()):
    print(f"  {chrom}: {len(te_intervals[chrom])} TEs")

# ── 2. Load bQTL data ──────────────────────────────────────────────────
print("\n--- Loading bQTL data ---")
bqtl_all = pd.read_csv(config.bqtl, usecols=['chr', 'pos'])
causal_ww = pd.read_csv(RESULTS_DIR / 'hybrid_motif_linkage.csv')
causal_ds = pd.read_csv(RESULTS_DIR / 'hybrid_motif_linkage_ds.csv')

causal_positions = set()
for _, row in pd.concat([causal_ww[['chr', 'pos']], causal_ds[['chr', 'pos']]]).iterrows():
    causal_positions.add((row['chr'], row['pos']))

bqtl_all['is_causal'] = bqtl_all.apply(lambda r: (r['chr'], r['pos']) in causal_positions, axis=1)
print(f"Total bQTL: {len(bqtl_all)}")
print(f"Causal: {bqtl_all['is_causal'].sum()}")

# ── 3. Classify each bQTL by TE context ────────────────────────────────
print("\n--- Classifying bQTL by TE context ---")

def find_te_at_position(chrom, pos, intervals):
    """Find TE(s) overlapping a given position using binary search."""
    if chrom not in intervals:
        return []
    tes = intervals[chrom]
    # Binary search for TEs that could overlap
    # Find rightmost TE with start <= pos
    starts = [t[0] for t in tes]
    idx = bisect.bisect_right(starts, pos) - 1

    results = []
    # Check nearby TEs (some may be very long)
    # Look backwards from idx
    for i in range(max(0, idx - 5), min(len(tes), idx + 10)):
        s, e, sf, sub = tes[i]
        if s <= pos <= e:
            results.append((sf, sub))
        elif s > pos:
            break
    return results

# Classify all bQTL
te_classes = []
for i, (_, row) in enumerate(bqtl_all.iterrows()):
    if i % 20000 == 0:
        print(f"  {i}/{len(bqtl_all)}...")
    hits = find_te_at_position(row['chr'], row['pos'], te_intervals)
    if hits:
        # Take the most specific (smallest) TE
        te_classes.append(hits[0])
    else:
        te_classes.append(('non-TE', 'non-TE'))

bqtl_all['te_superfamily'] = [t[0] for t in te_classes]
bqtl_all['te_subfamily'] = [t[1] for t in te_classes]

# ── 4. Compare causal vs non-causal ────────────────────────────────────
print("\n--- TE context comparison ---")

causal = bqtl_all[bqtl_all['is_causal']]
noncausal = bqtl_all[~bqtl_all['is_causal']]

# Superfamily distribution
print("\nSuperfamily distribution:")
print(f"{'Superfamily':>20s}  {'Causal':>8s}  {'Non-causal':>10s}  {'Enrich':>8s}  {'p':>10s}")
print("-" * 65)

enrichment_results = {}
for sf in sorted(bqtl_all['te_superfamily'].unique()):
    n_c = (causal['te_superfamily'] == sf).sum()
    n_nc = (noncausal['te_superfamily'] == sf).sum()
    frac_c = n_c / len(causal) if len(causal) > 0 else 0
    frac_nc = n_nc / len(noncausal) if len(noncausal) > 0 else 0
    enrichment = frac_c / frac_nc if frac_nc > 0 else np.inf

    # Fisher's exact test
    ct = np.array([
        [n_c, len(causal) - n_c],
        [n_nc, len(noncausal) - n_nc]
    ])
    _, p_val = stats.fisher_exact(ct)

    enrichment_results[sf] = {
        'causal_n': int(n_c),
        'causal_frac': float(frac_c),
        'noncausal_n': int(n_nc),
        'noncausal_frac': float(frac_nc),
        'enrichment': float(enrichment),
        'p_value': float(p_val),
    }

    sig = '*' if p_val < 0.05 else ''
    print(f"{sf:>20s}  {frac_c:>8.3f}  {frac_nc:>10.3f}  {enrichment:>8.2f}x  {p_val:>10.4f} {sig}")

# ── 5. Subfamily breakdown for LTR ─────────────────────────────────────
print("\n--- LTR subfamily breakdown ---")
ltr_causal = causal[causal['te_superfamily'] == 'LTR']
ltr_noncausal = noncausal[noncausal['te_superfamily'] == 'LTR']

if len(ltr_causal) > 0:
    print(f"\n{'Subfamily':>15s}  {'Causal':>8s}  {'Non-causal':>10s}  {'Enrich':>8s}")
    for sub in sorted(bqtl_all[bqtl_all['te_superfamily'] == 'LTR']['te_subfamily'].unique()):
        n_c = (ltr_causal['te_subfamily'] == sub).sum()
        n_nc = (ltr_noncausal['te_subfamily'] == sub).sum()
        frac_c = n_c / len(ltr_causal) if len(ltr_causal) > 0 else 0
        frac_nc = n_nc / len(ltr_noncausal) if len(ltr_noncausal) > 0 else 0
        enrichment = frac_c / frac_nc if frac_nc > 0 else np.inf
        print(f"{sub:>15s}  {frac_c:>8.3f}  {frac_nc:>10.3f}  {enrichment:>8.2f}x")

# ── 6. Effect size by TE context ────────────────────────────────────────
print("\n--- Effect size by TE context ---")

# Merge effect sizes from causal data
effect_data = pd.concat([
    causal_ww[['chr', 'pos', 'abs_d']],
    causal_ds[['chr', 'pos', 'abs_d']]
]).drop_duplicates(subset=['chr', 'pos'])

causal_with_effect = causal.merge(effect_data, on=['chr', 'pos'], how='left')

print(f"\n{'TE context':>15s}  {'n':>5s}  {'median |d|':>10s}  {'mean |d|':>10s}")
for sf in ['non-TE', 'LTR', 'DNA', 'MITE', 'LINE']:
    subset = causal_with_effect[causal_with_effect['te_superfamily'] == sf]
    if len(subset) > 0:
        print(f"{sf:>15s}  {len(subset):>5d}  {subset['abs_d'].median():>10.3f}  {subset['abs_d'].mean():>10.3f}")

# Compare TE vs non-TE effect sizes
te_effects = causal_with_effect[causal_with_effect['te_superfamily'] != 'non-TE']['abs_d'].dropna()
nonte_effects = causal_with_effect[causal_with_effect['te_superfamily'] == 'non-TE']['abs_d'].dropna()
if len(te_effects) > 2 and len(nonte_effects) > 2:
    u_stat, u_p = stats.mannwhitneyu(te_effects, nonte_effects, alternative='two-sided')
    print(f"\nTE vs non-TE effect size: U={u_stat:.0f}, p={u_p:.4f}")
    print(f"  TE median: {te_effects.median():.3f}, non-TE median: {nonte_effects.median():.3f}")

# ── 7. Figure ────────────────────────────────────────────────────────────
print("\n--- Generating figure ---")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: TE context pie chart — causal bQTL
ax = axes[0, 0]
sf_counts = causal['te_superfamily'].value_counts()
colors_pie = {'non-TE': '#2ecc71', 'LTR': '#e74c3c', 'DNA': '#3498db',
              'MITE': '#f39c12', 'LINE': '#9b59b6', 'Cent': '#95a5a6',
              'knob': '#1abc9c', 'Simple_repeat': '#e67e22', 'subtelomere': '#34495e'}
pie_colors = [colors_pie.get(sf, '#bdc3c7') for sf in sf_counts.index]
wedges, texts, autotexts = ax.pie(sf_counts, labels=sf_counts.index, autopct='%1.1f%%',
                                   colors=pie_colors, pctdistance=0.85)
for t in texts:
    t.set_fontsize(8)
for t in autotexts:
    t.set_fontsize(7)
ax.set_title('A. TE context of causal bQTL')

# Panel B: Enrichment bar chart
ax = axes[0, 1]
sfs = sorted([sf for sf in enrichment_results if sf != 'non-TE'],
             key=lambda x: enrichment_results[x]['enrichment'], reverse=True)
enrichments = [enrichment_results[sf]['enrichment'] for sf in sfs]
p_vals = [enrichment_results[sf]['p_value'] for sf in sfs]
bar_colors = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in p_vals]
bars = ax.barh(sfs, enrichments, color=bar_colors, alpha=0.7)
ax.axvline(1.0, color='black', linewidth=0.5, linestyle='--')
ax.set_xlabel('Enrichment (causal/non-causal)')
ax.set_title('B. TE superfamily enrichment')
for i, (e, p) in enumerate(zip(enrichments, p_vals)):
    ax.text(max(e, 1) + 0.02, i, f'p={p:.3f}', va='center', fontsize=7)

# Panel C: LTR subfamily breakdown
ax = axes[1, 0]
if len(ltr_causal) > 0:
    ltr_subs = ltr_causal['te_subfamily'].value_counts().head(6)
    ax.bar(ltr_subs.index, ltr_subs.values, color='#e74c3c', alpha=0.7)
    ax.set_xlabel('LTR subfamily')
    ax.set_ylabel('Causal bQTL count')
    ax.set_title('C. LTR subfamily distribution')
    ax.tick_params(axis='x', rotation=30)
else:
    ax.text(0.5, 0.5, 'No LTR-associated causal bQTL', transform=ax.transAxes, ha='center')
    ax.set_title('C. LTR subfamily distribution')

# Panel D: Effect size by TE context
ax = axes[1, 1]
te_cats = ['non-TE', 'LTR', 'DNA', 'MITE']
box_data = [causal_with_effect.loc[causal_with_effect['te_superfamily'] == sf, 'abs_d'].dropna().values
            for sf in te_cats]
valid_cats = [c for c, d in zip(te_cats, box_data) if len(d) > 0]
valid_data = [d for d in box_data if len(d) > 0]
if valid_data:
    bp = ax.boxplot(valid_data, tick_labels=valid_cats, patch_artist=True, showfliers=False)
    te_colors = [colors_pie.get(c, '#bdc3c7') for c in valid_cats]
    for patch, c in zip(bp['boxes'], te_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xlabel('TE context')
    ax.set_ylabel('Effect size |d|')
    if len(te_effects) > 2 and len(nonte_effects) > 2:
        ax.set_title(f'D. Effect size by TE context\n(TE vs non-TE p={u_p:.4f})')
    else:
        ax.set_title('D. Effect size by TE context')

plt.suptitle('Transposable Element Context of Causal bQTL', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / 'te_context.pdf', bbox_inches='tight', dpi=150)
print(f"Saved: {FIG_DIR / 'te_context.pdf'}")

# ── 8. Save results ─────────────────────────────────────────────────────
results = {
    'te_enrichment': enrichment_results,
    'n_causal': int(len(causal)),
    'n_noncausal': int(len(noncausal)),
    'causal_in_te_frac': float((causal['te_superfamily'] != 'non-TE').mean()),
    'noncausal_in_te_frac': float((noncausal['te_superfamily'] != 'non-TE').mean()),
    'te_vs_nonte_effect': {
        'te_median_d': float(te_effects.median()) if len(te_effects) > 0 else None,
        'nonte_median_d': float(nonte_effects.median()) if len(nonte_effects) > 0 else None,
        'p_value': float(u_p) if len(te_effects) > 2 and len(nonte_effects) > 2 else None,
    },
}

with open(RESULTS_DIR / 'te_context.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {RESULTS_DIR / 'te_context.json'}")

bqtl_all[['chr', 'pos', 'is_causal', 'te_superfamily', 'te_subfamily']].to_csv(
    RESULTS_DIR / 'bqtl_te_context.csv', index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Causal bQTL in TE: {(causal['te_superfamily'] != 'non-TE').mean():.1%}")
print(f"Non-causal bQTL in TE: {(noncausal['te_superfamily'] != 'non-TE').mean():.1%}")
for sf in ['LTR', 'DNA', 'MITE', 'LINE']:
    r = enrichment_results.get(sf, {})
    if r:
        sig = '*' if r['p_value'] < 0.05 else ''
        print(f"  {sf}: {r['enrichment']:.2f}x enrichment (p={r['p_value']:.4f}) {sig}")
print("=" * 70)
