"""
33_gwas_overlap.py

Cross-reference causal bQTL with maize GWAS atlas (NGDC, 43K hits, 322 traits).
Tests whether causal bQTL are enriched near agronomic GWAS hits compared to
non-causal bQTL.

Approach:
  1. Load NGDC GWAS atlas (B73v4/NAM5 coordinates)
  2. For each bQTL, find nearest GWAS hit and count GWAS hits within window
  3. Compare causal vs non-causal: distance to nearest GWAS, hit density
  4. Trait enrichment: which traits are overrepresented near causal bQTL?
  5. Gene-level: which causal genes overlap GWAS loci?
"""
import sys
sys.path.insert(0, '..')
from pipeline_config import config

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter, defaultdict
import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

FIG_DIR = config.figures_dir
RESULTS_DIR = config.results_dir
GWAS_FILE = config.data_dir / 'raw' / 'gwas' / 'ngdc_gwas_maize.txt.gz'

WINDOW = 100_000  # 100kb window for overlap

# ── 1. Load GWAS data ────────────────────────────────────────────────────
print("=" * 70)
print("GWAS OVERLAP ANALYSIS")
print("=" * 70)

print("Loading NGDC GWAS atlas...")
gwas_records = []
with gzip.open(GWAS_FILE, 'rt') as f:
    header = f.readline().strip().split('\t')
    chr_col = header.index('Chr')
    pos_col = header.index('Pos')
    trait_col = header.index('Trait')
    pval_col = header.index('P-value')
    ref_col = header.index('Ref ver')

    for line in f:
        parts = line.strip().split('\t')
        if len(parts) <= max(chr_col, pos_col, trait_col):
            continue
        try:
            chrom = int(parts[chr_col])
            pos = int(parts[pos_col])
            trait = parts[trait_col]
            pval = parts[pval_col] if pval_col < len(parts) else ''
            ref = parts[ref_col] if ref_col < len(parts) else ''
            gwas_records.append({
                'chr': f'chr{chrom}',
                'pos': pos,
                'trait': trait,
                'pval': pval,
                'ref': ref,
            })
        except (ValueError, IndexError):
            continue

gwas = pd.DataFrame(gwas_records)
print(f"GWAS hits loaded: {len(gwas)}")
print(f"Unique traits: {gwas['trait'].nunique()}")
print(f"Reference versions: {gwas['ref'].value_counts().head()}")

# Build chromosome-indexed GWAS positions for fast lookup
gwas_by_chr = defaultdict(list)
for _, row in gwas.iterrows():
    gwas_by_chr[row['chr']].append((row['pos'], row['trait']))

# Sort by position for binary search
for c in gwas_by_chr:
    gwas_by_chr[c].sort(key=lambda x: x[0])

# ── 2. Load bQTL data ────────────────────────────────────────────────────
print("\nLoading bQTL data...")
bqtl = pd.read_csv(config.bqtl)
causal = pd.read_csv(RESULTS_DIR / 'causal_bqtl_728.csv')
causal_positions = set(zip(causal['chr'], causal['pos']))
print(f"All bQTL: {len(bqtl)}, Causal: {len(causal_positions)}")

# ── 3. Find GWAS hits near each bQTL ─────────────────────────────────────
print(f"\n--- Finding GWAS hits within {WINDOW//1000}kb of each bQTL ---")

import bisect

def find_gwas_near(chrom, pos, window, gwas_chr):
    """Find GWAS hits within window of position."""
    if chrom not in gwas_chr:
        return []
    positions = gwas_chr[chrom]
    # Binary search for start
    pos_list = [p[0] for p in positions]
    left = bisect.bisect_left(pos_list, pos - window)
    right = bisect.bisect_right(pos_list, pos + window)
    return positions[left:right]

results = []
for _, row in bqtl.iterrows():
    chrom, pos = row['chr'], row['pos']
    nearby = find_gwas_near(chrom, pos, WINDOW, gwas_by_chr)
    is_causal = (chrom, pos) in causal_positions

    min_dist = min(abs(pos - g[0]) for g in nearby) if nearby else np.nan
    n_hits = len(nearby)
    traits = set(g[1] for g in nearby)

    results.append({
        'chr': chrom,
        'pos': pos,
        'is_causal': is_causal,
        'n_gwas_hits': n_hits,
        'min_dist': min_dist,
        'n_traits': len(traits),
        'traits': '|'.join(sorted(traits)) if traits else '',
    })

res_df = pd.DataFrame(results)
causal_res = res_df[res_df['is_causal']]
noncausal_res = res_df[~res_df['is_causal']]

# ── 4. Compare causal vs non-causal ──────────────────────────────────────
print("\n--- GWAS hit density comparison ---")
print(f"{'Metric':<30} {'Causal':>10} {'Non-causal':>12}")

has_hit_causal = (causal_res['n_gwas_hits'] > 0).mean()
has_hit_nc = (noncausal_res['n_gwas_hits'] > 0).mean()
print(f"{'Has GWAS hit within 100kb':<30} {has_hit_causal:>10.3f} {has_hit_nc:>12.3f}")

mean_hits_c = causal_res['n_gwas_hits'].mean()
mean_hits_nc = noncausal_res['n_gwas_hits'].mean()
print(f"{'Mean GWAS hits (100kb)':<30} {mean_hits_c:>10.2f} {mean_hits_nc:>12.2f}")

# Only among those with hits
c_with = causal_res[causal_res['n_gwas_hits'] > 0]
nc_with = noncausal_res[noncausal_res['n_gwas_hits'] > 0]
if len(c_with) > 0 and len(nc_with) > 0:
    mw_stat, mw_p = stats.mannwhitneyu(c_with['min_dist'], nc_with['min_dist'], alternative='two-sided')
    print(f"{'Median min distance (bp)':<30} {c_with['min_dist'].median():>10.0f} {nc_with['min_dist'].median():>12.0f}")
    print(f"{'Mann-Whitney p (distance)':<30} {mw_p:>23.4e}")

# Hit density test
mw2_stat, mw2_p = stats.mannwhitneyu(causal_res['n_gwas_hits'], noncausal_res['n_gwas_hits'], alternative='two-sided')
print(f"{'Mann-Whitney p (density)':<30} {mw2_p:>23.4e}")

# Enrichment ratio
enrichment = has_hit_causal / has_hit_nc if has_hit_nc > 0 else np.inf
print(f"\nGWAS overlap enrichment: {enrichment:.2f}×")

# ── 5. Trait enrichment ──────────────────────────────────────────────────
print("\n--- Trait enrichment near causal bQTL ---")

# Count trait occurrences near causal vs non-causal
causal_traits = Counter()
noncausal_traits = Counter()

for _, row in res_df.iterrows():
    if not row['traits']:
        continue
    traits = row['traits'].split('|')
    if row['is_causal']:
        causal_traits.update(traits)
    else:
        noncausal_traits.update(traits)

# Normalize
total_causal = len(causal_res)
total_nc = len(noncausal_res)

all_traits = set(causal_traits.keys()) | set(noncausal_traits.keys())
trait_enrich = []
for t in all_traits:
    c_rate = causal_traits[t] / total_causal
    nc_rate = noncausal_traits[t] / total_nc
    fold = c_rate / nc_rate if nc_rate > 0 else np.inf
    # Fisher's exact test
    a = causal_traits[t]
    b = total_causal - a
    c = noncausal_traits[t]
    d = total_nc - c
    _, fisher_p = stats.fisher_exact([[a, b], [c, d]])
    trait_enrich.append({
        'trait': t,
        'causal_count': a,
        'noncausal_count': c,
        'causal_rate': c_rate,
        'noncausal_rate': nc_rate,
        'fold_enrichment': fold,
        'fisher_p': fisher_p,
    })

trait_df = pd.DataFrame(trait_enrich).sort_values('fisher_p')

# FDR correction
from statsmodels.stats.multitest import multipletests
if len(trait_df) > 0:
    _, fdr, _, _ = multipletests(trait_df['fisher_p'], method='fdr_bh')
    trait_df['fdr'] = fdr

sig_traits = trait_df[trait_df['fdr'] < 0.05] if 'fdr' in trait_df.columns else trait_df.head(0)
print(f"Significant trait enrichments (FDR<0.05): {len(sig_traits)}")
if len(sig_traits) > 0:
    print(sig_traits[['trait', 'fold_enrichment', 'fdr']].head(20).to_string(index=False))

print(f"\nTop 15 traits by fold enrichment (p<0.05):")
top = trait_df[trait_df['fisher_p'] < 0.05].nlargest(15, 'fold_enrichment')
if len(top) > 0:
    for _, row in top.iterrows():
        print(f"  {row['trait']:<40} {row['fold_enrichment']:>6.2f}× (p={row['fisher_p']:.4e}, n={row['causal_count']})")

# ── 6. Gene-level GWAS overlap ───────────────────────────────────────────
print("\n--- Gene-level GWAS overlap ---")
causal_with_gwas = causal_res[causal_res['n_gwas_hits'] > 0]
print(f"Causal bQTL near GWAS hits: {len(causal_with_gwas)}/{len(causal_res)} ({len(causal_with_gwas)/len(causal_res)*100:.1f}%)")

# Map back to genes
causal_genes_gwas = causal.merge(
    causal_with_gwas[['chr', 'pos', 'n_gwas_hits', 'min_dist', 'traits']],
    on=['chr', 'pos'], how='inner'
)
print(f"Causal genes near GWAS: {causal_genes_gwas['gene_id'].nunique()}")

# ── 7. Trait categories ──────────────────────────────────────────────────
print("\n--- Trait category summary ---")
# Categorize traits
categories = {
    'flowering': ['anthesis', 'silking', 'flowering', 'photoperiod', 'DTA', 'DTS'],
    'height': ['height', 'stature', 'internode'],
    'leaf': ['leaf', 'angle', 'width'],
    'yield': ['yield', 'kernel', 'grain', 'ear', 'cob'],
    'stress': ['drought', 'stress', 'resistance', 'disease', 'heat'],
    'quality': ['protein', 'starch', 'oil', 'sugar', 'carotenoid'],
    'root': ['root'],
}

cat_counts_causal = Counter()
cat_counts_nc = Counter()

for _, row in res_df.iterrows():
    if not row['traits']:
        continue
    for trait in row['traits'].split('|'):
        trait_lower = trait.lower()
        matched = False
        for cat, keywords in categories.items():
            if any(kw in trait_lower for kw in keywords):
                if row['is_causal']:
                    cat_counts_causal[cat] += 1
                else:
                    cat_counts_nc[cat] += 1
                matched = True
                break
        if not matched:
            if row['is_causal']:
                cat_counts_causal['other'] += 1
            else:
                cat_counts_nc['other'] += 1

print(f"{'Category':<15} {'Causal':>8} {'Non-causal':>12} {'Enrichment':>12}")
for cat in sorted(set(list(cat_counts_causal.keys()) + list(cat_counts_nc.keys()))):
    c = cat_counts_causal[cat]
    nc = cat_counts_nc[cat]
    c_rate = c / total_causal
    nc_rate = nc / total_nc
    fold = c_rate / nc_rate if nc_rate > 0 else np.inf
    print(f"{cat:<15} {c:>8} {nc:>12} {fold:>12.2f}×")

# ── 8. Figure ─────────────────────────────────────────────────────────────
print("\n--- Generating figure ---")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: GWAS hit density
ax = axes[0, 0]
hit_bins = [-0.5, 0.5, 1.5, 4.5, 9.5, 49.5, 500]
hit_labels = ['0', '1', '2-4', '5-9', '10-49', '50+']
causal_binned = pd.cut(causal_res['n_gwas_hits'], bins=hit_bins, labels=hit_labels)
nc_binned = pd.cut(noncausal_res['n_gwas_hits'], bins=hit_bins, labels=hit_labels)
x = np.arange(len(hit_labels))
c_fracs = [causal_binned.value_counts(normalize=True).get(l, 0) for l in hit_labels]
nc_fracs = [nc_binned.value_counts(normalize=True).get(l, 0) for l in hit_labels]
width = 0.35
ax.bar(x - width/2, c_fracs, width, label='Causal', color='crimson', alpha=0.7)
ax.bar(x + width/2, nc_fracs, width, label='Non-causal', color='steelblue', alpha=0.7)
ax.set_xlabel('GWAS hits within 100kb')
ax.set_ylabel('Fraction')
ax.set_xticks(x)
ax.set_xticklabels(hit_labels, fontsize=8)
ax.set_title(f'A. GWAS hit density\n(p={mw2_p:.2e})')
ax.legend(fontsize=8)

# Panel B: Distance to nearest GWAS hit (among those with hits)
ax = axes[0, 1]
if len(c_with) > 0:
    ax.hist(c_with['min_dist'] / 1000, bins=30, alpha=0.6, density=True, color='crimson', label='Causal', edgecolor='white')
if len(nc_with) > 0:
    ax.hist(nc_with['min_dist'] / 1000, bins=30, alpha=0.4, density=True, color='steelblue', label='Non-causal', edgecolor='white')
ax.set_xlabel('Distance to nearest GWAS hit (kb)')
ax.set_ylabel('Density')
ax.set_title('B. Distance to nearest GWAS hit')
ax.legend(fontsize=8)

# Panel C: Top enriched traits
ax = axes[1, 0]
top_traits = trait_df[trait_df['fisher_p'] < 0.05].nlargest(12, 'fold_enrichment')
if len(top_traits) > 0:
    # Truncate long names
    names = [t[:35] for t in top_traits['trait']]
    folds = top_traits['fold_enrichment'].values
    colors = ['#e74c3c' if f > 1 else '#3498db' for f in folds]
    ax.barh(range(len(names)), folds, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Fold enrichment (causal vs non-causal)')
ax.set_title('C. Top enriched GWAS traits')

# Panel D: Trait category enrichment
ax = axes[1, 1]
cats = sorted(set(list(cat_counts_causal.keys()) + list(cat_counts_nc.keys())))
cat_folds = []
for cat in cats:
    c_rate = cat_counts_causal[cat] / total_causal
    nc_rate = cat_counts_nc[cat] / total_nc
    cat_folds.append(c_rate / nc_rate if nc_rate > 0 else 0)
colors = ['#e74c3c' if f > 1 else '#3498db' for f in cat_folds]
ax.barh(cats, cat_folds, color=colors, alpha=0.7)
ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Fold enrichment')
ax.set_title('D. Enrichment by trait category')

plt.suptitle('GWAS Atlas Overlap Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / 'gwas_overlap.pdf', bbox_inches='tight', dpi=150)
print(f"Saved: {FIG_DIR / 'gwas_overlap.pdf'}")

# ── 9. Save results ──────────────────────────────────────────────────────
# Save trait enrichment table
trait_df.to_csv(RESULTS_DIR / 'gwas_trait_enrichment.csv', index=False)

# Save gene-GWAS overlap
causal_genes_gwas.to_csv(RESULTS_DIR / 'causal_gwas_overlap.csv', index=False)

# Save summary
summary = {
    'gwas_hits_total': len(gwas),
    'gwas_traits': int(gwas['trait'].nunique()),
    'window_bp': WINDOW,
    'causal_with_gwas_hit': int(len(causal_with_gwas)),
    'causal_total': int(len(causal_res)),
    'causal_gwas_pct': float(len(causal_with_gwas) / len(causal_res) * 100),
    'noncausal_with_gwas_hit_pct': float(has_hit_nc * 100),
    'enrichment': float(enrichment),
    'density_p': float(mw2_p),
    'sig_traits': len(sig_traits),
    'top_enriched_traits': trait_df.head(10)[['trait', 'fold_enrichment', 'fisher_p']].to_dict('records'),
}

with open(RESULTS_DIR / 'gwas_overlap.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {RESULTS_DIR / 'gwas_overlap.json'}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"GWAS hits: {len(gwas)} across {gwas['trait'].nunique()} traits")
print(f"Causal bQTL near GWAS: {len(causal_with_gwas)}/{len(causal_res)} ({len(causal_with_gwas)/len(causal_res)*100:.1f}%)")
print(f"Non-causal near GWAS: {has_hit_nc*100:.1f}%")
print(f"Enrichment: {enrichment:.2f}×")
print(f"Significant trait enrichments: {len(sig_traits)}")
