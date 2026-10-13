"""
38_chromatin_context.py

Classifies causal bQTL by chromatin context using gene-relative position
and Marand et al. 2021 scATAC ACR data as a proxy for chromatin states:
  - Genic (within gene body)
  - Proximal promoter (within 1kb upstream of TSS)
  - Distal regulatory (1-10kb from gene, in ACR = enhancer-like)
  - Intergenic (>10kb from any gene)

Also tests: do causal bQTL in different chromatin contexts have different
effect sizes? This addresses whether distal regulation is more evolvable.
"""
import sys
sys.path.insert(0, '..')
from pipeline_config import config

import pandas as pd
import numpy as np
from scipy import stats
import gzip
import bisect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from collections import defaultdict

np.random.seed(42)

FIG_DIR = config.figures_dir
RESULTS_DIR = config.results_dir

# ── 1. Load gene annotations ────────────────────────────────────────────
print("=" * 70)
print("CHROMATIN CONTEXT OF CAUSAL bQTL")
print("=" * 70)

print("Loading gene annotations...")
genes = []
with open(config.gff3) as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9 or parts[2] != 'gene':
            continue
        chrom = 'chr' + parts[0] if not parts[0].startswith('chr') else parts[0]
        if chrom not in config.valid_chroms:
            continue
        start = int(parts[3])
        end = int(parts[4])
        strand = parts[6]
        gene_id = parts[8].split('ID=')[1].split(';')[0]
        genes.append((chrom, start, end, strand, gene_id))

genes_df = pd.DataFrame(genes, columns=['chr', 'start', 'end', 'strand', 'gene_id'])
print(f"Genes: {len(genes_df)}")

# Build sorted gene start lists per chromosome for binary search
gene_by_chr = {}
for chrom in config.valid_chroms:
    gdf = genes_df[genes_df['chr'] == chrom].sort_values('start')
    gene_by_chr[chrom] = gdf[['start', 'end', 'strand', 'gene_id']].values.tolist()

# ── 2. Load ACR data ────────────────────────────────────────────────────
print("Loading Marand et al. ACR data...")
acr_file = config.data_dir / 'raw' / 'scatac' / 'tmp_repo' / 'all_ACRs.celltype_calls.overlap.bed.gz'
acr_by_chr = defaultdict(list)  # chr -> [(start, end)]

with gzip.open(acr_file, 'rt') as f:
    header = f.readline()
    for line in f:
        parts = line.strip().split('\t')
        chrom = parts[0]
        if chrom not in config.valid_chroms:
            continue
        start = int(parts[1])
        end = int(parts[2])
        acr_by_chr[chrom].append((start, end))

for chrom in acr_by_chr:
    acr_by_chr[chrom].sort()

total_acrs = sum(len(v) for v in acr_by_chr.values())
print(f"ACRs loaded: {total_acrs}")

# ── 3. Classify genomic positions ──────────────────────────────────────
print("\nClassifying bQTL positions...")

def classify_position(chrom, pos, gene_list, acr_list):
    """Classify a genomic position by chromatin context."""
    # Check if in gene body
    for start, end, strand, gene_id in gene_list:
        if start <= pos <= end:
            return 'genic'

    # Check distance to nearest gene TSS
    min_dist = float('inf')
    nearest_gene = None
    for start, end, strand, gene_id in gene_list:
        tss = start if strand == '+' else end
        dist = abs(pos - tss)
        if dist < min_dist:
            min_dist = dist
            nearest_gene = gene_id

    # Check if in ACR
    in_acr = False
    acr_starts = [a[0] for a in acr_list]
    idx = bisect.bisect_right(acr_starts, pos) - 1
    for i in range(max(0, idx - 2), min(len(acr_list), idx + 3)):
        if acr_list[i][0] <= pos <= acr_list[i][1]:
            in_acr = True
            break

    if min_dist <= 1000:
        return 'proximal_promoter'
    elif min_dist <= 3000:
        return 'proximal_regulatory'
    elif min_dist <= 10000:
        if in_acr:
            return 'distal_enhancer'
        else:
            return 'distal_regulatory'
    else:
        if in_acr:
            return 'intergenic_acr'
        else:
            return 'intergenic'

# Load bQTL data
bqtl_all = pd.read_csv(config.bqtl, usecols=['chr', 'pos'])
causal_ww = pd.read_csv(RESULTS_DIR / 'hybrid_motif_linkage.csv')
causal_ds = pd.read_csv(RESULTS_DIR / 'hybrid_motif_linkage_ds.csv')

causal_positions = set()
for _, row in pd.concat([causal_ww[['chr', 'pos']], causal_ds[['chr', 'pos']]]).iterrows():
    causal_positions.add((row['chr'], row['pos']))

bqtl_all['is_causal'] = bqtl_all.apply(lambda r: (r['chr'], r['pos']) in causal_positions, axis=1)
print(f"Total bQTL: {len(bqtl_all)}, Causal: {bqtl_all['is_causal'].sum()}")

# Classify all bQTL
contexts = []
for i, (_, row) in enumerate(bqtl_all.iterrows()):
    if i % 20000 == 0:
        print(f"  {i}/{len(bqtl_all)}...")
    gene_list = gene_by_chr.get(row['chr'], [])
    acr_list = acr_by_chr.get(row['chr'], [])
    ctx = classify_position(row['chr'], row['pos'], gene_list, acr_list)
    contexts.append(ctx)

bqtl_all['context'] = contexts

# ── 4. Compare causal vs non-causal ────────────────────────────────────
print("\n--- Chromatin context comparison ---")

causal = bqtl_all[bqtl_all['is_causal']]
noncausal = bqtl_all[~bqtl_all['is_causal']]

context_order = ['genic', 'proximal_promoter', 'proximal_regulatory',
                 'distal_enhancer', 'distal_regulatory', 'intergenic_acr', 'intergenic']

print(f"\n{'Context':>25s}  {'Causal':>8s}  {'Non-causal':>10s}  {'Enrich':>8s}  {'p':>10s}")
print("-" * 70)

context_results = {}
for ctx in context_order:
    n_c = (causal['context'] == ctx).sum()
    n_nc = (noncausal['context'] == ctx).sum()
    frac_c = n_c / len(causal) if len(causal) > 0 else 0
    frac_nc = n_nc / len(noncausal) if len(noncausal) > 0 else 0
    enrichment = frac_c / frac_nc if frac_nc > 0 else np.inf

    ct = np.array([[n_c, len(causal) - n_c], [n_nc, len(noncausal) - n_nc]])
    _, p_val = stats.fisher_exact(ct)

    context_results[ctx] = {
        'causal_n': int(n_c),
        'causal_frac': float(frac_c),
        'noncausal_n': int(n_nc),
        'noncausal_frac': float(frac_nc),
        'enrichment': float(enrichment),
        'p_value': float(p_val),
    }

    sig = '*' if p_val < 0.05 else ''
    print(f"{ctx:>25s}  {frac_c:>8.3f}  {frac_nc:>10.3f}  {enrichment:>8.2f}x  {p_val:>10.4f} {sig}")

# ── 5. Effect size by context ──────────────────────────────────────────
print("\n--- Effect size by chromatin context ---")

effect_data = pd.concat([
    causal_ww[['chr', 'pos', 'abs_d']],
    causal_ds[['chr', 'pos', 'abs_d']]
]).drop_duplicates(subset=['chr', 'pos'])

causal_with_effect = causal.merge(effect_data, on=['chr', 'pos'], how='left')

print(f"\n{'Context':>25s}  {'n':>5s}  {'median |d|':>10s}  {'mean |d|':>10s}")
for ctx in context_order:
    subset = causal_with_effect[causal_with_effect['context'] == ctx]
    if len(subset) > 0:
        print(f"{ctx:>25s}  {len(subset):>5d}  {subset['abs_d'].median():>10.3f}  {subset['abs_d'].mean():>10.3f}")

# Kruskal-Wallis test across contexts
groups = [causal_with_effect.loc[causal_with_effect['context'] == ctx, 'abs_d'].dropna().values
          for ctx in context_order]
groups = [g for g in groups if len(g) > 2]
if len(groups) >= 2:
    h_stat, h_p = stats.kruskal(*groups)
    print(f"\nKruskal-Wallis across contexts: H={h_stat:.2f}, p={h_p:.4f}")
else:
    h_stat, h_p = np.nan, np.nan

# Proximal vs distal comparison
proximal = causal_with_effect[causal_with_effect['context'].isin(
    ['proximal_promoter', 'proximal_regulatory'])]['abs_d'].dropna()
distal = causal_with_effect[causal_with_effect['context'].isin(
    ['distal_enhancer', 'distal_regulatory', 'intergenic_acr'])]['abs_d'].dropna()
if len(proximal) > 2 and len(distal) > 2:
    u_stat, u_p = stats.mannwhitneyu(proximal, distal, alternative='two-sided')
    print(f"Proximal vs distal effect: U={u_stat:.0f}, p={u_p:.4f}")
    print(f"  Proximal median: {proximal.median():.3f}, Distal median: {distal.median():.3f}")

# ── 6. Figure ────────────────────────────────────────────────────────────
print("\n--- Generating figure ---")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Panel A: Context distribution comparison
ax = axes[0, 0]
short_labels = ['Genic', 'Prox\nprom', 'Prox\nreg', 'Distal\nenhancer', 'Distal\nreg',
                'Intergenic\nACR', 'Intergenic']
causal_fracs = [context_results[ctx]['causal_frac'] for ctx in context_order]
noncausal_fracs = [context_results[ctx]['noncausal_frac'] for ctx in context_order]
x = np.arange(len(context_order))
w = 0.35
ax.bar(x - w/2, causal_fracs, w, label='Causal', color='coral', alpha=0.7)
ax.bar(x + w/2, noncausal_fracs, w, label='Non-causal', color='steelblue', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=7)
ax.set_ylabel('Fraction')
ax.set_title('A. Chromatin context distribution')
ax.legend(fontsize=8)
# Add significance stars
for i, ctx in enumerate(context_order):
    if context_results[ctx]['p_value'] < 0.05:
        y_max = max(causal_fracs[i], noncausal_fracs[i])
        ax.text(i, y_max + 0.005, '*', ha='center', fontsize=12, color='red')

# Panel B: Effect size by context (boxplot)
ax = axes[0, 1]
box_data = []
box_labels = []
for ctx, label in zip(context_order, short_labels):
    vals = causal_with_effect.loc[causal_with_effect['context'] == ctx, 'abs_d'].dropna().values
    if len(vals) > 2:
        box_data.append(vals)
        box_labels.append(label)

if box_data:
    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showfliers=False)
    colors = ['#2ecc71', '#e74c3c', '#e67e22', '#3498db', '#9b59b6', '#f39c12', '#95a5a6']
    for patch, c in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel('Effect size |d|')
    kw_label = f'H={h_stat:.1f}, p={h_p:.4f}' if not np.isnan(h_stat) else ''
    ax.set_title(f'B. Effect size by context\n({kw_label})')

# Panel C: Enrichment bars
ax = axes[1, 0]
enrichments = [context_results[ctx]['enrichment'] for ctx in context_order]
p_vals = [context_results[ctx]['p_value'] for ctx in context_order]
bar_colors = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in p_vals]
ax.barh(short_labels, enrichments, color=bar_colors, alpha=0.7)
ax.axvline(1.0, color='black', linewidth=0.5, linestyle='--')
ax.set_xlabel('Enrichment (causal / non-causal)')
ax.set_title('C. Context enrichment')

# Panel D: Proximal vs distal summary
ax = axes[1, 1]
categories = ['Genic', 'Proximal\n(≤3kb)', 'Distal\n(3-10kb)', 'Intergenic\n(>10kb)']
# Aggregate
genic_c = context_results['genic']['causal_frac']
prox_c = context_results['proximal_promoter']['causal_frac'] + context_results['proximal_regulatory']['causal_frac']
dist_c = context_results['distal_enhancer']['causal_frac'] + context_results['distal_regulatory']['causal_frac']
inter_c = context_results.get('intergenic_acr', {}).get('causal_frac', 0) + context_results['intergenic']['causal_frac']

genic_nc = context_results['genic']['noncausal_frac']
prox_nc = context_results['proximal_promoter']['noncausal_frac'] + context_results['proximal_regulatory']['noncausal_frac']
dist_nc = context_results['distal_enhancer']['noncausal_frac'] + context_results['distal_regulatory']['noncausal_frac']
inter_nc = context_results.get('intergenic_acr', {}).get('noncausal_frac', 0) + context_results['intergenic']['noncausal_frac']

x2 = np.arange(4)
ax.bar(x2 - w/2, [genic_c, prox_c, dist_c, inter_c], w, label='Causal', color='coral', alpha=0.7)
ax.bar(x2 + w/2, [genic_nc, prox_nc, dist_nc, inter_nc], w, label='Non-causal', color='steelblue', alpha=0.7)
ax.set_xticks(x2)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel('Fraction')
ax.set_title('D. Aggregated chromatin context')
ax.legend(fontsize=8)

plt.suptitle('Chromatin Context of Causal Regulatory Variants', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / 'chromatin_context.pdf', bbox_inches='tight', dpi=150)
print(f"Saved: {FIG_DIR / 'chromatin_context.pdf'}")

# ── 7. Save results ─────────────────────────────────────────────────────
results = {
    'context_enrichment': context_results,
    'n_causal': int(len(causal)),
    'n_noncausal': int(len(noncausal)),
    'kruskal_wallis': {
        'H': float(h_stat) if not np.isnan(h_stat) else None,
        'p': float(h_p) if not np.isnan(h_p) else None,
    },
    'proximal_vs_distal_effect': {
        'proximal_median_d': float(proximal.median()) if len(proximal) > 0 else None,
        'distal_median_d': float(distal.median()) if len(distal) > 0 else None,
        'p_value': float(u_p) if len(proximal) > 2 and len(distal) > 2 else None,
    },
}

with open(RESULTS_DIR / 'chromatin_context.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {RESULTS_DIR / 'chromatin_context.json'}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for ctx in context_order:
    r = context_results[ctx]
    sig = '*' if r['p_value'] < 0.05 else ''
    print(f"  {ctx:>25s}: {r['enrichment']:.2f}x (p={r['p_value']:.4f}) {sig}")
print("=" * 70)
