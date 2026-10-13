"""
36_bqtl_epistasis.py

Tests whether multiple bQTL affecting the same gene combine additively
or show epistatic (non-additive) interactions.

Approach: For genes with multiple bQTL in their promoter, compare the
observed ASE in hybrids carrying multiple variant alleles vs the sum
of individual bQTL effects. Deviation from additivity = epistasis.

Also tests: does regulatory complexity (# bQTL per gene) predict
expression variability?
"""
import sys
sys.path.insert(0, '..')
from pipeline_config import config

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

np.random.seed(42)

FIG_DIR = config.figures_dir
RESULTS_DIR = config.results_dir

# ── 1. Load data ────────────────────────────────────────────────────────
print("=" * 70)
print("bQTL EPISTASIS: ADDITIVE vs NON-ADDITIVE EFFECTS")
print("=" * 70)

# Full bQTL positions
bqtl = pd.read_csv(config.bqtl, usecols=['chr', 'pos'])
print(f"Total bQTL: {len(bqtl)}")

# Load GFF3 for gene coordinates
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

# ── 2. Assign bQTL to gene promoters ───────────────────────────────────
print("\n--- Assigning bQTL to gene promoters ---")
PROMOTER = config.promoter_size  # 3000bp

# Build promoter regions (strand-aware)
genes_df['prom_start'] = np.where(
    genes_df['strand'] == '+',
    genes_df['start'] - PROMOTER,
    genes_df['start']
)
genes_df['prom_end'] = np.where(
    genes_df['strand'] == '+',
    genes_df['end'],
    genes_df['end'] + PROMOTER
)

# Count bQTL per gene
gene_bqtl_count = {}
gene_bqtl_positions = {}
for chrom in config.valid_chroms:
    bqtl_chr = bqtl[bqtl['chr'] == chrom]['pos'].values
    genes_chr = genes_df[genes_df['chr'] == chrom]
    for _, g in genes_chr.iterrows():
        mask = (bqtl_chr >= g['prom_start']) & (bqtl_chr <= g['prom_end'])
        n = mask.sum()
        if n > 0:
            gene_bqtl_count[g['gene_id']] = n
            gene_bqtl_positions[g['gene_id']] = bqtl_chr[mask].tolist()

count_dist = pd.Series(gene_bqtl_count)
print(f"Genes with >=1 bQTL: {len(count_dist)}")
print(f"Genes with >=2 bQTL: {(count_dist >= 2).sum()}")
print(f"Genes with >=5 bQTL: {(count_dist >= 5).sum()}")
print(f"Genes with >=10 bQTL: {(count_dist >= 10).sum()}")

# ── 3. Load ASE data ───────────────────────────────────────────────────
print("\n--- Loading ASE data ---")
ase = pd.read_csv(config.ase)
print(f"ASE records: {len(ase)}")
print(f"ASE columns: {list(ase.columns[:10])}")

# Load genotype data
geno = pd.read_csv(config.genotypes, sep='\t')
print(f"Genotype matrix: {geno.shape}")
print(f"Genotype columns: {list(geno.columns[:6])}")

# ── 4. Per-gene ASE variability vs bQTL burden ─────────────────────────
print("\n--- ASE variability vs bQTL burden ---")

# Gene-level ASE summary
gene_summary = pd.read_csv(config.ase_summary)
print(f"Gene ASE summary: {len(gene_summary)}")

# Detect column names (different naming conventions)
ase_mean_col = 'mean_abs_ase' if 'mean_abs_ase' in gene_summary.columns else 'mean_abs_log2'
ase_std_col = 'std_abs_ase' if 'std_abs_ase' in gene_summary.columns else 'median_abs_log2'
print(f"Using ASE columns: {ase_mean_col}, {ase_std_col}")

# Add bQTL count
gene_summary['n_bqtl'] = gene_summary['gene_id'].map(gene_bqtl_count).fillna(0).astype(int)
has_bqtl = gene_summary[gene_summary['n_bqtl'] > 0].copy()

# Bin by bQTL count
bins = [0, 1, 2, 5, 10, 100]
labels = ['1', '2', '3-5', '6-10', '>10']
has_bqtl['bqtl_bin'] = pd.cut(has_bqtl['n_bqtl'], bins=bins, labels=labels, right=True)

print("\nbQTL burden vs ASE variability:")
for label in labels:
    subset = has_bqtl[has_bqtl['bqtl_bin'] == label]
    if len(subset) > 0:
        print(f"  {label:>5s} bQTL: n={len(subset):5d}, "
              f"mean |ASE|={subset[ase_mean_col].mean():.3f}, "
              f"std ASE={subset[ase_std_col].mean():.3f}")

# Spearman correlation
rho_ase, p_ase = stats.spearmanr(has_bqtl['n_bqtl'], has_bqtl[ase_mean_col])
rho_var, p_var = stats.spearmanr(has_bqtl['n_bqtl'], has_bqtl[ase_std_col])
print(f"\nBurden vs |ASE|: rho={rho_ase:.3f}, p={p_ase:.2e}")
print(f"Burden vs ASE std: rho={rho_var:.3f}, p={p_var:.2e}")

# ── 5. Epistasis test: multi-bQTL genes ────────────────────────────────
print("\n--- Epistasis test ---")

# For genes with >=2 bQTL, test: does having more variant alleles at
# different bQTL positions produce proportionally larger ASE?

# Get per-hybrid genotypes at each bQTL position
founder_cols = [c for c in geno.columns if c not in ['chr', 'pos', 'Position', 'ref']]
print(f"Founders in genotype data: {len(founder_cols)}")

# Map ASE hybrid names (B73xFOUNDER) to founder names
ase_hybrids = ase['hybrid'].unique()
hybrid_to_founder = {}
for h in ase_hybrids:
    # Extract founder from 'B73xFOUNDER' format
    founder = h.replace('B73x', '') if 'x' in h else h
    if founder in founder_cols:
        hybrid_to_founder[h] = founder
print(f"ASE hybrids mapped to founders: {len(hybrid_to_founder)}")

# For multi-bQTL genes, count variant alleles per hybrid
multi_bqtl_genes = [g for g, n in gene_bqtl_count.items() if n >= 2]
print(f"Testing {len(multi_bqtl_genes)} multi-bQTL genes")

# Pre-index ASE by gene for speed
ase_by_gene = ase.groupby('gene_id')

# Filter to genes that exist in ASE data
genes_with_ase = set(ase_by_gene.groups.keys())
multi_bqtl_with_ase = [g for g in multi_bqtl_genes if g in genes_with_ase]
print(f"Multi-bQTL genes with ASE data: {len(multi_bqtl_with_ase)}")

epistasis_results = []
tested = 0
for gi, gene_id in enumerate(multi_bqtl_with_ase):
    if gi % 1000 == 0:
        print(f"  {gi}/{len(multi_bqtl_with_ase)}...")
    positions = gene_bqtl_positions[gene_id]

    # Get genotypes at these positions
    gene_geno = geno[geno['pos'].isin(positions)]
    if len(gene_geno) == 0:
        continue

    # Count variant alleles per founder
    # Genotype format: '0|0' = reference, nucleotide letter = variant
    variant_counts = {}
    for founder in founder_cols:
        founder_vals = gene_geno[founder].values
        n_variant = int(np.sum([v not in ('0|0', '0', None) and pd.notna(v) for v in founder_vals]))
        variant_counts[founder] = n_variant

    # Get ASE for this gene per hybrid
    if gene_id not in ase_by_gene.groups:
        continue
    gene_ase = ase_by_gene.get_group(gene_id)

    # Build per-hybrid data: map ASE hybrid to founder genotype
    data_points = []
    for _, row in gene_ase.iterrows():
        h = row['hybrid']
        if h in hybrid_to_founder:
            founder = hybrid_to_founder[h]
            data_points.append({
                'hybrid': h,
                'founder': founder,
                'n_variant_bqtl': variant_counts[founder],
                'ase': abs(row['abs_log2_ratio']) if pd.notna(row.get('abs_log2_ratio')) else np.nan,
            })

    if len(data_points) < 5:
        continue

    dp = pd.DataFrame(data_points).dropna()
    if dp['n_variant_bqtl'].nunique() < 2:
        continue

    tested += 1
    # Test: Spearman correlation between variant count and |ASE|
    rho, p = stats.spearmanr(dp['n_variant_bqtl'], dp['ase'])

    # Test additivity: linear model
    if len(dp) >= 5:
        slope, intercept, r_val, p_lin, se = stats.linregress(dp['n_variant_bqtl'], dp['ase'])
    else:
        slope, r_val, p_lin = np.nan, np.nan, np.nan

    epistasis_results.append({
        'gene_id': gene_id,
        'n_bqtl': gene_bqtl_count[gene_id],
        'n_hybrids_tested': len(dp),
        'rho': rho,
        'rho_p': p,
        'slope': slope,
        'r_squared': r_val**2 if not np.isnan(r_val) else np.nan,
        'slope_p': p_lin,
    })

print(f"Genes tested for dose-response: {tested}")

epi_df = pd.DataFrame(epistasis_results)
print(f"Genes with epistasis test: {len(epi_df)}")

if len(epi_df) > 0:
    # How many show significant dose-response?
    sig_pos = ((epi_df['rho_p'] < 0.05) & (epi_df['rho'] > 0)).sum()
    sig_neg = ((epi_df['rho_p'] < 0.05) & (epi_df['rho'] < 0)).sum()
    print(f"Significant positive dose-response (additive): {sig_pos}")
    print(f"Significant negative dose-response (saturation): {sig_neg}")
    print(f"Non-significant: {len(epi_df) - sig_pos - sig_neg}")

    # Mean slope: positive = additive, near-zero or negative at high doses = sub-additive
    mean_slope = epi_df['slope'].mean()
    print(f"\nMean slope (ASE per variant allele): {mean_slope:.4f}")
    print(f"Median rho: {epi_df['rho'].median():.3f}")

    # One-sample t-test: is mean slope significantly different from 0?
    t_stat, t_p = stats.ttest_1samp(epi_df['slope'].dropna(), 0)
    print(f"Slope vs 0: t={t_stat:.3f}, p={t_p:.4f}")

# ── 6. Regulatory complexity vs expression noise ───────────────────────
print("\n--- Regulatory complexity vs expression noise ---")

# Load expression data
expr = pd.read_csv(config.expression, sep='\t')
print(f"Expression data: {len(expr)} genes")

# Compute expression CV (coefficient of variation) across conditions
# Look for columns with expression values
expr_cols = [c for c in expr.columns if c != 'gene_id' and 'gene' not in c.lower()]
if len(expr_cols) >= 2:
    expr['expr_mean'] = expr[expr_cols].mean(axis=1)
    expr['expr_std'] = expr[expr_cols].std(axis=1)
    expr['expr_cv'] = expr['expr_std'] / expr['expr_mean'].replace(0, np.nan)

    expr['n_bqtl'] = expr['gene_id'].map(gene_bqtl_count).fillna(0).astype(int)
    has_expr = expr[(expr['n_bqtl'] > 0) & (expr['expr_mean'] > 0)].copy()

    rho_cv, p_cv = stats.spearmanr(has_expr['n_bqtl'], has_expr['expr_cv'])
    print(f"bQTL burden vs expression CV: rho={rho_cv:.3f}, p={p_cv:.2e}")
    print(f"(positive = more bQTL → more variable expression)")
else:
    rho_cv, p_cv = np.nan, np.nan
    has_expr = pd.DataFrame()
    print("Insufficient expression columns for CV analysis")

# ── 7. Figure ────────────────────────────────────────────────────────────
print("\n--- Generating figure ---")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: bQTL burden distribution
ax = axes[0, 0]
counts = pd.Series(gene_bqtl_count)
ax.hist(counts, bins=np.arange(0.5, min(counts.max(), 35) + 1.5),
        color='steelblue', alpha=0.7, edgecolor='white')
ax.set_xlabel('bQTL per gene')
ax.set_ylabel('Number of genes')
ax.set_title(f'A. bQTL burden distribution\n(n={len(counts)} genes)')
ax.set_yscale('log')

# Panel B: bQTL burden vs |ASE|
ax = axes[0, 1]
if len(has_bqtl) > 0:
    for label in labels:
        subset = has_bqtl[has_bqtl['bqtl_bin'] == label]
        if len(subset) > 0:
            ax.scatter(subset['n_bqtl'] + np.random.normal(0, 0.1, len(subset)),
                      subset[ase_mean_col], alpha=0.1, s=5, color='steelblue')

    # Add binned means
    bin_means = has_bqtl.groupby('bqtl_bin')[ase_mean_col].agg(['mean', 'sem', 'count'])
    bin_centers = [1, 2, 4, 8, 15][:len(bin_means)]
    ax.errorbar(bin_centers, bin_means['mean'], yerr=bin_means['sem'] * 1.96,
               fmt='ro-', markersize=6, linewidth=2, capsize=4)
    ax.set_xlabel('bQTL per gene')
    ax.set_ylabel('Mean |ASE|')
    ax.set_title(f'B. bQTL burden vs ASE magnitude\n(rho={rho_ase:.3f}, p={p_ase:.2e})')

# Panel C: Epistasis — dose-response slopes
ax = axes[1, 0]
if len(epi_df) > 0:
    ax.hist(epi_df['slope'].dropna(), bins=30, color='mediumpurple', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(epi_df['slope'].mean(), color='red', linestyle='-', linewidth=2, label=f'mean={mean_slope:.3f}')
    ax.set_xlabel('Dose-response slope (|ASE| per variant bQTL)')
    ax.set_ylabel('Number of genes')
    ax.set_title(f'C. Epistasis test: dose-response slopes\n(t={t_stat:.2f}, p={t_p:.4f})')
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
    ax.set_title('C. Epistasis test')

# Panel D: bQTL burden vs expression CV
ax = axes[1, 1]
if len(has_expr) > 0 and not np.isnan(rho_cv):
    bins_cv = [0, 1, 2, 5, 10, 100]
    labels_cv = ['1', '2', '3-5', '6-10', '>10']
    has_expr['bqtl_bin'] = pd.cut(has_expr['n_bqtl'], bins=bins_cv, labels=labels_cv, right=True)

    box_data = [has_expr.loc[has_expr['bqtl_bin'] == l, 'expr_cv'].dropna().values
                for l in labels_cv]
    box_data = [d for d in box_data if len(d) > 0]
    valid_labels = [l for l, d in zip(labels_cv,
                    [has_expr.loc[has_expr['bqtl_bin'] == l, 'expr_cv'].dropna().values
                     for l in labels_cv]) if len(d) > 0]

    if box_data:
        bp = ax.boxplot(box_data, tick_labels=valid_labels, patch_artist=True,
                       showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor('lightcoral')
            patch.set_alpha(0.6)
        ax.set_xlabel('bQTL per gene')
        ax.set_ylabel('Expression CV')
        ax.set_title(f'D. bQTL burden vs expression variability\n(rho={rho_cv:.3f}, p={p_cv:.2e})')
else:
    ax.text(0.5, 0.5, 'Insufficient expression data', transform=ax.transAxes, ha='center')
    ax.set_title('D. bQTL burden vs expression CV')

plt.suptitle('bQTL Epistasis and Regulatory Complexity', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / 'bqtl_epistasis.pdf', bbox_inches='tight', dpi=150)
print(f"Saved: {FIG_DIR / 'bqtl_epistasis.pdf'}")

# ── 8. Save results ─────────────────────────────────────────────────────
results = {
    'burden_distribution': {
        'n_genes_with_bqtl': len(gene_bqtl_count),
        'genes_with_2plus': int((count_dist >= 2).sum()),
        'genes_with_5plus': int((count_dist >= 5).sum()),
        'genes_with_10plus': int((count_dist >= 10).sum()),
        'mean_bqtl_per_gene': float(count_dist.mean()),
        'max_bqtl_per_gene': int(count_dist.max()),
    },
    'burden_vs_ase': {
        'rho': float(rho_ase),
        'p': float(p_ase),
    },
    'burden_vs_ase_variance': {
        'rho': float(rho_var),
        'p': float(p_var),
    },
    'burden_vs_expression_cv': {
        'rho': float(rho_cv) if not np.isnan(rho_cv) else None,
        'p': float(p_cv) if not np.isnan(p_cv) else None,
    },
    'epistasis': {
        'n_genes_tested': len(epi_df),
        'n_additive': int(sig_pos) if len(epi_df) > 0 else 0,
        'n_saturating': int(sig_neg) if len(epi_df) > 0 else 0,
        'mean_slope': float(mean_slope) if len(epi_df) > 0 and not np.isnan(mean_slope) else None,
        'slope_ttest_p': float(t_p) if len(epi_df) > 0 and not np.isnan(t_p) else None,
    },
}

with open(RESULTS_DIR / 'bqtl_epistasis.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {RESULTS_DIR / 'bqtl_epistasis.json'}")

if len(epi_df) > 0:
    epi_df.to_csv(RESULTS_DIR / 'bqtl_epistasis_genes.csv', index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"bQTL burden: {len(gene_bqtl_count)} genes, mean {count_dist.mean():.1f}/gene")
print(f"Burden vs |ASE|: rho={rho_ase:.3f} (p={p_ase:.2e})")
if len(epi_df) > 0:
    print(f"Epistasis: {sig_pos} additive, {sig_neg} saturating, {len(epi_df)-sig_pos-sig_neg} neutral")
    print(f"Mean dose-response slope: {mean_slope:.4f} (p={t_p:.4f})")
if not np.isnan(rho_cv):
    print(f"Burden vs expression CV: rho={rho_cv:.3f} (p={p_cv:.2e})")
print("=" * 70)
