"""
39_eqtl_proxy.py

Tests whether causal bQTL translate into allele-specific expression (ASE)
differences — the binding→expression causal chain.

Uses the Engelhorn ASE data from the same F1 hybrids to test:
1. Do genes with causal bQTL show larger ASE than genes with non-causal bQTL?
2. Does the bQTL effect size (Cohen's d) predict ASE magnitude?
3. Does bQTL-driven TF motif disruption predict expression change direction?

This serves as an eQTL proxy: if binding variation → expression variation,
then causal bQTL are functional eQTL.
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
print("bQTL → EXPRESSION: ASE AS eQTL PROXY")
print("=" * 70)

# Per-gene ASE summary
gene_summary = pd.read_csv(config.ase_summary)
ase_col = 'mean_abs_log2' if 'mean_abs_log2' in gene_summary.columns else 'mean_abs_ase'
print(f"Gene ASE summary: {len(gene_summary)} genes, using {ase_col}")

# Per-hybrid ASE
ase = pd.read_csv(config.ase)
print(f"Per-hybrid ASE records: {len(ase)}")

# Causal bQTL
causal_ww = pd.read_csv(RESULTS_DIR / 'hybrid_motif_linkage.csv')
causal_ds = pd.read_csv(RESULTS_DIR / 'hybrid_motif_linkage_ds.csv')
causal_genes_ww = set(causal_ww['gene_id'])
causal_genes_ds = set(causal_ds['gene_id'])
causal_genes = causal_genes_ww | causal_genes_ds
print(f"Causal genes: {len(causal_genes)} (WW: {len(causal_genes_ww)}, DS: {len(causal_genes_ds)})")

# Full bQTL
bqtl = pd.read_csv(config.bqtl, usecols=['chr', 'pos'])

# ── 2. Assign bQTL to genes ────────────────────────────────────────────
print("\n--- Assigning all bQTL to genes ---")

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
        gene_id = parts[8].split('ID=')[1].split(';')[0]
        genes.append((chrom, start - config.promoter_size, end, gene_id))

# Genes with any bQTL vs genes with causal bQTL
gene_has_bqtl = set()
for chrom, pstart, end, gene_id in genes:
    bqtl_chr = bqtl[bqtl['chr'] == chrom]['pos'].values
    if np.any((bqtl_chr >= pstart) & (bqtl_chr <= end)):
        gene_has_bqtl.add(gene_id)

print(f"Genes with any bQTL: {len(gene_has_bqtl)}")

# ── 3. Compare ASE: causal vs non-causal bQTL genes ───────────────────
print("\n--- ASE comparison: causal vs non-causal bQTL genes ---")

gene_summary['has_causal'] = gene_summary['gene_id'].isin(causal_genes)
gene_summary['has_any_bqtl'] = gene_summary['gene_id'].isin(gene_has_bqtl)

# Three groups: causal bQTL, non-causal bQTL, no bQTL
causal_ase = gene_summary.loc[gene_summary['has_causal'], ase_col].dropna()
noncausal_bqtl_ase = gene_summary.loc[
    gene_summary['has_any_bqtl'] & ~gene_summary['has_causal'], ase_col].dropna()
no_bqtl_ase = gene_summary.loc[~gene_summary['has_any_bqtl'], ase_col].dropna()

print(f"Causal bQTL genes: n={len(causal_ase)}, median |ASE|={causal_ase.median():.3f}")
print(f"Non-causal bQTL genes: n={len(noncausal_bqtl_ase)}, median |ASE|={noncausal_bqtl_ase.median():.3f}")
print(f"No bQTL genes: n={len(no_bqtl_ase)}, median |ASE|={no_bqtl_ase.median():.3f}")

# Causal vs non-causal bQTL
u1, p1 = stats.mannwhitneyu(causal_ase, noncausal_bqtl_ase, alternative='greater')
print(f"\nCausal vs non-causal bQTL: U={u1:.0f}, p={p1:.2e} (one-sided greater)")

# Causal vs no bQTL
u2, p2 = stats.mannwhitneyu(causal_ase, no_bqtl_ase, alternative='greater')
print(f"Causal vs no bQTL: U={u2:.0f}, p={p2:.2e} (one-sided greater)")

# Non-causal bQTL vs no bQTL
u3, p3 = stats.mannwhitneyu(noncausal_bqtl_ase, no_bqtl_ase, alternative='greater')
print(f"Non-causal bQTL vs no bQTL: U={u3:.0f}, p={p3:.2e}")

# ── 4. bQTL effect size predicts ASE magnitude ────────────────────────
print("\n--- bQTL effect size vs ASE magnitude ---")

# Merge causal bQTL effect sizes with gene ASE
causal_effects = pd.concat([
    causal_ww[['gene_id', 'cohens_d', 'abs_d']],
    causal_ds[['gene_id', 'cohens_d', 'abs_d']]
]).drop_duplicates(subset=['gene_id'])

merged = causal_effects.merge(gene_summary[['gene_id', ase_col]], on='gene_id')
merged = merged.dropna(subset=[ase_col, 'abs_d'])

if len(merged) > 10:
    rho, rho_p = stats.spearmanr(merged['abs_d'], merged[ase_col])
    print(f"bQTL |d| vs |ASE|: rho={rho:.3f}, p={rho_p:.2e} (n={len(merged)})")

    # Bin by effect size
    merged['d_bin'] = pd.qcut(merged['abs_d'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    print("\nBy effect size quartile:")
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        subset = merged[merged['d_bin'] == q]
        print(f"  {q}: n={len(subset)}, median |d|={subset['abs_d'].median():.2f}, "
              f"median |ASE|={subset[ase_col].median():.3f}")
else:
    rho, rho_p = np.nan, np.nan
    print("Insufficient data for effect size correlation")

# ── 5. Per-hybrid concordance ─────────────────────────────────────────
print("\n--- Per-hybrid binding-expression concordance ---")

# For each causal bQTL gene, check: do hybrids with the variant allele
# show higher |ASE| than hybrids with reference allele?
concordance_results = []
for _, row in causal_ww.iterrows():
    gene_id = row['gene_id']
    var_hybrids = set(row['variant_hybrids'].split('|')) if pd.notna(row.get('variant_hybrids')) else set()
    ref_hybrids = set(row['ref_hybrids'].split('|')) if pd.notna(row.get('ref_hybrids')) else set()

    gene_ase = ase[ase['gene_id'] == gene_id]
    if len(gene_ase) == 0:
        continue

    # Map hybrid names: ASE uses B73xFOUNDER, linkage uses FOUNDER
    var_ase = gene_ase[gene_ase['hybrid'].str.replace('B73x', '').isin(var_hybrids)]
    ref_ase = gene_ase[gene_ase['hybrid'].str.replace('B73x', '').isin(ref_hybrids)]

    if len(var_ase) < 2 or len(ref_ase) < 2:
        continue

    var_mean = var_ase['abs_log2_ratio'].mean()
    ref_mean = ref_ase['abs_log2_ratio'].mean()

    concordance_results.append({
        'gene_id': gene_id,
        'var_mean_ase': var_mean,
        'ref_mean_ase': ref_mean,
        'ase_diff': var_mean - ref_mean,
        'bqtl_d': row['abs_d'],
        'concordant': var_mean > ref_mean,
    })

conc_df = pd.DataFrame(concordance_results)
if len(conc_df) > 0:
    n_concordant = conc_df['concordant'].sum()
    n_total = len(conc_df)
    binom_p = stats.binom_test(n_concordant, n_total, 0.5) if hasattr(stats, 'binom_test') else \
              stats.binomtest(n_concordant, n_total, 0.5).pvalue
    print(f"Genes tested: {n_total}")
    print(f"Concordant (variant allele → higher |ASE|): {n_concordant} ({n_concordant/n_total:.1%})")
    print(f"Binomial test p={binom_p:.4f}")

    # Effect size correlation within concordance data
    rho_conc, p_conc = stats.spearmanr(conc_df['bqtl_d'], conc_df['ase_diff'])
    print(f"bQTL |d| vs ASE difference: rho={rho_conc:.3f}, p={p_conc:.4f}")
else:
    n_concordant, n_total, binom_p = 0, 0, 1.0
    rho_conc, p_conc = np.nan, np.nan

# ── 6. Expression change under drought ────────────────────────────────
print("\n--- Causal bQTL genes and drought expression change ---")

expr = pd.read_csv(config.expression, sep='\t')
expr['has_causal'] = expr['gene_id'].isin(causal_genes)
expr['has_any_bqtl'] = expr['gene_id'].isin(gene_has_bqtl)

causal_fc = expr.loc[expr['has_causal'], 'log2fc_DS_vs_WW'].dropna().abs()
noncausal_fc = expr.loc[expr['has_any_bqtl'] & ~expr['has_causal'], 'log2fc_DS_vs_WW'].dropna().abs()
no_bqtl_fc = expr.loc[~expr['has_any_bqtl'], 'log2fc_DS_vs_WW'].dropna().abs()

print(f"Causal bQTL genes: median |log2FC|={causal_fc.median():.3f} (n={len(causal_fc)})")
print(f"Non-causal bQTL genes: median |log2FC|={noncausal_fc.median():.3f} (n={len(noncausal_fc)})")
print(f"No bQTL genes: median |log2FC|={no_bqtl_fc.median():.3f} (n={len(no_bqtl_fc)})")

u_fc, p_fc = stats.mannwhitneyu(causal_fc, noncausal_fc, alternative='two-sided')
print(f"Causal vs non-causal |log2FC|: U={u_fc:.0f}, p={p_fc:.4f}")

# ── 7. Figure ────────────────────────────────────────────────────────────
print("\n--- Generating figure ---")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: ASE distribution by gene category
ax = axes[0, 0]
data_box = [no_bqtl_ase.values, noncausal_bqtl_ase.values, causal_ase.values]
bp = ax.boxplot(data_box, tick_labels=['No bQTL', 'Non-causal\nbQTL', 'Causal\nbQTL'],
               patch_artist=True, showfliers=False)
colors = ['#95a5a6', 'steelblue', 'coral']
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
ax.set_ylabel(f'Mean |ASE| ({ase_col})')
ax.set_title(f'A. ASE by bQTL category\n(causal vs non-causal p={p1:.2e})')
# Add significance brackets
y_max = max(causal_ase.quantile(0.75), noncausal_bqtl_ase.quantile(0.75))
ax.plot([2, 2, 3, 3], [y_max+0.05, y_max+0.08, y_max+0.08, y_max+0.05], 'k-', linewidth=1)
sig_label = '***' if p1 < 0.001 else '**' if p1 < 0.01 else '*' if p1 < 0.05 else 'ns'
ax.text(2.5, y_max+0.09, sig_label, ha='center', fontsize=10)

# Panel B: bQTL effect size vs ASE
ax = axes[0, 1]
if len(merged) > 10:
    ax.scatter(merged['abs_d'], merged[ase_col], alpha=0.3, s=10, color='mediumpurple')
    z = np.polyfit(merged['abs_d'], merged[ase_col], 1)
    x_line = np.linspace(merged['abs_d'].min(), merged['abs_d'].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'r--', linewidth=2)
    ax.set_xlabel('bQTL effect size |d|')
    ax.set_ylabel(f'Gene mean |ASE|')
    ax.set_title(f'B. Binding effect predicts expression\n(rho={rho:.3f}, p={rho_p:.2e})')
else:
    ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')

# Panel C: Per-hybrid concordance
ax = axes[1, 0]
if len(conc_df) > 0:
    ax.hist(conc_df['ase_diff'], bins=30, color='teal', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(conc_df['ase_diff'].mean(), color='red', linestyle='-', linewidth=2,
              label=f'mean={conc_df["ase_diff"].mean():.3f}')
    ax.set_xlabel('ASE difference (variant - reference allele)')
    ax.set_ylabel('Number of genes')
    ax.set_title(f'C. Binding-expression concordance\n({n_concordant}/{n_total} concordant, p={binom_p:.4f})')
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')

# Panel D: Expression change under drought
ax = axes[1, 1]
data_fc = [no_bqtl_fc.values, noncausal_fc.values, causal_fc.values]
bp2 = ax.boxplot(data_fc, tick_labels=['No bQTL', 'Non-causal\nbQTL', 'Causal\nbQTL'],
                patch_artist=True, showfliers=False)
for patch, c in zip(bp2['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
ax.set_ylabel('|log2FC| drought vs well-watered')
ax.set_title(f'D. Drought response by bQTL category\n(p={p_fc:.4f})')

plt.suptitle('bQTL → Expression: The Causal Chain', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / 'eqtl_proxy.pdf', bbox_inches='tight', dpi=150)
print(f"Saved: {FIG_DIR / 'eqtl_proxy.pdf'}")

# ── 8. Save results ─────────────────────────────────────────────────────
results = {
    'ase_comparison': {
        'causal_median': float(causal_ase.median()),
        'noncausal_bqtl_median': float(noncausal_bqtl_ase.median()),
        'no_bqtl_median': float(no_bqtl_ase.median()),
        'causal_vs_noncausal_p': float(p1),
        'causal_vs_no_bqtl_p': float(p2),
    },
    'effect_size_correlation': {
        'rho': float(rho) if not np.isnan(rho) else None,
        'p': float(rho_p) if not np.isnan(rho_p) else None,
        'n': int(len(merged)) if len(merged) > 0 else 0,
    },
    'concordance': {
        'n_tested': int(n_total),
        'n_concordant': int(n_concordant),
        'concordance_rate': float(n_concordant / n_total) if n_total > 0 else None,
        'binom_p': float(binom_p),
    },
    'drought_expression': {
        'causal_median_fc': float(causal_fc.median()),
        'noncausal_median_fc': float(noncausal_fc.median()),
        'p_value': float(p_fc),
    },
}

with open(RESULTS_DIR / 'eqtl_proxy.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {RESULTS_DIR / 'eqtl_proxy.json'}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"ASE: causal {causal_ase.median():.3f} > non-causal {noncausal_bqtl_ase.median():.3f} (p={p1:.2e})")
if not np.isnan(rho):
    print(f"Effect size correlation: rho={rho:.3f} (p={rho_p:.2e})")
if n_total > 0:
    print(f"Concordance: {n_concordant}/{n_total} ({n_concordant/n_total:.1%}, p={binom_p:.4f})")
print(f"Drought |log2FC|: causal {causal_fc.median():.3f} vs non-causal {noncausal_fc.median():.3f} (p={p_fc:.4f})")
print("=" * 70)
