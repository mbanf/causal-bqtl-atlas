#!/usr/bin/env python3
"""
60_condition_switching_deepdive.py — Characterizing condition-dependent bQTL switching.

Part of: "Condition-dependent bQTL switching" paper (Banf & Hartwig)
Paper sections: "Switching is a threshold effect", "Target gene drought response
  predicts bQTL switching", "TF family enrichment at switching bQTL"
Pipeline step: 8 of 8

Deep dive into:
1. What characterizes constitutive vs condition-specific functional bQTL?
2. What drives drought-specific vs WW-specific behavior?
3. How does gene expression change relate to bQTL switching?
4. Per-gene story: do genes switch their bQTL or gain/lose them?

Key results (paper numbers):
  WW-only |d| drops 50% under drought (1.989→0.985); DS-only gains 83% (1.052→1.920)
  DS-only target genes more drought-responsive (log2FC +0.413 vs +0.219, p=4.1e-4)
  NAC OR=1.30, TCP OR=1.51 at drought-specific bQTL
  1,039 genes with functional bQTL; 341 both, 386 WW-only, 312 DS-only

Input:
  ww_vs_drought_genotype_bqtl.csv (script 58)
  functional_bqtl_grammar.csv (script 56)
  engelhorn_ww_vs_ds_expression.tsv

Output:
  condition_switching_deepdive.csv — 7,888 paired bQTL classified by category
  figures/fig17_switching_deepdive.pdf — Paper Fig. 3 supplementary panels
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data' / 'processed'
FIG_DIR = BASE / 'figures' / 'paper_figures'

print("=" * 70)
print("CONDITION-DEPENDENT bQTL SWITCHING: DEEP DIVE")
print("=" * 70)

# ── 1. Load data ──
print("\n1. Loading data...")
rdf = pd.read_csv(BASE / 'ww_vs_drought_genotype_bqtl.csv')
rdf_ww = rdf[rdf['condition'] == 'WW'].copy()
rdf_ds = rdf[rdf['condition'] == 'DS'].copy()

# Functional bQTL (FDR < 0.05)
ww_sig = set(rdf_ww[rdf_ww['fdr_mw'] < 0.05].apply(lambda r: (r['chr'], r['pos'], r['gene_id']), axis=1))
ds_sig = set(rdf_ds[rdf_ds['fdr_mw'] < 0.05].apply(lambda r: (r['chr'], r['pos'], r['gene_id']), axis=1))

constitutive = ww_sig & ds_sig
ww_only = ww_sig - ds_sig
ds_only = ds_sig - ww_sig

print(f"  Constitutive: {len(constitutive)}")
print(f"  WW-only: {len(ww_only)}")
print(f"  Drought-only: {len(ds_only)}")

# ── 2. Build paired dataset ──
print("\n2. Building paired bQTL-gene dataset...")
merged = rdf_ww.merge(rdf_ds, on=['chr', 'pos', 'gene_id'], suffixes=('_ww', '_ds'))
print(f"  Paired observations: {len(merged)}")

# Classify each pair
def classify(row):
    key = (row['chr'], row['pos'], row['gene_id'])
    if key in constitutive:
        return 'constitutive'
    elif key in ww_only:
        return 'ww_only'
    elif key in ds_only:
        return 'ds_only'
    else:
        return 'non_functional'

merged['category'] = merged.apply(classify, axis=1)
print(f"  Category counts:")
for cat in ['constitutive', 'ww_only', 'ds_only', 'non_functional']:
    n = (merged['category'] == cat).sum()
    print(f"    {cat}: {n}")

# ── 3. Gene expression changes ──
print("\n3. Gene expression under WW vs drought...")
expr = pd.read_csv(DATA / 'engelhorn_ww_vs_ds_expression.tsv', sep='\t')
expr_cols = [c for c in expr.columns if c not in ['gene_id']]

# Compute mean WW and DS expression per gene
ww_cols = [c for c in expr_cols if '_WW_' in c or c.startswith('B73_WW') or 'WW' in c]
ds_cols = [c for c in expr_cols if '_DS_' in c or c.startswith('B73_DS') or 'DS' in c]

if not ww_cols or not ds_cols:
    # Try to find the right columns
    print(f"  Columns sample: {expr_cols[:10]}")
    # The expression file might have log2FC directly
    if 'log2FoldChange' in expr.columns:
        print("  Using log2FoldChange column directly")
        expr_lookup = dict(zip(expr['gene_id'], expr['log2FoldChange']))
    elif 'log2fc' in expr.columns:
        print("  Using log2fc column directly")
        expr_lookup = dict(zip(expr['gene_id'], expr['log2fc']))
    else:
        # Check what columns we have
        print(f"  All columns: {list(expr.columns)}")
        expr_lookup = None
else:
    expr['ww_mean'] = expr[ww_cols].mean(axis=1)
    expr['ds_mean'] = expr[ds_cols].mean(axis=1)
    expr['log2fc'] = np.log2((expr['ds_mean'] + 1) / (expr['ww_mean'] + 1))
    expr_lookup = dict(zip(expr['gene_id'], expr['log2fc']))

if expr_lookup:
    merged['gene_log2fc'] = merged['gene_id'].map(expr_lookup)
    print(f"  Mapped expression changes: {merged['gene_log2fc'].notna().sum()}/{len(merged)}")

    # Compare gene expression changes across categories
    print("\n  Gene expression log2FC (drought/WW) by bQTL category:")
    for cat in ['constitutive', 'ww_only', 'ds_only', 'non_functional']:
        vals = merged.loc[merged['category'] == cat, 'gene_log2fc'].dropna()
        if len(vals) > 0:
            print(f"    {cat:20s}: median={vals.median():+.3f}, mean={vals.mean():+.3f}, "
                  f"n={len(vals)}, up={( vals > 0.5).sum()}, down={(vals < -0.5).sum()}")

    # Statistical tests: is gene expression change different between categories?
    print("\n  Statistical comparisons (gene log2FC):")
    for a, b in [('ww_only', 'ds_only'), ('constitutive', 'ww_only'),
                  ('constitutive', 'ds_only'), ('ww_only', 'non_functional'),
                  ('ds_only', 'non_functional')]:
        va = merged.loc[merged['category'] == a, 'gene_log2fc'].dropna()
        vb = merged.loc[merged['category'] == b, 'gene_log2fc'].dropna()
        if len(va) > 10 and len(vb) > 10:
            u, p = stats.mannwhitneyu(va, vb, alternative='two-sided')
            print(f"    {a} vs {b}: p={p:.2e}, "
                  f"median {va.median():+.3f} vs {vb.median():+.3f}")

# ── 4. Effect size dynamics ──
print("\n4. Effect size dynamics (paired |d|)...")
for cat in ['constitutive', 'ww_only', 'ds_only', 'non_functional']:
    sub = merged[merged['category'] == cat]
    if len(sub) > 0:
        ww_d = sub['cohens_d_ww'].abs()
        ds_d = sub['cohens_d_ds'].abs()
        delta = ds_d - ww_d
        print(f"  {cat:20s}: WW |d|={ww_d.median():.3f}, DS |d|={ds_d.median():.3f}, "
              f"Δ|d|={delta.median():+.3f}, n={len(sub)}")

# Direction of effect: does the sign flip?
print("\n  Direction consistency (same sign of Cohen's d in both conditions):")
for cat in ['constitutive', 'ww_only', 'ds_only']:
    sub = merged[merged['category'] == cat]
    if len(sub) > 0:
        same_sign = ((sub['cohens_d_ww'] > 0) == (sub['cohens_d_ds'] > 0)).mean()
        print(f"    {cat:20s}: {same_sign:.1%} same direction")

# ── 5. Distance to TSS ──
print("\n5. Genomic architecture by category...")
print("  Distance to TSS:")
for cat in ['constitutive', 'ww_only', 'ds_only', 'non_functional']:
    sub = merged[merged['category'] == cat]
    if len(sub) > 0:
        d = sub['dist_to_tss_ww']
        print(f"    {cat:20s}: median={d.median():.0f}bp, mean={d.mean():.0f}bp")

# TSS proximity comparison
for a, b in [('constitutive', 'non_functional'), ('ww_only', 'ds_only')]:
    va = merged.loc[merged['category'] == a, 'dist_to_tss_ww']
    vb = merged.loc[merged['category'] == b, 'dist_to_tss_ww']
    u, p = stats.mannwhitneyu(va, vb, alternative='two-sided')
    print(f"    {a} vs {b}: p={p:.2e}")

# ── 6. TF motif context ──
print("\n6. TF motif context at switching bQTL...")
grammar = pd.read_csv(BASE / 'functional_bqtl_grammar.csv')
# Join grammar features to merged data
grammar_key = grammar[['chr', 'pos', 'gene_id', 'n_families_at_variant',
                         'n_motifs_at_variant', 'families_at_variant',
                         'n_families_in_region', 'total_motifs_in_region',
                         'mean_redundancy', 'gene_expression', 'n_bqtl_in_promoter']].copy()
merged2 = merged.merge(grammar_key, on=['chr', 'pos', 'gene_id'], how='left')

for feat in ['n_families_at_variant', 'n_motifs_at_variant', 'mean_redundancy',
             'gene_expression', 'n_bqtl_in_promoter']:
    print(f"\n  {feat}:")
    for cat in ['constitutive', 'ww_only', 'ds_only', 'non_functional']:
        vals = merged2.loc[merged2['category'] == cat, feat].dropna()
        if len(vals) > 0:
            print(f"    {cat:20s}: median={vals.median():.2f}, mean={vals.mean():.2f}")

# ── 7. Per-TF family: which motifs are at switching bQTL? ──
print("\n7. TF families at switching bQTL...")
# Explode families_at_variant to get per-family counts
merged2['fam_list'] = merged2['families_at_variant'].fillna('').str.split('|')
exploded = merged2.explode('fam_list')
exploded = exploded[exploded['fam_list'] != '']

for cat in ['constitutive', 'ww_only', 'ds_only']:
    sub = exploded[exploded['category'] == cat]
    counts = sub['fam_list'].value_counts()
    total = len(merged2[merged2['category'] == cat])
    print(f"\n  {cat} (n={total}):")
    for fam in counts.head(8).index:
        n = counts[fam]
        pct = 100 * n / total
        print(f"    {fam:15s}: {n:4d} ({pct:5.1f}%)")

# ── 8. Enrichment: which TF families are OVER-represented in each switching category? ──
print("\n8. TF family enrichment in switching categories...")
# For each TF family, compute odds ratio: (in category / total in category) / (not in category / total not)
all_cats = ['constitutive', 'ww_only', 'ds_only']
families = exploded['fam_list'].value_counts().head(20).index.tolist()

print(f"\n  {'Family':15s} | {'Constit OR':>10s} | {'WW-only OR':>10s} | {'DS-only OR':>10s} | {'Best cat':>10s}")
print("  " + "-" * 70)

enrichment_rows = []
for fam in families:
    ors = {}
    for cat in all_cats:
        # bQTL with this family AND in this category
        cat_with = merged2[(merged2['category'] == cat) &
                           merged2['families_at_variant'].fillna('').str.contains(fam, regex=False)]
        cat_without = merged2[(merged2['category'] == cat) &
                              ~merged2['families_at_variant'].fillna('').str.contains(fam, regex=False)]
        other_with = merged2[(merged2['category'] != cat) &
                             merged2['families_at_variant'].fillna('').str.contains(fam, regex=False)]
        other_without = merged2[(merged2['category'] != cat) &
                                ~merged2['families_at_variant'].fillna('').str.contains(fam, regex=False)]
        a, b, c, d = len(cat_with), len(other_with), len(cat_without), len(other_without)
        OR = (a * d) / (b * c) if b * c > 0 else np.nan
        ors[cat] = OR

    best = max(ors, key=lambda k: abs(np.log(ors[k])) if not np.isnan(ors[k]) else 0)
    print(f"  {fam:15s} | {ors['constitutive']:10.2f} | {ors['ww_only']:10.2f} | {ors['ds_only']:10.2f} | {best:>10s}")
    enrichment_rows.append({'tf_family': fam, **{f'OR_{c}': ors[c] for c in all_cats}})

enr_df = pd.DataFrame(enrichment_rows)

# ── 9. Gene-level view: do genes switch or accumulate functional bQTL? ──
print("\n9. Gene-level switching patterns...")
gene_cats = merged[merged['category'] != 'non_functional'].groupby('gene_id')['category'].apply(set)
gene_types = {
    'constitutive_only': sum(1 for s in gene_cats if s == {'constitutive'}),
    'ww_only_genes': sum(1 for s in gene_cats if s == {'ww_only'}),
    'ds_only_genes': sum(1 for s in gene_cats if s == {'ds_only'}),
    'mixed': sum(1 for s in gene_cats if len(s) > 1),
}
print(f"  Genes with ONLY constitutive bQTL: {gene_types['constitutive_only']}")
print(f"  Genes with ONLY WW-specific bQTL: {gene_types['ww_only_genes']}")
print(f"  Genes with ONLY drought-specific bQTL: {gene_types['ds_only_genes']}")
print(f"  Genes with MIXED switching: {gene_types['mixed']}")

# How many genes have bQTL in both conditions (at ANY position)?
ww_genes = set(rdf_ww[rdf_ww['fdr_mw'] < 0.05]['gene_id'])
ds_genes = set(rdf_ds[rdf_ds['fdr_mw'] < 0.05]['gene_id'])
print(f"\n  Genes with functional bQTL in WW: {len(ww_genes)}")
print(f"  Genes with functional bQTL in DS: {len(ds_genes)}")
print(f"  Genes in both: {len(ww_genes & ds_genes)}")
print(f"  WW-only genes: {len(ww_genes - ds_genes)}")
print(f"  DS-only genes: {len(ds_genes - ww_genes)}")

# ── 10. Gene expression of switching genes ──
if expr_lookup:
    print("\n10. Expression dynamics of switching genes...")
    for label, gene_set in [('WW-only genes', ww_genes - ds_genes),
                             ('DS-only genes', ds_genes - ww_genes),
                             ('Both-condition genes', ww_genes & ds_genes)]:
        fc_vals = [expr_lookup[g] for g in gene_set if g in expr_lookup]
        if fc_vals:
            fc = np.array(fc_vals)
            print(f"  {label:25s}: n={len(fc)}, median log2FC={np.median(fc):+.3f}, "
                  f"mean={np.mean(fc):+.3f}, up={( fc > 0.5).sum()}, down={(fc < -0.5).sum()}")
            # Are DS-only genes more drought-responsive?

    # Direct comparison
    ww_only_fc = [expr_lookup[g] for g in (ww_genes - ds_genes) if g in expr_lookup]
    ds_only_fc = [expr_lookup[g] for g in (ds_genes - ww_genes) if g in expr_lookup]
    if ww_only_fc and ds_only_fc:
        u, p = stats.mannwhitneyu(ww_only_fc, ds_only_fc, alternative='two-sided')
        print(f"\n  WW-only vs DS-only gene log2FC: p={p:.2e}")
        print(f"    WW-only genes: median FC = {np.median(ww_only_fc):+.3f}")
        print(f"    DS-only genes: median FC = {np.median(ds_only_fc):+.3f}")

    # Fraction drought-responsive
    for label, gene_set in [('WW-only', ww_genes - ds_genes),
                             ('DS-only', ds_genes - ww_genes),
                             ('Both', ww_genes & ds_genes)]:
        fc_vals = np.array([expr_lookup[g] for g in gene_set if g in expr_lookup])
        if len(fc_vals) > 0:
            drought_resp = (np.abs(fc_vals) > 1).sum()
            print(f"  {label:10s}: {drought_resp}/{len(fc_vals)} ({100*drought_resp/len(fc_vals):.1f}%) drought-responsive (|log2FC|>1)")

# ── 11. ASE magnitude: what happens to individual bQTL across conditions ──
print("\n11. Individual bQTL behavior across conditions...")

# For paired bQTL, look at how effect changes
paired_func = merged[merged['category'] != 'non_functional'].copy()
paired_func['d_ww'] = paired_func['cohens_d_ww'].abs()
paired_func['d_ds'] = paired_func['cohens_d_ds'].abs()
paired_func['d_ratio'] = paired_func['d_ds'] / paired_func['d_ww'].clip(lower=0.01)

print(f"\n  Effect size ratio (DS/WW) by category:")
for cat in ['constitutive', 'ww_only', 'ds_only']:
    sub = paired_func[paired_func['category'] == cat]
    ratio = sub['d_ratio']
    print(f"    {cat:20s}: median ratio={ratio.median():.2f}, "
          f"mean={ratio.mean():.2f}, n={len(sub)}")

# WW-only bQTL: what's their drought |d|? (should be low)
ww_only_pairs = merged[merged['category'] == 'ww_only']
print(f"\n  WW-only bQTL: WW |d| = {ww_only_pairs['cohens_d_ww'].abs().median():.3f}, "
      f"DS |d| = {ww_only_pairs['cohens_d_ds'].abs().median():.3f}")
print(f"  → Effect drops {(1 - ww_only_pairs['cohens_d_ds'].abs().median() / ww_only_pairs['cohens_d_ww'].abs().median()) * 100:.0f}%")

# DS-only bQTL: what's their WW |d|?
ds_only_pairs = merged[merged['category'] == 'ds_only']
print(f"\n  DS-only bQTL: WW |d| = {ds_only_pairs['cohens_d_ww'].abs().median():.3f}, "
      f"DS |d| = {ds_only_pairs['cohens_d_ds'].abs().median():.3f}")
print(f"  → Effect gains {(ds_only_pairs['cohens_d_ds'].abs().median() / ds_only_pairs['cohens_d_ww'].abs().median() - 1) * 100:.0f}%")

# ── 12. Hybrid-specific effects ──
print("\n12. Do specific hybrids drive switching?")
# Check if n_variant/n_reference changes between conditions
for cat in ['ww_only', 'ds_only', 'constitutive']:
    sub = merged[merged['category'] == cat]
    if len(sub) > 5:
        # Sample sizes should be same (same genotypes tested)
        n_var_diff = (sub['n_variant_ww'] - sub['n_variant_ds']).abs().mean()
        print(f"  {cat:20s}: mean |Δn_variant| = {n_var_diff:.2f} "
              f"(WW n={sub['n_variant_ww'].median():.0f}, DS n={sub['n_variant_ds'].median():.0f})")

# ── 13. Generate comprehensive figure ──
print("\n13. Generating figure...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(22, 11))

# A: Effect size distributions by category
ax = axes[0, 0]
for cat, color, label in [('constitutive', '#7B1FA2', 'Constitutive'),
                           ('ww_only', '#1976D2', 'WW-only'),
                           ('ds_only', '#D32F2F', 'DS-only')]:
    sub = merged[merged['category'] == cat]
    ax.hist(sub['cohens_d_ww'].abs().clip(0, 4), bins=40, alpha=0.5,
            color=color, density=True, label=f'{label} (n={len(sub)})')
ax.set_xlabel("|Cohen's d| under WW")
ax.set_ylabel('Density')
ax.set_title('A. WW effect sizes by switching category')
ax.legend(fontsize=7)

# B: Same for drought
ax = axes[0, 1]
for cat, color, label in [('constitutive', '#7B1FA2', 'Constitutive'),
                           ('ww_only', '#1976D2', 'WW-only'),
                           ('ds_only', '#D32F2F', 'DS-only')]:
    sub = merged[merged['category'] == cat]
    ax.hist(sub['cohens_d_ds'].abs().clip(0, 4), bins=40, alpha=0.5,
            color=color, density=True, label=f'{label} (n={len(sub)})')
ax.set_xlabel("|Cohen's d| under drought")
ax.set_ylabel('Density')
ax.set_title('B. Drought effect sizes by switching category')
ax.legend(fontsize=7)

# C: Paired scatter — WW vs DS effect, colored by category
ax = axes[0, 2]
for cat, color, marker, zorder in [('non_functional', '#BDBDBD', '.', 1),
                                    ('ww_only', '#1976D2', 'o', 3),
                                    ('ds_only', '#D32F2F', 'o', 3),
                                    ('constitutive', '#7B1FA2', 'D', 4)]:
    sub = merged[merged['category'] == cat]
    s = 2 if cat == 'non_functional' else 15
    alpha = 0.1 if cat == 'non_functional' else 0.6
    ax.scatter(sub['cohens_d_ww'].abs().clip(0, 5),
               sub['cohens_d_ds'].abs().clip(0, 5),
               s=s, alpha=alpha, color=color, marker=marker,
               label=f'{cat} ({len(sub)})', zorder=zorder, rasterized=True)
ax.plot([0, 5], [0, 5], 'k--', alpha=0.3)
ax.set_xlabel("|d| Well-watered")
ax.set_ylabel("|d| Drought")
ax.set_title('C. Paired effect sizes by category')
ax.legend(fontsize=6, markerscale=2)

# D: Gene expression change by category
if expr_lookup:
    ax = axes[0, 3]
    cat_data = []
    cat_labels = []
    cat_colors = []
    for cat, color, label in [('constitutive', '#7B1FA2', 'Const.'),
                               ('ww_only', '#1976D2', 'WW-only'),
                               ('ds_only', '#D32F2F', 'DS-only'),
                               ('non_functional', '#BDBDBD', 'Non-func.')]:
        vals = merged.loc[merged['category'] == cat, 'gene_log2fc'].dropna()
        if len(vals) > 0:
            cat_data.append(vals.values)
            cat_labels.append(f'{label}\n(n={len(vals)})')
            cat_colors.append(color)

    parts = ax.violinplot(cat_data, positions=range(len(cat_data)),
                           showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(cat_colors[i])
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(cat_labels)))
    ax.set_xticklabels(cat_labels, fontsize=8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.set_ylabel('Gene expression log2FC (drought/WW)')
    ax.set_title('D. Target gene drought response\nby bQTL switching category')

# E: Distance to TSS by category
ax = axes[1, 0]
cat_data = []
cat_labels = []
cat_colors = []
for cat, color, label in [('constitutive', '#7B1FA2', 'Const.'),
                           ('ww_only', '#1976D2', 'WW-only'),
                           ('ds_only', '#D32F2F', 'DS-only'),
                           ('non_functional', '#BDBDBD', 'Non-func.')]:
    vals = merged.loc[merged['category'] == cat, 'dist_to_tss_ww']
    cat_data.append(vals.values)
    cat_labels.append(f'{label}\n(n={len(vals)})')
    cat_colors.append(color)
bp = ax.boxplot(cat_data, tick_labels=cat_labels, patch_artist=True,
                showfliers=False, widths=0.6)
for patch, color in zip(bp['boxes'], cat_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Distance to TSS (bp)')
ax.set_title('E. Genomic position by switching category')

# F: Gene-level Venn (bar fallback)
ax = axes[1, 1]
bars = ax.bar([0, 1, 2],
              [len(ww_genes - ds_genes), len(ww_genes & ds_genes), len(ds_genes - ww_genes)],
              color=['#1976D2', '#7B1FA2', '#D32F2F'], edgecolor='white', width=0.6)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['WW-only\ngenes', 'Both-condition\ngenes', 'DS-only\ngenes'])
ax.set_ylabel('Number of genes')
ax.set_title(f'F. Gene-level switching\n{len(ww_genes)} WW, {len(ds_genes)} DS')
for bar, val in zip(bars, [len(ww_genes - ds_genes), len(ww_genes & ds_genes), len(ds_genes - ww_genes)]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(val), ha='center', va='bottom', fontsize=10)

# G: TF family enrichment in DS-only vs WW-only
ax = axes[1, 2]
enr_plot = enr_df.copy()
enr_plot['log2_ratio'] = np.log2(enr_plot['OR_ds_only'] / enr_plot['OR_ww_only'])
enr_plot = enr_plot.sort_values('log2_ratio')
colors = ['#D32F2F' if v > 0.1 else '#1976D2' if v < -0.1 else '#757575'
          for v in enr_plot['log2_ratio']]
ax.barh(range(len(enr_plot)), enr_plot['log2_ratio'], color=colors,
        edgecolor='white', height=0.7)
ax.set_yticks(range(len(enr_plot)))
ax.set_yticklabels(enr_plot['tf_family'], fontsize=7)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('log2(OR_drought / OR_WW)')
ax.set_title('G. TF family enrichment:\ndrought-specific vs WW-specific bQTL')

# H: Effect ratio distribution
ax = axes[1, 3]
for cat, color, label in [('ww_only', '#1976D2', 'WW-only'),
                           ('ds_only', '#D32F2F', 'DS-only'),
                           ('constitutive', '#7B1FA2', 'Constitutive')]:
    sub = paired_func[paired_func['category'] == cat]
    ratio = np.log2(sub['d_ratio'].clip(0.01, 100))
    ax.hist(ratio.clip(-4, 4), bins=40, alpha=0.5, color=color,
            density=True, label=f'{label} (n={len(sub)})')
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('log2(|d|_drought / |d|_WW)')
ax.set_ylabel('Density')
ax.set_title('H. Effect size ratio (drought/WW)')
ax.legend(fontsize=7)

plt.suptitle('Condition-dependent bQTL switching: deep dive', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig17_switching_deepdive.pdf', bbox_inches='tight', dpi=150)
plt.savefig(FIG_DIR / 'fig17_switching_deepdive.png', bbox_inches='tight', dpi=150)
print(f"  Saved: figures/fig17_switching_deepdive.pdf")
plt.close()

# ── Save results ──
merged.to_csv(BASE / 'condition_switching_deepdive.csv', index=False)
print(f"\n  Saved: condition_switching_deepdive.csv")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
