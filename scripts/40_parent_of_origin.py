"""
40_parent_of_origin.py

Parent-of-origin allele analysis for causal bQTL.

For each causal bQTL, determines whether the B73 (reference/mother) or the
NAM founder (father) allele is the one with higher TF binding. Then asks:

Module 1 — Classification:
  Split causal bQTL into B73-bound vs founder-bound based on Cohen's d sign
  and genotype direction. Summarize proportions, effect sizes.

Module 2 — Pathway coordination:
  Within each parental group, test whether genes in the same pathway
  (GO BP, KEGG) tend to share the same parental allele direction.
  Binomial test per pathway: is one parent overrepresented?

Module 3 — Spatial clustering:
  Test whether same-direction causal bQTL cluster on chromosomes.
  Sliding window and nearest-neighbor analysis.

Module 4 — Condition comparison:
  Compare B73-bound vs founder-bound proportions between WW and DS.
  Do drought-specific causal bQTL favor one parent's allele?

Addresses Thomas's question: is there coordinated allelic selection
across genes in the same pathway?
"""
import sys
sys.path.insert(0, '..')
from pipeline_config import config

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import re

np.random.seed(42)

FIG_DIR = config.figures_dir
RESULTS_DIR = config.results_dir

# ── 1. Load data ────────────────────────────────────────────────────────
print("=" * 70)
print("PARENT-OF-ORIGIN ALLELE ANALYSIS")
print("=" * 70)

# Causal bQTL (WW and DS)
causal_ww = pd.read_csv(RESULTS_DIR / 'causal_bqtl_728.csv')
causal_ds = pd.read_csv(RESULTS_DIR / 'causal_bqtl_ds.csv')
print(f"Causal bQTL: {len(causal_ww)} WW, {len(causal_ds)} DS")

# Genotypes: tells us which hybrids carry ref (B73) vs alt (founder) allele
genotypes = pd.read_csv(config.genotypes, sep='\t')
print(f"Genotype matrix: {genotypes.shape[0]} positions × {genotypes.shape[1]-3} hybrids")

# ASE data: per-hybrid, per-gene allele-specific expression
ase_ww = pd.read_csv(config.ase)
print(f"ASE data (WW): {len(ase_ww)} entries")

# Gene annotations for pathway analysis
gene_atlas = pd.read_csv(RESULTS_DIR / 'gene_atlas_728.csv')
print(f"Gene atlas: {len(gene_atlas)} genes with annotations")

# Try loading DS gene atlas if available
ds_atlas_path = RESULTS_DIR / 'gene_atlas_ds.csv'
if ds_atlas_path.exists():
    gene_atlas_ds = pd.read_csv(ds_atlas_path)
    print(f"Gene atlas (DS): {len(gene_atlas_ds)} genes")
else:
    gene_atlas_ds = None
    print("No DS gene atlas found — DS pathway analysis will use WW annotations")

# ── 2. Module 1: Classify B73-bound vs founder-bound ───────────────────
print("\n" + "=" * 70)
print("MODULE 1: PARENT-OF-ORIGIN CLASSIFICATION")
print("=" * 70)

# Cohen's d interpretation in the causal bQTL framework:
# The test compares ASE between hybrids carrying the VARIANT allele vs REF allele.
# variant_hybrids = hybrids where the NAM founder carries the alt allele at this bQTL
# ref_hybrids = hybrids where the NAM founder carries the ref (B73) allele
#
# Cohen's d < 0: variant-carrying hybrids have LOWER ASE → variant disrupts binding
#   → B73 allele is the "better bound" allele
# Cohen's d > 0: variant-carrying hybrids have HIGHER ASE → variant enhances binding
#   → Founder allele is the "better bound" allele
#
# Note: ASE = |log2(B73/NAM)|, so higher ASE = more allelic imbalance.
# But the DIRECTION of imbalance matters too. We need to check:
# does the variant allele increase B73-biased or NAM-biased expression?

def classify_parent_of_origin(causal_df, ase_df, label='WW'):
    """
    For each causal bQTL, determine which parent's allele drives binding.

    Logic:
    - For each bQTL, look at ASE in variant vs ref hybrids
    - If variant hybrids show higher |ASE| (Cohen's d < 0 in our framework),
      the variant disrupts binding → B73 allele is the functional one
    - If variant hybrids show lower |ASE|, the variant enhances binding
      → founder allele is the functional one

    But more precisely: we check the SIGNED ASE (log2 B73/NAM ratio).
    - Variant hybrids where NAM founder carries alt allele:
      if log2(B73/NAM) > 0 → B73 expressed more → founder allele disrupted
      if log2(B73/NAM) < 0 → NAM expressed more → founder allele enhanced
    """
    results = []

    for _, row in causal_df.iterrows():
        gene = row['gene_id']
        d = row['cohens_d']

        # Parse hybrid lists
        variant_hybs = str(row['variant_hybrids']).split('|') if pd.notna(row.get('variant_hybrids')) else []
        ref_hybs = str(row['ref_hybrids']).split('|') if pd.notna(row.get('ref_hybrids')) else []

        # Get ASE for this gene across hybrids
        gene_ase = ase_df[ase_df['gene_id'] == gene].copy()
        if len(gene_ase) == 0:
            results.append({
                'chr': row['chr'], 'pos': row['pos'], 'gene_id': gene,
                'cohens_d': d, 'parent_bound': 'unknown',
                'n_variant': len(variant_hybs), 'n_ref': len(ref_hybs),
                'mean_ase_variant': np.nan, 'mean_ase_ref': np.nan,
                'mean_signed_ase_variant': np.nan, 'mean_signed_ase_ref': np.nan,
            })
            continue

        # Extract hybrid name from "B73xFOUNDER" format
        gene_ase['founder'] = gene_ase['hybrid'].str.replace('B73x', '', regex=False)

        # Signed ASE: log2(B73/NAM) — positive means B73 allele more expressed
        # This is already in the 'log2_ratio' column
        var_ase = gene_ase[gene_ase['founder'].isin(variant_hybs)]
        ref_ase = gene_ase[gene_ase['founder'].isin(ref_hybs)]

        mean_signed_var = var_ase['log2_ratio'].mean() if len(var_ase) > 0 else np.nan
        mean_signed_ref = ref_ase['log2_ratio'].mean() if len(ref_ase) > 0 else np.nan
        mean_abs_var = var_ase['abs_log2_ratio'].mean() if len(var_ase) > 0 else np.nan
        mean_abs_ref = ref_ase['abs_log2_ratio'].mean() if len(ref_ase) > 0 else np.nan

        # Classification logic:
        # The variant allele is in the NAM founder genome.
        # Cohen's d < 0: variant hybrids have MORE allelic imbalance
        #   → the variant allele DISRUPTS binding → B73 allele is the better-bound one
        # Cohen's d > 0: variant hybrids have LESS imbalance or opposite direction
        #   → the variant allele CREATES/ENHANCES binding → founder allele is better
        #
        # More nuanced: check signed ASE direction in variant hybrids.
        # If variant hybrids have log2(B73/NAM) > 0 → B73 allele expressed more
        #   → consistent with founder allele being disrupted → B73-bound
        # If variant hybrids have log2(B73/NAM) < 0 → NAM allele expressed more
        #   → founder allele is the functional one → founder-bound

        if pd.notna(mean_signed_var) and len(var_ase) >= 2:
            if mean_signed_var > 0:
                parent = 'B73'  # B73 allele expressed more in variant hybrids
            elif mean_signed_var < 0:
                parent = 'founder'  # NAM founder allele expressed more
            else:
                parent = 'balanced'
        elif d < 0:
            parent = 'B73'  # fallback to Cohen's d direction
        elif d > 0:
            parent = 'founder'
        else:
            parent = 'balanced'

        results.append({
            'chr': row['chr'], 'pos': row['pos'], 'gene_id': gene,
            'cohens_d': d, 'abs_d': row['abs_d'],
            'parent_bound': parent,
            'n_variant': len(variant_hybs), 'n_ref': len(ref_hybs),
            'mean_ase_variant': mean_abs_var, 'mean_ase_ref': mean_abs_ref,
            'mean_signed_ase_variant': mean_signed_var,
            'mean_signed_ase_ref': mean_signed_ref,
            'variant_hybrids': row.get('variant_hybrids', ''),
            'ref_hybrids': row.get('ref_hybrids', ''),
        })

    return pd.DataFrame(results)

print("\nClassifying WW causal bQTL...")
poo_ww = classify_parent_of_origin(causal_ww, ase_ww, 'WW')

# Load DS ASE if available
ase_ds_path = config.data_dir / 'processed' / 'engelhorn_ase_ds.csv'
if ase_ds_path.exists():
    ase_ds = pd.read_csv(ase_ds_path)
    print("Classifying DS causal bQTL...")
    poo_ds = classify_parent_of_origin(causal_ds, ase_ds, 'DS')
else:
    print("No DS ASE data — skipping DS classification")
    poo_ds = None

# Summary statistics
for label, df in [('WW', poo_ww), ('DS', poo_ds)]:
    if df is None:
        continue
    counts = df['parent_bound'].value_counts()
    total = len(df)
    print(f"\n{label} Parent-of-origin classification:")
    for parent, n in counts.items():
        print(f"  {parent}: {n} ({100*n/total:.1f}%)")

    # Effect size comparison
    for parent in ['B73', 'founder']:
        sub = df[df['parent_bound'] == parent]
        if len(sub) > 0:
            print(f"  {parent}-bound mean |d|: {sub['abs_d'].mean():.3f} "
                  f"(median {sub['abs_d'].median():.3f})")

# Save classification
poo_ww.to_csv(RESULTS_DIR / 'parent_of_origin_ww.csv', index=False)
print(f"\nSaved: {RESULTS_DIR / 'parent_of_origin_ww.csv'}")
if poo_ds is not None:
    poo_ds.to_csv(RESULTS_DIR / 'parent_of_origin_ds.csv', index=False)
    print(f"Saved: {RESULTS_DIR / 'parent_of_origin_ds.csv'}")

# ── 3. Module 2: Pathway coordination ──────────────────────────────────
print("\n" + "=" * 70)
print("MODULE 2: PATHWAY COORDINATION")
print("=" * 70)

def test_pathway_coordination(poo_df, gene_atlas_df, pathway_col, label):
    """
    For each pathway, test whether one parent's allele is overrepresented
    among the causal bQTL targeting genes in that pathway.
    """
    # Merge parent classification with gene annotations
    merged = poo_df.merge(gene_atlas_df[['gene_id', pathway_col]].dropna(),
                          on='gene_id', how='inner')
    merged = merged[merged['parent_bound'].isin(['B73', 'founder'])]

    if len(merged) == 0:
        print(f"  No {label} genes with {pathway_col} annotations")
        return pd.DataFrame()

    # Overall B73 fraction as null expectation
    overall_b73_frac = (merged['parent_bound'] == 'B73').mean()
    print(f"  Overall B73 fraction: {overall_b73_frac:.3f} "
          f"({(merged['parent_bound'] == 'B73').sum()}/{len(merged)})")

    # Explode pathway annotations (they may be semicolon-separated)
    rows = []
    for _, r in merged.iterrows():
        pathways = str(r[pathway_col]).split(';')
        for pw in pathways:
            pw = pw.strip()
            if pw and pw != 'nan':
                rows.append({'gene_id': r['gene_id'], 'parent_bound': r['parent_bound'],
                             'pathway': pw, 'abs_d': r.get('abs_d', np.nan)})

    if not rows:
        print(f"  No valid pathway annotations found")
        return pd.DataFrame()

    pw_df = pd.DataFrame(rows)

    # Test each pathway with ≥3 genes
    results = []
    for pw, grp in pw_df.groupby('pathway'):
        n_total = len(grp.drop_duplicates('gene_id'))
        if n_total < 3:
            continue
        n_b73 = (grp.drop_duplicates('gene_id')['parent_bound'] == 'B73').sum()
        n_founder = n_total - n_b73

        # Binomial test: is B73 fraction different from overall?
        binom_p = stats.binomtest(n_b73, n_total, overall_b73_frac).pvalue

        results.append({
            'pathway': pw, 'n_genes': n_total,
            'n_b73': n_b73, 'n_founder': n_founder,
            'b73_fraction': n_b73 / n_total,
            'expected_b73_frac': overall_b73_frac,
            'binom_p': binom_p,
            'dominant_parent': 'B73' if n_b73 > n_founder else 'founder',
        })

    if not results:
        print(f"  No pathways with ≥3 genes")
        return pd.DataFrame()

    result_df = pd.DataFrame(results).sort_values('binom_p')

    # FDR correction
    from statsmodels.stats.multitest import multipletests
    _, fdr, _, _ = multipletests(result_df['binom_p'], method='fdr_bh')
    result_df['fdr'] = fdr

    n_sig = (result_df['fdr'] < 0.05).sum()
    n_nom = (result_df['binom_p'] < 0.05).sum()
    print(f"  {len(result_df)} pathways tested, {n_nom} nominal p<0.05, {n_sig} FDR<0.05")

    if n_nom > 0:
        print(f"\n  Top pathways (nominal p<0.05):")
        for _, r in result_df[result_df['binom_p'] < 0.05].head(10).iterrows():
            print(f"    {r['pathway']}: {r['n_b73']}/{r['n_genes']} B73 "
                  f"(exp {r['expected_b73_frac']:.2f}), p={r['binom_p']:.4f}, "
                  f"FDR={r['fdr']:.4f}, dominant={r['dominant_parent']}")

    return result_df

# Test with GO biological process
print("\n--- GO Biological Process ---")
go_results_ww = test_pathway_coordination(poo_ww, gene_atlas, 'go_biological', 'WW')

# Test with KEGG pathways
print("\n--- KEGG Pathways ---")
kegg_results_ww = test_pathway_coordination(poo_ww, gene_atlas, 'kegg_pathway', 'WW')

# Test with functional categories
print("\n--- Functional Categories ---")
func_results_ww = test_pathway_coordination(poo_ww, gene_atlas, 'func_category', 'WW')

# Test with signaling layers
print("\n--- Signaling Layers ---")
layer_results_ww = test_pathway_coordination(poo_ww, gene_atlas, 'signaling_layer', 'WW')

# Save pathway results
if len(go_results_ww) > 0:
    go_results_ww.to_csv(RESULTS_DIR / 'parent_origin_go_coordination.csv', index=False)
if len(kegg_results_ww) > 0:
    kegg_results_ww.to_csv(RESULTS_DIR / 'parent_origin_kegg_coordination.csv', index=False)

# ── 4. Module 3: Spatial clustering ────────────────────────────────────
print("\n" + "=" * 70)
print("MODULE 3: SPATIAL CLUSTERING")
print("=" * 70)

def test_spatial_clustering(poo_df, label='WW'):
    """
    Test whether same-parent causal bQTL cluster spatially on chromosomes.
    """
    df = poo_df[poo_df['parent_bound'].isin(['B73', 'founder'])].copy()
    df = df.sort_values(['chr', 'pos'])

    # Nearest-neighbor analysis: for each bQTL, distance to nearest same-parent bQTL
    # vs nearest opposite-parent bQTL
    same_dists = []
    diff_dists = []

    for chrom in df['chr'].unique():
        chr_df = df[df['chr'] == chrom].reset_index(drop=True)
        if len(chr_df) < 2:
            continue

        positions = chr_df['pos'].values
        parents = chr_df['parent_bound'].values

        for i in range(len(chr_df)):
            min_same = np.inf
            min_diff = np.inf
            for j in range(len(chr_df)):
                if i == j:
                    continue
                dist = abs(positions[i] - positions[j])
                if parents[i] == parents[j]:
                    min_same = min(min_same, dist)
                else:
                    min_diff = min(min_diff, dist)

            if min_same < np.inf:
                same_dists.append(min_same)
            if min_diff < np.inf:
                diff_dists.append(min_diff)

    same_dists = np.array(same_dists)
    diff_dists = np.array(diff_dists)

    print(f"\n{label} Nearest-neighbor distances:")
    print(f"  Same-parent:  median {np.median(same_dists)/1e6:.2f} Mb "
          f"(mean {np.mean(same_dists)/1e6:.2f} Mb, n={len(same_dists)})")
    print(f"  Diff-parent:  median {np.median(diff_dists)/1e6:.2f} Mb "
          f"(mean {np.mean(diff_dists)/1e6:.2f} Mb, n={len(diff_dists)})")

    mw_stat, mw_p = stats.mannwhitneyu(same_dists, diff_dists, alternative='less')
    print(f"  Mann-Whitney (same < diff): U={mw_stat:.0f}, p={mw_p:.4e}")

    # Runs test: are same-parent bQTL clustered along each chromosome?
    runs_results = []
    for chrom in df['chr'].unique():
        chr_df = df[df['chr'] == chrom].sort_values('pos')
        if len(chr_df) < 5:
            continue

        # Binary sequence: 1 = B73, 0 = founder
        binary = (chr_df['parent_bound'] == 'B73').astype(int).values
        n_b73 = binary.sum()
        n_founder = len(binary) - n_b73

        if n_b73 == 0 or n_founder == 0:
            continue

        # Count runs
        runs = 1
        for k in range(1, len(binary)):
            if binary[k] != binary[k-1]:
                runs += 1

        # Expected runs under random arrangement
        n = len(binary)
        exp_runs = 1 + 2 * n_b73 * n_founder / n
        var_runs = (2 * n_b73 * n_founder * (2 * n_b73 * n_founder - n)) / (n**2 * (n - 1))

        if var_runs > 0:
            z = (runs - exp_runs) / np.sqrt(var_runs)
            p = 2 * stats.norm.sf(abs(z))  # two-sided
            runs_results.append({
                'chr': chrom, 'n_bqtl': n, 'n_b73': n_b73, 'n_founder': n_founder,
                'observed_runs': runs, 'expected_runs': exp_runs,
                'z_score': z, 'p_value': p,
            })

    runs_df = pd.DataFrame(runs_results)
    if len(runs_df) > 0:
        print(f"\n  Wald-Wolfowitz runs test per chromosome:")
        for _, r in runs_df.iterrows():
            sig = '*' if r['p_value'] < 0.05 else ''
            direction = 'clustered' if r['z_score'] < 0 else 'dispersed'
            print(f"    {r['chr']}: {r['observed_runs']} runs "
                  f"(exp {r['expected_runs']:.1f}), z={r['z_score']:.2f}, "
                  f"p={r['p_value']:.3f} {sig} [{direction}]")

        # Fisher combined p-value across chromosomes
        fisher_stat = -2 * np.sum(np.log(runs_df['p_value']))
        fisher_p = stats.chi2.sf(fisher_stat, 2 * len(runs_df))
        print(f"\n  Fisher combined p-value: {fisher_p:.4e}")
        n_clustered = (runs_df['z_score'] < 0).sum()
        print(f"  Chromosomes with clustering trend: {n_clustered}/{len(runs_df)}")

    return same_dists, diff_dists, runs_df

same_d, diff_d, runs_ww = test_spatial_clustering(poo_ww, 'WW')

# ── 5. Module 4: Condition comparison ──────────────────────────────────
print("\n" + "=" * 70)
print("MODULE 4: CONDITION COMPARISON (WW vs DS)")
print("=" * 70)

if poo_ds is not None:
    # Compare parent-of-origin proportions
    ww_counts = poo_ww['parent_bound'].value_counts()
    ds_counts = poo_ds['parent_bound'].value_counts()

    print("\nParent-of-origin by condition:")
    print(f"  {'':20s} {'WW':>10s} {'DS':>10s}")
    for parent in ['B73', 'founder', 'balanced', 'unknown']:
        ww_n = ww_counts.get(parent, 0)
        ds_n = ds_counts.get(parent, 0)
        ww_pct = 100 * ww_n / len(poo_ww)
        ds_pct = 100 * ds_n / len(poo_ds) if len(poo_ds) > 0 else 0
        print(f"  {parent:20s} {ww_n:>5d} ({ww_pct:4.1f}%) {ds_n:>5d} ({ds_pct:4.1f}%)")

    # Chi-square test for B73 vs founder proportions
    ww_b73 = ww_counts.get('B73', 0)
    ww_founder = ww_counts.get('founder', 0)
    ds_b73 = ds_counts.get('B73', 0)
    ds_founder = ds_counts.get('founder', 0)

    if ww_b73 + ww_founder > 0 and ds_b73 + ds_founder > 0:
        contingency = [[ww_b73, ww_founder], [ds_b73, ds_founder]]
        chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
        print(f"\n  Chi-square (B73 vs founder × WW vs DS): "
              f"chi2={chi2:.2f}, p={chi_p:.4f}")

    # Overlap analysis: shared positions
    ww_pos = set(zip(poo_ww['chr'], poo_ww['pos']))
    ds_pos = set(zip(poo_ds['chr'], poo_ds['pos']))
    shared = ww_pos & ds_pos

    if shared:
        print(f"\n  Shared positions: {len(shared)}")
        # For shared positions, do they switch parent direction?
        ww_shared = poo_ww[poo_ww.apply(lambda r: (r['chr'], r['pos']) in shared, axis=1)]
        ds_shared = poo_ds[poo_ds.apply(lambda r: (r['chr'], r['pos']) in shared, axis=1)]

        merged_shared = ww_shared[['chr', 'pos', 'parent_bound']].merge(
            ds_shared[['chr', 'pos', 'parent_bound']],
            on=['chr', 'pos'], suffixes=('_ww', '_ds'))

        same_dir = (merged_shared['parent_bound_ww'] == merged_shared['parent_bound_ds']).sum()
        diff_dir = (merged_shared['parent_bound_ww'] != merged_shared['parent_bound_ds']).sum()
        print(f"  Same parent direction: {same_dir} ({100*same_dir/len(merged_shared):.1f}%)")
        print(f"  Different parent direction: {diff_dir} ({100*diff_dir/len(merged_shared):.1f}%)")
else:
    print("No DS data — skipping condition comparison")

# ── 6. Figures ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel A: Parent-of-origin proportions (WW vs DS)
ax = axes[0, 0]
datasets = [('WW', poo_ww)]
if poo_ds is not None:
    datasets.append(('DS', poo_ds))

x_pos = np.arange(len(datasets))
width = 0.35
for i, (lbl, df) in enumerate(datasets):
    classified = df[df['parent_bound'].isin(['B73', 'founder'])]
    n_b73 = (classified['parent_bound'] == 'B73').sum()
    n_founder = (classified['parent_bound'] == 'founder').sum()
    total = n_b73 + n_founder
    ax.bar(i - width/2, 100 * n_b73 / total, width, color='#2196F3', label='B73' if i == 0 else '')
    ax.bar(i + width/2, 100 * n_founder / total, width, color='#FF9800', label='Founder' if i == 0 else '')
    ax.text(i - width/2, 100 * n_b73 / total + 1, f'{n_b73}', ha='center', fontsize=9)
    ax.text(i + width/2, 100 * n_founder / total + 1, f'{n_founder}', ha='center', fontsize=9)

ax.set_xticks(x_pos)
ax.set_xticklabels([d[0] for d in datasets])
ax.set_ylabel('Percentage of causal bQTL')
ax.set_title('A. Parent-of-origin classification')
ax.legend()
ax.set_ylim(0, 75)

# Panel B: Effect sizes by parent
ax = axes[0, 1]
b73_effects = poo_ww[poo_ww['parent_bound'] == 'B73']['abs_d'].dropna()
founder_effects = poo_ww[poo_ww['parent_bound'] == 'founder']['abs_d'].dropna()
bp = ax.boxplot([b73_effects, founder_effects],
                patch_artist=True, showfliers=False)
ax.set_xticks([1, 2])
ax.set_xticklabels(['B73-bound', 'Founder-bound'])
bp['boxes'][0].set_facecolor('#2196F3')
bp['boxes'][0].set_alpha(0.5)
bp['boxes'][1].set_facecolor('#FF9800')
bp['boxes'][1].set_alpha(0.5)
if len(b73_effects) > 0 and len(founder_effects) > 0:
    mw_stat, mw_p = stats.mannwhitneyu(b73_effects, founder_effects)
    ax.set_title(f'B. Effect size by parent (p={mw_p:.3f})')
else:
    ax.set_title('B. Effect size by parent')
ax.set_ylabel('|Cohen\'s d|')

# Panel C: Spatial nearest-neighbor distances
ax = axes[1, 0]
if len(same_d) > 0 and len(diff_d) > 0:
    bins = np.logspace(np.log10(max(1, min(min(same_d), min(diff_d)))),
                       np.log10(max(max(same_d), max(diff_d))), 40)
    ax.hist(same_d / 1e6, bins=bins / 1e6, alpha=0.6, label='Same parent', color='#4CAF50')
    ax.hist(diff_d / 1e6, bins=bins / 1e6, alpha=0.6, label='Diff parent', color='#F44336')
    ax.set_xscale('log')
    ax.set_xlabel('Distance to nearest neighbor (Mb)')
    ax.set_ylabel('Count')
    ax.legend()
ax.set_title('C. Spatial clustering')

# Panel D: Chromosome runs test
ax = axes[1, 1]
if len(runs_ww) > 0:
    chroms = runs_ww['chr'].values
    z_scores = runs_ww['z_score'].values
    colors = ['#4CAF50' if z < 0 else '#F44336' for z in z_scores]
    bars = ax.barh(range(len(chroms)), z_scores, color=colors, alpha=0.7)
    ax.set_yticks(range(len(chroms)))
    ax.set_yticklabels(chroms, fontsize=8)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.axvline(-1.96, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(1.96, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Z-score (negative = clustered)')
    ax.set_title('D. Runs test (parent clustering)')

plt.tight_layout()
fig_path = FIG_DIR / 'parent_of_origin.pdf'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path}")

# ── 7. Summary JSON ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

summary = {
    'ww': {
        'total': len(poo_ww),
        'b73_bound': int((poo_ww['parent_bound'] == 'B73').sum()),
        'founder_bound': int((poo_ww['parent_bound'] == 'founder').sum()),
        'balanced': int((poo_ww['parent_bound'] == 'balanced').sum()),
        'unknown': int((poo_ww['parent_bound'] == 'unknown').sum()),
        'b73_fraction': float((poo_ww['parent_bound'] == 'B73').sum() /
                              max(1, poo_ww['parent_bound'].isin(['B73', 'founder']).sum())),
        'mean_abs_d_b73': float(poo_ww[poo_ww['parent_bound'] == 'B73']['abs_d'].mean())
            if (poo_ww['parent_bound'] == 'B73').any() else None,
        'mean_abs_d_founder': float(poo_ww[poo_ww['parent_bound'] == 'founder']['abs_d'].mean())
            if (poo_ww['parent_bound'] == 'founder').any() else None,
    },
    'spatial': {
        'median_same_parent_dist_mb': float(np.median(same_d) / 1e6) if len(same_d) > 0 else None,
        'median_diff_parent_dist_mb': float(np.median(diff_d) / 1e6) if len(diff_d) > 0 else None,
        'nearest_neighbor_p': float(stats.mannwhitneyu(same_d, diff_d, alternative='less').pvalue)
            if len(same_d) > 0 and len(diff_d) > 0 else None,
    },
    'pathway_coordination': {
        'go_n_tested': len(go_results_ww),
        'go_n_nominal': int((go_results_ww['binom_p'] < 0.05).sum()) if len(go_results_ww) > 0 else 0,
        'go_n_fdr': int((go_results_ww['fdr'] < 0.05).sum()) if len(go_results_ww) > 0 else 0,
        'kegg_n_tested': len(kegg_results_ww),
        'kegg_n_nominal': int((kegg_results_ww['binom_p'] < 0.05).sum()) if len(kegg_results_ww) > 0 else 0,
        'kegg_n_fdr': int((kegg_results_ww['fdr'] < 0.05).sum()) if len(kegg_results_ww) > 0 else 0,
    },
}

if poo_ds is not None:
    summary['ds'] = {
        'total': len(poo_ds),
        'b73_bound': int((poo_ds['parent_bound'] == 'B73').sum()),
        'founder_bound': int((poo_ds['parent_bound'] == 'founder').sum()),
        'b73_fraction': float((poo_ds['parent_bound'] == 'B73').sum() /
                              max(1, poo_ds['parent_bound'].isin(['B73', 'founder']).sum())),
    }

json_path = RESULTS_DIR / 'parent_of_origin.json'
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {json_path}")

print("\nKey findings:")
ww_classified = poo_ww[poo_ww['parent_bound'].isin(['B73', 'founder'])]
b73_frac = (ww_classified['parent_bound'] == 'B73').mean()
print(f"  B73 allele is the functional allele in {100*b73_frac:.1f}% of WW causal bQTL")
if poo_ds is not None:
    ds_classified = poo_ds[poo_ds['parent_bound'].isin(['B73', 'founder'])]
    ds_b73_frac = (ds_classified['parent_bound'] == 'B73').mean()
    print(f"  B73 allele is the functional allele in {100*ds_b73_frac:.1f}% of DS causal bQTL")

print("\nDone.")
