#!/usr/bin/env python3
"""
55_motif_genotype_ase_chain.py — TF motif identity is irrelevant under baseline.

Part of: "Condition-dependent bQTL switching" paper (Banf & Hartwig)
Paper section: "TF motif identity is irrelevant under baseline conditions"
Pipeline step: 3 of 8

Tests whether specific TF motif disruptions predict genotype→ASE effect strength.
Result: they do NOT. All TF families produce nearly identical effects (CV = 4.6%).

Tests:
  A. Do bQTL with TF motif at the variant have stronger ASE effects?
  B. Which TF family disruptions produce the strongest ASE effects?
  C. Does motif redundancy (more copies nearby) buffer the ASE effect?
  D. Do cis-variable (enriched) TF families show stronger effects than cis-conserved?

Input:
  genotype_bqtl_results.csv (script 54)
  bound_region_motifs.csv (previous pipeline)

Output:
  motif_genotype_ase_chain.csv
  figures/fig12_motif_genotype_ase.pdf — Paper Fig. 2
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUTDIR = BASE / "results"
FIG_DIR = BASE / "figures" / "paper_figures"
FIG_DIR.mkdir(exist_ok=True)

GENO_RESULTS = OUTDIR / "genotype_bqtl_results.csv"
MOTIF_DATA = OUTDIR / "bound_region_motifs.csv"

# TF enrichment from script 26 (variant-overlap, threshold=6.0)
# Enriched = cis-variable (motifs disrupted by bQTL), depleted = cis-conserved
CIS_VARIABLE = ['BES1', 'E2F/DP', 'CAMTA', 'BBR-BPC', 'bHLH', 'bZIP', 'TCP',
                 'MYB', 'ERF', 'NAC', 'C2H2', 'SBP', 'Trihelix', 'G2-like',
                 'ARF', 'B3', 'LBD', 'Nin-like']
CIS_CONSERVED = ['YABBY', 'WOX', 'HD-ZIP', 'CPP', 'HSF', 'Dof', 'GATA',
                  'GRAS', 'TALE', 'MIKC_MADS', 'M-type_MADS', 'AP2']


def main():
    print("=" * 70)
    print("MOTIF DISRUPTION → GENOTYPE → ASE CAUSAL CHAIN")
    print("=" * 70)

    # ── Load data ──
    geno = pd.read_csv(GENO_RESULTS)
    motifs = pd.read_csv(MOTIF_DATA)
    print(f"Genotype→ASE results: {len(geno):,} bQTL-gene pairs")
    print(f"Motif scan data: {len(motifs):,} rows")

    # Merge
    merged = geno.merge(motifs, on=['chr', 'pos', 'gene_id'], how='inner',
                         suffixes=('', '_motif'))
    print(f"Merged: {len(merged):,} rows ({merged[['chr','pos','gene_id']].drop_duplicates().shape[0]:,} unique pairs)")

    # ================================================================
    # A. Do bQTL with TF motif at variant have stronger ASE effects?
    # ================================================================
    print(f"\n{'='*60}")
    print("A. MOTIF AT VARIANT → STRONGER ASE?")
    print(f"{'='*60}")

    # For each bQTL-gene pair, check if ANY motif overlaps the variant
    pair_key = merged.groupby(['chr', 'pos', 'gene_id']).agg(
        has_motif_at_var=('motifs_at_variant', lambda x: (x > 0).any()),
        n_families_at_var=('motifs_at_variant', lambda x: (x > 0).sum()),
        cohens_d=('cohens_d', 'first'),
        mw_p=('mw_p', 'first'),
        fdr_mw=('fdr_mw', 'first'),
        abs_ase_diff=('abs_ase_diff', 'first'),
        n_variant=('n_variant', 'first'),
        n_reference=('n_reference', 'first'),
    ).reset_index()

    with_motif = pair_key[pair_key['has_motif_at_var']]
    without_motif = pair_key[~pair_key['has_motif_at_var']]

    print(f"  Pairs with motif at variant: {len(with_motif):,}")
    print(f"  Pairs without motif at variant: {len(without_motif):,}")

    # Compare |d|
    d_with = with_motif['cohens_d'].abs()
    d_without = without_motif['cohens_d'].abs()
    mw_stat, mw_p = stats.mannwhitneyu(d_with, d_without, alternative='greater')
    print(f"\n  |Cohen's d| with motif: {d_with.mean():.4f} (median {d_with.median():.4f})")
    print(f"  |Cohen's d| without:    {d_without.mean():.4f} (median {d_without.median():.4f})")
    print(f"  Mann-Whitney p (motif > no motif): {mw_p:.4e}")

    # Compare significance rates
    sig_with = (with_motif['fdr_mw'] < 0.05).mean()
    sig_without = (without_motif['fdr_mw'] < 0.05).mean()
    print(f"\n  FDR<0.05 rate with motif: {100*sig_with:.1f}%")
    print(f"  FDR<0.05 rate without:    {100*sig_without:.1f}%")
    print(f"  Enrichment: {sig_with/sig_without:.2f}x" if sig_without > 0 else "  (no sig without)")

    # Compare |ASE| difference
    ase_with = with_motif['abs_ase_diff']
    ase_without = without_motif['abs_ase_diff']
    mw2, p2 = stats.mannwhitneyu(ase_with, ase_without, alternative='greater')
    print(f"\n  |ASE| diff with motif: {ase_with.mean():.4f}")
    print(f"  |ASE| diff without:    {ase_without.mean():.4f}")
    print(f"  Mann-Whitney p: {p2:.4e}")

    # ================================================================
    # B. Per-TF-family effect sizes
    # ================================================================
    print(f"\n{'='*60}")
    print("B. PER-TF-FAMILY ASE EFFECT SIZES")
    print(f"{'='*60}")

    # For each TF family, get the effect sizes of bQTL that disrupt that family's motif
    family_effects = []
    for fam in merged['tf_family'].unique():
        if fam == 'NONE':
            continue
        fam_rows = merged[(merged['tf_family'] == fam) & (merged['motifs_at_variant'] > 0)]
        if len(fam_rows) < 10:
            continue

        fam_pairs = fam_rows[['chr', 'pos', 'gene_id', 'cohens_d', 'mw_p', 'fdr_mw',
                               'abs_ase_diff', 'redundant_in_region']].drop_duplicates(
                               subset=['chr', 'pos', 'gene_id'])

        abs_d = fam_pairs['cohens_d'].abs()
        sig_rate = (fam_pairs['fdr_mw'] < 0.05).mean()
        nom_rate = (fam_pairs['mw_p'] < 0.05).mean()
        mean_redundancy = fam_pairs['redundant_in_region'].mean()

        # Classify
        if fam in CIS_VARIABLE:
            category = 'cis-variable'
        elif fam in CIS_CONSERVED:
            category = 'cis-conserved'
        else:
            category = 'unclassified'

        family_effects.append({
            'tf_family': fam,
            'n_pairs': len(fam_pairs),
            'mean_abs_d': abs_d.mean(),
            'median_abs_d': abs_d.median(),
            'sig_rate_fdr': sig_rate,
            'sig_rate_nom': nom_rate,
            'mean_abs_ase_diff': fam_pairs['abs_ase_diff'].mean(),
            'mean_redundancy': mean_redundancy,
            'category': category,
        })

    fam_df = pd.DataFrame(family_effects).sort_values('mean_abs_d', ascending=False)

    print(f"\n  {'Family':<15} {'N':>5} {'|d|':>7} {'FDR%':>6} {'nom%':>6} "
          f"{'|ASE|':>7} {'Redun':>6} {'Category'}")
    print("  " + "-" * 75)
    for _, row in fam_df.iterrows():
        print(f"  {row['tf_family']:<15} {row['n_pairs']:5d} {row['mean_abs_d']:7.3f} "
              f"{100*row['sig_rate_fdr']:5.1f}% {100*row['sig_rate_nom']:5.1f}% "
              f"{row['mean_abs_ase_diff']:7.4f} {row['mean_redundancy']:6.1f} "
              f"{row['category']}")

    # ================================================================
    # C. Redundancy buffering
    # ================================================================
    print(f"\n{'='*60}")
    print("C. MOTIF REDUNDANCY VS ASE EFFECT SIZE")
    print(f"{'='*60}")

    motif_pairs = merged[merged['motifs_at_variant'] > 0].copy()
    motif_pairs = motif_pairs.drop_duplicates(subset=['chr', 'pos', 'gene_id', 'tf_family'])

    # Bin by redundancy
    bins = [(0, 0, 'No redundancy (0)'),
            (1, 5, 'Low (1-5)'),
            (6, 20, 'Medium (6-20)'),
            (21, 1000, 'High (>20)')]

    print(f"\n  {'Redundancy':<20} {'N':>6} {'|d|':>7} {'FDR%':>6} {'|ASE|':>7}")
    print("  " + "-" * 50)
    redundancy_results = []
    for lo, hi, label in bins:
        mask = (motif_pairs['redundant_in_region'] >= lo) & (motif_pairs['redundant_in_region'] <= hi)
        subset = motif_pairs[mask]
        if len(subset) < 10:
            continue
        abs_d = subset['cohens_d'].abs().mean()
        sig = (subset['fdr_mw'] < 0.05).mean()
        ase_diff = subset['abs_ase_diff'].mean()
        print(f"  {label:<20} {len(subset):6d} {abs_d:7.3f} {100*sig:5.1f}% {ase_diff:7.4f}")
        redundancy_results.append({'bin': label, 'n': len(subset),
                                    'mean_abs_d': abs_d, 'sig_rate': sig})

    # Correlation: redundancy vs |d|
    rho, p_rho = stats.spearmanr(motif_pairs['redundant_in_region'],
                                  motif_pairs['cohens_d'].abs())
    print(f"\n  Spearman correlation (redundancy vs |d|): rho={rho:.4f}, p={p_rho:.4e}")
    if rho < 0:
        print("  → Negative correlation: more redundancy = weaker ASE effect (buffering)")
    else:
        print("  → Positive or no correlation: redundancy does not buffer ASE effect")

    # ================================================================
    # D. Cis-variable vs cis-conserved comparison
    # ================================================================
    print(f"\n{'='*60}")
    print("D. CIS-VARIABLE vs CIS-CONSERVED TF FAMILIES")
    print(f"{'='*60}")

    cis_var = fam_df[fam_df['category'] == 'cis-variable']
    cis_con = fam_df[fam_df['category'] == 'cis-conserved']

    if len(cis_var) > 0 and len(cis_con) > 0:
        # Weighted mean by number of pairs
        wmean_var = np.average(cis_var['mean_abs_d'], weights=cis_var['n_pairs'])
        wmean_con = np.average(cis_con['mean_abs_d'], weights=cis_con['n_pairs'])
        wsig_var = np.average(cis_var['sig_rate_fdr'], weights=cis_var['n_pairs'])
        wsig_con = np.average(cis_con['sig_rate_fdr'], weights=cis_con['n_pairs'])

        print(f"  Cis-variable families ({len(cis_var)}):")
        print(f"    Weighted mean |d|: {wmean_var:.4f}")
        print(f"    Weighted FDR<0.05 rate: {100*wsig_var:.1f}%")
        print(f"    Total pairs: {cis_var['n_pairs'].sum():,}")

        print(f"  Cis-conserved families ({len(cis_con)}):")
        print(f"    Weighted mean |d|: {wmean_con:.4f}")
        print(f"    Weighted FDR<0.05 rate: {100*wsig_con:.1f}%")
        print(f"    Total pairs: {cis_con['n_pairs'].sum():,}")

        # Mann-Whitney on family-level effect sizes
        u_stat, u_p = stats.mannwhitneyu(cis_var['mean_abs_d'], cis_con['mean_abs_d'],
                                          alternative='two-sided')
        print(f"\n  Mann-Whitney (family-level |d|): p={u_p:.4f}")

        # Get individual pair-level data for the two categories
        var_pairs = motif_pairs[motif_pairs['tf_family'].isin(CIS_VARIABLE)]
        con_pairs = motif_pairs[motif_pairs['tf_family'].isin(CIS_CONSERVED)]
        if len(var_pairs) > 0 and len(con_pairs) > 0:
            u2, p2 = stats.mannwhitneyu(var_pairs['cohens_d'].abs(),
                                         con_pairs['cohens_d'].abs(),
                                         alternative='two-sided')
            print(f"  Mann-Whitney (pair-level |d|): p={p2:.4e}")
            print(f"    Cis-variable pairs: n={len(var_pairs):,}, mean |d|={var_pairs['cohens_d'].abs().mean():.4f}")
            print(f"    Cis-conserved pairs: n={len(con_pairs):,}, mean |d|={con_pairs['cohens_d'].abs().mean():.4f}")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*60}")
    print("SUMMARY: CAUSAL CHAIN ESTABLISHED?")
    print(f"{'='*60}")

    chain_tests = []

    # Test 1: motif at variant → stronger effect
    chain_tests.append(('Motif at variant → stronger |d|', mw_p < 0.05, mw_p))

    # Test 2: motif at variant → more significant
    if sig_without > 0:
        fisher_tab = np.array([
            [(with_motif['fdr_mw'] < 0.05).sum(), (with_motif['fdr_mw'] >= 0.05).sum()],
            [(without_motif['fdr_mw'] < 0.05).sum(), (without_motif['fdr_mw'] >= 0.05).sum()]
        ])
        _, fisher_p = stats.fisher_exact(fisher_tab, alternative='greater')
        chain_tests.append(('Motif at variant → higher FDR sig rate', fisher_p < 0.05, fisher_p))

    # Test 3: redundancy buffers effect
    chain_tests.append(('Redundancy negatively correlates with |d|', rho < 0 and p_rho < 0.05, p_rho))

    for desc, passed, p in chain_tests:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc} (p={p:.4e})")

    # Save
    fam_df.to_csv(OUTDIR / 'motif_genotype_ase_chain.csv', index=False)
    print(f"\nSaved: motif_genotype_ase_chain.csv")

    # ================================================================
    # Figures
    # ================================================================
    print("\nGenerating figures...")
    make_figures(pair_key, fam_df, motif_pairs, redundancy_results)


def make_figures(pair_key, fam_df, motif_pairs, redundancy_results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # A: Effect size with vs without motif at variant
    ax = axes[0, 0]
    with_motif = pair_key[pair_key['has_motif_at_var']]['cohens_d'].abs().clip(0, 5)
    without_motif = pair_key[~pair_key['has_motif_at_var']]['cohens_d'].abs().clip(0, 5)
    bins_hist = np.linspace(0, 5, 40)
    ax.hist(with_motif, bins=bins_hist, alpha=0.6, color='#D32F2F', density=True,
            edgecolor='white', label=f'Motif at variant (n={len(with_motif):,})')
    ax.hist(without_motif, bins=bins_hist, alpha=0.6, color='#1976D2', density=True,
            edgecolor='white', label=f'No motif (n={len(without_motif):,})')
    ax.axvline(with_motif.median(), color='#D32F2F', linestyle='--', alpha=0.8)
    ax.axvline(without_motif.median(), color='#1976D2', linestyle='--', alpha=0.8)
    ax.set_xlabel("|Cohen's d|")
    ax.set_ylabel('Density')
    ax.set_title("A. TF motif at variant → stronger ASE effect")
    ax.legend(fontsize=8)

    # B: Per-TF-family effect sizes (horizontal bar)
    ax = axes[0, 1]
    plot_df = fam_df.sort_values('mean_abs_d', ascending=True).tail(20)
    colors = []
    for _, row in plot_df.iterrows():
        if row['category'] == 'cis-variable':
            colors.append('#D32F2F')
        elif row['category'] == 'cis-conserved':
            colors.append('#1976D2')
        else:
            colors.append('#757575')
    ax.barh(range(len(plot_df)), plot_df['mean_abs_d'], color=colors, edgecolor='white')
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['tf_family'], fontsize=8)
    ax.set_xlabel("Mean |Cohen's d|")
    ax.set_title("B. ASE effect by disrupted TF family")
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#D32F2F', label='Cis-variable (enriched)'),
        Patch(color='#1976D2', label='Cis-conserved (depleted)'),
        Patch(color='#757575', label='Unclassified'),
    ], fontsize=7, loc='lower right')

    # C: Redundancy vs effect size
    ax = axes[1, 0]
    red = motif_pairs['redundant_in_region'].clip(0, 100)
    abs_d = motif_pairs['cohens_d'].abs().clip(0, 5)
    ax.scatter(red, abs_d, s=3, alpha=0.1, color='#7B1FA2', rasterized=True)
    # Binned means
    bin_edges = [0, 1, 3, 6, 10, 20, 50, 100]
    for i in range(len(bin_edges) - 1):
        mask = (red >= bin_edges[i]) & (red < bin_edges[i + 1])
        if mask.sum() > 20:
            mid = (bin_edges[i] + bin_edges[i + 1]) / 2
            mean_d = abs_d[mask].mean()
            se = abs_d[mask].std() / np.sqrt(mask.sum())
            ax.errorbar(mid, mean_d, yerr=se, fmt='rs', markersize=6, capsize=3)
    ax.set_xlabel('Redundant motifs in region')
    ax.set_ylabel("|Cohen's d|")
    rho, p_rho = stats.spearmanr(red, abs_d)
    ax.set_title(f"C. Motif redundancy vs ASE effect\nrho={rho:.3f}, p={p_rho:.2e}")

    # D: Cis-variable vs cis-conserved
    ax = axes[1, 1]
    var_fams = fam_df[fam_df['category'] == 'cis-variable'].sort_values('mean_abs_d')
    con_fams = fam_df[fam_df['category'] == 'cis-conserved'].sort_values('mean_abs_d')

    y_pos = 0
    yticks = []
    ylabels = []
    for _, row in con_fams.iterrows():
        ax.barh(y_pos, row['mean_abs_d'], color='#1976D2', edgecolor='white', height=0.7)
        yticks.append(y_pos)
        ylabels.append(row['tf_family'])
        y_pos += 1
    gap_y = y_pos
    y_pos += 0.5
    for _, row in var_fams.iterrows():
        ax.barh(y_pos, row['mean_abs_d'], color='#D32F2F', edgecolor='white', height=0.7)
        yticks.append(y_pos)
        ylabels.append(row['tf_family'])
        y_pos += 1

    ax.axhline(gap_y - 0.25, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=6)
    ax.set_xlabel("Mean |Cohen's d|")
    ax.set_title("D. Cis-variable vs cis-conserved\nTF family ASE effects")
    ax.text(0.95, 0.15, 'Cis-conserved\n(motifs preserved)', transform=ax.transAxes,
            fontsize=8, color='#1976D2', ha='right', va='bottom')
    ax.text(0.95, 0.85, 'Cis-variable\n(motifs disrupted)', transform=ax.transAxes,
            fontsize=8, color='#D32F2F', ha='right', va='top')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig12_motif_genotype_ase.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(FIG_DIR / 'fig12_motif_genotype_ase.png', bbox_inches='tight', dpi=150)
    print(f"  Saved: figures/fig12_motif_genotype_ase.pdf")
    plt.close()


if __name__ == '__main__':
    main()
