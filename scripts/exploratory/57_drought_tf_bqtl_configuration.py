#!/usr/bin/env python3
"""
57_drought_tf_bqtl_configuration.py — TF expression variation and drought activation.

Part of: "Condition-dependent bQTL switching" paper (Banf & Hartwig)
Paper section: "The TF concentration landscape gates bQTL functionality" (WW baseline)
Pipeline step: 5 of 8

Key hypothesis: Under well-watered (WW) conditions, most TFs are at baseline,
so disrupting any TF motif has a similar ASE effect (script 55 finding).
Under drought, specific TFs are activated → disrupting THEIR motifs should
have amplified effects → bQTL become condition-dependent regulators.

Key results: TF CV=0.580 across hybrids; NAC log2FC=+4.0, HSF +1.15, ERF +0.94.

Analysis:
  1. TF expression variation across hybrids (do TFs differ between hybrids?)
  2. Drought-responsive TF families (which TFs activate under stress?)
  3. For functional bQTL: are disrupted TF families enriched for drought-responsive ones?
  4. Regulatory configuration: WW baseline vs drought-activated TF landscape
  5. Predict which bQTL become MORE functional under drought

Input:
  genotype_bqtl_results.csv       (script 54)
  bound_region_motifs.csv          (motif scan)
  engelhorn_ase_ww.csv             (per-hybrid ASE)
  engelhorn_ww_vs_ds_expression.tsv (drought FC)
  EntAP annotations               (TF gene identification)

Output:
  drought_tf_bqtl_config.csv
  figures/fig14_drought_tf_configuration.pdf
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUTDIR = BASE / "results"
FIG_DIR = BASE / "figures" / "paper_figures"
FIG_DIR.mkdir(exist_ok=True)

GENO_RESULTS = OUTDIR / "genotype_bqtl_results.csv"
MOTIF_DATA = OUTDIR / "bound_region_motifs.csv"
ASE_FILE = DATA / "processed" / "engelhorn_ase_ww.csv"
EXPR_FILE = DATA / "processed" / "engelhorn_ww_vs_ds_expression.tsv"
ENTAP_FILE = DATA / "raw" / "Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1_entap_results.tsv.gz"
CHAIN_DATA = BASE.parent / "bqtl_predict" / "results" / "regulatory_network" / "regulatory_chains.csv"

# TF family keywords for EntAP description matching
TF_FAMILY_PATTERNS = {
    'MYB': r'\bMYB\d*\b(?!.*related)',
    'ERF': r'ethylene.responsive|ERF\d|AP2.ERF|\bERF\b',
    'bHLH': r'\bbHLH\b',
    'NAC': r'\bNAC\b|NAM.B\d',
    'WRKY': r'\bWRKY\b',
    'bZIP': r'\bbZIP\b',
    'HSF': r'heat.stress.transcription|heat.shock.factor|\bHSF\b',
    'TCP': r'\bTCP\b',
    'GATA': r'\bGATA\b',
    'HD-ZIP': r'HD.ZIP|homeodomain.leucine',
    'Dof': r'\bDof\b|DNA.binding.with.one.finger',
    'SBP': r'\bSBP\b|squamosa',
    'ARF': r'auxin.response.factor',
    'C2H2': r'C2H2',
    'MADS': r'\bMADS\b|MIKC',
    'G2-like': r'G2.like|GLK|golden',
    'Trihelix': r'trihelix',
    'LBD': r'\bLBD\b|LOB.domain',
}


def classify_tf_families(entap_df):
    """Classify TF genes into families using EntAP descriptions."""
    import re

    tf_mask = entap_df['Description'].astype(str).str.contains(
        'transcription factor', case=False, na=False)
    tf_df = entap_df[tf_mask].copy()

    gene_families = defaultdict(set)
    for _, row in tf_df.iterrows():
        gene_id = row['gene_id']
        desc = str(row['Description'])
        for family, pattern in TF_FAMILY_PATTERNS.items():
            if re.search(pattern, desc, re.IGNORECASE):
                gene_families[gene_id].add(family)

    # Also assign unmatched TF genes to 'other_TF'
    all_tf_genes = set(tf_df['gene_id'].unique())
    classified = set(gene_families.keys())
    for g in all_tf_genes - classified:
        gene_families[g].add('other_TF')

    return gene_families, all_tf_genes


def main():
    print("=" * 70)
    print("CONDITION-DEPENDENT REGULATORY CONFIGURATION AT bQTL")
    print("=" * 70)

    # ── Load data ──
    print("\n1. Loading data...")
    geno = pd.read_csv(GENO_RESULTS)
    motifs = pd.read_csv(MOTIF_DATA)
    ase = pd.read_csv(ASE_FILE)
    expr = pd.read_csv(EXPR_FILE, sep='\t')
    entap = pd.read_csv(ENTAP_FILE, sep='\t', compression='gzip', low_memory=False)
    entap['gene_id'] = entap['Query Sequence'].str.replace(r'_T\d+$', '', regex=True)
    chains = pd.read_csv(CHAIN_DATA)

    geno['functional'] = geno['fdr_mw'] < 0.05

    # ── Classify TF genes ──
    gene_families, all_tf_genes = classify_tf_families(entap)
    print(f"  TF genes classified: {len(gene_families):,} (of {len(all_tf_genes):,} total)")

    # ================================================================
    # 2. TF EXPRESSION VARIATION ACROSS HYBRIDS
    # ================================================================
    print(f"\n{'='*60}")
    print("2. TF EXPRESSION VARIATION ACROSS HYBRIDS")
    print(f"{'='*60}")

    ase['total_expr'] = ase['b73_mean'] + ase['nam_mean']
    tf_ase = ase[ase['gene_id'].isin(all_tf_genes)].copy()
    nontf_ase = ase[~ase['gene_id'].isin(all_tf_genes)].copy()

    # Per-gene CV across hybrids
    tf_cv = tf_ase.groupby('gene_id')['total_expr'].agg(['mean', 'std', 'count'])
    tf_cv['cv'] = tf_cv['std'] / tf_cv['mean']
    tf_cv = tf_cv[tf_cv['count'] >= 10]

    nontf_cv = nontf_ase.groupby('gene_id')['total_expr'].agg(['mean', 'std', 'count'])
    nontf_cv['cv'] = nontf_cv['std'] / nontf_cv['mean']
    nontf_cv = nontf_cv[nontf_cv['count'] >= 10]

    print(f"  TF genes (≥10 hybrids): {len(tf_cv):,}")
    print(f"  Non-TF genes: {len(nontf_cv):,}")
    print(f"\n  Expression CV across 24 hybrids:")
    print(f"    TF mean CV: {tf_cv['cv'].mean():.3f}, median: {tf_cv['cv'].median():.3f}")
    print(f"    Non-TF mean CV: {nontf_cv['cv'].mean():.3f}, median: {nontf_cv['cv'].median():.3f}")
    _, p_cv = stats.mannwhitneyu(tf_cv['cv'].dropna(), nontf_cv['cv'].dropna())
    print(f"    Mann-Whitney p: {p_cv:.4e}")
    print(f"    TFs with CV > 0.5: {(tf_cv['cv']>0.5).sum()} ({100*(tf_cv['cv']>0.5).mean():.1f}%)")

    # Per-family expression variation
    print(f"\n  Per-family expression variation:")
    family_expr_stats = []
    for family in sorted(TF_FAMILY_PATTERNS.keys()):
        fam_genes = [g for g, fams in gene_families.items() if family in fams]
        fam_expr = expr[expr['gene_id'].isin(fam_genes)]
        if len(fam_expr) < 3:
            continue
        mean_ww = fam_expr['mean_count_WW'].mean()
        mean_fc = fam_expr['log2fc_DS_vs_WW'].mean()
        n_up = (fam_expr['log2fc_DS_vs_WW'] > 1).sum()
        n_down = (fam_expr['log2fc_DS_vs_WW'] < -1).sum()

        # CV across hybrids for this family's genes
        fam_ase = tf_ase[tf_ase['gene_id'].isin(fam_genes)]
        fam_cv_vals = fam_ase.groupby('gene_id')['total_expr'].agg(['mean', 'std'])
        fam_cv_vals['cv'] = fam_cv_vals['std'] / fam_cv_vals['mean']
        mean_hybrid_cv = fam_cv_vals['cv'].mean() if len(fam_cv_vals) > 0 else np.nan

        family_expr_stats.append({
            'family': family, 'n_genes': len(fam_expr),
            'mean_ww_expr': mean_ww, 'mean_drought_fc': mean_fc,
            'n_drought_up': n_up, 'n_drought_down': n_down,
            'mean_hybrid_cv': mean_hybrid_cv,
        })

    fes = pd.DataFrame(family_expr_stats).sort_values('mean_drought_fc', ascending=False)
    print(f"\n  {'Family':<12} {'N':>4} {'WW expr':>8} {'Drought FC':>10} {'Up':>4} {'Down':>4} {'Hybrid CV':>10}")
    print("  " + "-" * 60)
    for _, row in fes.iterrows():
        print(f"  {row['family']:<12} {row['n_genes']:4d} {row['mean_ww_expr']:8.0f} "
              f"{row['mean_drought_fc']:+10.2f} {row['n_drought_up']:4d} {row['n_drought_down']:4d} "
              f"{row['mean_hybrid_cv']:10.3f}")

    # ================================================================
    # 3. DROUGHT-RESPONSIVE TFs AT FUNCTIONAL bQTL
    # ================================================================
    print(f"\n{'='*60}")
    print("3. ARE DROUGHT-RESPONSIVE TF MOTIFS ENRICHED AT FUNCTIONAL bQTL?")
    print(f"{'='*60}")

    # Build family-level drought response
    family_drought_fc = {}
    for _, row in fes.iterrows():
        family_drought_fc[row['family']] = row['mean_drought_fc']

    # Merge genotype results with motif data
    merged = geno.merge(motifs, on=['chr', 'pos', 'gene_id'], how='inner',
                         suffixes=('', '_motif'))
    disrupted = merged[merged['motifs_at_variant'] > 0].copy()

    # For each bQTL-gene pair, classify disrupted families as drought-responsive or stable
    pair_drought = []
    for (chrom, pos, gene_id), group in disrupted.groupby(['chr', 'pos', 'gene_id']):
        functional = group.iloc[0]['functional']
        cohens_d = group.iloc[0]['cohens_d']
        abs_d = abs(cohens_d)
        fdr = group.iloc[0]['fdr_mw']

        families = group['tf_family'].unique()
        n_drought_up_families = 0
        n_drought_down_families = 0
        n_stable_families = 0
        max_drought_fc = -999
        drought_families = []

        for fam in families:
            fc = family_drought_fc.get(fam, 0)
            if fc > 0.5:
                n_drought_up_families += 1
                drought_families.append(fam)
            elif fc < -0.5:
                n_drought_down_families += 1
            else:
                n_stable_families += 1
            max_drought_fc = max(max_drought_fc, fc)

        pair_drought.append({
            'chr': chrom, 'pos': pos, 'gene_id': gene_id,
            'functional': functional, 'cohens_d': cohens_d, 'abs_d': abs_d, 'fdr_mw': fdr,
            'n_drought_up_families': n_drought_up_families,
            'n_drought_down_families': n_drought_down_families,
            'n_stable_families': n_stable_families,
            'max_drought_fc': max_drought_fc if max_drought_fc > -999 else np.nan,
            'has_drought_up_tf': n_drought_up_families > 0,
            'drought_families': '|'.join(drought_families),
        })

    pd_df = pd.DataFrame(pair_drought)
    func = pd_df[pd_df['functional']]
    nonfunc = pd_df[~pd_df['functional']]

    # Test: are drought-responsive TF motif disruptions enriched at functional bQTL?
    func_drought = func['has_drought_up_tf'].mean()
    nonfunc_drought = nonfunc['has_drought_up_tf'].mean()
    a = int(func['has_drought_up_tf'].sum())
    b = int(len(func) - a)
    c = int(nonfunc['has_drought_up_tf'].sum())
    d = int(len(nonfunc) - c)
    or_val, fisher_p = stats.fisher_exact([[a, b], [c, d]])
    print(f"  Functional bQTL disrupting drought-UP TF: {100*func_drought:.1f}%")
    print(f"  Non-functional bQTL disrupting drought-UP TF: {100*nonfunc_drought:.1f}%")
    print(f"  Odds ratio: {or_val:.3f}, Fisher p: {fisher_p:.4e}")

    # Effect size comparison
    drought_pairs = pd_df[pd_df['has_drought_up_tf']]
    stable_pairs = pd_df[~pd_df['has_drought_up_tf']]
    _, p_d = stats.mannwhitneyu(drought_pairs['abs_d'], stable_pairs['abs_d'],
                                 alternative='two-sided')
    print(f"\n  |d| for drought-TF disrupted: {drought_pairs['abs_d'].mean():.4f}")
    print(f"  |d| for stable-TF disrupted:  {stable_pairs['abs_d'].mean():.4f}")
    print(f"  Mann-Whitney p: {p_d:.4e}")

    # ================================================================
    # 4. PER-FAMILY: DROUGHT FC vs ASE EFFECT AT DISRUPTED bQTL
    # ================================================================
    print(f"\n{'='*60}")
    print("4. DROUGHT RESPONSE vs ASE EFFECT BY TF FAMILY")
    print(f"{'='*60}")

    family_ase_effects = []
    for fam in sorted(TF_FAMILY_PATTERNS.keys()):
        fam_rows = disrupted[disrupted['tf_family'] == fam]
        fam_pairs = fam_rows.drop_duplicates(subset=['chr', 'pos', 'gene_id'])
        if len(fam_pairs) < 20:
            continue

        mean_abs_d = fam_pairs['cohens_d'].abs().mean()
        sig_rate = (fam_pairs['fdr_mw'] < 0.05).mean()
        drought_fc = family_drought_fc.get(fam, np.nan)

        family_ase_effects.append({
            'family': fam, 'n_pairs': len(fam_pairs),
            'mean_abs_d': mean_abs_d, 'sig_rate': sig_rate,
            'drought_fc': drought_fc,
        })

    fae = pd.DataFrame(family_ase_effects)

    # Correlation: drought FC vs ASE effect
    fae_valid = fae.dropna(subset=['drought_fc'])
    rho, p = stats.spearmanr(fae_valid['drought_fc'], fae_valid['mean_abs_d'])
    print(f"  Spearman (drought FC vs mean |d|): rho={rho:.4f}, p={p:.4f}")

    rho2, p2 = stats.spearmanr(fae_valid['drought_fc'], fae_valid['sig_rate'])
    print(f"  Spearman (drought FC vs FDR sig rate): rho={rho2:.4f}, p={p2:.4f}")

    print(f"\n  {'Family':<12} {'N':>5} {'|d|':>7} {'FDR%':>6} {'Drought FC':>10}")
    print("  " + "-" * 45)
    for _, row in fae.sort_values('drought_fc', ascending=False).iterrows():
        print(f"  {row['family']:<12} {row['n_pairs']:5d} {row['mean_abs_d']:7.3f} "
              f"{100*row['sig_rate']:5.1f}% {row['drought_fc']:+10.2f}")

    # ================================================================
    # 5. WELL-WATERED vs DROUGHT REGULATORY CONFIGURATION
    # ================================================================
    print(f"\n{'='*60}")
    print("5. REGULATORY CONFIGURATION: WW BASELINE vs DROUGHT")
    print(f"{'='*60}")

    print(f"\n  Under WELL-WATERED conditions (our ASE data):")
    print(f"    All TF families at similar baseline → disrupting any motif ≈ same ASE effect")
    print(f"    |d| range across families: {fae['mean_abs_d'].min():.3f} - {fae['mean_abs_d'].max():.3f}")
    print(f"    Coefficient of variation of family |d|: {fae['mean_abs_d'].std()/fae['mean_abs_d'].mean():.3f}")

    print(f"\n  Under DROUGHT conditions (predicted):")
    # Families with the highest drought upregulation
    top_drought = fae.nlargest(5, 'drought_fc')
    print(f"    Most drought-activated families:")
    for _, row in top_drought.iterrows():
        print(f"      {row['family']}: drought FC = {row['drought_fc']:+.2f}, "
              f"bQTL disrupting: {row['n_pairs']}")

    # How many bQTL would become more impactful under drought?
    drought_sensitive = pd_df[pd_df['has_drought_up_tf']].copy()
    n_currently_func = drought_sensitive['functional'].sum()
    n_total = len(drought_sensitive)
    print(f"\n    bQTL disrupting drought-responsive TF motifs: {n_total:,}")
    print(f"    Currently functional (WW): {n_currently_func:,} ({100*n_currently_func/n_total:.1f}%)")
    print(f"    Predicted to gain function under drought: additional bQTL among remaining "
          f"{n_total - n_currently_func:,}")

    # ================================================================
    # 6. HYBRID-SPECIFIC TF EXPRESSION × GENOTYPE INTERACTION
    # ================================================================
    print(f"\n{'='*60}")
    print("6. HYBRID-SPECIFIC TF EXPRESSION")
    print(f"{'='*60}")

    # For the most variable TF families, show per-hybrid expression ranges
    ase['hybrid_short'] = ase['hybrid'].str.replace('B73x', '', regex=False)
    hybrids = sorted(ase['hybrid_short'].unique())

    print(f"\n  Per-hybrid TF expression (mean total reads):")
    print(f"  {'Family':<12}", end='')
    for h in hybrids[:8]:
        print(f"  {h:>7}", end='')
    print("  ...")

    for family in ['ERF', 'NAC', 'HSF', 'MYB', 'bHLH', 'WRKY']:
        fam_genes = [g for g, fams in gene_families.items() if family in fams]
        if not fam_genes:
            continue
        print(f"  {family:<12}", end='')
        for h in hybrids[:8]:
            h_expr = ase[(ase['gene_id'].isin(fam_genes)) & (ase['hybrid_short'] == h)]
            mean_val = h_expr['total_expr'].mean() if len(h_expr) > 0 else 0
            print(f"  {mean_val:7.0f}", end='')
        # CV across hybrids
        hybrid_means = []
        for h in hybrids:
            h_expr = ase[(ase['gene_id'].isin(fam_genes)) & (ase['hybrid_short'] == h)]
            if len(h_expr) > 0:
                hybrid_means.append(h_expr['total_expr'].mean())
        cv = np.std(hybrid_means) / np.mean(hybrid_means) if hybrid_means else 0
        print(f"  CV={cv:.2f}")

    # ── Save ──
    pd_df.to_csv(OUTDIR / 'drought_tf_bqtl_config.csv', index=False)
    fae.to_csv(OUTDIR / 'family_drought_ase_effects.csv', index=False)
    fes.to_csv(OUTDIR / 'tf_family_expression_stats.csv', index=False)
    print(f"\n  Saved: drought_tf_bqtl_config.csv")

    # ── Figures ──
    print("\n7. Generating figures...")
    make_figures(fes, fae, pd_df, tf_cv, nontf_cv)


def make_figures(fes, fae, pd_df, tf_cv, nontf_cv):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # A: TF expression CV across hybrids
    ax = axes[0, 0]
    bins = np.linspace(0, 2, 40)
    ax.hist(tf_cv['cv'].dropna().clip(0, 2), bins=bins, alpha=0.6, color='#D32F2F',
            density=True, edgecolor='white', label=f'TF genes (n={len(tf_cv)})')
    ax.hist(nontf_cv['cv'].dropna().clip(0, 2), bins=bins, alpha=0.4, color='#1976D2',
            density=True, edgecolor='white', label=f'Non-TF genes (n={len(nontf_cv)})')
    ax.set_xlabel('Coefficient of variation across 24 hybrids')
    ax.set_ylabel('Density')
    ax.set_title('A. TF expression variability across hybrids')
    ax.legend(fontsize=8)

    # B: TF family drought response (bar chart)
    ax = axes[0, 1]
    fes_sorted = fes.sort_values('mean_drought_fc')
    colors = ['#D32F2F' if fc > 0.5 else '#1976D2' if fc < -0.5 else '#757575'
              for fc in fes_sorted['mean_drought_fc']]
    ax.barh(range(len(fes_sorted)), fes_sorted['mean_drought_fc'],
            color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(fes_sorted)))
    ax.set_yticklabels(fes_sorted['family'], fontsize=8)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean log2FC (drought / well-watered)')
    ax.set_title('B. TF family drought response')

    # C: Drought FC vs ASE effect (scatter)
    ax = axes[0, 2]
    ax.scatter(fae['drought_fc'], fae['mean_abs_d'], s=fae['n_pairs']/5,
               color='#7B1FA2', alpha=0.7, edgecolors='white')
    for _, row in fae.iterrows():
        ax.annotate(row['family'], (row['drought_fc'], row['mean_abs_d']),
                   fontsize=6, ha='center', va='bottom')
    rho, p = stats.spearmanr(fae['drought_fc'], fae['mean_abs_d'])
    ax.set_xlabel('TF family drought FC')
    ax.set_ylabel("Mean |Cohen's d| at disrupted bQTL")
    ax.set_title(f"C. Drought response vs ASE effect\nrho={rho:.3f}, p={p:.3f}")

    # D: Family-level hybrid CV vs drought FC
    ax = axes[1, 0]
    ax.scatter(fes['mean_hybrid_cv'], fes['mean_drought_fc'],
               s=fes['n_genes']*5, color='#FF7043', alpha=0.7, edgecolors='white')
    for _, row in fes.iterrows():
        ax.annotate(row['family'], (row['mean_hybrid_cv'], row['mean_drought_fc']),
                   fontsize=6, ha='center', va='bottom')
    ax.set_xlabel('Mean CV across hybrids')
    ax.set_ylabel('Mean drought log2FC')
    ax.set_title('D. TF variability vs drought response')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)

    # E: Effect size distribution for drought-UP vs stable TF disruptions
    ax = axes[1, 1]
    drought_up = pd_df[pd_df['has_drought_up_tf']]['abs_d'].clip(0, 5)
    stable = pd_df[~pd_df['has_drought_up_tf']]['abs_d'].clip(0, 5)
    bins = np.linspace(0, 5, 40)
    ax.hist(drought_up, bins=bins, alpha=0.6, color='#D32F2F', density=True,
            edgecolor='white', label=f'Drought-UP TF disrupted (n={len(drought_up):,})')
    ax.hist(stable, bins=bins, alpha=0.6, color='#1976D2', density=True,
            edgecolor='white', label=f'Stable TF disrupted (n={len(stable):,})')
    ax.set_xlabel("|Cohen's d|")
    ax.set_ylabel('Density')
    ax.set_title("E. ASE effect: drought-responsive vs stable TF disruption")
    ax.legend(fontsize=7)

    # F: Regulatory configuration schematic (text/annotation panel)
    ax = axes[1, 2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_axis_off()

    # WW condition
    ax.text(2.5, 9.5, 'Well-watered', fontsize=12, fontweight='bold', ha='center', color='#1976D2')
    ax.text(2.5, 8.5, 'All TFs at baseline\n→ Any motif disruption\n≈ same ASE effect',
            fontsize=9, ha='center', color='#333333')
    ax.text(2.5, 6.5, f"|d| range: {fae['mean_abs_d'].min():.2f}–{fae['mean_abs_d'].max():.2f}",
            fontsize=9, ha='center', color='#1976D2')
    ax.text(2.5, 5.8, f"CV across families: {fae['mean_abs_d'].std()/fae['mean_abs_d'].mean():.2f}",
            fontsize=9, ha='center', color='#1976D2')

    # Drought condition
    ax.text(7.5, 9.5, 'Drought', fontsize=12, fontweight='bold', ha='center', color='#D32F2F')
    drought_up_fams = fae.nlargest(3, 'drought_fc')['family'].tolist()
    ax.text(7.5, 8.5, f'Activated: {", ".join(drought_up_fams)}\n→ Their motif disruptions\n→ amplified ASE effects',
            fontsize=9, ha='center', color='#333333')
    n_drought_bqtl = pd_df['has_drought_up_tf'].sum()
    ax.text(7.5, 6.5, f'{n_drought_bqtl:,} bQTL disrupt\ndrought-responsive TF motifs',
            fontsize=9, ha='center', color='#D32F2F')

    ax.annotate('', xy=(5.5, 7.5), xytext=(4.5, 7.5),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

    ax.text(5, 3.5, 'Prediction: condition-dependent\nbQTL functionality',
            fontsize=10, ha='center', fontweight='bold', color='#333333',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4', edgecolor='#F9A825'))

    # Title
    ax.set_title('F. Regulatory configuration model')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig14_drought_tf_configuration.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(FIG_DIR / 'fig14_drought_tf_configuration.png', bbox_inches='tight', dpi=150)
    print(f"  Saved: figures/fig14_drought_tf_configuration.pdf")
    plt.close()


if __name__ == '__main__':
    main()
