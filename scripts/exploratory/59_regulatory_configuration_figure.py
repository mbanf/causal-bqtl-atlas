#!/usr/bin/env python3
"""
59_regulatory_configuration_figure.py — TF saturation model and summary figure.

Part of: "Condition-dependent bQTL switching" paper (Banf & Hartwig)
Paper section: "The TF concentration landscape gates bQTL functionality"
Pipeline step: 7 of 8

Tests the TF saturation model:
  High TF expression → saturated binding → variant effect masked (lower |d|)
  Low TF expression → subsaturated binding → variant effect revealed (higher |d|)

Key results (paper numbers):
  WW: TF expression vs |d|: rho=-0.830, p=0.0008 (saturation confirmed)
  DS: rho=-0.067, p=0.84 (relationship collapses under drought)
  NAC +5.25 log2FC; ERF, HSF, bHLH lose bQTL effect; LFY, YABBY gain

Input:
  ww_vs_drought_genotype_bqtl.csv (script 58)
  ww_vs_drought_tf_family_effects.csv (script 58)
  engelhorn_ww_vs_ds_expression.tsv
  entap_results.tsv.gz

Output:
  tf_expression_vs_bqtl_effect.csv
  figures/fig16_regulatory_configuration.pdf — Paper Fig. 4
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUTDIR = BASE / "results"
FIG_DIR = BASE / "figures" / "paper_figures"

MOESM5 = DATA / "raw" / "supp_table_MOESM5.xlsx"
ASE_WW_FILE = DATA / "processed" / "engelhorn_ase_ww.csv"
EXPR_FILE = DATA / "processed" / "engelhorn_ww_vs_ds_expression.tsv"
ENTAP_FILE = DATA / "raw" / "Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1_entap_results.tsv.gz"
MOTIF_DATA = OUTDIR / "bound_region_motifs.csv"
WW_DS_RESULTS = OUTDIR / "ww_vs_drought_genotype_bqtl.csv"
TF_FAMILY_EFFECTS = OUTDIR / "ww_vs_drought_tf_family_effects.csv"
TF_FAMILY_EXPR = OUTDIR / "tf_family_expression_stats.csv"

# TF family patterns for EntAP matching
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
    import re
    from collections import defaultdict
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
    return gene_families


def main():
    print("=" * 70)
    print("REGULATORY CONFIGURATION: TF EXPRESSION vs bQTL EFFECT")
    print("=" * 70)

    # ── Load data ──
    print("\n1. Loading data...")
    combined = pd.read_csv(WW_DS_RESULTS)
    tf_effects = pd.read_csv(TF_FAMILY_EFFECTS)
    expr = pd.read_csv(EXPR_FILE, sep='\t')
    ase_ww = pd.read_csv(ASE_WW_FILE)
    ase_ww['total_expr'] = ase_ww['b73_mean'] + ase_ww['nam_mean']

    entap = pd.read_csv(ENTAP_FILE, sep='\t', compression='gzip', low_memory=False)
    entap['gene_id'] = entap['Query Sequence'].str.replace(r'_T\d+$', '', regex=True)
    gene_families = classify_tf_families(entap)

    rdf_ww = combined[combined['condition'] == 'WW'].copy()
    rdf_ds = combined[combined['condition'] == 'DS'].copy()
    print(f"  WW results: {len(rdf_ww):,}")
    print(f"  DS results: {len(rdf_ds):,}")

    # ── TF expression per family under WW and DS ──
    print("\n2. Computing TF family expression levels...")

    family_expr_ww = {}
    family_expr_ds = {}
    for family in sorted(TF_FAMILY_PATTERNS.keys()):
        fam_genes = [g for g, fams in gene_families.items() if family in fams]
        fam_expr = expr[expr['gene_id'].isin(fam_genes)]
        if len(fam_expr) > 0:
            family_expr_ww[family] = fam_expr['mean_count_WW'].mean()
            family_expr_ds[family] = fam_expr['mean_count_DS'].mean()

    # ── Correlate TF expression CHANGE with bQTL effect CHANGE ──
    print("\n3. Testing saturation model...")
    print("  Hypothesis: TF upregulation → decreased bQTL effect (binding saturation)")

    correlation_data = []
    for _, row in tf_effects.iterrows():
        fam = row['tf_family']
        # Map motif family names to expression family names
        fam_key = fam
        if fam == 'MIKC_MADS':
            fam_key = 'MADS'
        elif fam == 'MYB_related':
            fam_key = 'MYB'  # group with MYB

        ww_expr = family_expr_ww.get(fam_key)
        ds_expr = family_expr_ds.get(fam_key)

        if ww_expr is not None and ds_expr is not None and ww_expr > 0:
            expr_fc = np.log2((ds_expr + 1) / (ww_expr + 1))
            correlation_data.append({
                'tf_family': fam,
                'ww_expr': ww_expr,
                'ds_expr': ds_expr,
                'expr_log2fc': expr_fc,
                'd_change': row['d_change'],
                'ww_abs_d': row['ww_abs_d'],
                'ds_abs_d': row['ds_abs_d'],
                'sig_change': row['sig_change'],
            })

    corr_df = pd.DataFrame(correlation_data)

    if len(corr_df) > 3:
        rho, p = stats.spearmanr(corr_df['expr_log2fc'], corr_df['d_change'])
        print(f"  TF expression log2FC vs Δ|d|: rho={rho:.4f}, p={p:.4f}")
        if rho < 0:
            print(f"  → NEGATIVE: TF upregulation correlates with DECREASED bQTL effect")
            print(f"  → Supports saturation model")
        else:
            print(f"  → Positive/neutral: no support for saturation")

        # Also test: absolute expression level vs |d|
        rho2, p2 = stats.spearmanr(corr_df['ww_expr'], corr_df['ww_abs_d'])
        print(f"\n  WW TF expression vs WW |d|: rho={rho2:.4f}, p={p2:.4f}")
        rho3, p3 = stats.spearmanr(corr_df['ds_expr'], corr_df['ds_abs_d'])
        print(f"  DS TF expression vs DS |d|: rho={rho3:.4f}, p={p3:.4f}")

        print(f"\n  Per-family detail:")
        print(f"  {'Family':<15} {'WW expr':>8} {'DS expr':>8} {'log2FC':>7} "
              f"{'Δ|d|':>7} {'Direction'}")
        print("  " + "-" * 60)
        for _, row in corr_df.sort_values('expr_log2fc', ascending=False).iterrows():
            direction = '↑TF ↓effect' if row['expr_log2fc'] > 0.3 and row['d_change'] < 0 else \
                        '↓TF ↑effect' if row['expr_log2fc'] < -0.3 and row['d_change'] > 0 else \
                        '↑TF ↑effect' if row['expr_log2fc'] > 0.3 and row['d_change'] > 0 else \
                        '↓TF ↓effect' if row['expr_log2fc'] < -0.3 and row['d_change'] < 0 else \
                        'stable'
            print(f"  {row['tf_family']:<15} {row['ww_expr']:8.0f} {row['ds_expr']:8.0f} "
                  f"{row['expr_log2fc']:+7.2f} {row['d_change']:+7.3f} {direction}")

    # ── Condition switching stats ──
    print(f"\n{'='*60}")
    print("4. CONDITION-DEPENDENT bQTL SWITCHING")
    print(f"{'='*60}")

    ww_sig = set(zip(rdf_ww[rdf_ww['fdr_mw'] < 0.05]['chr'],
                     rdf_ww[rdf_ww['fdr_mw'] < 0.05]['pos'],
                     rdf_ww[rdf_ww['fdr_mw'] < 0.05]['gene_id']))
    ds_sig = set(zip(rdf_ds[rdf_ds['fdr_mw'] < 0.05]['chr'],
                     rdf_ds[rdf_ds['fdr_mw'] < 0.05]['pos'],
                     rdf_ds[rdf_ds['fdr_mw'] < 0.05]['gene_id']))
    both = ww_sig & ds_sig
    ww_only = ww_sig - ds_sig
    ds_only = ds_sig - ww_sig

    total_unique = len(ww_sig | ds_sig)
    print(f"  Constitutive (both): {len(both):,} ({100*len(both)/total_unique:.1f}%)")
    print(f"  WW-specific: {len(ww_only):,} ({100*len(ww_only)/total_unique:.1f}%)")
    print(f"  Drought-specific: {len(ds_only):,} ({100*len(ds_only)/total_unique:.1f}%)")
    print(f"  Total unique functional: {total_unique:,}")
    print(f"  Jaccard index: {len(both)/total_unique:.3f}")

    # Save correlation data
    corr_df.to_csv(OUTDIR / 'tf_expression_vs_bqtl_effect.csv', index=False)

    # ── Generate figure ──
    print("\n5. Generating figure...")
    make_figure(rdf_ww, rdf_ds, tf_effects, corr_df, ww_sig, ds_sig)


def make_figure(rdf_ww, rdf_ds, tf_effects, corr_df, ww_sig, ds_sig):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    try:
        from matplotlib_venn import venn2
        has_venn = True
    except ImportError:
        has_venn = False

    fig = plt.figure(figsize=(18, 14))

    # Layout: 3 rows × 3 cols, with panel E spanning bottom right
    ax_a = fig.add_subplot(2, 3, 1)
    ax_b = fig.add_subplot(2, 3, 2)
    ax_c = fig.add_subplot(2, 3, 3)
    ax_d = fig.add_subplot(2, 3, 4)
    ax_e = fig.add_subplot(2, 3, 5)
    ax_f = fig.add_subplot(2, 3, 6)

    # ── A: Condition-dependent switching (Venn) ──
    both = ww_sig & ds_sig
    ww_only = ww_sig - ds_sig
    ds_only = ds_sig - ww_sig

    if has_venn:
        venn2(subsets=(len(ww_only), len(ds_only), len(both)),
              set_labels=('Well-watered', 'Drought'),
              set_colors=('#1976D2', '#D32F2F'), alpha=0.6, ax=ax_a)
    else:
        # Fallback if matplotlib_venn not available
        ax_a.bar([0, 1, 2], [len(ww_only), len(both), len(ds_only)],
                 color=['#1976D2', '#7B1FA2', '#D32F2F'], edgecolor='white')
        ax_a.set_xticks([0, 1, 2])
        ax_a.set_xticklabels(['WW only', 'Both', 'DS only'])
        ax_a.set_ylabel('bQTL-gene pairs')

    total = len(ww_sig | ds_sig)
    ax_a.set_title(f'A. Condition-dependent bQTL switching\n'
                   f'{len(both)} constitutive, {len(ww_only)} WW-only, {len(ds_only)} drought-only\n'
                   f'Jaccard = {len(both)/total:.3f}')

    # ── B: Per-TF-family Δ|d| (bar chart) ──
    tf_sorted = tf_effects.sort_values('d_change')
    # Top and bottom 10
    top_bot = pd.concat([tf_sorted.head(10), tf_sorted.tail(5)])
    colors = ['#D32F2F' if d > 0.02 else '#1976D2' if d < -0.02 else '#757575'
              for d in top_bot['d_change']]
    ax_b.barh(range(len(top_bot)), top_bot['d_change'], color=colors,
              edgecolor='white', height=0.7)
    ax_b.set_yticks(range(len(top_bot)))
    ax_b.set_yticklabels(top_bot['tf_family'], fontsize=8)
    ax_b.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_b.set_xlabel("Δ|Cohen's d| (drought − well-watered)")
    ax_b.set_title("B. Change in bQTL effect under drought\n"
                    "Red = gained, Blue = lost")

    # ── C: TF expression change vs Δ|d| (saturation test) ──
    if len(corr_df) > 3:
        ax_c.scatter(corr_df['expr_log2fc'], corr_df['d_change'],
                     s=80, color='#7B1FA2', alpha=0.7, edgecolors='white', zorder=5)
        for _, row in corr_df.iterrows():
            ax_c.annotate(row['tf_family'],
                         (row['expr_log2fc'], row['d_change']),
                         fontsize=6, ha='center', va='bottom',
                         xytext=(0, 5), textcoords='offset points')
        # Regression line
        x = corr_df['expr_log2fc'].values
        y = corr_df['d_change'].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 3:
            slope, intercept, r, p, se = stats.linregress(x[mask], y[mask])
            x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax_c.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5)
            rho, p_s = stats.spearmanr(x[mask], y[mask])
            ax_c.text(0.05, 0.95, f'rho={rho:.3f}\np={p_s:.3f}',
                     transform=ax_c.transAxes, fontsize=9, va='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_c.axhline(0, color='gray', linestyle=':', alpha=0.3)
        ax_c.axvline(0, color='gray', linestyle=':', alpha=0.3)
        ax_c.set_xlabel('TF family expression log2FC (drought/WW)')
        ax_c.set_ylabel("Δ|Cohen's d| (drought − WW)")
        ax_c.set_title("C. TF expression change vs bQTL effect change\n"
                        "Saturation model: ↑TF → ↓effect")

    # ── D: P-value distributions ──
    ax_d.hist(rdf_ww['mw_p'], bins=50, alpha=0.5, color='#1976D2',
              density=True, edgecolor='white', label='Well-watered')
    ax_d.hist(rdf_ds['mw_p'], bins=50, alpha=0.5, color='#D32F2F',
              density=True, edgecolor='white', label='Drought')
    ax_d.axhline(1, color='gray', linestyle='--', alpha=0.5)
    ww_fdr = (rdf_ww['fdr_mw'] < 0.05).sum()
    ds_fdr = (rdf_ds['fdr_mw'] < 0.05).sum()
    ax_d.set_xlabel('Mann-Whitney p-value')
    ax_d.set_ylabel('Density')
    ax_d.set_title(f'D. P-value distributions\n'
                   f'WW: {ww_fdr} FDR hits, DS: {ds_fdr} FDR hits')
    ax_d.legend(fontsize=8)

    # ── E: Paired effect sizes (shared bQTL) ──
    shared = rdf_ww.merge(rdf_ds, on=['chr', 'pos', 'gene_id'], suffixes=('_ww', '_ds'))
    if len(shared) > 0:
        ax_e.scatter(shared['cohens_d_ww'].abs().clip(0, 5),
                     shared['cohens_d_ds'].abs().clip(0, 5),
                     s=4, alpha=0.15, color='#7B1FA2', rasterized=True)
        ax_e.plot([0, 5], [0, 5], 'k--', alpha=0.5, label='y=x')
        rho, p = stats.spearmanr(shared['cohens_d_ww'].abs(),
                                  shared['cohens_d_ds'].abs())
        ax_e.set_xlabel("|d| Well-watered")
        ax_e.set_ylabel("|d| Drought")
        ax_e.set_title(f'E. Paired bQTL effects (n={len(shared):,})\n'
                       f'rho={rho:.3f}, p={p:.2e}')
        ax_e.legend(fontsize=8)

    # ── F: Summary metrics ──
    ax_f.set_xlim(0, 10)
    ax_f.set_ylim(0, 10)
    ax_f.set_axis_off()

    # Summary text
    ww_enr = (rdf_ww['mw_p'] < 0.05).sum() / (0.05 * len(rdf_ww))
    ds_enr = (rdf_ds['mw_p'] < 0.05).sum() / (0.05 * len(rdf_ds))
    ww_pi1 = 1 - min(1, 2 * (rdf_ww['mw_p'] > 0.5).mean())
    ds_pi1 = 1 - min(1, 2 * (rdf_ds['mw_p'] > 0.5).mean())

    text_lines = [
        ("Regulatory Configuration Model", 9.8, 14, 'bold'),
        ("", 0, 0, 'normal'),
        (f"Well-watered:", 9.0, 11, 'bold'),
        (f"  {ww_fdr} FDR hits • {ww_enr:.1f}x enrich • pi1={ww_pi1:.3f}", 8.5, 9, 'normal'),
        (f"  Mean |d| = {rdf_ww['cohens_d'].abs().mean():.3f}", 8.0, 9, 'normal'),
        ("", 0, 0, 'normal'),
        (f"Drought:", 7.2, 11, 'bold'),
        (f"  {ds_fdr} FDR hits • {ds_enr:.1f}x enrich • pi1={ds_pi1:.3f}", 6.7, 9, 'normal'),
        (f"  Mean |d| = {rdf_ds['cohens_d'].abs().mean():.3f}", 6.2, 9, 'normal'),
        ("", 0, 0, 'normal'),
        (f"Switching:", 5.2, 11, 'bold'),
        (f"  {len(ww_sig & ds_sig)} constitutive ({100*len(ww_sig&ds_sig)/total:.0f}%)", 4.7, 9, 'normal'),
        (f"  {len(ww_sig - ds_sig)} WW-specific", 4.2, 9, 'normal'),
        (f"  {len(ds_sig - ww_sig)} drought-specific", 3.7, 9, 'normal'),
        ("", 0, 0, 'normal'),
        (f"Key insight:", 2.7, 11, 'bold'),
        ("  bQTL functionality is gated by", 2.2, 9, 'normal'),
        ("  TF concentration landscape.", 1.7, 9, 'normal'),
        ("  Same variant, different impact.", 1.2, 9, 'normal'),
    ]

    for text, y, fontsize, weight in text_lines:
        if text:
            ax_f.text(0.5, y/10, text, fontsize=fontsize, fontweight=weight,
                     transform=ax_f.transAxes, va='top')

    ax_f.set_title('F. Summary')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig16_regulatory_configuration.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(FIG_DIR / 'fig16_regulatory_configuration.png', bbox_inches='tight', dpi=150)
    print(f"  Saved: figures/fig16_regulatory_configuration.pdf")
    plt.close()


if __name__ == '__main__':
    main()
