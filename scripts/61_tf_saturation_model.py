#!/usr/bin/env python3
"""
61_tf_saturation_model.py — TF dose-dependent binding variation sensitivity.

Part of: "Condition-dependent bQTL switching" paper (Banf & Hartwig)
Paper section: "Dose-dependent TF binding variation sensitivity"
Pipeline step: 9 of 9

Tests whether TF expression level modulates the effect of motif disruption on
allele-specific expression (ASE). The 25 Engelhorn hybrids (B73 x NAM founder)
serve as a natural TF dose series — each hybrid has different TF expression
levels (trans variation) and different motif disruption profiles (cis variation
via founder bQTL genotypes).

Core model for each TF family f, target gene g, hybrid h:

    ASE(g,h) ~ beta1 * disrupted(f,g,h) + beta2 * TF_expr(f,h)
               + beta3 * disrupted * TF_expr + epsilon

    beta3 > 0: sub-saturated (motif loss hurts more when TF is scarce)
    beta3 ~ 0: saturation-independent
    beta3 < 0: buffered (high TF rescues motif loss — saturation)

Since ASE controls for all trans effects within a cell, the only cis
difference is founder allele (motif intact vs disrupted). TF expression
varies across hybrids in trans, providing the "dose" axis.

Input:
  data/processed/nam_founder_genotypes_at_bqtl.tsv (script 53b)
  data/processed/engelhorn_ase_ww.csv (per-hybrid ASE)
  data/processed/engelhorn_ww_vs_ds_expression.tsv (mean expression)
  data/raw/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1_entap_results.tsv.gz
  results/.../bound_region_motifs.csv (script 55 prerequisite)
  results/.../genotype_bqtl_results.csv (script 54)

Output:
  tf_saturation_results.csv — per-family interaction statistics
  figures/fig_tf_saturation.pdf — dose-response visualization
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
OUTDIR = BASE / "results"
FIG_DIR = BASE / "figures" / "paper_figures"
FIG_DIR.mkdir(exist_ok=True)

# Hybrid name mapping: ASE uses "B73xB97", genotype uses "B97"
# Need to map between them
FOUNDER_TO_HYBRID = {
    'B97': 'B73xB97', 'CML247': 'B73xCML247', 'CML277': 'B73xCML277',
    'CML322': 'B73xCML322', 'CML333': 'B73xCML333', 'CML69': 'B73xCML69',
    'HP301': 'B73xHP301', 'Il14H': 'B73xIL14H', 'Ki11': 'B73xKi11',
    'Ki3': 'B73xKi3', 'Ky21': 'B73xKy21', 'M162W': 'B73xM162W',
    'Mo18W': 'B73xMo18W', 'Ms71': 'B73xMs71', 'NC358': 'B73xNC358',
    'Oh43': 'B73xOh43', 'Oh7B': 'B73xOh7b', 'P39': 'B73xP39',
    'Tx303': 'B73xTx303',
}
HYBRID_TO_FOUNDER = {v: k for k, v in FOUNDER_TO_HYBRID.items()}


# ── TF family identification ──

def identify_tf_genes():
    """Identify maize TF genes by family from EntAP annotations.

    Uses conservative keyword matching to avoid false positives.
    Returns dict of {gene_id: tf_family}.
    """
    entap = pd.read_csv(RAW / "Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1_entap_results.tsv.gz",
                         sep='\t', low_memory=False)
    entap['gene_id'] = entap['Query Sequence'].str.replace(r'_T\d+$', '', regex=True).str.split('.').str[0]

    # Conservative keywords — require specific domain names to avoid false positives
    tf_keywords = {
        'WRKY': ['WRKY transcription', 'WRKY DNA-binding'],
        'MYB': ['MYB transcription', 'MYB domain', 'MYB-related', 'R2R3-MYB'],
        'NAC': ['NAC domain', 'NAC transcription', 'NAM/ATAF/CUC'],
        'bHLH': ['bHLH', 'basic helix-loop-helix transcription'],
        'bZIP': ['bZIP transcription', 'basic leucine zipper'],
        'ERF': ['ethylene-responsive transcription', 'AP2/ERF', 'ERF transcription'],
        'C2H2': ['C2H2-type zinc finger', 'C2H2 zinc finger'],
        'TCP': ['TCP transcription', 'TCP family', 'TEOSINTE BRANCHED'],
        'HSF': ['heat shock factor', 'heat stress transcription'],
        'HD-ZIP': ['HD-ZIP', 'homeodomain-leucine zipper'],
        'Dof': ['Dof zinc finger', 'Dof domain', 'DOF transcription'],
        'ARF': ['auxin response factor'],
        'SBP': ['SBP domain', 'SQUAMOSA promoter', 'SPL transcription'],
        'GATA': ['GATA transcription', 'GATA zinc finger'],
        'G2-like': ['GLK transcription', 'G2-like', 'Golden2-like'],
        'BES1': ['BES1', 'BZR1', 'brassinosteroid signaling'],
        'CAMTA': ['CAMTA', 'calmodulin-binding transcription'],
        'YABBY': ['YABBY'],
        'WOX': ['WOX', 'WUSCHEL'],
        'Trihelix': ['Trihelix', 'GT factor'],
        'BBR-BPC': ['BBR-BPC', 'BASIC PENTACYSTEINE'],
        'LBD': ['LBD', 'LOB domain'],
        'TALE': ['TALE homeodomain', 'KNOX', 'BEL1-like'],
        'E2F/DP': ['E2F', 'DP transcription'],
        'GRAS': ['GRAS domain', 'SCARECROW', 'DELLA'],
        'MIKC_MADS': ['MADS-box', 'MIKC'],
        'Nin-like': ['NIN-like', 'Nin-like', 'NLP'],
        'CPP': ['CPP domain', 'CXC domain', 'tesmin'],
    }

    tf_genes = {}
    for _, row in entap.iterrows():
        gid = row['gene_id']
        if gid in tf_genes:
            continue
        for col in ['EggNOG Description', 'Description']:
            desc = str(row.get(col, ''))
            if desc == 'nan':
                continue
            for fam, keywords in tf_keywords.items():
                if any(kw.lower() in desc.lower() for kw in keywords):
                    tf_genes[gid] = fam
                    break
            if gid in tf_genes:
                break

    fam_counts = Counter(tf_genes.values())
    print(f"TF genes identified: {len(tf_genes)} across {len(fam_counts)} families")
    for fam, n in fam_counts.most_common(10):
        print(f"  {fam}: {n}")

    return tf_genes


def compute_per_hybrid_tf_expression(ase_df, tf_genes):
    """Compute TF family expression per hybrid from ASE total counts.

    For each hybrid h and TF family f, sum b73_mean + nam_mean across all
    TF genes of that family to get total family expression. Then take the
    median across TF genes as the representative "dose".

    Returns DataFrame: hybrid, tf_family, tf_expr_median, tf_expr_sum, n_tf_genes
    """
    # Filter ASE to TF genes
    tf_ase = ase_df[ase_df.gene_id.isin(tf_genes)].copy()
    tf_ase['tf_family'] = tf_ase.gene_id.map(tf_genes)
    tf_ase['total_expr'] = tf_ase['b73_mean'] + tf_ase['nam_mean']

    # Aggregate per hybrid × family
    agg = tf_ase.groupby(['hybrid', 'tf_family']).agg(
        tf_expr_median=('total_expr', 'median'),
        tf_expr_sum=('total_expr', 'sum'),
        tf_expr_mean=('total_expr', 'mean'),
        n_tf_genes=('gene_id', 'nunique'),
    ).reset_index()

    print(f"\nPer-hybrid TF expression: {len(agg)} entries "
          f"({agg.hybrid.nunique()} hybrids × {agg.tf_family.nunique()} families)")

    return agg


def build_disruption_matrix(genotypes, motif_data):
    """For each bQTL × founder, determine if the founder allele disrupts each TF family motif.

    Returns DataFrame: chr, pos, founder, tf_family, disrupted (0/1)
    """
    founders = [c for c in genotypes.columns if c not in ('chr', 'pos', 'ref')]

    # Get motif disruptions per bQTL position
    motifs_at_var = motif_data[
        (motif_data.tf_family != 'NONE') & (motif_data.motifs_at_variant > 0)
    ][['chr', 'pos', 'tf_family']].drop_duplicates()

    print(f"\nbQTL with motif at variant: {motifs_at_var[['chr','pos']].drop_duplicates().shape[0]:,}")
    print(f"TF families: {motifs_at_var.tf_family.nunique()}")

    # For each (chr, pos, tf_family), check which founders have the variant allele
    # Variant allele = anything other than '0|0' and './.'
    rows = []
    merged = motifs_at_var.merge(genotypes, on=['chr', 'pos'], how='inner')
    print(f"Merged with genotypes: {len(merged):,}")

    for founder in founders:
        sub = merged[['chr', 'pos', 'tf_family', founder]].copy()
        sub['founder'] = founder
        # disrupted = 1 if founder has variant allele (motif disrupted in founder)
        sub['disrupted'] = (~sub[founder].isin(['0|0', './.'])).astype(int)
        rows.append(sub[['chr', 'pos', 'tf_family', 'founder', 'disrupted']])

    result = pd.concat(rows, ignore_index=True)
    print(f"Disruption matrix: {len(result):,} entries")
    return result


def build_analysis_table(ase_df, genotypes, motif_data, tf_genes,
                          geno_results):
    """Build the main analysis table combining all data sources (vectorized).

    For each (bQTL, gene, TF_family, hybrid):
      - ASE effect (log2 ratio = founder/B73)
      - Whether founder allele disrupts the TF motif at this bQTL
      - TF family expression in this hybrid

    Returns DataFrame ready for regression.
    """
    founders = [c for c in genotypes.columns if c not in ('chr', 'pos', 'ref')]

    # Step 1: Get bQTL-gene pairs with motif info
    motifs_at_var = motif_data[
        (motif_data.tf_family != 'NONE') & (motif_data.motifs_at_variant > 0)
    ][['chr', 'pos', 'gene_id', 'tf_family', 'motifs_at_variant',
       'redundant_in_region']].copy()
    motifs_at_var = motifs_at_var.dropna(subset=['gene_id'])
    print(f"Motif-at-variant entries with gene_id: {len(motifs_at_var):,}")

    # Step 2: Get per-hybrid TF expression
    tf_expr = compute_per_hybrid_tf_expression(ase_df, tf_genes)

    # Step 3: Melt genotypes to long format (chr, pos, founder, genotype)
    print("  Melting genotype matrix...")
    geno_long = genotypes.melt(
        id_vars=['chr', 'pos', 'ref'],
        value_vars=founders,
        var_name='founder',
        value_name='allele'
    )
    # Filter out missing genotypes and map to hybrid names
    geno_long = geno_long[geno_long['allele'] != './.'].copy()
    geno_long['hybrid'] = geno_long.founder.map(FOUNDER_TO_HYBRID)
    geno_long = geno_long.dropna(subset=['hybrid'])
    geno_long['disrupted'] = (geno_long['allele'] != '0|0').astype(int)
    print(f"  Genotype long: {len(geno_long):,} entries")

    # Step 4: Merge motifs × genotypes
    print("  Merging motifs x genotypes...")
    merged = motifs_at_var.merge(
        geno_long[['chr', 'pos', 'founder', 'hybrid', 'disrupted']],
        on=['chr', 'pos'],
        how='inner'
    )
    print(f"  Motif x genotype: {len(merged):,} entries")

    # Step 5: Merge with ASE (only keep genes with sufficient reads)
    print("  Merging with ASE...")
    ase_filt = ase_df[ase_df.total_reads >= 10][
        ['gene_id', 'hybrid', 'log2_ratio', 'abs_log2_ratio', 'total_reads']
    ].copy()
    merged = merged.merge(ase_filt, on=['gene_id', 'hybrid'], how='inner')
    merged.rename(columns={'log2_ratio': 'log2_ase',
                            'abs_log2_ratio': 'abs_log2_ase'}, inplace=True)
    print(f"  After ASE merge: {len(merged):,} entries")

    # Step 6: Merge with TF expression
    print("  Merging with TF expression...")
    merged = merged.merge(
        tf_expr[['hybrid', 'tf_family', 'tf_expr_median']],
        on=['hybrid', 'tf_family'],
        how='inner'
    )
    merged.rename(columns={'tf_expr_median': 'tf_expr'}, inplace=True)
    print(f"  After TF expr merge: {len(merged):,} entries")

    # Rename columns
    merged.rename(columns={
        'motifs_at_variant': 'n_motifs_at_var',
        'redundant_in_region': 'n_redundant',
    }, inplace=True)

    df = merged[['chr', 'pos', 'gene_id', 'tf_family', 'founder', 'hybrid',
                  'disrupted', 'log2_ase', 'abs_log2_ase', 'total_reads',
                  'tf_expr', 'n_motifs_at_var', 'n_redundant']].copy()

    print(f"\nAnalysis table: {len(df):,} records")
    print(f"  Unique bQTL: {df[['chr','pos']].drop_duplicates().shape[0]:,}")
    print(f"  Unique genes: {df.gene_id.nunique():,}")
    print(f"  Unique families: {df.tf_family.nunique()}")
    print(f"  Disrupted fraction: {df.disrupted.mean():.3f}")
    return df


def fit_saturation_model(analysis_df):
    """Fit the interaction model per TF family.

    Model: abs(ASE) ~ disrupted + log_tf_expr + disrupted:log_tf_expr

    The interaction term tests whether the effect of motif disruption
    depends on TF expression level.

    Returns DataFrame with per-family results.
    """
    results = []

    # Log-transform TF expression (add pseudocount)
    analysis_df = analysis_df.copy()
    analysis_df['log_tf_expr'] = np.log2(analysis_df['tf_expr'] + 1)

    # Standardize TF expression within each family for comparable betas
    for fam in analysis_df.tf_family.unique():
        mask = analysis_df.tf_family == fam
        vals = analysis_df.loc[mask, 'log_tf_expr']
        if vals.std() > 0:
            analysis_df.loc[mask, 'log_tf_expr_z'] = (vals - vals.mean()) / vals.std()
        else:
            analysis_df.loc[mask, 'log_tf_expr_z'] = 0

    for fam, fam_df in analysis_df.groupby('tf_family'):
        n = len(fam_df)
        if n < 30:
            continue
        n_disrupted = fam_df.disrupted.sum()
        n_intact = (fam_df.disrupted == 0).sum()
        if n_disrupted < 10 or n_intact < 10:
            continue

        # Simple approach: compare ASE effect sizes
        ase_disrupted = fam_df.loc[fam_df.disrupted == 1, 'abs_log2_ase']
        ase_intact = fam_df.loc[fam_df.disrupted == 0, 'abs_log2_ase']

        # Mann-Whitney test: does disruption increase ASE?
        mw_stat, mw_p = stats.mannwhitneyu(ase_disrupted, ase_intact,
                                             alternative='greater')

        # Interaction model via OLS
        # Y = abs_log2_ase
        # X = [1, disrupted, log_tf_expr_z, disrupted * log_tf_expr_z]
        Y = fam_df['abs_log2_ase'].values
        X = np.column_stack([
            np.ones(n),
            fam_df['disrupted'].values,
            fam_df['log_tf_expr_z'].values,
            fam_df['disrupted'].values * fam_df['log_tf_expr_z'].values,
        ])

        try:
            betas, residuals, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
            # Standard errors via residual variance
            if len(residuals) > 0:
                mse = residuals[0] / (n - 4)
            else:
                mse = np.sum((Y - X @ betas) ** 2) / (n - 4)
            se = np.sqrt(np.diag(mse * np.linalg.pinv(X.T @ X)))
            t_stats = betas / se
            p_values = 2 * stats.t.sf(np.abs(t_stats), df=n - 4)

            beta_intercept, beta_disrupted, beta_tf_expr, beta_interaction = betas
            se_interaction = se[3]
            p_interaction = p_values[3]
            t_interaction = t_stats[3]

        except Exception:
            beta_disrupted = beta_tf_expr = beta_interaction = np.nan
            se_interaction = p_interaction = t_interaction = np.nan

        # Correlation: TF expression vs ASE effect among disrupted only
        if n_disrupted >= 10:
            rho, rho_p = stats.spearmanr(
                fam_df.loc[fam_df.disrupted == 1, 'log_tf_expr'],
                fam_df.loc[fam_df.disrupted == 1, 'abs_log2_ase']
            )
        else:
            rho = rho_p = np.nan

        # Median ASE split by TF expression tertiles (among disrupted)
        if n_disrupted >= 15:
            d_df = fam_df[fam_df.disrupted == 1].copy()
            d_df['tf_tertile'] = pd.qcut(d_df['log_tf_expr'], 3,
                                          labels=['low', 'mid', 'high'],
                                          duplicates='drop')
            if d_df.tf_tertile.nunique() == 3:
                ase_low = d_df.loc[d_df.tf_tertile == 'low', 'abs_log2_ase'].median()
                ase_high = d_df.loc[d_df.tf_tertile == 'high', 'abs_log2_ase'].median()
            else:
                ase_low = ase_high = np.nan
        else:
            ase_low = ase_high = np.nan

        results.append({
            'tf_family': fam,
            'n_total': n,
            'n_disrupted': n_disrupted,
            'n_intact': n_intact,
            'mean_ase_disrupted': ase_disrupted.mean(),
            'mean_ase_intact': ase_intact.mean(),
            'ase_diff': ase_disrupted.mean() - ase_intact.mean(),
            'mw_p': mw_p,
            'beta_disrupted': beta_disrupted,
            'beta_tf_expr': beta_tf_expr,
            'beta_interaction': beta_interaction,
            'se_interaction': se_interaction,
            't_interaction': t_interaction,
            'p_interaction': p_interaction,
            'spearman_rho': rho,
            'spearman_p': rho_p,
            'ase_tf_low': ase_low,
            'ase_tf_high': ase_high,
        })

    result_df = pd.DataFrame(results)

    # FDR correction
    if len(result_df) > 1:
        for col in ['mw_p', 'p_interaction', 'spearman_p']:
            valid = result_df[col].notna()
            if valid.sum() > 1:
                _, fdr, _, _ = multipletests(result_df.loc[valid, col],
                                              method='fdr_bh')
                result_df.loc[valid, f'fdr_{col}'] = fdr

    return result_df.sort_values('p_interaction')


def plot_results(result_df, analysis_df, outpath):
    """Create multi-panel saturation figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # ── Panel A: Interaction coefficient forest plot ──
    ax = axes[0]
    df = result_df.dropna(subset=['beta_interaction']).sort_values('beta_interaction')
    n_fam = len(df)
    y_pos = np.arange(n_fam)

    colors = []
    for _, row in df.iterrows():
        if row.get('fdr_p_interaction', 1) < 0.05:
            colors.append('#d62728' if row['beta_interaction'] > 0 else '#1f77b4')
        else:
            colors.append('#888888')

    ax.barh(y_pos, df['beta_interaction'], xerr=df['se_interaction'] * 1.96,
            color=colors, alpha=0.7, height=0.7, capsize=2, ecolor='#555555')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['tf_family'], fontsize=8)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xlabel('Interaction coefficient (beta_3)', fontsize=10)
    ax.set_title('A. TF dose x motif disruption\ninteraction', fontsize=11)
    ax.text(0.02, 0.98, 'sub-saturated\n(motif loss hurts\nmore when TF low)',
            transform=ax.transAxes, fontsize=7, va='top', color='#d62728', alpha=0.7)
    ax.text(0.98, 0.98, 'buffered\n(high TF rescues\nmotif loss)',
            transform=ax.transAxes, fontsize=7, va='top', ha='right',
            color='#1f77b4', alpha=0.7)

    # ── Panel B: ASE by TF expression tertile (top families) ──
    ax = axes[1]
    top_fams = result_df.nsmallest(6, 'p_interaction')['tf_family'].tolist()

    analysis_df = analysis_df.copy()
    analysis_df['log_tf_expr'] = np.log2(analysis_df['tf_expr'] + 1)

    bar_data = []
    for fam in top_fams:
        d = analysis_df[(analysis_df.tf_family == fam) & (analysis_df.disrupted == 1)]
        if len(d) < 15:
            continue
        try:
            d = d.copy()
            d['tertile'] = pd.qcut(d['log_tf_expr'], 3,
                                    labels=['Low TF', 'Mid TF', 'High TF'],
                                    duplicates='drop')
            if d.tertile.nunique() == 3:
                for t in ['Low TF', 'Mid TF', 'High TF']:
                    bar_data.append({
                        'family': fam,
                        'tertile': t,
                        'median_ase': d.loc[d.tertile == t, 'abs_log2_ase'].median(),
                        'n': (d.tertile == t).sum(),
                    })
        except Exception:
            continue

    if bar_data:
        bar_df = pd.DataFrame(bar_data)
        families = bar_df.family.unique()
        x = np.arange(len(families))
        width = 0.25
        tertile_colors = {'Low TF': '#d62728', 'Mid TF': '#ff7f0e', 'High TF': '#2ca02c'}

        for i, t in enumerate(['Low TF', 'Mid TF', 'High TF']):
            vals = [bar_df[(bar_df.family == f) & (bar_df.tertile == t)]['median_ase'].values
                    for f in families]
            vals = [v[0] if len(v) > 0 else 0 for v in vals]
            ax.bar(x + i * width, vals, width, label=t,
                   color=tertile_colors[t], alpha=0.8)

        ax.set_xticks(x + width)
        ax.set_xticklabels(families, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Median |log2 ASE| when disrupted', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
    ax.set_title('B. ASE effect by TF expression\ntertile (disrupted sites)', fontsize=11)

    # ── Panel C: Scatter — enrichment OR vs interaction beta ──
    ax = axes[2]

    # Enrichment ORs from script 26
    enrichment_ors = {
        'BES1': 2.72, 'E2F/DP': 2.40, 'CAMTA': 1.92, 'BBR-BPC': 1.68,
        'bHLH': 1.57, 'bZIP': 1.52, 'TCP': 1.47, 'MYB': 1.39,
        'ERF': 1.35, 'NAC': 1.33, 'C2H2': 1.33, 'SBP': 1.27,
        'Trihelix': 1.22, 'G2-like': 1.19, 'ARF': 1.16, 'B3': 1.14,
        'LBD': 1.13, 'Nin-like': 1.32,
        'YABBY': 0.60, 'WOX': 0.73, 'HD-ZIP': 0.75, 'CPP': 0.75,
        'HSF': 0.77, 'Dof': 0.82, 'GATA': 0.85, 'GRAS': 0.87,
        'TALE': 0.88, 'MIKC_MADS': 0.89, 'M-type_MADS': 0.93, 'AP2': 0.94,
    }

    for _, row in result_df.iterrows():
        fam = row['tf_family']
        if fam not in enrichment_ors or np.isnan(row['beta_interaction']):
            continue
        log2or = np.log2(enrichment_ors[fam])
        beta = row['beta_interaction']
        sig = row.get('fdr_p_interaction', 1) < 0.05
        color = '#d62728' if log2or > 0 else '#1f77b4'
        marker = 'o' if sig else 'x'
        ax.scatter(log2or, beta, c=color, marker=marker, s=60, alpha=0.8)
        ax.annotate(fam, (log2or, beta), fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords='offset points')

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('log2(Enrichment OR) from script 26', fontsize=10)
    ax.set_ylabel('Interaction coefficient (beta_3)', fontsize=10)
    ax.set_title('C. Enrichment vs dose-dependent\nsensitivity', fontsize=11)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {outpath}")
    plt.close()


def main():
    print("=" * 70)
    print("TF DOSE-DEPENDENT BINDING VARIATION SENSITIVITY")
    print("(Saturation model: 25 hybrids as natural TF dose series)")
    print("=" * 70)

    # ── Load data ──
    print("\n── Loading data ──")

    # 1. Genotypes
    genotypes = pd.read_csv(PROC / "nam_founder_genotypes_at_bqtl.tsv", sep='\t')
    print(f"Genotypes: {len(genotypes):,} bQTL x "
          f"{len([c for c in genotypes.columns if c not in ('chr','pos','ref')])} founders")

    # 2. ASE
    ase = pd.read_csv(PROC / "engelhorn_ase_ww.csv")
    print(f"ASE: {len(ase):,} gene-hybrid pairs")

    # 3. Motif disruption data
    motif_path = OUTDIR / "bound_region_motifs.csv"
    motifs = pd.read_csv(motif_path)
    print(f"Motifs: {len(motifs):,} bQTL-gene-family entries")

    # 4. Genotype-bQTL results (for bQTL-gene pairs)
    geno_results = pd.read_csv(OUTDIR / "genotype_bqtl_results.csv")
    print(f"Genotype-bQTL pairs: {len(geno_results):,}")

    # 5. TF gene identification
    print(f"\n── Identifying TF genes ──")
    tf_genes = identify_tf_genes()

    # ── Build analysis table ──
    print(f"\n── Building analysis table ──")
    analysis_df = build_analysis_table(ase, genotypes, motifs, tf_genes,
                                        geno_results)

    if len(analysis_df) == 0:
        print("ERROR: No analysis records built")
        return

    # ── Fit saturation models ──
    print(f"\n── Fitting interaction models ──")
    result_df = fit_saturation_model(analysis_df)

    # ── Print results ──
    print(f"\n{'='*70}")
    print(f"RESULTS: Dose-dependent binding variation sensitivity")
    print(f"{'='*70}")

    print(f"\nTested {len(result_df)} TF families")

    sig = result_df[result_df.get('fdr_p_interaction', pd.Series(dtype=float)).fillna(1) < 0.05]
    print(f"Significant interaction (FDR<0.05): {len(sig)} families")

    if len(sig) > 0:
        print(f"\nSignificant families:")
        for _, row in sig.iterrows():
            direction = "sub-saturated" if row['beta_interaction'] > 0 else "buffered"
            print(f"  {row['tf_family']:15s}: beta3={row['beta_interaction']:+.4f} "
                  f"(p={row['p_interaction']:.2e}), {direction}")

    print(f"\nAll families (sorted by interaction p-value):")
    for _, row in result_df.head(20).iterrows():
        fdr = row.get('fdr_p_interaction', np.nan)
        sig_marker = '*' if fdr < 0.05 else ' '
        direction = '+' if row['beta_interaction'] > 0 else '-'
        print(f" {sig_marker} {row['tf_family']:15s}: beta3={row['beta_interaction']:+.4f} "
              f"(p={row['p_interaction']:.2e}, FDR={fdr:.3f}), "
              f"n={row['n_total']}, rho={row['spearman_rho']:+.3f}")

    # ── Interpretation ──
    print(f"\n── Interpretation ──")
    pos_int = result_df[result_df.beta_interaction > 0]
    neg_int = result_df[result_df.beta_interaction < 0]
    print(f"Positive interaction (sub-saturated): {len(pos_int)} families")
    print(f"Negative interaction (buffered/saturated): {len(neg_int)} families")

    if len(result_df) > 5:
        mean_beta = result_df['beta_interaction'].mean()
        print(f"Mean interaction coefficient: {mean_beta:+.4f}")
        _, global_p = stats.wilcoxon(result_df['beta_interaction'].dropna())
        print(f"Global test (Wilcoxon): p={global_p:.4f}")

    # ── Save results ──
    out_csv = OUTDIR / "tf_saturation_results.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"\nResults saved: {out_csv}")

    # Save analysis table for downstream use
    analysis_out = OUTDIR / "tf_saturation_analysis_table.csv"
    analysis_df.to_csv(analysis_out, index=False)
    print(f"Analysis table saved: {analysis_out}")

    # ── Plot ──
    plot_results(result_df, analysis_df, FIG_DIR / "fig_tf_saturation.pdf")


if __name__ == '__main__':
    main()
