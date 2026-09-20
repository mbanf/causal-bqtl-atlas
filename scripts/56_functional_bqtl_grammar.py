#!/usr/bin/env python3
"""
56_functional_bqtl_grammar.py — What distinguishes functional from non-functional bQTL.

Part of: "Condition-dependent bQTL switching" paper (Banf & Hartwig)
Paper section: "Regulatory grammar of functional bQTL"
Pipeline step: 4 of 8

Key results: TSS proximity (mean 521 vs 649 bp, p=1.2e-6) and gene expression
level (mean 698 vs 1000 CPM, p=0.007) distinguish functional from non-functional.
TF motif redundancy and specific TF identity do NOT.

For genes with genotype→ASE (script 54), characterize the regulatory landscape:
  1. Which TF motifs are disrupted at the variant?
  2. Which of those TFs are expressed?
  3. How are motifs distributed across peaks in the promoter?
  4. What motif combinations (grammar) distinguish functional from non-functional bQTL?
  5. Does the number of bQTL per promoter correlate with ASE magnitude?

Focuses on the complete regulatory chain:
  Expressed TF → motif at variant → bQTL in peak → target gene → ASE

Input:
  genotype_bqtl_results.csv      (script 54: genotype→ASE pairs)
  bound_region_motifs.csv         (script 47: motifs per bQTL region)
  regulatory_chains.csv           (script 41: TF→bQTL→target with expression)
  pan_cistrome_peaks.csv          (peak coordinates)
  gene_peak_disruption.csv        (per-gene peak stats)
  engelhorn_ww_vs_ds_expression.tsv (expression levels)
  bqtl_snp_ww.csv                (all bQTL)

Output:
  functional_bqtl_grammar.csv
  figures/fig13_regulatory_grammar.pdf
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUTDIR = BASE / "results"
FIG_DIR = BASE / "figures" / "paper_figures"
FIG_DIR.mkdir(exist_ok=True)
RESULTS = BASE / "results"

# Input files
GENO_RESULTS = OUTDIR / "genotype_bqtl_results.csv"
MOTIF_DATA = OUTDIR / "bound_region_motifs.csv"
CHAIN_DATA = BASE.parent / "bqtl_predict" / "results" / "regulatory_network" / "regulatory_chains.csv"
PEAKS = OUTDIR / "pan_cistrome_peaks.csv"
GENE_PEAKS = OUTDIR / "gene_peak_disruption.csv"
BQTL = DATA / "processed" / "bqtl_snp_ww.csv"
EXPR = DATA / "processed" / "engelhorn_ww_vs_ds_expression.tsv"
GFF3 = DATA / "raw" / "Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3"

VALID_CHROMS = {f'chr{i}' for i in range(1, 11)}
PROMOTER_BP = 2000


def parse_gff3():
    genes = {}
    with open(GFF3) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'gene':
                continue
            chrom = parts[0]
            if chrom not in VALID_CHROMS:
                continue
            start, end = int(parts[3]), int(parts[4])
            strand = parts[6]
            gene_id = None
            for attr in parts[8].split(';'):
                if attr.startswith('ID=gene:'):
                    gene_id = attr.split(':')[1]
                    break
                elif attr.startswith('ID=Zm'):
                    gene_id = attr[3:]
                    break
            if gene_id:
                tss = start if strand == '+' else end
                genes[gene_id] = {'chr': chrom, 'start': start, 'end': end,
                                  'strand': strand, 'tss': tss}
    return genes


def main():
    print("=" * 70)
    print("REGULATORY GRAMMAR AT FUNCTIONAL bQTL")
    print("=" * 70)

    # ── Load data ──
    print("\n1. Loading data...")
    geno = pd.read_csv(GENO_RESULTS)
    motifs = pd.read_csv(MOTIF_DATA)
    chains = pd.read_csv(CHAIN_DATA)
    peaks = pd.read_csv(PEAKS)
    gene_peaks = pd.read_csv(GENE_PEAKS)
    bqtl_df = pd.read_csv(BQTL)
    expr = pd.read_csv(EXPR, sep='\t')
    genes = parse_gff3()

    print(f"  Genotype→ASE results: {len(geno):,} pairs")
    print(f"  Motif scan: {len(motifs):,} rows")
    print(f"  Regulatory chains: {len(chains):,}")
    print(f"  Pan-cistrome peaks: {len(peaks):,}")
    print(f"  Gene peak data: {len(gene_peaks):,}")
    print(f"  Expression data: {len(expr):,} genes")

    # Label functional vs non-functional
    geno['functional'] = geno['fdr_mw'] < 0.05
    n_func = geno['functional'].sum()
    n_nonfunc = (~geno['functional']).sum()
    print(f"\n  Functional (FDR<0.05): {n_func:,}")
    print(f"  Non-functional: {n_nonfunc:,}")

    # ================================================================
    # 2. Identify disrupted TF motifs and whether TF is expressed
    # ================================================================
    print(f"\n{'='*60}")
    print("2. DISRUPTED MOTIFS + TF EXPRESSION AT FUNCTIONAL bQTL")
    print(f"{'='*60}")

    # Merge genotype results with motif data
    merged = geno.merge(motifs, on=['chr', 'pos', 'gene_id'], how='inner',
                         suffixes=('', '_motif'))

    # For each bQTL-gene pair, get the disrupted TF families
    # (families with motifs_at_variant > 0)
    disrupted = merged[merged['motifs_at_variant'] > 0].copy()
    print(f"  bQTL-gene-TF rows with motif at variant: {len(disrupted):,}")

    # Merge with regulatory chains to get TF expression
    # chains has: chr, bqtl_pos, tf_family, tf_expressed, target_gene, chain_complete
    disrupted_expr = disrupted.merge(
        chains[['chr', 'bqtl_pos', 'tf_family', 'tf_expressed', 'chain_complete',
                'target_gene', 'motif_score']].rename(
                    columns={'bqtl_pos': 'pos', 'target_gene': 'chain_target'}),
        on=['chr', 'pos', 'tf_family'],
        how='left'
    )

    # For cases where chain target matches our gene
    disrupted_expr['chain_matches'] = disrupted_expr['chain_target'] == disrupted_expr['gene_id']

    print(f"  With chain data: {disrupted_expr['tf_expressed'].notna().sum():,}")
    print(f"  TF expressed: {(disrupted_expr['tf_expressed'] == True).sum():,}")
    print(f"  Complete chain (TF expressed + target expressed): {(disrupted_expr['chain_complete'] == True).sum():,}")
    print(f"  Chain matches gene: {disrupted_expr['chain_matches'].sum():,}")

    # ── Per bQTL-gene pair: summarize regulatory grammar ──
    print(f"\n{'='*60}")
    print("3. BUILDING PER-PAIR REGULATORY GRAMMAR FEATURES")
    print(f"{'='*60}")

    # Build expression lookup
    expr_lookup = dict(zip(expr['gene_id'], expr['mean_expression']))

    # Build a per-pair feature table
    pair_features = []
    pair_groups = merged.groupby(['chr', 'pos', 'gene_id'])

    for (chrom, pos, gene_id), group in pair_groups:
        # Basic genotype→ASE info
        row0 = group.iloc[0]
        functional = row0['functional']
        cohens_d = row0['cohens_d']
        mw_p = row0['mw_p']
        fdr = row0['fdr_mw']
        abs_ase_diff = row0['abs_ase_diff']

        # Motif features from the scanned region
        families_at_var = group[group['motifs_at_variant'] > 0]['tf_family'].tolist()
        n_families_at_var = len(families_at_var)
        total_motifs_at_var = group[group['motifs_at_variant'] > 0]['motifs_at_variant'].sum()

        # Region-level features (same across all TF rows for this pair)
        n_families_in_region = row0.get('n_families_in_region', 0)
        total_motifs_in_region = row0.get('total_motifs_in_region', 0)

        # Redundancy: for disrupted families, how many copies nearby?
        redundancy_scores = group[group['motifs_at_variant'] > 0]['redundant_in_region'].values
        mean_redundancy = redundancy_scores.mean() if len(redundancy_scores) > 0 else 0
        max_redundancy = redundancy_scores.max() if len(redundancy_scores) > 0 else 0
        min_redundancy = redundancy_scores.min() if len(redundancy_scores) > 0 else 0

        # Check if disrupted TF is expressed (from chains)
        chain_rows = disrupted_expr[
            (disrupted_expr['chr'] == chrom) &
            (disrupted_expr['pos'] == pos) &
            (disrupted_expr['gene_id'] == gene_id) &
            (disrupted_expr['chain_matches'] == True)
        ]
        n_expressed_tfs = (chain_rows['tf_expressed'] == True).sum() if len(chain_rows) > 0 else 0
        n_complete_chains = (chain_rows['chain_complete'] == True).sum() if len(chain_rows) > 0 else 0
        has_expressed_tf = n_expressed_tfs > 0
        has_complete_chain = n_complete_chains > 0

        # Gene expression level
        gene_expr = expr_lookup.get(gene_id, np.nan)

        # Gene info
        ginfo = genes.get(gene_id, {})
        dist_to_tss = row0.get('dist_to_tss', np.nan)

        # Count bQTL in this gene's promoter
        if gene_id in genes:
            g = genes[gene_id]
            tss = g['tss']
            if g['strand'] == '+':
                prom_start = max(0, tss - PROMOTER_BP)
                prom_end = tss
            else:
                prom_start = tss
                prom_end = tss + PROMOTER_BP
            n_bqtl_in_promoter = len(bqtl_df[
                (bqtl_df['chr'] == chrom) &
                (bqtl_df['pos'] >= prom_start) &
                (bqtl_df['pos'] <= prom_end)
            ])
        else:
            n_bqtl_in_promoter = 0

        # Count peaks in promoter (from gene_peak_disruption)
        gpd_row = gene_peaks[gene_peaks['gene_id'] == gene_id]
        total_peaks = gpd_row['total_peaks'].values[0] if len(gpd_row) > 0 else 0
        disrupted_peaks = gpd_row['disrupted_peaks'].values[0] if len(gpd_row) > 0 else 0

        pair_features.append({
            'chr': chrom, 'pos': pos, 'gene_id': gene_id,
            'functional': functional,
            'cohens_d': cohens_d, 'abs_cohens_d': abs(cohens_d),
            'mw_p': mw_p, 'fdr_mw': fdr,
            'abs_ase_diff': abs_ase_diff,
            'dist_to_tss': dist_to_tss,
            # Motif grammar features
            'n_families_at_variant': n_families_at_var,
            'n_motifs_at_variant': total_motifs_at_var,
            'families_at_variant': '|'.join(sorted(families_at_var)),
            'n_families_in_region': n_families_in_region,
            'total_motifs_in_region': total_motifs_in_region,
            'mean_redundancy': mean_redundancy,
            'max_redundancy': max_redundancy,
            'min_redundancy': min_redundancy,
            # TF expression
            'n_expressed_tfs': n_expressed_tfs,
            'n_complete_chains': n_complete_chains,
            'has_expressed_tf': has_expressed_tf,
            'has_complete_chain': has_complete_chain,
            # Gene/promoter context
            'gene_expression': gene_expr,
            'n_bqtl_in_promoter': n_bqtl_in_promoter,
            'total_peaks_at_gene': total_peaks,
            'disrupted_peaks_at_gene': disrupted_peaks,
        })

    pf = pd.DataFrame(pair_features)
    print(f"  Built feature table: {len(pf):,} pairs × {len(pf.columns)} features")

    func = pf[pf['functional']]
    nonfunc = pf[~pf['functional']]
    print(f"  Functional: {len(func):,}, Non-functional: {len(nonfunc):,}")

    # ================================================================
    # 4. COMPARE GRAMMAR: FUNCTIONAL vs NON-FUNCTIONAL
    # ================================================================
    print(f"\n{'='*60}")
    print("4. REGULATORY GRAMMAR: FUNCTIONAL vs NON-FUNCTIONAL")
    print(f"{'='*60}")

    features_to_compare = [
        ('n_families_at_variant', 'TF families disrupted at variant'),
        ('n_motifs_at_variant', 'Total motifs disrupted at variant'),
        ('n_families_in_region', 'TF families in region'),
        ('total_motifs_in_region', 'Total motifs in region'),
        ('mean_redundancy', 'Mean redundancy of disrupted motifs'),
        ('n_bqtl_in_promoter', 'bQTL in promoter'),
        ('total_peaks_at_gene', 'Peaks at gene'),
        ('disrupted_peaks_at_gene', 'Disrupted peaks at gene'),
        ('has_expressed_tf', 'Has expressed TF'),
        ('has_complete_chain', 'Has complete regulatory chain'),
        ('gene_expression', 'Gene expression (CPM)'),
        ('dist_to_tss', 'Distance to TSS'),
    ]

    print(f"\n  {'Feature':<40} {'Func':>8} {'NonF':>8} {'MW p':>10} {'Dir':>5}")
    print("  " + "-" * 75)

    for feat, label in features_to_compare:
        f_vals = func[feat].dropna()
        nf_vals = nonfunc[feat].dropna()
        if len(f_vals) < 10 or len(nf_vals) < 10:
            continue

        # For boolean features, compare proportions
        if feat.startswith('has_'):
            f_prop = f_vals.mean()
            nf_prop = nf_vals.mean()
            # Fisher's exact test
            a = int(f_vals.sum())
            b = int(len(f_vals) - a)
            c = int(nf_vals.sum())
            d = int(len(nf_vals) - c)
            _, p = stats.fisher_exact([[a, b], [c, d]])
            direction = '+' if f_prop > nf_prop else '-'
            print(f"  {label:<40} {100*f_prop:7.1f}% {100*nf_prop:7.1f}% {p:10.4e} {direction:>5}")
        else:
            f_mean = f_vals.mean()
            nf_mean = nf_vals.mean()
            _, p = stats.mannwhitneyu(f_vals, nf_vals, alternative='two-sided')
            direction = '+' if f_mean > nf_mean else '-'
            print(f"  {label:<40} {f_mean:8.2f} {nf_mean:8.2f} {p:10.4e} {direction:>5}")

    # ================================================================
    # 5. WHICH TF FAMILIES ARE DISRUPTED AT FUNCTIONAL bQTL?
    # ================================================================
    print(f"\n{'='*60}")
    print("5. TF FAMILIES DISRUPTED AT FUNCTIONAL vs NON-FUNCTIONAL")
    print(f"{'='*60}")

    # For each TF family, count how often it's disrupted in functional vs non-functional
    all_families = set()
    for fams in pf['families_at_variant']:
        if fams:
            all_families.update(fams.split('|'))
    all_families.discard('')

    tf_enrichment = []
    for fam in sorted(all_families):
        func_has = func['families_at_variant'].str.contains(fam, na=False).sum()
        func_total = len(func)
        nonfunc_has = nonfunc['families_at_variant'].str.contains(fam, na=False).sum()
        nonfunc_total = len(nonfunc)

        # Fisher's exact
        table = [[func_has, func_total - func_has],
                 [nonfunc_has, nonfunc_total - nonfunc_has]]
        odds_ratio, p = stats.fisher_exact(table)

        tf_enrichment.append({
            'tf_family': fam,
            'func_count': func_has, 'func_pct': 100 * func_has / func_total,
            'nonfunc_count': nonfunc_has, 'nonfunc_pct': 100 * nonfunc_has / nonfunc_total,
            'odds_ratio': odds_ratio, 'fisher_p': p,
        })

    tf_enr = pd.DataFrame(tf_enrichment).sort_values('odds_ratio', ascending=False)
    from statsmodels.stats.multitest import multipletests
    _, tf_enr['fdr'], _, _ = multipletests(tf_enr['fisher_p'], method='fdr_bh')

    print(f"\n  {'Family':<15} {'Func%':>6} {'NonF%':>6} {'OR':>6} {'p':>10} {'FDR':>8}")
    print("  " + "-" * 55)
    for _, row in tf_enr.iterrows():
        sig = '*' if row['fdr'] < 0.05 else ''
        print(f"  {row['tf_family']:<15} {row['func_pct']:5.1f}% {row['nonfunc_pct']:5.1f}% "
              f"{row['odds_ratio']:6.2f} {row['fisher_p']:10.4e} {row['fdr']:8.4f} {sig}")

    # ================================================================
    # 6. MOTIF COMBINATIONS (CO-OCCURRENCE GRAMMAR)
    # ================================================================
    print(f"\n{'='*60}")
    print("6. MOTIF COMBINATIONS AT FUNCTIONAL bQTL")
    print(f"{'='*60}")

    # Most common TF family combinations at functional bQTL variant positions
    func_combos = Counter()
    nonfunc_combos = Counter()
    for _, row in pf.iterrows():
        fams = row['families_at_variant']
        if not fams:
            continue
        key = tuple(sorted(fams.split('|')))
        if row['functional']:
            func_combos[key] += 1
        else:
            nonfunc_combos[key] += 1

    print(f"\n  Top 20 motif combinations at functional bQTL variants:")
    print(f"  {'Combination':<55} {'Func':>5} {'NonF':>5} {'Ratio':>6}")
    print("  " + "-" * 75)
    for combo, count in func_combos.most_common(20):
        nf_count = nonfunc_combos.get(combo, 0)
        ratio = count / max(nf_count, 1) * (len(nonfunc) / len(func))
        combo_str = ' + '.join(combo)
        if len(combo_str) > 53:
            combo_str = combo_str[:50] + '...'
        print(f"  {combo_str:<55} {count:5d} {nf_count:5d} {ratio:6.2f}")

    # ================================================================
    # 7. EXPRESSED TF → DISRUPTED MOTIF → ASE EFFECT
    # ================================================================
    print(f"\n{'='*60}")
    print("7. COMPLETE CHAINS: EXPRESSED TF → DISRUPTED MOTIF → ASE")
    print(f"{'='*60}")

    # Among functional bQTL, how many have a complete chain?
    func_chain = func[func['has_complete_chain']]
    func_no_chain = func[~func['has_complete_chain']]

    print(f"  Functional with complete chain: {len(func_chain):,} ({100*len(func_chain)/len(func):.1f}%)")
    print(f"  Functional without: {len(func_no_chain):,}")

    if len(func_chain) > 5 and len(func_no_chain) > 5:
        d_chain = func_chain['abs_cohens_d']
        d_no = func_no_chain['abs_cohens_d']
        _, p = stats.mannwhitneyu(d_chain, d_no, alternative='two-sided')
        print(f"  |d| with chain: {d_chain.mean():.3f} vs without: {d_no.mean():.3f} (p={p:.4e})")

    # List top functional hits with complete chains
    if len(func_chain) > 0:
        top_chains = func_chain.nlargest(15, 'abs_cohens_d')
        print(f"\n  Top functional bQTL with complete regulatory chains:")
        print(f"  {'Position':<25} {'Gene':<20} {'|d|':>6} {'TFs disrupted':<30} {'Expr TFs':>4}")
        print("  " + "-" * 90)
        for _, row in top_chains.iterrows():
            fams = row['families_at_variant'][:28] if row['families_at_variant'] else 'none'
            print(f"  {row['chr']}:{row['pos']:<15,} {row['gene_id']:<20} "
                  f"{row['abs_cohens_d']:6.2f} {fams:<30} {row['n_expressed_tfs']:4d}")

    # ================================================================
    # 8. PROMOTER ARCHITECTURE: bQTL DENSITY vs ASE
    # ================================================================
    print(f"\n{'='*60}")
    print("8. PROMOTER ARCHITECTURE: bQTL DENSITY vs EFFECT SIZE")
    print(f"{'='*60}")

    # Group by gene — aggregate bQTL effects
    gene_level = pf.groupby('gene_id').agg(
        n_bqtl_tested=('pos', 'nunique'),
        n_functional=('functional', 'sum'),
        max_abs_d=('abs_cohens_d', 'max'),
        mean_abs_d=('abs_cohens_d', 'mean'),
        n_bqtl_in_promoter=('n_bqtl_in_promoter', 'first'),
        total_peaks=('total_peaks_at_gene', 'first'),
        disrupted_peaks=('disrupted_peaks_at_gene', 'first'),
        mean_n_families=('n_families_at_variant', 'mean'),
        gene_expression=('gene_expression', 'first'),
    ).reset_index()

    # Correlation: bQTL density vs ASE effect
    rho, p = stats.spearmanr(gene_level['n_bqtl_in_promoter'],
                              gene_level['max_abs_d'])
    print(f"  bQTL per promoter vs max |d|: rho={rho:.4f}, p={p:.4e}")

    rho2, p2 = stats.spearmanr(gene_level['disrupted_peaks'],
                                gene_level['max_abs_d'])
    print(f"  Disrupted peaks vs max |d|: rho={rho2:.4f}, p={p2:.4e}")

    rho3, p3 = stats.spearmanr(gene_level['total_peaks'],
                                gene_level['max_abs_d'])
    print(f"  Total peaks vs max |d|: rho={rho3:.4f}, p={p3:.4e}")

    # Genes with multiple functional bQTL
    multi_func = gene_level[gene_level['n_functional'] > 1]
    print(f"\n  Genes with >1 functional bQTL: {len(multi_func):,}")
    if len(multi_func) > 0:
        print(f"  Their mean max |d|: {multi_func['max_abs_d'].mean():.3f}")
        single_func = gene_level[gene_level['n_functional'] == 1]
        if len(single_func) > 5:
            print(f"  Single functional bQTL mean max |d|: {single_func['max_abs_d'].mean():.3f}")

    # Binned analysis: bQTL count vs functional rate
    print(f"\n  bQTL per promoter vs functional rate:")
    print(f"  {'bQTL count':<15} {'N genes':>8} {'Func rate':>10} {'Mean |d|':>10}")
    print("  " + "-" * 45)
    for lo, hi, label in [(0, 1, '0-1'), (2, 3, '2-3'), (4, 6, '4-6'),
                           (7, 15, '7-15'), (16, 1000, '16+')]:
        mask = (gene_level['n_bqtl_in_promoter'] >= lo) & (gene_level['n_bqtl_in_promoter'] <= hi)
        subset = gene_level[mask]
        if len(subset) < 5:
            continue
        func_rate = (subset['n_functional'] > 0).mean()
        mean_d = subset['max_abs_d'].mean()
        print(f"  {label:<15} {len(subset):8d} {100*func_rate:9.1f}% {mean_d:10.3f}")

    # ── Save ──
    pf.to_csv(OUTDIR / 'functional_bqtl_grammar.csv', index=False)
    gene_level.to_csv(OUTDIR / 'gene_level_grammar.csv', index=False)
    tf_enr.to_csv(OUTDIR / 'tf_disruption_enrichment_functional.csv', index=False)
    print(f"\n  Saved: functional_bqtl_grammar.csv ({len(pf):,} pairs)")
    print(f"  Saved: gene_level_grammar.csv ({len(gene_level):,} genes)")
    print(f"  Saved: tf_disruption_enrichment_functional.csv")

    # ── Figures ──
    print("\n9. Generating figures...")
    make_figures(pf, gene_level, tf_enr)


def make_figures(pf, gene_level, tf_enr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    func = pf[pf['functional']]
    nonfunc = pf[~pf['functional']]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # A: Number of TF families disrupted at variant
    ax = axes[0, 0]
    bins = range(0, 15)
    ax.hist(func['n_families_at_variant'], bins=bins, alpha=0.6, color='#D32F2F',
            density=True, edgecolor='white', label=f'Functional (n={len(func):,})')
    ax.hist(nonfunc['n_families_at_variant'], bins=bins, alpha=0.6, color='#1976D2',
            density=True, edgecolor='white', label=f'Non-functional (n={len(nonfunc):,})')
    ax.set_xlabel('TF families with motif at variant')
    ax.set_ylabel('Density')
    ax.set_title('A. TF family diversity at variant')
    ax.legend(fontsize=8)

    # B: TF family enrichment in functional bQTL (forest plot)
    ax = axes[0, 1]
    plot_tf = tf_enr[tf_enr['func_count'] >= 5].sort_values('odds_ratio')
    colors = ['#D32F2F' if row['fdr'] < 0.05 else '#757575'
              for _, row in plot_tf.iterrows()]
    ax.barh(range(len(plot_tf)), np.log2(plot_tf['odds_ratio'].clip(0.1, 10)),
            color=colors, edgecolor='white', height=0.7)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(range(len(plot_tf)))
    ax.set_yticklabels(plot_tf['tf_family'], fontsize=7)
    ax.set_xlabel('log2(Odds Ratio) functional vs non-functional')
    ax.set_title('B. TF enrichment at functional bQTL')
    n_sig = (tf_enr['fdr'] < 0.05).sum()
    ax.text(0.95, 0.05, f'{n_sig} FDR<0.05', transform=ax.transAxes,
            fontsize=8, ha='right', color='#D32F2F')

    # C: Total motifs in region: functional vs non-functional
    ax = axes[0, 2]
    ax.hist(func['total_motifs_in_region'].clip(0, 400), bins=40, alpha=0.6,
            color='#D32F2F', density=True, edgecolor='white', label='Functional')
    ax.hist(nonfunc['total_motifs_in_region'].clip(0, 400), bins=40, alpha=0.6,
            color='#1976D2', density=True, edgecolor='white', label='Non-functional')
    ax.set_xlabel('Total motifs in 500bp region')
    ax.set_ylabel('Density')
    ax.set_title('C. Motif density in region')
    ax.legend(fontsize=8)

    # D: Complete regulatory chain enrichment
    ax = axes[1, 0]
    categories = ['Has motif\nat variant', 'Has expressed\nTF', 'Complete\nchain']
    func_rates = [
        func['n_families_at_variant'].gt(0).mean(),
        func['has_expressed_tf'].mean(),
        func['has_complete_chain'].mean(),
    ]
    nonfunc_rates = [
        nonfunc['n_families_at_variant'].gt(0).mean(),
        nonfunc['has_expressed_tf'].mean(),
        nonfunc['has_complete_chain'].mean(),
    ]
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, [100*r for r in func_rates], w, color='#D32F2F', label='Functional', edgecolor='white')
    ax.bar(x + w/2, [100*r for r in nonfunc_rates], w, color='#1976D2', label='Non-functional', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('% of bQTL-gene pairs')
    ax.set_title('D. Regulatory chain completeness')
    ax.legend(fontsize=8)

    # E: bQTL per promoter vs ASE effect
    ax = axes[1, 1]
    ax.scatter(gene_level['n_bqtl_in_promoter'].clip(0, 40),
               gene_level['max_abs_d'].clip(0, 5),
               s=8, alpha=0.3, color='#7B1FA2', rasterized=True)
    # Binned means
    for lo, hi in [(0, 1), (2, 3), (4, 6), (7, 10), (11, 20), (21, 100)]:
        mask = (gene_level['n_bqtl_in_promoter'] >= lo) & (gene_level['n_bqtl_in_promoter'] <= hi)
        if mask.sum() > 10:
            mid = (lo + hi) / 2
            mean_d = gene_level.loc[mask, 'max_abs_d'].mean()
            se = gene_level.loc[mask, 'max_abs_d'].std() / np.sqrt(mask.sum())
            ax.errorbar(mid, mean_d, yerr=se, fmt='rs', markersize=7, capsize=3)
    ax.set_xlabel('bQTL per promoter')
    ax.set_ylabel("Max |Cohen's d| per gene")
    rho, p = stats.spearmanr(gene_level['n_bqtl_in_promoter'], gene_level['max_abs_d'])
    ax.set_title(f"E. Promoter bQTL density vs effect\nrho={rho:.3f}, p={p:.2e}")

    # F: Motif combination size vs effect
    ax = axes[1, 2]
    pf_with = pf[pf['n_families_at_variant'] > 0].copy()
    bins_n = range(1, min(12, pf_with['n_families_at_variant'].max() + 1))
    means = []
    sems = []
    ns = []
    for n in bins_n:
        subset = pf_with[pf_with['n_families_at_variant'] == n]
        if len(subset) > 10:
            means.append(subset['abs_cohens_d'].mean())
            sems.append(subset['abs_cohens_d'].std() / np.sqrt(len(subset)))
            ns.append(len(subset))
        else:
            means.append(np.nan)
            sems.append(np.nan)
            ns.append(0)
    ax.errorbar(list(bins_n), means, yerr=sems, fmt='o-', color='#1976D2',
                markersize=6, capsize=3)
    ax.set_xlabel('Number of TF families disrupted at variant')
    ax.set_ylabel("Mean |Cohen's d|")
    ax.set_title('F. Motif complexity vs ASE effect')
    # Add sample sizes
    for i, (n, y) in enumerate(zip(bins_n, means)):
        if ns[i] > 0 and not np.isnan(y):
            ax.annotate(f'n={ns[i]}', (n, y), textcoords='offset points',
                       xytext=(0, 10), fontsize=6, ha='center')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig13_regulatory_grammar.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(FIG_DIR / 'fig13_regulatory_grammar.png', bbox_inches='tight', dpi=150)
    print(f"  Saved: figures/fig13_regulatory_grammar.pdf")
    plt.close()


if __name__ == '__main__':
    main()
