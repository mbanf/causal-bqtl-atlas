#!/usr/bin/env python3
"""
54_genotype_aware_bqtl_test.py — Causal test: variant genotype → ASE magnitude.

Part of: "Condition-dependent bQTL switching" paper (Banf & Hartwig)
Paper section: "Variant genotype causally determines allele-specific expression"
Pipeline step: 2 of 8

Uses NAM founder genotypes (from assembly alignment via script 53b) to determine
which F1 hybrids carry each bQTL variant, then tests whether carrying the
variant allele is associated with allele-specific expression (ASE) at nearby genes.

Logic:
  Each F1 hybrid = B73 × NAM_founder.
  If NAM_founder has ALT allele at a bQTL position → F1 is heterozygous → bQTL active.
  If NAM_founder has REF allele → F1 is homozygous REF → bQTL inactive.

  For each bQTL-gene pair (bQTL within 2kb of TSS):
    Split hybrids into VARIANT (NAM founder has ALT) vs REFERENCE (NAM founder has REF).
    Compare signed ASE (log2 B73/NAM) between groups using Mann-Whitney U.
    If bQTL is functional, variant-carrying hybrids should show more extreme ASE.

Key results (paper numbers):
  7,817 testable pairs → 785 FDR<0.05 (script 54 subset)
  5.18x enrichment, pi1=0.364, median |d|=2.13

Input:
  data/processed/nam_founder_genotypes_at_bqtl.tsv (from script 53)
  data/processed/engelhorn_ase_ww.csv
  data/processed/bqtl_snp_ww.csv
  Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3

Output:
  genotype_bqtl_results.csv — per bQTL-gene pair test results
  figures/fig11_genotype_bqtl_test.pdf — Paper Fig. 1
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

BQTL = DATA / "processed" / "bqtl_snp_ww.csv"
ASE_FILE = DATA / "processed" / "engelhorn_ase_ww.csv"
GENOTYPE_FILE = DATA / "processed" / "nam_founder_genotypes_at_bqtl.tsv"
GFF3 = DATA / "raw" / "Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3"

VALID_CHROMS = {f'chr{i}' for i in range(1, 11)}
PROMOTER_BP = 2000

# Map assembly founder column names → hybrid short names (as used in ASE data)
FOUNDER_TO_HYBRID = {
    'B97': 'B97', 'CML247': 'CML247', 'CML277': 'CML277',
    'CML322': 'CML322', 'CML333': 'CML333', 'CML69': 'CML69',
    'HP301': 'HP301', 'Il14H': 'IL14H', 'Ki11': 'Ki11', 'Ki3': 'Ki3',
    'Ky21': 'Ky21', 'M162W': 'M162W', 'Mo18W': 'Mo18W', 'Ms71': 'Ms71',
    'NC358': 'NC358', 'Oh43': 'Oh43', 'Oh7B': 'Oh7b', 'P39': 'P39',
    'Tx303': 'Tx303',
}

# ASE hybrids use "B73x<NAM>" format
HYBRIDS = ['A188', 'A619', 'B97', 'CML103', 'CML247', 'CML277', 'CML322',
           'CML333', 'CML69', 'HP301', 'IL14H', 'Ki11', 'Ki3', 'Ky21',
           'M162W', 'Mo17', 'Mo18W', 'Ms71', 'NC358', 'Oh43', 'Oh7b',
           'P39', 'Tx303', 'W22']


def parse_gff3():
    """Parse GFF3 to get gene TSS positions."""
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


def load_genotype_matrix(genotype_file):
    """Load genotype TSV (from assembly alignment) and convert to per-hybrid binary matrix.

    Format: chr, pos, ref, <founder columns>
    Values: '0|0' = ref, 'A/C/G/T' = SNP, 'DEL' = deletion, './.' = missing

    Returns:
        dict: {(chr, pos): {hybrid_name: 0 or 1}}
        where 1 = NAM founder carries non-reference allele (bQTL active in F1)
    """
    df = pd.read_csv(genotype_file, sep='\t', dtype=str)
    founder_cols = [c for c in df.columns if c not in ('chr', 'pos', 'ref')]
    print(f"  Loaded genotypes: {len(df):,} variants × {len(founder_cols)} founders")

    # Vectorized binary conversion per founder
    binary = pd.DataFrame(index=df.index)
    binary['chr'] = df['chr']
    binary['pos'] = df['pos'].astype(int)
    hybrid_cols = []

    for col in founder_cols:
        hybrid = FOUNDER_TO_HYBRID.get(col)
        if hybrid is None:
            continue
        vals = df[col]
        is_ref = vals == '0|0'
        is_miss = vals.isin(['./.', 'nan']) | vals.isna()
        # Anything not ref and not missing = variant (SNP or DEL)
        binary[hybrid] = np.where(is_miss, np.nan, np.where(is_ref, 0, 1))
        hybrid_cols.append(hybrid)

    # Build dict from vectorized result
    genotype_matrix = {}
    n_informative = 0
    arr = binary[hybrid_cols].values  # shape: (N, n_hybrids)

    for i in range(len(binary)):
        chrom = binary.iloc[i]['chr']
        pos = int(binary.iloc[i]['pos'])
        geno = {}
        for j, hybrid in enumerate(hybrid_cols):
            val = arr[i, j]
            if not np.isnan(val):
                geno[hybrid] = int(val)
        if geno:
            genotype_matrix[(chrom, pos)] = geno
            n_ref = sum(1 for v in geno.values() if v == 0)
            n_alt = sum(1 for v in geno.values() if v == 1)
            if n_ref >= 3 and n_alt >= 3:
                n_informative += 1

    print(f"  Genotype matrix: {len(genotype_matrix):,} positions")
    print(f"  Informative (≥3 ref + ≥3 alt): {n_informative:,}")
    return genotype_matrix


def map_bqtl_to_genes(bqtl_df, genes):
    """Map bQTL to nearby genes (within promoter region)."""
    mappings = []
    for _, bqtl in bqtl_df.iterrows():
        chrom = bqtl['chr']
        pos = bqtl['pos']
        for gene_id, ginfo in genes.items():
            if ginfo['chr'] != chrom:
                continue
            tss = ginfo['tss']
            dist = abs(pos - tss)
            if dist <= PROMOTER_BP:
                mappings.append({
                    'chr': chrom, 'pos': pos, 'gene_id': gene_id,
                    'dist_to_tss': dist, 'strand': ginfo['strand'],
                    'bqtl_type': bqtl.get('type', ''),
                    'bqtl_fdr': bqtl.get('fdr', np.nan),
                })
    return pd.DataFrame(mappings)


def run_genotype_test(genotype_matrix, bqtl_gene_df, ase_lookup):
    """Run genotype-aware functional bQTL test.

    For each bQTL-gene pair:
      Split hybrids by genotype at the bQTL position.
      Compare signed ASE between variant vs reference groups.
    """
    results = []
    for _, row in bqtl_gene_df.iterrows():
        chrom, pos, gene_id = row['chr'], row['pos'], row['gene_id']
        geno = genotype_matrix.get((chrom, pos))
        if geno is None:
            continue

        ase_variant = []
        ase_reference = []

        for hybrid in HYBRIDS:
            g = geno.get(hybrid)
            if g is None:
                continue
            ase_val = ase_lookup.get((gene_id, f'B73x{hybrid}'))
            if ase_val is None or np.isnan(ase_val):
                continue

            if g == 1:
                ase_variant.append(ase_val)
            else:
                ase_reference.append(ase_val)

        n_var = len(ase_variant)
        n_ref = len(ase_reference)

        if n_var >= 3 and n_ref >= 3:
            # Mann-Whitney U test
            mw_stat, mw_p = stats.mannwhitneyu(ase_variant, ase_reference,
                                                alternative='two-sided')
            # Welch's t-test
            t_stat, t_p = stats.ttest_ind(ase_variant, ase_reference,
                                           equal_var=False)

            mean_v = np.mean(ase_variant)
            mean_r = np.mean(ase_reference)
            sd_v = np.std(ase_variant, ddof=1) if n_var > 1 else 0
            sd_r = np.std(ase_reference, ddof=1) if n_ref > 1 else 0

            # Cohen's d (pooled SD)
            pooled_sd = np.sqrt(((n_var - 1) * sd_v**2 + (n_ref - 1) * sd_r**2) /
                                (n_var + n_ref - 2))
            cohens_d = (mean_v - mean_r) / pooled_sd if pooled_sd > 0 else 0

            # |ASE| comparison: do variant-carriers have MORE extreme ASE?
            abs_ase_var = np.mean([abs(x) for x in ase_variant])
            abs_ase_ref = np.mean([abs(x) for x in ase_reference])

            results.append({
                'chr': chrom, 'pos': pos, 'gene_id': gene_id,
                'dist_to_tss': row.get('dist_to_tss', np.nan),
                'n_variant': n_var, 'n_reference': n_ref,
                'mean_ase_variant': mean_v, 'mean_ase_reference': mean_r,
                'sd_ase_variant': sd_v, 'sd_ase_reference': sd_r,
                'ase_diff': mean_v - mean_r,
                'abs_ase_variant': abs_ase_var, 'abs_ase_reference': abs_ase_ref,
                'abs_ase_diff': abs_ase_var - abs_ase_ref,
                'cohens_d': cohens_d,
                'mw_p': mw_p, 't_p': t_p,
            })

    rdf = pd.DataFrame(results)
    if len(rdf) > 0:
        _, rdf['fdr_mw'], _, _ = multipletests(rdf['mw_p'], method='fdr_bh')
        _, rdf['fdr_t'], _, _ = multipletests(rdf['t_p'], method='fdr_bh')
    return rdf


def make_figures(rdf, outdir):
    """Generate summary figures for the genotype-aware test."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # A: P-value histogram (MW)
    ax = axes[0, 0]
    ax.hist(rdf['mw_p'], bins=50, color='#1976D2', edgecolor='white', alpha=0.8)
    ax.axhline(len(rdf) / 50, color='red', linestyle='--', alpha=0.7,
               label='Uniform expectation')
    ax.set_xlabel('Mann-Whitney p-value')
    ax.set_ylabel('Count')
    n_sig = (rdf['fdr_mw'] < 0.05).sum()
    ax.set_title(f'A. P-value distribution\n{n_sig:,} FDR<0.05 of {len(rdf):,}')
    ax.legend(fontsize=8)

    # B: Effect size (Cohen's d) distribution
    ax = axes[0, 1]
    d_vals = rdf['cohens_d'].clip(-3, 3)
    ax.hist(d_vals, bins=50, color='#FF7043', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel("Cohen's d (variant − reference)")
    ax.set_ylabel('Count')
    ax.set_title(f"B. Effect size distribution\nmedian |d| = {rdf['cohens_d'].abs().median():.3f}")

    # C: Volcano plot (effect size vs -log10 p)
    ax = axes[0, 2]
    log_p = -np.log10(rdf['mw_p'].clip(1e-20))
    sig_mask = rdf['fdr_mw'] < 0.05
    ax.scatter(rdf.loc[~sig_mask, 'cohens_d'].clip(-3, 3),
               log_p[~sig_mask], s=3, alpha=0.3, color='gray', rasterized=True)
    if sig_mask.any():
        ax.scatter(rdf.loc[sig_mask, 'cohens_d'].clip(-3, 3),
                   log_p[sig_mask], s=8, alpha=0.7, color='#D32F2F', rasterized=True,
                   label=f'FDR<0.05 (n={sig_mask.sum():,})')
        ax.legend(fontsize=8)
    ax.set_xlabel("Cohen's d")
    ax.set_ylabel('-log10(p)')
    ax.set_title('C. Volcano plot')

    # D: |ASE| comparison (variant vs reference groups)
    ax = axes[1, 0]
    ax.scatter(rdf['abs_ase_reference'], rdf['abs_ase_variant'],
               s=5, alpha=0.2, color='#1976D2', rasterized=True)
    lim = max(rdf['abs_ase_reference'].quantile(0.99),
              rdf['abs_ase_variant'].quantile(0.99))
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
    ax.set_xlabel('Mean |ASE| in reference hybrids')
    ax.set_ylabel('Mean |ASE| in variant hybrids')
    frac_above = (rdf['abs_ase_variant'] > rdf['abs_ase_reference']).mean()
    ax.set_title(f'D. |ASE| magnitude by genotype\n{100*frac_above:.1f}% higher in variant')

    # E: Sample size distribution
    ax = axes[1, 1]
    ax.hist(rdf['n_variant'], bins=range(0, 25), alpha=0.6, color='#D32F2F',
            label='Variant', edgecolor='white')
    ax.hist(rdf['n_reference'], bins=range(0, 25), alpha=0.6, color='#1976D2',
            label='Reference', edgecolor='white')
    ax.set_xlabel('Number of hybrids per group')
    ax.set_ylabel('Count')
    ax.set_title(f'E. Group sizes\nmedian variant={rdf["n_variant"].median():.0f}, '
                 f'ref={rdf["n_reference"].median():.0f}')
    ax.legend(fontsize=8)

    # F: Distance to TSS vs effect size
    ax = axes[1, 2]
    ax.scatter(rdf['dist_to_tss'], rdf['cohens_d'].abs().clip(0, 3),
               s=5, alpha=0.2, color='#7B1FA2', rasterized=True)
    ax.set_xlabel('Distance to TSS (bp)')
    ax.set_ylabel("|Cohen's d|")
    # Binned means
    bins = [0, 200, 500, 1000, 2000]
    for i in range(len(bins) - 1):
        mask = (rdf['dist_to_tss'] >= bins[i]) & (rdf['dist_to_tss'] < bins[i+1])
        if mask.sum() > 10:
            mean_d = rdf.loc[mask, 'cohens_d'].abs().mean()
            mid = (bins[i] + bins[i+1]) / 2
            ax.plot(mid, mean_d, 'rs', markersize=8)
    ax.set_title('F. Effect size vs distance to TSS')

    plt.tight_layout()
    plt.savefig(outdir / 'fig11_genotype_bqtl_test.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(outdir / 'fig11_genotype_bqtl_test.png', bbox_inches='tight', dpi=150)
    print(f"  Saved: figures/fig11_genotype_bqtl_test.pdf")
    plt.close()


def main():
    print("=" * 70)
    print("GENOTYPE-AWARE FUNCTIONAL bQTL TEST")
    print("=" * 70)

    # ── Check genotype file ──
    if not GENOTYPE_FILE.exists():
        print(f"\nERROR: Genotype file not found: {GENOTYPE_FILE}")
        print("Run script 53_download_nam_genotypes.py first.")
        return

    # ── Load data ──
    print("\n1. Loading data...")
    bqtl_df = pd.read_csv(BQTL)
    print(f"  bQTL: {len(bqtl_df):,}")

    ase_df = pd.read_csv(ASE_FILE)
    print(f"  ASE measurements: {len(ase_df):,}")

    genotype_matrix = load_genotype_matrix(GENOTYPE_FILE)

    # ── Parse genes and map bQTL to promoters ──
    print("\n2. Mapping bQTL to gene promoters...")
    genes = parse_gff3()
    print(f"  Genes in GFF3: {len(genes):,}")

    # Only map bQTL that have genotype data
    bqtl_with_geno = bqtl_df[
        bqtl_df.apply(lambda r: (r['chr'], r['pos']) in genotype_matrix, axis=1)
    ].copy()
    print(f"  bQTL with genotype data: {len(bqtl_with_geno):,}")

    bqtl_gene_df = map_bqtl_to_genes(bqtl_with_geno, genes)
    print(f"  bQTL-gene pairs in promoters: {len(bqtl_gene_df):,}")

    if len(bqtl_gene_df) == 0:
        print("\nNo bQTL-gene pairs found. Check promoter mapping.")
        return

    # ── Build ASE lookup ──
    print("\n3. Building ASE lookup...")
    ase_lookup = {}
    for _, row in ase_df.iterrows():
        ase_lookup[(row['gene_id'], row['hybrid'])] = row['log2_ratio']
    print(f"  ASE lookup: {len(ase_lookup):,} gene-hybrid pairs")

    # ── Genotype summary ──
    print("\n4. Genotype summary...")

    # Per-hybrid ALT allele rates
    hybrid_alt_counts = {h: 0 for h in HYBRIDS if h != 'CML103'}
    hybrid_total = {h: 0 for h in HYBRIDS if h != 'CML103'}
    for (chrom, pos), geno in genotype_matrix.items():
        for hybrid, g in geno.items():
            if hybrid in hybrid_alt_counts:
                hybrid_total[hybrid] += 1
                if g == 1:
                    hybrid_alt_counts[hybrid] += 1

    print(f"  Per-hybrid ALT allele rates:")
    for h in sorted(hybrid_alt_counts.keys()):
        if hybrid_total[h] > 0:
            rate = hybrid_alt_counts[h] / hybrid_total[h]
            print(f"    {h:8s}: {hybrid_alt_counts[h]:6,}/{hybrid_total[h]:6,} ({100*rate:.1f}%)")

    # ── Run the test ──
    print(f"\n5. Running genotype-aware test...")
    rdf = run_genotype_test(genotype_matrix, bqtl_gene_df, ase_lookup)
    print(f"  Tested pairs: {len(rdf):,}")

    if len(rdf) == 0:
        print("\nNo testable pairs (need ≥3 hybrids in each genotype group with ASE data).")
        return

    # ── Results summary ──
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    n_sig_mw = (rdf['fdr_mw'] < 0.05).sum()
    n_sig_t = (rdf['fdr_t'] < 0.05).sum()
    print(f"  Significant (FDR<0.05):")
    print(f"    Mann-Whitney: {n_sig_mw:,} / {len(rdf):,} ({100*n_sig_mw/len(rdf):.1f}%)")
    print(f"    Welch t-test: {n_sig_t:,} / {len(rdf):,} ({100*n_sig_t/len(rdf):.1f}%)")

    n_nom_mw = (rdf['mw_p'] < 0.05).sum()
    n_nom_t = (rdf['t_p'] < 0.05).sum()
    print(f"  Nominally significant (p<0.05):")
    print(f"    Mann-Whitney: {n_nom_mw:,} / {len(rdf):,} ({100*n_nom_mw/len(rdf):.1f}%)")
    print(f"    Welch t-test: {n_nom_t:,} / {len(rdf):,} ({100*n_nom_t/len(rdf):.1f}%)")
    print(f"    Expected by chance: {0.05*len(rdf):.0f} (5.0%)")

    # Pi1 estimate (fraction of true positives)
    from scipy.stats import gaussian_kde
    pvals = rdf['mw_p'].values
    if len(pvals) > 100:
        # Storey's pi0 estimate: fraction of p-values > 0.5, doubled
        pi0 = min(1.0, 2 * (pvals > 0.5).mean())
        pi1 = 1 - pi0
        print(f"  pi1 estimate (Storey): {pi1:.3f} (fraction true positives)")

    # Effect sizes
    print(f"\n  Effect sizes (Cohen's d):")
    print(f"    Mean:   {rdf['cohens_d'].mean():.4f}")
    print(f"    Median: {rdf['cohens_d'].median():.4f}")
    print(f"    Mean |d|: {rdf['cohens_d'].abs().mean():.4f}")
    print(f"    |d| > 0.2: {(rdf['cohens_d'].abs() > 0.2).sum():,} ({100*(rdf['cohens_d'].abs()>0.2).mean():.1f}%)")
    print(f"    |d| > 0.5: {(rdf['cohens_d'].abs() > 0.5).sum():,} ({100*(rdf['cohens_d'].abs()>0.5).mean():.1f}%)")
    print(f"    |d| > 0.8: {(rdf['cohens_d'].abs() > 0.8).sum():,} ({100*(rdf['cohens_d'].abs()>0.8).mean():.1f}%)")

    # |ASE| magnitude test
    frac_higher = (rdf['abs_ase_variant'] > rdf['abs_ase_reference']).mean()
    print(f"\n  |ASE| magnitude:")
    print(f"    Variant > Reference: {100*frac_higher:.1f}%")
    # Binomial test: is frac_higher significantly > 50%?
    n_higher = (rdf['abs_ase_variant'] > rdf['abs_ase_reference']).sum()
    binom_p = stats.binomtest(n_higher, len(rdf), 0.5, alternative='greater').pvalue
    print(f"    Binomial test p: {binom_p:.4e}")

    # Paired Wilcoxon on |ASE| difference
    wx_stat, wx_p = stats.wilcoxon(rdf['abs_ase_variant'] - rdf['abs_ase_reference'],
                                    alternative='greater')
    print(f"    Wilcoxon signed-rank p: {wx_p:.4e}")
    print(f"    Mean |ASE| diff: {rdf['abs_ase_diff'].mean():.4f}")

    # Top hits
    if n_sig_mw > 0:
        print(f"\n  Top significant hits (FDR<0.05, by effect size):")
        sig = rdf[rdf['fdr_mw'] < 0.05].sort_values('cohens_d', key=abs, ascending=False)
        for _, hit in sig.head(20).iterrows():
            print(f"    {hit['chr']}:{hit['pos']:,} → {hit['gene_id']} "
                  f"d={hit['cohens_d']:.3f} p={hit['mw_p']:.2e} "
                  f"n_var={hit['n_variant']} n_ref={hit['n_reference']}")
    elif n_nom_mw > 20:
        print(f"\n  Top nominally significant hits (p<0.05, by effect size):")
        nom = rdf[rdf['mw_p'] < 0.05].sort_values('cohens_d', key=abs, ascending=False)
        for _, hit in nom.head(20).iterrows():
            print(f"    {hit['chr']}:{hit['pos']:,} → {hit['gene_id']} "
                  f"d={hit['cohens_d']:.3f} p={hit['mw_p']:.2e} fdr={hit['fdr_mw']:.3f} "
                  f"n_var={hit['n_variant']} n_ref={hit['n_reference']}")

    # ── Save results ──
    rdf.to_csv(OUTDIR / 'genotype_bqtl_results.csv', index=False)
    print(f"\n  Saved: genotype_bqtl_results.csv ({len(rdf):,} pairs)")

    # ── Figures ──
    print("\n6. Generating figures...")
    make_figures(rdf, FIG_DIR)

    # ── Summary interpretation ──
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    enrichment = n_nom_mw / (0.05 * len(rdf)) if len(rdf) > 0 else 0
    print(f"  Nominal enrichment over chance: {enrichment:.2f}x")
    if enrichment > 1.5:
        print("  → STRONG SIGNAL: bQTL genotype predicts ASE direction")
    elif enrichment > 1.1:
        print("  → MODERATE SIGNAL: bQTL genotype has weak but detectable ASE association")
    else:
        print("  → WEAK/NO SIGNAL: genotype-ASE association not above chance")
        print("  Possible explanations:")
        print("    - bQTL effects on ASE may be too subtle for n=24 hybrids")
        print("    - Many bQTL may affect binding without affecting expression")
        print("    - ASE measurement noise may mask small genotype effects")
        print("    - bQTL may act through mechanisms other than cis-regulation")


if __name__ == '__main__':
    main()
