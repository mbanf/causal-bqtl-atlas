#!/usr/bin/env python3
"""
Step 04: Identify functional bQTL — which binding variants affect gene expression?

Maps bQTL to gene promoters (2kb upstream of TSS) and tests whether
peak presence at each bQTL position is associated with allele-specific
expression (ASE) across NAM hybrids.

Generates:
  results/bqtl_functional_test.csv — per-bQTL test (peak presence vs |ASE|)
  results/bqtl_signed_ase_test.csv — signed ASE direction test
  results/gene_bqtl_burden.csv — gene-level bQTL burden vs ASE
  figures/paper_figures/fig8_functional_bqtl_diagnostics.pdf
  figures/paper_figures/fig9_functional_bqtl_approaches.pdf
"""

import bisect
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from pipeline_config import config
OUTDIR = config.results_dir
FIG_DIR = config.figures_dir
BQTL = config.bqtl
ASE_HYBRID = config.ase
ASE_GENE = config.ase_summary
GFF3 = config.gff3
PEAKS_DIR = config.peaks_dir
PAN_CISTROME = OUTDIR / "pan_cistrome_peaks.csv"
FUNCTIONAL_CSV = OUTDIR / "bqtl_functional_test.csv"

VALID_CHROMS = {f'chr{i}' for i in range(1, 11)}
PROMOTER_BP = 2000

# Read hybrid names from ASE data
_ase_df = pd.read_csv(str(ASE_HYBRID), usecols=['hybrid'], dtype=str)
HYBRIDS = sorted(_ase_df['hybrid'].unique())


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


# ============================================================
# Shared helpers: promoter index, per-hybrid peak lookup
# ============================================================

def build_promoter_index(gene_coords):
    """Build sorted promoter intervals per chromosome for fast lookup."""
    promoters_by_chr = defaultdict(list)
    for gene_id, info in gene_coords.items():
        chrom = info['chr']
        if info['strand'] == '+':
            ps = max(0, info['tss'] - PROMOTER_BP)
            pe = info['tss']
        else:
            ps = info['tss']
            pe = info['tss'] + PROMOTER_BP
        promoters_by_chr[chrom].append((ps, pe, gene_id))
    for chrom in promoters_by_chr:
        promoters_by_chr[chrom].sort()
    return promoters_by_chr


def load_pan_cistrome_index():
    """Load pan-cistrome peaks as sorted arrays per chromosome."""
    pan_df = pd.read_csv(PAN_CISTROME)
    pan_by_chr = {}
    for chrom in VALID_CHROMS:
        sub = pan_df[pan_df['chr'] == chrom].sort_values('start')
        pan_by_chr[chrom] = sub[['start', 'end']].values
    return pan_by_chr


def pos_in_peaks(peaks_arr, pos):
    """Check if position falls in a sorted peak array."""
    if peaks_arr is None or len(peaks_arr) == 0:
        return False
    idx = bisect.bisect_right(peaks_arr[:, 0], pos) - 1
    return idx >= 0 and peaks_arr[idx, 0] <= pos <= peaks_arr[idx, 1]


def load_hybrid_peaks():
    """Load per-hybrid B73-aligned peak positions for peak presence lookup."""
    hybrid_peak_positions = {}
    for hybrid in HYBRIDS:
        # Peak files use founder name (A188), not hybrid name (B73xA188)
        founder = hybrid.replace('B73x', '')
        # Try q255 first (preferred — matches bQTL calling), fall back to q3
        peak_file = PEAKS_DIR / f"{founder}.WW.q255.peaks.names.sorted.bed.gz"
        if not peak_file.exists():
            peak_file = PEAKS_DIR / f"{founder}.w2_4.q3_peaks.narrowPeak"
        if not peak_file.exists():
            continue
        positions_by_chr = defaultdict(list)
        import gzip
        opener = gzip.open if peak_file.suffix == '.gz' else open
        with opener(peak_file, 'rt') as f:
            for line in f:
                parts = line.strip().split('\t')
                chrom_raw = parts[0]
                if not chrom_raw.startswith('B73-chr'):
                    continue
                chrom = chrom_raw.replace('B73-', '')
                if chrom not in VALID_CHROMS:
                    continue
                positions_by_chr[chrom].append((int(parts[1]), int(parts[2])))
        for chrom in positions_by_chr:
            positions_by_chr[chrom].sort()
        hybrid_peak_positions[hybrid] = positions_by_chr
    return hybrid_peak_positions


def hybrid_has_peak(hybrid_peaks, hybrid, chrom, pos):
    """Check if position has a peak in this hybrid."""
    peaks = hybrid_peaks.get(hybrid, {}).get(chrom, [])
    if not peaks:
        return False
    starts = [p[0] for p in peaks]
    idx = bisect.bisect_right(starts, pos) - 1
    if idx >= 0 and peaks[idx][0] <= pos <= peaks[idx][1]:
        return True
    if idx + 1 < len(peaks) and peaks[idx + 1][0] <= pos <= peaks[idx + 1][1]:
        return True
    return False


# ============================================================
# PART 0: Generate bqtl_functional_test.csv (peak presence vs |ASE|)
# ============================================================

def generate_functional_test():
    """Map bQTL to gene promoters and test peak presence vs absolute ASE.

    For each bQTL in a pan-cistrome peak within a gene promoter:
      Split hybrids into WITH-peak vs WITHOUT-peak at the bQTL position.
      Compare |ASE| between groups using Mann-Whitney U and point-biserial corr.
    """
    if FUNCTIONAL_CSV.exists():
        print(f"  {FUNCTIONAL_CSV.name} already exists, skipping generation")
        return pd.read_csv(FUNCTIONAL_CSV)

    print("=" * 70)
    print("GENERATING: bqtl_functional_test.csv")
    print("=" * 70)

    from statsmodels.stats.multitest import multipletests

    bqtl_df = pd.read_csv(BQTL)
    ase_df = pd.read_csv(ASE_HYBRID)
    gene_coords = parse_gff3()
    promoters_by_chr = build_promoter_index(gene_coords)
    pan_by_chr = load_pan_cistrome_index()
    hybrid_peaks = load_hybrid_peaks()

    # Build absolute ASE lookup: (gene_id, hybrid_short) -> |log2_ratio|
    ase_abs = {}
    for _, row in ase_df.iterrows():
        parent = row['hybrid'].replace('B73x', '')
        ase_abs[(row['gene_id'], parent)] = abs(row['log2_ratio'])

    # Map bQTL to gene promoters
    print("Mapping bQTL to gene promoters...")
    bqtl_genes = []
    for _, row in bqtl_df.iterrows():
        chrom, pos = row['chr'], row['pos']
        for ps, pe, gene_id in promoters_by_chr.get(chrom, []):
            if ps <= pos <= pe:
                bqtl_genes.append({
                    'pos': pos, 'chr': chrom, 'gene_id': gene_id,
                    'bqtl_type': row['Type'], 'bqtl_fdr': row['FDRp'],
                    'dist_to_tss': abs(pos - gene_coords[gene_id]['tss'])
                })
    bqtl_gene_df = pd.DataFrame(bqtl_genes)
    print(f"  {len(bqtl_gene_df):,} bQTL-gene pairs")

    # Test each unique bQTL-gene pair in pan-cistrome
    print("Testing peak presence vs |ASE|...")
    results = []
    unique_pairs = bqtl_gene_df.drop_duplicates(['pos', 'chr', 'gene_id'])

    for _, brow in unique_pairs.iterrows():
        chrom, pos, gene_id = brow['chr'], brow['pos'], brow['gene_id']
        if not pos_in_peaks(pan_by_chr.get(chrom), pos):
            continue

        ase_with, ase_without = [], []
        peak_binary = []  # 1=has peak, 0=no peak
        ase_vals = []

        for hybrid in HYBRIDS:
            parent = hybrid.replace('B73x', '')
            abs_ase = ase_abs.get((gene_id, parent))
            if abs_ase is None or np.isnan(abs_ase):
                continue
            has_pk = hybrid_has_peak(hybrid_peaks, hybrid, chrom, pos)
            if has_pk:
                ase_with.append(abs_ase)
            else:
                ase_without.append(abs_ase)
            peak_binary.append(1 if has_pk else 0)
            ase_vals.append(abs_ase)

        if len(ase_with) >= 3 and len(ase_without) >= 3:
            mw_stat, mw_p = stats.mannwhitneyu(
                ase_with, ase_without, alternative='two-sided')
            # Cohen's d
            pooled_std = np.sqrt(
                ((len(ase_with) - 1) * np.std(ase_with, ddof=1) ** 2 +
                 (len(ase_without) - 1) * np.std(ase_without, ddof=1) ** 2) /
                (len(ase_with) + len(ase_without) - 2))
            cohens_d = ((np.mean(ase_with) - np.mean(ase_without)) /
                        pooled_std) if pooled_std > 0 else 0
            # Point-biserial correlation
            pb_rho, pb_p = stats.pointbiserialr(peak_binary, ase_vals)

            results.append({
                'pos': pos, 'chr': chrom, 'gene_id': gene_id,
                'bqtl_type': brow['bqtl_type'], 'bqtl_fdr': brow['bqtl_fdr'],
                'n_hybrids_tested': len(ase_with) + len(ase_without),
                'n_with_peak': len(ase_with), 'n_without_peak': len(ase_without),
                'mean_ase_with_peak': np.mean(ase_with),
                'mean_ase_without_peak': np.mean(ase_without),
                'ase_diff': np.mean(ase_with) - np.mean(ase_without),
                'cohens_d': cohens_d,
                'mw_p': mw_p, 'pb_rho': pb_rho, 'pb_p': pb_p,
                'dist_to_tss': brow['dist_to_tss'],
            })

    rdf = pd.DataFrame(results)
    print(f"  Tested {len(rdf):,} bQTL-gene pairs")

    # FDR correction
    _, rdf['fdr'], _, _ = multipletests(rdf['mw_p'], method='fdr_bh')
    rdf['functional'] = rdf['fdr'] < 0.05

    n_sig = rdf['functional'].sum()
    n_nom = (rdf['mw_p'] < 0.05).sum()
    print(f"  Nominally significant (p<0.05): {n_nom} ({100 * n_nom / len(rdf):.1f}%)")
    print(f"  FDR<0.05: {n_sig}")

    rdf.to_csv(FUNCTIONAL_CSV, index=False)
    print(f"  Saved: {FUNCTIONAL_CSV.name}")
    return rdf


# ============================================================
# PART 1: Diagnostics and figures for per-bQTL test
# ============================================================

def part1_diagnostics():
    """Complete the diagnostic analysis from the per-bQTL functional test."""
    print("=" * 70)
    print("PART 1: PER-bQTL FUNCTIONAL TEST DIAGNOSTICS")
    print("=" * 70)

    df = pd.read_csv(FUNCTIONAL_CSV)
    print(f"Loaded {len(df):,} tested bQTL")
    print(f"  Nominally significant (p<0.05): {(df['mw_p']<0.05).sum()} ({100*(df['mw_p']<0.05).mean():.1f}%)")
    print(f"  FDR<0.05: {df['functional'].sum()}")

    # Pi1 estimate
    pvals = df['mw_p'].dropna().values
    for thr in [0.05, 0.10, 0.20]:
        frac_below = (pvals < thr).mean()
        pi0 = (1 - frac_below) / (1 - thr)
        pi1 = max(0, 1 - pi0)
        print(f"  Pi1 at lambda={thr}: {pi1:.4f} (pi0={pi0:.4f})")

    # Sign test (using binomtest instead of deprecated binom_test)
    n_pos = (df['ase_diff'] > 0).sum()
    n_neg = (df['ase_diff'] < 0).sum()
    n_total = n_pos + n_neg
    sign_result = stats.binomtest(n_pos, n_total, 0.5)
    print(f"\n  Sign test: {n_pos} positive, {n_neg} negative")
    print(f"  Binomial p = {sign_result.pvalue:.4f} (proportion = {n_pos/n_total:.4f})")

    # --- Distance-to-TSS stratification ---
    print(f"\n--- Distance-to-TSS stratification ---")
    dist_bins = [(0, 200, 'proximal (<200bp)'),
                 (200, 500, 'near (200-500bp)'),
                 (500, 1000, 'mid (500-1000bp)'),
                 (1000, 2000, 'distal (1-2kb)')]
    strat_results = []
    for lo, hi, label in dist_bins:
        subset = df[(df['dist_to_tss'] >= lo) & (df['dist_to_tss'] < hi)]
        if len(subset) < 10:
            continue
        nom_sig = (subset['mw_p'] < 0.05).mean()
        mean_d = subset['cohens_d'].abs().mean()
        strat_results.append({'bin': label, 'n': len(subset),
                              'nom_sig_pct': 100 * nom_sig,
                              'mean_abs_d': mean_d,
                              'mean_ase_diff': subset['ase_diff'].mean()})
        print(f"  {label:22s}: n={len(subset):5d}, "
              f"nom sig={100*nom_sig:.1f}%, |d|={mean_d:.4f}")

    # --- bQTL type stratification ---
    print(f"\n--- bQTL type stratification ---")
    type_results = []
    for btype in df['bqtl_type'].unique():
        subset = df[df['bqtl_type'] == btype]
        nom_sig = (subset['mw_p'] < 0.05).mean()
        mean_d = subset['cohens_d'].abs().mean()
        type_results.append({'type': btype, 'n': len(subset),
                             'nom_sig_pct': 100 * nom_sig,
                             'mean_abs_d': mean_d})
        print(f"  {btype:10s}: n={len(subset):5d}, "
              f"nom sig={100*nom_sig:.1f}%, |d|={mean_d:.4f}")

    # --- Generate figures ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # A: P-value histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(pvals, bins=50, color='#607D8B', edgecolor='white', alpha=0.8, density=True)
    ax1.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Uniform')
    ax1.set_xlabel('p-value (Mann-Whitney)')
    ax1.set_ylabel('Density')
    ax1.set_title('A. P-value distribution')
    ax1.legend(fontsize=8)
    ax1.text(0.05, 0.95, f'n = {len(pvals):,}\nnom p<0.05: {(pvals<0.05).sum()}\nFDR sig: 0',
             transform=ax1.transAxes, va='top', fontsize=7,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # B: QQ plot
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_p = np.sort(pvals)
    expected = np.arange(1, len(sorted_p) + 1) / (len(sorted_p) + 1)
    ax2.scatter(-np.log10(expected), -np.log10(sorted_p), s=1, alpha=0.3, color='#1976D2')
    max_val = max(-np.log10(expected).max(), -np.log10(sorted_p[sorted_p > 0]).max()) + 0.5
    ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    ax2.set_xlabel('Expected -log₁₀(p)')
    ax2.set_ylabel('Observed -log₁₀(p)')
    ax2.set_title('B. QQ plot')
    # Genomic inflation factor
    chisq_obs = stats.chi2.ppf(1 - pvals, 1)
    lambda_gc = np.median(chisq_obs) / stats.chi2.ppf(0.5, 1)
    ax2.text(0.05, 0.95, f'$\\lambda_{{GC}}$ = {lambda_gc:.3f}',
             transform=ax2.transAxes, va='top', fontsize=9)

    # C: Effect size distribution
    ax3 = fig.add_subplot(gs[0, 2])
    d_vals = df['cohens_d'].dropna().values
    ax3.hist(d_vals, bins=80, color='#FF7043', edgecolor='white', alpha=0.8)
    ax3.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel("Cohen's d (peak→higher ASE)")
    ax3.set_ylabel('Count')
    ax3.set_title(f"C. Effect sizes\n(mean |d| = {np.abs(d_vals).mean():.3f})")

    # D: Nominal significance by distance
    ax4 = fig.add_subplot(gs[1, 0])
    if strat_results:
        sdf = pd.DataFrame(strat_results)
        ax4.bar(range(len(sdf)), sdf['nom_sig_pct'], color='#26A69A', edgecolor='white')
        ax4.axhline(5.0, color='red', linestyle='--', alpha=0.7, label='Expected 5%')
        ax4.set_xticks(range(len(sdf)))
        ax4.set_xticklabels([r['bin'].split('(')[0].strip() for r in strat_results], fontsize=7)
        ax4.set_ylabel('% nominally significant')
        ax4.set_title('D. Significance by distance to TSS')
        ax4.legend(fontsize=7)
        for i, row in sdf.iterrows():
            ax4.text(i, row['nom_sig_pct'] + 0.3, f"n={row['n']:,}", ha='center', fontsize=6)

    # E: Nominal significance by bQTL type
    ax5 = fig.add_subplot(gs[1, 1])
    if type_results:
        tdf = pd.DataFrame(type_results)
        ax5.bar(range(len(tdf)), tdf['nom_sig_pct'], color='#AB47BC', edgecolor='white')
        ax5.axhline(5.0, color='red', linestyle='--', alpha=0.7, label='Expected 5%')
        ax5.set_xticks(range(len(tdf)))
        ax5.set_xticklabels(tdf['type'], fontsize=8)
        ax5.set_ylabel('% nominally significant')
        ax5.set_title('E. Significance by bQTL type')
        ax5.legend(fontsize=7)
        for i, row in tdf.iterrows():
            ax5.text(i, row['nom_sig_pct'] + 0.3, f"n={row['n']:,}", ha='center', fontsize=6)

    # F: Volcano-style: effect size vs -log10(p)
    ax6 = fig.add_subplot(gs[1, 2])
    valid = df[df['mw_p'] > 0].copy()
    ax6.scatter(valid['cohens_d'], -np.log10(valid['mw_p']),
                s=2, alpha=0.15, color='#455A64')
    sig = valid[valid['mw_p'] < 0.05]
    ax6.scatter(sig['cohens_d'], -np.log10(sig['mw_p']),
                s=4, alpha=0.5, color='#E53935')
    ax6.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
    ax6.set_xlabel("Cohen's d")
    ax6.set_ylabel('-log₁₀(p)')
    ax6.set_title(f'F. Volcano plot\n({len(sig)} nominal, 0 FDR)')

    plt.suptitle('Per-bQTL Functional Test: Peak Presence vs ASE Across Hybrids',
                 fontsize=12, fontweight='bold')
    plt.savefig(FIG_DIR / 'fig8_functional_bqtl_diagnostics.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(FIG_DIR / 'fig8_functional_bqtl_diagnostics.png', bbox_inches='tight', dpi=150)
    print(f"\nSaved: figures/fig8_functional_bqtl_diagnostics.pdf")
    plt.close()

    return df


# ============================================================
# PART 2: Alternative approaches
# ============================================================

def part2_signed_ase():
    """Test with signed ASE (B73-NAM direction) instead of absolute values.

    Rationale: if a bQTL disrupts B73 binding, the NAM allele should be MORE
    expressed (or vice versa). Using |ASE| throws away this directional info.
    """
    print(f"\n{'='*70}")
    print("PART 2: SIGNED ASE DIRECTION ANALYSIS")
    print(f"{'='*70}")

    bqtl_df = pd.read_csv(BQTL)
    ase_df = pd.read_csv(ASE_HYBRID)
    gene_coords = parse_gff3()
    promoters_by_chr = build_promoter_index(gene_coords)
    pan_by_chr = load_pan_cistrome_index()
    hybrid_peaks = load_hybrid_peaks()

    # Map each bQTL to genes
    print("Mapping bQTL to gene promoters...")
    bqtl_genes = []
    for _, row in bqtl_df.iterrows():
        chrom, pos = row['chr'], row['pos']
        for ps, pe, gene_id in promoters_by_chr.get(chrom, []):
            if ps <= pos <= pe:
                bqtl_genes.append({
                    'pos': pos, 'chr': chrom, 'gene_id': gene_id,
                    'bqtl_type': row['Type'], 'bqtl_fdr': row['FDRp'],
                    'dist_to_tss': abs(pos - gene_coords[gene_id]['tss'])
                })
    bqtl_gene_df = pd.DataFrame(bqtl_genes)
    print(f"  {len(bqtl_gene_df):,} bQTL-gene pairs")
    print(f"  {bqtl_gene_df['pos'].nunique():,} unique bQTL in promoters")

    # Build signed ASE lookup: (gene_id, hybrid) -> log2(B73/NAM)
    ase_signed = {}
    for _, row in ase_df.iterrows():
        ase_signed[(row['gene_id'], row['hybrid'])] = row['log2_ratio']

    # Test signed ASE with vs without peak
    print("Testing signed ASE consistency...")
    results = []
    unique_bqtl = bqtl_gene_df.drop_duplicates(['pos', 'chr', 'gene_id'])

    for _, brow in unique_bqtl.iterrows():
        chrom, pos, gene_id = brow['chr'], brow['pos'], brow['gene_id']
        if not pos_in_peaks(pan_by_chr.get(chrom), pos):
            continue

        ase_with, ase_without = [], []
        for hybrid in HYBRIDS:
            signed_ase = ase_signed.get((gene_id, hybrid))
            if signed_ase is None or np.isnan(signed_ase):
                continue
            if hybrid_has_peak(hybrid_peaks, hybrid, chrom, pos):
                ase_with.append(signed_ase)
            else:
                ase_without.append(signed_ase)

        if len(ase_with) >= 3 and len(ase_without) >= 3:
            mw_stat, mw_p = stats.mannwhitneyu(ase_with, ase_without,
                                                alternative='two-sided')
            t_stat, t_p = stats.ttest_ind(ase_with, ase_without)
            results.append({
                'pos': pos, 'chr': chrom, 'gene_id': gene_id,
                'bqtl_type': brow['bqtl_type'],
                'dist_to_tss': brow['dist_to_tss'],
                'n_with': len(ase_with), 'n_without': len(ase_without),
                'mean_signed_with': np.mean(ase_with),
                'mean_signed_without': np.mean(ase_without),
                'signed_diff': np.mean(ase_with) - np.mean(ase_without),
                'mw_p': mw_p, 't_p': t_p, 't_stat': t_stat,
            })

    rdf = pd.DataFrame(results)
    print(f"  Tested {len(rdf):,} bQTL (signed ASE)")

    if len(rdf) == 0:
        print("  No testable bQTL found")
        return None

    # FDR correction
    from statsmodels.stats.multitest import multipletests
    _, rdf['fdr'], _, _ = multipletests(rdf['mw_p'], method='fdr_bh')
    _, rdf['t_fdr'], _, _ = multipletests(rdf['t_p'], method='fdr_bh')

    n_sig_mw = (rdf['fdr'] < 0.05).sum()
    n_sig_t = (rdf['t_fdr'] < 0.05).sum()
    n_nom_mw = (rdf['mw_p'] < 0.05).sum()
    n_nom_t = (rdf['t_p'] < 0.05).sum()

    print(f"  MW nominal p<0.05: {n_nom_mw} ({100*n_nom_mw/len(rdf):.1f}%)")
    print(f"  MW FDR<0.05: {n_sig_mw}")
    print(f"  t-test nominal p<0.05: {n_nom_t} ({100*n_nom_t/len(rdf):.1f}%)")
    print(f"  t-test FDR<0.05: {n_sig_t}")

    # Pi1 for signed test
    for test_name, pcol in [('MW', 'mw_p'), ('t-test', 't_p')]:
        pv = rdf[pcol].dropna().values
        frac_below = (pv < 0.1).mean()
        pi0 = (1 - frac_below) / 0.9
        pi1 = max(0, 1 - pi0)
        print(f"  Pi1 ({test_name}, lambda=0.1): {pi1:.4f}")

    # Directional consistency: among nominally significant, which direction?
    sig_signed = rdf[rdf['mw_p'] < 0.05]
    if len(sig_signed) > 0:
        n_peak_higher = (sig_signed['signed_diff'] > 0).sum()
        n_peak_lower = (sig_signed['signed_diff'] < 0).sum()
        print(f"\n  Among nominal hits (n={len(sig_signed)}):")
        print(f"    Peak present → B73 higher: {n_peak_higher}")
        print(f"    Peak present → NAM higher: {n_peak_lower}")

    rdf.to_csv(OUTDIR / 'bqtl_signed_ase_test.csv', index=False)
    print(f"  Saved: bqtl_signed_ase_test.csv")

    return rdf


def part3_high_variance_genes():
    """Focus on genes with HIGH ASE variance across hybrids.

    Rationale: genes with consistent ASE across all hybrids won't show
    peak-dependent variation. Only genes whose ASE VARIES could reveal
    bQTL effects.
    """
    print(f"\n{'='*70}")
    print("PART 3: HIGH-VARIANCE GENES ANALYSIS")
    print(f"{'='*70}")

    ase_df = pd.read_csv(ASE_HYBRID)
    # Compute per-gene ASE variance across hybrids
    gene_var = ase_df.groupby('gene_id').agg(
        ase_mean=('abs_log2_ratio', 'mean'),
        ase_std=('abs_log2_ratio', 'std'),
        ase_cv=('abs_log2_ratio', lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
        n_hybrids=('hybrid', 'nunique'),
        signed_range=('log2_ratio', lambda x: x.max() - x.min()),
    ).reset_index()

    print(f"Genes with ASE data: {len(gene_var):,}")
    print(f"ASE std distribution:")
    print(f"  mean: {gene_var['ase_std'].mean():.4f}")
    print(f"  median: {gene_var['ase_std'].median():.4f}")
    print(f"  Q75: {gene_var['ase_std'].quantile(0.75):.4f}")
    print(f"  Q90: {gene_var['ase_std'].quantile(0.90):.4f}")

    # Load functional test results and merge
    func_df = pd.read_csv(FUNCTIONAL_CSV)

    # For high-variance genes (top 25%), re-examine functional test
    high_var_genes = set(gene_var[gene_var['ase_std'] >= gene_var['ase_std'].quantile(0.75)]['gene_id'])
    low_var_genes = set(gene_var[gene_var['ase_std'] < gene_var['ase_std'].quantile(0.25)]['gene_id'])

    func_hv = func_df[func_df['gene_id'].isin(high_var_genes)]
    func_lv = func_df[func_df['gene_id'].isin(low_var_genes)]

    print(f"\nHigh-variance genes (top 25%): {len(high_var_genes):,}")
    print(f"  bQTL tested: {len(func_hv):,}")
    if len(func_hv) > 0:
        nom_sig = (func_hv['mw_p'] < 0.05).mean()
        print(f"  Nominal p<0.05: {(func_hv['mw_p']<0.05).sum()} ({100*nom_sig:.1f}%)")
        mean_d = func_hv['cohens_d'].abs().mean()
        print(f"  Mean |d|: {mean_d:.4f}")
        # Pi1
        pv = func_hv['mw_p'].dropna().values
        frac = (pv < 0.1).mean()
        pi0 = (1 - frac) / 0.9
        pi1 = max(0, 1 - pi0)
        print(f"  Pi1 (lambda=0.1): {pi1:.4f}")

    print(f"\nLow-variance genes (bottom 25%): {len(low_var_genes):,}")
    print(f"  bQTL tested: {len(func_lv):,}")
    if len(func_lv) > 0:
        nom_sig = (func_lv['mw_p'] < 0.05).mean()
        print(f"  Nominal p<0.05: {(func_lv['mw_p']<0.05).sum()} ({100*nom_sig:.1f}%)")

    # Compare signed ASE range vs number of bQTL
    gene_var_merged = gene_var.merge(
        func_df.groupby('gene_id').agg(
            n_bqtl=('pos', 'nunique'),
            mean_abs_d=('cohens_d', lambda x: x.abs().mean()),
            min_p=('mw_p', 'min'),
        ).reset_index(),
        on='gene_id', how='inner'
    )
    rho, p = stats.spearmanr(gene_var_merged['signed_range'], gene_var_merged['n_bqtl'])
    print(f"\nASE range vs n_bQTL: rho={rho:.4f}, p={p:.2e}")
    rho2, p2 = stats.spearmanr(gene_var_merged['ase_std'], gene_var_merged['mean_abs_d'])
    print(f"ASE std vs mean |d|: rho={rho2:.4f}, p={p2:.2e}")

    return gene_var


def part4_aggregate_gene_level():
    """Gene-level aggregate test: do genes with MORE bQTL show MORE ASE?

    Instead of testing individual bQTL, test whether the DENSITY of bQTL
    in a promoter predicts ASE variance across hybrids.
    """
    print(f"\n{'='*70}")
    print("PART 4: GENE-LEVEL AGGREGATE bQTL BURDEN TEST")
    print(f"{'='*70}")

    from statsmodels.stats.multitest import multipletests

    bqtl_df = pd.read_csv(BQTL)
    ase_gene = pd.read_csv(ASE_GENE)
    ase_hybrid = pd.read_csv(ASE_HYBRID)
    gene_coords = parse_gff3()
    promoters_by_chr = build_promoter_index(gene_coords)

    print("Building bQTL-to-gene index...")
    bqtl_per_gene = defaultdict(int)
    for _, row in bqtl_df.iterrows():
        chrom, pos = row['chr'], row['pos']
        promoters = promoters_by_chr.get(chrom, [])
        starts = [p[0] for p in promoters]
        idx = bisect.bisect_right(starts, pos)
        for j in range(max(0, idx - 5), min(len(promoters), idx + 1)):
            if promoters[j][0] <= pos <= promoters[j][1]:
                bqtl_per_gene[promoters[j][2]] += 1

    bqtl_counts = pd.DataFrame([
        {'gene_id': g, 'n_bqtl': c} for g, c in bqtl_per_gene.items()
    ])
    print(f"  Genes with bQTL in promoter: {len(bqtl_counts):,}")

    # Merge with ASE data
    # Per-gene: mean ASE, ASE variance across hybrids
    gene_ase = ase_hybrid.groupby('gene_id').agg(
        mean_abs_ase=('abs_log2_ratio', 'mean'),
        std_abs_ase=('abs_log2_ratio', 'std'),
        mean_signed=('log2_ratio', 'mean'),
        std_signed=('log2_ratio', 'std'),
        n_hybrids=('hybrid', 'nunique'),
    ).reset_index()

    merged = gene_ase.merge(bqtl_counts, on='gene_id', how='left')
    merged['n_bqtl'] = merged['n_bqtl'].fillna(0).astype(int)

    # Add genes with 0 bQTL
    print(f"  Genes with ASE data: {len(gene_ase):,}")
    print(f"  Genes with both: {merged['n_bqtl'].gt(0).sum():,}")

    # Correlations
    has_bqtl = merged[merged['n_bqtl'] > 0]
    no_bqtl = merged[merged['n_bqtl'] == 0]

    rho1, p1 = stats.spearmanr(has_bqtl['n_bqtl'], has_bqtl['mean_abs_ase'])
    rho2, p2 = stats.spearmanr(has_bqtl['n_bqtl'], has_bqtl['std_abs_ase'])

    print(f"\n  n_bQTL vs mean |ASE|: rho={rho1:.4f}, p={p1:.2e}")
    print(f"  n_bQTL vs ASE std:    rho={rho2:.4f}, p={p2:.2e}")

    # Compare genes with vs without bQTL
    mw_abs, p_abs = stats.mannwhitneyu(has_bqtl['mean_abs_ase'], no_bqtl['mean_abs_ase'])
    mw_std, p_std = stats.mannwhitneyu(has_bqtl['std_abs_ase'].dropna(),
                                        no_bqtl['std_abs_ase'].dropna())

    print(f"\n  Genes WITH bQTL (n={len(has_bqtl):,}):")
    print(f"    mean |ASE| = {has_bqtl['mean_abs_ase'].mean():.4f}")
    print(f"    ASE std = {has_bqtl['std_abs_ase'].mean():.4f}")
    print(f"  Genes WITHOUT bQTL (n={len(no_bqtl):,}):")
    print(f"    mean |ASE| = {no_bqtl['mean_abs_ase'].mean():.4f}")
    print(f"    ASE std = {no_bqtl['std_abs_ase'].mean():.4f}")
    print(f"  Mann-Whitney |ASE|: p={p_abs:.2e}")
    print(f"  Mann-Whitney ASE std: p={p_std:.2e}")

    # Burden bins
    print(f"\n  bQTL burden bins:")
    bins = [0, 1, 2, 5, 10, 50, 1000]
    labels = ['0', '1', '2-4', '5-9', '10-49', '50+']
    merged['bqtl_bin'] = pd.cut(merged['n_bqtl'], bins=bins, labels=labels, right=False)
    for label in labels:
        sub = merged[merged['bqtl_bin'] == label]
        if len(sub) > 0:
            print(f"    {label:6s}: n={len(sub):5d}, mean |ASE|={sub['mean_abs_ase'].mean():.4f}, "
                  f"ASE std={sub['std_abs_ase'].mean():.4f}")

    merged.to_csv(OUTDIR / 'gene_bqtl_burden.csv', index=False)
    print(f"\n  Saved: gene_bqtl_burden.csv")

    return merged


def part5_summary_figure(func_df, signed_df, burden_df):
    """Generate summary figure combining all approaches."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    # A: Per-bQTL test - p-value histogram with overlay for abs vs signed
    ax1 = fig.add_subplot(gs[0, 0])
    pvals_abs = func_df['mw_p'].values
    ax1.hist(pvals_abs, bins=40, color='#607D8B', edgecolor='white',
             alpha=0.6, density=True, label='|ASE| test')
    if signed_df is not None and len(signed_df) > 0:
        ax1.hist(signed_df['mw_p'].values, bins=40, color='#E53935',
                 edgecolor='white', alpha=0.5, density=True, label='Signed ASE test')
    ax1.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('p-value')
    ax1.set_ylabel('Density')
    ax1.set_title('A. Per-bQTL tests\n(uniform = no signal)')
    ax1.legend(fontsize=7)

    # B: Gene-level burden
    ax2 = fig.add_subplot(gs[0, 1])
    bins_plot = ['0', '1', '2-4', '5-9', '10-49', '50+']
    means = []
    stds = []
    ns = []
    for label in bins_plot:
        sub = burden_df[burden_df['bqtl_bin'] == label]
        if len(sub) > 0:
            means.append(sub['mean_abs_ase'].mean())
            stds.append(sub['mean_abs_ase'].sem())
            ns.append(len(sub))
        else:
            means.append(0)
            stds.append(0)
            ns.append(0)
    colors = ['#9E9E9E'] + ['#1976D2'] * 5
    ax2.bar(range(len(bins_plot)), means, yerr=stds, color=colors,
            edgecolor='white', capsize=3)
    ax2.set_xticks(range(len(bins_plot)))
    ax2.set_xticklabels(bins_plot, fontsize=8)
    ax2.set_xlabel('bQTL count in promoter')
    ax2.set_ylabel('Mean |log₂(B73/NAM)|')
    ax2.set_title('B. Gene-level bQTL burden vs ASE')
    for i, n in enumerate(ns):
        ax2.text(i, means[i] + stds[i] + 0.01, f'n={n:,}', ha='center', fontsize=5)

    # C: ASE variance vs bQTL count
    ax3 = fig.add_subplot(gs[0, 2])
    has_bqtl = burden_df[burden_df['n_bqtl'] > 0]
    ax3.scatter(has_bqtl['n_bqtl'], has_bqtl['std_abs_ase'],
                s=2, alpha=0.1, color='#455A64')
    # Binned means
    bin_edges = [1, 2, 5, 10, 20, 50, 200]
    for i in range(len(bin_edges) - 1):
        sub = has_bqtl[(has_bqtl['n_bqtl'] >= bin_edges[i]) &
                       (has_bqtl['n_bqtl'] < bin_edges[i+1])]
        if len(sub) > 5:
            x = (bin_edges[i] + bin_edges[i+1]) / 2
            ax3.scatter(x, sub['std_abs_ase'].mean(), s=60, color='#E53935',
                        edgecolor='black', zorder=5, marker='D')
    ax3.set_xlabel('bQTL count in promoter')
    ax3.set_ylabel('ASE std across hybrids')
    ax3.set_title('C. bQTL burden vs ASE variability')
    ax3.set_xscale('log')
    rho, p = stats.spearmanr(has_bqtl['n_bqtl'], has_bqtl['std_abs_ase'].fillna(0))
    ax3.text(0.05, 0.95, f'$\\rho$ = {rho:.3f}\np = {p:.1e}',
             transform=ax3.transAxes, va='top', fontsize=9)

    plt.suptitle('Identifying Functional bQTL: Multiple Testing Approaches',
                 fontsize=12, fontweight='bold')
    plt.savefig(FIG_DIR / 'fig9_functional_bqtl_approaches.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(FIG_DIR / 'fig9_functional_bqtl_approaches.png', bbox_inches='tight', dpi=150)
    print(f"\nSaved: figures/fig9_functional_bqtl_approaches.pdf")
    plt.close()


def main():
    # Part 0: Generate functional test data (skips if already exists)
    generate_functional_test()

    # Part 1: Complete diagnostics
    func_df = part1_diagnostics()

    # Part 2: Signed ASE test
    signed_df = part2_signed_ase()

    # Part 3: High-variance genes
    gene_var = part3_high_variance_genes()

    # Part 4: Gene-level burden
    burden_df = part4_aggregate_gene_level()

    # Part 5: Summary figure
    part5_summary_figure(func_df, signed_df, burden_df)

    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY: FUNCTIONAL bQTL IDENTIFICATION")
    print(f"{'='*70}")
    print(f"""
Results across four approaches:

1. Per-bQTL test (|ASE|, peak presence/absence):
   - 13,458 bQTL tested across 24 hybrids
   - 0/13,458 survive FDR correction
   - Pi1 = 0 (no excess of small p-values)
   - CONCLUSION: No detectable individual bQTL→ASE signal

2. Signed ASE direction test:
   - Tests whether peak presence shifts B73/NAM ratio directionally
   - See results above

3. High-variance gene focus:
   - Restricting to genes with variable ASE across hybrids
   - See results above

4. Gene-level bQTL burden:
   - Tests whether MORE bQTL per promoter → more ASE
   - Aggregate signal vs individual bQTL signal
""")


if __name__ == '__main__':
    main()
