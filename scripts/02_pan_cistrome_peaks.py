#!/usr/bin/env python3
"""
MOA-seq peak-level disruption analysis.

Uses actual Engelhorn narrowPeak data to count independent binding events
per gene promoter, determine which peaks are disrupted by bQTL, and test
whether the fraction disrupted predicts allelic expression.

Key concept: instead of scanning for PWM motifs (too dense, ~200 per 500bp),
use the experimental MOA-seq peak calls as ground truth for where TFs bind.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
GFF3 = DATA / "raw" / "Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3"
PEAKS_DIR = DATA / "raw" / "engelhorn_peaks"
BQTL = DATA / "processed" / "bqtl_snp_ww.csv"
ASE = DATA / "processed" / "engelhorn_ase_ww_gene_summary.csv"
ASE_HYBRID = DATA / "processed" / "engelhorn_ase_ww.csv"
EXPR = DATA / "processed" / "engelhorn_ww_vs_ds_expression.tsv"
ENTAP = DATA / "processed" / "entap_annotations.tsv"
OUTDIR = BASE / "results"

PROMOTER_BP = 2000  # upstream of TSS


def parse_gff3():
    """Parse GFF3 for gene coordinates."""
    genes = {}
    with open(GFF3) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'gene':
                continue
            chrom = parts[0]
            if chrom not in {f'chr{i}' for i in range(1, 11)}:
                continue
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attrs = parts[8]
            gene_id = None
            for attr in attrs.split(';'):
                if attr.startswith('ID=gene:'):
                    gene_id = attr.split(':')[1]
                    break
                elif attr.startswith('ID=Zm'):
                    gene_id = attr[3:]
                    break
            if gene_id:
                tss = start if strand == '+' else end
                genes[gene_id] = {
                    'chr': chrom, 'start': start, 'end': end,
                    'strand': strand, 'tss': tss
                }
    return genes


def load_b73_peaks_ww():
    """Load all B73 peaks from well-watered narrowPeak files across all hybrids."""
    peak_files = sorted(PEAKS_DIR.glob('*.w2_4.q3_peaks.narrowPeak'))
    print(f"Loading B73 peaks from {len(peak_files)} WW hybrid files...")

    all_peaks = []
    for pf in peak_files:
        hybrid = pf.name.split('.')[0]
        with open(pf) as f:
            for line in f:
                parts = line.strip().split('\t')
                chrom_raw = parts[0]
                if not chrom_raw.startswith('B73-chr'):
                    continue
                chrom = chrom_raw.replace('B73-', '')  # B73-chr1 -> chr1
                start = int(parts[1])
                end = int(parts[2])
                score = int(parts[4])
                signal = float(parts[6])
                pval = float(parts[7])
                qval = float(parts[8])
                summit = int(parts[9])
                all_peaks.append({
                    'chr': chrom, 'start': start, 'end': end,
                    'hybrid': hybrid, 'score': score, 'signal': signal,
                    'pval': pval, 'qval': qval, 'summit_offset': summit,
                    'summit_pos': start + summit,
                })

    df = pd.DataFrame(all_peaks)
    print(f"  Total B73 peaks (WW): {len(df):,}")
    print(f"  Hybrids: {df['hybrid'].nunique()}")
    widths = df['end'] - df['start']
    print(f"  Peak width: mean={widths.mean():.1f}bp, median={widths.median():.0f}bp")
    return df


def build_pan_cistrome(peaks_df, merge_distance=100):
    """Merge overlapping peaks across hybrids into a pan-cistrome.

    Returns DataFrame with merged peak coordinates and hybrid count.
    """
    print(f"\nBuilding pan-cistrome (merge distance={merge_distance}bp)...")
    merged_peaks = []

    for chrom in sorted(peaks_df['chr'].unique()):
        chr_peaks = peaks_df[peaks_df['chr'] == chrom].sort_values('start')
        starts = chr_peaks['start'].values
        ends = chr_peaks['end'].values
        hybrids = chr_peaks['hybrid'].values

        # Merge overlapping/nearby peaks
        i = 0
        while i < len(starts):
            merge_start = starts[i]
            merge_end = ends[i]
            merge_hybrids = {hybrids[i]}
            j = i + 1
            while j < len(starts) and starts[j] <= merge_end + merge_distance:
                merge_end = max(merge_end, ends[j])
                merge_hybrids.add(hybrids[j])
                j += 1
            merged_peaks.append({
                'chr': chrom,
                'start': merge_start,
                'end': merge_end,
                'width': merge_end - merge_start,
                'n_hybrids': len(merge_hybrids),
                'n_raw_peaks': j - i,
            })
            i = j

    merged_df = pd.DataFrame(merged_peaks)
    print(f"  Pan-cistrome peaks: {len(merged_df):,}")
    print(f"  Width: mean={merged_df['width'].mean():.0f}bp, "
          f"median={merged_df['width'].median():.0f}bp")
    print(f"  Hybrid breadth: mean={merged_df['n_hybrids'].mean():.1f}, "
          f"median={merged_df['n_hybrids'].median():.0f}")
    print(f"  Genome coverage: {merged_df['width'].sum()/1e6:.1f} Mb "
          f"({100*merged_df['width'].sum()/2.1e9:.1f}% of genome)")

    # Hybrid breadth distribution
    for n in [1, 5, 10, 15, 20, 25]:
        pct = 100 * (merged_df['n_hybrids'] >= n).sum() / len(merged_df)
        print(f"    ≥{n:2d} hybrids: {(merged_df['n_hybrids'] >= n).sum():,} ({pct:.1f}%)")

    return merged_df


def map_peaks_to_promoters(merged_peaks, gene_coords):
    """Map pan-cistrome peaks to gene promoters (2kb upstream of TSS)."""
    print(f"\nMapping peaks to gene promoters ({PROMOTER_BP}bp upstream)...")

    import bisect

    # Build promoter intervals per chromosome
    promoters_by_chr = defaultdict(list)  # chr -> [(start, end, gene_id)]
    for gene_id, info in gene_coords.items():
        chrom = info['chr']
        if info['strand'] == '+':
            prom_start = max(0, info['tss'] - PROMOTER_BP)
            prom_end = info['tss']
        else:
            prom_start = info['tss']
            prom_end = info['tss'] + PROMOTER_BP
        promoters_by_chr[chrom].append((prom_start, prom_end, gene_id))

    for chrom in promoters_by_chr:
        promoters_by_chr[chrom].sort()

    # Map each peak to overlapping promoters
    peak_gene_map = []  # (peak_idx, gene_id)
    for idx, row in merged_peaks.iterrows():
        chrom = row['chr']
        peak_start = row['start']
        peak_end = row['end']
        for prom_start, prom_end, gene_id in promoters_by_chr.get(chrom, []):
            # Check overlap
            if peak_start < prom_end and peak_end > prom_start:
                peak_gene_map.append({
                    'peak_idx': idx,
                    'gene_id': gene_id,
                    'peak_chr': chrom,
                    'peak_start': peak_start,
                    'peak_end': peak_end,
                    'peak_width': row['width'],
                    'peak_n_hybrids': row['n_hybrids'],
                })

    map_df = pd.DataFrame(peak_gene_map)
    n_genes_with_peaks = map_df['gene_id'].nunique()
    n_peaks_in_promoters = map_df['peak_idx'].nunique()
    print(f"  Peak-gene mappings: {len(map_df):,}")
    print(f"  Genes with ≥1 peak in promoter: {n_genes_with_peaks:,} "
          f"({100*n_genes_with_peaks/len(gene_coords):.1f}%)")
    print(f"  Peaks in gene promoters: {n_peaks_in_promoters:,} "
          f"({100*n_peaks_in_promoters/len(merged_peaks):.1f}%)")

    # Peaks per gene
    peaks_per_gene = map_df.groupby('gene_id')['peak_idx'].nunique()
    print(f"  Peaks per gene: mean={peaks_per_gene.mean():.1f}, "
          f"median={peaks_per_gene.median():.0f}, max={peaks_per_gene.max()}")

    return map_df, peaks_per_gene


def map_bqtl_to_peaks(merged_peaks, bqtl_df):
    """Identify which pan-cistrome peaks contain bQTL."""
    print(f"\nMapping bQTL to pan-cistrome peaks...")

    import bisect

    # Build sorted peak index per chromosome
    peaks_by_chr = {}
    for chrom in merged_peaks['chr'].unique():
        chr_peaks = merged_peaks[merged_peaks['chr'] == chrom].sort_values('start')
        peaks_by_chr[chrom] = {
            'starts': chr_peaks['start'].values,
            'ends': chr_peaks['end'].values,
            'indices': chr_peaks.index.values,
        }

    bqtl_in_peak = []
    for _, row in bqtl_df.iterrows():
        chrom = row['chr']
        pos = row['pos']
        if chrom not in peaks_by_chr:
            continue
        pk = peaks_by_chr[chrom]
        idx = bisect.bisect_right(pk['starts'], pos) - 1
        if idx >= 0 and pos <= pk['ends'][idx]:
            bqtl_in_peak.append({
                'bqtl_pos': pos,
                'bqtl_chr': chrom,
                'peak_idx': pk['indices'][idx],
            })

    bqtl_peak_df = pd.DataFrame(bqtl_in_peak)
    n_bqtl_in_peaks = len(bqtl_peak_df)
    n_peaks_with_bqtl = bqtl_peak_df['peak_idx'].nunique() if len(bqtl_peak_df) > 0 else 0

    print(f"  bQTL in pan-cistrome peaks: {n_bqtl_in_peaks:,}/{len(bqtl_df):,} "
          f"({100*n_bqtl_in_peaks/len(bqtl_df):.1f}%)")
    print(f"  Peaks containing ≥1 bQTL: {n_peaks_with_bqtl:,}/{len(merged_peaks):,} "
          f"({100*n_peaks_with_bqtl/len(merged_peaks):.1f}%)")

    # bQTL per peak
    if len(bqtl_peak_df) > 0:
        bqtl_per_peak = bqtl_peak_df.groupby('peak_idx')['bqtl_pos'].nunique()
        print(f"  bQTL per disrupted peak: mean={bqtl_per_peak.mean():.1f}, "
              f"median={bqtl_per_peak.median():.0f}, max={bqtl_per_peak.max()}")

    return bqtl_peak_df


def build_gene_disruption_table(peak_gene_map, bqtl_peak_df, peaks_per_gene,
                                 ase_dict, expr_dict, drought_dict, annot_dict):
    """Build per-gene table: total peaks, disrupted peaks, fraction, ASE."""
    print(f"\nBuilding gene-level disruption table...")

    # Peaks with bQTL
    disrupted_peaks = set(bqtl_peak_df['peak_idx'].unique()) if len(bqtl_peak_df) > 0 else set()

    # Per gene: count total peaks and disrupted peaks
    gene_stats = []
    for gene_id, group in peak_gene_map.groupby('gene_id'):
        total_peaks = group['peak_idx'].nunique()
        gene_peak_ids = set(group['peak_idx'])
        disrupted = len(gene_peak_ids & disrupted_peaks)
        fraction_disrupted = disrupted / total_peaks if total_peaks > 0 else 0

        # Count bQTL in this gene's peaks
        gene_bqtl = bqtl_peak_df[bqtl_peak_df['peak_idx'].isin(gene_peak_ids)]
        n_bqtl = len(gene_bqtl)

        # Mean hybrid breadth of peaks
        mean_breadth = group['peak_n_hybrids'].mean()

        gene_stats.append({
            'gene_id': gene_id,
            'total_peaks': total_peaks,
            'disrupted_peaks': disrupted,
            'intact_peaks': total_peaks - disrupted,
            'fraction_disrupted': fraction_disrupted,
            'n_bqtl': n_bqtl,
            'mean_peak_breadth': mean_breadth,
            'abs_log2_ase': ase_dict.get(gene_id, np.nan),
            'expression': expr_dict.get(gene_id, np.nan),
            'drought_fc': drought_dict.get(gene_id, np.nan),
            'annotation': annot_dict.get(gene_id, ''),
        })

    gene_df = pd.DataFrame(gene_stats)
    print(f"  Genes in table: {len(gene_df):,}")
    print(f"  Genes with ≥1 disrupted peak: "
          f"{(gene_df['disrupted_peaks'] > 0).sum():,} "
          f"({100*(gene_df['disrupted_peaks'] > 0).sum()/len(gene_df):.1f}%)")

    return gene_df


def analyze_disruption(gene_df):
    """Analyze relationship between peak disruption and ASE."""
    from scipy import stats

    print(f"\n{'='*70}")
    print("PEAK-LEVEL DISRUPTION ANALYSIS")
    print(f"{'='*70}")

    has_ase = gene_df[gene_df['abs_log2_ase'].notna()].copy()
    print(f"Genes with ASE data: {len(has_ase):,}")

    # --- 1. Basic distributions ---
    print(f"\n--- 1. Peak count distributions ---")
    print(f"Total peaks per gene: mean={gene_df['total_peaks'].mean():.1f}, "
          f"median={gene_df['total_peaks'].median():.0f}")
    print(f"Disrupted peaks per gene: mean={gene_df['disrupted_peaks'].mean():.1f}, "
          f"median={gene_df['disrupted_peaks'].median():.0f}")
    print(f"Fraction disrupted: mean={gene_df['fraction_disrupted'].mean():.3f}, "
          f"median={gene_df['fraction_disrupted'].median():.3f}")

    for n in [0, 1, 2, 3, 5, 10]:
        pct = 100 * (gene_df['disrupted_peaks'] >= n).sum() / len(gene_df)
        print(f"  ≥{n} disrupted peaks: {(gene_df['disrupted_peaks'] >= n).sum():,} ({pct:.1f}%)")

    # --- 2. Total peaks vs ASE ---
    print(f"\n--- 2. Total peaks vs ASE ---")
    rho, p = stats.spearmanr(has_ase['total_peaks'], has_ase['abs_log2_ase'])
    print(f"  Spearman: rho={rho:.4f}, p={p:.2e}")

    # --- 3. Disrupted peaks vs ASE ---
    print(f"\n--- 3. Number of disrupted peaks vs ASE ---")
    rho, p = stats.spearmanr(has_ase['disrupted_peaks'], has_ase['abs_log2_ase'])
    print(f"  Spearman: rho={rho:.4f}, p={p:.2e}")

    bins = [0, 1, 2, 3, 5, 100]
    labels = ['0', '1', '2', '3-4', '5+']
    has_ase['disrupt_bin'] = pd.cut(has_ase['disrupted_peaks'], bins=bins, labels=labels, right=False)
    bin_stats = has_ase.groupby('disrupt_bin', observed=True).agg(
        n=('abs_log2_ase', 'count'),
        mean=('abs_log2_ase', 'mean'),
        median=('abs_log2_ase', 'median'),
        se=('abs_log2_ase', 'sem'),
    )
    print(f"  Disrupted peaks | N     | Mean |ASE| | Median | SE")
    for idx, row in bin_stats.iterrows():
        print(f"    {idx:10s}     | {row['n']:5.0f} | {row['mean']:.4f}  | "
              f"{row['median']:.4f} | {row['se']:.4f}")

    # Effect size: 0 disrupted vs 1+
    zero = has_ase[has_ase['disrupted_peaks'] == 0]['abs_log2_ase']
    one_plus = has_ase[has_ase['disrupted_peaks'] >= 1]['abs_log2_ase']
    if len(zero) > 10 and len(one_plus) > 10:
        u, up = stats.mannwhitneyu(one_plus, zero, alternative='greater')
        d = (one_plus.mean() - zero.mean()) / np.sqrt((one_plus.var() + zero.var()) / 2)
        print(f"\n  0 disrupted vs ≥1: MWU p={up:.2e}, Cohen's d={d:.3f}")
        print(f"    0: mean={zero.mean():.4f}, n={len(zero)}")
        print(f"    ≥1: mean={one_plus.mean():.4f}, n={len(one_plus)}")

    # --- 4. Fraction disrupted vs ASE ---
    print(f"\n--- 4. Fraction of peaks disrupted vs ASE ---")
    has_ase_with_peaks = has_ase[has_ase['total_peaks'] > 0].copy()
    rho, p = stats.spearmanr(has_ase_with_peaks['fraction_disrupted'],
                              has_ase_with_peaks['abs_log2_ase'])
    print(f"  Spearman: rho={rho:.4f}, p={p:.2e}")

    frac_bins = [0, 0.01, 0.1, 0.25, 0.5, 1.01]
    frac_labels = ['0%', '1-9%', '10-24%', '25-49%', '50-100%']
    has_ase_with_peaks['frac_bin'] = pd.cut(has_ase_with_peaks['fraction_disrupted'],
                                             bins=frac_bins, labels=frac_labels, right=False)
    frac_stats = has_ase_with_peaks.groupby('frac_bin', observed=True).agg(
        n=('abs_log2_ase', 'count'),
        mean=('abs_log2_ase', 'mean'),
        median=('abs_log2_ase', 'median'),
        se=('abs_log2_ase', 'sem'),
    )
    print(f"  Fraction disrupted | N     | Mean |ASE| | Median | SE")
    for idx, row in frac_stats.iterrows():
        print(f"    {idx:12s}       | {row['n']:5.0f} | {row['mean']:.4f}  | "
              f"{row['median']:.4f} | {row['se']:.4f}")

    # --- 5. Intact peaks as buffer ---
    print(f"\n--- 5. Intact peaks as buffer ---")
    disrupted_genes = has_ase[has_ase['disrupted_peaks'] >= 1].copy()
    if len(disrupted_genes) > 50:
        rho, p = stats.spearmanr(disrupted_genes['intact_peaks'],
                                  disrupted_genes['abs_log2_ase'])
        print(f"  Among genes with ≥1 disrupted peak (n={len(disrupted_genes)}):")
        print(f"  Intact peaks vs |ASE|: rho={rho:.4f}, p={p:.2e}")

        # Binned
        intact_bins = [0, 1, 3, 5, 10, 100]
        intact_labels = ['0', '1-2', '3-4', '5-9', '10+']
        disrupted_genes['intact_bin'] = pd.cut(disrupted_genes['intact_peaks'],
                                                bins=intact_bins, labels=intact_labels, right=False)
        intact_stats = disrupted_genes.groupby('intact_bin', observed=True).agg(
            n=('abs_log2_ase', 'count'),
            mean=('abs_log2_ase', 'mean'),
            se=('abs_log2_ase', 'sem'),
        )
        print(f"  Intact peaks | N     | Mean |ASE| | SE")
        for idx, row in intact_stats.iterrows():
            print(f"    {idx:8s}     | {row['n']:5.0f} | {row['mean']:.4f}  | {row['se']:.4f}")

    # --- 6. Peak hybrid breadth vs vulnerability ---
    print(f"\n--- 6. Peak conservation (hybrid breadth) ---")
    if len(disrupted_genes) > 50:
        rho, p = stats.spearmanr(disrupted_genes['mean_peak_breadth'],
                                  disrupted_genes['abs_log2_ase'])
        print(f"  Peak breadth vs |ASE|: rho={rho:.4f}, p={p:.2e}")

    return has_ase


def generate_figures(gene_df, has_ase):
    """Generate publication figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy import stats

    fig_dir = OUTDIR / 'figures'
    fig_dir.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Panel A: Peaks per gene histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(gene_df['total_peaks'], bins=range(0, 30), color='#2196F3',
             edgecolor='white', alpha=0.8)
    ax1.set_xlabel('MOA-seq peaks in promoter')
    ax1.set_ylabel('Genes')
    ax1.set_title(f'A. Binding events per gene\n'
                  f'(mean={gene_df["total_peaks"].mean():.1f}, '
                  f'median={gene_df["total_peaks"].median():.0f})')
    ax1.axvline(gene_df['total_peaks'].median(), color='k', ls='--', alpha=0.5)

    # Panel B: Disrupted peaks vs ASE (binned)
    ax2 = fig.add_subplot(gs[0, 1])
    bins = [0, 1, 2, 3, 5, 100]
    labels = ['0', '1', '2', '3-4', '5+']
    has_ase_copy = has_ase.copy()
    has_ase_copy['disrupt_bin'] = pd.cut(has_ase_copy['disrupted_peaks'],
                                          bins=bins, labels=labels, right=False)
    bin_stats = has_ase_copy.groupby('disrupt_bin', observed=True).agg(
        mean=('abs_log2_ase', 'mean'),
        se=('abs_log2_ase', 'sem'),
        n=('abs_log2_ase', 'count'),
    ).reset_index()
    x = range(len(bin_stats))
    ax2.bar(x, bin_stats['mean'], yerr=bin_stats['se'], color='#FF5722',
            edgecolor='white', capsize=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_stats['disrupt_bin'])
    ax2.set_xlabel('Disrupted peaks (with bQTL)')
    ax2.set_ylabel('Mean |log₂(B73/NAM)|')
    rho, p = stats.spearmanr(has_ase['disrupted_peaks'], has_ase['abs_log2_ase'])
    ax2.set_title(f'B. Disrupted peaks vs ASE\nρ={rho:.3f}, p={p:.1e}')
    for i, row in bin_stats.iterrows():
        ax2.text(i, row['mean'] + row['se'] + 0.01,
                 f'n={row["n"]:.0f}', ha='center', fontsize=6)

    # Panel C: Fraction disrupted vs ASE
    ax3 = fig.add_subplot(gs[0, 2])
    has_peaks = has_ase[has_ase['total_peaks'] > 0].copy()
    frac_bins = [0, 0.01, 0.1, 0.25, 0.5, 1.01]
    frac_labels = ['0%', '1-9%', '10-24%', '25-49%', '50-100%']
    has_peaks['frac_bin'] = pd.cut(has_peaks['fraction_disrupted'],
                                    bins=frac_bins, labels=frac_labels, right=False)
    frac_stats = has_peaks.groupby('frac_bin', observed=True).agg(
        mean=('abs_log2_ase', 'mean'),
        se=('abs_log2_ase', 'sem'),
        n=('abs_log2_ase', 'count'),
    ).reset_index()
    x = range(len(frac_stats))
    ax3.bar(x, frac_stats['mean'], yerr=frac_stats['se'], color='#9C27B0',
            edgecolor='white', capsize=3)
    ax3.set_xticks(x)
    ax3.set_xticklabels(frac_stats['frac_bin'], fontsize=8)
    ax3.set_xlabel('Fraction of peaks disrupted')
    ax3.set_ylabel('Mean |log₂(B73/NAM)|')
    rho2, p2 = stats.spearmanr(has_peaks['fraction_disrupted'], has_peaks['abs_log2_ase'])
    ax3.set_title(f'C. Fraction disrupted vs ASE\nρ={rho2:.3f}, p={p2:.1e}')
    for i, row in frac_stats.iterrows():
        ax3.text(i, row['mean'] + row['se'] + 0.01,
                 f'n={row["n"]:.0f}', ha='center', fontsize=6)

    # Panel D: Intact peaks as buffer (among disrupted genes)
    ax4 = fig.add_subplot(gs[1, 0])
    disrupted = has_ase[has_ase['disrupted_peaks'] >= 1].copy()
    if len(disrupted) > 50:
        intact_bins = [0, 1, 3, 5, 10, 100]
        intact_labels = ['0', '1-2', '3-4', '5-9', '10+']
        disrupted['intact_bin'] = pd.cut(disrupted['intact_peaks'],
                                          bins=intact_bins, labels=intact_labels, right=False)
        intact_stats = disrupted.groupby('intact_bin', observed=True).agg(
            mean=('abs_log2_ase', 'mean'),
            se=('abs_log2_ase', 'sem'),
            n=('abs_log2_ase', 'count'),
        ).reset_index()
        x = range(len(intact_stats))
        ax4.bar(x, intact_stats['mean'], yerr=intact_stats['se'], color='#4CAF50',
                edgecolor='white', capsize=3)
        ax4.set_xticks(x)
        ax4.set_xticklabels(intact_stats['intact_bin'])
        ax4.set_xlabel('Intact (non-disrupted) peaks')
        ax4.set_ylabel('Mean |log₂(B73/NAM)|')
        rho3, p3 = stats.spearmanr(disrupted['intact_peaks'], disrupted['abs_log2_ase'])
        ax4.set_title(f'D. Intact peaks buffer ASE?\nρ={rho3:.3f}, p={p3:.1e}')
        for i, row in intact_stats.iterrows():
            ax4.text(i, row['mean'] + row['se'] + 0.01,
                     f'n={row["n"]:.0f}', ha='center', fontsize=6)

    # Panel E: Disrupted peaks vs hybrid breadth scatter
    ax5 = fig.add_subplot(gs[1, 1])
    if len(disrupted) > 50:
        ax5.scatter(disrupted['mean_peak_breadth'], disrupted['abs_log2_ase'],
                    alpha=0.1, s=5, color='#FF9800')
        ax5.set_xlabel('Mean peak hybrid breadth')
        ax5.set_ylabel('|log₂(B73/NAM)|')
        ax5.set_title('E. Peak conservation vs ASE')

    # Panel F: Total peaks vs disrupted peaks
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(gene_df['total_peaks'], gene_df['disrupted_peaks'],
                alpha=0.05, s=3, color='#607D8B')
    ax6.plot([0, gene_df['total_peaks'].max()],
             [0, gene_df['total_peaks'].max()], 'r--', alpha=0.3, label='100% disrupted')
    ax6.set_xlabel('Total MOA-seq peaks in promoter')
    ax6.set_ylabel('Peaks with bQTL (disrupted)')
    ax6.set_title(f'F. Total vs disrupted peaks\n'
                  f'mean fraction: {gene_df["fraction_disrupted"].mean():.1%}')
    ax6.legend(fontsize=8)

    plt.suptitle('MOA-seq Peak Disruption Analysis', fontsize=13, fontweight='bold')
    plt.savefig(fig_dir / 'fig6_peak_disruption.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(fig_dir / 'fig6_peak_disruption.png', bbox_inches='tight', dpi=150)
    print(f"\nSaved: figures/fig6_peak_disruption.pdf")
    plt.close()


def main():
    print("=" * 70)
    print("MOA-SEQ PEAK DISRUPTION ANALYSIS")
    print("=" * 70)

    # Load data
    gene_coords = parse_gff3()
    bqtl = pd.read_csv(BQTL)
    ase_df = pd.read_csv(ASE)
    ase_dict = dict(zip(ase_df['gene_id'], ase_df['mean_abs_log2']))

    expr_df = pd.read_csv(EXPR, sep='\t')
    expr_dict = dict(zip(expr_df['gene_id'], expr_df['mean_expression']))
    fc_col = next((c for c in expr_df.columns if 'log2' in c.lower() and 'fc' in c.lower()), None)
    drought_dict = dict(zip(expr_df['gene_id'], expr_df[fc_col])) if fc_col else {}

    annot_dict = {}
    if ENTAP.exists():
        annot_df = pd.read_csv(ENTAP, sep='\t')
        for _, row in annot_df.iterrows():
            gid = str(row.get('Query Sequence', ''))
            desc = str(row.get('Description', ''))
            if gid.startswith('Zm') and desc != 'nan':
                annot_dict[gid] = desc

    print(f"Genes: {len(gene_coords):,}")
    print(f"bQTL: {len(bqtl):,}")
    print(f"ASE: {len(ase_dict):,} genes")

    # 1. Load peaks
    peaks_df = load_b73_peaks_ww()

    # 2. Build pan-cistrome
    merged = build_pan_cistrome(peaks_df, merge_distance=100)

    # 3. Map peaks to promoters
    peak_gene_map, peaks_per_gene = map_peaks_to_promoters(merged, gene_coords)

    # 4. Map bQTL to peaks
    bqtl_peak_df = map_bqtl_to_peaks(merged, bqtl)

    # 5. Build gene-level table
    gene_df = build_gene_disruption_table(
        peak_gene_map, bqtl_peak_df, peaks_per_gene,
        ase_dict, expr_dict, drought_dict, annot_dict
    )

    # 6. Analyze
    has_ase = analyze_disruption(gene_df)

    # 7. Figures
    generate_figures(gene_df, has_ase)

    # 8. Save
    gene_df.to_csv(OUTDIR / 'gene_peak_disruption.csv', index=False)
    merged.to_csv(OUTDIR / 'pan_cistrome_peaks.csv', index=False)
    print(f"\nSaved: gene_peak_disruption.csv, pan_cistrome_peaks.csv")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Pan-cistrome: {len(merged):,} merged peaks")
    print(f"Genes with peaks: {gene_df['gene_id'].nunique():,}")
    print(f"Mean peaks/gene: {gene_df['total_peaks'].mean():.1f}")
    print(f"Mean disrupted/gene: {gene_df['disrupted_peaks'].mean():.1f}")
    print(f"Mean fraction disrupted: {gene_df['fraction_disrupted'].mean():.1%}")


if __name__ == '__main__':
    main()
