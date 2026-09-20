#!/usr/bin/env python3
"""
ASE direction as genotype proxy — functional bQTL identification.

Key insight: For genes where ASE direction FLIPS across hybrids (some B73-high,
some NAM-high), the direction grouping approximates genotype. If a bQTL in the
promoter drives ASE, peak presence should predict ASE direction.

Test: Fisher's exact on 2x2 table {peak_present, peak_absent} x {B73-high, NAM-high}
This is more powerful than magnitude-based tests because it's binary, not continuous.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import defaultdict
from statsmodels.stats.multitest import multipletests
import bisect
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUTDIR = BASE / "results"
FIG_DIR = BASE / "figures" / "paper_figures"
FIG_DIR.mkdir(exist_ok=True)

BQTL = DATA / "processed" / "bqtl_snp_ww.csv"
ASE_HYBRID = DATA / "processed" / "engelhorn_ase_ww.csv"
GFF3 = DATA / "raw" / "Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3"
PEAKS_DIR = DATA / "raw" / "engelhorn_peaks"
PAN_CISTROME = OUTDIR / "pan_cistrome_peaks.csv"

VALID_CHROMS = {f'chr{i}' for i in range(1, 11)}
PROMOTER_BP = 2000

HYBRIDS = ['A188', 'A619', 'B97', 'CML103', 'CML247', 'CML277', 'CML322',
           'CML333', 'CML69', 'HP301', 'IL14H', 'Ki11', 'Ki3', 'Ky21',
           'M162W', 'Mo17', 'Mo18W', 'Ms71', 'NC358', 'Oh43', 'Oh7b',
           'P39', 'Tx303', 'W22']


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


def load_hybrid_peaks():
    """Load per-hybrid peak intervals for fast lookup."""
    hybrid_peaks = {}
    for hybrid in HYBRIDS:
        peak_file = PEAKS_DIR / f"{hybrid}.w2_4.q3_peaks.narrowPeak"
        if not peak_file.exists():
            continue
        peaks_by_chr = defaultdict(list)
        with open(peak_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                chrom_raw = parts[0]
                if not chrom_raw.startswith('B73-chr'):
                    continue
                chrom = chrom_raw.replace('B73-', '')
                if chrom not in VALID_CHROMS:
                    continue
                peaks_by_chr[chrom].append((int(parts[1]), int(parts[2])))
        for chrom in peaks_by_chr:
            peaks_by_chr[chrom].sort()
        hybrid_peaks[hybrid] = peaks_by_chr
    return hybrid_peaks


def pos_has_peak(hybrid_peaks, hybrid, chrom, pos):
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


def load_pan_cistrome():
    """Load pan-cistrome peaks for fast positional lookup."""
    pan_df = pd.read_csv(PAN_CISTROME)
    pan_by_chr = {}
    for chrom, grp in pan_df.groupby('chr'):
        pan_by_chr[chrom] = grp[['start', 'end']].values
        pan_by_chr[chrom].sort(axis=0)
    return pan_by_chr


def pos_in_pan_peak(pan_by_chr, chrom, pos):
    peaks = pan_by_chr.get(chrom)
    if peaks is None or len(peaks) == 0:
        return False
    idx = bisect.bisect_right(peaks[:, 0], pos) - 1
    if idx >= 0 and peaks[idx, 0] <= pos <= peaks[idx, 1]:
        return True
    return False


# ============================================================
# PART 1: Identify variable-direction ASE genes
# ============================================================

def identify_variable_ase_genes(ase_df):
    """Find genes where ASE direction flips across hybrids."""
    print("=" * 70)
    print("PART 1: VARIABLE ASE DIRECTION GENES")
    print("=" * 70)

    # Per gene per hybrid: direction of ASE
    gene_hybrid = ase_df.groupby(['gene_id', 'hybrid']).agg(
        log2_ratio=('log2_ratio', 'mean'),
        abs_ase=('abs_log2_ratio', 'mean'),
        total_reads=('total_reads', 'mean'),
    ).reset_index()

    # Normalize hybrid names: "B73xMo17" -> "Mo17"
    gene_hybrid['hybrid_short'] = gene_hybrid['hybrid'].str.replace('B73x', '', regex=False)

    # Classify direction per hybrid
    gene_hybrid['direction'] = np.where(gene_hybrid['log2_ratio'] > 0, 'B73', 'NAM')

    # Per gene: count B73-high vs NAM-high hybrids
    gene_dir = gene_hybrid.groupby('gene_id').agg(
        n_hybrids=('hybrid', 'nunique'),
        n_b73_high=('direction', lambda x: (x == 'B73').sum()),
        n_nam_high=('direction', lambda x: (x == 'NAM').sum()),
        mean_abs_ase=('abs_ase', 'mean'),
        mean_signed=('log2_ratio', 'mean'),
        std_signed=('log2_ratio', 'std'),
    ).reset_index()

    gene_dir['frac_b73'] = gene_dir['n_b73_high'] / gene_dir['n_hybrids']
    gene_dir['minority_frac'] = gene_dir[['frac_b73', 'frac_b73']].apply(
        lambda x: min(x.iloc[0], 1 - x.iloc[0]), axis=1)

    # Variable = both groups have >= 3 hybrids (Fisher's needs cell counts)
    gene_dir['variable'] = (gene_dir['n_b73_high'] >= 3) & (gene_dir['n_nam_high'] >= 3)
    gene_dir['consistent_b73'] = gene_dir['frac_b73'] >= 0.8
    gene_dir['consistent_nam'] = gene_dir['frac_b73'] <= 0.2

    n_var = gene_dir['variable'].sum()
    n_b73 = gene_dir['consistent_b73'].sum()
    n_nam = gene_dir['consistent_nam'].sum()
    n_total = len(gene_dir)

    print(f"Total genes with ASE: {n_total:,}")
    print(f"  Consistently B73-high (>=80%): {n_b73:,} ({100*n_b73/n_total:.1f}%)")
    print(f"  Consistently NAM-high (>=80%): {n_nam:,} ({100*n_nam/n_total:.1f}%)")
    print(f"  Variable direction (>=3 each): {n_var:,} ({100*n_var/n_total:.1f}%)")
    print(f"  Median minority fraction (variable): "
          f"{gene_dir.loc[gene_dir['variable'], 'minority_frac'].median():.3f}")

    # Stronger ASE genes are more interesting
    strong_ase = gene_dir[(gene_dir['mean_abs_ase'] >= 0.3) & gene_dir['variable']]
    print(f"  Variable + strong ASE (|log2|>=0.3): {len(strong_ase):,}")

    return gene_dir, gene_hybrid


# ============================================================
# PART 2: Per-bQTL concordance test (Fisher's exact)
# ============================================================

def per_bqtl_concordance_test(gene_dir, gene_hybrid, gene_coords, hybrid_peaks, pan_by_chr):
    """For each bQTL in a variable ASE gene's promoter, test if peak presence
    predicts ASE direction (Fisher's exact 2x2)."""
    print(f"\n{'='*70}")
    print("PART 2: PER-bQTL CONCORDANCE TEST (peak x ASE direction)")
    print(f"{'='*70}")

    bqtl_df = pd.read_csv(BQTL)
    print(f"Total bQTL: {len(bqtl_df):,}")

    # Get variable genes
    var_genes = set(gene_dir[gene_dir['variable']]['gene_id'])
    print(f"Variable ASE genes: {len(var_genes):,}")

    # Build per-gene direction lookup: {gene_id: {hybrid_short: 'B73'/'NAM'}}
    dir_lookup = {}
    for _, row in gene_hybrid.iterrows():
        gene_id = row['gene_id']
        hybrid_short = row['hybrid_short']
        if gene_id in var_genes:
            if gene_id not in dir_lookup:
                dir_lookup[gene_id] = {}
            dir_lookup[gene_id][hybrid_short] = 'B73' if row['log2_ratio'] > 0 else 'NAM'

    # Build promoter index
    promoters_by_chr = defaultdict(list)
    for gene_id, info in gene_coords.items():
        if gene_id not in var_genes:
            continue
        chrom = info['chr']
        if info['strand'] == '+':
            ps = max(0, info['tss'] - PROMOTER_BP)
            pe = info['tss']
        else:
            ps = info['tss']
            pe = info['tss'] + PROMOTER_BP
        promoters_by_chr[chrom].append((ps, pe, gene_id, info['tss']))
    for chrom in promoters_by_chr:
        promoters_by_chr[chrom].sort()

    # Map bQTL to variable ASE genes
    print("Mapping bQTL to variable ASE gene promoters...")
    bqtl_gene_pairs = []
    for _, row in bqtl_df.iterrows():
        chrom, pos = row['chr'], row['pos']
        promoters = promoters_by_chr.get(chrom, [])
        starts = [p[0] for p in promoters]
        idx = bisect.bisect_right(starts, pos)
        for j in range(max(0, idx - 5), min(len(promoters), idx + 1)):
            if promoters[j][0] <= pos <= promoters[j][1]:
                gene_id = promoters[j][2]
                tss = promoters[j][3]
                bqtl_gene_pairs.append({
                    'chr': chrom, 'pos': pos, 'gene_id': gene_id,
                    'bqtl_type': row['Type'], 'bqtl_fdr': row['FDRp'],
                    'dist_to_tss': abs(pos - tss),
                })

    pairs_df = pd.DataFrame(bqtl_gene_pairs)
    print(f"  bQTL-gene pairs (variable genes): {len(pairs_df):,}")
    print(f"  Unique bQTL: {pairs_df[['chr','pos']].drop_duplicates().shape[0]:,}")
    print(f"  Unique genes: {pairs_df['gene_id'].nunique():,}")

    # Run Fisher's exact for each bQTL-gene pair
    print("\nRunning Fisher's exact tests...")
    results = []
    for _, pair in pairs_df.iterrows():
        chrom, pos, gene_id = pair['chr'], pair['pos'], pair['gene_id']
        directions = dir_lookup.get(gene_id, {})

        # Build 2x2: rows = peak {present, absent}, cols = ASE direction {B73, NAM}
        a, b, c, d = 0, 0, 0, 0  # a=peak+B73, b=peak+NAM, c=nopeak+B73, d=nopeak+NAM
        for hybrid in HYBRIDS:
            ase_dir = directions.get(hybrid)
            if ase_dir is None:
                continue
            has_peak = pos_has_peak(hybrid_peaks, hybrid, chrom, pos)
            if has_peak:
                if ase_dir == 'B73':
                    a += 1
                else:
                    b += 1
            else:
                if ase_dir == 'B73':
                    c += 1
                else:
                    d += 1

        total = a + b + c + d
        if total < 6:
            continue
        # Need at least 1 in each margin for a valid test
        if (a + b) < 2 or (c + d) < 2 or (a + c) < 2 or (b + d) < 2:
            continue

        odds_ratio, fisher_p = stats.fisher_exact([[a, b], [c, d]])

        # Also compute concordance: fraction of hybrids where peak<->direction matches
        # If OR > 1: concordant = peak+B73 or nopeak+NAM
        # If OR < 1: concordant = peak+NAM or nopeak+B73
        if odds_ratio >= 1:
            concordant = a + d
        else:
            concordant = b + c
        concordance = concordant / total

        results.append({
            'chr': chrom, 'pos': pos, 'gene_id': gene_id,
            'bqtl_type': pair['bqtl_type'],
            'bqtl_fdr': pair['bqtl_fdr'],
            'dist_to_tss': pair['dist_to_tss'],
            'peak_b73': a, 'peak_nam': b, 'nopeak_b73': c, 'nopeak_nam': d,
            'n_hybrids': total,
            'odds_ratio': odds_ratio,
            'fisher_p': fisher_p,
            'concordance': concordance,
        })

    rdf = pd.DataFrame(results)
    print(f"  Tested: {len(rdf):,} bQTL-gene pairs")

    if len(rdf) == 0:
        print("  No testable pairs!")
        return None

    # FDR correction
    _, rdf['fdr'], _, _ = multipletests(rdf['fisher_p'], method='fdr_bh')

    n_nom = (rdf['fisher_p'] < 0.05).sum()
    n_fdr = (rdf['fdr'] < 0.05).sum()
    n_fdr10 = (rdf['fdr'] < 0.10).sum()
    n_fdr20 = (rdf['fdr'] < 0.20).sum()

    print(f"\n  Nominal p<0.05: {n_nom} ({100*n_nom/len(rdf):.1f}%)")
    print(f"  Expected by chance: {int(len(rdf)*0.05)} (5%)")
    print(f"  Enrichment ratio: {n_nom / (len(rdf)*0.05):.2f}x")
    print(f"  FDR<0.05: {n_fdr}")
    print(f"  FDR<0.10: {n_fdr10}")
    print(f"  FDR<0.20: {n_fdr20}")

    # Pi1 estimate
    pvals = rdf['fisher_p'].dropna().values
    for thr in [0.05, 0.10, 0.20]:
        frac_below = (pvals < thr).mean()
        pi0 = (1 - frac_below) / (1 - thr)
        pi1 = max(0, 1 - pi0)
        print(f"  Pi1 at lambda={thr}: {pi1:.4f}")

    # Mean concordance
    print(f"\n  Mean concordance: {rdf['concordance'].mean():.4f} (random=0.50)")
    print(f"  Concordance > 0.6: {(rdf['concordance'] > 0.6).sum()} "
          f"({100*(rdf['concordance'] > 0.6).mean():.1f}%)")
    print(f"  Concordance > 0.7: {(rdf['concordance'] > 0.7).sum()} "
          f"({100*(rdf['concordance'] > 0.7).mean():.1f}%)")
    print(f"  Concordance > 0.8: {(rdf['concordance'] > 0.8).sum()} "
          f"({100*(rdf['concordance'] > 0.8).mean():.1f}%)")

    # Effect size: median odds ratio among nominal hits
    nom_hits = rdf[rdf['fisher_p'] < 0.05]
    if len(nom_hits) > 0:
        # OR can be inf, use log
        finite_or = nom_hits['odds_ratio'].replace([np.inf, 0], np.nan).dropna()
        if len(finite_or) > 0:
            log_or = np.log2(finite_or)
            print(f"\n  Nominal hits (n={len(nom_hits)}):")
            print(f"    Median |log2(OR)|: {np.median(np.abs(log_or)):.3f}")
            print(f"    Mean concordance: {nom_hits['concordance'].mean():.4f}")
            print(f"    OR>1 (peak→B73): {(nom_hits['odds_ratio'] > 1).sum()}")
            print(f"    OR<1 (peak→NAM): {(nom_hits['odds_ratio'] < 1).sum()}")

    # Stratify by bQTL type
    print(f"\n  By bQTL type:")
    for btype in sorted(rdf['bqtl_type'].unique()):
        sub = rdf[rdf['bqtl_type'] == btype]
        n_sig = (sub['fisher_p'] < 0.05).sum()
        print(f"    {btype:10s}: n={len(sub):5d}, nom sig={n_sig:4d} ({100*n_sig/len(sub):.1f}%), "
              f"conc={sub['concordance'].mean():.4f}")

    # Stratify by distance to TSS
    print(f"\n  By distance to TSS:")
    for lo, hi, label in [(0, 200, '<200bp'), (200, 500, '200-500bp'),
                           (500, 1000, '500-1000bp'), (1000, 2000, '1-2kb')]:
        sub = rdf[(rdf['dist_to_tss'] >= lo) & (rdf['dist_to_tss'] < hi)]
        if len(sub) > 0:
            n_sig = (sub['fisher_p'] < 0.05).sum()
            print(f"    {label:10s}: n={len(sub):5d}, nom sig={n_sig:4d} ({100*n_sig/len(sub):.1f}%), "
                  f"conc={sub['concordance'].mean():.4f}")

    rdf.to_csv(OUTDIR / 'bqtl_ase_direction_concordance.csv', index=False)
    print(f"\n  Saved: bqtl_ase_direction_concordance.csv")

    return rdf


# ============================================================
# PART 3: Gene-level concordance aggregation
# ============================================================

def gene_level_concordance(rdf, gene_dir):
    """For each variable ASE gene, aggregate concordance across its bQTL."""
    print(f"\n{'='*70}")
    print("PART 3: GENE-LEVEL CONCORDANCE AGGREGATION")
    print(f"{'='*70}")

    gene_agg = rdf.groupby('gene_id').agg(
        n_bqtl=('pos', 'nunique'),
        mean_concordance=('concordance', 'mean'),
        max_concordance=('concordance', 'max'),
        min_fisher_p=('fisher_p', 'min'),
        n_nominal=('fisher_p', lambda x: (x < 0.05).sum()),
        mean_odds_ratio=('odds_ratio', lambda x: x.replace([np.inf, 0], np.nan).dropna().mean()),
    ).reset_index()

    # Merge with gene direction info
    gene_agg = gene_agg.merge(
        gene_dir[['gene_id', 'n_hybrids', 'n_b73_high', 'n_nam_high',
                  'mean_abs_ase', 'std_signed']],
        on='gene_id', how='left'
    )

    print(f"Genes tested: {len(gene_agg):,}")
    print(f"  With >=1 nominal hit: {(gene_agg['n_nominal'] >= 1).sum()}")
    print(f"  With >=2 nominal hits: {(gene_agg['n_nominal'] >= 2).sum()}")

    # Gene-level Fisher combined p-value (Fisher's method)
    gene_fisher_p = []
    for _, grow in gene_agg.iterrows():
        gene_pvals = rdf[rdf['gene_id'] == grow['gene_id']]['fisher_p'].values
        if len(gene_pvals) >= 2:
            # Fisher's combined test
            chi2 = -2 * np.sum(np.log(gene_pvals + 1e-300))
            combined_p = stats.chi2.sf(chi2, 2 * len(gene_pvals))
            gene_fisher_p.append(combined_p)
        elif len(gene_pvals) == 1:
            gene_fisher_p.append(gene_pvals[0])
        else:
            gene_fisher_p.append(np.nan)

    gene_agg['combined_p'] = gene_fisher_p
    valid = gene_agg['combined_p'].notna()
    if valid.sum() > 0:
        _, gene_agg.loc[valid, 'combined_fdr'], _, _ = multipletests(
            gene_agg.loc[valid, 'combined_p'], method='fdr_bh')
    else:
        gene_agg['combined_fdr'] = np.nan

    n_gene_sig = (gene_agg['combined_fdr'] < 0.05).sum()
    n_gene_sig10 = (gene_agg['combined_fdr'] < 0.10).sum()
    n_gene_sig20 = (gene_agg['combined_fdr'] < 0.20).sum()

    print(f"\n  Gene-level combined test (Fisher's method):")
    print(f"    FDR<0.05: {n_gene_sig}")
    print(f"    FDR<0.10: {n_gene_sig10}")
    print(f"    FDR<0.20: {n_gene_sig20}")

    # Top genes by combined p-value
    top = gene_agg.nsmallest(20, 'combined_p')
    print(f"\n  Top 20 genes by combined p:")
    print(f"  {'gene_id':20s} {'n_bqtl':>6s} {'mean_conc':>9s} {'comb_p':>10s} {'fdr':>10s} "
          f"{'|ASE|':>6s} {'n_B73':>5s} {'n_NAM':>5s}")
    for _, row in top.iterrows():
        print(f"  {row['gene_id']:20s} {row['n_bqtl']:6d} {row['mean_concordance']:9.4f} "
              f"{row['combined_p']:10.2e} {row['combined_fdr']:10.4f} "
              f"{row['mean_abs_ase']:6.3f} {row['n_b73_high']:5.0f} {row['n_nam_high']:5.0f}")

    # Correlation: gene mean concordance vs ASE magnitude
    rho, p = stats.spearmanr(gene_agg['mean_concordance'], gene_agg['mean_abs_ase'])
    print(f"\n  Gene concordance vs |ASE|: rho={rho:.4f}, p={p:.2e}")
    rho2, p2 = stats.spearmanr(gene_agg['mean_concordance'], gene_agg['std_signed'])
    print(f"  Gene concordance vs ASE variability: rho={rho2:.4f}, p={p2:.2e}")

    # Do genes with more bQTL show higher concordance?
    rho3, p3 = stats.spearmanr(gene_agg['n_bqtl'], gene_agg['mean_concordance'])
    print(f"  n_bQTL vs concordance: rho={rho3:.4f}, p={p3:.2e}")

    gene_agg.to_csv(OUTDIR / 'gene_ase_direction_concordance.csv', index=False)
    print(f"\n  Saved: gene_ase_direction_concordance.csv")

    return gene_agg


# ============================================================
# PART 4: Comparison — direction test vs magnitude test
# ============================================================

def compare_tests(rdf):
    """Compare this direction-based test to the magnitude-based test from script 50."""
    print(f"\n{'='*70}")
    print("PART 4: DIRECTION TEST vs MAGNITUDE TEST COMPARISON")
    print(f"{'='*70}")

    mag_csv = OUTDIR / 'bqtl_functional_test.csv'
    if not mag_csv.exists():
        print("  Magnitude test results not found, skipping comparison.")
        return

    mag_df = pd.read_csv(mag_csv)

    # Find overlap
    rdf_key = rdf[['chr', 'pos', 'gene_id']].drop_duplicates()
    mag_key = mag_df[['chr', 'pos', 'gene_id']].drop_duplicates()

    merged = rdf.merge(mag_df[['chr', 'pos', 'gene_id', 'mw_p', 'cohens_d', 'ase_diff']],
                        on=['chr', 'pos', 'gene_id'], how='inner',
                        suffixes=('_dir', '_mag'))

    if len(merged) > 0:
        print(f"  Overlapping bQTL-gene pairs: {len(merged):,}")

        # Compare p-values
        rho, p = stats.spearmanr(-np.log10(merged['fisher_p']),
                                  -np.log10(merged['mw_p']))
        print(f"  -log10(p) correlation: rho={rho:.4f}, p={p:.2e}")

        # How many nominally sig in each?
        sig_dir = merged['fisher_p'] < 0.05
        sig_mag = merged['mw_p'] < 0.05
        print(f"  Direction test sig: {sig_dir.sum()}")
        print(f"  Magnitude test sig: {sig_mag.sum()}")
        print(f"  Both sig: {(sig_dir & sig_mag).sum()}")
        print(f"  Direction only: {(sig_dir & ~sig_mag).sum()}")
        print(f"  Magnitude only: {(~sig_dir & sig_mag).sum()}")
    else:
        print("  No overlapping bQTL-gene pairs found.")


# ============================================================
# PART 5: Figures
# ============================================================

def make_figures(rdf, gene_agg, gene_dir):
    """Generate summary figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # A: ASE direction consistency across genes
    ax1 = fig.add_subplot(gs[0, 0])
    frac_b73 = gene_dir['frac_b73'].values
    ax1.hist(frac_b73, bins=50, color='#1976D2', edgecolor='white', alpha=0.8)
    ax1.axvline(0.5, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Fraction of hybrids with B73-high ASE')
    ax1.set_ylabel('Number of genes')
    ax1.set_title('A. ASE direction consistency')
    n_var = gene_dir['variable'].sum()
    ax1.text(0.05, 0.95, f'Variable genes:\n{n_var:,} ({100*n_var/len(gene_dir):.0f}%)',
             transform=ax1.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # B: P-value histogram from concordance test
    ax2 = fig.add_subplot(gs[0, 1])
    pvals = rdf['fisher_p'].values
    ax2.hist(pvals, bins=40, color='#E53935', edgecolor='white', alpha=0.8, density=True)
    ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='H0 (uniform)')
    ax2.set_xlabel("Fisher's exact p-value")
    ax2.set_ylabel('Density')
    ax2.set_title('B. Per-bQTL direction concordance test')
    n_nom = (pvals < 0.05).sum()
    ax2.text(0.95, 0.95, f'Nominal: {n_nom}/{len(rdf)}\n'
             f'({100*n_nom/len(rdf):.1f}%, expect 5%)',
             transform=ax2.transAxes, va='top', ha='right', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax2.legend(fontsize=8)

    # C: Concordance distribution
    ax3 = fig.add_subplot(gs[0, 2])
    conc = rdf['concordance'].values
    ax3.hist(conc, bins=30, color='#43A047', edgecolor='white', alpha=0.8)
    ax3.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax3.set_xlabel('Concordance (peak ↔ ASE direction)')
    ax3.set_ylabel('Count')
    ax3.set_title('C. Concordance distribution')
    ax3.text(0.95, 0.95, f'Mean: {conc.mean():.4f}\nMedian: {np.median(conc):.4f}',
             transform=ax3.transAxes, va='top', ha='right', fontsize=9)
    ax3.legend(fontsize=8)

    # D: Odds ratio distribution (log2 scale, cap at ±5)
    ax4 = fig.add_subplot(gs[1, 0])
    finite_or = rdf['odds_ratio'].replace([np.inf, 0], np.nan).dropna()
    log_or = np.log2(finite_or).clip(-5, 5)
    ax4.hist(log_or, bins=40, color='#7B1FA2', edgecolor='white', alpha=0.8)
    ax4.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('log₂(odds ratio)')
    ax4.set_ylabel('Count')
    ax4.set_title('D. Effect sizes (peak → ASE direction)')
    pos_or = (finite_or > 1).sum()
    neg_or = (finite_or < 1).sum()
    ax4.text(0.05, 0.95, f'Peak→B73: {pos_or}\nPeak→NAM: {neg_or}',
             transform=ax4.transAxes, va='top', fontsize=9)

    # E: Gene-level combined p-value histogram
    ax5 = fig.add_subplot(gs[1, 1])
    if gene_agg is not None:
        gene_p = gene_agg['combined_p'].dropna().values
        ax5.hist(gene_p, bins=40, color='#FF6F00', edgecolor='white', alpha=0.8, density=True)
        ax5.axhline(1.0, color='black', linestyle='--', alpha=0.5)
        ax5.set_xlabel("Gene-level combined p-value")
        ax5.set_ylabel('Density')
        ax5.set_title("E. Gene-level concordance test\n(Fisher's combined)")
        n_gene_sig = (gene_agg['combined_fdr'] < 0.05).sum()
        ax5.text(0.95, 0.95, f'FDR<0.05: {n_gene_sig}\nFDR<0.20: {(gene_agg["combined_fdr"]<0.20).sum()}',
                 transform=ax5.transAxes, va='top', ha='right', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # F: Gene concordance vs ASE magnitude
    ax6 = fig.add_subplot(gs[1, 2])
    if gene_agg is not None:
        ax6.scatter(gene_agg['mean_abs_ase'], gene_agg['mean_concordance'],
                    s=8, alpha=0.3, color='#00695C')
        ax6.axhline(0.5, color='red', linestyle='--', alpha=0.3)
        ax6.set_xlabel('Mean |log₂(B73/NAM)|')
        ax6.set_ylabel('Mean bQTL concordance')
        ax6.set_title('F. Gene ASE magnitude vs concordance')
        rho, p = stats.spearmanr(gene_agg['mean_abs_ase'], gene_agg['mean_concordance'])
        ax6.text(0.05, 0.95, f'$\\rho$ = {rho:.4f}\np = {p:.1e}',
                 transform=ax6.transAxes, va='top', fontsize=9)

    plt.suptitle('ASE Direction as Genotype Proxy: bQTL Concordance Analysis',
                 fontsize=13, fontweight='bold')
    plt.savefig(FIG_DIR / 'fig10_ase_direction_concordance.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(FIG_DIR / 'fig10_ase_direction_concordance.png', bbox_inches='tight', dpi=150)
    print(f"\nSaved: figures/fig10_ase_direction_concordance.pdf")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading data...")
    ase_df = pd.read_csv(ASE_HYBRID)
    gene_coords = parse_gff3()
    hybrid_peaks = load_hybrid_peaks()
    pan_by_chr = load_pan_cistrome()
    print(f"  ASE records: {len(ase_df):,}")
    print(f"  Genes in GFF3: {len(gene_coords):,}")
    print(f"  Hybrids with peaks: {len(hybrid_peaks)}")

    # Part 1: Identify variable ASE genes
    gene_dir, gene_hybrid = identify_variable_ase_genes(ase_df)

    # Part 2: Per-bQTL concordance test
    rdf = per_bqtl_concordance_test(gene_dir, gene_hybrid, gene_coords, hybrid_peaks, pan_by_chr)

    if rdf is None or len(rdf) == 0:
        print("\nNo testable bQTL found. Exiting.")
        return

    # Part 3: Gene-level aggregation
    gene_agg = gene_level_concordance(rdf, gene_dir)

    # Part 4: Compare with magnitude test
    compare_tests(rdf)

    # Part 5: Figures
    make_figures(rdf, gene_agg, gene_dir)

    # Final summary
    print(f"\n{'='*70}")
    print("SUMMARY: ASE DIRECTION CONCORDANCE ANALYSIS")
    print(f"{'='*70}")
    n_tested = len(rdf)
    n_nom = (rdf['fisher_p'] < 0.05).sum()
    n_fdr = (rdf['fdr'] < 0.05).sum()
    enrichment = (n_nom / n_tested) / 0.05 if n_tested > 0 else 0

    print(f"""
Approach: Use ASE direction (B73-high vs NAM-high) as a genotype proxy.
For each bQTL in a variable ASE gene's promoter, test whether peak presence
at that position predicts which hybrids are B73-high vs NAM-high (Fisher's exact).

Key numbers:
  Variable ASE genes (>=3 hybrids each direction): {gene_dir['variable'].sum():,}
  bQTL-gene pairs tested: {n_tested:,}
  Nominal p<0.05: {n_nom} ({100*n_nom/n_tested:.1f}%, expect 5%)
  Enrichment over null: {enrichment:.2f}x
  FDR<0.05: {n_fdr}
  Mean concordance: {rdf['concordance'].mean():.4f} (random=0.50)

Interpretation:
  - Enrichment >{1.5:.1f}x nominal hits → real signal from bQTL→ASE direction
  - Mean concordance >0.50 → systematic peak-direction association
  - Gene-level FDR hits → genes where promoter bQTL collectively explain ASE direction
""")


if __name__ == '__main__':
    main()
