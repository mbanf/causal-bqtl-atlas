#!/usr/bin/env python3
"""
05b_expanded_distance_test.py — Genotype→ASE test at expanded distances.

Extends step 05 from 2kb upstream to 10kb upstream + 10kb downstream (nearest gene),
to measure how causality decays with distance and whether the 2kb cutoff is too strict.

Also tests bidirectional mapping (upstream AND downstream of TSS) since bQTL in
gene bodies or downstream enhancers could also affect expression.

Output:
  results/expanded_distance_test.csv — all tested pairs with distance
  figures/distance_decay_curve.pdf — causality rate vs distance
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
from collections import defaultdict
import bisect
import warnings
warnings.filterwarnings('ignore')

from pipeline_config import config

OUTDIR = config.results_dir
FIG_DIR = config.figures_dir
BQTL = config.bqtl
ASE_FILE = config.ase
GENOTYPE_FILE = config.genotypes
GFF3 = config.gff3

MAX_DISTANCE = 10000  # 10kb in each direction
VALID_CHROMS = {f'chr{i}' for i in range(1, 11)}


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


def load_genotype_matrix():
    """Load genotype matrix and return {(chr, pos): {hybrid: 0/1}}."""
    ase_df = pd.read_csv(str(ASE_FILE), usecols=['hybrid'], dtype=str)
    hybrids = sorted(ase_df['hybrid'].unique())
    hyb_founder = {h.replace('B73x', '').upper(): h for h in hybrids}

    df = pd.read_csv(str(GENOTYPE_FILE), sep='\t', dtype=str)
    founder_cols = [c for c in df.columns if c not in ('chr', 'pos', 'ref')]

    founder_to_hybrid = {}
    for f in founder_cols:
        match = hyb_founder.get(f.upper())
        if match:
            founder_to_hybrid[f] = match

    genotype_matrix = {}
    for _, row in df.iterrows():
        chrom = row['chr']
        pos = int(row['pos'])
        geno = {}
        for f_col in founder_cols:
            hybrid = founder_to_hybrid.get(f_col)
            if hybrid is None:
                continue
            val = str(row[f_col])
            if val in ('./.', 'nan') or pd.isna(row[f_col]):
                continue
            geno[hybrid] = 0 if val == '0|0' else 1
        if geno:
            genotype_matrix[(chrom, pos)] = geno

    return genotype_matrix, hybrids


def map_bqtl_to_genes_expanded(bqtl_df, genes, max_dist=10000):
    """Map each bQTL to ALL genes within max_dist of TSS (any direction)."""
    genes_by_chr = defaultdict(list)
    for gid, info in genes.items():
        genes_by_chr[info['chr']].append((info['tss'], gid, info['strand'],
                                          info['start'], info['end']))
    for c in genes_by_chr:
        genes_by_chr[c].sort()

    mappings = []
    for _, row in bqtl_df.iterrows():
        chrom, pos = row['chr'], row['pos']
        gene_list = genes_by_chr.get(chrom, [])
        if not gene_list:
            continue
        tss_list = [g[0] for g in gene_list]
        idx = bisect.bisect_left(tss_list, pos)

        for i in range(max(0, idx - 20), min(len(gene_list), idx + 20)):
            tss, gid, strand, gstart, gend = gene_list[i]
            abs_dist = abs(pos - tss)
            if abs_dist > max_dist:
                continue

            # Classify position relative to gene
            if strand == '+':
                signed_dist = tss - pos  # positive = upstream
            else:
                signed_dist = pos - tss  # positive = upstream

            # Is bQTL in gene body?
            in_gene_body = gstart <= pos <= gend

            mappings.append({
                'chr': chrom, 'pos': pos, 'gene_id': gid,
                'abs_dist': abs_dist, 'signed_dist': signed_dist,
                'strand': strand, 'in_gene_body': in_gene_body,
                'position_type': 'upstream' if signed_dist > 0 else
                                 ('downstream' if signed_dist < 0 else 'at_TSS'),
            })

    return pd.DataFrame(mappings)


def run_test(genotype_matrix, pairs_df, ase_lookup, hybrids):
    """Run Mann-Whitney test for each bQTL-gene pair."""
    results = []
    for _, row in pairs_df.iterrows():
        chrom, pos, gene_id = row['chr'], row['pos'], row['gene_id']
        geno = genotype_matrix.get((chrom, pos))
        if geno is None:
            continue

        ase_var, ase_ref = [], []
        for hybrid in hybrids:
            g = geno.get(hybrid)
            if g is None:
                continue
            ase_val = ase_lookup.get((gene_id, hybrid))
            if ase_val is None or np.isnan(ase_val):
                continue
            if g == 1:
                ase_var.append(ase_val)
            else:
                ase_ref.append(ase_val)

        if len(ase_var) >= 3 and len(ase_ref) >= 3:
            mw_stat, mw_p = stats.mannwhitneyu(ase_var, ase_ref,
                                                alternative='two-sided')
            mean_v, mean_r = np.mean(ase_var), np.mean(ase_ref)
            sd_v = np.std(ase_var, ddof=1)
            sd_r = np.std(ase_ref, ddof=1)
            pooled_sd = np.sqrt(((len(ase_var)-1)*sd_v**2 +
                                 (len(ase_ref)-1)*sd_r**2) /
                                (len(ase_var)+len(ase_ref)-2))
            cohens_d = (mean_v - mean_r) / pooled_sd if pooled_sd > 0 else 0

            results.append({
                'chr': chrom, 'pos': pos, 'gene_id': gene_id,
                'abs_dist': row['abs_dist'],
                'signed_dist': row['signed_dist'],
                'position_type': row['position_type'],
                'in_gene_body': row['in_gene_body'],
                'n_variant': len(ase_var), 'n_reference': len(ase_ref),
                'mean_ase_variant': mean_v, 'mean_ase_reference': mean_r,
                'cohens_d': cohens_d, 'mw_p': mw_p,
            })

    rdf = pd.DataFrame(results)
    if len(rdf) > 0:
        _, rdf['fdr_mw'], _, _ = multipletests(rdf['mw_p'], method='fdr_bh')
    return rdf


def analyze_and_plot(rdf):
    """Analyze distance decay and generate figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print(f"\n{'='*70}")
    print("DISTANCE DECAY ANALYSIS")
    print(f"{'='*70}")
    print(f"Total pairs tested: {len(rdf):,}")
    print(f"FDR < 0.05: {(rdf['fdr_mw'] < 0.05).sum():,} ({100*(rdf['fdr_mw']<0.05).mean():.1f}%)")

    # Distance bins
    dist_bins = [0, 200, 500, 1000, 2000, 3000, 5000, 7500, 10000]
    labels = ['0-200', '200-500', '500-1k', '1-2k', '2-3k', '3-5k', '5-7.5k', '7.5-10k']
    rdf['dist_bin'] = pd.cut(rdf['abs_dist'], bins=dist_bins, labels=labels, right=False)

    print(f"\n--- Causality rate by distance ---")
    print(f"{'Distance':>12s} | {'N':>6s} | {'FDR sig':>7s} | {'Rate':>6s} | {'Nom sig':>7s} | {'|d| med':>7s}")
    bin_data = []
    for label in labels:
        subset = rdf[rdf['dist_bin'] == label]
        if len(subset) == 0:
            continue
        n_sig = (subset['fdr_mw'] < 0.05).sum()
        n_nom = (subset['mw_p'] < 0.05).sum()
        rate = n_sig / len(subset) if len(subset) > 0 else 0
        nom_rate = n_nom / len(subset) if len(subset) > 0 else 0
        med_d = subset['cohens_d'].abs().median()
        print(f"{label:>12s} | {len(subset):6,} | {n_sig:7,} | {100*rate:5.1f}% | {n_nom:7,} | {med_d:7.3f}")
        bin_data.append({
            'label': label, 'n': len(subset), 'fdr_sig': n_sig,
            'fdr_rate': rate, 'nom_rate': nom_rate, 'med_d': med_d,
        })

    # Upstream vs downstream
    print(f"\n--- Upstream vs downstream ---")
    for ptype in ['upstream', 'downstream']:
        subset = rdf[rdf['position_type'] == ptype]
        if len(subset) == 0:
            continue
        n_sig = (subset['fdr_mw'] < 0.05).sum()
        print(f"  {ptype:>12s}: n={len(subset):,}, FDR sig={n_sig:,} ({100*n_sig/len(subset):.1f}%)")

    # In gene body vs not
    print(f"\n--- In gene body vs intergenic ---")
    for in_body in [True, False]:
        subset = rdf[rdf['in_gene_body'] == in_body]
        if len(subset) == 0:
            continue
        n_sig = (subset['fdr_mw'] < 0.05).sum()
        label = "Gene body" if in_body else "Intergenic"
        print(f"  {label:>12s}: n={len(subset):,}, FDR sig={n_sig:,} ({100*n_sig/len(subset):.1f}%)")

    # New causal bQTL beyond 2kb
    beyond_2kb = rdf[rdf['abs_dist'] > 2000]
    beyond_2kb_sig = beyond_2kb[beyond_2kb['fdr_mw'] < 0.05]
    print(f"\n--- New discoveries beyond 2kb ---")
    print(f"  Pairs tested beyond 2kb: {len(beyond_2kb):,}")
    print(f"  FDR significant: {len(beyond_2kb_sig):,} ({100*len(beyond_2kb_sig)/max(len(beyond_2kb),1):.1f}%)")
    if len(beyond_2kb_sig) > 0:
        print(f"  Unique genes: {beyond_2kb_sig['gene_id'].nunique()}")
        print(f"  Unique bQTL positions: {beyond_2kb_sig[['chr','pos']].drop_duplicates().shape[0]}")
        # How many are NEW (gene not already found within 2kb)?
        within_2kb_sig = rdf[(rdf['abs_dist'] <= 2000) & (rdf['fdr_mw'] < 0.05)]
        known_genes = set(within_2kb_sig['gene_id'])
        new_genes = set(beyond_2kb_sig['gene_id']) - known_genes
        print(f"  NEW genes (not found within 2kb): {len(new_genes)}")

    # Single-bQTL gene analysis at expanded distance
    print(f"\n--- Causal bQTL yield at different cutoffs ---")
    for max_d in [2000, 3000, 5000, 7500, 10000]:
        subset = rdf[rdf['abs_dist'] <= max_d]
        sig = subset[subset['fdr_mw'] < 0.05]
        bqtl_per_gene = sig.groupby('gene_id').size()
        single = bqtl_per_gene[bqtl_per_gene == 1].index
        multi = bqtl_per_gene[bqtl_per_gene > 1].index
        print(f"  ≤{max_d/1000:.0f}kb: {len(sig):,} sig pairs, "
              f"{sig['gene_id'].nunique():,} genes, "
              f"{len(single):,} single-bQTL, {len(multi):,} multi-bQTL")

    # === FIGURES ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    bin_df = pd.DataFrame(bin_data)

    # A: FDR sig rate by distance
    ax = axes[0, 0]
    ax.bar(range(len(bin_df)), 100 * bin_df['fdr_rate'], color='#1976D2',
           edgecolor='white')
    ax.set_xticks(range(len(bin_df)))
    ax.set_xticklabels(bin_df['label'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('FDR-significant pairs (%)')
    ax.set_title('A. Causality rate decays with distance')
    ax.axhline(5, color='red', linestyle='--', alpha=0.5, label='FDR=5% baseline')
    ax.legend(fontsize=8)
    for i, row in bin_df.iterrows():
        ax.text(i, 100*row['fdr_rate'] + 0.3, f"n={row['n']:,}",
                ha='center', fontsize=6)

    # B: Median |Cohen's d| by distance
    ax = axes[0, 1]
    ax.plot(range(len(bin_df)), bin_df['med_d'], 'o-', color='#FF5722',
            markersize=8, linewidth=2)
    ax.set_xticks(range(len(bin_df)))
    ax.set_xticklabels(bin_df['label'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Median |Cohen's d|")
    ax.set_title("B. Effect size decays with distance")

    # C: Upstream vs downstream decay
    ax = axes[0, 2]
    for ptype, color, marker in [('upstream', '#1976D2', 'o'),
                                  ('downstream', '#E91E63', 's')]:
        sub = rdf[rdf['position_type'] == ptype]
        sub_binned = sub.copy()
        sub_binned['dist_bin'] = pd.cut(sub_binned['abs_dist'], bins=dist_bins,
                                         labels=labels, right=False)
        rates = []
        for label in labels:
            s = sub_binned[sub_binned['dist_bin'] == label]
            if len(s) > 50:
                rates.append(100 * (s['fdr_mw'] < 0.05).mean())
            else:
                rates.append(np.nan)
        ax.plot(range(len(labels)), rates, f'{marker}-', color=color,
                label=ptype, markersize=6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('FDR-significant (%)')
    ax.set_title('C. Upstream vs downstream')
    ax.legend(fontsize=8)

    # D: P-value histogram
    ax = axes[1, 0]
    ax.hist(rdf['mw_p'], bins=50, color='#1976D2', edgecolor='white', alpha=0.8)
    ax.axhline(len(rdf) / 50, color='red', linestyle='--', alpha=0.7)
    n_sig = (rdf['fdr_mw'] < 0.05).sum()
    ax.set_xlabel('Mann-Whitney p-value')
    ax.set_ylabel('Count')
    ax.set_title(f'D. P-value distribution (all distances)\n'
                 f'{n_sig:,} FDR<0.05 of {len(rdf):,}')

    # E: Cumulative causal bQTL vs distance
    ax = axes[1, 1]
    sig_pairs = rdf[rdf['fdr_mw'] < 0.05].copy()
    cum_dists = sorted(range(0, 10001, 100))
    cum_genes = []
    cum_single = []
    for d in cum_dists:
        sub = sig_pairs[sig_pairs['abs_dist'] <= d]
        cum_genes.append(sub['gene_id'].nunique())
        bpg = sub.groupby('gene_id').size()
        cum_single.append((bpg == 1).sum())
    ax.plot(cum_dists, cum_genes, '-', color='#1976D2', linewidth=2,
            label='All sig genes')
    ax.plot(cum_dists, cum_single, '-', color='#4CAF50', linewidth=2,
            label='Single-bQTL genes')
    ax.axvline(3000, color='red', linestyle='--', alpha=0.5, label='3kb threshold')
    ax.set_xlabel('Max distance to TSS (bp)')
    ax.set_ylabel('Cumulative genes')
    ax.set_title('E. Cumulative causal genes vs distance')
    ax.legend(fontsize=8)

    # F: Scatter: distance vs effect size (sig only)
    ax = axes[1, 2]
    if len(sig_pairs) > 0:
        ax.scatter(sig_pairs['abs_dist'], sig_pairs['cohens_d'].abs().clip(0, 5),
                   s=8, alpha=0.3, color='#D32F2F', rasterized=True)
        # Binned means
        for i in range(len(dist_bins) - 1):
            mask = ((sig_pairs['abs_dist'] >= dist_bins[i]) &
                    (sig_pairs['abs_dist'] < dist_bins[i+1]))
            if mask.sum() > 5:
                mean_d = sig_pairs.loc[mask, 'cohens_d'].abs().mean()
                mid = (dist_bins[i] + dist_bins[i+1]) / 2
                ax.plot(mid, mean_d, 'ko', markersize=8)
    ax.axvline(3000, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Distance to TSS (bp)')
    ax.set_ylabel("|Cohen's d|")
    ax.set_title('F. Effect size vs distance (sig pairs)')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'distance_decay_curve.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(FIG_DIR / 'distance_decay_curve.png', bbox_inches='tight', dpi=150)
    print(f"\nSaved: figures/distance_decay_curve.pdf")
    plt.close()


def main():
    print("=" * 70)
    print("EXPANDED DISTANCE GENOTYPE→ASE TEST (0–10kb)")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    bqtl_df = pd.read_csv(BQTL)
    print(f"  bQTL: {len(bqtl_df):,}")

    genotype_matrix, hybrids = load_genotype_matrix()
    print(f"  Genotype matrix: {len(genotype_matrix):,} positions")
    print(f"  Hybrids: {len(hybrids)}")

    genes = parse_gff3()
    print(f"  Genes: {len(genes):,}")

    ase_df = pd.read_csv(ASE_FILE)
    ase_lookup = {}
    for _, row in ase_df.iterrows():
        ase_lookup[(row['gene_id'], row['hybrid'])] = row['log2_ratio']
    print(f"  ASE lookup: {len(ase_lookup):,} gene-hybrid pairs")

    # Map bQTL to genes at expanded distance
    print(f"\n2. Mapping bQTL to genes (≤{MAX_DISTANCE/1000:.0f}kb)...")
    bqtl_with_geno = bqtl_df[
        bqtl_df.apply(lambda r: (r['chr'], r['pos']) in genotype_matrix, axis=1)
    ].copy()
    print(f"  bQTL with genotype data: {len(bqtl_with_geno):,}")

    pairs = map_bqtl_to_genes_expanded(bqtl_with_geno, genes, MAX_DISTANCE)
    print(f"  bQTL-gene pairs: {len(pairs):,}")
    print(f"  Unique genes: {pairs['gene_id'].nunique():,}")
    print(f"  Position types: {pairs['position_type'].value_counts().to_dict()}")

    # Run tests
    print(f"\n3. Running genotype→ASE tests...")
    rdf = run_test(genotype_matrix, pairs, ase_lookup, hybrids)
    print(f"  Testable pairs (≥3 per group): {len(rdf):,}")

    if len(rdf) == 0:
        print("No testable pairs found.")
        return

    # Save
    rdf.to_csv(OUTDIR / 'expanded_distance_test.csv', index=False)
    print(f"  Saved: results/expanded_distance_test.csv")

    # Analyze and plot
    analyze_and_plot(rdf)


if __name__ == '__main__':
    main()
