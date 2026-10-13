"""
26_population_motif_occupancy.py — Layer 1: Population-level TF motif occupancy across hybrids.

For each gene, across all 23 WW hybrids:
  1. Check if an MOA-seq peak overlaps the gene's promoter region (B73 coordinates)
  2. If yes, extract the B73 reference sequence under that peak
  3. Scan with 405 DAP-seq PWMs (threshold 6.0)
  4. Build a gene × TF family × hybrid occupancy matrix
  5. Test for population-level enrichment: which TF families show variable
     occupancy across the hybrid panel (some hybrids bound, others not)?

This analysis is independent of bQTL — it captures the baseline TF binding
landscape across the population. Combined with Layer 2 (bQTL motif linkage),
it reveals whether a TF binding site that is disrupted by a bQTL was actually
occupied in the MOA-seq experiment.

Output:
  - population_motif_occupancy.csv:      Gene × TF family enrichment results
  - population_occupancy_matrix.csv:     Full gene × hybrid × family matrix
  - population_motif_summary.csv:        Per-family summary statistics
  - population_motif_occupancy.pdf:      Summary figure
"""

import sys, json, gzip, time, warnings, functools
print = functools.partial(print, flush=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_config import config


# ── Vectorized PWM scanning ──

BG = np.array([0.25, 0.25, 0.25, 0.25])
BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def seq_to_idx(seq):
    """Convert sequence string to int8 array (0-3 for ACGT, -1 for N)."""
    mapping = np.full(256, -1, dtype=np.int8)
    for base, val in BASE_MAP.items():
        mapping[ord(base)] = val
        mapping[ord(base.lower())] = val
    return mapping[np.frombuffer(seq.encode(), dtype=np.uint8)]


def load_pwms():
    """Load PWMs and compile into width-grouped numpy stacks for vectorized scanning."""
    db = json.load(open(config.pwm_db))
    families_list = []
    by_width = defaultdict(list)
    for m in db['motifs']:
        mat = np.array(m['pwm'], dtype=np.float64)
        mat = np.clip(mat, 1e-4, None)
        lo = np.log2(mat / BG[np.newaxis, :])
        rc = lo[::-1, ::-1].copy()
        by_width[len(mat)].append((lo, rc, m['tf_family'], m['tf_name']))
        families_list.append(m['tf_family'])

    compiled = {}
    for w, tuples in by_width.items():
        fwd_stack = np.array([t[0] for t in tuples])
        rc_stack = np.array([t[1] for t in tuples])
        families = [t[2] for t in tuples]
        tf_names = [t[3] for t in tuples]
        compiled[w] = (fwd_stack, rc_stack, families, tf_names)

    all_families = sorted(set(families_list))
    n_motifs = len(families_list)
    print(f"  PWMs: {n_motifs} motifs, {len(all_families)} families, "
          f"{len(compiled)} width groups")
    return compiled, all_families


def scan_families_fast(seq_idx, compiled, threshold=6.0):
    """Fast vectorized PWM scan using one-hot encoding + tensordot.

    Converts sequence to one-hot (L, 4), creates sliding windows via stride
    tricks, then scores all PWMs at all positions with a single tensordot per
    width group. ~10x faster than fancy-indexing approach.
    """
    L = len(seq_idx)
    hit_families = set()

    # One-hot encode sequence (N bases stay as all-zeros)
    onehot = np.zeros((L, 4), dtype=np.float32)
    for b in range(4):
        onehot[seq_idx == b, b] = 1.0

    # Mark N positions for validity check
    has_n = np.any(seq_idx < 0)

    for w, (fwd_stack, rc_stack, families, tf_names) in compiled.items():
        if L < w:
            continue
        n_pwms = fwd_stack.shape[0]
        n_windows = L - w + 1

        # Sliding windows of one-hot: (n_windows, w, 4)
        windows_oh = np.lib.stride_tricks.as_strided(
            onehot,
            shape=(n_windows, w, 4),
            strides=(onehot.strides[0], onehot.strides[0], onehot.strides[1])
        )

        # Score all PWMs at all windows: tensordot over (w, 4) axes
        # fwd_stack: (n_pwms, w, 4), windows_oh: (n_windows, w, 4)
        # Result: (n_windows, n_pwms)
        fwd_scores = np.tensordot(
            windows_oh, fwd_stack.astype(np.float32),
            axes=([1, 2], [1, 2])
        )
        rc_scores = np.tensordot(
            windows_oh, rc_stack.astype(np.float32),
            axes=([1, 2], [1, 2])
        )

        # Invalidate windows containing N bases
        if has_n:
            seq_windows = np.lib.stride_tricks.as_strided(
                seq_idx,
                shape=(n_windows, w),
                strides=(seq_idx.strides[0], seq_idx.strides[0])
            )
            invalid = np.any(seq_windows < 0, axis=1)
            fwd_scores[invalid] = -np.inf
            rc_scores[invalid] = -np.inf

        # Best score per PWM across all positions
        best = np.maximum(fwd_scores.max(axis=0), rc_scores.max(axis=0))
        for i in np.where(best >= threshold)[0]:
            hit_families.add(families[i])

    return hit_families


# ── GFF3 parsing ──

def parse_gff3():
    """Parse GFF3 for gene coordinates and TSS."""
    genes = {}
    with open(config.gff3) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'gene':
                continue
            chrom = parts[0]
            if not chrom.startswith('chr'):
                chrom = 'chr' + chrom
            if chrom not in config.valid_chroms:
                continue
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            gene_id = None
            for attr in parts[8].split(';'):
                if attr.startswith('ID=gene:'):
                    gene_id = attr.split(':')[1]
                    break
                elif attr.startswith('ID=Zm'):
                    gene_id = attr[3:]  # strip 'ID='
                    break
            if gene_id:
                tss = start if strand == '+' else end
                genes[gene_id] = {
                    'chr': chrom, 'start': start, 'end': end,
                    'strand': strand, 'tss': tss,
                }
    return genes


# ── Peak loading ──

def load_hybrid_peaks_b73(hybrid, condition='WW'):
    """Load peaks in B73 coordinates for one hybrid.
    Returns dict of chrom -> (starts_array, ends_array) as numpy arrays for fast search."""
    path = config.peaks_dir / f'{hybrid}.{condition}.q255.peaks.names.sorted.bed.gz'
    if not path.exists():
        return {}
    peaks_by_chrom = defaultdict(list)
    with gzip.open(path, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            chrom = parts[0]
            if not chrom.startswith('B73-chr'):
                continue
            chrom = chrom.replace('B73-', '')
            peaks_by_chrom[chrom].append((int(parts[1]), int(parts[2])))
    # Convert to sorted numpy arrays for binary search
    result = {}
    for chrom, peaks in peaks_by_chrom.items():
        peaks.sort()
        starts = np.array([p[0] for p in peaks], dtype=np.int64)
        ends = np.array([p[1] for p in peaks], dtype=np.int64)
        result[chrom] = (starts, ends)
    return result


def peaks_overlap_region(peak_arrays, region_start, region_end):
    """Fast overlap query using numpy searchsorted.
    peak_arrays: (starts, ends) numpy arrays, sorted by start.
    Returns list of (start, end) tuples that overlap [region_start, region_end)."""
    if peak_arrays is None:
        return []
    starts, ends = peak_arrays
    if len(starts) == 0:
        return []
    # Peaks that could overlap: start < region_end AND end > region_start
    # Use searchsorted to find candidate range
    right = np.searchsorted(starts, region_end, side='left')  # first peak with start >= region_end
    if right == 0:
        return []
    # Among peaks[0:right], find those with end > region_start
    candidate_ends = ends[:right]
    mask = candidate_ends > region_start
    indices = np.where(mask)[0]
    return [(int(starts[i]), int(ends[i])) for i in indices]


# ── Main ──

def main():
    import argparse
    from pyfaidx import Fasta

    parser = argparse.ArgumentParser(
        description='Layer 1: Population-level TF motif occupancy across hybrids')
    parser.add_argument('--promoter-size', type=int, default=None,
                        help='Promoter window upstream of TSS (default: from config)')
    parser.add_argument('--output-prefix', type=str, default=None)
    parser.add_argument('--condition', type=str, default='WW',
                        choices=['WW', 'DS'])
    args = parser.parse_args()

    promoter_size = args.promoter_size or config.promoter_size
    condition = args.condition

    print("=" * 70)
    print("  LAYER 1: POPULATION-LEVEL TF MOTIF OCCUPANCY")
    print("=" * 70)

    # Load PWMs (compiled for vectorized scanning)
    compiled, all_families = load_pwms()
    threshold = config.motif_threshold

    # Load genome
    print("Loading genome...")
    genome = Fasta(str(config.genome))

    # Parse genes
    print("Parsing GFF3...")
    genes = parse_gff3()
    print(f"  Genes: {len(genes):,}")

    # Discover hybrids
    peak_files = sorted(config.peaks_dir.glob(f'*.{condition}.q255.peaks.names.sorted.bed.gz'))
    hybrids = [f.name.split('.')[0] for f in peak_files]
    print(f"  Hybrids ({condition}): {len(hybrids)}")

    # Load all hybrid peaks (B73 coordinates only, as numpy arrays)
    print("Loading hybrid peaks (B73 coordinates)...")
    hybrid_peaks = {}
    for hyb in hybrids:
        hybrid_peaks[hyb] = load_hybrid_peaks_b73(hyb, condition)
        n_peaks = sum(len(v[0]) for v in hybrid_peaks[hyb].values())
        print(f"    {hyb}: {n_peaks:,} B73-coordinate peaks")

    # Strategy: scan each unique peak region ONCE on B73 reference, then map
    # motifs back to hybrids. Since all peaks are in B73 coordinates and we
    # scan the B73 sequence, the motif content of a given genomic interval is
    # the same regardless of which hybrid's peak covers it.
    print(f"\nScanning promoter regions ({promoter_size}bp upstream of TSS)...")
    print(f"  Motif threshold: {threshold}")

    # Build promoter intervals per chromosome
    chrom_genes = defaultdict(list)
    for gene_id, info in genes.items():
        chrom_genes[info['chr']].append((gene_id, info))

    occupancy_records = []
    gene_peak_counts = defaultdict(lambda: defaultdict(int))

    total_genes_with_peaks = 0
    total_scans = 0

    for chrom in sorted(config.valid_chroms):
        gene_list = chrom_genes.get(chrom, [])
        if not gene_list:
            continue

        chrom_key = chrom
        if chrom not in genome:
            chrom_key = chrom.replace('chr', '')
            if chrom_key not in genome:
                print(f"  Skipping {chrom}: not in genome")
                continue
        chrom_seq = str(genome[chrom_key]).upper()
        chrom_len = len(chrom_seq)

        t0 = time.time()
        print(f"  {chrom}: {len(gene_list)} genes, {chrom_len:,} bp ...")

        # Step 1: For each gene, collect union of peak intervals across ALL hybrids
        # and record which hybrids contribute peaks
        gene_data = []  # (gene_id, prom_start, prom_end, union_intervals, hybrid_peaks_map)
        for gene_id, info in gene_list:
            tss = info['tss']
            if info['strand'] == '+':
                prom_start = max(0, tss - promoter_size)
                prom_end = tss
            else:
                prom_start = tss
                prom_end = min(chrom_len, tss + promoter_size)

            # Collect all peak intervals from all hybrids that overlap this promoter
            all_intervals = []  # (start, end)
            hyb_peak_intervals = {}  # hybrid -> list of (start, end)
            for hyb in hybrids:
                hyb_chrom_peaks = hybrid_peaks[hyb].get(chrom)
                overlaps = peaks_overlap_region(hyb_chrom_peaks, prom_start, prom_end)
                if overlaps:
                    hyb_peak_intervals[hyb] = overlaps
                    all_intervals.extend(overlaps)

            if not all_intervals:
                continue

            # Merge overlapping intervals into union regions
            all_intervals.sort()
            merged = [all_intervals[0]]
            for s, e in all_intervals[1:]:
                if s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))

            gene_data.append((gene_id, prom_start, prom_end, merged, hyb_peak_intervals))

        # Step 2: Collect all unique merged intervals across all genes on this chrom
        # and scan each ONCE
        interval_motifs = {}  # (start, end) -> set of families
        all_unique = set()
        for _, _, _, merged, _ in gene_data:
            for iv in merged:
                all_unique.add(iv)

        t_collect = time.time() - t0
        print(f"    -> {len(gene_data)} genes with peaks, {len(all_unique)} unique intervals "
              f"(collect: {t_collect:.1f}s)")
        t_scan = time.time()
        sorted_intervals = sorted(all_unique)
        widths = [pe-ps for ps,pe in sorted_intervals]
        print(f"    -> Scanning: median width {np.median(widths):.0f}bp, "
              f"max {max(widths)}bp, total {sum(widths):,}bp")
        n_scanned = 0
        for ps, pe in sorted_intervals:
            ps_c = max(0, ps)
            pe_c = min(chrom_len, pe)
            if pe_c - ps_c < 6:
                interval_motifs[(ps, pe)] = set()
                continue
            peak_idx = seq_to_idx(chrom_seq[ps_c:pe_c])
            fams = scan_families_fast(peak_idx, compiled, threshold)
            interval_motifs[(ps, pe)] = fams
            total_scans += 1
            n_scanned += 1
            if n_scanned % 2000 == 0:
                rate = n_scanned / max(time.time() - t_scan, 0.001)
                print(f"      {n_scanned}/{len(all_unique)} scanned ({rate:.0f}/s) ...")
        t_scan_elapsed = time.time() - t_scan
        print(f"    -> Scanning done: {t_scan_elapsed:.1f}s")

        # Step 3: For each gene × hybrid, map peak overlaps to scanned motifs
        for gene_id, prom_start, prom_end, merged, hyb_peak_intervals in gene_data:
            total_genes_with_peaks += 1

            # Pre-collect all motif families found in any merged interval
            # for this gene's promoter
            all_gene_families = set()
            for iv in merged:
                all_gene_families.update(interval_motifs.get(iv, set()))

            for hyb, hyb_overlaps in hyb_peak_intervals.items():
                gene_peak_counts[gene_id][hyb] = len(hyb_overlaps)

                # Which families are in the specific peaks this hybrid has?
                hyb_families = set()
                for ps, pe in hyb_overlaps:
                    # Find which merged interval(s) this peak falls into
                    for ms, me in merged:
                        if ps < me and pe > ms:
                            hyb_families.update(interval_motifs.get((ms, me), set()))
                for fam in hyb_families:
                    occupancy_records.append((gene_id, hyb, fam))

        elapsed = time.time() - t0
        print(f"    -> {len(gene_data)} genes with peaks, {len(all_unique)} unique intervals, "
              f"{total_scans} total scans, {elapsed:.1f}s")

    print(f"\nGenes with at least one peak in any hybrid: {total_genes_with_peaks:,}")
    print(f"Total peak scans: {total_scans:,}")
    print(f"Occupancy records: {len(occupancy_records):,}")

    if not occupancy_records:
        print("ERROR: No occupancy records — check peak paths and genome")
        sys.exit(1)

    # ── Build occupancy matrix ──
    print("\nBuilding occupancy matrix...")
    occ_df = pd.DataFrame(occupancy_records, columns=['gene_id', 'hybrid', 'tf_family'])
    occ_df['occupied'] = 1

    # Pivot: gene × (hybrid, family) -> 0/1
    matrix = occ_df.pivot_table(
        index='gene_id', columns=['hybrid', 'tf_family'],
        values='occupied', fill_value=0, aggfunc='max'
    )

    # Gene-level peak count matrix
    peak_count_df = pd.DataFrame([
        {'gene_id': g, 'hybrid': h, 'n_peaks': c}
        for g, hc in gene_peak_counts.items()
        for h, c in hc.items()
    ])

    # ── Per-gene × family: test for variable occupancy ──
    print("Testing for variable occupancy per gene × TF family...")

    # For each gene and each TF family: count how many hybrids have that family
    # in a peak overlapping the promoter
    gene_family_hyb = defaultdict(lambda: defaultdict(set))
    for gene_id, hyb, fam in occupancy_records:
        gene_family_hyb[gene_id][fam].add(hyb)

    # Also need to know which hybrids have ANY peak at each gene
    gene_bound_hybrids = defaultdict(set)
    for gene_id, hyb_counts in gene_peak_counts.items():
        for hyb in hyb_counts:
            gene_bound_hybrids[gene_id].add(hyb)

    results = []
    n_hybrids = len(hybrids)

    for gene_id in sorted(gene_bound_hybrids.keys()):
        bound_hybs = gene_bound_hybrids[gene_id]
        n_bound = len(bound_hybs)
        info = genes[gene_id]

        for fam in all_families:
            hybs_with_motif = gene_family_hyb[gene_id].get(fam, set())
            n_with_motif = len(hybs_with_motif)

            # Fraction of bound hybrids that have this motif
            frac_bound = n_with_motif / n_bound if n_bound > 0 else 0

            # Fraction of total hybrids
            frac_total = n_with_motif / n_hybrids

            # Variability: motif present in some but not all bound hybrids
            # Shannon entropy of presence/absence among bound hybrids
            if n_bound > 0 and 0 < n_with_motif < n_bound:
                p = n_with_motif / n_bound
                entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
                variable = True
            else:
                entropy = 0.0
                variable = False

            results.append({
                'gene_id': gene_id,
                'chr': info['chr'],
                'tss': info['tss'],
                'strand': info['strand'],
                'tf_family': fam,
                'n_hybrids_bound': n_bound,
                'n_hybrids_with_motif': n_with_motif,
                'frac_of_bound': round(frac_bound, 4),
                'frac_of_total': round(frac_total, 4),
                'entropy': round(entropy, 4),
                'variable': variable,
                'hybrids_with_motif': '|'.join(sorted(hybs_with_motif)) if hybs_with_motif else '',
            })

    results_df = pd.DataFrame(results)
    print(f"  Gene × family entries: {len(results_df):,}")

    # ── Per-family summary ──
    print("\nPer-family summary:")
    summary_rows = []
    for fam in all_families:
        fam_data = results_df[results_df['tf_family'] == fam]
        n_genes_with_motif = (fam_data['n_hybrids_with_motif'] > 0).sum()
        n_genes_variable = fam_data['variable'].sum()
        mean_frac = fam_data.loc[fam_data['n_hybrids_bound'] > 0, 'frac_of_bound'].mean()
        mean_entropy = fam_data.loc[fam_data['variable'], 'entropy'].mean() if n_genes_variable > 0 else 0

        # Population-level: how many genes have this family in >50% of bound hybrids?
        n_conserved = ((fam_data['frac_of_bound'] >= 0.5) & (fam_data['n_hybrids_bound'] >= 5)).sum()
        # Variable: present in 10-90% of bound hybrids (with at least 5 bound)
        n_variable = (
            (fam_data['frac_of_bound'] >= 0.1) &
            (fam_data['frac_of_bound'] <= 0.9) &
            (fam_data['n_hybrids_bound'] >= 5)
        ).sum()

        summary_rows.append({
            'tf_family': fam,
            'n_genes_with_motif': n_genes_with_motif,
            'n_genes_variable': int(n_genes_variable),
            'n_genes_conserved': int(n_conserved),
            'n_genes_variable_strict': int(n_variable),
            'mean_frac_of_bound': round(mean_frac, 4) if not np.isnan(mean_frac) else 0,
            'mean_entropy_variable': round(mean_entropy, 4),
        })
        if n_genes_with_motif > 0:
            print(f"    {fam:12s}: {n_genes_with_motif:5d} genes, "
                  f"{n_genes_variable:5d} variable, "
                  f"{n_conserved:5d} conserved (>50%), "
                  f"mean occ {mean_frac:.3f}")

    summary_df = pd.DataFrame(summary_rows)

    # ── Save outputs ──
    out_dir = config.results_dir
    prefix = args.output_prefix or 'population_motif'

    results_path = out_dir / f'{prefix}_occupancy.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")

    summary_path = out_dir / f'{prefix}_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # Save peak count matrix
    peak_matrix_path = out_dir / f'{prefix}_peak_counts.csv'
    if not peak_count_df.empty:
        peak_pivot = peak_count_df.pivot_table(
            index='gene_id', columns='hybrid',
            values='n_peaks', fill_value=0
        )
        peak_pivot.to_csv(peak_matrix_path)
        print(f"Saved: {peak_matrix_path}")

    # ── Cross-reference with causal bQTL ──
    print("\nCross-referencing with causal bQTL...")
    causal_path = config.results_dir / 'causal_bqtl_728.csv'
    if causal_path.exists():
        causal = pd.read_csv(causal_path)
        causal_genes = set(causal['gene_id'].dropna().unique())

        # For causal bQTL genes: what is the motif occupancy profile?
        causal_occ = results_df[results_df['gene_id'].isin(causal_genes)].copy()
        n_causal_genes = causal_occ['gene_id'].nunique()
        print(f"  Causal bQTL genes with occupancy data: {n_causal_genes}")

        # Compare variable vs conserved motifs at causal vs non-causal genes
        results_df['is_causal'] = results_df['gene_id'].isin(causal_genes)
        for fam in all_families:
            fam_data = results_df[
                (results_df['tf_family'] == fam) &
                (results_df['n_hybrids_bound'] >= 5)
            ]
            if len(fam_data) < 10:
                continue
            causal_frac = fam_data.loc[fam_data['is_causal'], 'frac_of_bound']
            noncausal_frac = fam_data.loc[~fam_data['is_causal'], 'frac_of_bound']
            if len(causal_frac) >= 5 and len(noncausal_frac) >= 5:
                u_stat, p_val = stats.mannwhitneyu(
                    causal_frac, noncausal_frac, alternative='two-sided')
                if p_val < 0.05:
                    diff = causal_frac.mean() - noncausal_frac.mean()
                    print(f"    {fam:12s}: causal={causal_frac.mean():.3f} vs "
                          f"non-causal={noncausal_frac.mean():.3f} "
                          f"(diff={diff:+.3f}, p={p_val:.2e})")

        causal_occ_path = out_dir / f'{prefix}_causal_crossref.csv'
        causal_occ.to_csv(causal_occ_path, index=False)
        print(f"  Saved: {causal_occ_path}")
    else:
        print(f"  Causal bQTL file not found: {causal_path}")

    # ── Summary figure ──
    print("\nGenerating summary figure...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Population-Level TF Motif Occupancy ({condition})', fontsize=14)

        # A: Genes per family
        ax = axes[0, 0]
        top = summary_df.nlargest(20, 'n_genes_with_motif')
        ax.barh(range(len(top)), top['n_genes_with_motif'], color='steelblue')
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top['tf_family'])
        ax.set_xlabel('Genes with motif')
        ax.set_title('A. TF families by gene coverage')
        ax.invert_yaxis()

        # B: Variable vs conserved
        ax = axes[0, 1]
        top_var = summary_df.nlargest(20, 'n_genes_variable')
        ax.barh(range(len(top_var)), top_var['n_genes_variable'], color='coral', label='Variable')
        ax.barh(range(len(top_var)), top_var['n_genes_conserved'], color='teal',
                alpha=0.6, label='Conserved (>50%)')
        ax.set_yticks(range(len(top_var)))
        ax.set_yticklabels(top_var['tf_family'])
        ax.set_xlabel('Number of genes')
        ax.set_title('B. Variable vs conserved occupancy')
        ax.legend(fontsize=8)
        ax.invert_yaxis()

        # C: Distribution of peak counts per gene across hybrids
        ax = axes[1, 0]
        n_bound_per_gene = results_df.groupby('gene_id')['n_hybrids_bound'].first()
        ax.hist(n_bound_per_gene, bins=range(0, n_hybrids + 2), color='gray', edgecolor='black')
        ax.set_xlabel('Number of hybrids with peak at promoter')
        ax.set_ylabel('Number of genes')
        ax.set_title('C. Peak coverage across hybrids')
        ax.axvline(n_bound_per_gene.median(), color='red', linestyle='--',
                   label=f'Median: {n_bound_per_gene.median():.0f}')
        ax.legend(fontsize=8)

        # D: Mean motif occupancy fraction by family
        ax = axes[1, 1]
        top_occ = summary_df[summary_df['mean_frac_of_bound'] > 0].nlargest(20, 'mean_frac_of_bound')
        ax.barh(range(len(top_occ)), top_occ['mean_frac_of_bound'], color='purple')
        ax.set_yticks(range(len(top_occ)))
        ax.set_yticklabels(top_occ['tf_family'])
        ax.set_xlabel('Mean fraction of bound hybrids with motif')
        ax.set_title('D. Mean motif occupancy rate')
        ax.invert_yaxis()

        plt.tight_layout()
        fig_path = config.figures_dir / f'{prefix}_occupancy.pdf'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")
    except ImportError:
        print("  matplotlib not available — skipping figure")

    print("\n" + "=" * 70)
    print(f"  LAYER 1 COMPLETE")
    print(f"  Genes analyzed: {total_genes_with_peaks:,}")
    print(f"  Hybrids: {len(hybrids)}")
    print(f"  TF families: {len(all_families)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
