#!/usr/bin/env python3
"""
MOA-seq bound region motif analysis.

Instead of scanning entire 2kb promoters, this analyzes motif content within
experimentally-defined binding regions:
  1. ±250bp around each bQTL (proxy for MOA-seq peak, median cluster width = 254bp)
  2. Marand 2021 scATAC-seq ACR peaks (leaf tissue, cell-type resolved)

Key questions:
  - What is the motif composition within MOA-seq bound regions?
  - How many motifs exist per bound region? Which families?
  - What is the redundancy per family within bound regions?
  - Which motifs at the variant position are disrupted vs created by bQTL?
  - How does motif density in bound regions compare to random promoter sequence?
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
GFF3 = DATA / "raw" / "Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3"
GENOME = DATA / "raw" / "B73_NAM5.fa"
PWM_DB = DATA / "processed" / "maize_tf_pwm_database.json"
BQTL = DATA / "processed" / "bqtl_snp_ww.csv"
ASE = DATA / "processed" / "engelhorn_ase_ww_gene_summary.csv"
EXPR = DATA / "processed" / "engelhorn_ww_vs_ds_expression.tsv"
ENTAP = DATA / "processed" / "entap_annotations.tsv"
SAVADEL = DATA / "raw" / "savadel2021_mf_bc7_nam5.bed"
MARAND = DATA / "external" / "marand2021_cell_type_peaks.bed"
OUTDIR = BASE / "results"

# Parameters
BOUND_REGION_BP = 250     # ±250bp around bQTL as MOA peak proxy
VARIANT_WINDOW = 28       # ±28bp for variant-overlap (disrupted motif)
PWM_THRESHOLD = 6.0
BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
BG = np.array([0.2642, 0.2358, 0.2358, 0.2642])


# ================================================================
# PWM SCORING (reused from 44_full_quantitative_modulation.py)
# ================================================================

def compile_pwm(pwm_list):
    """Precompile log-odds matrices grouped by width."""
    by_width = defaultdict(list)
    for entry in pwm_list:
        mat = np.array(entry['pwm'], dtype=np.float64)
        mat = np.clip(mat, 1e-4, None)
        lo = np.log2(mat / BG[np.newaxis, :])
        rc = lo[::-1, ::-1].copy()
        by_width[len(mat)].append((lo, rc, entry.get('family', '?')))
    compiled = {}
    for w, triples in by_width.items():
        fwd_stack = np.array([t[0] for t in triples])
        rc_stack = np.array([t[1] for t in triples])
        families = [t[2] for t in triples]
        compiled[w] = (fwd_stack, rc_stack, families)
    return compiled


def scan_region_detailed(seq_idx, compiled, threshold=6.0):
    """Scan sequence for all TF motif hits. Returns list of (position, family, score, strand).

    seq_idx: int8 array of base indices (0-3, -1 for N)
    """
    L = len(seq_idx)
    hits = []

    for w, (fwd_stack, rc_stack, families) in compiled.items():
        if L < w:
            continue
        n_pwms = fwd_stack.shape[0]
        n_windows = L - w + 1

        windows = np.lib.stride_tricks.as_strided(
            seq_idx,
            shape=(n_windows, w),
            strides=(seq_idx.strides[0], seq_idx.strides[0])
        ).copy()

        valid = np.all(windows >= 0, axis=1)
        if not np.any(valid):
            continue

        valid_idx = np.where(valid)[0]
        valid_windows = windows[valid]
        n_valid = valid_windows.shape[0]
        pos_idx = np.arange(w)

        # Score all PWMs at all valid windows
        fwd_scores = np.sum(
            fwd_stack[
                np.arange(n_pwms)[:, np.newaxis, np.newaxis],
                pos_idx[np.newaxis, np.newaxis, :],
                valid_windows[np.newaxis, :, :]
            ], axis=2
        )  # (n_pwms, n_valid)

        rc_scores = np.sum(
            rc_stack[
                np.arange(n_pwms)[:, np.newaxis, np.newaxis],
                pos_idx[np.newaxis, np.newaxis, :],
                valid_windows[np.newaxis, :, :]
            ], axis=2
        )

        for p in range(n_pwms):
            for v_i in range(n_valid):
                fwd_s = fwd_scores[p, v_i]
                rc_s = rc_scores[p, v_i]
                best_score = max(fwd_s, rc_s)
                if best_score >= threshold:
                    strand = '+' if fwd_s >= rc_s else '-'
                    hits.append({
                        'pos_in_seq': int(valid_idx[v_i]),
                        'width': w,
                        'family': families[p],
                        'score': float(best_score),
                        'strand': strand,
                    })
    return hits


def seq_to_idx(seq):
    """Convert sequence string to int8 array."""
    idx = np.full(len(seq), -1, dtype=np.int8)
    for base, val in BASE_MAP.items():
        mask = np.array([c == base for c in seq.upper()])
        idx[mask] = val
    return idx


def load_genome():
    """Load genome as dict of chr -> sequence."""
    from pyfaidx import Fasta
    return Fasta(str(GENOME))


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
            if not chrom.startswith('chr'):
                chrom = 'chr' + chrom
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
                    gene_id = attr[3:]  # strip 'ID='
                    break
            if gene_id:
                if strand == '+':
                    tss = start
                else:
                    tss = end
                genes[gene_id] = {'chr': chrom, 'start': start, 'end': end,
                                  'strand': strand, 'tss': tss}
    return genes


def load_pwm_db(pwm_path):
    """Load PWM database and group by family."""
    with open(pwm_path) as f:
        raw = json.load(f)
    # DB stores motifs as flat list under 'motifs' key
    motif_list = raw.get('motifs', [])
    by_family = defaultdict(list)
    for entry in motif_list:
        fam = entry.get('tf_family', 'unknown')
        by_family[fam].append(entry)
    return by_family


def process_chromosome(args):
    """Process all bQTL on one chromosome."""
    chrom, bqtl_positions, genome_path, pwm_data, gene_coords, ase_dict = args

    from pyfaidx import Fasta
    genome = Fasta(str(genome_path))

    # Load and compile PWMs
    by_family = load_pwm_db(pwm_data)

    # Build global compiled (all families together, with family labels)
    all_pwms = []
    for family, pwms in by_family.items():
        for p in pwms:
            all_pwms.append(dict(p, family=family))
    global_compiled = compile_pwm(all_pwms)

    chrom_seq = str(genome[chrom])
    chrom_len = len(chrom_seq)

    # Map bQTL to genes (promoter = 2kb upstream of TSS)
    promoter_map = {}  # pos -> (gene_id, dist_to_tss)
    for gene_id, info in gene_coords.items():
        if info['chr'] != chrom:
            continue
        if info['strand'] == '+':
            prom_start = max(0, info['tss'] - 2000)
            prom_end = info['tss']
        else:
            prom_start = info['tss']
            prom_end = min(chrom_len, info['tss'] + 2000)
        for pos in bqtl_positions:
            if prom_start <= pos <= prom_end:
                dist = abs(pos - info['tss'])
                if pos not in promoter_map or dist < promoter_map[pos][1]:
                    promoter_map[pos] = (gene_id, dist)

    results = []

    for pos in bqtl_positions:
        gene_id = None
        dist_to_tss = None
        if pos in promoter_map:
            gene_id, dist_to_tss = promoter_map[pos]

        # Extract bound region: ±250bp around bQTL
        region_start = max(0, pos - BOUND_REGION_BP)
        region_end = min(chrom_len, pos + BOUND_REGION_BP + 1)
        region_seq = chrom_seq[region_start:region_end].upper()
        region_idx = seq_to_idx(region_seq)

        # Position of variant within region
        var_pos_in_region = pos - region_start

        # Scan entire bound region for ALL motifs
        all_hits = scan_region_detailed(region_idx, global_compiled, PWM_THRESHOLD)

        # Count motifs by family in the bound region
        family_hits = defaultdict(list)
        for hit in all_hits:
            family_hits[hit['family']].append(hit)

        total_motifs = len(all_hits)
        n_families = len(family_hits)

        # Identify which motifs overlap the variant position (±28bp)
        variant_motifs = []
        for hit in all_hits:
            hit_start = hit['pos_in_seq']
            hit_end = hit_start + hit['width']
            if hit_start <= var_pos_in_region < hit_end:
                variant_motifs.append(hit)

        variant_families = set(h['family'] for h in variant_motifs)

        # For each family that overlaps variant: count redundancy
        # (other motifs of SAME family in bound region, NOT overlapping variant)
        family_details = []
        for fam, fam_hits in family_hits.items():
            overlaps_variant = fam in variant_families
            # Count motifs NOT overlapping variant
            non_variant_count = 0
            for hit in fam_hits:
                hit_start = hit['pos_in_seq']
                hit_end = hit_start + hit['width']
                if not (hit_start <= var_pos_in_region < hit_end):
                    non_variant_count += 1

            family_details.append({
                'family': fam,
                'total_in_region': len(fam_hits),
                'overlaps_variant': overlaps_variant,
                'redundant_copies': non_variant_count,
            })

        # Per-family redundancy for variant-overlapping families
        for fd in family_details:
            if fd['overlaps_variant']:
                ase_val = ase_dict.get(gene_id, np.nan) if gene_id else np.nan
                results.append({
                    'chr': chrom,
                    'pos': pos,
                    'gene_id': gene_id or '',
                    'dist_to_tss': dist_to_tss if dist_to_tss is not None else -1,
                    'tf_family': fd['family'],
                    'motifs_at_variant': sum(1 for h in variant_motifs if h['family'] == fd['family']),
                    'redundant_in_region': fd['redundant_copies'],
                    'total_family_in_region': fd['total_in_region'],
                    'total_motifs_in_region': total_motifs,
                    'n_families_in_region': n_families,
                    'families_in_region': ','.join(sorted(family_hits.keys())),
                    'variant_families': ','.join(sorted(variant_families)),
                    'n_variant_families': len(variant_families),
                    'abs_log2_ase': ase_val,
                    'region_width': region_end - region_start,
                })

        # Also store a region-level summary (one row per bQTL position)
        # This goes to a separate output
        if not variant_families:
            ase_val = ase_dict.get(gene_id, np.nan) if gene_id else np.nan
            results.append({
                'chr': chrom,
                'pos': pos,
                'gene_id': gene_id or '',
                'dist_to_tss': dist_to_tss if dist_to_tss is not None else -1,
                'tf_family': 'NONE',
                'motifs_at_variant': 0,
                'redundant_in_region': 0,
                'total_family_in_region': 0,
                'total_motifs_in_region': total_motifs,
                'n_families_in_region': n_families,
                'families_in_region': ','.join(sorted(family_hits.keys())),
                'variant_families': '',
                'n_variant_families': 0,
                'abs_log2_ase': ase_val,
                'region_width': region_end - region_start,
            })

    return results


def analyze_results(df):
    """Analyze bound region motif scan results."""
    print("=" * 70)
    print("BOUND REGION MOTIF ANALYSIS")
    print("=" * 70)

    # 1. Overall motif density in bound regions
    region_summary = df.drop_duplicates(['chr', 'pos'])
    print(f"\n--- 1. Motif density in ±{BOUND_REGION_BP}bp bound regions ---")
    print(f"Total bQTL positions: {len(region_summary)}")
    print(f"Total motifs per region: mean={region_summary['total_motifs_in_region'].mean():.1f}, "
          f"median={region_summary['total_motifs_in_region'].median():.0f}")
    print(f"TF families per region: mean={region_summary['n_families_in_region'].mean():.1f}, "
          f"median={region_summary['n_families_in_region'].median():.0f}")

    # Distribution of motif counts
    for thresh in [0, 5, 10, 20, 50]:
        pct = 100 * (region_summary['total_motifs_in_region'] > thresh).sum() / len(region_summary)
        print(f"  >{thresh} motifs: {pct:.1f}%")

    # 2. Which families are most common in bound regions?
    print(f"\n--- 2. Family prevalence in bound regions ---")
    family_presence = defaultdict(int)
    for families_str in region_summary['families_in_region']:
        if families_str:
            for fam in families_str.split(','):
                family_presence[fam] += 1
    total_regions = len(region_summary)
    print(f"Family | Present in N regions | % of regions")
    for fam, count in sorted(family_presence.items(), key=lambda x: -x[1]):
        print(f"  {fam:15s} | {count:6d} | {100*count/total_regions:.1f}%")

    # 3. Variant-overlapping motifs
    variant_rows = df[df['tf_family'] != 'NONE']
    no_variant = df[df['tf_family'] == 'NONE']
    print(f"\n--- 3. Motifs at variant position ---")
    print(f"bQTL with motif at variant: {variant_rows['pos'].nunique()} "
          f"({100*variant_rows['pos'].nunique()/len(region_summary):.1f}%)")
    print(f"bQTL with NO motif at variant: {len(no_variant)} "
          f"({100*len(no_variant)/len(region_summary):.1f}%)")

    if len(variant_rows) > 0:
        print(f"\nVariant-overlapping families:")
        var_fam_counts = variant_rows['tf_family'].value_counts()
        for fam, count in var_fam_counts.items():
            print(f"  {fam:15s} | {count:5d} bQTL")

    # 4. Redundancy within bound regions
    print(f"\n--- 4. Redundancy within bound regions ---")
    if len(variant_rows) > 0:
        print(f"For families whose motif overlaps the variant:")
        print(f"  Additional same-family motifs in region: "
              f"mean={variant_rows['redundant_in_region'].mean():.1f}, "
              f"median={variant_rows['redundant_in_region'].median():.0f}")
        sole = (variant_rows['redundant_in_region'] == 0).sum()
        print(f"  SOLE (no redundant): {sole} ({100*sole/len(variant_rows):.1f}%)")
        for thresh in [1, 2, 3, 5, 10]:
            pct = 100 * (variant_rows['redundant_in_region'] >= thresh).sum() / len(variant_rows)
            print(f"  ≥{thresh} redundant: {pct:.1f}%")

        # Per-family redundancy
        print(f"\n  Per-family redundancy (within ±{BOUND_REGION_BP}bp bound region):")
        for fam in sorted(variant_rows['tf_family'].unique()):
            fam_data = variant_rows[variant_rows['tf_family'] == fam]
            if len(fam_data) < 10:
                continue
            mean_red = fam_data['redundant_in_region'].mean()
            sole_pct = 100 * (fam_data['redundant_in_region'] == 0).sum() / len(fam_data)
            print(f"    {fam:15s} | n={len(fam_data):5d} | "
                  f"mean_redundancy={mean_red:.1f} | sole={sole_pct:.0f}%")

    # 5. Redundancy vs ASE (the key question)
    print(f"\n--- 5. Redundancy vs allelic expression ---")
    if len(variant_rows) > 0:
        has_ase = variant_rows[variant_rows['abs_log2_ase'].notna()]
        if len(has_ase) > 0:
            from scipy import stats
            rho, p = stats.spearmanr(has_ase['redundant_in_region'], has_ase['abs_log2_ase'])
            print(f"  Spearman correlation (redundancy vs |ASE|): rho={rho:.4f}, p={p:.2e}")
            print(f"  N = {len(has_ase)}")

            # Binned analysis
            bins = [0, 1, 3, 6, 100]
            labels = ['sole(0)', '1-2', '3-5', '6+']
            has_ase = has_ase.copy()
            has_ase['red_bin'] = pd.cut(has_ase['redundant_in_region'], bins=bins, labels=labels, right=False)
            bin_stats = has_ase.groupby('red_bin', observed=True).agg(
                n=('abs_log2_ase', 'count'),
                mean_ase=('abs_log2_ase', 'mean'),
                median_ase=('abs_log2_ase', 'median'),
                se=('abs_log2_ase', 'sem'),
            )
            print(f"\n  Redundancy bin | N     | Mean |ASE| | Median | SE")
            for idx, row in bin_stats.iterrows():
                print(f"    {idx:10s}   | {row['n']:5.0f} | {row['mean_ase']:.4f}  | "
                      f"{row['median_ase']:.4f} | {row['se']:.4f}")

            # Compare sole vs redundant (Mann-Whitney)
            sole_ase = has_ase[has_ase['redundant_in_region'] == 0]['abs_log2_ase']
            redundant_ase = has_ase[has_ase['redundant_in_region'] > 0]['abs_log2_ase']
            if len(sole_ase) > 10 and len(redundant_ase) > 10:
                u_stat, u_p = stats.mannwhitneyu(sole_ase, redundant_ase, alternative='greater')
                d = (sole_ase.mean() - redundant_ase.mean()) / np.sqrt(
                    (sole_ase.var() + redundant_ase.var()) / 2)
                print(f"\n  Sole vs Redundant (Mann-Whitney, sole > redundant):")
                print(f"    Sole: n={len(sole_ase)}, mean={sole_ase.mean():.4f}")
                print(f"    Redundant: n={len(redundant_ase)}, mean={redundant_ase.mean():.4f}")
                print(f"    U={u_stat:.0f}, p={u_p:.2e}, Cohen's d={d:.3f}")

    # 6. Compare to motif density in RANDOM promoter regions
    print(f"\n--- 6. Motif density: bound regions vs random ---")
    print(f"  (See control analysis below)")

    return variant_rows


def analyze_marand_overlap(df, bqtl):
    """Cross-reference bQTL with Marand ACR peaks."""
    if not MARAND.exists():
        print("Marand ACR data not found, skipping")
        return

    marand = pd.read_csv(MARAND, sep='\t', header=None,
                         names=['chr', 'start', 'end', 'cell_type', 'score'])
    unique_peaks = marand.drop_duplicates(['chr', 'start', 'end'])

    print(f"\n{'='*70}")
    print(f"MARAND ACR PEAK OVERLAP")
    print(f"{'='*70}")

    # Build interval index
    import bisect
    mar_by_chr = defaultdict(list)
    for _, row in unique_peaks.iterrows():
        mar_by_chr[row['chr']].append((row['start'], row['end']))
    for c in mar_by_chr:
        mar_by_chr[c].sort()

    # Check which bQTL fall in ACR peaks
    acr_bqtl = set()
    bqtl_to_acr = {}
    for chrom, intervals in mar_by_chr.items():
        starts = [iv[0] for iv in intervals]
        ends = [iv[1] for iv in intervals]
        chr_bqtl = bqtl[bqtl['chr'] == chrom]
        for _, row in chr_bqtl.iterrows():
            pos = row['pos']
            idx = bisect.bisect_right(starts, pos) - 1
            if idx >= 0 and pos <= ends[idx]:
                acr_bqtl.add(pos)
                bqtl_to_acr[pos] = (starts[idx], ends[idx])

    print(f"bQTL in Marand ACR peaks: {len(acr_bqtl)}/{len(bqtl)} ({100*len(acr_bqtl)/len(bqtl):.1f}%)")

    # Compare motif density: ACR-overlapping vs non-ACR bQTL
    region_summary = df.drop_duplicates(['chr', 'pos'])
    in_acr = region_summary[region_summary['pos'].isin(acr_bqtl)]
    not_acr = region_summary[~region_summary['pos'].isin(acr_bqtl)]

    print(f"\nMotif density comparison:")
    print(f"  In ACR: mean={in_acr['total_motifs_in_region'].mean():.1f} motifs, "
          f"{in_acr['n_families_in_region'].mean():.1f} families")
    print(f"  Not in ACR: mean={not_acr['total_motifs_in_region'].mean():.1f} motifs, "
          f"{not_acr['n_families_in_region'].mean():.1f} families")

    # Cell-type breakdown
    print(f"\nCell-type breakdown of bQTL in ACR peaks:")
    acr_positions = set(acr_bqtl)
    cell_type_counts = defaultdict(int)
    for _, row in marand.iterrows():
        # Check if any bQTL falls in this peak
        chr_bqtl = bqtl[bqtl['chr'] == row['chr']]
        mask = (chr_bqtl['pos'] >= row['start']) & (chr_bqtl['pos'] <= row['end'])
        if mask.any():
            cell_type_counts[row['cell_type']] += mask.sum()

    for ct, count in sorted(cell_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ct:20s}: {count:5d} bQTL")


def run_control_analysis(genome, gene_coords, pwm_data, n_controls=5000):
    """Compare motif density in bQTL-centered regions vs random promoter regions."""
    import random
    random.seed(42)

    by_family = load_pwm_db(pwm_data)
    all_pwms = []
    for family, pwms in by_family.items():
        for p in pwms:
            all_pwms.append(dict(p, family=family))
    global_compiled = compile_pwm(all_pwms)

    from pyfaidx import Fasta
    genome_fa = Fasta(str(genome))

    # Sample random promoter positions (within 2kb upstream of genes)
    promoter_genes = [g for g, info in gene_coords.items()
                      if info['chr'] in {f'chr{i}' for i in range(1, 11)}]
    sampled_genes = random.sample(promoter_genes, min(n_controls, len(promoter_genes)))

    control_motif_counts = []
    control_family_counts = []

    for gene_id in sampled_genes:
        info = gene_coords[gene_id]
        chrom = info['chr']
        chrom_seq = str(genome_fa[chrom])
        chrom_len = len(chrom_seq)

        if info['strand'] == '+':
            prom_start = max(0, info['tss'] - 2000)
            prom_end = info['tss']
        else:
            prom_start = info['tss']
            prom_end = min(chrom_len, info['tss'] + 2000)

        # Random position within promoter
        if prom_end - prom_start < 2 * BOUND_REGION_BP:
            continue
        rand_pos = random.randint(prom_start + BOUND_REGION_BP,
                                  prom_end - BOUND_REGION_BP)

        region_start = rand_pos - BOUND_REGION_BP
        region_end = rand_pos + BOUND_REGION_BP + 1
        region_seq = chrom_seq[region_start:region_end].upper()
        region_idx = seq_to_idx(region_seq)

        hits = scan_region_detailed(region_idx, global_compiled, PWM_THRESHOLD)
        control_motif_counts.append(len(hits))
        families = set(h['family'] for h in hits)
        control_family_counts.append(len(families))

    return np.array(control_motif_counts), np.array(control_family_counts)


def generate_figures(df, control_motifs, control_families):
    """Generate publication figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy import stats

    fig_dir = OUTDIR / 'figures'
    fig_dir.mkdir(exist_ok=True)

    region_summary = df.drop_duplicates(['chr', 'pos'])
    variant_rows = df[df['tf_family'] != 'NONE']

    # ============================================================
    # Figure: Bound Region Motif Landscape
    # ============================================================
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Panel A: Motif count distribution (bQTL regions vs controls)
    ax1 = fig.add_subplot(gs[0, 0])
    bqtl_counts = region_summary['total_motifs_in_region'].values
    ax1.hist(bqtl_counts, bins=50, alpha=0.6, density=True, label='bQTL regions', color='#2196F3')
    ax1.hist(control_motifs, bins=50, alpha=0.6, density=True, label='Random promoter', color='#9E9E9E')
    ax1.set_xlabel('Total motifs in ±250bp region')
    ax1.set_ylabel('Density')
    ax1.set_title('A. Motif density: bound vs random')
    ax1.legend(fontsize=8)
    u, p = stats.mannwhitneyu(bqtl_counts, control_motifs, alternative='two-sided')
    ax1.text(0.95, 0.95, f'MWU p={p:.2e}\nbQTL: {np.mean(bqtl_counts):.1f}±{np.std(bqtl_counts):.1f}'
             f'\nCtrl: {np.mean(control_motifs):.1f}±{np.std(control_motifs):.1f}',
             transform=ax1.transAxes, va='top', ha='right', fontsize=7,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel B: Family count distribution
    ax2 = fig.add_subplot(gs[0, 1])
    bqtl_fam = region_summary['n_families_in_region'].values
    ax2.hist(bqtl_fam, bins=range(0, 20), alpha=0.6, density=True, label='bQTL regions', color='#2196F3')
    ax2.hist(control_families, bins=range(0, 20), alpha=0.6, density=True, label='Random promoter', color='#9E9E9E')
    ax2.set_xlabel('TF families in ±250bp region')
    ax2.set_ylabel('Density')
    ax2.set_title('B. Family diversity: bound vs random')
    ax2.legend(fontsize=8)

    # Panel C: Redundancy distribution for variant-overlapping families
    ax3 = fig.add_subplot(gs[0, 2])
    if len(variant_rows) > 0:
        red_vals = variant_rows['redundant_in_region'].values
        ax3.hist(red_vals, bins=range(0, max(15, int(np.max(red_vals))+2)), alpha=0.7,
                 color='#FF5722', edgecolor='white')
        ax3.axvline(np.median(red_vals), color='k', linestyle='--', label=f'Median={np.median(red_vals):.0f}')
        sole_pct = 100 * (red_vals == 0).sum() / len(red_vals)
        ax3.set_xlabel('Redundant same-family motifs in region')
        ax3.set_ylabel('Count')
        ax3.set_title(f'C. Redundancy in bound region\n(sole: {sole_pct:.0f}%)')
        ax3.legend(fontsize=8)

    # Panel D: Per-family redundancy bar chart
    ax4 = fig.add_subplot(gs[1, 0])
    if len(variant_rows) > 0:
        fam_red = variant_rows.groupby('tf_family').agg(
            mean_red=('redundant_in_region', 'mean'),
            n=('redundant_in_region', 'count'),
            sole_pct=('redundant_in_region', lambda x: 100 * (x == 0).sum() / len(x))
        ).reset_index()
        fam_red = fam_red[fam_red['n'] >= 20].sort_values('mean_red', ascending=False)
        colors = ['#E91E63' if r > 50 else '#2196F3' for r in fam_red['sole_pct']]
        ax4.barh(range(len(fam_red)), fam_red['mean_red'].values, color=colors, edgecolor='white')
        ax4.set_yticks(range(len(fam_red)))
        ax4.set_yticklabels(fam_red['tf_family'].values, fontsize=7)
        ax4.set_xlabel('Mean redundant copies in bound region')
        ax4.set_title('D. Per-family redundancy')
        ax4.invert_yaxis()

    # Panel E: Redundancy vs ASE
    ax5 = fig.add_subplot(gs[1, 1])
    if len(variant_rows) > 0:
        has_ase = variant_rows[variant_rows['abs_log2_ase'].notna()].copy()
        if len(has_ase) > 0:
            bins = [0, 1, 2, 3, 5, 8, 100]
            labels = ['0', '1', '2', '3-4', '5-7', '8+']
            has_ase['red_bin'] = pd.cut(has_ase['redundant_in_region'], bins=bins, labels=labels, right=False)
            bin_stats = has_ase.groupby('red_bin', observed=True).agg(
                mean=('abs_log2_ase', 'mean'),
                se=('abs_log2_ase', 'sem'),
                n=('abs_log2_ase', 'count'),
            ).reset_index()
            x = range(len(bin_stats))
            ax5.bar(x, bin_stats['mean'], yerr=bin_stats['se'], color='#4CAF50',
                    edgecolor='white', capsize=3)
            ax5.set_xticks(x)
            ax5.set_xticklabels(bin_stats['red_bin'], fontsize=8)
            ax5.set_xlabel('Redundant motifs in bound region')
            ax5.set_ylabel('Mean |log₂(B73/NAM)|')
            ax5.set_title('E. Redundancy vs allelic expression')
            for i, row in bin_stats.iterrows():
                ax5.text(i, row['mean'] + row['se'] + 0.01,
                         f'n={row["n"]:.0f}', ha='center', fontsize=6)

    # Panel F: Combinatorial disruption in bound region
    ax6 = fig.add_subplot(gs[1, 2])
    if len(variant_rows) > 0:
        has_ase2 = region_summary[region_summary['abs_log2_ase'].notna()].copy()
        has_ase2['n_var_fam_bin'] = pd.cut(has_ase2['n_variant_families'],
                                            bins=[0, 1, 2, 3, 5, 100],
                                            labels=['0', '1', '2', '3-4', '5+'], right=False)
        bin_stats2 = has_ase2.groupby('n_var_fam_bin', observed=True).agg(
            mean=('abs_log2_ase', 'mean'),
            se=('abs_log2_ase', 'sem'),
            n=('abs_log2_ase', 'count'),
        ).reset_index()
        x = range(len(bin_stats2))
        ax6.bar(x, bin_stats2['mean'], yerr=bin_stats2['se'], color='#9C27B0',
                edgecolor='white', capsize=3)
        ax6.set_xticks(x)
        ax6.set_xticklabels(bin_stats2['n_var_fam_bin'], fontsize=8)
        ax6.set_xlabel('TF families disrupted at variant')
        ax6.set_ylabel('Mean |log₂(B73/NAM)|')
        ax6.set_title('F. Multi-family disruption vs ASE')
        for i, row in bin_stats2.iterrows():
            ax6.text(i, row['mean'] + row['se'] + 0.01,
                     f'n={row["n"]:.0f}', ha='center', fontsize=6)

    plt.savefig(fig_dir / 'fig5_bound_region_motifs.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(fig_dir / 'fig5_bound_region_motifs.png', bbox_inches='tight', dpi=150)
    print(f"\nSaved: figures/fig5_bound_region_motifs.pdf")
    plt.close()


def main():
    print("=" * 70)
    print("MOA-SEQ BOUND REGION MOTIF ANALYSIS")
    print(f"Region: ±{BOUND_REGION_BP}bp around each bQTL")
    print("=" * 70)

    # Load data
    bqtl = pd.read_csv(BQTL)
    ase_df = pd.read_csv(ASE)
    ase_dict = dict(zip(ase_df['gene_id'], ase_df['mean_abs_log2']))
    gene_coords = parse_gff3()

    print(f"bQTL: {len(bqtl)} positions")
    print(f"ASE: {len(ase_dict)} genes")
    print(f"Gene annotations: {len(gene_coords)}")

    # Prepare per-chromosome args
    chromosomes = sorted(bqtl['chr'].unique())
    args_list = []
    for chrom in chromosomes:
        chr_bqtl = sorted(bqtl[bqtl['chr'] == chrom]['pos'].values)
        args_list.append((
            chrom, chr_bqtl, str(GENOME), str(PWM_DB), gene_coords, ase_dict
        ))

    print(f"\nProcessing {len(chromosomes)} chromosomes with multiprocessing...")

    # Run with multiprocessing
    all_results = []
    with Pool(processes=min(10, len(chromosomes))) as pool:
        for i, results in enumerate(pool.imap_unordered(process_chromosome, args_list)):
            n_pos = len(set(r['pos'] for r in results))
            print(f"  {args_list[i][0]}: {n_pos} positions, {len(results)} records")
            all_results.extend(results)

    # Combine results
    df = pd.DataFrame(all_results)
    print(f"\nTotal records: {len(df)}")
    print(f"Unique positions: {df[['chr','pos']].drop_duplicates().shape[0]}")

    # Save raw results
    df.to_csv(OUTDIR / 'bound_region_motifs.csv', index=False)
    print(f"Saved: bound_region_motifs.csv")

    # Analyze
    variant_rows = analyze_results(df)

    # Marand ACR overlap
    analyze_marand_overlap(df, bqtl)

    # Control analysis
    print(f"\n{'='*70}")
    print("CONTROL ANALYSIS: random promoter regions")
    print(f"{'='*70}")
    control_motifs, control_families = run_control_analysis(
        GENOME, gene_coords, str(PWM_DB), n_controls=5000
    )
    print(f"Random promoter regions (n={len(control_motifs)}):")
    print(f"  Motifs: mean={control_motifs.mean():.1f}, median={np.median(control_motifs):.0f}")
    print(f"  Families: mean={control_families.mean():.1f}, median={np.median(control_families):.0f}")

    region_summary = df.drop_duplicates(['chr', 'pos'])
    bqtl_motifs = region_summary['total_motifs_in_region'].values
    from scipy import stats
    u, p = stats.mannwhitneyu(bqtl_motifs, control_motifs, alternative='two-sided')
    print(f"\n  bQTL regions: mean={bqtl_motifs.mean():.1f}")
    print(f"  Control regions: mean={control_motifs.mean():.1f}")
    print(f"  Ratio: {bqtl_motifs.mean() / control_motifs.mean():.2f}x")
    print(f"  Mann-Whitney U p={p:.2e}")

    # Generate figures
    generate_figures(df, control_motifs, control_families)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    n_total = df[['chr','pos']].drop_duplicates().shape[0]
    n_with_variant_motif = df[df['tf_family'] != 'NONE']['pos'].nunique()
    print(f"Total bQTL analyzed: {n_total}")
    print(f"bQTL with motif at variant: {n_with_variant_motif} ({100*n_with_variant_motif/n_total:.1f}%)")
    print(f"Mean motifs per bound region: {bqtl_motifs.mean():.1f} (control: {control_motifs.mean():.1f})")
    if len(variant_rows) > 0:
        sole = (variant_rows['redundant_in_region'] == 0).sum()
        print(f"Sole motifs (no redundancy): {sole} ({100*sole/len(variant_rows):.1f}%)")


if __name__ == '__main__':
    main()
