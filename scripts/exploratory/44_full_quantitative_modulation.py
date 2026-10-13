#!/usr/bin/env python3
"""
Full-scale quantitative modulation analysis: ALL bQTL in promoters.

Key optimizations over 43_quantitative_modulation.py:
1. Fully vectorized PWM scoring with numpy (no Python inner loops)
2. Multiprocessing across chromosomes
3. Batch sequence extraction
4. Pre-encoded sequences as int8 arrays

Tests: motif redundancy vs allelic expression, per-family effects,
combinatorial disruption, disruption fraction, compound variants.
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
GFF3 = DATA / "raw" / "Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3"
GENOME = DATA / "raw" / "B73_NAM5.fa"
PWM_DB = DATA / "processed" / "maize_tf_pwm_database.json"
BQTL = DATA / "processed" / "bqtl_snp_ww.csv"
ASE = DATA / "processed" / "engelhorn_ase_ww_gene_summary.csv"
ASE_HYBRID = DATA / "processed" / "engelhorn_ase_ww.csv"
EXPR = DATA / "processed" / "engelhorn_ww_vs_ds_expression.tsv"
OUTDIR = BASE / "results"

PROMOTER_BP = 2000
VARIANT_WINDOW = 28   # ±28bp for variant-overlap scan
REDUNDANCY_WINDOW = 200  # ±200bp for redundancy count
PWM_THRESHOLD = 6.0
BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
BG = np.array([0.2642, 0.2358, 0.2358, 0.2642])

# ================================================================
# VECTORIZED PWM SCORING
# ================================================================

def compile_pwm(pwm_list):
    """Precompile log-odds matrices grouped by width for batch scoring."""
    by_width = defaultdict(list)
    for entry in pwm_list:
        mat = np.array(entry['pwm'], dtype=np.float64)
        mat = np.clip(mat, 1e-4, None)
        lo = np.log2(mat / BG[np.newaxis, :])
        rc = lo[::-1, ::-1].copy()
        by_width[len(mat)].append((lo, rc))
    # Stack matrices of same width for batch scoring
    compiled = {}
    for w, pairs in by_width.items():
        fwd_stack = np.array([p[0] for p in pairs])  # (n_pwms, w, 4)
        rc_stack = np.array([p[1] for p in pairs])
        compiled[w] = (fwd_stack, rc_stack)
    return compiled


def score_seq_vectorized(seq_idx, compiled, threshold=6.0):
    """Score sequence against all PWMs in compiled dict. Returns hit count.

    seq_idx: int8 array of base indices (0-3, -1 for N)
    compiled: {width: (fwd_stack, rc_stack)} from compile_pwm
    """
    L = len(seq_idx)
    total_hits = 0

    for w, (fwd_stack, rc_stack) in compiled.items():
        if L < w:
            continue
        n_pwms = fwd_stack.shape[0]
        n_windows = L - w + 1

        # Build windows: (n_windows, w)
        windows = np.lib.stride_tricks.as_strided(
            seq_idx,
            shape=(n_windows, w),
            strides=(seq_idx.strides[0], seq_idx.strides[0])
        ).copy()

        # Mask windows with N bases
        valid = np.all(windows >= 0, axis=1)  # (n_windows,)

        if not np.any(valid):
            continue

        valid_windows = windows[valid]  # (n_valid, w)
        n_valid = valid_windows.shape[0]

        # Score all PWMs at all valid windows
        # fwd_stack: (n_pwms, w, 4), valid_windows: (n_valid, w)
        # We need scores[p, i] = sum_j fwd_stack[p, j, valid_windows[i, j]]
        pos_idx = np.arange(w)[np.newaxis, :]  # (1, w)
        base_idx = valid_windows  # (n_valid, w)

        # Use advanced indexing: for each pwm p, position j, get log_odds[p, j, base]
        # Reshape for broadcasting: (n_pwms, 1, w) and (1, n_valid, w)
        fwd_scores = np.sum(
            fwd_stack[:, np.arange(w), :][
                :, :, :  # (n_pwms, w, 4)
            ][:, np.newaxis, :, :][  # (n_pwms, 1, w, 4)
                np.arange(n_pwms)[:, np.newaxis, np.newaxis],
                0,
                pos_idx[np.newaxis, :, :],  # (1, 1, w)
                base_idx[np.newaxis, :, :]  # (1, n_valid, w)
            ],
            axis=2
        )  # (n_pwms, n_valid)

        rc_scores = np.sum(
            rc_stack[
                np.arange(n_pwms)[:, np.newaxis, np.newaxis],
                pos_idx[np.newaxis, :, :],
                base_idx[np.newaxis, :, :]
            ],
            axis=2
        )  # (n_pwms, n_valid)

        # Count hits where either fwd or rc exceeds threshold
        hits = np.any(
            (fwd_scores >= threshold) | (rc_scores >= threshold),
            axis=0  # across PWMs
        )
        total_hits += np.sum(hits)

    return total_hits


def has_hit_vectorized(seq_idx, compiled, threshold=6.0):
    """Check if ANY PWM hits in sequence (early exit)."""
    L = len(seq_idx)
    for w, (fwd_stack, rc_stack) in compiled.items():
        if L < w:
            continue
        n_windows = L - w + 1
        windows = np.lib.stride_tricks.as_strided(
            seq_idx,
            shape=(n_windows, w),
            strides=(seq_idx.strides[0], seq_idx.strides[0])
        ).copy()
        valid = np.all(windows >= 0, axis=1)
        if not np.any(valid):
            continue
        valid_windows = windows[valid]
        n_valid = valid_windows.shape[0]
        n_pwms = fwd_stack.shape[0]
        pos_idx = np.arange(w)
        base_idx = valid_windows

        fwd_scores = np.sum(
            fwd_stack[
                np.arange(n_pwms)[:, np.newaxis, np.newaxis],
                pos_idx[np.newaxis, np.newaxis, :],
                base_idx[np.newaxis, :, :]
            ],
            axis=2
        )
        if np.any(fwd_scores >= threshold):
            return True
        rc_scores = np.sum(
            rc_stack[
                np.arange(n_pwms)[:, np.newaxis, np.newaxis],
                pos_idx[np.newaxis, np.newaxis, :],
                base_idx[np.newaxis, :, :]
            ],
            axis=2
        )
        if np.any(rc_scores >= threshold):
            return True
    return False


def seq_to_idx(seq):
    """Convert sequence string to int8 array."""
    return np.array([BASE_MAP.get(b, -1) for b in seq], dtype=np.int8)


# ================================================================
# GFF3 PARSING
# ================================================================

def parse_gene_coords(gff3_path):
    """Extract gene coordinates and feature structure from GFF3."""
    genes = {}
    features = defaultdict(list)  # gene_id -> [(type, start, end)]
    current_gene = None
    with open(gff3_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            ftype = parts[2]
            chrom = 'chr' + parts[0] if not parts[0].startswith('chr') else parts[0]
            start, end = int(parts[3]), int(parts[4])
            strand = parts[6]
            attrs = dict(kv.split('=', 1) for kv in parts[8].split(';') if '=' in kv)

            if ftype == 'gene':
                gene_id = attrs.get('ID', '').replace('gene:', '')
                if not gene_id.startswith('Zm'):
                    continue
                genes[gene_id] = {
                    'chr': chrom, 'start': start, 'end': end, 'strand': strand
                }
                current_gene = gene_id
            elif ftype in ('CDS', 'five_prime_UTR', 'three_prime_UTR', 'exon'):
                parent = attrs.get('Parent', '')
                # Find gene from mRNA parent
                if current_gene:
                    features[current_gene].append((ftype, start, end))
    return genes, features


def classify_position(pos, gene_info, gene_features):
    """Classify a genomic position relative to gene structure."""
    if pos < gene_info['start'] or pos > gene_info['end']:
        # Upstream or downstream
        if gene_info['strand'] == '+':
            if pos < gene_info['start']:
                dist = gene_info['start'] - pos
                return 'promoter' if dist <= PROMOTER_BP else 'distal'
        else:
            if pos > gene_info['end']:
                dist = pos - gene_info['end']
                return 'promoter' if dist <= PROMOTER_BP else 'distal'
        return 'intergenic'

    # Within gene body
    for ftype, fstart, fend in gene_features:
        if fstart <= pos <= fend:
            if ftype == 'CDS':
                return 'CDS'
            elif ftype == 'five_prime_UTR':
                return '5UTR'
            elif ftype == 'three_prime_UTR':
                return '3UTR'
    return 'intron'


# ================================================================
# WORKER FUNCTION FOR MULTIPROCESSING
# ================================================================

def process_chromosome(args):
    """Process all bQTL positions on one chromosome."""
    chrom, positions, gene_mappings, pos_types, family_compiled_data, \
        ase_dict, ase_n_dict, genome_path = args

    import pyfaidx
    genome = pyfaidx.Fasta(str(genome_path))

    # Reconstruct compiled PWMs from serializable data
    family_compiled = {}
    for fam, width_data in family_compiled_data.items():
        compiled = {}
        for w_str, (fwd_arr, rc_arr) in width_data.items():
            compiled[int(w_str)] = (np.array(fwd_arr), np.array(rc_arr))
        family_compiled[fam] = compiled

    chrom_key = chrom if chrom in genome else chrom.replace('chr', '')
    if chrom_key not in genome:
        return []

    chrom_len = len(genome[chrom_key])
    results = []

    for pos in positions:
        # Extract variant-overlap window
        var_start = max(0, pos - VARIANT_WINDOW - 1)
        var_end = min(chrom_len, pos + VARIANT_WINDOW)
        var_seq = seq_to_idx(str(genome[chrom_key][var_start:var_end]).upper())

        # Extract redundancy window
        red_start = max(0, pos - REDUNDANCY_WINDOW - 1)
        red_end = min(chrom_len, pos + REDUNDANCY_WINDOW)
        red_seq = seq_to_idx(str(genome[chrom_key][red_start:red_end]).upper())

        for fam, compiled in family_compiled.items():
            if has_hit_vectorized(var_seq, compiled, PWM_THRESHOLD):
                n_total = score_seq_vectorized(red_seq, compiled, PWM_THRESHOLD)
                n_other = max(0, n_total - 1)

                for gene_id in gene_mappings.get((chrom, pos), []):
                    results.append({
                        'chr': chrom,
                        'pos': pos,
                        'gene_id': gene_id,
                        'tf_family': fam,
                        'bqtl_type': pos_types.get((chrom, pos), ''),
                        'n_other_motifs': n_other,
                        'is_sole': n_other == 0,
                        'abs_log2_ase': ase_dict.get(gene_id, np.nan),
                        'n_hybrids': ase_n_dict.get(gene_id, 0),
                    })

    return results


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 70, flush=True)
    print("FULL-SCALE QUANTITATIVE MODULATION ANALYSIS", flush=True)
    print("All bQTL in promoters × 18 TF families", flush=True)
    print("=" * 70, flush=True)

    # ----------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------
    print("\n[1] Loading data...", flush=True)

    bqtl_df = pd.read_csv(BQTL)
    print(f"  {len(bqtl_df)} bQTL total", flush=True)

    ase_df = pd.read_csv(ASE)
    ase_dict = dict(zip(ase_df['gene_id'], ase_df['mean_abs_log2']))
    ase_n_dict = dict(zip(ase_df['gene_id'], ase_df['n_hybrids']))
    print(f"  {len(ase_df)} genes with ASE data", flush=True)

    # Load per-hybrid ASE for cross-hybrid analyses
    ase_hybrid_exists = ASE_HYBRID.exists()
    if ase_hybrid_exists:
        ase_hybrid_df = pd.read_csv(ASE_HYBRID)
        print(f"  {len(ase_hybrid_df)} hybrid-level ASE entries", flush=True)

    # Load expression data
    expr_df = pd.read_csv(EXPR, sep='\t')
    expr_dict = dict(zip(expr_df['gene_id'], expr_df['mean_expression']))
    drought_dict = {}
    fc_col = next((c for c in expr_df.columns if 'log2' in c.lower() and 'fc' in c.lower()), None)
    if fc_col:
        drought_dict = dict(zip(expr_df['gene_id'], expr_df[fc_col]))
        print(f"  Drought FC column: {fc_col}", flush=True)
    print(f"  {len(expr_df)} genes with expression data", flush=True)

    gene_coords, gene_features = parse_gene_coords(GFF3)
    print(f"  {len(gene_coords)} gene coordinates", flush=True)

    with open(PWM_DB) as f:
        db = json.load(f)

    # Group and compile PWMs by family
    family_pwms_raw = defaultdict(list)
    for m in db['motifs']:
        family_pwms_raw[m['tf_family']].append(m)

    TOP_FAMILIES = sorted(family_pwms_raw.keys(),
                          key=lambda f: len(family_pwms_raw[f]), reverse=True)[:18]
    family_compiled = {}
    family_compiled_serial = {}  # serializable version for multiprocessing
    for fam in TOP_FAMILIES:
        compiled = compile_pwm(family_pwms_raw[fam])
        family_compiled[fam] = compiled
        # Convert to lists for pickling
        serial = {}
        for w, (fwd, rc) in compiled.items():
            serial[str(w)] = (fwd.tolist(), rc.tolist())
        family_compiled_serial[fam] = serial

    total_pwms = sum(
        sum(fwd.shape[0] for fwd, rc in c.values())
        for c in family_compiled.values()
    )
    print(f"  {total_pwms} PWMs across {len(family_compiled)} families", flush=True)

    # ----------------------------------------------------------
    # 2. Build promoter index and map bQTL
    # ----------------------------------------------------------
    print("\n[2] Building promoter index...", flush=True)
    chrom_promoters = defaultdict(list)
    for gene_id, info in gene_coords.items():
        if gene_id not in ase_dict:
            continue
        chrom = info['chr']
        if info['strand'] == '+':
            pstart = max(1, info['start'] - PROMOTER_BP)
            pend = info['start']
        else:
            pstart = info['end']
            pend = info['end'] + PROMOTER_BP
        chrom_promoters[chrom].append((pstart, pend, gene_id))
    for chrom in chrom_promoters:
        chrom_promoters[chrom].sort()
    n_promoters = sum(len(v) for v in chrom_promoters.values())
    print(f"  {n_promoters} promoters indexed", flush=True)

    print("\n[3] Mapping bQTL to promoters...", flush=True)
    pos_to_genes = defaultdict(list)
    pos_to_type = {}
    for _, row in bqtl_df.iterrows():
        chrom = row['chr']
        pos = row['pos']
        if chrom not in chrom_promoters:
            continue
        for pstart, pend, gene_id in chrom_promoters[chrom]:
            if pstart > pos + 1000:
                break
            if pstart <= pos <= pend:
                pos_to_genes[(chrom, pos)].append(gene_id)
                pos_to_type[(chrom, pos)] = row['Type']

    unique_positions = sorted(pos_to_genes.keys())
    print(f"  {len(unique_positions)} unique bQTL positions in promoters", flush=True)

    # ----------------------------------------------------------
    # 3. Build gene body interval index (for compound variant analysis)
    # ----------------------------------------------------------
    print("\n[4] Building gene body interval index...", flush=True)
    # Build sorted interval lists per chromosome for fast lookup
    chrom_gene_intervals = defaultdict(list)
    for gene_id, info in gene_coords.items():
        chrom = info['chr']
        # Extended region: gene body + promoter
        if info['strand'] == '+':
            ext_start = max(1, info['start'] - PROMOTER_BP)
            ext_end = info['end']
        else:
            ext_start = info['start']
            ext_end = info['end'] + PROMOTER_BP
        chrom_gene_intervals[chrom].append((ext_start, ext_end, gene_id))
    for chrom in chrom_gene_intervals:
        chrom_gene_intervals[chrom].sort()
    print(f"  Gene interval index built", flush=True)

    # ----------------------------------------------------------
    # 4. Scan motifs — multiprocessing by chromosome
    # ----------------------------------------------------------
    print(f"\n[5] Scanning motifs at {len(unique_positions)} positions...", flush=True)
    print(f"    Using {min(cpu_count(), 10)} processes", flush=True)

    # Group positions by chromosome
    chrom_positions = defaultdict(list)
    for chrom, pos in unique_positions:
        chrom_positions[chrom].append(pos)

    # Prepare worker args
    worker_args = []
    for chrom in sorted(chrom_positions.keys()):
        positions = chrom_positions[chrom]
        # Filter gene_mappings and pos_types for this chromosome
        gene_map = {(c, p): genes for (c, p), genes in pos_to_genes.items() if c == chrom}
        type_map = {(c, p): t for (c, p), t in pos_to_type.items() if c == chrom}
        worker_args.append((
            chrom, positions, gene_map, type_map,
            family_compiled_serial, ase_dict, ase_n_dict, str(GENOME)
        ))

    n_procs = min(cpu_count(), 10)
    all_results = []

    with Pool(n_procs) as pool:
        for i, chunk_results in enumerate(pool.imap_unordered(process_chromosome, worker_args)):
            all_results.extend(chunk_results)
            chrom_name = worker_args[i][0] if i < len(worker_args) else "?"
            print(f"    Chromosome done: {len(chunk_results)} triples", flush=True)

    print(f"\n  Total: {len(all_results)} bQTL-gene-TF triples", flush=True)

    if not all_results:
        print("ERROR: No results generated.", flush=True)
        return

    res_df = pd.DataFrame(all_results)
    res_df.to_csv(OUTDIR / 'quantitative_modulation_full.csv', index=False)
    print(f"  Saved: quantitative_modulation_full.csv", flush=True)

    # ----------------------------------------------------------
    # 5. ANALYSIS
    # ----------------------------------------------------------
    analyze_results(res_df, ase_dict, ase_n_dict, expr_dict, drought_dict,
                    pos_to_genes, pos_to_type, gene_coords, gene_features)


def analyze_results(res_df, ase_dict, ase_n_dict, expr_dict, drought_dict,
                    pos_to_genes, pos_to_type, gene_coords, gene_features):
    """Comprehensive analysis of quantitative modulation results."""
    from scipy import stats

    print("\n" + "=" * 70, flush=True)
    print("RESULTS: FULL-SCALE QUANTITATIVE MODULATION", flush=True)
    print("=" * 70, flush=True)

    # Filter to rows with ASE data
    df = res_df.dropna(subset=['abs_log2_ase']).copy()

    sole = df[df['is_sole']]
    redundant = df[~df['is_sole']]

    print(f"\n  Total triples: {len(df)}", flush=True)
    print(f"  Unique bQTL: {df[['chr','pos']].drop_duplicates().shape[0]}", flush=True)
    print(f"  Unique genes: {df['gene_id'].nunique()}", flush=True)
    print(f"  TF families: {df['tf_family'].nunique()}", flush=True)
    print(f"  Sole motifs: {len(sole)} ({100*len(sole)/len(df):.1f}%)", flush=True)
    print(f"  Redundant: {len(redundant)} ({100*len(redundant)/len(df):.1f}%)", flush=True)

    # ====================================================
    # A. Overall redundancy vs ASE
    # ====================================================
    print(f"\n--- A. OVERALL REDUNDANCY vs ASE ---", flush=True)
    rho, p_rho = stats.spearmanr(df['n_other_motifs'], df['abs_log2_ase'])
    print(f"  Spearman (n_other vs |log2 ASE|): rho={rho:+.4f}, p={p_rho:.2e}", flush=True)

    print(f"\n  Sole vs Redundant |log2(B73/NAM)|:", flush=True)
    print(f"  Sole:      median = {sole['abs_log2_ase'].median():.4f}, "
          f"mean = {sole['abs_log2_ase'].mean():.4f} (n={len(sole)})", flush=True)
    print(f"  Redundant: median = {redundant['abs_log2_ase'].median():.4f}, "
          f"mean = {redundant['abs_log2_ase'].mean():.4f} (n={len(redundant)})", flush=True)
    if len(sole) > 10 and len(redundant) > 10:
        u, p = stats.mannwhitneyu(sole['abs_log2_ase'], redundant['abs_log2_ase'],
                                   alternative='greater')
        print(f"  Mann-Whitney (sole > redundant): p = {p:.4f}", flush=True)

    # Redundancy bins
    print(f"\n  |log2 ASE| by redundancy level:", flush=True)
    for n in range(10):
        sub = df[df['n_other_motifs'] == n]
        if len(sub) > 20:
            print(f"    {n}: median={sub['abs_log2_ase'].median():.4f}, n={len(sub)}", flush=True)
    high = df[df['n_other_motifs'] >= 10]
    if len(high) > 20:
        print(f"    10+: median={high['abs_log2_ase'].median():.4f}, n={len(high)}", flush=True)

    # ====================================================
    # B. Per-family analysis (THE KEY RESULT)
    # ====================================================
    print(f"\n--- B. PER-FAMILY REDUNDANCY vs ASE ---", flush=True)
    print(f"  {'Family':15s} {'n':>6s} {'%sole':>6s} {'rho':>7s} {'p':>12s} {'sig':>4s} {'direction':>10s}", flush=True)

    family_results = []
    for fam in sorted(df['tf_family'].unique()):
        sub = df[df['tf_family'] == fam]
        if len(sub) < 50:
            continue
        pct_sole = 100 * sub['is_sole'].sum() / len(sub)
        r, p = stats.spearmanr(sub['n_other_motifs'], sub['abs_log2_ase'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        direction = 'Buffered' if r < 0 else 'Sensitive' if r > 0 else ''
        print(f"  {fam:15s} {len(sub):6d} {pct_sole:5.1f}% {r:+7.4f} {p:12.3e} {sig:>4s} {direction:>10s}", flush=True)
        family_results.append({
            'family': fam, 'n': len(sub), 'pct_sole': pct_sole,
            'rho': r, 'p': p, 'direction': direction
        })

    fam_df = pd.DataFrame(family_results)
    fam_df.to_csv(OUTDIR / 'per_family_redundancy.csv', index=False)

    # ====================================================
    # C. Disruption fraction analysis
    # ====================================================
    print(f"\n--- C. DISRUPTION FRACTION ---", flush=True)
    df['disruption_fraction'] = 1.0 / (df['n_other_motifs'] + 1)
    rho_df, p_df = stats.spearmanr(df['disruption_fraction'], df['abs_log2_ase'])
    print(f"  Overall: rho={rho_df:+.4f}, p={p_df:.2e}", flush=True)

    for fam in ['MYB', 'MYB_related', 'Dof', 'WRKY', 'bHLH', 'SBP', 'bZIP']:
        sub = df[df['tf_family'] == fam]
        if len(sub) < 50:
            continue
        r, p = stats.spearmanr(sub['disruption_fraction'], sub['abs_log2_ase'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {fam:15s}: rho={r:+.4f}, p={p:.3e} {sig}", flush=True)

    # ====================================================
    # D. Combinatorial disruption (# TF families per gene)
    # ====================================================
    print(f"\n--- D. COMBINATORIAL DISRUPTION ---", flush=True)

    gene_level = df.groupby('gene_id').agg(
        n_families=('tf_family', 'nunique'),
        n_bqtl=('pos', 'nunique'),
        mean_redundancy=('n_other_motifs', 'mean'),
        max_redundancy=('n_other_motifs', 'max'),
        abs_log2_ase=('abs_log2_ase', 'first'),
    ).reset_index()

    rho_comb, p_comb = stats.spearmanr(gene_level['n_families'], gene_level['abs_log2_ase'])
    print(f"  # TF families disrupted vs |log2 ASE|: rho={rho_comb:+.4f}, p={p_comb:.2e}", flush=True)

    print(f"\n  |log2 ASE| by # families disrupted:", flush=True)
    for n in sorted(gene_level['n_families'].unique()):
        sub = gene_level[gene_level['n_families'] == n]
        if len(sub) > 10:
            print(f"    {n:2d} families: median={sub['abs_log2_ase'].median():.4f}, n={len(sub)}", flush=True)

    # Which families predict ASE when disrupted?
    print(f"\n  Family disruption → ASE magnitude:", flush=True)
    families_present = []
    for fam in sorted(df['tf_family'].unique()):
        genes_with = set(df[df['tf_family'] == fam]['gene_id'])
        genes_without = set(gene_level['gene_id']) - genes_with

        with_ase = gene_level[gene_level['gene_id'].isin(genes_with)]['abs_log2_ase']
        without_ase = gene_level[gene_level['gene_id'].isin(genes_without)]['abs_log2_ase']

        if len(with_ase) > 20 and len(without_ase) > 20:
            u, p = stats.mannwhitneyu(with_ase, without_ase, alternative='two-sided')
            diff = with_ase.median() - without_ase.median()
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            if p < 0.05:
                print(f"    {fam:15s}: with={with_ase.median():.3f} vs without={without_ase.median():.3f} "
                      f"diff={diff:+.3f} p={p:.3e} {sig}", flush=True)
                families_present.append({
                    'family': fam, 'with_median': with_ase.median(),
                    'without_median': without_ase.median(),
                    'diff': diff, 'p': p, 'n_with': len(with_ase), 'n_without': len(without_ase)
                })

    # ====================================================
    # E. By bQTL mechanism type
    # ====================================================
    print(f"\n--- E. BY MECHANISM TYPE ---", flush=True)
    for btype in ['GenoOnly', 'MetOnly', 'GenoP', 'MetP']:
        sub = df[df['bqtl_type'] == btype]
        if len(sub) > 50:
            rho_t, p_t = stats.spearmanr(sub['n_other_motifs'], sub['abs_log2_ase'])
            pct_sole = 100 * sub['is_sole'].sum() / len(sub)
            print(f"  {btype:10s}: n={len(sub):6d}, %sole={pct_sole:.1f}%, "
                  f"rho={rho_t:+.4f}, p={p_t:.3e}", flush=True)

    # ====================================================
    # F. Compound variant analysis
    # ====================================================
    print(f"\n--- F. COMPOUND VARIANT ANALYSIS ---", flush=True)
    # For genes in our analysis, classify ALL bQTL positions relative to gene structure
    # Use the promoter-mapped genes as the gene set
    analysis_genes = set(df['gene_id'].unique())

    # Collect ALL bQTL that fall anywhere in/near these genes
    gene_all_bqtl = defaultdict(set)
    for (chrom, pos), genes in pos_to_genes.items():
        for gene_id in genes:
            if gene_id in analysis_genes:
                gene_all_bqtl[gene_id].add((chrom, pos))

    # Classify positions relative to gene structure
    gene_position_classes = {}
    for gene_id, positions in gene_all_bqtl.items():
        if gene_id not in gene_coords:
            continue
        info = gene_coords[gene_id]
        feats = gene_features.get(gene_id, [])
        classes = set()
        for chrom, pos in positions:
            cls = classify_position(pos, info, feats)
            classes.add(cls)
        gene_position_classes[gene_id] = classes

    # Compare: promoter-only vs compound (promoter + gene body)
    promoter_only_genes = [g for g, c in gene_position_classes.items()
                           if c == {'promoter'} and g in ase_dict]
    compound_genes = [g for g, c in gene_position_classes.items()
                      if 'promoter' in c and len(c) > 1 and g in ase_dict]

    if promoter_only_genes and compound_genes:
        prom_ase = [ase_dict[g] for g in promoter_only_genes]
        comp_ase = [ase_dict[g] for g in compound_genes]
        u, p = stats.mannwhitneyu(comp_ase, prom_ase, alternative='greater')
        print(f"  Promoter-only: median |ASE| = {np.median(prom_ase):.4f} (n={len(prom_ase)})", flush=True)
        print(f"  Compound (promoter + body): median |ASE| = {np.median(comp_ase):.4f} (n={len(comp_ase)})", flush=True)
        print(f"  Mann-Whitney (compound > promoter): p = {p:.4f}", flush=True)

        # Break down by position class
        for cls in ['5UTR', '3UTR', 'CDS', 'intron']:
            cls_genes = [g for g, c in gene_position_classes.items()
                         if 'promoter' in c and cls in c and g in ase_dict]
            if len(cls_genes) > 10:
                cls_ase = [ase_dict[g] for g in cls_genes]
                print(f"    Promoter + {cls}: median={np.median(cls_ase):.4f} (n={len(cls_genes)})", flush=True)

    # ====================================================
    # G. Cross-hybrid consistency
    # ====================================================
    if ASE_HYBRID.exists():
        print(f"\n--- G. CROSS-HYBRID CONSISTENCY ---", flush=True)
        ase_hybrid_df = pd.read_csv(ASE_HYBRID)
        # For each gene, compute ASE variance across hybrids
        gene_var = ase_hybrid_df.groupby('gene_id')['abs_log2_ratio'].agg(['std', 'count'])
        gene_var = gene_var[gene_var['count'] >= 5].reset_index()
        gene_var.columns = ['gene_id', 'ase_std', 'n_hybrids']

        merged = gene_level.merge(gene_var, on='gene_id', how='inner')
        if len(merged) > 50:
            rho_v, p_v = stats.spearmanr(merged['n_families'], merged['ase_std'])
            print(f"  # families disrupted vs ASE std: rho={rho_v:+.4f}, p={p_v:.2e}", flush=True)
            print(f"  (n={len(merged)} genes)", flush=True)

    # ====================================================
    # H. Gene-level summary
    # ====================================================
    gene_level.to_csv(OUTDIR / 'gene_level_modulation.csv', index=False)
    print(f"\n  Saved: gene_level_modulation.csv ({len(gene_level)} genes)", flush=True)

    # ====================================================
    # SAVE SUMMARY
    # ====================================================
    summary = {
        'total_triples': len(df),
        'unique_bqtl': df[['chr','pos']].drop_duplicates().shape[0],
        'unique_genes': df['gene_id'].nunique(),
        'pct_redundant': 100 * len(redundant) / len(df),
        'overall_rho': rho,
        'overall_p': p_rho,
        'combinatorial_rho': rho_comb,
        'combinatorial_p': p_comb,
    }
    pd.DataFrame([summary]).to_csv(OUTDIR / 'analysis_summary.csv', index=False)

    # ====================================================
    # FIGURES
    # ====================================================
    print("\n[6] Generating figures...", flush=True)
    generate_all_figures(df, gene_level, fam_df)
    print("  Done.", flush=True)


def generate_all_figures(df, gene_level, fam_df):
    """Generate publication-quality multi-panel figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats
    import matplotlib.gridspec as gridspec

    # ========================================
    # FIGURE 1: Redundancy overview (3 panels)
    # ========================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Redundancy distribution
    ax = axes[0]
    bins = list(range(21)) + [50]
    counts = df['n_other_motifs'].clip(upper=20)
    ax.hist(counts, bins=range(22), color='steelblue', edgecolor='white', alpha=0.8)
    pct_sole = 100 * (df['is_sole'].sum() / len(df))
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5, label=f'Sole: {pct_sole:.1f}%')
    ax.set_xlabel('Additional same-family motifs (±200bp)', fontsize=11)
    ax.set_ylabel('Count (bQTL-gene-TF triples)', fontsize=11)
    ax.set_title('A. Motif redundancy distribution', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)

    # Panel B: Redundancy vs ASE boxplot
    ax = axes[1]
    data_groups = []
    positions = []
    labels = []
    for n in range(8):
        sub = df[df['n_other_motifs'] == n]
        if len(sub) > 20:
            data_groups.append(sub['abs_log2_ase'].values)
            positions.append(n)
            labels.append(str(n))
    high = df[df['n_other_motifs'] >= 8]
    if len(high) > 20:
        data_groups.append(high['abs_log2_ase'].values)
        positions.append(8)
        labels.append('8+')

    if data_groups:
        bp = ax.boxplot(data_groups, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='darkred', linewidth=2))
        colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(positions)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    rho, p = stats.spearmanr(df['n_other_motifs'], df['abs_log2_ase'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax.set_xlabel('Additional same-family motifs', fontsize=11)
    ax.set_ylabel('|log₂(B73/NAM)|', fontsize=11)
    ax.set_title(f'B. Redundancy vs allelic imbalance\nρ={rho:+.3f} {sig}',
                 fontweight='bold', fontsize=12)
    ax.set_xticklabels(labels)

    # Panel C: Per-family bars
    ax = axes[2]
    fam_sig = fam_df.sort_values('rho')
    colors = ['#d32f2f' if p < 0.001 else '#e57373' if p < 0.01
              else '#ef9a9a' if p < 0.05 else '#e0e0e0'
              for p in fam_sig['p']]
    ax.barh(range(len(fam_sig)), fam_sig['rho'], color=colors,
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(fam_sig)))
    ax.set_yticklabels(fam_sig['family'], fontsize=9)
    ax.set_xlabel('Spearman ρ (redundancy vs |ASE|)', fontsize=11)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('C. Per-family buffering effect\n(darker = more significant)',
                 fontweight='bold', fontsize=12)

    plt.suptitle('bQTL as Quantitative Modulators of Gene Regulation',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTDIR / 'figures' / 'fig1_redundancy_overview.pdf',
                dpi=300, bbox_inches='tight')
    fig.savefig(OUTDIR / 'figures' / 'fig1_redundancy_overview.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================
    # FIGURE 2: Combinatorial disruption (2 panels)
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: # families vs ASE
    ax = axes[0]
    data_groups = []
    positions = []
    for n in sorted(gene_level['n_families'].unique()):
        sub = gene_level[gene_level['n_families'] == n]
        if len(sub) > 10:
            data_groups.append(sub['abs_log2_ase'].values)
            positions.append(n)

    if data_groups:
        bp = ax.boxplot(data_groups, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='darkred', linewidth=2))
        colors = plt.cm.YlOrRd(np.linspace(0.1, 0.8, len(positions)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    rho_c, p_c = stats.spearmanr(gene_level['n_families'], gene_level['abs_log2_ase'])
    sig = '***' if p_c < 0.001 else '**' if p_c < 0.01 else '*' if p_c < 0.05 else 'ns'
    ax.set_xlabel('# TF families with motifs disrupted', fontsize=11)
    ax.set_ylabel('|log₂(B73/NAM)|', fontsize=11)
    ax.set_title(f'A. Combinatorial disruption vs ASE\nρ={rho_c:+.3f} {sig}',
                 fontweight='bold', fontsize=12)

    # Panel B: Two-tier model - buffered vs sensitive families
    ax = axes[1]
    buffered = fam_df[fam_df['rho'] < 0].sort_values('rho')
    sensitive = fam_df[fam_df['rho'] > 0].sort_values('rho', ascending=False)

    all_fams = pd.concat([buffered, sensitive])
    y_pos = range(len(all_fams))
    colors = ['#2e7d32' if r < 0 and p < 0.05 else '#c8e6c9' if r < 0
              else '#c62828' if r > 0 and p < 0.05 else '#ffcdd2'
              for r, p in zip(all_fams['rho'], all_fams['p'])]

    ax.barh(y_pos, all_fams['rho'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_fams['family'], fontsize=9)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Spearman ρ (redundancy vs |ASE|)', fontsize=11)
    ax.set_title('B. Two-tier regulatory architecture\n(green=buffered, red=sensitive)',
                 fontweight='bold', fontsize=12)

    # Add tier labels
    n_buff = len(buffered)
    if n_buff > 0:
        ax.axhspan(-0.5, n_buff - 0.5, alpha=0.05, color='green')
    if len(sensitive) > 0:
        ax.axhspan(n_buff - 0.5, len(all_fams) - 0.5, alpha=0.05, color='red')

    plt.tight_layout()
    fig.savefig(OUTDIR / 'figures' / 'fig2_combinatorial_two_tier.pdf',
                dpi=300, bbox_inches='tight')
    fig.savefig(OUTDIR / 'figures' / 'fig2_combinatorial_two_tier.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================
    # FIGURE 3: Disruption fraction (2 panels)
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Disruption fraction scatter for MYB
    ax = axes[0]
    for fam, color, marker in [('MYB', '#1565c0', 'o'), ('MYB_related', '#42a5f5', 's'),
                                ('Dof', '#66bb6a', '^')]:
        sub = df[df['tf_family'] == fam]
        if len(sub) > 50:
            frac = 1.0 / (sub['n_other_motifs'] + 1)
            jitter = np.random.RandomState(42).normal(0, 0.01, len(sub))
            ax.scatter(frac + jitter, sub['abs_log2_ase'],
                      alpha=0.1, s=10, c=color, label=fam, edgecolor='none')
            # Binned means
            bins = [0, 0.1, 0.2, 0.35, 0.55, 1.01]
            for i in range(len(bins)-1):
                mask = (frac >= bins[i]) & (frac < bins[i+1])
                if mask.sum() > 10:
                    ax.plot((bins[i] + bins[i+1])/2,
                           sub.loc[mask, 'abs_log2_ase'].median(),
                           marker=marker, markersize=10, color=color,
                           markeredgecolor='black', markeredgewidth=1)

    ax.set_xlabel('Disruption fraction (1 / total motifs)', fontsize=11)
    ax.set_ylabel('|log₂(B73/NAM)|', fontsize=11)
    ax.set_title('A. Buffered families: disruption fraction vs ASE',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)

    # Panel B: Disruption fraction scatter for sensitive families
    ax = axes[1]
    for fam, color, marker in [('WRKY', '#c62828', 'o'), ('bHLH', '#e65100', 's'),
                                ('SBP', '#f9a825', '^'), ('bZIP', '#ad1457', 'D')]:
        sub = df[df['tf_family'] == fam]
        if len(sub) > 50:
            frac = 1.0 / (sub['n_other_motifs'] + 1)
            jitter = np.random.RandomState(42).normal(0, 0.01, len(sub))
            ax.scatter(frac + jitter, sub['abs_log2_ase'],
                      alpha=0.1, s=10, c=color, label=fam, edgecolor='none')
            bins = [0, 0.1, 0.2, 0.35, 0.55, 1.01]
            for i in range(len(bins)-1):
                mask = (frac >= bins[i]) & (frac < bins[i+1])
                if mask.sum() > 10:
                    ax.plot((bins[i] + bins[i+1])/2,
                           sub.loc[mask, 'abs_log2_ase'].median(),
                           marker=marker, markersize=10, color=color,
                           markeredgecolor='black', markeredgewidth=1)

    ax.set_xlabel('Disruption fraction (1 / total motifs)', fontsize=11)
    ax.set_ylabel('|log₂(B73/NAM)|', fontsize=11)
    ax.set_title('B. Sensitive families: disruption fraction vs ASE',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTDIR / 'figures' / 'fig3_disruption_fraction.pdf',
                dpi=300, bbox_inches='tight')
    fig.savefig(OUTDIR / 'figures' / 'fig3_disruption_fraction.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print("  Saved: fig1_redundancy_overview.pdf", flush=True)
    print("  Saved: fig2_combinatorial_two_tier.pdf", flush=True)
    print("  Saved: fig3_disruption_fraction.pdf", flush=True)


if __name__ == '__main__':
    main()
