#!/usr/bin/env python3
"""
Bidirectional motif scan: for each bQTL, classify whether the variant
is more likely to DISRUPT or CREATE TF binding motifs.

Approach:
1. Extract B73 reference sequence around each bQTL (±28bp)
2. Scan reference for motifs → "disruption potential"
3. For each of 3 possible alt bases, substitute and rescan → "creation potential"
4. Classify: net disruption, net creation, or neutral
5. Link to ASE to test if creation vs disruption has different allelic effects

Key insight: without knowing the actual NAM allele, we test all 3 alternatives.
If 2/3 alternatives create a motif, the position is more likely a creation site.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
GENOME = DATA / "raw" / "B73_NAM5.fa"
PWM_DB = DATA / "processed" / "maize_tf_pwm_database.json"
BQTL = DATA / "processed" / "bqtl_snp_ww.csv"
ASE = DATA / "processed" / "engelhorn_ase_ww_gene_summary.csv"
OUTDIR = BASE / "results"

PWM_THRESHOLD = 6.0
VARIANT_WINDOW = 28
BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
IDX_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
BG = np.array([0.2642, 0.2358, 0.2358, 0.2642])
BASES = ['A', 'C', 'G', 'T']


def compile_pwm(pwm_list):
    """Precompile log-odds matrices grouped by width."""
    by_width = defaultdict(list)
    for entry in pwm_list:
        mat = np.array(entry['pwm'], dtype=np.float64)
        mat = np.clip(mat, 1e-4, None)
        lo = np.log2(mat / BG[np.newaxis, :])
        rc = lo[::-1, ::-1].copy()
        by_width[len(mat)].append((lo, rc, entry.get('tf_id', '')))
    compiled = {}
    for w, triples in by_width.items():
        fwd_stack = np.array([t[0] for t in triples])
        rc_stack = np.array([t[1] for t in triples])
        compiled[w] = (fwd_stack, rc_stack)
    return compiled


def count_hits(seq_idx, compiled, threshold=6.0):
    """Count total motif hits in encoded sequence."""
    L = len(seq_idx)
    total = 0
    for w, (fwd_stack, rc_stack) in compiled.items():
        if L < w:
            continue
        n_pwms = fwd_stack.shape[0]
        n_windows = L - w + 1
        windows = np.lib.stride_tricks.as_strided(
            seq_idx, shape=(n_windows, w),
            strides=(seq_idx.strides[0], seq_idx.strides[0])
        ).copy()
        valid = np.all(windows >= 0, axis=1)
        if not np.any(valid):
            continue
        valid_windows = windows[valid]
        n_valid = valid_windows.shape[0]
        pos_idx = np.arange(w)

        fwd_scores = np.sum(
            fwd_stack[np.arange(n_pwms)[:, None, None],
                      pos_idx[None, None, :],
                      valid_windows[None, :, :]],
            axis=2)
        rc_scores = np.sum(
            rc_stack[np.arange(n_pwms)[:, None, None],
                     pos_idx[None, None, :],
                     valid_windows[None, :, :]],
            axis=2)
        hits = np.any((fwd_scores >= threshold) | (rc_scores >= threshold), axis=0)
        total += np.sum(hits)
    return total


def seq_to_idx(seq):
    return np.array([BASE_MAP.get(b, -1) for b in seq], dtype=np.int8)


def process_chromosome(args):
    """Process bQTL positions on one chromosome: ref scan + 3 alt substitutions."""
    chrom, positions, family_compiled_data, genome_path = args

    import pyfaidx
    genome = pyfaidx.Fasta(str(genome_path))

    # Reconstruct compiled PWMs
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
        # Extract window around variant
        win_start = max(0, pos - VARIANT_WINDOW - 1)
        win_end = min(chrom_len, pos + VARIANT_WINDOW)
        ref_seq_str = str(genome[chrom_key][win_start:win_end]).upper()
        ref_seq = seq_to_idx(ref_seq_str)

        # Position of the variant within the window
        var_idx = pos - 1 - win_start
        if var_idx < 0 or var_idx >= len(ref_seq):
            continue

        ref_base = ref_seq[var_idx]
        if ref_base < 0:  # N base
            continue

        # Scan reference for each family
        for fam, compiled in family_compiled.items():
            ref_hits = count_hits(ref_seq, compiled, PWM_THRESHOLD)

            # Try all 3 alternative bases
            alt_hits_list = []
            for alt_base in range(4):
                if alt_base == ref_base:
                    continue
                alt_seq = ref_seq.copy()
                alt_seq[var_idx] = alt_base
                alt_hits = count_hits(alt_seq, compiled, PWM_THRESHOLD)
                alt_hits_list.append(alt_hits)

            # Classify
            mean_alt_hits = np.mean(alt_hits_list)
            max_alt_hits = max(alt_hits_list)
            min_alt_hits = min(alt_hits_list)

            # Net effect: positive = alt creates more motifs, negative = alt disrupts
            net_effect = mean_alt_hits - ref_hits

            # Classification
            if ref_hits > 0 and mean_alt_hits < ref_hits:
                direction = 'disruption'
            elif ref_hits < mean_alt_hits:
                direction = 'creation'
            elif ref_hits == 0 and mean_alt_hits == 0:
                direction = 'neutral'
            else:
                direction = 'mixed'

            # How many alts create vs disrupt?
            n_create = sum(1 for h in alt_hits_list if h > ref_hits)
            n_disrupt = sum(1 for h in alt_hits_list if h < ref_hits)
            n_neutral = sum(1 for h in alt_hits_list if h == ref_hits)

            results.append({
                'chr': chrom,
                'pos': pos,
                'ref_base': IDX_BASE[ref_base],
                'tf_family': fam,
                'ref_hits': ref_hits,
                'mean_alt_hits': mean_alt_hits,
                'max_alt_hits': max_alt_hits,
                'min_alt_hits': min_alt_hits,
                'net_effect': net_effect,
                'direction': direction,
                'n_create': n_create,
                'n_disrupt': n_disrupt,
                'n_neutral': n_neutral,
            })

    return results


def main():
    print("=" * 70, flush=True)
    print("BIDIRECTIONAL MOTIF SCAN: DISRUPTION vs CREATION", flush=True)
    print("=" * 70, flush=True)

    # Load data
    print("\n[1] Loading data...", flush=True)
    bqtl_df = pd.read_csv(BQTL)
    print(f"  {len(bqtl_df)} bQTL total", flush=True)

    ase_df = pd.read_csv(ASE)
    ase_dict = dict(zip(ase_df['gene_id'], ase_df['mean_abs_log2']))

    with open(PWM_DB) as f:
        db = json.load(f)

    # Use top 12 families for speed
    family_pwms_raw = defaultdict(list)
    for m in db['motifs']:
        family_pwms_raw[m['tf_family']].append(m)
    TOP_FAMILIES = ['MYB', 'MYB_related', 'Dof', 'SBP', 'bHLH', 'bZIP',
                    'NAC', 'ARF', 'ERF', 'WRKY', 'HD-ZIP', 'C2H2']

    family_compiled_serial = {}
    for fam in TOP_FAMILIES:
        if fam in family_pwms_raw:
            compiled = compile_pwm(family_pwms_raw[fam])
            serial = {}
            for w, (fwd, rc) in compiled.items():
                serial[str(w)] = (fwd.tolist(), rc.tolist())
            family_compiled_serial[fam] = serial

    # Subsample for computational feasibility (3 alt scans per position per family)
    # Full scan = 147K × 12 families × 4 scans each = too slow
    # Use 10K random bQTL
    MAX_POS = 10000
    rng = np.random.RandomState(42)
    if len(bqtl_df) > MAX_POS:
        sample_idx = rng.choice(len(bqtl_df), MAX_POS, replace=False)
        sample_df = bqtl_df.iloc[sample_idx]
    else:
        sample_df = bqtl_df
    print(f"  Sampling {len(sample_df)} bQTL for bidirectional scan", flush=True)

    # Group by chromosome
    chrom_positions = defaultdict(list)
    for _, row in sample_df.iterrows():
        chrom_positions[row['chr']].append(row['pos'])

    # Prepare worker args
    worker_args = []
    for chrom in sorted(chrom_positions.keys()):
        worker_args.append((
            chrom, chrom_positions[chrom],
            family_compiled_serial, str(GENOME)
        ))

    # Run multiprocessing
    print(f"\n[2] Scanning {len(sample_df)} positions × {len(TOP_FAMILIES)} families × 4 alleles...", flush=True)
    n_procs = min(cpu_count(), 10)
    all_results = []
    with Pool(n_procs) as pool:
        for chunk in pool.imap_unordered(process_chromosome, worker_args):
            all_results.extend(chunk)
            print(f"    Chunk done: {len(chunk)} entries", flush=True)

    print(f"  Total: {len(all_results)} position-family entries", flush=True)

    res_df = pd.DataFrame(all_results)
    res_df.to_csv(OUTDIR / 'bidirectional_motif_scan.csv', index=False)
    print(f"  Saved: bidirectional_motif_scan.csv", flush=True)

    # Analysis
    from scipy import stats

    print("\n" + "=" * 70, flush=True)
    print("RESULTS", flush=True)
    print("=" * 70, flush=True)

    # Overall direction breakdown
    print(f"\n--- Overall direction at bQTL positions ---", flush=True)
    for direction in ['disruption', 'creation', 'mixed', 'neutral']:
        n = (res_df['direction'] == direction).sum()
        pct = 100 * n / len(res_df)
        print(f"  {direction:12s}: {n:7d} ({pct:5.1f}%)", flush=True)

    # Per family
    print(f"\n--- Per-family direction breakdown ---", flush=True)
    print(f"  {'Family':15s} {'Disruption':>10s} {'Creation':>10s} {'Mixed':>10s} {'Neutral':>10s} {'%Disrupt':>8s} {'%Create':>8s}", flush=True)

    family_summary = []
    for fam in TOP_FAMILIES:
        sub = res_df[res_df['tf_family'] == fam]
        if len(sub) == 0:
            continue
        n_dis = (sub['direction'] == 'disruption').sum()
        n_cre = (sub['direction'] == 'creation').sum()
        n_mix = (sub['direction'] == 'mixed').sum()
        n_neu = (sub['direction'] == 'neutral').sum()
        pct_dis = 100 * n_dis / len(sub)
        pct_cre = 100 * n_cre / len(sub)
        print(f"  {fam:15s} {n_dis:10d} {n_cre:10d} {n_mix:10d} {n_neu:10d} {pct_dis:7.1f}% {pct_cre:7.1f}%", flush=True)
        family_summary.append({
            'family': fam, 'n_disruption': n_dis, 'n_creation': n_cre,
            'n_mixed': n_mix, 'n_neutral': n_neu,
            'pct_disruption': pct_dis, 'pct_creation': pct_cre,
            'ratio': n_dis / max(n_cre, 1),
        })

    # Key question: is disruption or creation more common?
    total_dis = (res_df['direction'] == 'disruption').sum()
    total_cre = (res_df['direction'] == 'creation').sum()
    print(f"\n  Overall disruption:creation ratio = {total_dis}:{total_cre} = {total_dis/max(total_cre,1):.2f}", flush=True)

    # Net effect distribution
    print(f"\n--- Net effect (mean_alt - ref hits) ---", flush=True)
    print(f"  Mean net effect: {res_df['net_effect'].mean():+.3f}", flush=True)
    print(f"  Median net effect: {res_df['net_effect'].median():+.3f}", flush=True)
    print(f"  % positions where any alt creates motif: "
          f"{100*(res_df['n_create'] > 0).sum()/len(res_df):.1f}%", flush=True)
    print(f"  % positions where any alt disrupts motif: "
          f"{100*(res_df['n_disrupt'] > 0).sum()/len(res_df):.1f}%", flush=True)

    # Per-family: are buffered vs sensitive families different in disruption:creation ratio?
    print(f"\n--- Buffered vs Sensitive: disruption:creation ratio ---", flush=True)
    buffered = ['MYB', 'MYB_related', 'Dof', 'HD-ZIP']
    sensitive = ['SBP', 'bHLH', 'bZIP', 'NAC', 'ARF', 'ERF']

    for tier_name, tier_fams in [('Buffered', buffered), ('Sensitive', sensitive)]:
        tier_data = res_df[res_df['tf_family'].isin(tier_fams)]
        n_d = (tier_data['direction'] == 'disruption').sum()
        n_c = (tier_data['direction'] == 'creation').sum()
        ratio = n_d / max(n_c, 1)
        print(f"  {tier_name:10s}: disruption={n_d}, creation={n_c}, ratio={ratio:.2f}", flush=True)

    # Generate figure
    print("\n[3] Generating figure...", flush=True)
    plot_bidirectional(res_df, TOP_FAMILIES, family_summary)
    print("  Done.", flush=True)


def plot_bidirectional(df, families, family_summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Overall pie chart
    ax = axes[0]
    directions = ['disruption', 'creation', 'mixed', 'neutral']
    counts = [df[df['direction'] == d].shape[0] for d in directions]
    colors = ['#c62828', '#2e7d32', '#f9a825', '#9e9e9e']
    labels = [f'{d}\n({c:,})' for d, c in zip(directions, counts)]
    ax.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 9})
    ax.set_title('A. Direction of motif effect at bQTL', fontweight='bold')

    # Panel B: Per-family disruption:creation ratio
    ax = axes[1]
    fam_df = pd.DataFrame(family_summary).sort_values('ratio')
    y = range(len(fam_df))
    # Stacked bar: disruption vs creation
    ax.barh(y, fam_df['pct_disruption'], color='#c62828', alpha=0.8, label='Disruption')
    ax.barh(y, -fam_df['pct_creation'], color='#2e7d32', alpha=0.8, label='Creation')
    ax.set_yticks(y)
    ax.set_yticklabels(fam_df['family'], fontsize=9)
    ax.set_xlabel('% of bQTL positions', fontsize=11)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.legend(fontsize=9)
    ax.set_title('B. Per-family disruption vs creation', fontweight='bold')

    # Panel C: Net effect histogram
    ax = axes[2]
    # Filter to positions with non-zero effects
    nonzero = df[df['net_effect'] != 0]['net_effect']
    ax.hist(nonzero, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linewidth=2, linestyle='--')
    ax.set_xlabel('Net effect (alt - ref hits)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    mean_ne = nonzero.mean()
    ax.set_title(f'C. Net motif effect distribution\nmean = {mean_ne:+.3f}',
                 fontweight='bold')

    plt.suptitle('Bidirectional Motif Analysis: Do bQTL Disrupt or Create TF Binding?',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUTDIR / 'figures' / 'fig4_bidirectional.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUTDIR / 'figures' / 'fig4_bidirectional.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig4_bidirectional.pdf", flush=True)


if __name__ == '__main__':
    main()
