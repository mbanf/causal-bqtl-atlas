#!/usr/bin/env python3
"""
68_verify_sole_motif.py

Critical verification: For each of the 728 causal bQTL, within the actual
MOA-seq binding peak, is the disrupted TF family's motif the SOLE copy?

Steps:
1. Load actual narrowPeak files per hybrid
2. For each bQTL, find the tightest peak containing it
3. Extract peak sequence from reference genome
4. Scan with all 259 PWMs (threshold=6.0)
5. For each TF family at the variant: count OTHER copies of that family in the peak
6. Report: sole copy fraction per family
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict, Counter
import pyfaidx

BASE = str(Path(__file__).resolve().parent.parent)
RESULTS = f'{BASE}/results'
PEAK_DIR = f'{BASE}/data/raw/engelhorn_peaks'
GENOME = f'{BASE}/data/raw/B73_NAM5.fa'

# ── Load data ──
causal = pd.read_csv(f'{RESULTS}/causal_bqtl_728.csv')
print(f"Loaded {len(causal)} causal bQTL")

# ── Load PWM database ──
with open(f'{BASE}/data/processed/maize_tf_pwm_database.json') as f:
    pwm_db = json.load(f)
motifs = pwm_db['motifs']
print(f"PWM database: {len(motifs)} motifs")

# Build PWM matrices
def build_pwm_matrix(pwm_entry):
    """Convert PWM list-of-lists to numpy log-odds matrix."""
    pwm = pwm_entry['pwm']  # list of [A, C, G, T] per position
    mat = np.array(pwm).T  # shape (4, width)
    # Convert frequencies to log-odds
    mat = np.clip(mat, 1e-4, 1.0)
    mat = np.log2(mat / 0.25)
    return mat

pwm_matrices = []
pwm_families = []
for m in motifs:
    try:
        mat = build_pwm_matrix(m)
        pwm_matrices.append(mat)
        pwm_families.append(m['tf_family'])
    except:
        pass
print(f"Built {len(pwm_matrices)} PWM matrices")
print(f"Families: {len(set(pwm_families))}")

# ── Load peaks for WW condition ──
def load_peaks(hybrid):
    """Load narrowPeak file, return list of (chr, start, end)."""
    # Try WW first
    path = f'{PEAK_DIR}/{hybrid}.w2_4.q3_peaks.narrowPeak'
    if not os.path.exists(path):
        return []
    peaks = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            chrom = parts[0].replace('B73-', '')  # B73-chr1 → chr1
            start = int(parts[1])
            end = int(parts[2])
            peaks.append((chrom, start, end))
    return peaks

# Load peaks for all hybrids
print("\nLoading peaks...")
hybrids = ['B97','CML247','CML277','CML322','CML333','CML69','HP301',
           'IL14H','Ki11','Ki3','Ky21','M162W','M37W','Mo18W','Ms71',
           'NC358','Oh43','Oh7b','P39','Tx303']

all_peaks = {}
for hyb in hybrids:
    peaks = load_peaks(hyb)
    all_peaks[hyb] = peaks
    # Also add non-genotyped hybrids
for hyb in ['A188','A619','CML103','Mo17','W22']:
    peaks = load_peaks(hyb)
    all_peaks[hyb] = peaks

total_peaks = sum(len(v) for v in all_peaks.values())
print(f"Loaded {total_peaks} peaks across {len(all_peaks)} hybrids")

# ── Load reference genome ──
print("Loading genome...")
genome = pyfaidx.Fasta(GENOME)
print(f"Genome loaded: {len(genome.keys())} sequences")

# Map chromosome names
chr_name_map = {}
for key in genome.keys():
    # Map chr1 → actual fasta key
    simple = key.replace('B73-', '').lower()
    chr_name_map[simple] = key
    chr_name_map[key.lower()] = key

def get_sequence(chrom, start, end):
    """Get sequence from reference genome."""
    key = chr_name_map.get(chrom.lower())
    if key is None:
        return None
    try:
        seq = str(genome[key][start:end])
        return seq.upper()
    except:
        return None

# ── PWM scanning ──
BASE_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def scan_sequence(seq, pwm_mat, threshold=6.0):
    """Scan sequence with PWM, return list of (position, strand, score)."""
    hits = []
    w = pwm_mat.shape[1]
    if len(seq) < w:
        return hits

    for i in range(len(seq) - w + 1):
        subseq = seq[i:i+w]
        # Forward strand
        score = 0
        valid = True
        for j, base in enumerate(subseq):
            idx = BASE_IDX.get(base)
            if idx is None:
                valid = False
                break
            score += pwm_mat[idx, j]
        if valid and score >= threshold:
            hits.append((i, '+', score))

        # Reverse complement
        rc = subseq[::-1].translate(str.maketrans('ACGT', 'TGCA'))
        score = 0
        valid = True
        for j, base in enumerate(rc):
            idx = BASE_IDX.get(base)
            if idx is None:
                valid = False
                break
            score += pwm_mat[idx, j]
        if valid and score >= threshold:
            hits.append((i, '-', score))

    return hits

# ── Main analysis ──
print("\n" + "="*70)
print("SCANNING 728 bQTL WITHIN ACTUAL MOA-SEQ PEAKS")
print("="*70)

results = []
threshold = 6.0

for idx, row in causal.iterrows():
    chrom = row['chr']
    pos = row['pos']
    gene_id = row['gene_id']

    # Find the tightest peak containing this position across all hybrids
    best_peak = None
    best_width = float('inf')
    peak_hybrids = []

    for hyb, peaks in all_peaks.items():
        for (pc, ps, pe) in peaks:
            if pc == chrom and ps <= pos <= pe:
                width = pe - ps
                peak_hybrids.append(hyb)
                if width < best_width:
                    best_width = width
                    best_peak = (pc, ps, pe)

    if best_peak is None:
        results.append({
            'gene_id': gene_id, 'chr': chrom, 'pos': pos,
            'in_peak': False, 'peak_width': 0,
            'n_hybrids_in_peak': 0
        })
        continue

    peak_chr, peak_start, peak_end = best_peak
    peak_seq = get_sequence(peak_chr, peak_start, peak_end)

    if peak_seq is None or len(peak_seq) < 10:
        results.append({
            'gene_id': gene_id, 'chr': chrom, 'pos': pos,
            'in_peak': True, 'peak_width': best_width,
            'n_hybrids_in_peak': len(set(peak_hybrids)),
            'error': 'no_sequence'
        })
        continue

    # Position of variant within peak
    var_pos_in_peak = pos - peak_start

    # Scan all PWMs within the peak
    family_hits = defaultdict(list)  # family → [(pos_in_peak, strand, score)]
    family_at_variant = defaultdict(list)  # family → hits overlapping variant

    for mi, (pwm_mat, fam) in enumerate(zip(pwm_matrices, pwm_families)):
        hits = scan_sequence(peak_seq, pwm_mat, threshold)
        w = pwm_mat.shape[1]
        for (hit_pos, strand, score) in hits:
            family_hits[fam].append((hit_pos, strand, score, w))
            # Does this hit overlap the variant position?
            if hit_pos <= var_pos_in_peak < hit_pos + w:
                family_at_variant[fam].append((hit_pos, strand, score, w))

    # For each family at the variant: count OTHER copies in the peak
    variant_families = list(family_at_variant.keys())
    sole_copy_families = []
    redundant_families = []

    for fam in variant_families:
        total_hits = len(family_hits[fam])
        hits_at_var = len(family_at_variant[fam])
        other_copies = total_hits - hits_at_var

        if other_copies == 0:
            sole_copy_families.append(fam)
        else:
            redundant_families.append((fam, other_copies))

    # Summary for this bQTL
    has_sole = len(sole_copy_families) > 0
    all_sole = len(redundant_families) == 0 and len(variant_families) > 0

    results.append({
        'gene_id': gene_id,
        'chr': chrom,
        'pos': pos,
        'in_peak': True,
        'peak_width': best_width,
        'n_hybrids_in_peak': len(set(peak_hybrids)),
        'n_families_at_variant': len(variant_families),
        'variant_families': ','.join(variant_families),
        'n_sole_copy': len(sole_copy_families),
        'sole_families': ','.join(sole_copy_families),
        'n_redundant': len(redundant_families),
        'redundant_families': ','.join(f"{f}(+{n})" for f, n in redundant_families),
        'has_any_sole': has_sole,
        'all_sole': all_sole,
        'total_motifs_in_peak': sum(len(v) for v in family_hits.values()),
        'n_families_in_peak': len(family_hits),
    })

    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx+1}/728...")

res_df = pd.DataFrame(results)
print(f"\nDone. Results: {len(res_df)} bQTL")

# ── Summary statistics ──
in_peak = res_df[res_df['in_peak'] == True]
has_motif = in_peak[in_peak['n_families_at_variant'] > 0]

print(f"\n{'='*70}")
print(f"RESULTS: Motif redundancy within actual MOA-seq peaks")
print(f"{'='*70}")
print(f"Total bQTL: {len(res_df)}")
print(f"In at least one peak: {len(in_peak)} ({100*len(in_peak)/len(res_df):.1f}%)")
print(f"With TF motif at variant: {len(has_motif)} ({100*len(has_motif)/len(in_peak):.1f}%)")

print(f"\n--- Per-bQTL: at least one family is sole copy ---")
any_sole = has_motif['has_any_sole'].sum()
print(f"At least one sole-copy family: {any_sole} / {len(has_motif)} = {100*any_sole/len(has_motif):.1f}%")

print(f"\n--- Per-bQTL: ALL families at variant are sole copy ---")
all_sole = has_motif['all_sole'].sum()
print(f"All families sole copy: {all_sole} / {len(has_motif)} = {100*all_sole/len(has_motif):.1f}%")

print(f"\n--- Per-family level (across all bQTL) ---")
total_family_instances = 0
sole_family_instances = 0
family_sole_counts = Counter()
family_total_counts = Counter()

for _, row in has_motif.iterrows():
    for fam in str(row.get('sole_families', '')).split(','):
        if fam:
            sole_family_instances += 1
            total_family_instances += 1
            family_sole_counts[fam] += 1
            family_total_counts[fam] += 1
    for entry in str(row.get('redundant_families', '')).split(','):
        if entry and '(' in entry:
            fam = entry.split('(')[0]
            total_family_instances += 1
            family_total_counts[fam] += 1

print(f"Total family-at-variant instances: {total_family_instances}")
print(f"Sole copy instances: {sole_family_instances} ({100*sole_family_instances/max(total_family_instances,1):.1f}%)")
print(f"Redundant instances: {total_family_instances - sole_family_instances}")

print(f"\n--- Per-family sole-copy rates ---")
for fam in sorted(family_total_counts.keys()):
    total = family_total_counts[fam]
    sole = family_sole_counts.get(fam, 0)
    if total >= 5:
        print(f"  {fam}: {sole}/{total} sole ({100*sole/total:.0f}%)")

print(f"\n--- Peak width statistics ---")
print(f"Median peak width: {in_peak['peak_width'].median():.0f} bp")
print(f"Mean peak width: {in_peak['peak_width'].mean():.0f} bp")
print(f"Median motifs per peak: {has_motif['total_motifs_in_peak'].median():.0f}")
print(f"Median families per peak: {has_motif['n_families_in_peak'].median():.0f}")

# ── Effect size comparison: sole vs redundant ──
print(f"\n--- Effect size: sole-copy vs redundant bQTL ---")
sole_d = causal.set_index('gene_id').loc[
    has_motif[has_motif['all_sole']]['gene_id'], 'abs_d'
].dropna()
red_d = causal.set_index('gene_id').loc[
    has_motif[~has_motif['all_sole']]['gene_id'], 'abs_d'
].dropna()
from scipy import stats
if len(sole_d) > 5 and len(red_d) > 5:
    stat, p = stats.mannwhitneyu(sole_d, red_d, alternative='two-sided')
    print(f"All-sole: median |d| = {sole_d.median():.2f} (n={len(sole_d)})")
    print(f"Has redundant: median |d| = {red_d.median():.2f} (n={len(red_d)})")
    print(f"Mann-Whitney p = {p:.4f}")

# Save
res_df.to_csv(f'{RESULTS}/sole_motif_verification_728.csv', index=False)
print(f"\nSaved: sole_motif_verification_728.csv")
