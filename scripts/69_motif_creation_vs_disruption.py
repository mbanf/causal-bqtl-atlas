#!/usr/bin/env python3
"""
69_motif_creation_vs_disruption.py

Bidirectional motif analysis: for each of 728 causal bQTL, compare motif hits
in the B73 reference sequence vs the NAM variant sequence.

- Disrupted: motif present in ref, absent in alt
- Created: motif absent in ref, present in alt
- Unchanged: motif present in both (variant outside core positions)

Uses ±14bp window around variant (max PWM width = 28bp).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
from collections import defaultdict, Counter

BASE = str(Path(__file__).resolve().parent.parent)
RESULTS = f'{BASE}/results'
GENOME = f'{BASE}/data/raw/B73_NAM5.fa'

# ── Load data ──
causal = pd.read_csv(f'{RESULTS}/causal_bqtl_728.csv')
geno = pd.read_csv(f'{BASE}/data/processed/nam_founder_genotypes_at_bqtl.tsv', sep='\t')
print(f"Loaded {len(causal)} causal bQTL")

# Merge to get ref/alt alleles
hybrids = ['B97','CML247','CML277','CML322','CML333','CML69','HP301',
           'Il14H','Ki11','Ki3','Ky21','M162W','Mo18W','Ms71','NC358',
           'Oh43','Oh7B','P39','Tx303']

merged = causal.merge(geno, on=['chr', 'pos'], how='left')

# Extract alt allele for each bQTL
def get_alt_allele(row):
    alts = set()
    for h in hybrids:
        val = str(row.get(h, ''))
        if val not in ['0|0', './.', 'nan', '']:
            alts.add(val)
    if len(alts) == 1:
        return alts.pop()
    elif len(alts) > 1:
        return list(alts)[0]  # Take first (multi-allelic rare)
    return None

merged['alt'] = merged.apply(get_alt_allele, axis=1)
has_alt = merged['alt'].notna().sum()
print(f"bQTL with alt allele: {has_alt}/{len(merged)}")

# Check for multi-allelic
multi = merged.apply(lambda r: len(set(
    str(r.get(h, '')) for h in hybrids
    if str(r.get(h, '')) not in ['0|0', './.', 'nan', '']
)), axis=1)
print(f"Multi-allelic sites: {(multi > 1).sum()}")

# ── Load PWM database ──
with open(f'{BASE}/data/processed/maize_tf_pwm_database.json') as f:
    pwm_db = json.load(f)
motifs = pwm_db['motifs']

pwm_matrices = []
pwm_families = []
pwm_widths = []
for m in motifs:
    try:
        mat = np.array(m['pwm']).T  # (4, width)
        mat = np.clip(mat, 1e-4, 1.0)
        mat = np.log2(mat / 0.25)
        pwm_matrices.append(mat)
        pwm_families.append(m['tf_family'])
        pwm_widths.append(mat.shape[1])
    except:
        pass
print(f"Built {len(pwm_matrices)} PWMs, max width {max(pwm_widths)}")

# ── Load reference genome ──
import pyfaidx
print("Loading genome...")
genome = pyfaidx.Fasta(GENOME)

chr_name_map = {}
for key in genome.keys():
    simple = key.replace('B73-', '').lower()
    chr_name_map[simple] = key
    chr_name_map[key.lower()] = key

def get_sequence(chrom, start, end):
    key = chr_name_map.get(chrom.lower())
    if key is None:
        return None
    try:
        return str(genome[key][start:end]).upper()
    except:
        return None

# ── PWM scanning ──
BASE_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def scan_sequence_families(seq, threshold=6.0):
    """Scan sequence with all PWMs, return dict of family → list of (pos, strand, score, width)."""
    family_hits = defaultdict(list)
    for pwm_mat, fam in zip(pwm_matrices, pwm_families):
        w = pwm_mat.shape[1]
        if len(seq) < w:
            continue
        for i in range(len(seq) - w + 1):
            subseq = seq[i:i+w]
            # Forward
            score = 0
            valid = True
            for j, base in enumerate(subseq):
                idx = BASE_IDX.get(base)
                if idx is None:
                    valid = False
                    break
                score += pwm_mat[idx, j]
            if valid and score >= threshold:
                family_hits[fam].append((i, '+', score, w))
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
                family_hits[fam].append((i, '-', score, w))
    return family_hits

# ── Main analysis ──
print("\n" + "="*70)
print("BIDIRECTIONAL MOTIF ANALYSIS: DISRUPTION vs CREATION")
print("="*70)

FLANK = 28  # max PWM width
threshold = 6.0
results = []

for idx, row in merged.iterrows():
    chrom = row['chr']
    pos = row['pos']
    gene_id = row['gene_id']
    ref_base = row['ref']
    alt_base = row['alt']

    if pd.isna(alt_base) or pd.isna(ref_base):
        continue

    # Extract ±FLANK around variant (pos is 1-based, pyfaidx is 0-based)
    var_pos_0 = pos - 1  # convert to 0-based
    start = var_pos_0 - FLANK
    end = var_pos_0 + FLANK + 1
    ref_seq = get_sequence(chrom, start, end)
    if ref_seq is None or len(ref_seq) < 2 * FLANK + 1:
        continue

    # Verify ref base matches
    var_idx = FLANK  # position of variant in extracted sequence
    genome_base = ref_seq[var_idx]
    if genome_base.upper() != ref_base.upper():
        continue

    # Create alt sequence
    alt_seq = ref_seq[:var_idx] + alt_base + ref_seq[var_idx+1:]

    # Scan both sequences
    ref_hits = scan_sequence_families(ref_seq, threshold)
    alt_hits = scan_sequence_families(alt_seq, threshold)

    # For each family, check hits that OVERLAP the variant position
    disrupted_families = []
    created_families = []
    unchanged_families = []

    all_families = set(list(ref_hits.keys()) + list(alt_hits.keys()))

    for fam in all_families:
        # Hits overlapping variant in ref
        ref_at_var = [(p, s, sc, w) for p, s, sc, w in ref_hits.get(fam, [])
                      if p <= var_idx < p + w]
        # Hits overlapping variant in alt
        alt_at_var = [(p, s, sc, w) for p, s, sc, w in alt_hits.get(fam, [])
                      if p <= var_idx < p + w]

        has_ref = len(ref_at_var) > 0
        has_alt = len(alt_at_var) > 0

        if has_ref and not has_alt:
            disrupted_families.append(fam)
        elif not has_ref and has_alt:
            created_families.append(fam)
        elif has_ref and has_alt:
            unchanged_families.append(fam)

    results.append({
        'gene_id': gene_id,
        'chr': chrom,
        'pos': pos,
        'ref': ref_base,
        'alt': alt_base,
        'n_disrupted': len(disrupted_families),
        'disrupted_families': ','.join(sorted(disrupted_families)),
        'n_created': len(created_families),
        'created_families': ','.join(sorted(created_families)),
        'n_unchanged': len(unchanged_families),
        'unchanged_families': ','.join(sorted(unchanged_families)),
        'abs_d': row['abs_d'],
        'cohens_d': row['cohens_d'],
    })

    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx+1}/{len(merged)}...")

res_df = pd.DataFrame(results)
print(f"\nAnalyzed: {len(res_df)} bQTL")

# ── Summary ──
print(f"\n{'='*70}")
print(f"RESULTS: Motif disruption vs creation at 728 causal bQTL")
print(f"{'='*70}")

has_any = res_df[(res_df['n_disrupted'] > 0) | (res_df['n_created'] > 0)]
only_disrupted = res_df[(res_df['n_disrupted'] > 0) & (res_df['n_created'] == 0)]
only_created = res_df[(res_df['n_created'] > 0) & (res_df['n_disrupted'] == 0)]
both = res_df[(res_df['n_disrupted'] > 0) & (res_df['n_created'] > 0)]
neither = res_df[(res_df['n_disrupted'] == 0) & (res_df['n_created'] == 0)]

print(f"\nTotal bQTL analyzed: {len(res_df)}")
print(f"Any motif change: {len(has_any)} ({100*len(has_any)/len(res_df):.1f}%)")
print(f"  Only disrupted: {len(only_disrupted)} ({100*len(only_disrupted)/len(res_df):.1f}%)")
print(f"  Only created:   {len(only_created)} ({100*len(only_created)/len(res_df):.1f}%)")
print(f"  Both:           {len(both)} ({100*len(both)/len(res_df):.1f}%)")
print(f"  Neither:        {len(neither)} ({100*len(neither)/len(res_df):.1f}%)")

# Per-family counts
print(f"\n--- Disrupted families (total instances) ---")
disrupted_counts = Counter()
for fams in res_df['disrupted_families']:
    for f in str(fams).split(','):
        if f and f != 'nan':
            disrupted_counts[f] += 1

created_counts = Counter()
for fams in res_df['created_families']:
    for f in str(fams).split(','):
        if f and f != 'nan':
            created_counts[f] += 1

all_fams = sorted(set(list(disrupted_counts.keys()) + list(created_counts.keys())))
print(f"\n{'Family':<16} {'Disrupted':>10} {'Created':>10} {'Net':>10} {'Direction':>12}")
print("-" * 62)
total_d = 0
total_c = 0
for fam in all_fams:
    d = disrupted_counts.get(fam, 0)
    c = created_counts.get(fam, 0)
    total_d += d
    total_c += c
    net = c - d
    direction = "CREATION" if net > 0 else ("DISRUPTION" if net < 0 else "BALANCED")
    if d + c >= 5:
        print(f"  {fam:<14} {d:>10} {c:>10} {net:>+10} {direction:>12}")
print("-" * 62)
print(f"  {'TOTAL':<14} {total_d:>10} {total_c:>10} {total_c-total_d:>+10}")

# Effect sizes
print(f"\n--- Effect sizes by category ---")
from scipy import stats

for label, subset in [('Only disrupted', only_disrupted),
                       ('Only created', only_created),
                       ('Both', both)]:
    if len(subset) > 0:
        print(f"  {label}: n={len(subset)}, median |d|={subset['abs_d'].median():.2f}")

if len(only_disrupted) > 5 and len(only_created) > 5:
    stat, p = stats.mannwhitneyu(only_disrupted['abs_d'], only_created['abs_d'],
                                  alternative='two-sided')
    print(f"\n  Disrupted vs Created effect size: Mann-Whitney p = {p:.4f}")

# Direction of Cohen's d
print(f"\n--- Direction of expression change ---")
for label, subset in [('Only disrupted', only_disrupted),
                       ('Only created', only_created),
                       ('Both', both)]:
    if len(subset) > 0:
        pos_d = (subset['cohens_d'] > 0).sum()
        neg_d = (subset['cohens_d'] < 0).sum()
        print(f"  {label}: {pos_d} positive d, {neg_d} negative d ({100*pos_d/len(subset):.0f}%/{100*neg_d/len(subset):.0f}%)")

# Save
res_df.to_csv(f'{RESULTS}/motif_creation_vs_disruption_728.csv', index=False)
print(f"\nSaved: motif_creation_vs_disruption_728.csv")
