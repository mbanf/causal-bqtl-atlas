#!/usr/bin/env python3
"""
72_threshold_sensitivity.py

Sweep PWM threshold from 4.0 to 8.0 and check stability of key findings:
- % disrupted, created, rewired, neither
- Zero self-switches
- Equal effect sizes across categories
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
from collections import defaultdict, Counter
import pyfaidx

BASE = str(Path(__file__).resolve().parent.parent)
RESULTS = f'{BASE}/results'

# ── Load data ──
causal = pd.read_csv(f'{RESULTS}/causal_bqtl_728.csv')
geno = pd.read_csv(f'{BASE}/data/processed/nam_founder_genotypes_at_bqtl.tsv', sep='\t')

hybrids = ['B97','CML247','CML277','CML322','CML333','CML69','HP301',
           'Il14H','Ki11','Ki3','Ky21','M162W','Mo18W','Ms71','NC358',
           'Oh43','Oh7B','P39','Tx303']

merged = causal.merge(geno, on=['chr', 'pos'], how='left')

def get_alt_allele(row):
    alts = set()
    for h in hybrids:
        val = str(row.get(h, ''))
        if val not in ['0|0', './.', 'nan', '']:
            alts.add(val)
    return list(alts)[0] if alts else None

merged['alt'] = merged.apply(get_alt_allele, axis=1)

# ── Load PWMs ──
with open(f'{BASE}/data/processed/maize_tf_pwm_database.json') as f:
    pwm_db = json.load(f)

pwm_matrices = []
pwm_families = []
for m in pwm_db['motifs']:
    try:
        mat = np.array(m['pwm']).T
        mat = np.clip(mat, 1e-4, 1.0)
        mat = np.log2(mat / 0.25)
        pwm_matrices.append(mat)
        pwm_families.append(m['tf_family'])
    except:
        pass

# ── Load genome ──
genome = pyfaidx.Fasta(f'{BASE}/data/raw/B73_NAM5.fa')
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

BASE_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
FLANK = 28

def scan_families(seq, threshold):
    family_hits = defaultdict(list)
    for pwm_mat, fam in zip(pwm_matrices, pwm_families):
        w = pwm_mat.shape[1]
        if len(seq) < w:
            continue
        for i in range(len(seq) - w + 1):
            subseq = seq[i:i+w]
            for strand_seq, strand in [(subseq, '+'),
                                        (subseq[::-1].translate(str.maketrans('ACGT', 'TGCA')), '-')]:
                score = 0
                valid = True
                for j, base in enumerate(strand_seq):
                    idx = BASE_IDX.get(base)
                    if idx is None:
                        valid = False
                        break
                    score += pwm_mat[idx, j]
                if valid and score >= threshold:
                    family_hits[fam].append((i, strand, score, w))
    return family_hits

# ── Precompute sequences ──
print("Precomputing sequences...")
seqs = {}
for idx, row in merged.iterrows():
    chrom, pos, ref, alt = row['chr'], row['pos'], row.get('ref'), row.get('alt')
    if pd.isna(alt) or pd.isna(ref):
        continue
    var_pos_0 = pos - 1
    start = var_pos_0 - FLANK
    end = var_pos_0 + FLANK + 1
    ref_seq = get_sequence(chrom, start, end)
    if ref_seq is None or len(ref_seq) < 2 * FLANK + 1:
        continue
    var_idx = FLANK
    if ref_seq[var_idx].upper() != str(ref).upper():
        continue
    alt_seq = ref_seq[:var_idx] + str(alt) + ref_seq[var_idx+1:]
    seqs[idx] = (ref_seq, alt_seq, var_idx, row['abs_d'], row['cohens_d'], row['gene_id'])

print(f"Precomputed {len(seqs)} sequence pairs")

# ── Sweep thresholds ──
thresholds = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0]

print(f"\n{'='*90}")
print(f"{'Threshold':>10} {'Disrupted':>10} {'Created':>10} {'Rewired':>10} {'Neither':>10} "
      f"{'Self-sw':>8} {'d_disrupt':>10} {'d_create':>10} {'d_rewire':>10} {'d_neither':>10}")
print(f"{'='*90}")

sweep_results = []

for threshold in thresholds:
    cats = Counter()
    d_by_cat = defaultdict(list)
    self_switches = 0

    for idx, (ref_seq, alt_seq, var_idx, abs_d, cohens_d, gene_id) in seqs.items():
        ref_hits = scan_families(ref_seq, threshold)
        alt_hits = scan_families(alt_seq, threshold)

        disrupted = []
        created = []
        all_fams = set(list(ref_hits.keys()) + list(alt_hits.keys()))

        for fam in all_fams:
            ref_at_var = [h for h in ref_hits.get(fam, []) if h[0] <= var_idx < h[0] + h[3]]
            alt_at_var = [h for h in alt_hits.get(fam, []) if h[0] <= var_idx < h[0] + h[3]]
            has_ref = len(ref_at_var) > 0
            has_alt = len(alt_at_var) > 0
            if has_ref and not has_alt:
                disrupted.append(fam)
            elif not has_ref and has_alt:
                created.append(fam)

        # Check self-switches
        d_set = set(disrupted)
        c_set = set(created)
        self_switches += len(d_set & c_set)

        if disrupted and created:
            cat = 'rewired'
        elif disrupted and not created:
            cat = 'only_disrupted'
        elif created and not disrupted:
            cat = 'only_created'
        else:
            cat = 'neither'

        cats[cat] += 1
        d_by_cat[cat].append(abs_d)

    n = sum(cats.values())
    row_data = {
        'threshold': threshold,
        'n': n,
        'pct_disrupted': 100 * cats['only_disrupted'] / n,
        'pct_created': 100 * cats['only_created'] / n,
        'pct_rewired': 100 * cats['rewired'] / n,
        'pct_neither': 100 * cats['neither'] / n,
        'self_switches': self_switches,
        'median_d_disrupted': np.median(d_by_cat['only_disrupted']) if d_by_cat['only_disrupted'] else 0,
        'median_d_created': np.median(d_by_cat['only_created']) if d_by_cat['only_created'] else 0,
        'median_d_rewired': np.median(d_by_cat['rewired']) if d_by_cat['rewired'] else 0,
        'median_d_neither': np.median(d_by_cat['neither']) if d_by_cat['neither'] else 0,
    }
    sweep_results.append(row_data)

    print(f"{threshold:>10.1f} {row_data['pct_disrupted']:>9.1f}% {row_data['pct_created']:>9.1f}% "
          f"{row_data['pct_rewired']:>9.1f}% {row_data['pct_neither']:>9.1f}% "
          f"{self_switches:>8} {row_data['median_d_disrupted']:>10.2f} "
          f"{row_data['median_d_created']:>10.2f} {row_data['median_d_rewired']:>10.2f} "
          f"{row_data['median_d_neither']:>10.2f}")

sweep_df = pd.DataFrame(sweep_results)
sweep_df.to_csv(f'{RESULTS}/threshold_sensitivity_sweep.csv', index=False)

# ── Summary ──
print(f"\n{'='*70}")
print("STABILITY ASSESSMENT")
print("="*70)
print(f"Rewired % range: {sweep_df['pct_rewired'].min():.1f}% - {sweep_df['pct_rewired'].max():.1f}%")
print(f"Self-switches across ALL thresholds: {sweep_df['self_switches'].sum()}")

# Check if effect sizes are always comparable
print(f"\nEffect size ranges across thresholds:")
print(f"  Disrupted: {sweep_df['median_d_disrupted'].min():.2f} - {sweep_df['median_d_disrupted'].max():.2f}")
print(f"  Created:   {sweep_df['median_d_created'].min():.2f} - {sweep_df['median_d_created'].max():.2f}")
print(f"  Rewired:   {sweep_df['median_d_rewired'].min():.2f} - {sweep_df['median_d_rewired'].max():.2f}")
print(f"  Neither:   {sweep_df['median_d_neither'].min():.2f} - {sweep_df['median_d_neither'].max():.2f}")

print(f"\nSaved: threshold_sensitivity_sweep.csv")
