#!/usr/bin/env python3
"""
70_rewiring_deep_analysis.py

Deep analysis of TF motif rewiring at 728 causal bQTL:
1. For "rewired" bQTL: are disrupted AND created motifs sole copies in the peak?
2. Family→family transition matrix: which TFs replace which?
3. Biological interpretation of switching patterns
4. Integration with signaling layers, drought response, effect sizes
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict, Counter
import pyfaidx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats

BASE = str(Path(__file__).resolve().parent.parent)
RESULTS = f'{BASE}/results'
PEAK_DIR = f'{BASE}/data/raw/engelhorn_peaks'
GENOME = f'{BASE}/data/raw/B73_NAM5.fa'
FIG_DIR = f'{BASE}/figures/paper_figures'

# ── Load all data ──
causal = pd.read_csv(f'{RESULTS}/causal_bqtl_728.csv')
creation = pd.read_csv(f'{RESULTS}/motif_creation_vs_disruption_728.csv')
sole = pd.read_csv(f'{RESULTS}/sole_motif_verification_728.csv')
geno = pd.read_csv(f'{BASE}/data/processed/nam_founder_genotypes_at_bqtl.tsv', sep='\t')

# Gene atlas for descriptions and layers
atlas = pd.read_csv(f'{RESULTS}/gene_atlas_728.csv')

print(f"Loaded: {len(causal)} causal, {len(creation)} creation/disruption, {len(sole)} sole motif")

# ── Load PWM database ──
with open(f'{BASE}/data/processed/maize_tf_pwm_database.json') as f:
    pwm_db = json.load(f)
motifs = pwm_db['motifs']

pwm_matrices = []
pwm_families = []
for m in motifs:
    try:
        mat = np.array(m['pwm']).T
        mat = np.clip(mat, 1e-4, 1.0)
        mat = np.log2(mat / 0.25)
        pwm_matrices.append(mat)
        pwm_families.append(m['tf_family'])
    except:
        pass

# ── Load peaks ──
def load_peaks(hybrid):
    path = f'{PEAK_DIR}/{hybrid}.w2_4.q3_peaks.narrowPeak'
    if not os.path.exists(path):
        return []
    peaks = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            chrom = parts[0].replace('B73-', '')
            peaks.append((chrom, int(parts[1]), int(parts[2])))
    return peaks

all_peaks = {}
for hyb in ['B97','CML247','CML277','CML322','CML333','CML69','HP301',
            'IL14H','Ki11','Ki3','Ky21','M162W','M37W','Mo18W','Ms71',
            'NC358','Oh43','Oh7b','P39','Tx303',
            'A188','A619','CML103','Mo17','W22']:
    all_peaks[hyb] = load_peaks(hyb)

# ── Load genome ──
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

def scan_sequence(seq, pwm_mat, threshold=6.0):
    hits = []
    w = pwm_mat.shape[1]
    if len(seq) < w:
        return hits
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
                hits.append((i, strand, score, w))
    return hits

# ── Get ref/alt alleles ──
hybrids_geno = ['B97','CML247','CML277','CML322','CML333','CML69','HP301',
                'Il14H','Ki11','Ki3','Ky21','M162W','Mo18W','Ms71','NC358',
                'Oh43','Oh7B','P39','Tx303']

merged = causal.merge(geno, on=['chr', 'pos'], how='left')

def get_alt_allele(row):
    alts = set()
    for h in hybrids_geno:
        val = str(row.get(h, ''))
        if val not in ['0|0', './.', 'nan', '']:
            alts.add(val)
    return list(alts)[0] if alts else None

merged['alt'] = merged.apply(get_alt_allele, axis=1)

# ── ANALYSIS 1: Sole-copy check for rewired bQTL within actual peaks ──
print("\n" + "="*70)
print("ANALYSIS 1: Are rewired motifs sole copies within actual peaks?")
print("="*70)

threshold = 6.0
rewire_results = []

for idx, row in merged.iterrows():
    chrom = row['chr']
    pos = row['pos']
    gene_id = row['gene_id']
    ref_base = row.get('ref')
    alt_base = row.get('alt')

    if pd.isna(alt_base) or pd.isna(ref_base):
        continue

    # Find tightest peak
    best_peak = None
    best_width = float('inf')
    for hyb, peaks in all_peaks.items():
        for (pc, ps, pe) in peaks:
            if pc == chrom and ps <= pos <= pe:
                width = pe - ps
                if width < best_width:
                    best_width = width
                    best_peak = (pc, ps, pe)

    if best_peak is None:
        continue

    peak_chr, peak_start, peak_end = best_peak

    # Get ref peak sequence
    ref_peak_seq = get_sequence(peak_chr, peak_start, peak_end)
    if ref_peak_seq is None or len(ref_peak_seq) < 10:
        continue

    # Create alt peak sequence (substitute variant)
    var_pos_0 = pos - 1  # 1-based to 0-based
    var_in_peak = var_pos_0 - peak_start
    if var_in_peak < 0 or var_in_peak >= len(ref_peak_seq):
        continue

    # Verify ref matches
    if ref_peak_seq[var_in_peak].upper() != str(ref_base).upper():
        continue

    alt_peak_seq = ref_peak_seq[:var_in_peak] + str(alt_base) + ref_peak_seq[var_in_peak+1:]

    # Scan both ref and alt peak sequences with all PWMs
    ref_family_hits = defaultdict(list)  # family → all hits in ref peak
    alt_family_hits = defaultdict(list)  # family → all hits in alt peak
    ref_at_var = defaultdict(list)  # family → hits overlapping variant in ref
    alt_at_var = defaultdict(list)  # family → hits overlapping variant in alt

    for pwm_mat, fam in zip(pwm_matrices, pwm_families):
        w = pwm_mat.shape[1]

        # Scan ref
        for hit_pos, strand, score, hw in scan_sequence(ref_peak_seq, pwm_mat, threshold):
            ref_family_hits[fam].append((hit_pos, strand, score, hw))
            if hit_pos <= var_in_peak < hit_pos + hw:
                ref_at_var[fam].append((hit_pos, strand, score, hw))

        # Scan alt
        for hit_pos, strand, score, hw in scan_sequence(alt_peak_seq, pwm_mat, threshold):
            alt_family_hits[fam].append((hit_pos, strand, score, hw))
            if hit_pos <= var_in_peak < hit_pos + hw:
                alt_at_var[fam].append((hit_pos, strand, score, hw))

    # Classify each family
    all_fams = set(list(ref_at_var.keys()) + list(alt_at_var.keys()))
    disrupted = []
    created = []

    for fam in all_fams:
        has_ref = len(ref_at_var.get(fam, [])) > 0
        has_alt = len(alt_at_var.get(fam, [])) > 0

        if has_ref and not has_alt:
            # Disrupted: count OTHER copies of this family in ref peak
            total_in_ref = len(ref_family_hits[fam])
            at_var_in_ref = len(ref_at_var[fam])
            other_copies_ref = total_in_ref - at_var_in_ref
            disrupted.append((fam, other_copies_ref == 0))  # (family, is_sole)

        elif not has_ref and has_alt:
            # Created: count OTHER copies of this family in alt peak
            total_in_alt = len(alt_family_hits[fam])
            at_var_in_alt = len(alt_at_var[fam])
            other_copies_alt = total_in_alt - at_var_in_alt
            created.append((fam, other_copies_alt == 0))  # (family, is_sole)

    is_rewired = len(disrupted) > 0 and len(created) > 0
    is_only_disrupted = len(disrupted) > 0 and len(created) == 0
    is_only_created = len(disrupted) == 0 and len(created) > 0

    # For rewired: check if BOTH sides are sole
    disrupted_sole = [fam for fam, sole in disrupted if sole]
    created_sole = [fam for fam, sole in created if sole]
    disrupted_redundant = [fam for fam, sole in disrupted if not sole]
    created_redundant = [fam for fam, sole in created if not sole]

    rewire_results.append({
        'gene_id': gene_id,
        'chr': chrom,
        'pos': pos,
        'ref': ref_base,
        'alt': alt_base,
        'category': 'rewired' if is_rewired else ('only_disrupted' if is_only_disrupted
                     else ('only_created' if is_only_created else 'neither')),
        'n_disrupted': len(disrupted),
        'n_created': len(created),
        'n_disrupted_sole': len(disrupted_sole),
        'n_created_sole': len(created_sole),
        'n_disrupted_redundant': len(disrupted_redundant),
        'n_created_redundant': len(created_redundant),
        'disrupted_families': ','.join(f for f, _ in disrupted),
        'created_families': ','.join(f for f, _ in created),
        'disrupted_sole_families': ','.join(disrupted_sole),
        'created_sole_families': ','.join(created_sole),
        'both_sides_have_sole': len(disrupted_sole) > 0 and len(created_sole) > 0,
        'all_disrupted_sole': len(disrupted) > 0 and len(disrupted_redundant) == 0,
        'all_created_sole': len(created) > 0 and len(created_redundant) == 0,
        'clean_switch': (len(disrupted_sole) > 0 and len(created_sole) > 0
                        and len(disrupted_redundant) == 0 and len(created_redundant) == 0),
        'peak_width': best_width,
        'abs_d': row['abs_d'],
        'cohens_d': row['cohens_d'],
    })

    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx+1}/{len(merged)}...")

rw_df = pd.DataFrame(rewire_results)
print(f"Analyzed: {len(rw_df)} bQTL")

# ── Summary: sole copy in rewired ──
rewired = rw_df[rw_df['category'] == 'rewired']
print(f"\nRewired bQTL (disrupt + create): {len(rewired)}")
print(f"  Both sides have ≥1 sole-copy family: {rewired['both_sides_have_sole'].sum()} "
      f"({100*rewired['both_sides_have_sole'].mean():.1f}%)")
print(f"  ALL disrupted are sole copy: {rewired['all_disrupted_sole'].sum()} "
      f"({100*rewired['all_disrupted_sole'].mean():.1f}%)")
print(f"  ALL created are sole copy: {rewired['all_created_sole'].sum()} "
      f"({100*rewired['all_created_sole'].mean():.1f}%)")
print(f"  Clean switch (all sole on both sides): {rewired['clean_switch'].sum()} "
      f"({100*rewired['clean_switch'].mean():.1f}%)")

# Compare effect sizes
for cat in ['rewired', 'only_disrupted', 'only_created', 'neither']:
    sub = rw_df[rw_df['category'] == cat]
    if len(sub) > 0:
        print(f"  {cat}: n={len(sub)}, median |d|={sub['abs_d'].median():.2f}")

# For clean switches specifically
clean = rewired[rewired['clean_switch']]
if len(clean) > 5:
    not_clean = rewired[~rewired['clean_switch']]
    print(f"\n  Clean switch median |d|: {clean['abs_d'].median():.2f} (n={len(clean)})")
    print(f"  Non-clean rewired median |d|: {not_clean['abs_d'].median():.2f} (n={len(not_clean)})")
    stat, p = stats.mannwhitneyu(clean['abs_d'], not_clean['abs_d'], alternative='two-sided')
    print(f"  Mann-Whitney p = {p:.4f}")

# ── ANALYSIS 2: Family→family transition matrix ──
print(f"\n{'='*70}")
print("ANALYSIS 2: TF family switching matrix (disrupted → created)")
print("="*70)

transition_counts = Counter()
for _, row in rewired.iterrows():
    d_fams = [f for f in str(row['disrupted_families']).split(',') if f and f != 'nan']
    c_fams = [f for f in str(row['created_families']).split(',') if f and f != 'nan']
    for df in d_fams:
        for cf in c_fams:
            transition_counts[(df, cf)] += 1

# Top transitions
print("\nTop 20 family switches (disrupted → created):")
for (df, cf), count in transition_counts.most_common(20):
    print(f"  {df} → {cf}: {count}")

# Build matrix for heatmap
all_d_fams = sorted(set(df for df, cf in transition_counts))
all_c_fams = sorted(set(cf for df, cf in transition_counts))
trans_matrix = pd.DataFrame(0, index=all_d_fams, columns=all_c_fams)
for (df, cf), count in transition_counts.items():
    trans_matrix.loc[df, cf] = count

# Self-switches (same family disrupted and created at same bQTL)
print(f"\nSelf-switches (same family disrupted and created):")
for fam in sorted(set(all_d_fams) & set(all_c_fams)):
    if trans_matrix.loc[fam, fam] > 0:
        print(f"  {fam}: {trans_matrix.loc[fam, fam]} bQTL")

# ── ANALYSIS 3: Integrate with signaling layers and drought ──
print(f"\n{'='*70}")
print("ANALYSIS 3: Rewiring by signaling layer and drought response")
print("="*70)

# Merge with atlas
rw_atlas = rw_df.merge(atlas[['gene_id', 'signaling_layer', 'drought_cat',
                               'description']].drop_duplicates(),
                        on='gene_id', how='left')

print("\nCategory breakdown by signaling layer:")
for cat in ['rewired', 'only_disrupted', 'only_created']:
    sub = rw_atlas[rw_atlas['category'] == cat]
    if len(sub) > 0:
        print(f"\n  {cat} (n={len(sub)}):")
        layer_counts = sub['signaling_layer'].value_counts().head(5)
        for layer, n in layer_counts.items():
            print(f"    {layer}: {n} ({100*n/len(sub):.1f}%)")

print("\nCategory breakdown by drought response:")
for cat in ['rewired', 'only_disrupted', 'only_created']:
    sub = rw_atlas[rw_atlas['category'] == cat]
    if len(sub) > 0:
        drought_counts = sub['drought_cat'].value_counts()
        print(f"  {cat}: ", end='')
        for dc, n in drought_counts.items():
            print(f"{dc}={n} ", end='')
        print()

# ── ANALYSIS 4: Specific interesting switches ──
print(f"\n{'='*70}")
print("ANALYSIS 4: Biologically interesting switches")
print("="*70)

# Find clean switches with gene descriptions
clean_atlas = rw_atlas[(rw_atlas['clean_switch'] == True)]
print(f"\nClean sole-copy switches (n={len(clean_atlas)}):")
for _, row in clean_atlas.head(20).iterrows():
    d_fams = row['disrupted_sole_families']
    c_fams = row['created_sole_families']
    desc = str(row.get('description', ''))[:60]
    d_val = row['abs_d']
    drought = row.get('drought_cat', '')
    print(f"  {row['gene_id']}: {d_fams} → {c_fams} |d|={d_val:.2f} {drought}")
    print(f"    {desc}")

# ── FIGURE: Transition heatmap ──
print(f"\n{'='*70}")
print("Generating figures...")
print("="*70)

# Filter to families with ≥3 transitions
min_trans = 3
row_sums = trans_matrix.sum(axis=1)
col_sums = trans_matrix.sum(axis=0)
keep_rows = row_sums[row_sums >= min_trans].index
keep_cols = col_sums[col_sums >= min_trans].index
plot_matrix = trans_matrix.loc[keep_rows, keep_cols]

fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [2, 1]})

# Panel A: Transition heatmap
ax = axes[0]
im = ax.imshow(plot_matrix.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(plot_matrix.columns)))
ax.set_xticklabels(plot_matrix.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(plot_matrix.index)))
ax.set_yticklabels(plot_matrix.index, fontsize=8)
ax.set_xlabel('Created (NAM allele)', fontsize=10)
ax.set_ylabel('Disrupted (B73 allele)', fontsize=10)
ax.set_title('A. TF family switching at rewired bQTL', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8, label='Number of bQTL')

# Add text annotations
for i in range(len(plot_matrix.index)):
    for j in range(len(plot_matrix.columns)):
        val = plot_matrix.values[i, j]
        if val > 0:
            color = 'white' if val > plot_matrix.values.max() * 0.6 else 'black'
            ax.text(j, i, str(int(val)), ha='center', va='center',
                    fontsize=6, color=color)

# Panel B: Category breakdown with sole-copy info
ax = axes[1]
categories = ['Rewired\n(both)', 'Only\ndisrupted', 'Only\ncreated', 'Neither']
counts = [len(rw_df[rw_df['category'] == c]) for c in ['rewired', 'only_disrupted', 'only_created', 'neither']]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#95a5a6']

bars = ax.bar(range(4), counts, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(4))
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel('Number of bQTL', fontsize=10)
ax.set_title('B. Motif change categories', fontsize=11, fontweight='bold')

# Add sole-copy annotation for rewired
rewired_n = len(rewired)
clean_n = rewired['clean_switch'].sum()
ax.annotate(f'{clean_n} clean\nsole-copy\nswitches\n({100*clean_n/rewired_n:.0f}%)',
            xy=(0, counts[0]), xytext=(0.5, counts[0] + 20),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', color='red'))

for i, (count, cat) in enumerate(zip(counts, categories)):
    ax.text(i, count + 3, str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig_motif_rewiring.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{FIG_DIR}/fig_motif_rewiring.png', bbox_inches='tight', dpi=150)
print(f"Saved: fig_motif_rewiring.pdf")

# ── FIGURE 2: Effect sizes by category and sole-copy status ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Effect sizes by category
ax = axes[0]
cat_data = []
cat_labels = []
cat_colors = []
for cat, color, label in [('rewired', '#e74c3c', 'Rewired'),
                           ('only_disrupted', '#3498db', 'Only disrupted'),
                           ('only_created', '#2ecc71', 'Only created'),
                           ('neither', '#95a5a6', 'Neither')]:
    sub = rw_df[rw_df['category'] == cat]
    if len(sub) > 0:
        cat_data.append(sub['abs_d'].values)
        cat_labels.append(f'{label}\n(n={len(sub)})')
        cat_colors.append(color)

bp = ax.boxplot(cat_data, tick_labels=cat_labels, patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], cat_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel("|Cohen's d|", fontsize=10)
ax.set_title('A. Effect size by motif change category', fontsize=11, fontweight='bold')

# Panel B: Clean switch vs other rewired
ax = axes[1]
if len(clean) > 5:
    not_clean = rewired[~rewired['clean_switch']]
    data = [clean['abs_d'].values, not_clean['abs_d'].values]
    labels = [f'Clean switch\n(n={len(clean)})', f'Partial switch\n(n={len(not_clean)})']
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('#f39c12')
    bp['boxes'][1].set_alpha(0.6)
    ax.set_ylabel("|Cohen's d|", fontsize=10)
    ax.set_title('B. Clean vs partial sole-copy switches', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig_rewiring_effects.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{FIG_DIR}/fig_rewiring_effects.png', bbox_inches='tight', dpi=150)
print(f"Saved: fig_rewiring_effects.pdf")

# ── Save results ──
rw_df.to_csv(f'{RESULTS}/rewiring_deep_analysis_728.csv', index=False)
trans_matrix.to_csv(f'{RESULTS}/tf_family_transition_matrix.csv')
print(f"\nSaved: rewiring_deep_analysis_728.csv, tf_family_transition_matrix.csv")

# ── Final summary ──
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print("="*70)
print(f"Total bQTL analyzed: {len(rw_df)}")
for cat in ['rewired', 'only_disrupted', 'only_created', 'neither']:
    n = len(rw_df[rw_df['category'] == cat])
    print(f"  {cat}: {n} ({100*n/len(rw_df):.1f}%)")

print(f"\nAmong {len(rewired)} rewired bQTL:")
print(f"  Clean sole-copy switch: {rewired['clean_switch'].sum()} ({100*rewired['clean_switch'].mean():.1f}%)")
print(f"  ≥1 sole disrupted + ≥1 sole created: {rewired['both_sides_have_sole'].sum()} ({100*rewired['both_sides_have_sole'].mean():.1f}%)")

print(f"\nTotal unique transitions: {len(transition_counts)}")
print(f"Self-switches (same family): {sum(1 for (d,c) in transition_counts if d == c)}")
print(f"Cross-family switches: {sum(1 for (d,c) in transition_counts if d != c)}")
