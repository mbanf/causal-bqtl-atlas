#!/usr/bin/env python3
"""
02_train_compare.py — Phase 2: Train RF on ramp vs summit bQTLs.

Experiment design:
  1. bQTLs stratified by peak position (ramp / middle / summit)
  2. Hard negatives shared across conditions (MOA non-bQTL + promoter non-bQTL)
  3. Subsample each zone to matched sizes
  4. Train RF with 6-mer features on each subset
  5. 3-fold CV + cross-condition evaluation

Key question: Do ramp-positioned bQTLs produce a BETTER model per training
sample? If Thomas's ramp hypothesis is correct, ramp bQTLs should be cleaner
(direct TF motif disruptions) → higher AUC even with fewer samples.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from itertools import product
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results"
OUTDIR = Path(__file__).resolve().parent
FIG_DIR = OUTDIR / "figures"

# ── Config ────────────────────────────────────────────────────────────────────

WINDOW = 1001       # bp around each position
HALF = WINDOW // 2
K = 6               # k-mer length
N_FOLDS = 3
SEED = 42
RAMP_FRAC = 0.2     # outer 20% = ramp zone
N_PER_ZONE = 15000  # subsample positives per zone (ramp has ~37K)

np.random.seed(SEED)

# ── Load peak-mapped bQTLs ────────────────────────────────────────────────────

print("=" * 70)
print("PHASE 2: RF training — ramp vs summit comparison")
print("=" * 70)

mapping = pd.read_csv(OUTDIR / 'bqtl_peak_mapping.csv')
print(f"Loaded {len(mapping):,} peak-mapped bQTLs")

# Zone assignment
mapping['zone'] = pd.cut(
    mapping['rel_pos_sym'],
    bins=[0, RAMP_FRAC, 0.5 - RAMP_FRAC, 0.5],
    labels=['ramp', 'middle', 'summit'],
    include_lowest=True
)

zone_counts = mapping['zone'].value_counts()
print(f"\nZone counts:")
for z in ['ramp', 'middle', 'summit']:
    print(f"  {z:8s}: {zone_counts.get(z, 0):>7,}")

# ── Load genome ───────────────────────────────────────────────────────────────

print("\nLoading B73 genome...")
from pyfaidx import Fasta
genome = Fasta(str(DATA / "raw" / "B73_NAM5.fa"))
chrom_sizes = {name: len(genome[name]) for name in genome.keys()
               if name.startswith('chr') and name[3:].isdigit()}
print(f"  {len(chrom_sizes)} chromosomes loaded")

def extract_seq(chrom, pos, half=HALF):
    """Extract centered window, return uppercase or None."""
    if chrom not in chrom_sizes:
        return None
    start = max(0, pos - half)
    end = min(chrom_sizes[chrom], pos + half + 1)
    seq = str(genome[chrom][start:end]).upper()
    if len(seq) < WINDOW * 0.9:  # too short (near chr boundary)
        return None
    if 'N' * 10 in seq:  # too many Ns
        return None
    return seq

# ── Generate k-mer feature vector ─────────────────────────────────────────────

KMERS = [''.join(x) for x in product('ACGT', repeat=K)]
KMER_IDX = {km: i for i, km in enumerate(KMERS)}
N_KMERS = len(KMERS)

def seq_to_kmer(seq):
    """Count k-mer frequencies in sequence."""
    counts = np.zeros(N_KMERS, dtype=np.float32)
    for i in range(len(seq) - K + 1):
        kmer = seq[i:i+K]
        if kmer in KMER_IDX:
            counts[KMER_IDX[kmer]] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts

# ── Extract positive samples by zone ──────────────────────────────────────────

print("\n── Extracting positive samples ──")
zone_data = {}  # zone -> (X, positions)
valid_chroms = {f'chr{i}' for i in range(1, 11)}

for zone in ['ramp', 'middle', 'summit']:
    zone_bqtl = mapping[mapping['zone'] == zone].copy()
    zone_bqtl = zone_bqtl[zone_bqtl['chr'].isin(valid_chroms)]

    # Subsample
    if len(zone_bqtl) > N_PER_ZONE:
        zone_bqtl = zone_bqtl.sample(N_PER_ZONE, random_state=SEED)

    seqs, positions = [], []
    for _, row in zone_bqtl.iterrows():
        s = extract_seq(row['chr'], row['pos'])
        if s is not None:
            seqs.append(s)
            positions.append((row['chr'], row['pos']))

    print(f"  {zone:8s}: {len(seqs):,} sequences extracted (from {len(zone_bqtl):,})")

    X = np.array([seq_to_kmer(s) for s in seqs])
    zone_data[zone] = {'X': X, 'positions': positions, 'seqs': seqs}

# ── Generate hard negatives ──────────────────────────────────────────────────

print("\n── Generating hard negatives ──")

# Strategy: MOA non-bQTL positions (within peaks but >500bp from any bQTL)
# + promoter positions (2kb upstream of genes, no bQTL within 500bp)

bqtl_all = pd.read_csv(DATA / "processed" / "bqtl_snp_ww.csv")
bqtl_set = set(zip(bqtl_all['chr'], bqtl_all['pos']))

# Build bQTL position lookup for buffer check
from collections import defaultdict
bqtl_by_chrom = defaultdict(list)
for _, row in bqtl_all.iterrows():
    if row['chr'] in valid_chroms:
        bqtl_by_chrom[row['chr']].append(row['pos'])
for chrom in bqtl_by_chrom:
    bqtl_by_chrom[chrom] = sorted(bqtl_by_chrom[chrom])

def is_near_bqtl(chrom, pos, buffer=500):
    """Check if position is within buffer of any bQTL."""
    positions = bqtl_by_chrom.get(chrom, [])
    import bisect
    idx = bisect.bisect_left(positions, pos - buffer)
    for i in range(idx, min(idx + 20, len(positions))):
        if abs(positions[i] - pos) <= buffer:
            return True
        if positions[i] > pos + buffer:
            break
    return False

# MOA non-bQTL: sample from peaks but away from bQTLs
peaks = pd.read_csv(RESULTS / "pan_cistrome_peaks.csv")
peaks = peaks[peaks['chr'].isin(valid_chroms)]

n_neg_needed = N_PER_ZONE  # match to smallest positive set
neg_seqs = []
neg_positions = []

# Sample from random positions within peaks
peak_sample = peaks.sample(min(len(peaks), n_neg_needed * 5), random_state=SEED)
for _, p in peak_sample.iterrows():
    if len(neg_seqs) >= n_neg_needed:
        break
    pos = np.random.randint(p['start'] + HALF, max(p['start'] + HALF + 1, p['end'] - HALF))
    if not is_near_bqtl(p['chr'], pos):
        s = extract_seq(p['chr'], pos)
        if s is not None:
            neg_seqs.append(s)
            neg_positions.append((p['chr'], pos))

# If not enough from peaks, add promoter negatives
if len(neg_seqs) < n_neg_needed:
    print(f"  Adding promoter negatives (have {len(neg_seqs)} from peaks)...")
    import gzip
    gff = DATA / "raw" / "Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3"
    genes = []
    opener = gzip.open if str(gff).endswith('.gz') else open
    with opener(str(gff), 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 9 and parts[2] == 'gene' and parts[0] in valid_chroms:
                chrom = parts[0]
                start = int(parts[3])
                strand = parts[6]
                tss = start if strand == '+' else int(parts[4])
                genes.append((chrom, tss))

    np.random.shuffle(genes)
    for chrom, tss in genes:
        if len(neg_seqs) >= n_neg_needed:
            break
        # 2kb upstream promoter
        pos = tss - np.random.randint(500, 2000) if np.random.random() > 0.5 else tss + np.random.randint(500, 2000)
        if pos > HALF and not is_near_bqtl(chrom, pos):
            s = extract_seq(chrom, pos)
            if s is not None:
                neg_seqs.append(s)
                neg_positions.append((chrom, pos))

print(f"  Total negatives: {len(neg_seqs):,}")
X_neg = np.array([seq_to_kmer(s) for s in neg_seqs])

# ── Build training sets ──────────────────────────────────────────────────────

print("\n── Building training sets ──")

# Matched size: use minimum across zones
min_n = min(len(zone_data[z]['X']) for z in ['ramp', 'middle', 'summit'])
min_n = min(min_n, len(X_neg))
print(f"  Matched set size: {min_n:,} positives + {min_n:,} negatives per condition")

datasets = {}
for zone in ['ramp', 'middle', 'summit']:
    X_pos = zone_data[zone]['X'][:min_n]
    X_n = X_neg[:min_n]
    X = np.vstack([X_pos, X_n])
    y = np.concatenate([np.ones(min_n), np.zeros(min_n)])
    datasets[zone] = (X, y)
    print(f"  {zone:8s}: {X.shape[0]:,} samples ({min_n:,} pos + {min_n:,} neg)")

# Also build a "full" dataset (balanced sample from all zones)
X_full_pos = np.vstack([zone_data[z]['X'][:min_n // 3] for z in ['ramp', 'middle', 'summit']])
n_full = len(X_full_pos)
X_full = np.vstack([X_full_pos, X_neg[:n_full]])
y_full = np.concatenate([np.ones(n_full), np.zeros(n_full)])
datasets['full_mix'] = (X_full, y_full)
print(f"  {'full_mix':8s}: {X_full.shape[0]:,} samples ({n_full:,} pos + {n_full:,} neg)")

# ── Train RF with 3-fold CV ──────────────────────────────────────────────────

print("\n── Training Random Forest models (3-fold CV) ──")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

results = {}

for name, (X, y) in datasets.items():
    print(f"\n  Training on: {name} ({len(X):,} samples)...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    aucs, aps = [], []
    fold_preds = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1,
                                     random_state=SEED + fold)
        rf.fit(X[train_idx], y[train_idx])
        y_prob = rf.predict_proba(X[test_idx])[:, 1]

        auc = roc_auc_score(y[test_idx], y_prob)
        ap = average_precision_score(y[test_idx], y_prob)
        aucs.append(auc)
        aps.append(ap)
        fold_preds.append((y[test_idx], y_prob))
        print(f"    Fold {fold+1}: AUC={auc:.4f}, AUPRC={ap:.4f}")

    results[name] = {
        'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs),
        'ap_mean': np.mean(aps), 'ap_std': np.std(aps),
        'fold_preds': fold_preds
    }
    print(f"    → Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"    → Mean AUPRC: {np.mean(aps):.4f} ± {np.std(aps):.4f}")

# ── Cross-condition evaluation ────────────────────────────────────────────────

print("\n── Cross-condition evaluation (train on A, test on B) ──")

cross_results = {}
for train_name in ['ramp', 'summit']:
    X_train, y_train = datasets[train_name]
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1,
                                 random_state=SEED)
    rf.fit(X_train, y_train)

    for test_name in ['ramp', 'summit', 'middle']:
        if test_name == train_name:
            continue
        X_test, y_test = datasets[test_name]
        y_prob = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        key = f"{train_name}→{test_name}"
        cross_results[key] = {'auc': auc, 'ap': ap}
        print(f"  {key:20s}: AUC={auc:.4f}, AUPRC={ap:.4f}")

# ── Results visualization ─────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# A: Within-condition CV AUC comparison
ax = axes[0]
names = ['ramp', 'middle', 'summit', 'full_mix']
colors = {'ramp': '#E53935', 'middle': '#FFA726', 'summit': '#42A5F5', 'full_mix': '#66BB6A'}
aucs = [results[n]['auc_mean'] for n in names]
stds = [results[n]['auc_std'] for n in names]
bars = ax.bar(names, aucs, yerr=stds, color=[colors[n] for n in names],
              capsize=5, edgecolor='white', linewidth=0.5)
ax.set_ylabel('AUC (3-fold CV)')
ax.set_title('A. Within-condition AUC')
ax.set_ylim(0.5, max(aucs) + 0.05)
for bar, auc, std in zip(bars, aucs, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
            f'{auc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# B: Cross-condition AUC heatmap
ax = axes[1]
cross_matrix = np.zeros((2, 3))
train_labels = ['ramp', 'summit']
test_labels = ['ramp', 'middle', 'summit']
for i, tr in enumerate(train_labels):
    for j, te in enumerate(test_labels):
        key = f"{tr}→{te}"
        if key in cross_results:
            cross_matrix[i, j] = cross_results[key]['auc']
        else:
            # Self: use CV result
            cross_matrix[i, j] = results[tr]['auc_mean']

im = ax.imshow(cross_matrix, cmap='RdYlGn', vmin=0.6, vmax=0.9, aspect='auto')
ax.set_xticks(range(3))
ax.set_xticklabels(test_labels)
ax.set_yticks(range(2))
ax.set_yticklabels([f'Train: {t}' for t in train_labels])
ax.set_title('B. Cross-condition AUC')
for i in range(2):
    for j in range(3):
        ax.text(j, i, f'{cross_matrix[i,j]:.3f}', ha='center', va='center',
                fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

# C: AUPRC comparison
ax = axes[2]
aps = [results[n]['ap_mean'] for n in names]
ap_stds = [results[n]['ap_std'] for n in names]
bars = ax.bar(names, aps, yerr=ap_stds, color=[colors[n] for n in names],
              capsize=5, edgecolor='white', linewidth=0.5)
ax.set_ylabel('AUPRC (3-fold CV)')
ax.set_title('C. Within-condition AUPRC')
ax.set_ylim(0.5, max(aps) + 0.05)
for bar, ap, std in zip(bars, aps, ap_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
            f'{ap:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle("Ramp vs Summit Training Comparison (RF, 6-mer features)\n"
             "Thomas's hypothesis: ramp bQTLs = cleaner TF motif disruptions",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'ramp_vs_summit_rf.pdf', dpi=150, bbox_inches='tight')
plt.savefig(FIG_DIR / 'ramp_vs_summit_rf.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {FIG_DIR / 'ramp_vs_summit_rf.pdf'}")

# ── Summary table ─────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"\n{'Condition':<12} {'AUC':>12} {'AUPRC':>12} {'N_train':>10}")
print("-" * 50)
for name in names:
    r = results[name]
    n = len(datasets[name][0])
    print(f"{name:<12} {r['auc_mean']:.4f}±{r['auc_std']:.4f} "
          f"{r['ap_mean']:.4f}±{r['ap_std']:.4f} {n:>10,}")

print(f"\n{'Cross-eval':<20} {'AUC':>8} {'AUPRC':>8}")
print("-" * 40)
for key, r in sorted(cross_results.items()):
    print(f"{key:<20} {r['auc']:.4f}   {r['ap']:.4f}")

# Save results
results_df = pd.DataFrame([
    {'condition': name,
     'auc_mean': results[name]['auc_mean'],
     'auc_std': results[name]['auc_std'],
     'auprc_mean': results[name]['ap_mean'],
     'auprc_std': results[name]['ap_std'],
     'n_samples': len(datasets[name][0])}
    for name in names
])
results_df.to_csv(OUTDIR / 'rf_ramp_vs_summit_results.csv', index=False)

cross_df = pd.DataFrame([
    {'train_on': k.split('→')[0], 'test_on': k.split('→')[1],
     'auc': v['auc'], 'auprc': v['ap']}
    for k, v in cross_results.items()
])
cross_df.to_csv(OUTDIR / 'rf_cross_condition_results.csv', index=False)
print(f"\nSaved results to {OUTDIR}")
