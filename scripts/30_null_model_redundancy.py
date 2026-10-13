"""
30_null_model_redundancy.py

Permutation null model for functional redundancy.
Tests whether observed GO-term functional convergence exceeds random expectation.

Two null models:
  1. Label shuffle: permute gene-GO mapping (break which genes have which GO terms)
  2. Genome random: draw random gene sets from all GO-annotated maize genes
"""
import sys
sys.path.insert(0, '..')
from pipeline_config import config

import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import gzip
import re

np.random.seed(42)

FIG_DIR = config.figures_dir
RESULTS_DIR = config.results_dir
N_PERMS = 500

# ── 1. Load data ──────────────────────────────────────────────────────────
print("=" * 70)
print("NULL MODEL FOR FUNCTIONAL REDUNDANCY")
print("=" * 70)

ml = pd.read_csv(RESULTS_DIR / 'hybrid_motif_linkage.csv')

# Build gene × hybrid sets
all_founders = set()
gene_hybrid = {}
for _, row in ml.iterrows():
    gene = row['gene_id']
    var_hybrids = set(row['variant_hybrids'].split('|')) if pd.notna(row['variant_hybrids']) else set()
    all_founders.update(var_hybrids)
    gene_hybrid[gene] = var_hybrids
founders = sorted(all_founders)
genes = sorted(gene_hybrid.keys())
n_founders = len(founders)
print(f"Genes: {len(genes)}, Founders: {n_founders}")

# Binary matrix
matrix = np.zeros((len(genes), n_founders), dtype=int)
gene_idx = {g: i for i, g in enumerate(genes)}
founder_idx = {f: i for i, f in enumerate(founders)}
for gene, hybs in gene_hybrid.items():
    for h in hybs:
        if h in founder_idx:
            matrix[gene_idx[gene], founder_idx[h]] = 1

# ── 2. Load GO annotations ────────────────────────────────────────────────
print("\n--- Loading GO annotations ---")

entap_path = config.entap_gz
gene_go = {}      # causal gene -> set of GO terms
all_go_genes = {} # ALL genes in EntAP

with gzip.open(entap_path, 'rt') as f:
    header = f.readline().strip().split('\t')
    go_bio_col = None
    query_col = None
    for i, h in enumerate(header):
        if 'biological' in h.lower() and 'go' in h.lower():
            go_bio_col = i
        if 'query' in h.lower() and 'seq' in h.lower():
            query_col = i
    if query_col is None:
        query_col = 0

    for line in f:
        parts = line.strip().split('\t')
        if go_bio_col is None or go_bio_col >= len(parts):
            continue
        raw_id = parts[query_col]
        gene_id = re.sub(r'_T\d+$', '', raw_id)
        if not parts[go_bio_col].strip():
            continue
        bp_terms = {t.strip() for t in parts[go_bio_col].split(',') if t.strip().startswith('GO:')}
        if bp_terms:
            all_go_genes[gene_id] = bp_terms
            if gene_id in gene_idx:
                gene_go[gene_id] = bp_terms

annotated_causal = set(gene_go.keys())
print(f"Causal genes with GO BP: {len(annotated_causal)}/{len(genes)}")
print(f"All genome genes with GO: {len(all_go_genes)}")

# ── 3. Precompute per-hybrid GO term sets ─────────────────────────────────
# This is the key optimization: precompute GO sets once, then just permute mappings

def compute_go_jaccards(mat, gene_list, go_map):
    """Fast computation of pairwise GO Jaccard between hybrids."""
    n_hyb = mat.shape[1]
    # Build per-hybrid GO sets
    hybrid_go = []
    for j in range(n_hyb):
        go_set = set()
        for i in range(len(gene_list)):
            if mat[i, j] == 1 and gene_list[i] in go_map:
                go_set.update(go_map[gene_list[i]])
        hybrid_go.append(go_set)

    # Pairwise Jaccard
    jaccards = []
    for i, j in combinations(range(n_hyb), 2):
        a, b = hybrid_go[i], hybrid_go[j]
        if a or b:
            jaccards.append(len(a & b) / len(a | b))
    return np.mean(jaccards) if jaccards else 0

def compute_redundancy(mat, gene_list, go_map):
    """For each hybrid pair, what fraction of A-only genes have GO match in B?"""
    n_hyb = mat.shape[1]
    # Pre-build hybrid GO sets and gene lists
    hybrid_genes = []
    hybrid_go_sets = []
    for j in range(n_hyb):
        genes_j = [gene_list[i] for i in range(len(gene_list)) if mat[i, j] == 1 and gene_list[i] in go_map]
        go_set_j = set()
        for g in genes_j:
            go_set_j.update(go_map[g])
        hybrid_genes.append(genes_j)
        hybrid_go_sets.append(go_set_j)

    # Sample 50 random pairs instead of all 276 for speed
    all_pairs = list(combinations(range(n_hyb), 2))
    if len(all_pairs) > 50:
        pair_sample = [all_pairs[k] for k in np.random.choice(len(all_pairs), 50, replace=False)]
    else:
        pair_sample = all_pairs

    scores = []
    for i, j in pair_sample:
        # Genes in i but not j
        genes_i_only = [g for g in hybrid_genes[i] if g not in set(hybrid_genes[j])]
        if not genes_i_only or not hybrid_go_sets[j]:
            continue
        matched = sum(1 for g in genes_i_only if go_map[g] & hybrid_go_sets[j])
        scores.append(matched / len(genes_i_only))
    return np.mean(scores) if scores else 0

def compute_gene_jaccard(mat, n_hyb):
    """Pairwise gene Jaccard."""
    jaccards = []
    for i, j in combinations(range(n_hyb), 2):
        a = set(np.where(mat[:, i] == 1)[0])
        b = set(np.where(mat[:, j] == 1)[0])
        if a or b:
            jaccards.append(len(a & b) / len(a | b))
    return np.mean(jaccards) if jaccards else 0

# ── 4. Observed metrics ──────────────────────────────────────────────────
print("\n--- Computing observed metrics ---")
obs_gene_j = compute_gene_jaccard(matrix, n_founders)
obs_go_j = compute_go_jaccards(matrix, genes, gene_go)
obs_redund = compute_redundancy(matrix, genes, gene_go)
obs_ratio = obs_go_j / obs_gene_j if obs_gene_j > 0 else 0

print(f"Observed gene Jaccard:  {obs_gene_j:.3f}")
print(f"Observed GO Jaccard:    {obs_go_j:.3f}")
print(f"Observed redundancy:    {obs_redund:.3f}")
print(f"Observed ratio:         {obs_ratio:.2f}×")

# ── 5. Permutation test 1: Shuffle GO labels ─────────────────────────────
print(f"\n--- Permutation test 1: Shuffle gene-GO mapping ({N_PERMS} perms) ---")
null_go_j = []
null_redund = []

go_genes_list = list(gene_go.keys())
go_values_list = [gene_go[g] for g in go_genes_list]

for perm in range(N_PERMS):
    if perm % 50 == 0:
        print(f"  Permutation {perm}/{N_PERMS}...", flush=True)

    # Shuffle GO assignments among causal genes
    perm_idx = np.random.permutation(len(go_genes_list))
    shuffled_go = {}
    for orig_i, perm_i in enumerate(perm_idx):
        shuffled_go[go_genes_list[orig_i]] = go_values_list[perm_i]

    null_go_j.append(compute_go_jaccards(matrix, genes, shuffled_go))
    null_redund.append(compute_redundancy(matrix, genes, shuffled_go))

null_go_j = np.array(null_go_j)
null_redund = np.array(null_redund)

p_go = (null_go_j >= obs_go_j).mean()
p_red = (null_redund >= obs_redund).mean()

print(f"\n  Null GO Jaccard:    {null_go_j.mean():.3f} ± {null_go_j.std():.3f}")
print(f"  Observed:           {obs_go_j:.3f}")
print(f"  p-value:            {p_go:.4f}")
print(f"\n  Null redundancy:    {null_redund.mean():.3f} ± {null_redund.std():.3f}")
print(f"  Observed:           {obs_redund:.3f}")
print(f"  p-value:            {p_red:.4f}")

# ── 6. Permutation test 2: Random genome gene sets ───────────────────────
print(f"\n--- Permutation test 2: Random genome gene sets ({N_PERMS} perms) ---")

genome_go_genes = list(all_go_genes.keys())
print(f"Genome GO gene pool: {len(genome_go_genes)}")

# How many annotated genes per hybrid?
n_ann_per_hyb = []
for j in range(n_founders):
    n = sum(1 for i in range(len(genes)) if matrix[i, j] == 1 and genes[i] in gene_go)
    n_ann_per_hyb.append(n)

null2_go_j = []
null2_redund = []

for perm in range(N_PERMS):
    if perm % 50 == 0:
        print(f"  Permutation {perm}/{N_PERMS}...", flush=True)

    # Draw random genes for each hybrid
    rand_sets = []
    for j in range(n_founders):
        chosen_idx = np.random.choice(len(genome_go_genes), size=min(n_ann_per_hyb[j], len(genome_go_genes)), replace=False)
        chosen_genes = [genome_go_genes[k] for k in chosen_idx]
        rand_sets.append(chosen_genes)

    # Build GO sets for random hybrid gene sets
    hybrid_go = []
    for gene_list_j in rand_sets:
        go_set = set()
        for g in gene_list_j:
            go_set.update(all_go_genes[g])
        hybrid_go.append(go_set)

    # GO Jaccard
    jaccards = []
    for i, j in combinations(range(n_founders), 2):
        a, b = hybrid_go[i], hybrid_go[j]
        if a or b:
            jaccards.append(len(a & b) / len(a | b))
    null2_go_j.append(np.mean(jaccards))

    # Redundancy (sample 50 pairs)
    all_pairs = list(combinations(range(n_founders), 2))
    pair_sample = [all_pairs[k] for k in np.random.choice(len(all_pairs), 50, replace=False)]
    scores = []
    for i, j in pair_sample:
        # Genes in i but not j
        i_only = [g for g in rand_sets[i] if g not in set(rand_sets[j])]
        if not i_only or not hybrid_go[j]:
            continue
        matched = sum(1 for g in i_only if all_go_genes[g] & hybrid_go[j])
        scores.append(matched / len(i_only))
    null2_redund.append(np.mean(scores) if scores else 0)

null2_go_j = np.array(null2_go_j)
null2_redund = np.array(null2_redund)

p2_go = (null2_go_j >= obs_go_j).mean()
p2_red = (null2_redund >= obs_redund).mean()

print(f"\n  Random genome GO J: {null2_go_j.mean():.3f} ± {null2_go_j.std():.3f}")
print(f"  Observed:           {obs_go_j:.3f}")
print(f"  p-value:            {p2_go:.4f}")
print(f"\n  Random redundancy:  {null2_redund.mean():.3f} ± {null2_redund.std():.3f}")
print(f"  Observed:           {obs_redund:.3f}")
print(f"  p-value:            {p2_red:.4f}")

# ── 7. Convergence ratio null ─────────────────────────────────────────────
print("\n--- Convergence ratio test ---")
null_ratios = []
for perm in range(N_PERMS):
    perm_mat = matrix.copy()
    np.random.shuffle(perm_mat)  # shuffle rows
    gene_j = compute_gene_jaccard(perm_mat, n_founders)

    perm_idx = np.random.permutation(len(go_genes_list))
    shuffled_go = {go_genes_list[i]: go_values_list[perm_idx[i]] for i in range(len(go_genes_list))}
    go_j = compute_go_jaccards(perm_mat, genes, shuffled_go)
    if gene_j > 0:
        null_ratios.append(go_j / gene_j)

null_ratios = np.array(null_ratios)
p_ratio = (null_ratios >= obs_ratio).mean()
print(f"Observed ratio: {obs_ratio:.2f}×")
print(f"Null ratio:     {null_ratios.mean():.2f}× ± {null_ratios.std():.2f}")
print(f"p-value:        {p_ratio:.4f}")

# ── 8. Figure ─────────────────────────────────────────────────────────────
print("\n--- Generating figure ---")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
ax.hist(null_go_j, bins=40, color='steelblue', alpha=0.7, edgecolor='white', label='Null (label shuffle)')
ax.axvline(obs_go_j, color='red', linewidth=2, linestyle='--', label=f'Observed ({obs_go_j:.3f})')
ax.set_xlabel('Mean GO BP Jaccard')
ax.set_ylabel('Count')
ax.set_title(f'A. GO Jaccard vs label-shuffle null\n(p={p_go:.4f})')
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.hist(null2_go_j, bins=40, color='darkorange', alpha=0.7, edgecolor='white', label='Null (random genome)')
ax.axvline(obs_go_j, color='red', linewidth=2, linestyle='--', label=f'Observed ({obs_go_j:.3f})')
ax.set_xlabel('Mean GO BP Jaccard')
ax.set_ylabel('Count')
ax.set_title(f'B. GO Jaccard vs genome-random null\n(p={p2_go:.4f})')
ax.legend(fontsize=8)

ax = axes[1, 0]
ax.hist(null_redund, bins=40, color='steelblue', alpha=0.7, edgecolor='white', label='Null (label shuffle)')
ax.axvline(obs_redund, color='red', linewidth=2, linestyle='--', label=f'Observed ({obs_redund:.3f})')
ax.set_xlabel('Mean functional redundancy')
ax.set_ylabel('Count')
ax.set_title(f'C. Redundancy vs label-shuffle null\n(p={p_red:.4f})')
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.hist(null_ratios, bins=40, color='mediumpurple', alpha=0.7, edgecolor='white', label='Null (row shuffle)')
ax.axvline(obs_ratio, color='red', linewidth=2, linestyle='--', label=f'Observed ({obs_ratio:.2f}×)')
ax.set_xlabel('GO Jaccard / Gene Jaccard ratio')
ax.set_ylabel('Count')
ax.set_title(f'D. Convergence ratio vs null\n(p={p_ratio:.4f})')
ax.legend(fontsize=8)

plt.suptitle('Functional Redundancy Null Model Tests', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / 'null_model_redundancy.pdf', bbox_inches='tight', dpi=150)
print(f"Saved: {FIG_DIR / 'null_model_redundancy.pdf'}")

# ── 9. Save results ──────────────────────────────────────────────────────
results = {
    'test1_label_shuffle': {
        'n_permutations': N_PERMS,
        'observed_go_jaccard': float(obs_go_j),
        'null_go_jaccard_mean': float(null_go_j.mean()),
        'null_go_jaccard_std': float(null_go_j.std()),
        'p_go_jaccard': float(p_go),
        'observed_redundancy': float(obs_redund),
        'null_redundancy_mean': float(null_redund.mean()),
        'null_redundancy_std': float(null_redund.std()),
        'p_redundancy': float(p_red),
    },
    'test2_genome_random': {
        'n_permutations': N_PERMS,
        'genome_go_gene_pool': len(genome_go_genes),
        'observed_go_jaccard': float(obs_go_j),
        'null_go_jaccard_mean': float(null2_go_j.mean()),
        'null_go_jaccard_std': float(null2_go_j.std()),
        'p_go_jaccard': float(p2_go),
        'observed_redundancy': float(obs_redund),
        'null_redundancy_mean': float(null2_redund.mean()),
        'null_redundancy_std': float(null2_redund.std()),
        'p_redundancy': float(p2_red),
    },
    'convergence_ratio': {
        'observed_ratio': float(obs_ratio),
        'null_ratio_mean': float(null_ratios.mean()),
        'null_ratio_std': float(null_ratios.std()),
        'p_value': float(p_ratio),
    },
    'observed_metrics': {
        'gene_jaccard_mean': float(obs_gene_j),
        'go_jaccard_mean': float(obs_go_j),
        'redundancy_mean': float(obs_redund),
        'n_genes': len(genes),
        'n_annotated': len(annotated_causal),
        'n_founders': n_founders,
    }
}

with open(RESULTS_DIR / 'null_model_redundancy.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {RESULTS_DIR / 'null_model_redundancy.json'}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Label-shuffle: GO Jaccard p={p_go:.4f}, Redundancy p={p_red:.4f}")
print(f"Genome-random: GO Jaccard p={p2_go:.4f}, Redundancy p={p2_red:.4f}")
print(f"Convergence:   {obs_ratio:.2f}× observed vs {null_ratios.mean():.2f}× null (p={p_ratio:.4f})")
print("=" * 70)
