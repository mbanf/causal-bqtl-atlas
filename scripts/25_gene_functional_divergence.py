"""
25_gene_functional_divergence.py

Analyzes what gene-level divergence across hybrids means functionally.
Key questions:
  - Are different genes in different hybrids functionally redundant (same pathways)?
  - Does Jaccard increase when we go from gene-level to pathway/GO-level?
  - What functional categories are hybrid-specific vs universally shared?
  - Can we build per-hybrid functional fingerprints?
"""
import sys
sys.path.insert(0, '..')
from pipeline_config import config

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import gzip
import re

FIG_DIR = config.figures_dir
RESULTS_DIR = config.results_dir

# ── 1. Load data ──────────────────────────────────────────────────────────
print("=" * 70)
print("GENE FUNCTIONAL DIVERGENCE ACROSS HYBRIDS")
print("=" * 70)

ml = pd.read_csv(RESULTS_DIR / 'hybrid_motif_linkage.csv')
ga = pd.read_csv(RESULTS_DIR / 'gene_atlas_728.csv')

# Get all 24 founders
all_founders = set()
for h in ml['variant_hybrids'].dropna():
    all_founders.update(h.split('|'))
founders = sorted(all_founders)
print(f"Founders: {len(founders)}")

# ── 2. Build gene × hybrid binary matrix ──────────────────────────────────
print("\n--- Building gene × hybrid matrix ---")
gene_hybrid = {}  # gene_id -> set of founders carrying variant
for _, row in ml.iterrows():
    gene = row['gene_id']
    var_hybrids = set(row['variant_hybrids'].split('|')) if pd.notna(row['variant_hybrids']) else set()
    gene_hybrid[gene] = var_hybrids

# Binary matrix: rows=genes, cols=founders
genes = sorted(gene_hybrid.keys())
matrix = np.zeros((len(genes), len(founders)), dtype=int)
gene_idx = {g: i for i, g in enumerate(genes)}
founder_idx = {f: i for i, f in enumerate(founders)}

for gene, hybs in gene_hybrid.items():
    for h in hybs:
        if h in founder_idx:
            matrix[gene_idx[gene], founder_idx[h]] = 1

print(f"Matrix: {matrix.shape[0]} genes × {matrix.shape[1]} founders")
print(f"Mean genes per hybrid: {matrix.sum(axis=0).mean():.0f} ± {matrix.sum(axis=0).std():.0f}")
print(f"Mean hybrids per gene: {matrix.sum(axis=1).mean():.1f} ± {matrix.sum(axis=1).std():.1f}")

# Gene sharing spectrum
n_hybrids_per_gene = matrix.sum(axis=1)
print(f"\nGene sharing spectrum:")
for n in [1, 2, 3, 5, 10, 15, 20, 24]:
    count = (n_hybrids_per_gene == n).sum()
    count_ge = (n_hybrids_per_gene >= n).sum()
    print(f"  Exactly {n:2d} hybrids: {count:4d} genes  |  ≥{n:2d}: {count_ge:4d}")

# ── 3. Load GO terms from EntAP ───────────────────────────────────────────
print("\n--- Loading GO annotations from EntAP ---")
entap_path = config.entap_gz

gene_go_bp = {}  # gene_id -> set of GO:XXXXXXX terms (biological process)
gene_go_mf = {}  # molecular function
gene_description = {}  # cleaned description

with gzip.open(entap_path, 'rt') as f:
    header = f.readline().strip().split('\t')
    # Find column indices
    col_map = {name: i for i, name in enumerate(header)}
    desc_col = col_map.get('Description', None)
    go_bp_uni = col_map.get('UniProt GO Biological', None)
    go_mf_uni = col_map.get('UniProt GO Molecular', None)
    go_bp_egg = col_map.get('EggNOG GO Biological', None)
    go_mf_egg = col_map.get('EggNOG GO Molecular', None)
    kegg_col = col_map.get('EggNOG KEGG Pathway', None)

    for line in f:
        fields = line.strip().split('\t')
        if len(fields) < 5:
            continue
        raw_id = fields[0]
        # Strip _T001 suffix
        gene_id = re.sub(r'_T\d+$', '', raw_id)

        if gene_id not in gene_idx:
            continue

        # Merge GO from UniProt and EggNOG
        go_bp = set()
        go_mf = set()
        for col in [go_bp_uni, go_bp_egg]:
            if col and col < len(fields) and fields[col]:
                for term in fields[col].split(','):
                    t = term.strip()
                    if t.startswith('GO:'):
                        go_bp.add(t)
        for col in [go_mf_uni, go_mf_egg]:
            if col and col < len(fields) and fields[col]:
                for term in fields[col].split(','):
                    t = term.strip()
                    if t.startswith('GO:'):
                        go_mf.add(t)

        if go_bp:
            gene_go_bp[gene_id] = gene_go_bp.get(gene_id, set()) | go_bp
        if go_mf:
            gene_go_mf[gene_id] = gene_go_mf.get(gene_id, set()) | go_mf

        # Description
        if desc_col and desc_col < len(fields) and fields[desc_col]:
            gene_description[gene_id] = fields[desc_col]

print(f"Genes with GO BP: {len(gene_go_bp)} / {len(genes)}")
print(f"Genes with GO MF: {len(gene_go_mf)} / {len(genes)}")
print(f"Unique GO BP terms: {len(set.union(*gene_go_bp.values()) if gene_go_bp else set())}")

# Also use signaling_layer and KEGG from gene_atlas
ga_dict = ga.set_index('gene_id').to_dict()
gene_layer = {}
gene_kegg = {}
for gene in genes:
    if gene in ga_dict['signaling_layer'] and pd.notna(ga_dict['signaling_layer'].get(gene)):
        gene_layer[gene] = ga_dict['signaling_layer'][gene]
    if gene in ga_dict.get('kegg_pathway', {}) and pd.notna(ga_dict.get('kegg_pathway', {}).get(gene)):
        kegg_str = str(ga_dict['kegg_pathway'][gene])
        gene_kegg[gene] = set(k.strip() for k in kegg_str.split(',') if k.strip().startswith('ko'))

print(f"Genes with signaling layer: {len(gene_layer)} / {len(genes)}")
print(f"Genes with KEGG pathways: {len(gene_kegg)} / {len(genes)}")

# ── 4. Compute Jaccard at multiple levels ─────────────────────────────────
print("\n--- Jaccard similarity at multiple levels ---")

def pairwise_jaccard(sets_by_hybrid):
    """Compute pairwise Jaccard for a dict of {hybrid: set}."""
    jaccards = []
    hybs = sorted(sets_by_hybrid.keys())
    for i, h1 in enumerate(hybs):
        for h2 in hybs[i+1:]:
            s1, s2 = sets_by_hybrid[h1], sets_by_hybrid[h2]
            if len(s1) == 0 and len(s2) == 0:
                continue
            j = len(s1 & s2) / len(s1 | s2)
            jaccards.append((h1, h2, j))
    return jaccards

# Build per-hybrid sets at different levels
hybrid_genes = {f: set() for f in founders}
hybrid_go_bp = {f: set() for f in founders}
hybrid_go_mf = {f: set() for f in founders}
hybrid_kegg = {f: set() for f in founders}
hybrid_layer = {f: Counter() for f in founders}

for gene, hybs in gene_hybrid.items():
    for h in hybs:
        if h not in founder_idx:
            continue
        hybrid_genes[h].add(gene)
        if gene in gene_go_bp:
            hybrid_go_bp[h].update(gene_go_bp[gene])
        if gene in gene_go_mf:
            hybrid_go_mf[h].update(gene_go_mf[gene])
        if gene in gene_kegg:
            hybrid_kegg[h].update(gene_kegg[gene])
        if gene in gene_layer:
            hybrid_layer[h][gene_layer[gene]] += 1

# Compute Jaccard at each level
levels = {
    'Gene': hybrid_genes,
    'GO Biological Process': hybrid_go_bp,
    'GO Molecular Function': hybrid_go_mf,
    'KEGG Pathway': hybrid_kegg,
}

level_jaccards = {}
print(f"\n{'Level':<25s} {'Mean Jaccard':>12s} {'Median':>8s} {'Min':>8s} {'Max':>8s} {'N items/hybrid':>15s}")
print("-" * 78)
for level_name, level_sets in levels.items():
    jacs = pairwise_jaccard(level_sets)
    vals = [j for _, _, j in jacs]
    mean_items = np.mean([len(s) for s in level_sets.values()])
    level_jaccards[level_name] = vals
    if vals:
        print(f"{level_name:<25s} {np.mean(vals):>12.3f} {np.median(vals):>8.3f} "
              f"{np.min(vals):>8.3f} {np.max(vals):>8.3f} {mean_items:>13.1f}")

# Signaling layer profile similarity (not Jaccard, but correlation)
layer_names = sorted(set(gene_layer.values()))
layer_profiles = {}
for f in founders:
    total = sum(hybrid_layer[f].values())
    if total > 0:
        layer_profiles[f] = np.array([hybrid_layer[f].get(l, 0) / total for l in layer_names])
    else:
        layer_profiles[f] = np.zeros(len(layer_names))

layer_corrs = []
for i, f1 in enumerate(founders):
    for f2 in founders[i+1:]:
        r, _ = stats.spearmanr(layer_profiles[f1], layer_profiles[f2])
        layer_corrs.append(r)
print(f"{'Signaling Layer (corr)':<25s} {np.mean(layer_corrs):>12.3f} {np.median(layer_corrs):>8.3f} "
      f"{np.min(layer_corrs):>8.3f} {np.max(layer_corrs):>8.3f} {len(layer_names):>13d}")

# ── 5. Functional redundancy test ─────────────────────────────────────────
print("\n--- Functional redundancy test ---")
print("Q: When hybrids share the same GO term but via DIFFERENT genes,")
print("   how often does this happen?")

# For each pair of hybrids, decompose overlap
shared_gene_count = []
diff_gene_same_go = []
diff_gene_diff_go = []

for i, f1 in enumerate(founders):
    for f2 in founders[i+1:]:
        g1 = hybrid_genes[f1]
        g2 = hybrid_genes[f2]
        shared = g1 & g2
        unique1 = g1 - g2
        unique2 = g2 - g1

        shared_gene_count.append(len(shared))

        # Unique genes that share GO terms with unique genes in the other hybrid
        go1 = {gene: gene_go_bp.get(gene, set()) for gene in unique1}
        go2 = {gene: gene_go_bp.get(gene, set()) for gene in unique2}

        # How many unique genes in f1 have a GO-term match in f2's unique genes?
        all_go2 = set()
        for gset in go2.values():
            all_go2.update(gset)

        n_functionally_redundant = 0
        n_functionally_unique = 0
        for gene, gos in go1.items():
            if gos and (gos & all_go2):
                n_functionally_redundant += 1
            elif gos:
                n_functionally_unique += 1
            # genes without GO are uncategorized

        if n_functionally_redundant + n_functionally_unique > 0:
            diff_gene_same_go.append(n_functionally_redundant)
            diff_gene_diff_go.append(n_functionally_unique)

if diff_gene_same_go:
    total_r = sum(diff_gene_same_go)
    total_u = sum(diff_gene_diff_go)
    pct = 100 * total_r / (total_r + total_u)
    print(f"  Hybrid-specific genes with GO-term overlap: {pct:.1f}%")
    print(f"  Mean per pair: {np.mean(diff_gene_same_go):.1f} redundant, "
          f"{np.mean(diff_gene_diff_go):.1f} unique")
    print(f"  → {pct:.0f}% of functionally annotated hybrid-specific genes")
    print(f"    have a functional equivalent in the other hybrid")

# ── 6. Per-gene uniqueness score ──────────────────────────────────────────
print("\n--- Gene uniqueness scoring ---")
# Score each gene by how many hybrids it appears in AND whether its function
# is redundantly covered

gene_uniqueness = []
for gene in genes:
    n_hyb = n_hybrids_per_gene[gene_idx[gene]]
    go_terms = gene_go_bp.get(gene, set())
    layer = gene_layer.get(gene, 'Unknown')
    desc = gene_description.get(gene, ga_dict.get('description', {}).get(gene, 'Unknown'))

    # How many other genes share at least one GO term?
    n_go_redundant = 0
    if go_terms:
        for other_gene, other_go in gene_go_bp.items():
            if other_gene != gene and (go_terms & other_go):
                n_go_redundant += 1

    gene_uniqueness.append({
        'gene_id': gene,
        'n_hybrids': int(n_hyb),
        'n_go_terms': len(go_terms),
        'n_go_redundant_genes': n_go_redundant,
        'signaling_layer': layer,
        'description': str(desc)[:100] if desc else 'Unknown',
    })

gu = pd.DataFrame(gene_uniqueness)

# Rare genes (≤5 hybrids)
rare_genes = gu[gu['n_hybrids'] <= 5]
print(f"Rare genes (≤5 hybrids): {len(rare_genes)}")
print(f"  With GO annotation: {(rare_genes['n_go_terms'] > 0).sum()}")
print(f"  Functionally redundant (GO overlap): "
      f"{(rare_genes['n_go_redundant_genes'] > 0).sum()}")
print(f"  Truly unique function: "
      f"{((rare_genes['n_go_terms'] > 0) & (rare_genes['n_go_redundant_genes'] == 0)).sum()}")

# Most common genes (≥20 hybrids)
very_common = gu[gu['n_hybrids'] >= 20]
print(f"\nMost common genes (≥20 hybrids): {len(very_common)}")
for _, row in very_common.iterrows():
    print(f"  {row['gene_id']} (n={row['n_hybrids']}): {row['description']}")

# ── 7. Per-hybrid functional fingerprints ─────────────────────────────────
print("\n--- Per-hybrid functional fingerprints ---")

# Build layer proportion table
layer_table = pd.DataFrame(index=founders, columns=layer_names, dtype=float)
for f in founders:
    total = sum(hybrid_layer[f].values()) or 1
    for l in layer_names:
        layer_table.loc[f, l] = hybrid_layer[f].get(l, 0) / total

print("Signaling layer proportions per hybrid:")
print(layer_table.round(3).to_string())

# Also build description-keyword fingerprint
# Extract key functional words from descriptions
def extract_keywords(desc):
    """Extract biological keywords from EntAP description."""
    if not desc or desc == 'Unknown' or pd.isna(desc):
        return set()
    desc_lower = str(desc).lower()
    keywords = set()
    kw_map = {
        'kinase': 'kinase', 'phosphatase': 'phosphatase', 'protease': 'protease',
        'transporter': 'transporter', 'channel': 'channel', 'permease': 'transporter',
        'aquaporin': 'transporter', 'pump': 'transporter',
        'transcription': 'transcription', 'zinc finger': 'zinc_finger',
        'dehydrogenase': 'dehydrogenase', 'oxidase': 'oxidase', 'reductase': 'reductase',
        'synthase': 'synthase', 'transferase': 'transferase',
        'ribosom': 'ribosome', 'histone': 'histone', 'chromatin': 'chromatin',
        'ubiquitin': 'ubiquitin', 'proteasome': 'proteasome',
        'receptor': 'receptor', 'hormone': 'hormone', 'auxin': 'hormone',
        'cytochrome': 'cytochrome', 'heat shock': 'heat_shock',
        'myb': 'myb', 'wrky': 'wrky', 'nac': 'nac', 'bzip': 'bzip',
        'abc ': 'abc_transporter', 'sugar': 'sugar', 'starch': 'starch',
        'cellulose': 'cell_wall', 'cell wall': 'cell_wall', 'pectin': 'cell_wall',
        'photosystem': 'photosynthesis', 'chlorophyll': 'photosynthesis',
        'tubulin': 'cytoskeleton', 'actin': 'cytoskeleton',
        'dna repair': 'dna_repair', 'helicase': 'helicase',
    }
    for pattern, kw in kw_map.items():
        if pattern in desc_lower:
            keywords.add(kw)
    return keywords

# Build keyword sets per hybrid
hybrid_keywords = {f: Counter() for f in founders}
for gene, hybs in gene_hybrid.items():
    desc = gene_description.get(gene, ga_dict.get('description', {}).get(gene, ''))
    kws = extract_keywords(desc)
    for h in hybs:
        if h in founder_idx:
            for kw in kws:
                hybrid_keywords[h][kw] += 1

# Get top keywords
all_kw_counts = Counter()
for c in hybrid_keywords.values():
    all_kw_counts.update(c)
top_keywords = [kw for kw, _ in all_kw_counts.most_common(20)]
print(f"\nTop functional keywords: {top_keywords}")

# ── 8. Hybrid-specific gene functional analysis ──────────────────────────
print("\n--- Hybrid-specific gene functions ---")

# Since minimum hybrids per gene = 3, use "rare" (≤5) vs "common" (≥15) instead
hybrid_rare = {}  # genes in ≤5 hybrids that include this founder
hybrid_common = {}  # genes in ≥15 hybrids that include this founder
for f in founders:
    rare_genes = set()
    common_genes = set()
    for gene in hybrid_genes[f]:
        n = n_hybrids_per_gene[gene_idx[gene]]
        if n <= 5:
            rare_genes.add(gene)
        elif n >= 15:
            common_genes.add(gene)
    hybrid_rare[f] = rare_genes
    hybrid_common[f] = common_genes

print(f"{'Founder':<12s} {'Rare≤5':>7s} {'Common≥15':>10s} {'Total':>6s}")
for f in founders:
    print(f"{f:<12s} {len(hybrid_rare[f]):>7d} {len(hybrid_common[f]):>10d} {len(hybrid_genes[f]):>6d}")

# Aggregate rare vs common gene functions by layer
print("\nRare vs common gene signaling layers:")
rare_layers = Counter()
common_layers = Counter()
for gene in genes:
    n_hyb = n_hybrids_per_gene[gene_idx[gene]]
    layer = gene_layer.get(gene, 'Unknown')
    if n_hyb <= 5:
        rare_layers[layer] += 1
    elif n_hyb >= 15:
        common_layers[layer] += 1

print(f"{'Layer':<25s} {'Rare≤5':>8s} {'Common≥15':>10s} {'Ratio':>8s}")
for layer in layer_names:
    r = rare_layers.get(layer, 0)
    c = common_layers.get(layer, 0)
    ratio = r / c if c > 0 else float('inf')
    print(f"{layer:<25s} {r:>8d} {c:>10d} {ratio:>8.2f}")

# ── 9. Description-based semantic clustering ──────────────────────────────
print("\n--- Description keyword overlap ---")
# For hybrid-specific genes, do they tend to have similar descriptions
# across hybrids?

# TF-IDF-like approach: keyword presence per hybrid
kw_matrix = np.zeros((len(founders), len(top_keywords)))
for i, f in enumerate(founders):
    total = sum(hybrid_keywords[f].values()) or 1
    for j, kw in enumerate(top_keywords):
        kw_matrix[i, j] = hybrid_keywords[f].get(kw, 0) / total

# Pairwise correlation of keyword profiles
kw_corrs = []
for i in range(len(founders)):
    for j in range(i+1, len(founders)):
        r, _ = stats.spearmanr(kw_matrix[i], kw_matrix[j])
        kw_corrs.append(r)
print(f"Keyword profile correlation: mean={np.mean(kw_corrs):.3f}, "
      f"min={np.min(kw_corrs):.3f}, max={np.max(kw_corrs):.3f}")

# ── 10. Save results ─────────────────────────────────────────────────────
print("\n--- Saving results ---")

# Gene uniqueness table
gu.to_csv(RESULTS_DIR / 'gene_hybrid_uniqueness.csv', index=False)
print(f"Saved: gene_hybrid_uniqueness.csv ({len(gu)} genes)")

# Per-hybrid summary
hybrid_summary = []
for f in founders:
    hybrid_summary.append({
        'founder': f,
        'n_genes': len(hybrid_genes[f]),
        'n_rare': len(hybrid_rare[f]),
        'n_common': len(hybrid_common[f]),
        'n_go_bp_terms': len(hybrid_go_bp[f]),
        'n_kegg_pathways': len(hybrid_kegg[f]),
        'pct_rare': 100 * len(hybrid_rare[f]) / max(len(hybrid_genes[f]), 1),
    })
hs = pd.DataFrame(hybrid_summary)
hs.to_csv(RESULTS_DIR / 'hybrid_functional_summary.csv', index=False)
print(f"Saved: hybrid_functional_summary.csv")

# Jaccard comparison
jaccard_summary = {}
for level_name, vals in level_jaccards.items():
    v = [j for _, _, j in vals] if vals and isinstance(vals[0], tuple) else vals
    jaccard_summary[level_name] = {
        'mean': float(np.mean(v)),
        'median': float(np.median(v)),
        'min': float(np.min(v)),
        'max': float(np.max(v)),
    }
jaccard_summary['Signaling Layer (corr)'] = {
    'mean': float(np.mean(layer_corrs)),
    'median': float(np.median(layer_corrs)),
    'min': float(np.min(layer_corrs)),
    'max': float(np.max(layer_corrs)),
}

with open(RESULTS_DIR / 'functional_redundancy_analysis.json', 'w') as f:
    json.dump({
        'jaccard_by_level': jaccard_summary,
        'n_genes': len(genes),
        'n_founders': len(founders),
        'go_bp_coverage': len(gene_go_bp),
        'kegg_coverage': len(gene_kegg),
        'functional_redundancy_pct': float(pct) if diff_gene_same_go else None,
        'rare_genes_le5': int((n_hybrids_per_gene <= 5).sum()),
        'common_genes_ge15': int((n_hybrids_per_gene >= 15).sum()),
        'min_hybrids_per_gene': int(n_hybrids_per_gene.min()),
        'max_hybrids_per_gene': int(n_hybrids_per_gene.max()),
        'median_hybrids_per_gene': float(np.median(n_hybrids_per_gene)),
    }, f, indent=2)
print(f"Saved: functional_redundancy_analysis.json")

# ── 11. FIGURES ───────────────────────────────────────────────────────────
print("\n--- Generating figures ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# A: Jaccard by level (bar chart showing convergence)
ax = axes[0, 0]
level_order = ['Gene', 'GO Biological Process', 'GO Molecular Function', 'KEGG Pathway']
def extract_vals(jac_list):
    if jac_list and isinstance(jac_list[0], tuple):
        return [j for _, _, j in jac_list]
    return jac_list
means = [np.mean(extract_vals(level_jaccards[l])) for l in level_order]
stds = [np.std(extract_vals(level_jaccards[l])) for l in level_order]
short_labels = ['Gene', 'GO BP', 'GO MF', 'KEGG']
colors = ['#E53935', '#FF9800', '#4CAF50', '#1976D2']
bars = ax.bar(range(len(level_order)), means, yerr=stds, color=colors,
              edgecolor='white', linewidth=1.5, capsize=5)
ax.set_xticks(range(len(level_order)))
ax.set_xticklabels(short_labels, fontsize=11)
ax.set_ylabel('Pairwise Jaccard similarity', fontsize=12)
ax.set_title('A. Functional convergence across levels', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1)
# Add value labels
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f'{m:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# B: Gene sharing spectrum (histogram)
ax = axes[0, 1]
bins = range(1, 26)
ax.hist(n_hybrids_per_gene, bins=bins, color='#1976D2', edgecolor='white',
        linewidth=0.5, alpha=0.85)
ax.set_xlabel('Number of hybrids carrying variant', fontsize=12)
ax.set_ylabel('Number of genes', fontsize=12)
ax.set_title('B. Gene sharing spectrum', fontsize=13, fontweight='bold')
ax.axvline(np.median(n_hybrids_per_gene), color='red', linestyle='--',
           label=f'Median = {np.median(n_hybrids_per_gene):.0f}')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# C: Functional fingerprint heatmap (layers × hybrids)
ax = axes[0, 2]
# Cluster hybrids by layer profile similarity
layer_mat = layer_table.values.astype(float)
# Cluster
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
if layer_mat.shape[0] > 2:
    dist = pdist(layer_mat, metric='cosine')
    Z = linkage(dist, method='ward')
    order = leaves_list(Z)
else:
    order = list(range(layer_mat.shape[0]))
ordered_founders = [founders[i] for i in order]
ordered_mat = layer_mat[order, :]

im = ax.imshow(ordered_mat.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_xticks(range(len(ordered_founders)))
ax.set_xticklabels(ordered_founders, rotation=90, fontsize=8)
ax.set_yticks(range(len(layer_names)))
ax.set_yticklabels(layer_names, fontsize=9)
ax.set_title('C. Signaling layer profiles', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.7, label='Proportion')

# D: Rare vs common gene functions
ax = axes[1, 0]
rare_data = []
common_data = []
for layer in layer_names:
    rare_data.append(rare_layers.get(layer, 0))
    common_data.append(common_layers.get(layer, 0))
# Normalize
rare_total = sum(rare_data) or 1
common_total = sum(common_data) or 1
rare_pct = [100 * x / rare_total for x in rare_data]
common_pct = [100 * x / common_total for x in common_data]

x = np.arange(len(layer_names))
width = 0.35
ax.barh(x - width/2, rare_pct, width, label='Rare (≤5 hybrids)', color='#E53935', alpha=0.85)
ax.barh(x + width/2, common_pct, width, label='Common (≥15 hybrids)', color='#1976D2', alpha=0.85)
ax.set_yticks(x)
ax.set_yticklabels(layer_names, fontsize=9)
ax.set_xlabel('Percentage of genes', fontsize=12)
ax.set_title('D. Rare vs common gene functions', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# E: Keyword heatmap
ax = axes[1, 1]
# Order hybrids same as panel C
kw_ordered = kw_matrix[order, :]
im2 = ax.imshow(kw_ordered, aspect='auto', cmap='viridis', interpolation='nearest')
ax.set_xticks(range(len(top_keywords)))
ax.set_xticklabels(top_keywords, rotation=90, fontsize=8)
ax.set_yticks(range(len(ordered_founders)))
ax.set_yticklabels(ordered_founders, fontsize=8)
ax.set_title('E. Functional keyword profiles', fontsize=13, fontweight='bold')
plt.colorbar(im2, ax=ax, shrink=0.7, label='Freq (normalized)')

# F: Redundancy decomposition pie
ax = axes[1, 2]
if diff_gene_same_go:
    total_annotated = sum(diff_gene_same_go) + sum(diff_gene_diff_go)
    n_unannotated = sum(len(hybrid_genes[f]) - len([g for g in hybrid_genes[f] if g in gene_go_bp])
                        for f in founders)
    sizes = [sum(diff_gene_same_go), sum(diff_gene_diff_go)]
    labels = [f'Functionally\nredundant\n({pct:.0f}%)',
              f'Functionally\nunique\n({100-pct:.0f}%)']
    colors_pie = ['#4CAF50', '#E53935']
    wedges, texts = ax.pie(sizes, labels=labels, colors=colors_pie,
                           startangle=90, textprops={'fontsize': 11})
    ax.set_title('F. Hybrid-specific gene redundancy', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(FIG_DIR / 'gene_functional_divergence.pdf', bbox_inches='tight', dpi=150)
plt.savefig(FIG_DIR / 'gene_functional_divergence.png', bbox_inches='tight', dpi=150)
print(f"Saved: gene_functional_divergence.pdf")
plt.close()

# ── 12. Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Gene-level Jaccard:  {jaccard_summary['Gene']['mean']:.3f}")
print(f"GO BP Jaccard:       {jaccard_summary['GO Biological Process']['mean']:.3f}")
print(f"GO MF Jaccard:       {jaccard_summary['GO Molecular Function']['mean']:.3f}")
print(f"KEGG Jaccard:        {jaccard_summary['KEGG Pathway']['mean']:.3f}")
print(f"Layer correlation:   {jaccard_summary['Signaling Layer (corr)']['mean']:.3f}")
ratio = jaccard_summary['GO Biological Process']['mean'] / jaccard_summary['Gene']['mean']
print(f"\nConvergence ratio (GO BP / Gene): {ratio:.2f}×")
print(f"→ {'Strong' if ratio > 1.5 else 'Moderate' if ratio > 1.2 else 'Weak'} "
      f"functional redundancy")
if diff_gene_same_go:
    print(f"\n{pct:.0f}% of hybrid-specific genes have functional equivalents")
    print(f"in the other hybrid → {'high' if pct > 60 else 'moderate' if pct > 40 else 'low'} redundancy")
print(f"\nRare genes (≤5 hybrids): {(n_hybrids_per_gene <= 5).sum()}")
print(f"Common genes (≥15 hybrids): {(n_hybrids_per_gene >= 15).sum()}")
print(f"Min hybrids per gene: {n_hybrids_per_gene.min()}")
print(f"Median hybrids per gene: {np.median(n_hybrids_per_gene):.0f}")
