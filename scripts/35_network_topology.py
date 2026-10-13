"""
35_network_topology.py

Tests whether causal bQTL target genes occupy different positions in the
regulatory network compared to non-causal targets.

The distributed OS model predicts causal targets are NOT hubs — the system
tolerates swapping them precisely because they are peripheral/replaceable.

Uses the tripartite TF → bQTL → target network from the causal atlas.
"""
import sys
sys.path.insert(0, '..')
from pipeline_config import config

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

np.random.seed(42)

FIG_DIR = config.figures_dir
RESULTS_DIR = config.results_dir

# ── 1. Load network ─────────────────────────────────────────────────────
print("=" * 70)
print("NETWORK TOPOLOGY OF CAUSAL vs NON-CAUSAL TARGETS")
print("=" * 70)

net = pd.read_csv(RESULTS_DIR / 'causal_regulatory_network.csv')
print(f"Network edges: {len(net)}")
print(f"Unique TFs: {net['tf_name'].nunique()}")
print(f"Unique targets: {net['target_gene'].nunique()}")

# Load causal gene sets
causal_ww = pd.read_csv(RESULTS_DIR / 'hybrid_motif_linkage.csv')
causal_ds = pd.read_csv(RESULTS_DIR / 'hybrid_motif_linkage_ds.csv')
causal_genes = set(causal_ww['gene_id']) | set(causal_ds['gene_id'])
print(f"Causal genes (WW+DS): {len(causal_genes)}")

# ── 2. Compute node-level metrics ───────────────────────────────────────
print("\n--- Computing network metrics ---")

# Target gene degree (how many TF families regulate it)
target_degree = net.groupby('target_gene')['tf_family'].nunique().reset_index()
target_degree.columns = ['gene_id', 'tf_family_degree']

# Total edge count per target
target_edges = net.groupby('target_gene').size().reset_index(name='total_edges')

# Disrupted vs created edges
effect_counts = net.groupby(['target_gene', 'effect_type']).size().unstack(fill_value=0).reset_index()
effect_counts.columns.name = None
if 'disrupted' not in effect_counts.columns:
    effect_counts['disrupted'] = 0
if 'created' not in effect_counts.columns:
    effect_counts['created'] = 0
effect_counts = effect_counts.rename(columns={'target_gene': 'gene_id'})

# TF diversity (Shannon entropy of TF family distribution)
def shannon_entropy(counts):
    counts = np.array(counts, dtype=float)
    counts = counts[counts > 0]
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p))

tf_diversity = []
for gene, gdf in net.groupby('target_gene'):
    fam_counts = gdf['tf_family'].value_counts().values
    tf_diversity.append({
        'gene_id': gene,
        'tf_entropy': shannon_entropy(fam_counts),
        'n_unique_tfs': gdf['tf_name'].nunique(),
    })
tf_diversity = pd.DataFrame(tf_diversity)

# Merge all metrics
metrics = target_degree.merge(target_edges, left_on='gene_id', right_on='target_gene', how='left')
metrics = metrics.drop(columns=['target_gene'], errors='ignore')
metrics = metrics.merge(effect_counts[['gene_id', 'disrupted', 'created']], on='gene_id', how='left')
metrics = metrics.merge(tf_diversity, on='gene_id', how='left')

# Add disruption ratio
metrics['disruption_ratio'] = metrics['disrupted'] / (metrics['disrupted'] + metrics['created'])

# Label causal vs non-causal
metrics['is_causal'] = metrics['gene_id'].isin(causal_genes)

print(f"Genes in network: {len(metrics)}")
print(f"  Causal: {metrics['is_causal'].sum()}")
print(f"  Non-causal: {(~metrics['is_causal']).sum()}")

# ── 3. Compare metrics ──────────────────────────────────────────────────
print("\n--- Comparing causal vs non-causal targets ---")

causal = metrics[metrics['is_causal']]
noncausal = metrics[~metrics['is_causal']]

compare_metrics = ['tf_family_degree', 'total_edges', 'n_unique_tfs',
                   'tf_entropy', 'disrupted', 'created', 'disruption_ratio']

comparison_results = {}
for m in compare_metrics:
    c_vals = causal[m].dropna()
    nc_vals = noncausal[m].dropna()
    if len(c_vals) < 3 or len(nc_vals) < 3:
        continue
    u_stat, u_p = stats.mannwhitneyu(c_vals, nc_vals, alternative='two-sided')
    comparison_results[m] = {
        'causal_median': float(c_vals.median()),
        'noncausal_median': float(nc_vals.median()),
        'causal_mean': float(c_vals.mean()),
        'noncausal_mean': float(nc_vals.mean()),
        'U_stat': float(u_stat),
        'p_value': float(u_p),
    }
    sig = '*' if u_p < 0.05 else ''
    print(f"  {m:25s}: causal={c_vals.median():.2f} vs non-causal={nc_vals.median():.2f} "
          f"(U={u_stat:.0f}, p={u_p:.4f}) {sig}")

# ── 4. Hub analysis ─────────────────────────────────────────────────────
print("\n--- Hub analysis ---")

# Define hubs as top 10% by degree
hub_threshold = metrics['tf_family_degree'].quantile(0.9)
metrics['is_hub'] = metrics['tf_family_degree'] >= hub_threshold
print(f"Hub threshold (top 10%): {hub_threshold:.0f} TF families")
print(f"Hub genes: {metrics['is_hub'].sum()}")

# Are causal genes less likely to be hubs?
ct = pd.crosstab(metrics['is_causal'], metrics['is_hub'])
if ct.shape == (2, 2):
    odds_ratio_result = stats.fisher_exact(ct)
    or_val = odds_ratio_result[0]
    or_p = odds_ratio_result[1]
    print(f"Causal genes as hubs: {ct.loc[True, True] if True in ct.index and True in ct.columns else 0} / {metrics['is_causal'].sum()}")
    print(f"Non-causal genes as hubs: {ct.loc[False, True] if False in ct.index and True in ct.columns else 0} / {(~metrics['is_causal']).sum()}")
    print(f"Fisher's exact OR={or_val:.3f}, p={or_p:.4f}")
    comparison_results['hub_enrichment'] = {
        'odds_ratio': float(or_val),
        'p_value': float(or_p),
        'causal_hub_frac': float(metrics.loc[metrics['is_causal'], 'is_hub'].mean()),
        'noncausal_hub_frac': float(metrics.loc[~metrics['is_causal'], 'is_hub'].mean()),
    }

# ── 5. Betweenness-like metric (simplified) ─────────────────────────────
# In bipartite TF→target network, compute how many TF families are "shared"
# between a target and other targets (measures redundancy in regulation)
print("\n--- Regulatory redundancy ---")

# For each gene, which TF families regulate it
gene_tf_sets = {}
for gene, gdf in net.groupby('target_gene'):
    gene_tf_sets[gene] = set(gdf['tf_family'])

# Mean Jaccard similarity with other genes in network
# (subsample for speed)
sample_genes = list(gene_tf_sets.keys())
if len(sample_genes) > 500:
    np.random.shuffle(sample_genes)
    sample_genes = sample_genes[:500]

jaccard_means = {}
for g in gene_tf_sets:
    s1 = gene_tf_sets[g]
    jaccards = []
    for g2 in sample_genes:
        if g2 == g:
            continue
        s2 = gene_tf_sets[g2]
        j = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0
        jaccards.append(j)
    jaccard_means[g] = np.mean(jaccards)

metrics['tf_reg_similarity'] = metrics['gene_id'].map(jaccard_means)

c_sim = metrics.loc[metrics['is_causal'], 'tf_reg_similarity'].dropna()
nc_sim = metrics.loc[~metrics['is_causal'], 'tf_reg_similarity'].dropna()
u_stat, u_p = stats.mannwhitneyu(c_sim, nc_sim, alternative='two-sided')
print(f"Regulatory similarity (mean Jaccard of TF families):")
print(f"  Causal: {c_sim.median():.3f}, Non-causal: {nc_sim.median():.3f}")
print(f"  Mann-Whitney p={u_p:.4f}")
comparison_results['tf_reg_similarity'] = {
    'causal_median': float(c_sim.median()),
    'noncausal_median': float(nc_sim.median()),
    'p_value': float(u_p),
}

# ── 6. Figure ────────────────────────────────────────────────────────────
print("\n--- Generating figure ---")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: TF family degree distribution
ax = axes[0, 0]
bins = np.arange(0, metrics['tf_family_degree'].max() + 2) - 0.5
ax.hist(noncausal['tf_family_degree'], bins=bins, alpha=0.6, color='steelblue',
        label=f'Non-causal (n={len(noncausal)})', density=True)
ax.hist(causal['tf_family_degree'], bins=bins, alpha=0.6, color='coral',
        label=f'Causal (n={len(causal)})', density=True)
ax.set_xlabel('TF family degree')
ax.set_ylabel('Density')
p_deg = comparison_results['tf_family_degree']['p_value']
ax.set_title(f'A. TF family degree (p={p_deg:.4f})')
ax.legend(fontsize=8)

# Panel B: TF entropy
ax = axes[0, 1]
ax.hist(noncausal['tf_entropy'].dropna(), bins=30, alpha=0.6, color='steelblue',
        label='Non-causal', density=True)
ax.hist(causal['tf_entropy'].dropna(), bins=30, alpha=0.6, color='coral',
        label='Causal', density=True)
ax.set_xlabel('TF family entropy (bits)')
ax.set_ylabel('Density')
p_ent = comparison_results['tf_entropy']['p_value']
ax.set_title(f'B. Regulatory diversity (p={p_ent:.4f})')
ax.legend(fontsize=8)

# Panel C: Disruption ratio
ax = axes[1, 0]
c_dr = causal['disruption_ratio'].dropna()
nc_dr = noncausal['disruption_ratio'].dropna()
ax.hist(nc_dr, bins=30, alpha=0.6, color='steelblue', label='Non-causal', density=True)
ax.hist(c_dr, bins=30, alpha=0.6, color='coral', label='Causal', density=True)
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Disruption ratio (disrupted / total)')
ax.set_ylabel('Density')
p_dr = comparison_results['disruption_ratio']['p_value']
ax.set_title(f'C. Disruption ratio (p={p_dr:.4f})')
ax.legend(fontsize=8)

# Panel D: Bar chart of key comparisons (effect sizes)
ax = axes[1, 1]
key_metrics = ['tf_family_degree', 'total_edges', 'tf_entropy', 'disruption_ratio']
effect_sizes = []
labels = []
colors_bar = []
for m in key_metrics:
    r = comparison_results[m]
    # Cohen's d effect size
    c_vals = causal[m].dropna()
    nc_vals = noncausal[m].dropna()
    pooled_std = np.sqrt((c_vals.std()**2 + nc_vals.std()**2) / 2)
    d = (c_vals.mean() - nc_vals.mean()) / pooled_std if pooled_std > 0 else 0
    effect_sizes.append(d)
    labels.append(m.replace('_', '\n'))
    colors_bar.append('#e74c3c' if r['p_value'] < 0.05 else '#95a5a6')

bars = ax.barh(labels, effect_sizes, color=colors_bar, alpha=0.7)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel("Cohen's d (causal - non-causal)")
ax.set_title('D. Effect sizes')
for i, (es, m) in enumerate(zip(effect_sizes, key_metrics)):
    p = comparison_results[m]['p_value']
    ax.text(max(es, 0) + 0.02, i, f'p={p:.3f}', va='center', fontsize=8)

plt.suptitle('Network Topology: Causal vs Non-Causal Targets', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / 'network_topology.pdf', bbox_inches='tight', dpi=150)
print(f"Saved: {FIG_DIR / 'network_topology.pdf'}")

# ── 7. Save results ─────────────────────────────────────────────────────
results = {
    'comparison': comparison_results,
    'n_causal_in_network': int(metrics['is_causal'].sum()),
    'n_noncausal_in_network': int((~metrics['is_causal']).sum()),
    'hub_threshold_tf_families': float(hub_threshold),
}

with open(RESULTS_DIR / 'network_topology.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {RESULTS_DIR / 'network_topology.json'}")

# Save per-gene metrics
metrics.to_csv(RESULTS_DIR / 'network_topology_metrics.csv', index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for m in compare_metrics:
    if m in comparison_results:
        r = comparison_results[m]
        direction = "higher" if r['causal_median'] > r['noncausal_median'] else "lower"
        sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else "ns"
        print(f"  {m}: causal {direction} ({r['causal_median']:.2f} vs {r['noncausal_median']:.2f}) [{sig}]")
print("=" * 70)
