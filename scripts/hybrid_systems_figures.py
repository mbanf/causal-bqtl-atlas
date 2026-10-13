"""
Systems-level hybrid analysis figures for the causal bQTL atlas paper.
Generates:
1. Hybrid similarity dendrogram + gene sharing heatmap
2. Regulatory burden comparison (stacked bars by functional layer)
3. Co-disruption module structure
4. Per-hybrid TF family disruption profiles
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from collections import defaultdict

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── Load data ──
causal = pd.read_csv('/Users/michael/perelyn/research/cell_biology/causal_bqtl_atlas/results/causal_bqtl_728.csv')
atlas = pd.read_csv('/Users/michael/perelyn/research/cell_biology/causal_bqtl_atlas/results/gene_atlas_728.csv')
motifs = pd.read_csv('/Users/michael/perelyn/research/cell_biology/causal_bqtl_atlas/results/allele_specific_motifs.csv')
geno = pd.read_csv('/Users/michael/perelyn/research/cell_biology/shared_data/processed/nam_founder_genotypes_at_bqtl.tsv', sep='\t')

nam_founders = list(geno.columns[3:])
geno_idx = {}
for _, row in geno.iterrows():
    geno_idx[(row['chr'], row['pos'])] = row

func_map = atlas.set_index('gene_id')['signaling_layer'].to_dict()
drought_map = atlas.set_index('gene_id')['drought_cat'].to_dict()
tf_fam_map = atlas.set_index('gene_id')['tf_families'].to_dict()

# ── Build gene x hybrid matrix ──
all_genes = sorted(causal['gene_id'].unique())
hybrids = sorted([f'B73x{f}' for f in nam_founders])
short_names = [h.replace('B73x', '') for h in hybrids]

gene_matrix = np.zeros((len(hybrids), len(all_genes)))
gene_effects = np.zeros((len(hybrids), len(all_genes)))

effect_map = causal.set_index('gene_id')['abs_d'].to_dict()

for j, gene in enumerate(all_genes):
    crow = causal[causal['gene_id'] == gene].iloc[0]
    geno_row = geno_idx.get((crow['chr'], crow['pos']))
    if geno_row is None:
        continue
    for i, h in enumerate(hybrids):
        founder = h.replace('B73x', '')
        gt = str(geno_row.get(founder, '0|0'))
        if gt != '0|0':
            gene_matrix[i, j] = 1
            gene_effects[i, j] = effect_map.get(gene, 0)

# ── Hybrid ancestry groups ──
ancestry = {
    'A188': 'Flint', 'HP301': 'Popcorn', 'IL14H': 'Iodent', 'P39': 'Popcorn',
    'A619': 'Stiff Stalk', 'B97': 'Stiff Stalk', 'Mo17': 'Lancaster',
    'Ms71': 'Stiff Stalk', 'Oh43': 'Iodent', 'Oh7b': 'Iodent', 'W22': 'Stiff Stalk',
    'CML103': 'Tropical', 'CML247': 'Tropical', 'CML277': 'Tropical',
    'CML322': 'Tropical', 'CML333': 'Tropical', 'CML69': 'Tropical',
    'Ki11': 'Tropical', 'Ki3': 'Tropical', 'NC358': 'Tropical',
    'Tx303': 'Tropical', 'M162W': 'Mixed',
    'Mo18W': 'Tropical', 'Ky21': 'Lancaster'
}
ancestry_colors = {
    'Tropical': '#e41a1c', 'Stiff Stalk': '#377eb8', 'Lancaster': '#4daf4a',
    'Iodent': '#984ea3', 'Flint': '#ff7f00', 'Popcorn': '#a65628',
    'Mixed': '#999999'
}


# ═══════════════════════════════════════════════════════
# FIGURE 1: Hybrid dendrogram + gene sharing heatmap
# ═══════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[1, 4],
                       hspace=0.02, wspace=0.02)

# Compute distances and linkage
hybrid_dists = pdist(gene_matrix, metric='jaccard')
Z = linkage(hybrid_dists, method='average')

# Top dendrogram
ax_dendro_top = fig.add_subplot(gs[0, 1])
dn = dendrogram(Z, labels=short_names, ax=ax_dendro_top, leaf_rotation=90,
                color_threshold=0, above_threshold_color='#333333')
ax_dendro_top.set_ylabel('Jaccard distance')
ax_dendro_top.set_title('Hybrid clustering by shared causal bQTL gene sets', fontsize=12, fontweight='bold')

# Reorder matrix by dendrogram
order = dn['leaves']
ordered_names = [short_names[i] for i in order]

# Similarity heatmap
ax_heat = fig.add_subplot(gs[1, 1])
sim_matrix = 1 - squareform(hybrid_dists)
sim_ordered = sim_matrix[np.ix_(order, order)]

im = ax_heat.imshow(sim_ordered, cmap='YlOrRd', vmin=0.2, vmax=0.9, aspect='auto')
ax_heat.set_xticks(range(len(ordered_names)))
ax_heat.set_xticklabels(ordered_names, rotation=90)
ax_heat.set_yticks(range(len(ordered_names)))
ax_heat.set_yticklabels(ordered_names)

# Add ancestry color strip
for i, name in enumerate(ordered_names):
    color = ancestry_colors.get(ancestry.get(name, 'Mixed'), '#999999')
    ax_heat.add_patch(plt.Rectangle((-1.5, i-0.5), 0.8, 1, color=color, clip_on=False))
    ax_heat.add_patch(plt.Rectangle((i-0.5, len(ordered_names)-0.5), 1, 0.8, color=color, clip_on=False))

# Colorbar
cbar = plt.colorbar(im, ax=ax_heat, shrink=0.6, label='Jaccard similarity')

# Annotations
ax_info = fig.add_subplot(gs[1, 0])
ax_info.axis('off')
# Stats text
stats_text = (
    "Regulatory burden\n"
    "────────────────\n"
    f"Genes/hybrid:\n  {gene_matrix.sum(axis=1).min():.0f}–{gene_matrix.sum(axis=1).max():.0f}\n"
    f"  (mean {gene_matrix.sum(axis=1).mean():.0f})\n\n"
    f"Hybrids/gene:\n  mean {gene_matrix.sum(axis=0).mean():.1f}\n\n"
    f"Mean Jaccard:\n  {sim_matrix[np.triu_indices_from(sim_matrix, k=1)].mean():.3f}\n\n"
    "Ancestry groups\n"
    "────────────────\n"
)
for grp, color in ancestry_colors.items():
    n = sum(1 for v in ancestry.values() if v == grp)
    if n > 0:
        stats_text += f"  ■ {grp} ({n})\n"

ax_info.text(0.1, 0.95, stats_text, transform=ax_info.transAxes,
             va='top', fontsize=8, family='monospace')

plt.savefig('/Users/michael/perelyn/research/cell_biology/causal_bqtl_atlas/figures/hybrid_similarity_heatmap.pdf')
plt.close()
print("Saved: hybrid_similarity_heatmap.pdf")


# ═══════════════════════════════════════════════════════
# FIGURE 2: Per-hybrid regulatory burden (stacked bars)
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})

# Panel A: Stacked bar by signaling layer
layer_order = ['Signal Transduction', 'Transcription Factor', 'Metabolism',
               'Transport', 'Protein Processing', 'Chromatin/Epigenetic',
               'Receptor/Sensor', 'Hormone', 'Translation', 'Redox/Stress',
               'Membrane/Transport', 'Unknown']
# Map to simpler names for display
layer_short = {
    'Signal Transduction': 'Signaling',
    'Transcription Factor': 'TF',
    'Membrane/Transport': 'Membrane',
    'Protein Processing': 'Protein QC',
    'Chromatin/Epigenetic': 'Chromatin',
    'Receptor/Sensor': 'Receptor',
    'Redox/Stress': 'Redox',
}
layer_colors = {
    'Signal Transduction': '#e41a1c',
    'Transcription Factor': '#377eb8',
    'Metabolism': '#4daf4a',
    'Transport': '#984ea3',
    'Protein Processing': '#ff7f00',
    'Chromatin/Epigenetic': '#a65628',
    'Receptor/Sensor': '#f781bf',
    'Hormone': '#999999',
    'Translation': '#66c2a5',
    'Redox/Stress': '#e78ac3',
    'Membrane/Transport': '#8da0cb',
    'Unknown': '#d9d9d9',
}

# Build layer counts per hybrid (in dendrogram order)
layer_data = defaultdict(list)
for i in order:
    h = hybrids[i]
    gene_ids = [all_genes[j] for j in range(len(all_genes)) if gene_matrix[i, j] == 1]
    total = len(gene_ids)
    for layer in layer_order:
        cnt = sum(1 for g in gene_ids if func_map.get(g) == layer)
        layer_data[layer].append(cnt)

# Existing layers only
existing_layers = [l for l in layer_order if sum(layer_data[l]) > 0]

ax = axes[0]
bottom = np.zeros(len(order))
for layer in existing_layers:
    vals = np.array(layer_data[layer])
    label = layer_short.get(layer, layer)
    ax.bar(range(len(order)), vals, bottom=bottom, color=layer_colors.get(layer, '#cccccc'),
           label=label, width=0.8, edgecolor='white', linewidth=0.3)
    bottom += vals

ax.set_xticks(range(len(order)))
ax.set_xticklabels(ordered_names, rotation=45, ha='right')
ax.set_ylabel('Number of causal target genes')
ax.set_title('A. Regulatory burden by functional layer', fontweight='bold', loc='left')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1, fontsize=7)
ax.set_xlim(-0.5, len(order)-0.5)

# Add ancestry colors at bottom
for idx, name in enumerate(ordered_names):
    color = ancestry_colors.get(ancestry.get(name, 'Mixed'), '#999999')
    ax.add_patch(plt.Rectangle((idx-0.4, -15), 0.8, 12, color=color, clip_on=False))

# Panel B: TF family disruption proportions
ax2 = axes[1]

# Get disrupted families per hybrid
motif_merged = motifs.merge(causal[['gene_id', 'chr', 'pos']], on=['gene_id', 'chr', 'pos'])
top_fams = ['C2H2', 'NAC', 'MYB', 'bZIP', 'ERF', 'Dof', 'Trihelix', 'WRKY', 'bHLH', 'HOX']
fam_colors = plt.cm.tab20(np.linspace(0, 1, len(top_fams)))

fam_data = defaultdict(list)
for i in order:
    h = hybrids[i]
    founder = h.replace('B73x', '')
    fam_counts = defaultdict(int)
    for _, row in motif_merged.iterrows():
        geno_row = geno_idx.get((row['chr'], row['pos']))
        if geno_row is None:
            continue
        gt = str(geno_row.get(founder, '0|0'))
        if gt != '0|0' and pd.notna(row['disrupted_families']) and row['disrupted_families']:
            for f in str(row['disrupted_families']).split(','):
                f = f.strip()
                if f:
                    fam_counts[f] += 1
    for fam in top_fams:
        fam_data[fam].append(fam_counts.get(fam, 0))

bottom2 = np.zeros(len(order))
for k, fam in enumerate(top_fams):
    vals = np.array(fam_data[fam])
    ax2.bar(range(len(order)), vals, bottom=bottom2, color=fam_colors[k],
            label=fam, width=0.8, edgecolor='white', linewidth=0.3)
    bottom2 += vals

ax2.set_xticks(range(len(order)))
ax2.set_xticklabels(ordered_names, rotation=45, ha='right')
ax2.set_ylabel('TF motif disruptions')
ax2.set_title('B. TF family disruption profile', fontweight='bold', loc='left')
ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1, fontsize=7)
ax2.set_xlim(-0.5, len(order)-0.5)

plt.tight_layout()
plt.savefig('/Users/michael/perelyn/research/cell_biology/causal_bqtl_atlas/figures/hybrid_regulatory_burden.pdf')
plt.close()
print("Saved: hybrid_regulatory_burden.pdf")


# ═══════════════════════════════════════════════════════
# FIGURE 3: Gene sharing distribution + module heatmap
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Gene sharing histogram
ax = axes[0]
sharing = gene_matrix.sum(axis=0)
ax.hist(sharing, bins=range(1, 26), color='#377eb8', edgecolor='white', alpha=0.8)
ax.axvline(sharing.mean(), color='red', ls='--', lw=1.5, label=f'Mean = {sharing.mean():.1f}')
ax.set_xlabel('Number of hybrids carrying the causal bQTL')
ax.set_ylabel('Number of genes')
ax.set_title('A. Gene sharing across hybrids', fontweight='bold', loc='left')
ax.legend()

# Panel B: Module heatmap (gene clusters x hybrids)
ax = axes[1]
gene_dists = pdist(gene_matrix.T, metric='jaccard')
Z_genes = linkage(gene_dists, method='average')
gene_clusters = fcluster(Z_genes, t=10, criterion='maxclust')

# Sort genes by cluster, then by sharing within cluster
gene_order = []
for c in range(1, 11):
    members = [i for i in range(len(all_genes)) if gene_clusters[i] == c]
    members.sort(key=lambda x: -gene_matrix[:, x].sum())
    gene_order.extend(members)

# Reorder both axes
ordered_matrix = gene_matrix[np.ix_(order, gene_order)]
im = ax.imshow(ordered_matrix, cmap='Blues', aspect='auto', interpolation='nearest')
ax.set_yticks(range(len(ordered_names)))
ax.set_yticklabels(ordered_names, fontsize=7)
ax.set_xlabel('Genes (grouped by co-disruption module)')
ax.set_title('B. Per-hybrid gene presence (10 modules)', fontweight='bold', loc='left')

# Add module boundaries
cluster_ordered = [gene_clusters[i] for i in gene_order]
for i in range(1, len(cluster_ordered)):
    if cluster_ordered[i] != cluster_ordered[i-1]:
        ax.axvline(i-0.5, color='red', lw=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig('/Users/michael/perelyn/research/cell_biology/causal_bqtl_atlas/figures/gene_sharing_modules.pdf')
plt.close()
print("Saved: gene_sharing_modules.pdf")


# ═══════════════════════════════════════════════════════
# FIGURE 4: Circular-style hybrid comparison (polar plot)
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Each hybrid gets a spoke; plot multiple metrics as radar chart
metrics = {}
for i, h in enumerate(hybrids):
    name = short_names[i]
    genes = [all_genes[j] for j in range(len(all_genes)) if gene_matrix[i, j] == 1]

    # Metrics (normalized 0-1)
    n_genes = len(genes)
    sum_d = gene_effects[i].sum()
    n_drought_resp = sum(1 for g in genes if drought_map.get(g) in ['Strong up', 'Strong down'])
    n_tfs = sum(1 for g in genes if func_map.get(g) == 'Transcription Factor')
    n_signal = sum(1 for g in genes if func_map.get(g) == 'Signal Transduction')
    n_rewired = sum(1 for g in genes
                    if any(motifs[motifs['gene_id']==g]['n_disrupted'].values > 0) and
                       any(motifs[motifs['gene_id']==g]['n_created'].values > 0))

    metrics[name] = {
        'Genes': n_genes,
        'Effect (Σ|d|)': sum_d,
        'Drought resp.': n_drought_resp,
        'TF targets': n_tfs,
        'Signaling': n_signal,
    }

# Normalize each metric to 0-1
metric_names = list(list(metrics.values())[0].keys())
raw_vals = {m: [metrics[n][m] for n in ordered_names] for m in metric_names}
norm_vals = {}
for m in metric_names:
    vals = raw_vals[m]
    mn, mx = min(vals), max(vals)
    norm_vals[m] = [(v - mn) / (mx - mn + 1e-10) for v in vals]

# Radar chart
angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
angles += angles[:1]

# Plot a few representative hybrids
representatives = ['IL14H', 'A188', 'CML277', 'Mo17', 'Oh7b', 'Tx303']
rep_colors = ['#e41a1c', '#ff7f00', '#e41a1c', '#4daf4a', '#984ea3', '#e41a1c']

for name, color in zip(representatives, rep_colors):
    if name in ordered_names:
        idx = ordered_names.index(name)
        values = [norm_vals[m][idx] for m in metric_names]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=1.5, label=name, color=color, markersize=4)
        ax.fill(angles, values, alpha=0.05, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_names, fontsize=9)
ax.set_title('Hybrid regulatory fingerprints\n(normalized metrics)', fontsize=12, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.savefig('/Users/michael/perelyn/research/cell_biology/causal_bqtl_atlas/figures/hybrid_radar_fingerprints.pdf')
plt.close()
print("Saved: hybrid_radar_fingerprints.pdf")

print("\nAll figures generated successfully.")
