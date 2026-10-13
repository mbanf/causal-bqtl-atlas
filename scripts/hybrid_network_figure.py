"""
Per-hybrid TF feedback network figure.
Shows: shared regulatory circuit topology across all ancestry groups.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict
import math

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# Load data
causal = pd.read_csv('/Users/michael/perelyn/research/cell_biology/causal_bqtl_atlas/results/causal_bqtl_728.csv')
atlas = pd.read_csv('/Users/michael/perelyn/research/cell_biology/causal_bqtl_atlas/results/gene_atlas_728.csv')
motifs = pd.read_csv('/Users/michael/perelyn/research/cell_biology/causal_bqtl_atlas/results/allele_specific_motifs.csv')
geno = pd.read_csv('/Users/michael/perelyn/research/cell_biology/shared_data/processed/nam_founder_genotypes_at_bqtl.tsv', sep='\t')

nam_founders = list(geno.columns[3:])
geno_idx = {}
for _, row in geno.iterrows():
    geno_idx[(row['chr'], row['pos'])] = row

func_map = atlas.set_index('gene_id')['signaling_layer'].to_dict()
tf_fam_map = atlas.set_index('gene_id')['tf_families'].to_dict()

all_genes = sorted(causal['gene_id'].unique())
hybrids = sorted([f'B73x{f}' for f in nam_founders])
short_names = [h.replace('B73x','') for h in hybrids]

gene_matrix = np.zeros((len(hybrids), len(all_genes)))
for j, gene in enumerate(all_genes):
    crow = causal[causal['gene_id'] == gene].iloc[0]
    geno_row = geno_idx.get((crow['chr'], crow['pos']))
    if geno_row is None:
        continue
    for i, h in enumerate(hybrids):
        founder = h.replace('B73x','')
        if str(geno_row.get(founder, '0|0')) != '0|0':
            gene_matrix[i, j] = 1

ancestry = {
    'A188': 'Flint', 'HP301': 'Popcorn', 'IL14H': 'Iodent', 'P39': 'Popcorn',
    'A619': 'Stiff Stalk', 'B97': 'Stiff Stalk', 'Mo17': 'Lancaster',
    'Ms71': 'Stiff Stalk', 'Oh43': 'Iodent', 'Oh7b': 'Iodent', 'W22': 'Stiff Stalk',
    'CML103': 'Tropical', 'CML247': 'Tropical', 'CML277': 'Tropical',
    'CML322': 'Tropical', 'CML333': 'Tropical', 'CML69': 'Tropical',
    'Ki11': 'Tropical', 'Ki3': 'Tropical', 'NC358': 'Tropical',
    'Tx303': 'Tropical', 'M162W': 'Mixed', 'Mo18W': 'Tropical', 'Ky21': 'Lancaster'
}
ancestry_colors = {
    'Tropical': '#e41a1c', 'Stiff Stalk': '#377eb8', 'Lancaster': '#4daf4a',
    'Iodent': '#984ea3', 'Flint': '#ff7f00', 'Popcorn': '#a65628', 'Mixed': '#999999'
}

# Build feedback edges per hybrid
tf_genes = set(atlas[atlas['signaling_layer'] == 'Transcription Factor']['gene_id'])

# Count edges: (disrupting_family → target_family) → set of hybrids
edge_hybrids = defaultdict(set)
edge_genes = defaultdict(set)  # which genes implement each edge

for _, row in causal.iterrows():
    gene = row['gene_id']
    if gene not in tf_genes:
        continue
    geno_row = geno_idx.get((row['chr'], row['pos']))
    if geno_row is None:
        continue

    mrow = motifs[motifs['gene_id'] == gene]
    disrupted_fams = []
    if len(mrow) > 0 and pd.notna(mrow.iloc[0].get('disrupted_families')):
        disrupted_fams = [f.strip() for f in str(mrow.iloc[0]['disrupted_families']).split(',') if f.strip()]

    target_fam = ''
    tf_str = tf_fam_map.get(gene, '')
    if pd.notna(tf_str) and tf_str:
        target_fam = str(tf_str).split(',')[0].strip()

    if not target_fam or target_fam == 'NONE':
        continue

    for i, h in enumerate(hybrids):
        founder = h.replace('B73x','')
        if str(geno_row.get(founder, '0|0')) != '0|0':
            for df in disrupted_fams:
                edge_hybrids[(df, target_fam)].add(h)
                edge_genes[(df, target_fam)].add(gene)


# ═══════════════════════════════════════════════════════
# FIGURE: Multi-panel network + systems overview
# ═══════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

# ── Panel A: Circular TF feedback network ──
ax = fig.add_subplot(gs[0, 0])

# Get TF families involved in feedback
fam_set = set()
for (df, tf) in edge_hybrids:
    fam_set.add(df)
    fam_set.add(tf)
fam_list = sorted(fam_set)
n_fams = len(fam_list)

# Position families in a circle
angles = np.linspace(0, 2*np.pi, n_fams, endpoint=False)
radius = 0.35
cx, cy = 0.5, 0.5
positions = {}
for i, fam in enumerate(fam_list):
    x = cx + radius * np.cos(angles[i] - np.pi/2)
    y = cy + radius * np.sin(angles[i] - np.pi/2)
    positions[fam] = (x, y)

# Draw edges (width = number of hybrids sharing the edge)
for (df, tf), hybs in edge_hybrids.items():
    if df == tf:  # autoregulatory
        continue
    x1, y1 = positions[df]
    x2, y2 = positions[tf]
    n = len(hybs)
    alpha = min(0.8, 0.1 + n/24 * 0.7)
    lw = 0.3 + n/24 * 2.5

    # Slight curve
    mx = (x1+x2)/2 + 0.02 * (y2-y1)
    my = (y1+y2)/2 - 0.02 * (x2-x1)

    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#333333',
                               lw=lw, alpha=alpha,
                               connectionstyle='arc3,rad=0.15'))

# Draw nodes
for fam in fam_list:
    x, y = positions[fam]
    # Count total edges involving this family
    n_edges = sum(len(h) for (df, tf), h in edge_hybrids.items() if df == fam or tf == fam)
    size = 200 + n_edges * 3

    # Check if autoregulatory
    auto_n = len(edge_hybrids.get((fam, fam), set()))

    if auto_n > 0:
        ax.scatter(x, y, s=size, c='#e41a1c', edgecolors='black', linewidths=1.5, zorder=5)
    else:
        ax.scatter(x, y, s=size, c='#377eb8', edgecolors='black', linewidths=1, zorder=5)

    # Label
    label_x = cx + (radius + 0.06) * np.cos(angles[fam_list.index(fam)] - np.pi/2)
    label_y = cy + (radius + 0.06) * np.sin(angles[fam_list.index(fam)] - np.pi/2)
    ha = 'left' if label_x > cx else 'right'
    if abs(label_x - cx) < 0.05:
        ha = 'center'
    ax.text(label_x, label_y, fam, ha=ha, va='center', fontsize=7, fontweight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('A. Shared TF feedback network\n(edge width ∝ hybrid sharing)', fontweight='bold', loc='left')

# Legend
ax.scatter(0.02, 0.08, s=100, c='#e41a1c', edgecolors='black', linewidths=1)
ax.text(0.06, 0.08, 'Autoregulatory', fontsize=7, va='center')
ax.scatter(0.02, 0.04, s=100, c='#377eb8', edgecolors='black', linewidths=1)
ax.text(0.06, 0.04, 'Cross-regulatory only', fontsize=7, va='center')


# ── Panel B: Edge sharing histogram ──
ax2 = fig.add_subplot(gs[0, 1])
sharing_counts = [len(h) for h in edge_hybrids.values()]
ax2.hist(sharing_counts, bins=range(1, 26), color='#377eb8', edgecolor='white', alpha=0.8)
ax2.axvline(np.mean(sharing_counts), color='red', ls='--', lw=1.5,
            label=f'Mean = {np.mean(sharing_counts):.1f}')
ax2.set_xlabel('Number of hybrids sharing the feedback edge')
ax2.set_ylabel('Number of feedback edges')
ax2.set_title('B. Feedback edge sharing across hybrids', fontweight='bold', loc='left')
ax2.legend()

# Annotate: top shared
auto_edges = [(k, len(v)) for k, v in edge_hybrids.items() if k[0] == k[1]]
auto_edges.sort(key=lambda x: -x[1])
text = "Top shared (autoregulatory):\n"
for (df, tf), n in auto_edges[:4]:
    text += f"  {df}→{tf}: {n}/24 hybrids\n"
ax2.text(0.98, 0.98, text, transform=ax2.transAxes, va='top', ha='right',
         fontsize=7, family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# ── Panel C: Ancestry-specific genes per group ──
ax3 = fig.add_subplot(gs[1, 0])

# Compute ancestry-specific genes
anc_groups = ['Tropical', 'Stiff Stalk', 'Iodent', 'Lancaster', 'Popcorn']
anc_specific = {}
anc_shared = {}

for grp in anc_groups:
    members = [i for i, n in enumerate(short_names) if ancestry.get(n) == grp]
    if len(members) < 2:
        continue
    group_freq = gene_matrix[members].mean(axis=0)
    other_idx = [i for i in range(len(hybrids)) if i not in members]
    other_freq = gene_matrix[other_idx].mean(axis=0)

    specific = ((group_freq >= 0.5) & (other_freq < 0.2)).sum()
    shared_broadly = (gene_matrix.sum(axis=0) >= 15).sum()
    anc_specific[grp] = specific
    anc_shared[grp] = len(members)

# Bar chart
x = range(len(anc_groups))
bars = ax3.bar(x, [anc_specific.get(g, 0) for g in anc_groups],
               color=[ancestry_colors[g] for g in anc_groups],
               edgecolor='black', linewidth=0.5)
ax3.set_xticks(x)
ax3.set_xticklabels([f'{g}\n(n={sum(1 for v in ancestry.values() if v==g)})' for g in anc_groups],
                     fontsize=8)
ax3.set_ylabel('Ancestry-specific genes')
ax3.set_title('C. Private regulatory targets per ancestry group', fontweight='bold', loc='left')

# Add values on bars
for bar, grp in zip(bars, anc_groups):
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.0f}', ha='center', va='bottom', fontsize=9)


# ── Panel D: Stacked: core vs private modules ──
ax4 = fig.add_subplot(gs[1, 1])

from scipy.cluster.hierarchy import linkage as hlink, fcluster as fclust
from scipy.spatial.distance import pdist as pdst

gene_dists = pdst(gene_matrix.T, metric='jaccard')
Z = hlink(gene_dists, method='average')
clusters = fclust(Z, t=10, criterion='maxclust')

# Classify modules by ancestry affinity
core_module = 0
core_n = 0
anc_modules = defaultdict(int)

for c in range(1, 11):
    members = [i for i in range(len(all_genes)) if clusters[i] == c]
    n = len(members)
    sharing = gene_matrix[:, members].sum(axis=0).mean()

    if n > 100:  # core module
        core_module = c
        core_n = n
    else:
        # Which ancestry dominates?
        coverage = gene_matrix[:, members].mean(axis=1)
        for grp in anc_groups:
            grp_idx = [i for i, sn in enumerate(short_names) if ancestry.get(sn) == grp]
            grp_cov = coverage[grp_idx].mean()
            if grp_cov > 0.4:
                anc_modules[grp] += n
                break
        else:
            anc_modules['Mixed'] += n

# Pie chart: core vs ancestry-specific modules
sizes = [core_n]
labels = [f'Core\n({core_n} genes)']
colors = ['#d9d9d9']
for grp in anc_groups:
    if anc_modules.get(grp, 0) > 0:
        sizes.append(anc_modules[grp])
        labels.append(f'{grp}\n({anc_modules[grp]})')
        colors.append(ancestry_colors[grp])
if anc_modules.get('Mixed', 0) > 0:
    sizes.append(anc_modules['Mixed'])
    labels.append(f'Mixed\n({anc_modules["Mixed"]})')
    colors.append('#999999')

wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                                     startangle=90, pctdistance=0.85,
                                     wedgeprops=dict(linewidth=1, edgecolor='white'))
for t in texts:
    t.set_fontsize(8)
for t in autotexts:
    t.set_fontsize(7)

ax4.set_title('D. Gene module composition:\ncore vs ancestry-private', fontweight='bold', loc='left')

plt.savefig('/Users/michael/perelyn/research/cell_biology/causal_bqtl_atlas/figures/hybrid_network_systems.pdf')
plt.close()
print("Saved: hybrid_network_systems.pdf")


# ═══════════════════════════════════════════════════════
# FIGURE: Summary "distributed OS" diagram
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Regulatory burden uniformity
ax = axes[0]
genes_per_hybrid = gene_matrix.sum(axis=1)
effects_per_hybrid = np.array([gene_matrix[i] @ np.array([causal[causal['gene_id']==g].iloc[0]['abs_d']
                                                           for g in all_genes])
                               for i in range(len(hybrids))])

# Sort by dendrogram order
from scipy.cluster.hierarchy import dendrogram as dg
hybrid_dists = pdst(gene_matrix, metric='jaccard')
Z_h = hlink(hybrid_dists, method='average')
dn = dg(Z_h, no_plot=True)
order = dn['leaves']

ax.barh(range(len(order)), [genes_per_hybrid[i] for i in order],
        color=[ancestry_colors.get(ancestry.get(short_names[i], 'Mixed'), '#999999') for i in order],
        edgecolor='white', linewidth=0.3)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([short_names[i] for i in order], fontsize=7)
ax.axvline(genes_per_hybrid.mean(), color='black', ls='--', lw=1)
ax.set_xlabel('Number of causal target genes')
ax.set_title('A. Uniform regulatory burden\n(all hybrids ≈ 452 genes)', fontweight='bold', loc='left')
ax.set_xlim(400, 510)

# Panel B: TF profile correlation (pairwise Spearman of disruption profiles)
ax = axes[1]

# Build TF disruption count matrix
motif_merged = motifs.copy()
top_fams = ['C2H2', 'NAC', 'MYB', 'bZIP', 'ERF', 'Dof', 'Trihelix', 'WRKY',
            'bHLH', 'HOX', 'C3H', 'MYB-related', 'AP2', 'MIKC-MADS', 'HSF']

fam_matrix = np.zeros((len(hybrids), len(top_fams)))
for j, gene in enumerate(all_genes):
    crow = causal[causal['gene_id'] == gene].iloc[0]
    geno_row = geno_idx.get((crow['chr'], crow['pos']))
    if geno_row is None:
        continue
    mrow = motifs[motifs['gene_id'] == gene]
    if len(mrow) == 0:
        continue
    dis_fams = []
    if pd.notna(mrow.iloc[0].get('disrupted_families')):
        dis_fams = [f.strip() for f in str(mrow.iloc[0]['disrupted_families']).split(',') if f.strip()]
    for i, h in enumerate(hybrids):
        founder = h.replace('B73x','')
        if str(geno_row.get(founder, '0|0')) != '0|0':
            for f in dis_fams:
                if f in top_fams:
                    fam_matrix[i, top_fams.index(f)] += 1

# Normalize
fam_props = fam_matrix / (fam_matrix.sum(axis=1, keepdims=True) + 1e-10)

# Pairwise Spearman
from scipy.stats import spearmanr
spearman_matrix = np.zeros((len(hybrids), len(hybrids)))
for i in range(len(hybrids)):
    for j in range(len(hybrids)):
        rho, _ = spearmanr(fam_props[i], fam_props[j])
        spearman_matrix[i, j] = rho

# Reorder
sp_ordered = spearman_matrix[np.ix_(order, order)]
im = ax.imshow(sp_ordered, cmap='RdYlGn', vmin=0.9, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(order)))
ax.set_xticklabels([short_names[i] for i in order], rotation=90, fontsize=6)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([short_names[i] for i in order], fontsize=6)
plt.colorbar(im, ax=ax, shrink=0.7, label='Spearman ρ')
ax.set_title('B. TF disruption profile similarity\n(all pairs ρ > 0.92)', fontweight='bold', loc='left')

# Panel C: Gene sharing spectrum
ax = axes[2]
sharing = gene_matrix.sum(axis=0)

# Color by functional layer
colors_by_func = []
func_color_map = {
    'Signal Transduction': '#e41a1c', 'Transcription Factor': '#377eb8',
    'Metabolism': '#4daf4a', 'Protein Processing': '#ff7f00',
    'Transport': '#984ea3', 'Unknown': '#d9d9d9'
}
for g in all_genes:
    f = func_map.get(g, 'Unknown')
    colors_by_func.append(func_color_map.get(f, '#d9d9d9'))

# Sort by sharing
sort_idx = np.argsort(sharing)
ax.scatter(range(len(all_genes)), sharing[sort_idx], s=3,
           c=[colors_by_func[i] for i in sort_idx], alpha=0.5)
ax.axhline(sharing.mean(), color='black', ls='--', lw=1, label=f'Mean = {sharing.mean():.1f}')
ax.set_xlabel('Genes (ranked by sharing)')
ax.set_ylabel('Number of hybrids')
ax.set_title('C. Gene sharing spectrum\n(functional layer coloring)', fontweight='bold', loc='left')
ax.legend(fontsize=7)

# Add legend for functional colors
patches = [mpatches.Patch(color=c, label=l) for l, c in
           [('Signaling', '#e41a1c'), ('TF', '#377eb8'), ('Metabolism', '#4daf4a'),
            ('Protein QC', '#ff7f00'), ('Unknown', '#d9d9d9')]]
ax.legend(handles=patches, loc='upper left', fontsize=6, ncol=2)

plt.tight_layout()
plt.savefig('/Users/michael/perelyn/research/cell_biology/causal_bqtl_atlas/figures/distributed_os_summary.pdf')
plt.close()
print("Saved: distributed_os_summary.pdf")
