"""
25b_pathway_hybrid_heatmap.py

Creates a detailed KEGG pathway × hybrid heatmap showing which pathways
are disrupted in which hybrids, and identifies hybrid-variable vs
universally disrupted pathways.
"""
import sys
sys.path.insert(0, '..')
from pipeline_config import config

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gzip
import re
import json

FIG_DIR = config.figures_dir
RESULTS_DIR = config.results_dir

# ── Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
ml = pd.read_csv(RESULTS_DIR / 'hybrid_motif_linkage.csv')
ga = pd.read_csv(RESULTS_DIR / 'gene_atlas_728.csv')

# Build gene → hybrid sets
all_founders = set()
for h in ml['variant_hybrids'].dropna():
    all_founders.update(h.split('|'))
founders = sorted(all_founders)

gene_hybrid = {}
for _, row in ml.iterrows():
    gene = row['gene_id']
    var_hybrids = set(row['variant_hybrids'].split('|')) if pd.notna(row['variant_hybrids']) else set()
    gene_hybrid[gene] = var_hybrids

# Build gene → KEGG pathways (use ko only)
ga_kegg = ga.set_index('gene_id')['kegg_pathway'].to_dict()
gene_kegg = {}
for gene, kegg_str in ga_kegg.items():
    if pd.notna(kegg_str):
        pathways = set()
        for k in str(kegg_str).split(','):
            k = k.strip()
            if k.startswith('ko'):
                pathways.add(k)
        if pathways:
            gene_kegg[gene] = pathways

# KEGG pathway name mapping (abbreviated)
kegg_names = {
    'ko00010': 'Glycolysis',
    'ko00020': 'TCA cycle',
    'ko00040': 'Pentose/glucuronate',
    'ko00051': 'Fructose/mannose',
    'ko00052': 'Galactose',
    'ko00061': 'Fatty acid biosynthesis',
    'ko00071': 'Fatty acid degradation',
    'ko00100': 'Steroid biosynthesis',
    'ko00130': 'Ubiquinone biosynthesis',
    'ko00190': 'Oxidative phosphorylation',
    'ko00195': 'Photosynthesis',
    'ko00196': 'Photosynthesis antenna',
    'ko00230': 'Purine metabolism',
    'ko00240': 'Pyrimidine metabolism',
    'ko00250': 'Ala/Asp/Glu metabolism',
    'ko00260': 'Gly/Ser/Thr metabolism',
    'ko00270': 'Cys/Met metabolism',
    'ko00280': 'Val/Leu/Ile degradation',
    'ko00290': 'Val/Leu/Ile biosynthesis',
    'ko00300': 'Lys biosynthesis',
    'ko00310': 'Lys degradation',
    'ko00330': 'Arg/Pro metabolism',
    'ko00350': 'Tyr metabolism',
    'ko00360': 'Phe metabolism',
    'ko00380': 'Trp metabolism',
    'ko00400': 'Phe/Tyr/Trp biosynthesis',
    'ko00410': 'beta-Alanine metabolism',
    'ko00430': 'Taurine/hypotaurine',
    'ko00440': 'Phosphonate/phosphinate',
    'ko00450': 'Selenocompound',
    'ko00460': 'Cyanoamino acid',
    'ko00480': 'Glutathione metabolism',
    'ko00500': 'Starch/sucrose',
    'ko00510': 'N-glycan biosynthesis',
    'ko00520': 'Sugar nucleotide biosynthesis',
    'ko00531': 'Glycosaminoglycan deg.',
    'ko00561': 'Glycerolipid',
    'ko00564': 'Glycerophospholipid',
    'ko00590': 'Arachidonic acid',
    'ko00592': 'alpha-Linolenic acid',
    'ko00600': 'Sphingolipid',
    'ko00620': 'Pyruvate metabolism',
    'ko00630': 'Glyoxylate/dicarboxylate',
    'ko00640': 'Propanoate metabolism',
    'ko00650': 'Butanoate metabolism',
    'ko00670': 'One carbon pool (folate)',
    'ko00710': 'Carbon fixation (photosynthetic)',
    'ko00720': 'Carbon fixation (prokaryotic)',
    'ko00730': 'Thiamine metabolism',
    'ko00740': 'Riboflavin metabolism',
    'ko00750': 'Vitamin B6 metabolism',
    'ko00760': 'Nicotinate/nicotinamide',
    'ko00770': 'Pantothenate/CoA',
    'ko00780': 'Biotin metabolism',
    'ko00785': 'Lipoic acid metabolism',
    'ko00790': 'Folate biosynthesis',
    'ko00860': 'Porphyrin metabolism',
    'ko00900': 'Terpenoid backbone',
    'ko00906': 'Carotenoid biosynthesis',
    'ko00910': 'Nitrogen metabolism',
    'ko00920': 'Sulfur metabolism',
    'ko00940': 'Phenylpropanoid',
    'ko00941': 'Flavonoid biosynthesis',
    'ko00945': 'Stilbenoid biosynthesis',
    'ko00960': 'Tropane/piperidine',
    'ko00970': 'Aminoacyl-tRNA biosynthesis',
    'ko01100': 'Metabolic pathways',
    'ko01200': 'Carbon metabolism',
    'ko01210': 'Oxocarboxylic acid',
    'ko01212': 'Fatty acid metabolism',
    'ko01230': 'Biosynthesis of amino acids',
    'ko01240': 'Biosynthesis of cofactors',
    'ko02010': 'ABC transporters',
    'ko03008': 'Ribosome biogenesis',
    'ko03010': 'Ribosome',
    'ko03013': 'Nucleocytoplasmic transport',
    'ko03015': 'mRNA surveillance',
    'ko03018': 'RNA degradation',
    'ko03020': 'RNA polymerase',
    'ko03022': 'Basal transcription',
    'ko03030': 'DNA replication',
    'ko03040': 'Spliceosome',
    'ko03050': 'Proteasome',
    'ko03060': 'Protein export',
    'ko03410': 'Base excision repair',
    'ko03420': 'Nucleotide excision repair',
    'ko03430': 'Mismatch repair',
    'ko03440': 'Homologous recombination',
    'ko04011': 'MAPK signaling (plant)',
    'ko04016': 'MAPK signaling (yeast)',
    'ko04024': 'cAMP signaling',
    'ko04070': 'Phosphatidylinositol',
    'ko04075': 'Plant hormone signal',
    'ko04110': 'Cell cycle',
    'ko04111': 'Cell cycle (yeast)',
    'ko04120': 'Ubiquitin proteolysis',
    'ko04130': 'SNARE vesicular transport',
    'ko04136': 'Autophagy',
    'ko04140': 'Autophagy (animal)',
    'ko04141': 'Protein processing in ER',
    'ko04144': 'Endocytosis',
    'ko04146': 'Peroxisome',
    'ko04150': 'mTOR signaling',
    'ko04152': 'AMPK signaling',
    'ko04310': 'Wnt signaling',
    'ko04350': 'TGF-beta signaling',
    'ko04530': 'Tight junction',
    'ko04540': 'Gap junction',
    'ko04626': 'Plant-pathogen',
    'ko04714': 'Thermogenesis',
    'ko04723': 'Retrograde signaling',
    'ko04932': 'Non-alcoholic fatty liver',
    'ko04933': 'AGE-RAGE signaling',
}

# ── Build pathway × hybrid matrix ─────────────────────────────────────────
print("Building pathway × hybrid matrix...")
pathway_hybrid_genes = defaultdict(lambda: defaultdict(set))  # pathway -> founder -> {genes}

for gene, hybs in gene_hybrid.items():
    if gene not in gene_kegg:
        continue
    for pathway in gene_kegg[gene]:
        for h in hybs:
            if h in founders:
                pathway_hybrid_genes[pathway][h].add(gene)

# Count genes per pathway per hybrid
all_pathways = sorted(pathway_hybrid_genes.keys())
pw_matrix = np.zeros((len(all_pathways), len(founders)))
for i, pw in enumerate(all_pathways):
    for j, f in enumerate(founders):
        pw_matrix[i, j] = len(pathway_hybrid_genes[pw][f])

# Filter: keep pathways with ≥3 genes total and variability across hybrids
pw_totals = pw_matrix.sum(axis=1)
pw_cv = np.array([np.std(pw_matrix[i]) / max(np.mean(pw_matrix[i]), 0.01) for i in range(len(all_pathways))])
keep = (pw_totals >= 10) & (pw_cv > 0.05)
filtered_pathways = [p for i, p in enumerate(all_pathways) if keep[i]]
filtered_matrix = pw_matrix[keep]
print(f"Pathways total: {len(all_pathways)}, filtered (≥10 genes, variable): {len(filtered_pathways)}")

# ── Identify most variable pathways ───────────────────────────────────────
print("\n--- Most variable pathways across hybrids ---")
pw_stats = []
for i, pw in enumerate(filtered_pathways):
    row = filtered_matrix[i]
    pw_stats.append({
        'pathway': pw,
        'name': kegg_names.get(pw, pw),
        'mean': row.mean(),
        'std': row.std(),
        'cv': row.std() / max(row.mean(), 0.01),
        'min': row.min(),
        'max': row.max(),
        'range': row.max() - row.min(),
    })
pw_df = pd.DataFrame(pw_stats).sort_values('cv', ascending=False)
print(f"{'Pathway':<35s} {'Mean':>6s} {'CV':>6s} {'Min':>5s} {'Max':>5s}")
for _, row in pw_df.head(20).iterrows():
    print(f"{row['name'][:35]:<35s} {row['mean']:>6.1f} {row['cv']:>6.2f} {row['min']:>5.0f} {row['max']:>5.0f}")

# ── Classify pathways: conserved vs variable ──────────────────────────────
conserved_pw = pw_df[pw_df['cv'] < 0.15]
variable_pw = pw_df[pw_df['cv'] >= 0.20]
print(f"\nConserved pathways (CV<0.15): {len(conserved_pw)}")
print(f"Variable pathways (CV≥0.20): {len(variable_pw)}")

# ── Which hybrids are most distinctive? ───────────────────────────────────
print("\n--- Hybrid distinctiveness ---")
# Pairwise pathway profile correlation
pw_corr_mat = np.corrcoef(filtered_matrix.T)
mean_corr = []
for i, f in enumerate(founders):
    others = [pw_corr_mat[i, j] for j in range(len(founders)) if j != i]
    mean_corr.append((f, np.mean(others)))
mean_corr.sort(key=lambda x: x[1])
print("Least similar (most distinctive):")
for f, c in mean_corr[:5]:
    print(f"  {f}: mean pathway corr = {c:.3f}")
print("Most similar:")
for f, c in mean_corr[-5:]:
    print(f"  {f}: mean pathway corr = {c:.3f}")

# ── GO-level analysis: which GO terms are hybrid-variable? ────────────────
print("\n--- GO term variability across hybrids ---")
# Load GO from EntAP
gene_go = {}
with gzip.open(config.entap_gz, 'rt') as f:
    header = f.readline().strip().split('\t')
    col_map = {name: i for i, name in enumerate(header)}
    go_bp_cols = [col_map.get('UniProt GO Biological'), col_map.get('EggNOG GO Biological')]
    for line in f:
        fields = line.strip().split('\t')
        raw_id = fields[0]
        gene_id = re.sub(r'_T\d+$', '', raw_id)
        if gene_id not in gene_hybrid:
            continue
        go_terms = set()
        for col in go_bp_cols:
            if col and col < len(fields) and fields[col]:
                for t in fields[col].split(','):
                    t = t.strip()
                    if t.startswith('GO:'):
                        go_terms.add(t)
        if go_terms:
            gene_go[gene_id] = gene_go.get(gene_id, set()) | go_terms

# Build GO × hybrid matrix (binary: is this GO term represented in this hybrid?)
go_hybrid = defaultdict(set)  # go_term -> set of founders
for gene, hybs in gene_hybrid.items():
    if gene not in gene_go:
        continue
    for go_term in gene_go[gene]:
        for h in hybs:
            go_hybrid[go_term].add(h)

# GO terms present in all 24 vs subset
go_coverage = {go: len(hybs) for go, hybs in go_hybrid.items()}
universal_go = sum(1 for n in go_coverage.values() if n == 24)
variable_go = sum(1 for n in go_coverage.values() if n < 20)
print(f"Total GO BP terms across all hybrids: {len(go_hybrid)}")
print(f"Universal (all 24): {universal_go} ({100*universal_go/len(go_hybrid):.1f}%)")
print(f"Variable (<20 hybrids): {variable_go} ({100*variable_go/len(go_hybrid):.1f}%)")

# What are the most hybrid-variable GO terms?
go_var = [(go, n) for go, n in go_coverage.items() if n < 15]
go_var.sort(key=lambda x: x[1])
print(f"\nMost hybrid-restricted GO terms (present in <15/24 hybrids):")
for go, n in go_var[:15]:
    # Find which genes carry this GO
    go_genes = [g for g, terms in gene_go.items() if go in terms and g in gene_hybrid]
    print(f"  {go} (n={n}): {len(go_genes)} genes")

# ── FIGURES ───────────────────────────────────────────────────────────────
print("\n--- Generating figures ---")

fig = plt.figure(figsize=(20, 14))

# Panel A: KEGG pathway × hybrid heatmap
ax1 = fig.add_axes([0.06, 0.08, 0.55, 0.85])

# Select top 30 most variable + top 10 most common pathways
top_variable = pw_df.head(25)['pathway'].tolist()
top_common = pw_df.tail(10)['pathway'].tolist()
selected = list(dict.fromkeys(top_variable + top_common))  # preserve order, deduplicate

sel_idx = [filtered_pathways.index(p) for p in selected if p in filtered_pathways]
sel_matrix = filtered_matrix[sel_idx]
sel_names = [kegg_names.get(filtered_pathways[i], filtered_pathways[i])[:40]
             for i in sel_idx]

# Cluster hybrids
if sel_matrix.shape[1] > 2:
    dist_h = pdist(sel_matrix.T, metric='correlation')
    dist_h = np.clip(dist_h, 0, None)  # fix numerical issues
    Z_h = linkage(dist_h, method='ward')
    h_order = leaves_list(Z_h)
else:
    h_order = list(range(sel_matrix.shape[1]))

# Cluster pathways
if sel_matrix.shape[0] > 2:
    dist_p = pdist(sel_matrix, metric='correlation')
    dist_p = np.clip(dist_p, 0, None)
    Z_p = linkage(dist_p, method='ward')
    p_order = leaves_list(Z_p)
else:
    p_order = list(range(sel_matrix.shape[0]))

ordered_mat = sel_matrix[p_order][:, h_order]
ordered_founders = [founders[i] for i in h_order]
ordered_names = [sel_names[i] for i in p_order]

im = ax1.imshow(ordered_mat, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax1.set_xticks(range(len(ordered_founders)))
ax1.set_xticklabels(ordered_founders, rotation=90, fontsize=9)
ax1.set_yticks(range(len(ordered_names)))
ax1.set_yticklabels(ordered_names, fontsize=8)
ax1.set_title('A. KEGG pathway disruption across hybrids\n(gene count per pathway per hybrid)',
              fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax1, shrink=0.5, label='Genes disrupted', pad=0.02)

# Panel B: Convergence bar (simplified)
ax2 = fig.add_axes([0.68, 0.55, 0.28, 0.35])
levels = ['Gene', 'GO BP', 'GO MF', 'KEGG']
jacs = [0.285, 0.584, 0.548, 0.667]
colors = ['#E53935', '#FF9800', '#4CAF50', '#1976D2']
bars = ax2.bar(range(4), jacs, color=colors, edgecolor='white', linewidth=1.5)
for bar, j in zip(bars, jacs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{j:.3f}', ha='center', fontsize=11, fontweight='bold')
ax2.set_xticks(range(4))
ax2.set_xticklabels(levels, fontsize=11)
ax2.set_ylabel('Pairwise Jaccard', fontsize=12)
ax2.set_title('B. Functional convergence', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 0.85)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# Add arrow showing convergence
ax2.annotate('', xy=(3.3, 0.667), xytext=(0.3, 0.285),
             arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))
ax2.text(1.8, 0.22, '2.3× convergence', fontsize=10, color='gray',
         fontstyle='italic', rotation=15)

# Panel C: GO term coverage histogram
ax3 = fig.add_axes([0.68, 0.08, 0.28, 0.35])
coverage_vals = list(go_coverage.values())
ax3.hist(coverage_vals, bins=range(1, 26), color='#7B1FA2', edgecolor='white',
         linewidth=0.5, alpha=0.85)
ax3.axvline(24, color='red', linestyle='--', alpha=0.7, label=f'All 24: {universal_go}')
ax3.set_xlabel('Number of hybrids with GO term', fontsize=12)
ax3.set_ylabel('Number of GO terms', fontsize=12)
ax3.set_title('C. GO term sharing across hybrids', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.savefig(FIG_DIR / 'pathway_hybrid_heatmap.pdf', bbox_inches='tight', dpi=150)
plt.savefig(FIG_DIR / 'pathway_hybrid_heatmap.png', bbox_inches='tight', dpi=150)
print("Saved: pathway_hybrid_heatmap.pdf")
plt.close()

# Save pathway variability table
pw_df.to_csv(RESULTS_DIR / 'pathway_variability.csv', index=False)
print("Saved: pathway_variability.csv")

print("\nDone.")
