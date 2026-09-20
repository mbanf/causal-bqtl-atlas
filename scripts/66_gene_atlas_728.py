#!/usr/bin/env python3
"""
66_gene_atlas_728.py

Deep characterization of ALL 728 causal bQTL target genes:
1. Gene × TF family heatmap (which motif disrupted at each gene's bQTL)
2. Co-regulation analysis (genes sharing same disrupted TF family)
3. Master regulator identification (TF families regulating coherent gene sets)
4. Signaling pathway reconstruction (receptor→kinase→TF→target chains)
5. Protein functional network (what do these 728 proteins DO together?)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from collections import defaultdict, Counter
import gzip
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

BASE = str(Path(__file__).resolve().parent.parent)
RESULTS = f'{BASE}/results'
FIGS = f'{BASE}/figures/paper_figures'
os.makedirs(FIGS, exist_ok=True)

PY = str(Path(__file__).resolve().parent.parent.parent / 'bqtl_predict' / '.venv' / 'bin' / 'python')

# ── Load data ──
reg = pd.read_csv(f'{RESULTS}/regulatory_map_728.csv')
causal = pd.read_csv(f'{RESULTS}/causal_bqtl_728.csv')
brm = pd.read_csv(f'{RESULTS}/bound_region_motifs.csv')
expr = pd.read_csv(f'{BASE}/data/processed/engelhorn_ww_vs_ds_expression.tsv', sep='\t')

# EntAP annotations
entap = pd.read_csv(
    f'{BASE}/data/raw/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1_entap_results.tsv.gz',
    sep='\t', compression='gzip', low_memory=False
)
entap['gene_id'] = entap['Query Sequence'].str.replace(r'_T\d+$', '', regex=True)
entap_genes = entap.drop_duplicates(subset='gene_id', keep='first').set_index('gene_id')

print(f"Loaded: {len(reg)} reg map, {len(causal)} causal bQTL, {len(entap_genes)} EntAP genes")

# ═══════════════════════════════════════════════════════════════
# STEP 1: Build Gene × TF Family Binary Matrix
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 1: Gene × TF Family Matrix")
print("="*70)

# Get variant_families per bQTL position
brm_agg = brm.groupby(['chr','pos']).agg({'variant_families': 'first'}).reset_index()
merged = causal.merge(brm_agg, on=['chr','pos'], how='left')
# Also merge in functional info from reg
merged = merged.merge(
    reg[['gene_id','families_at_variant','Description','EggNOG COG Description',
         'log2fc_DS_vs_WW','func_category','drought_cat']].drop_duplicates(subset='gene_id'),
    on='gene_id', how='left'
)

# Use variant_families (from bound_region_motifs) preferentially, fall back to families_at_variant
def get_families(row):
    vf = str(row.get('variant_families', ''))
    if vf and vf != 'nan':
        return sorted(set(f.strip() for f in vf.split(',') if f.strip()))
    fav = str(row.get('families_at_variant', ''))
    if fav and fav != 'nan':
        return sorted(set(f.strip() for f in fav.split('|') if f.strip()))
    return []

merged['tf_families_list'] = merged.apply(get_families, axis=1)
merged['n_families'] = merged['tf_families_list'].apply(len)
has_family = merged['n_families'] > 0
print(f"Genes with TF family at variant: {has_family.sum()} / {len(merged)} ({100*has_family.mean():.1f}%)")

# Collect all TF families
all_families = sorted(set(f for fams in merged['tf_families_list'] for f in fams))
print(f"TF families represented: {len(all_families)}")

# Build binary matrix
gene_ids = merged['gene_id'].values
matrix = np.zeros((len(merged), len(all_families)), dtype=int)
for i, fams in enumerate(merged['tf_families_list']):
    for f in fams:
        j = all_families.index(f)
        matrix[i, j] = 1

print(f"Matrix shape: {matrix.shape}")
print(f"Non-zero entries: {matrix.sum()} ({100*matrix.sum()/(matrix.shape[0]*matrix.shape[1]):.1f}%)")

# Per-family counts
fam_totals = matrix.sum(axis=0)
print("\nGenes per TF family (disrupted at variant):")
for j, fam in enumerate(all_families):
    print(f"  {fam}: {fam_totals[j]}")


# ═══════════════════════════════════════════════════════════════
# STEP 2: Co-regulation — genes sharing same disrupted motif
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 2: Co-regulation Analysis")
print("="*70)

# For each TF family: which genes have ONLY that family disrupted (unique motif)?
single_motif = merged[merged['n_families'] == 1].copy()
single_motif['sole_family'] = single_motif['tf_families_list'].apply(lambda x: x[0])
print(f"\nGenes with exactly 1 TF family disrupted: {len(single_motif)} ({100*len(single_motif)/len(merged):.1f}%)")

sole_family_counts = single_motif['sole_family'].value_counts()
print("\nSole-family gene groups (co-regulated by same motif):")
for fam, count in sole_family_counts.items():
    if count >= 3:
        genes = single_motif[single_motif['sole_family'] == fam]
        funcs = genes['func_category'].value_counts().head(3)
        func_str = ', '.join(f"{f}({c})" for f, c in funcs.items())
        mean_d = genes['abs_d'].mean()
        print(f"  {fam}: {count} genes, mean |d|={mean_d:.2f}, top funcs: {func_str}")

# Co-regulation groups: genes sharing exact same TF family combination
combo_counts = merged['tf_families_list'].apply(tuple).value_counts()
print(f"\nUnique TF family combinations: {len(combo_counts)}")
print("Top combinations (co-regulated gene groups):")
for combo, count in combo_counts.head(20).items():
    if count >= 3 and len(combo) > 0:
        genes_in = merged[merged['tf_families_list'].apply(tuple) == combo]
        funcs = genes_in['func_category'].value_counts().head(3)
        func_str = ', '.join(f"{f}({c})" for f, c in funcs.items())
        print(f"  {','.join(combo)}: {count} genes — {func_str}")


# ═══════════════════════════════════════════════════════════════
# STEP 3: Master Regulator Analysis
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 3: Master Regulator Analysis")
print("="*70)

# For each TF family: how many genes does it regulate?
# And do those genes form coherent functional groups?
print("\nTF Family → Target Gene Functional Coherence:")
for j, fam in enumerate(all_families):
    target_idx = np.where(matrix[:, j] == 1)[0]
    if len(target_idx) < 5:
        continue
    target_genes = merged.iloc[target_idx]
    n_targets = len(target_genes)

    # Functional coherence: what fraction share the dominant category?
    func_dist = target_genes['func_category'].value_counts()
    top_func = func_dist.index[0] if len(func_dist) > 0 else 'Unknown'
    top_frac = func_dist.iloc[0] / n_targets if len(func_dist) > 0 else 0

    # Drought coherence: what fraction respond same direction?
    drought_dist = target_genes['drought_cat'].value_counts()

    # Expression level
    mean_d = target_genes['abs_d'].mean()

    # Are targets of this TF enriched for specific categories?
    func_enrich = []
    for cat in target_genes['func_category'].unique():
        if cat == 'Unknown' or pd.isna(cat):
            continue
        a = (target_genes['func_category'] == cat).sum()
        b = n_targets - a
        c = (merged['func_category'] == cat).sum() - a
        d = len(merged) - n_targets - c
        if a >= 3 and c > 0:
            odds, p = stats.fisher_exact([[a, b], [c, d]])
            if p < 0.05:
                func_enrich.append((cat, odds, p, a))

    enrich_str = '; '.join(f"{cat}(OR={o:.1f},n={n})" for cat,o,p,n in sorted(func_enrich, key=lambda x: x[2]))

    print(f"\n  {fam} ({n_targets} targets):")
    print(f"    Mean |d| = {mean_d:.2f}")
    print(f"    Top function: {top_func} ({100*top_frac:.0f}%)")
    print(f"    Drought: {dict(drought_dist.head(3))}")
    if func_enrich:
        print(f"    Enriched: {enrich_str}")

# Identify TF genes among the 728 (master regulators: TFs whose binding is disrupted
# AND they themselves are targets of other bQTL)
print("\n── TF genes that are themselves bQTL targets (cascade nodes) ──")
tf_targets = merged[merged['func_category'] == 'Transcription factor'].copy()
print(f"TF genes in 728: {len(tf_targets)}")
for _, row in tf_targets.iterrows():
    desc = str(row.get('Description', ''))[:80]
    fams = ','.join(row['tf_families_list']) if row['tf_families_list'] else 'none'
    drought = row.get('drought_cat', '')
    print(f"  {row['gene_id']}: |d|={row['abs_d']:.2f}, disrupted by [{fams}], drought={drought}")
    print(f"    {desc}")


# ═══════════════════════════════════════════════════════════════
# STEP 4: Signaling Pathway Reconstruction
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 4: Signaling Pathway Reconstruction")
print("="*70)

# Classify ALL 728 proteins into signaling layers
SIGNALING_LAYERS = {
    'Receptor/Sensor': ['receptor', 'sensor', 'phototropin', 'phytochrome', 'transmembrane receptor',
                        'LRR receptor', 'kinase receptor', 'receptor-like'],
    'Membrane/Transport': ['transporter', 'channel', 'pump', 'carrier', 'aquaporin', 'ABC transporter',
                          'permease', 'membrane protein', 'ion channel'],
    'Signal Transduction': ['kinase', 'phosphatase', 'GTPase', 'G protein', 'calcium', 'calmodulin',
                           'MAP kinase', 'MAPK', 'CDK', 'CaM', 'signal transduction', 'phosphorylation'],
    'Second Messenger': ['adenylate cyclase', 'phospholipase', 'inositol', 'cyclic nucleotide',
                        'calcium-dependent'],
    'Protein Processing': ['chaperone', 'HSP', 'heat shock', 'proteasome', 'ubiquitin', 'protease',
                          'peptidase', 'folding', 'Clp protease'],
    'Transcription Factor': ['transcription factor', 'zinc finger', 'MYB', 'WRKY', 'NAC', 'bHLH',
                            'bZIP', 'ERF', 'AP2', 'MADS', 'homeobox', 'HD-ZIP', 'DNA-binding'],
    'Chromatin/Epigenetic': ['histone', 'methyltransferase', 'acetyltransferase', 'chromatin',
                            'deacetylase', 'SWI/SNF', 'polycomb'],
    'Translation': ['ribosom', 'translation', 'tRNA', 'elongation factor', 'initiation factor',
                   'aminoacyl'],
    'Metabolism': ['synthase', 'dehydrogenase', 'oxidase', 'reductase', 'kinase', 'transferase',
                  'hydrolase', 'isomerase', 'ligase', 'cytochrome P450'],
    'Redox/Stress': ['peroxidase', 'thioredoxin', 'glutaredoxin', 'superoxide', 'catalase',
                    'ascorbate', 'glutathione'],
    'Hormone': ['auxin', 'ABA', 'gibberellin', 'ethylene', 'jasmonate', 'brassinosteroid',
               'cytokinin', 'salicylic acid', 'IAA', 'ARF', 'PYR', 'GID']
}

def classify_layer(gene_id, desc, cog_desc, func_cat):
    """Classify protein into signaling layer based on description keywords."""
    text = f"{desc} {cog_desc}".lower()
    layers = []
    for layer, keywords in SIGNALING_LAYERS.items():
        for kw in keywords:
            if kw.lower() in text:
                layers.append(layer)
                break
    if not layers:
        if func_cat and func_cat != 'Unknown' and not pd.isna(func_cat):
            layers = [func_cat]
        else:
            layers = ['Unknown']
    return layers

# Classify all 728 genes
layer_data = []
for _, row in merged.iterrows():
    gene_id = row['gene_id']
    desc = str(row.get('Description', ''))
    cog = str(row.get('EggNOG COG Description', ''))
    func = row.get('func_category', '')

    # Get additional info from EntAP
    go_bio = ''
    kegg_path = ''
    if gene_id in entap_genes.index:
        go_bio = str(entap_genes.loc[gene_id, 'UniProt GO Biological'])
        kegg_path = str(entap_genes.loc[gene_id, 'EggNOG KEGG Pathway'])

    layers = classify_layer(gene_id, desc, cog, func)

    layer_data.append({
        'gene_id': gene_id,
        'chr': row['chr'],
        'pos': row['pos'],
        'abs_d': row['abs_d'],
        'cohens_d': row['cohens_d'],
        'description': desc[:120],
        'cog_description': cog[:80],
        'func_category': func,
        'signaling_layer': layers[0],  # Primary layer
        'all_layers': '|'.join(layers),
        'tf_families': ','.join(row['tf_families_list']),
        'n_families': row['n_families'],
        'log2fc_drought': row.get('log2fc_DS_vs_WW', np.nan),
        'drought_cat': row.get('drought_cat', ''),
        'go_biological': go_bio[:200] if go_bio != 'nan' else '',
        'kegg_pathway': kegg_path[:200] if kegg_path != 'nan' else '',
        'n_variant_hybrids': row['n_variant'],
        'n_ref_hybrids': row['n_ref']
    })

atlas = pd.DataFrame(layer_data)

# Layer distribution
print("\nSignaling layer distribution of 728 proteins:")
layer_counts = atlas['signaling_layer'].value_counts()
for layer, count in layer_counts.items():
    pct = 100 * count / len(atlas)
    mean_d = atlas.loc[atlas['signaling_layer'] == layer, 'abs_d'].mean()
    print(f"  {layer}: {count} ({pct:.1f}%), mean |d|={mean_d:.2f}")

# Signaling chains: find genes that form receptor→kinase→TF cascades
print("\n── Signaling Chain Analysis ──")
receptors = atlas[atlas['signaling_layer'] == 'Receptor/Sensor']
kinases = atlas[atlas['signaling_layer'] == 'Signal Transduction']
tfs = atlas[atlas['signaling_layer'] == 'Transcription Factor']
chaperones = atlas[atlas['signaling_layer'] == 'Protein Processing']
transporters = atlas[atlas['signaling_layer'] == 'Membrane/Transport']
hormone = atlas[atlas['signaling_layer'] == 'Hormone']

print(f"  Receptors/Sensors: {len(receptors)}")
print(f"  Signal Transduction (kinases etc): {len(kinases)}")
print(f"  Transcription Factors: {len(tfs)}")
print(f"  Protein Processing (chaperones): {len(chaperones)}")
print(f"  Membrane/Transport: {len(transporters)}")
print(f"  Hormone signaling: {len(hormone)}")

# Do receptor/kinase/TF genes share the SAME disrupted TF family?
# If so, a single TF's binding variation ripples through the entire cascade
print("\n── Shared TF motifs across signaling layers ──")
for fam in all_families:
    fam_genes = atlas[atlas['tf_families'].str.contains(fam, na=False)]
    if len(fam_genes) < 5:
        continue
    layer_dist = fam_genes['signaling_layer'].value_counts()
    n_layers = len(layer_dist)
    if n_layers >= 3:
        layer_str = ', '.join(f"{l}({c})" for l, c in layer_dist.head(5).items())
        print(f"  {fam} ({len(fam_genes)} genes, {n_layers} layers): {layer_str}")

# Drought-responsive signaling chains
print("\n── Drought-responsive signaling genes (|log2FC| > 1) ──")
drought_genes = atlas[atlas['log2fc_drought'].abs() > 1].sort_values('log2fc_drought')
print(f"  {len(drought_genes)} genes with strong drought response")
for layer in ['Receptor/Sensor', 'Signal Transduction', 'Transcription Factor', 'Hormone']:
    sub = drought_genes[drought_genes['signaling_layer'] == layer]
    if len(sub) > 0:
        print(f"\n  {layer} ({len(sub)} drought-responsive):")
        for _, r in sub.iterrows():
            direction = 'UP' if r['log2fc_drought'] > 0 else 'DOWN'
            print(f"    {r['gene_id']}: log2FC={r['log2fc_drought']:+.2f} ({direction}), "
                  f"|d|={r['abs_d']:.2f}, motif={r['tf_families']}")
            print(f"      {r['description'][:100]}")


# ═══════════════════════════════════════════════════════════════
# STEP 5: Specific Pathway Modules
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 5: Known Pathway Modules in 728 Genes")
print("="*70)

# Look for known plant signaling modules
KNOWN_MODULES = {
    'BR signaling': ['BES1', 'BZR', 'BRI1', 'BAK1', 'BSK', 'BIN2', 'brassinosteroid'],
    'Auxin signaling': ['IAA', 'ARF', 'TIR1', 'AFB', 'auxin', 'PIN', 'AUX1'],
    'ABA signaling': ['PYR', 'PYL', 'RCAR', 'PP2C', 'SnRK2', 'ABI', 'abscisic'],
    'MAPK cascade': ['MAPK', 'MAP kinase', 'MPK', 'MKK', 'MEKK'],
    'Calcium signaling': ['calmodulin', 'CaM', 'calcium', 'CDPK', 'CPK', 'CBL', 'CIPK'],
    'Ubiquitin-proteasome': ['ubiquitin', 'E3 ligase', 'proteasome', 'SCF', 'CUL', 'F-box'],
    'HSP/chaperone network': ['HSP', 'heat shock', 'chaperone', 'DnaJ', 'ClpB', 'Hsp70', 'Hsp90'],
    'Photosynthesis/Light': ['photosystem', 'chlorophyll', 'light-harvesting', 'phototropin', 'phytochrome'],
    'Cell wall': ['cellulose', 'pectin', 'xylan', 'expansin', 'cell wall'],
    'Redox homeostasis': ['thioredoxin', 'glutaredoxin', 'peroxidase', 'superoxide dismutase', 'catalase'],
}

for module_name, keywords in KNOWN_MODULES.items():
    module_genes = []
    for _, row in atlas.iterrows():
        text = f"{row['description']} {row['cog_description']} {row['go_biological']}".lower()
        if any(kw.lower() in text for kw in keywords):
            module_genes.append(row)

    if module_genes:
        mdf = pd.DataFrame(module_genes)
        # Check if they share TF motifs
        shared_fams = Counter()
        for fams in mdf['tf_families']:
            for f in str(fams).split(','):
                if f.strip():
                    shared_fams[f.strip()] += 1

        dominant = shared_fams.most_common(3)
        dom_str = ', '.join(f"{f}({c})" for f, c in dominant)

        print(f"\n  {module_name}: {len(mdf)} genes")
        print(f"    Shared disrupted motifs: {dom_str}")
        for _, r in mdf.iterrows():
            drought_str = f"drought={r['log2fc_drought']:+.1f}" if pd.notna(r['log2fc_drought']) else ""
            print(f"    {r['gene_id']}: |d|={r['abs_d']:.2f}, motif=[{r['tf_families']}], {drought_str}")
            print(f"      {r['description'][:100]}")


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Gene × TF Family Clustered Heatmap
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Creating figures...")
print("="*70)

# Filter to genes with at least 1 family
mask_has = merged['n_families'] > 0
mat_filt = matrix[mask_has]
genes_filt = merged.loc[mask_has, 'gene_id'].values
funcs_filt = merged.loc[mask_has, 'func_category'].values
effects_filt = merged.loc[mask_has, 'abs_d'].values

# Filter to families with >= 3 genes
fam_mask = mat_filt.sum(axis=0) >= 3
mat_filt2 = mat_filt[:, fam_mask]
fams_filt = [all_families[i] for i in range(len(all_families)) if fam_mask[i]]

print(f"Heatmap: {mat_filt2.shape[0]} genes × {mat_filt2.shape[1]} families")

# Cluster genes
if mat_filt2.shape[0] > 2:
    gene_dist = pdist(mat_filt2, metric='jaccard')
    # Replace NaN distances (genes with no motifs) with 1.0
    gene_dist = np.nan_to_num(gene_dist, nan=1.0)
    gene_link = linkage(gene_dist, method='ward')

    # Cluster families
    fam_dist = pdist(mat_filt2.T, metric='jaccard')
    fam_dist = np.nan_to_num(fam_dist, nan=1.0)
    fam_link = linkage(fam_dist, method='ward')

fig = plt.figure(figsize=(18, 22))
gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[1, 5],
             hspace=0.05, wspace=0.05)

# Dendrogram for genes (left side)
ax_dendro = fig.add_subplot(gs[0, 0])
dendro = dendrogram(gene_link, orientation='left', no_labels=True, ax=ax_dendro,
                    color_threshold=0, above_threshold_color='gray')
ax_dendro.set_xticks([])
ax_dendro.invert_yaxis()
ax_dendro.spines['top'].set_visible(False)
ax_dendro.spines['right'].set_visible(False)
ax_dendro.spines['bottom'].set_visible(False)

# Reorder matrix by dendrogram
gene_order = dendro['leaves']

# Heatmap
ax_heat = fig.add_subplot(gs[0, 1])
mat_ordered = mat_filt2[gene_order, :]

# Custom colormap: white for 0, blue for 1
cmap = ListedColormap(['#FFFFFF', '#1565C0'])
im = ax_heat.imshow(mat_ordered, aspect='auto', cmap=cmap, interpolation='nearest')
ax_heat.set_xticks(range(len(fams_filt)))
ax_heat.set_xticklabels(fams_filt, rotation=90, fontsize=8, ha='center')
ax_heat.xaxis.set_ticks_position('top')
ax_heat.xaxis.set_label_position('top')
ax_heat.set_yticks([])
ax_heat.set_ylabel(f'{mat_filt2.shape[0]} genes (clustered)', fontsize=10)

# Summary bar chart below (family counts)
ax_bar = fig.add_subplot(gs[1, 1])
fam_sums = mat_filt2.sum(axis=0)
ax_bar.bar(range(len(fams_filt)), fam_sums, color='#1565C0', alpha=0.7)
ax_bar.set_xticks(range(len(fams_filt)))
ax_bar.set_xticklabels(fams_filt, rotation=90, fontsize=8)
ax_bar.set_ylabel('Genes')
ax_bar.set_xlim(-0.5, len(fams_filt)-0.5)

# Functional category color strip (left of dendrogram)
# Actually put this as annotation
func_colors = {
    'Kinase/Phosphorylation': '#E53935',
    'Chaperone/HSP': '#FF9800',
    'Transcription factor': '#4CAF50',
    'Biosynthesis': '#2196F3',
    'Translation': '#9C27B0',
    'Signal transduction': '#F44336',
    'Transporter': '#00BCD4',
    'Redox/Metabolism': '#795548',
    'Chromatin/Epigenetic': '#607D8B',
    'Hormone signaling': '#FFEB3B',
    'Cell wall': '#8BC34A',
    'Other': '#BDBDBD',
    'Unknown': '#E0E0E0'
}

plt.suptitle('Gene × TF Family Heatmap — 728 Causal bQTL Targets\n'
             '(blue = TF family motif disrupted at bQTL position)',
             fontsize=13, fontweight='bold', y=0.995)

plt.savefig(f'{FIGS}/fig_gene_motif_heatmap.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{FIGS}/fig_gene_motif_heatmap.png', bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved: fig_gene_motif_heatmap.pdf")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Signaling Layer Network
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Panel A: Signaling layer distribution with effect sizes
ax = axes[0, 0]
layer_order = ['Receptor/Sensor', 'Membrane/Transport', 'Hormone',
               'Signal Transduction', 'Second Messenger',
               'Protein Processing', 'Transcription Factor',
               'Chromatin/Epigenetic', 'Translation', 'Metabolism',
               'Redox/Stress', 'Unknown']
layer_order = [l for l in layer_order if l in layer_counts.index]
counts = [layer_counts.get(l, 0) for l in layer_order]
colors_layer = plt.cm.Set3(np.linspace(0, 1, len(layer_order)))
bars = ax.barh(range(len(layer_order)), counts, color=colors_layer, alpha=0.8)
ax.set_yticks(range(len(layer_order)))
ax.set_yticklabels(layer_order, fontsize=9)
ax.set_xlabel('Number of genes')
ax.set_title('A. Signaling layer distribution', fontsize=11)
ax.invert_yaxis()
# Annotate with mean effect
for i, layer in enumerate(layer_order):
    mean_d = atlas.loc[atlas['signaling_layer'] == layer, 'abs_d'].mean()
    ax.text(counts[i] + 2, i, f'|d|={mean_d:.1f}', va='center', fontsize=8)

# Panel B: TF family → signaling layer heatmap (co-regulation across layers)
ax = axes[0, 1]
top_fams = [f for f in all_families if matrix[:, all_families.index(f)].sum() >= 10]
layer_fam_matrix = np.zeros((len(layer_order), len(top_fams)))
for i, layer in enumerate(layer_order):
    layer_genes = atlas[atlas['signaling_layer'] == layer].index
    for j, fam in enumerate(top_fams):
        fam_idx = all_families.index(fam)
        layer_fam_matrix[i, j] = matrix[layer_genes, fam_idx].sum()

im = ax.imshow(layer_fam_matrix, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(top_fams)))
ax.set_xticklabels(top_fams, rotation=90, fontsize=8)
ax.set_yticks(range(len(layer_order)))
ax.set_yticklabels(layer_order, fontsize=8)
ax.set_title('B. TF motif × signaling layer', fontsize=11)
plt.colorbar(im, ax=ax, label='Gene count', shrink=0.7)

# Panel C: Effect size by signaling layer
ax = axes[1, 0]
data_layers = []
labels_layers = []
for layer in layer_order:
    vals = atlas.loc[atlas['signaling_layer'] == layer, 'abs_d'].dropna()
    if len(vals) >= 5:
        data_layers.append(vals.values)
        labels_layers.append(layer)
bp = ax.boxplot(data_layers, tick_labels=labels_layers, patch_artist=True, vert=True)
for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(labels_layers)))):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels(labels_layers, rotation=45, ha='right', fontsize=8)
ax.set_ylabel("|Cohen's d|")
ax.set_title('C. Effect size by signaling layer', fontsize=11)

# Panel D: Drought response by signaling layer
ax = axes[1, 1]
drought_data = []
for layer in layer_order:
    vals = atlas.loc[atlas['signaling_layer'] == layer, 'log2fc_drought'].dropna()
    if len(vals) >= 5:
        up = (vals > 1).sum()
        down = (vals < -1).sum()
        stable = len(vals) - up - down
        drought_data.append({'layer': layer, 'up': up, 'down': down, 'stable': stable})

if drought_data:
    ddf = pd.DataFrame(drought_data)
    x = range(len(ddf))
    ax.bar(x, ddf['up'], color='#E53935', alpha=0.7, label='Drought UP')
    ax.bar(x, -ddf['down'], color='#2196F3', alpha=0.7, label='Drought DOWN')
    ax.set_xticks(x)
    ax.set_xticklabels(ddf['layer'], rotation=45, ha='right', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Gene count (up / -down)')
    ax.set_title('D. Drought response by layer', fontsize=11)
    ax.legend(fontsize=8)

plt.suptitle('Signaling Architecture of 728 Causal bQTL Target Genes',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{FIGS}/fig_signaling_layers.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{FIGS}/fig_signaling_layers.png', bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved: fig_signaling_layers.pdf")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Co-regulation Network (TF family → gene groups)
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(18, 10))

# Panel A: Genes per sole TF family (exclusive regulation)
ax = axes[0]
sole_counts = sole_family_counts[sole_family_counts >= 3]
bars = ax.barh(range(len(sole_counts)), sole_counts.values, color='#1565C0', alpha=0.7)
ax.set_yticks(range(len(sole_counts)))
ax.set_yticklabels(sole_counts.index, fontsize=9)
ax.set_xlabel('Genes exclusively regulated by this TF family')
ax.set_title('A. Sole-motif gene groups\n(only 1 TF family disrupted at bQTL)', fontsize=11)
ax.invert_yaxis()

# Panel B: Jaccard similarity between TF family target sets
ax = axes[1]
top_fams_all = [f for f in all_families if fam_totals[all_families.index(f)] >= 10]
n_fams = len(top_fams_all)
jaccard_mat = np.zeros((n_fams, n_fams))
for i in range(n_fams):
    fi = all_families.index(top_fams_all[i])
    for j in range(n_fams):
        fj = all_families.index(top_fams_all[j])
        genes_i = set(np.where(matrix[:, fi] == 1)[0])
        genes_j = set(np.where(matrix[:, fj] == 1)[0])
        if genes_i | genes_j:
            jaccard_mat[i, j] = len(genes_i & genes_j) / len(genes_i | genes_j)
        else:
            jaccard_mat[i, j] = 0

im = ax.imshow(jaccard_mat, cmap='YlOrRd', vmin=0, vmax=0.6)
ax.set_xticks(range(n_fams))
ax.set_xticklabels(top_fams_all, rotation=90, fontsize=8)
ax.set_yticks(range(n_fams))
ax.set_yticklabels(top_fams_all, fontsize=8)
ax.set_title('B. Jaccard overlap between TF family targets\n(do same genes have multiple motifs disrupted?)', fontsize=11)
plt.colorbar(im, ax=ax, label='Jaccard index', shrink=0.7)

plt.suptitle('Co-regulation Patterns — 728 Causal bQTL',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{FIGS}/fig_coregulation_network.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{FIGS}/fig_coregulation_network.png', bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved: fig_coregulation_network.pdf")


# ═══════════════════════════════════════════════════════════════
# Save Complete Gene Atlas
# ═══════════════════════════════════════════════════════════════
atlas.to_csv(f'{RESULTS}/gene_atlas_728.csv', index=False)
print(f"\nSaved: gene_atlas_728.csv ({len(atlas)} genes)")
print("Done.")
