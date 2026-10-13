#!/usr/bin/env python3
"""
24_embedding_tf_clustering.py — Do causal bQTL cluster by disrupted TF family
in Agro-NT embedding space?

Analysis:
  1. Load frozen + LoRA embeddings (1075 × 1500)
  2. Load disrupted TF family labels per bQTL
  3. UMAP project both embedding sets
  4. Quantify TF family clustering: silhouette score, kNN purity
  5. Compare frozen vs LoRA — does LoRA capture TF-specific grammar?
  6. Overlay condition labels (WW-only, DS-only, shared)
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')
import functools
print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_config import config

RESULTS = config.results_dir
FIGURES = config.figures_dir

# ── Load data ────────────────────────────────────────────────────────

print("=" * 70)
print("  EMBEDDING × TF FAMILY CLUSTERING ANALYSIS")
print("=" * 70)

# Embeddings
emb_frozen = np.load(RESULTS / 'agro_nt_embeddings_frozen.npy')
emb_lora = np.load(RESULTS / 'agro_nt_embeddings_lora.npy')
pos_df = pd.read_csv(RESULTS / 'agro_nt_embedding_positions.csv')
print(f"\n  Embeddings: {emb_frozen.shape[0]} bQTL × {emb_frozen.shape[1]} dims")

# TF family labels
asm = pd.read_csv(RESULTS / 'allele_specific_motifs.csv')
# Deduplicate asm on chr+pos (keep first) before merging
asm_dedup = asm.drop_duplicates(subset=['chr', 'pos'], keep='first')
# Merge on chr + pos
pos_df = pos_df.merge(asm_dedup[['chr', 'pos', 'disrupted_families', 'created_tfs',
                            'cohens_d', 'abs_d']],
                       on=['chr', 'pos'], how='left')
assert len(pos_df) == len(emb_frozen), f"Merge changed row count: {len(pos_df)} vs {len(emb_frozen)}"

# Also load condition labels (WW vs DS)
causal_ww = pd.read_csv(RESULTS / 'causal_bqtl_728.csv')
causal_ww_set = set(zip(causal_ww.chr, causal_ww.pos))

ds_path = RESULTS / 'causal_bqtl_ds.csv'
if ds_path.exists():
    causal_ds = pd.read_csv(ds_path)
    causal_ds_set = set(zip(causal_ds.chr, causal_ds.pos))
else:
    causal_ds_set = set()

pos_df['in_ww'] = pos_df.apply(lambda r: (r.chr, r.pos) in causal_ww_set, axis=1)
pos_df['in_ds'] = pos_df.apply(lambda r: (r.chr, r.pos) in causal_ds_set, axis=1)
pos_df['condition'] = 'WW-only'
pos_df.loc[pos_df['in_ww'] & pos_df['in_ds'], 'condition'] = 'Shared'
pos_df.loc[~pos_df['in_ww'] & pos_df['in_ds'], 'condition'] = 'DS-only'
pos_df.loc[~pos_df['in_ww'] & ~pos_df['in_ds'], 'condition'] = 'WW-only'  # default

print(f"  Condition: {pos_df['condition'].value_counts().to_dict()}")

# ── Assign primary TF family ────────────────────────────────────────

def get_primary_family(fam_str):
    """Return the first (primary) disrupted family, or 'none'."""
    if pd.isna(fam_str) or not str(fam_str).strip():
        return 'none'
    families = [f.strip() for f in str(fam_str).split(',') if f.strip()]
    return families[0] if families else 'none'

def get_all_families(fam_str):
    """Return all disrupted families as a set."""
    if pd.isna(fam_str) or not str(fam_str).strip():
        return set()
    return set(f.strip() for f in str(fam_str).split(',') if f.strip())

pos_df['primary_family'] = pos_df['disrupted_families'].apply(get_primary_family)

# Count families
fam_counts = pos_df['primary_family'].value_counts()
print(f"\n  Primary TF family distribution:")
for fam, cnt in fam_counts.head(15).items():
    print(f"    {fam:20s} {cnt:4d}")
print(f"  Total with family: {(pos_df['primary_family'] != 'none').sum()}")

# For clustering analysis, use families with >= 20 bQTL
MIN_COUNT = 20
major_families = set(fam_counts[fam_counts >= MIN_COUNT].index) - {'none'}
print(f"\n  Major families (n >= {MIN_COUNT}): {len(major_families)}")
for f in sorted(major_families):
    print(f"    {f}: {fam_counts[f]}")

# ── UMAP projection ─────────────────────────────────────────────────

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# PCA first to 50 dims for UMAP stability
pca = PCA(n_components=50, random_state=42)
frozen_pca = pca.fit_transform(emb_frozen)
var_frozen = pca.explained_variance_ratio_.sum()
print(f"\n  Frozen PCA 50d: {var_frozen:.1%} variance explained")

pca_lora = PCA(n_components=50, random_state=42)
lora_pca = pca_lora.fit_transform(emb_lora)
var_lora = pca_lora.explained_variance_ratio_.sum()
print(f"  LoRA PCA 50d: {var_lora:.1%} variance explained")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("\n  WARNING: umap-learn not installed, using t-SNE fallback")

if HAS_UMAP:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42, n_components=2)
    frozen_2d = reducer.fit_transform(frozen_pca)
    lora_2d = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42,
                         n_components=2).fit_transform(lora_pca)
    method_name = 'UMAP'
else:
    from sklearn.manifold import TSNE
    frozen_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(frozen_pca)
    lora_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(lora_pca)
    method_name = 't-SNE'

print(f"  {method_name} projection complete")

# ── Quantify clustering ─────────────────────────────────────────────

from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Filter to major families only
major_mask = pos_df['primary_family'].isin(major_families)
n_major = major_mask.sum()
labels_major = pos_df.loc[major_mask, 'primary_family'].values
print(f"\n  Clustering analysis on {n_major} bQTL with major families")

# Multi-label assignment: use ALL disrupted families, not just primary
# For each bQTL, create a binary vector over major families
all_fam_labels = []
for fam_str in pos_df.loc[major_mask, 'disrupted_families']:
    fams = get_all_families(fam_str)
    all_fam_labels.append(fams & major_families)

# ── 1. Silhouette score (primary family label) ──────────────────────

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels_enc = le.fit_transform(labels_major)

sil_frozen = silhouette_score(frozen_pca[major_mask], labels_enc, metric='cosine')
sil_lora = silhouette_score(lora_pca[major_mask], labels_enc, metric='cosine')
print(f"\n  Silhouette score (cosine, primary family):")
print(f"    Frozen: {sil_frozen:.4f}")
print(f"    LoRA:   {sil_lora:.4f}")

# Also in 2D projection
sil_frozen_2d = silhouette_score(frozen_2d[major_mask], labels_enc, metric='euclidean')
sil_lora_2d = silhouette_score(lora_2d[major_mask], labels_enc, metric='euclidean')
print(f"  Silhouette score ({method_name} 2D):")
print(f"    Frozen: {sil_frozen_2d:.4f}")
print(f"    LoRA:   {sil_lora_2d:.4f}")

# ── 2. kNN purity (can we predict TF family from embedding?) ───────

print(f"\n  kNN TF family prediction (5-fold CV):")
for k in [5, 15, 30]:
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')

    scores_frozen = cross_val_score(knn, frozen_pca[major_mask], labels_enc,
                                     cv=5, scoring='accuracy')
    scores_lora = cross_val_score(knn, lora_pca[major_mask], labels_enc,
                                   cv=5, scoring='accuracy')

    # Random baseline: most frequent class
    baseline = max(Counter(labels_enc).values()) / len(labels_enc)

    print(f"  k={k:2d}: Frozen={scores_frozen.mean():.3f}±{scores_frozen.std():.3f}, "
          f"LoRA={scores_lora.mean():.3f}±{scores_lora.std():.3f}, "
          f"baseline={baseline:.3f}")

# ── 3. Per-family kNN purity (is family X more clustered?) ──────────

print(f"\n  Per-family neighborhood purity (k=15, cosine):")
from sklearn.neighbors import NearestNeighbors
results_per_family = {}

for emb_name, emb_data in [('Frozen', frozen_pca), ('LoRA', lora_pca)]:
    nn = NearestNeighbors(n_neighbors=16, metric='cosine')
    nn.fit(emb_data)  # fit on ALL 1075

    fam_purity = {}
    for fam in sorted(major_families):
        fam_mask = pos_df['primary_family'] == fam
        fam_idx = np.where(fam_mask)[0]
        if len(fam_idx) < 5:
            continue

        # For each bQTL of this family, how many of its 15 nearest neighbors
        # also have this family as primary?
        distances, neighbors = nn.kneighbors(emb_data[fam_idx])
        purities = []
        for i, neigh_idx in enumerate(neighbors):
            # Exclude self (first neighbor)
            neigh_fams = pos_df.iloc[neigh_idx[1:]]['primary_family'].values
            purity = (neigh_fams == fam).mean()
            purities.append(purity)

        mean_purity = np.mean(purities)
        # Expected purity by chance = n_family / n_total
        expected = len(fam_idx) / len(pos_df)
        enrichment = mean_purity / expected if expected > 0 else 0

        fam_purity[fam] = {
            'purity': mean_purity,
            'expected': expected,
            'enrichment': enrichment,
            'n': len(fam_idx)
        }

    results_per_family[emb_name] = fam_purity

# Print comparison
print(f"\n  {'Family':20s} {'N':>4s}  {'Frozen_pur':>10s} {'LoRA_pur':>10s}  "
      f"{'Frozen_enr':>10s} {'LoRA_enr':>10s}  {'LoRA_gain':>10s}")
print("  " + "-" * 85)
for fam in sorted(major_families):
    if fam in results_per_family['Frozen'] and fam in results_per_family['LoRA']:
        fr = results_per_family['Frozen'][fam]
        lr = results_per_family['LoRA'][fam]
        gain = lr['enrichment'] - fr['enrichment']
        print(f"  {fam:20s} {fr['n']:4d}  {fr['purity']:10.4f} {lr['purity']:10.4f}  "
              f"{fr['enrichment']:10.2f}× {lr['enrichment']:10.2f}×  "
              f"{'+' if gain > 0 else ''}{gain:9.2f}×")

# ── 4. Embedding distance: within-family vs between-family ──────────

from scipy.spatial.distance import cdist

print(f"\n  Within-family vs between-family cosine distance:")
for emb_name, emb_data in [('Frozen', frozen_pca), ('LoRA', lora_pca)]:
    within_dists = []
    between_dists = []

    for fam in sorted(major_families):
        fam_mask = np.array(pos_df['primary_family'] == fam)
        other_mask = np.array(pos_df['primary_family'].isin(major_families) &
                              (pos_df['primary_family'] != fam))

        if fam_mask.sum() < 5:
            continue

        # Within-family distances
        fam_emb = emb_data[fam_mask]
        if len(fam_emb) > 1:
            w_dists = cdist(fam_emb, fam_emb, metric='cosine')
            np.fill_diagonal(w_dists, np.nan)
            within_dists.extend(w_dists[~np.isnan(w_dists)].tolist())

        # Between-family distances (sample to keep tractable)
        other_emb = emb_data[other_mask]
        if len(other_emb) > 100:
            rng = np.random.RandomState(42)
            other_emb = other_emb[rng.choice(len(other_emb), 100, replace=False)]
        b_dists = cdist(fam_emb, other_emb, metric='cosine')
        between_dists.extend(b_dists.flatten().tolist())

    within_mean = np.mean(within_dists)
    between_mean = np.mean(between_dists)
    ratio = between_mean / within_mean if within_mean > 0 else 0

    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(within_dists, between_dists, alternative='less')

    print(f"  {emb_name}: within={within_mean:.4f}, between={between_mean:.4f}, "
          f"ratio={ratio:.3f}, p={p:.2e}")

# ── 5. Effect size in embedding space ────────────────────────────────

print(f"\n  Does embedding distance correlate with Cohen's d?")
from scipy.stats import spearmanr

for emb_name, emb_data in [('Frozen', frozen_pca), ('LoRA', lora_pca)]:
    # Distance from centroid of all embeddings
    centroid = emb_data.mean(axis=0)
    dists_to_center = np.sqrt(((emb_data - centroid) ** 2).sum(axis=1))

    abs_d = pos_df['abs_d'].values
    valid = ~np.isnan(abs_d)
    rho, p = spearmanr(dists_to_center[valid], abs_d[valid])
    print(f"  {emb_name}: dist-to-centroid vs |Cohen's d|: ρ={rho:.4f}, p={p:.3e}")

# ── 6. Visualization ────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

fig, axes = plt.subplots(2, 3, figsize=(20, 13))

# Color map for top families
top_families = list(fam_counts.head(10).index)
top_families = [f for f in top_families if f != 'none']
cmap = plt.cm.tab10
fam_colors = {fam: cmap(i) for i, fam in enumerate(top_families)}
fam_colors['none'] = (0.85, 0.85, 0.85, 0.3)
fam_colors['other'] = (0.6, 0.6, 0.6, 0.4)

def get_color(fam):
    if fam in fam_colors:
        return fam_colors[fam]
    return fam_colors['other']

# Panel A: Frozen, colored by primary TF family
ax = axes[0, 0]
# Plot 'none' and 'other' first (background)
for fam in ['none', 'other']:
    mask = pos_df['primary_family'].apply(
        lambda f: f == fam if fam == 'none' else (f not in top_families and f != 'none'))
    if mask.sum() > 0:
        ax.scatter(frozen_2d[mask, 0], frozen_2d[mask, 1], c=[get_color(fam)],
                   s=8, alpha=0.3, rasterized=True)
# Then plot each major family
for fam in top_families:
    mask = pos_df['primary_family'] == fam
    if mask.sum() > 0:
        ax.scatter(frozen_2d[mask, 0], frozen_2d[mask, 1], c=[fam_colors[fam]],
                   s=20, alpha=0.7, label=f'{fam} ({mask.sum()})', rasterized=True)
ax.set_title(f'Frozen Agro-NT — TF family', fontsize=12, fontweight='bold')
ax.set_xlabel(f'{method_name} 1')
ax.set_ylabel(f'{method_name} 2')
ax.legend(fontsize=7, loc='best', framealpha=0.8, ncol=2)

# Panel B: LoRA, colored by primary TF family
ax = axes[0, 1]
for fam in ['none', 'other']:
    mask = pos_df['primary_family'].apply(
        lambda f: f == fam if fam == 'none' else (f not in top_families and f != 'none'))
    if mask.sum() > 0:
        ax.scatter(lora_2d[mask, 0], lora_2d[mask, 1], c=[get_color(fam)],
                   s=8, alpha=0.3, rasterized=True)
for fam in top_families:
    mask = pos_df['primary_family'] == fam
    if mask.sum() > 0:
        ax.scatter(lora_2d[mask, 0], lora_2d[mask, 1], c=[fam_colors[fam]],
                   s=20, alpha=0.7, label=f'{fam} ({mask.sum()})', rasterized=True)
ax.set_title(f'LoRA Agro-NT — TF family', fontsize=12, fontweight='bold')
ax.set_xlabel(f'{method_name} 1')
ax.set_ylabel(f'{method_name} 2')
ax.legend(fontsize=7, loc='best', framealpha=0.8, ncol=2)

# Panel C: LoRA colored by condition (WW/DS/Shared)
ax = axes[0, 2]
cond_colors = {'WW-only': '#2196F3', 'DS-only': '#FF5722', 'Shared': '#4CAF50'}
for cond in ['WW-only', 'DS-only', 'Shared']:
    mask = pos_df['condition'] == cond
    if mask.sum() > 0:
        ax.scatter(lora_2d[mask, 0], lora_2d[mask, 1], c=cond_colors[cond],
                   s=20, alpha=0.6, label=f'{cond} ({mask.sum()})', rasterized=True)
ax.set_title(f'LoRA Agro-NT — Condition', fontsize=12, fontweight='bold')
ax.set_xlabel(f'{method_name} 1')
ax.set_ylabel(f'{method_name} 2')
ax.legend(fontsize=9, loc='best', framealpha=0.8)

# Panel D: Per-family neighborhood enrichment comparison
ax = axes[1, 0]
families_sorted = sorted(major_families,
                          key=lambda f: results_per_family['LoRA'].get(f, {}).get('enrichment', 0),
                          reverse=True)
x = np.arange(len(families_sorted))
w = 0.35
frozen_enr = [results_per_family['Frozen'].get(f, {}).get('enrichment', 0) for f in families_sorted]
lora_enr = [results_per_family['LoRA'].get(f, {}).get('enrichment', 0) for f in families_sorted]
bars1 = ax.bar(x - w/2, frozen_enr, w, label='Frozen', color='#42A5F5', alpha=0.8)
bars2 = ax.bar(x + w/2, lora_enr, w, label='LoRA', color='#EF5350', alpha=0.8)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, label='Random expectation')
ax.set_xticks(x)
ax.set_xticklabels(families_sorted, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Neighborhood enrichment')
ax.set_title('Per-family kNN purity enrichment', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

# Panel E: Cohen's d by TF family
ax = axes[1, 1]
fam_d = []
for fam in families_sorted:
    mask = pos_df['primary_family'] == fam
    d_vals = pos_df.loc[mask, 'abs_d'].dropna()
    fam_d.append(d_vals.values)
bp = ax.boxplot(fam_d, tick_labels=families_sorted, patch_artist=True, showfliers=False)
for patch in bp['boxes']:
    patch.set_facecolor('#B0BEC5')
    patch.set_alpha(0.7)
ax.set_xticklabels(families_sorted, rotation=45, ha='right', fontsize=8)
ax.set_ylabel("|Cohen's d|")
ax.set_title('Effect size by disrupted TF family', fontsize=12, fontweight='bold')

# Panel F: LoRA colored by effect size
ax = axes[1, 2]
abs_d = pos_df['abs_d'].values
valid = ~np.isnan(abs_d)
sc = ax.scatter(lora_2d[valid, 0], lora_2d[valid, 1], c=abs_d[valid],
                cmap='YlOrRd', s=15, alpha=0.7, vmin=0, vmax=np.percentile(abs_d[valid], 95),
                rasterized=True)
plt.colorbar(sc, ax=ax, label="|Cohen's d|", shrink=0.8)
ax.set_title(f'LoRA Agro-NT — Effect size', fontsize=12, fontweight='bold')
ax.set_xlabel(f'{method_name} 1')
ax.set_ylabel(f'{method_name} 2')

plt.tight_layout()
fig_path = FIGURES / 'embedding_tf_clustering.pdf'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.savefig(fig_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
print(f"\n  Figure saved: {fig_path}")

# ── Save results ─────────────────────────────────────────────────────

results = {
    'n_bqtl': int(len(pos_df)),
    'n_with_family': int((pos_df['primary_family'] != 'none').sum()),
    'n_major_families': int(len(major_families)),
    'silhouette_frozen_50d': float(sil_frozen),
    'silhouette_lora_50d': float(sil_lora),
    'silhouette_frozen_2d': float(sil_frozen_2d),
    'silhouette_lora_2d': float(sil_lora_2d),
    'pca_variance_frozen': float(var_frozen),
    'pca_variance_lora': float(var_lora),
    'per_family_enrichment': {
        emb_name: {fam: vals for fam, vals in fam_data.items()}
        for emb_name, fam_data in results_per_family.items()
    }
}

out_path = RESULTS / 'embedding_tf_clustering.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Results saved: {out_path}")

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
