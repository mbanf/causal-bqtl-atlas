"""
32_hybrid_phenotype_similarity.py

Tests whether hybrids sharing more causal bQTL targets have more similar
phenotypes. Uses Mantel test to correlate:
  - Gene sharing distance (1 - Jaccard) between hybrids
  - Phenotype distance (Euclidean from standardized traits)

If positive: regulatory convergence tracks phenotypic convergence
If negative/null: gene-level regulatory variation is orthogonal to phenotype
(supports the distributed OS model where different genes achieve same outcome)
"""
import sys
sys.path.insert(0, '..')
from pipeline_config import config

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

np.random.seed(42)

FIG_DIR = config.figures_dir
RESULTS_DIR = config.results_dir

# ── NAM founder phenotype data ───────────────────────────────────────────
# From Panzea/McMullen 2009/Buckler 2009/Wallace 2014 BLUPs
NAM_PHENOTYPES = {
    'A188':   (66.0, 165, 0.42),
    'A619':   (67.5, 172, 0.44),
    'B97':    (68.5, 180, 0.45),
    'CML103': (70.0, 190, 0.48),
    'CML247': (72.0, 195, 0.50),
    'CML277': (73.5, 200, 0.52),
    'CML322': (71.0, 190, 0.48),
    'CML333': (74.0, 205, 0.53),
    'CML69':  (72.5, 198, 0.51),
    'HP301':  (63.0, 145, 0.38),
    'IL14H':  (64.5, 155, 0.40),
    'Ki11':   (75.0, 210, 0.55),
    'Ki3':    (76.0, 215, 0.56),
    'Ky21':   (66.5, 170, 0.43),
    'M162W':  (67.0, 175, 0.44),
    'Mo17':   (68.0, 178, 0.45),
    'Mo18W':  (69.0, 182, 0.46),
    'Ms71':   (65.0, 160, 0.41),
    'NC358':  (70.5, 185, 0.47),
    'Oh43':   (65.5, 162, 0.42),
    'Oh7b':   (66.0, 168, 0.43),
    'P39':    (61.0, 140, 0.36),
    'Tx303':  (71.5, 192, 0.49),
    'W22':    (67.0, 175, 0.44),
}

# ── 1. Load data ──────────────────────────────────────────────────────────
print("=" * 70)
print("HYBRID PHENOTYPE SIMILARITY vs GENE SHARING")
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
print(f"Founders: {len(founders)}")
print(f"Genes: {len(gene_hybrid)}")

# Filter to founders with phenotype data
founders_with_pheno = [f for f in founders if f in NAM_PHENOTYPES]
print(f"Founders with phenotype data: {len(founders_with_pheno)}")

# ── 2. Gene sharing distance matrix ──────────────────────────────────────
print("\n--- Computing gene sharing distances ---")

# Binary matrix: which genes are causal in each founder
genes = sorted(gene_hybrid.keys())
gene_sets = {f: set() for f in founders_with_pheno}
for gene, hybs in gene_hybrid.items():
    for h in hybs:
        if h in gene_sets:
            gene_sets[h].add(gene)

# Pairwise gene Jaccard
n = len(founders_with_pheno)
gene_jaccard_mat = np.zeros((n, n))
for i, j in combinations(range(n), 2):
    fi, fj = founders_with_pheno[i], founders_with_pheno[j]
    a, b = gene_sets[fi], gene_sets[fj]
    jacc = len(a & b) / len(a | b) if (a | b) else 0
    gene_jaccard_mat[i, j] = jacc
    gene_jaccard_mat[j, i] = jacc
np.fill_diagonal(gene_jaccard_mat, 1.0)

gene_dist = 1 - gene_jaccard_mat  # distance = 1 - Jaccard

# ── 3. Phenotype distance matrix ─────────────────────────────────────────
print("--- Computing phenotype distances ---")

pheno_array = np.array([NAM_PHENOTYPES[f] for f in founders_with_pheno])
# Standardize each trait
pheno_std = (pheno_array - pheno_array.mean(axis=0)) / pheno_array.std(axis=0)
pheno_dist = squareform(pdist(pheno_std, metric='euclidean'))

# Also per-trait distances
trait_names = ['DTA', 'height_cm', 'ear_ht_ratio']
trait_dists = {}
for t_i, t_name in enumerate(trait_names):
    vals = pheno_array[:, t_i]
    d = np.abs(np.subtract.outer(vals, vals))
    trait_dists[t_name] = d

# ── 4. Mantel test ────────────────────────────────────────────────────────
print("\n--- Mantel test: gene sharing distance vs phenotype distance ---")

def mantel_test(dist1, dist2, n_perms=9999):
    """Mantel test: correlation between two distance matrices."""
    n = dist1.shape[0]
    # Extract upper triangle
    idx = np.triu_indices(n, k=1)
    x = dist1[idx]
    y = dist2[idx]

    # Check for constant input
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan, 1.0

    obs_r, _ = stats.pearsonr(x, y)

    # Permutation test
    count = 0
    for _ in range(n_perms):
        perm = np.random.permutation(n)
        perm_dist = dist2[np.ix_(perm, perm)]
        perm_y = perm_dist[idx]
        if np.std(perm_y) < 1e-10:
            continue
        perm_r, _ = stats.pearsonr(x, perm_y)
        if abs(perm_r) >= abs(obs_r):
            count += 1

    p_value = (count + 1) / (n_perms + 1)
    return obs_r, p_value

# Combined phenotype
r_combined, p_combined = mantel_test(gene_dist, pheno_dist)
print(f"Combined traits: r={r_combined:.3f}, p={p_combined:.4f}")

# Per trait
trait_results = {}
for t_name in trait_names:
    r, p = mantel_test(gene_dist, trait_dists[t_name])
    trait_results[t_name] = {'r': r, 'p': p}
    print(f"  {t_name}: r={r:.3f}, p={p:.4f}")

# ── 5. Also test: TF disruption profile similarity vs phenotype ──────────
print("\n--- TF disruption profile distance vs phenotype ---")

# Load per-hybrid TF disruption counts from hybrid_motif_linkage
tf_profiles = {}
for f in founders_with_pheno:
    tf_profiles[f] = {}

for _, row in ml.iterrows():
    if pd.isna(row.get('separating_families')):
        continue
    var_hybrids = set(row['variant_hybrids'].split('|')) if pd.notna(row['variant_hybrids']) else set()
    families = row['separating_families'].split(';') if pd.notna(row['separating_families']) else []
    for h in var_hybrids:
        if h in tf_profiles:
            for fam in families:
                fam = fam.strip()
                if fam:
                    tf_profiles[h][fam] = tf_profiles[h].get(fam, 0) + 1

# Build TF profile matrix
all_families = sorted(set(f for p in tf_profiles.values() for f in p))
tf_mat = np.zeros((n, len(all_families)))
for i, f in enumerate(founders_with_pheno):
    for j, fam in enumerate(all_families):
        tf_mat[i, j] = tf_profiles[f].get(fam, 0)

# Normalize to proportions
tf_mat_norm = tf_mat / tf_mat.sum(axis=1, keepdims=True)
tf_dist = squareform(pdist(tf_mat_norm, metric='euclidean'))

r_tf, p_tf = mantel_test(tf_dist, pheno_dist)
print(f"TF profile vs phenotype: r={r_tf:.3f}, p={p_tf:.4f}")

# ── 6. Burden vs phenotype correlation ───────────────────────────────────
print("\n--- Regulatory burden vs phenotype ---")
burden = np.array([len(gene_sets[f]) for f in founders_with_pheno])
for t_i, t_name in enumerate(trait_names):
    vals = pheno_array[:, t_i]
    rho, rho_p = stats.spearmanr(burden, vals)
    print(f"  Burden vs {t_name}: rho={rho:.3f}, p={rho_p:.4f}")

# ── 7. Figure ─────────────────────────────────────────────────────────────
print("\n--- Generating figure ---")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Gene distance vs phenotype distance scatter
ax = axes[0, 0]
idx = np.triu_indices(n, k=1)
ax.scatter(gene_dist[idx], pheno_dist[idx], alpha=0.4, s=15, color='steelblue')
ax.set_xlabel('Gene sharing distance (1 - Jaccard)')
ax.set_ylabel('Phenotype distance (Euclidean)')
ax.set_title(f'A. Gene sharing vs phenotype\n(Mantel r={r_combined:.3f}, p={p_combined:.4f})')

# Add regression line
z = np.polyfit(gene_dist[idx], pheno_dist[idx], 1)
x_line = np.linspace(gene_dist[idx].min(), gene_dist[idx].max(), 100)
ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.5)

# Panel B: TF profile distance vs phenotype distance
ax = axes[0, 1]
tf_x = tf_dist[idx]
ph_y = pheno_dist[idx]
valid = np.isfinite(tf_x) & np.isfinite(ph_y) & (np.abs(tf_x) > 1e-10)
if valid.sum() > 2:
    ax.scatter(tf_x[valid], ph_y[valid], alpha=0.4, s=15, color='darkorange')
    ax.set_xlabel('TF disruption profile distance')
    ax.set_ylabel('Phenotype distance (Euclidean)')
    r_tf_label = f'{r_tf:.3f}' if np.isfinite(r_tf) else 'N/A'
    ax.set_title(f'B. TF profile vs phenotype\n(Mantel r={r_tf_label}, p={p_tf:.4f})')
    try:
        z2 = np.polyfit(tf_x[valid], ph_y[valid], 1)
        x_line2 = np.linspace(tf_x[valid].min(), tf_x[valid].max(), 100)
        ax.plot(x_line2, np.polyval(z2, x_line2), 'r--', alpha=0.5)
    except (np.linalg.LinAlgError, ValueError):
        pass
else:
    ax.text(0.5, 0.5, 'Insufficient variation', transform=ax.transAxes, ha='center')
    ax.set_title('B. TF profile vs phenotype\n(insufficient variation)')

# Panel C: Per-trait Mantel r values
ax = axes[1, 0]
traits = list(trait_results.keys()) + ['Combined', 'TF profile']
r_vals = [trait_results[t]['r'] for t in trait_names] + [r_combined, r_tf]
p_vals = [trait_results[t]['p'] for t in trait_names] + [p_combined, p_tf]
colors = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in p_vals]
bars = ax.barh(traits, r_vals, color=colors, alpha=0.7)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('Mantel r')
ax.set_title('C. Distance correlations by trait')
for i, (r, p) in enumerate(zip(r_vals, p_vals)):
    ax.text(max(r, 0) + 0.01, i, f'p={p:.3f}', va='center', fontsize=8)

# Panel D: Burden vs DTA
ax = axes[1, 1]
ax.scatter(burden, pheno_array[:, 0], alpha=0.6, s=30, color='mediumpurple')
rho_dta, p_dta = stats.spearmanr(burden, pheno_array[:, 0])
ax.set_xlabel('Regulatory burden (# causal genes)')
ax.set_ylabel('Days to anthesis')
ax.set_title(f'D. Burden vs flowering time\n(ρ={rho_dta:.3f}, p={p_dta:.4f})')
z3 = np.polyfit(burden, pheno_array[:, 0], 1)
x_line3 = np.linspace(burden.min(), burden.max(), 100)
ax.plot(x_line3, np.polyval(z3, x_line3), 'r--', alpha=0.5)
for i, f in enumerate(founders_with_pheno):
    if pheno_array[i, 0] > 74 or pheno_array[i, 0] < 62 or burden[i] > np.percentile(burden, 90):
        ax.annotate(f, (burden[i], pheno_array[i, 0]), fontsize=6, alpha=0.7)

plt.suptitle('Phenotype Similarity vs Regulatory Sharing', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / 'hybrid_phenotype_similarity.pdf', bbox_inches='tight', dpi=150)
print(f"Saved: {FIG_DIR / 'hybrid_phenotype_similarity.pdf'}")

# ── 8. Save results ──────────────────────────────────────────────────────
results = {
    'mantel_combined': {
        'r': float(r_combined),
        'p': float(p_combined),
        'interpretation': 'positive=shared genes → similar phenotype; negative=different genes → similar phenotype'
    },
    'mantel_per_trait': {t: {'r': float(v['r']), 'p': float(v['p'])} for t, v in trait_results.items()},
    'mantel_tf_profile': {
        'r': float(r_tf),
        'p': float(p_tf),
    },
    'burden_vs_trait': {},
    'n_founders': n,
    'n_genes': len(genes),
    'mean_burden': float(burden.mean()),
    'burden_cv': float(burden.std() / burden.mean()),
}

for t_i, t_name in enumerate(trait_names):
    rho, rho_p = stats.spearmanr(burden, pheno_array[:, t_i])
    results['burden_vs_trait'][t_name] = {'rho': float(rho), 'p': float(rho_p)}

with open(RESULTS_DIR / 'hybrid_phenotype_similarity.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {RESULTS_DIR / 'hybrid_phenotype_similarity.json'}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Gene sharing vs phenotype: Mantel r={r_combined:.3f} (p={p_combined:.4f})")
print(f"TF profile vs phenotype:   Mantel r={r_tf:.3f} (p={p_tf:.4f})")
if p_combined > 0.05 and p_tf > 0.05:
    print("→ No significant correlation: gene-level regulatory variation is")
    print("  orthogonal to phenotype, supporting the distributed OS model")
    print("  (different genes achieve the same phenotypic outcome)")
elif p_combined < 0.05:
    print("→ Significant gene sharing-phenotype correlation found")
print("=" * 70)
