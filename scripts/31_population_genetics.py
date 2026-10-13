"""
31_population_genetics.py

Population genetics analysis of causal vs non-causal bQTL.
Tests whether causal bQTL show different allele frequency spectra or
selection signatures compared to non-causal bQTL.

Analyses:
  1. Minor allele frequency (MAF) spectrum: causal vs non-causal
  2. Derived allele frequency (DAF) if ancestral can be inferred
  3. Heterozygosity at causal vs non-causal sites
  4. Ancestry group enrichment (Stiff Stalk, Non-Stiff Stalk, Tropical, etc.)
  5. Effect size vs allele frequency relationship
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

FIG_DIR = config.figures_dir
RESULTS_DIR = config.results_dir

# ── NAM founder ancestry groups ──────────────────────────────────────────
# Based on Flint-Garcia et al. 2005, Yu et al. 2006
ANCESTRY = {
    'Stiff Stalk': ['B97', 'IL14H', 'Mo17', 'Mo18W', 'Oh43', 'Oh7b'],
    'Non-Stiff Stalk': ['A619', 'Ky21', 'M162W', 'Ms71', 'W22'],
    'Iodent': ['HP301', 'P39'],
    'Lancaster': ['Ki11'],
    'Tropical': ['CML103', 'CML247', 'CML277', 'CML322', 'CML333', 'CML69', 'Ki3', 'NC358', 'Tx303'],
    'Popcorn/Sweet': ['A188'],
}

# Reverse lookup
FOUNDER_ANCESTRY = {}
for group, founders in ANCESTRY.items():
    for f in founders:
        FOUNDER_ANCESTRY[f] = group

# ── 1. Load data ──────────────────────────────────────────────────────────
print("=" * 70)
print("POPULATION GENETICS OF CAUSAL bQTL")
print("=" * 70)

# All bQTL
bqtl = pd.read_csv(config.bqtl)
print(f"All bQTL: {len(bqtl)}")

# Causal bQTL
causal = pd.read_csv(RESULTS_DIR / 'causal_bqtl_728.csv')
causal_positions = set(zip(causal['chr'], causal['pos']))
print(f"Causal bQTL: {len(causal_positions)}")

# Genotypes
print("Loading genotypes...")
geno = pd.read_csv(config.genotypes, sep='\t')
founders = [c for c in geno.columns if c not in ('chr', 'pos', 'ref')]
print(f"Founders: {len(founders)}")

# ── 2. Compute allele frequencies ─────────────────────────────────────────
print("\n--- Computing allele frequencies ---")

def compute_af(row, founders):
    """Compute variant allele frequency from genotype row."""
    ref = row['ref']
    n_ref = 0
    n_alt = 0
    n_missing = 0
    for f in founders:
        val = str(row[f])
        if val in ('./.', 'nan', ''):
            n_missing += 1
        elif val == '0|0':
            n_ref += 1
        else:
            n_alt += 1
    total = n_ref + n_alt
    if total == 0:
        return np.nan, n_alt, n_ref, n_missing
    return n_alt / total, n_alt, n_ref, n_missing

print("Computing AFs for all positions...")
afs = []
for idx, row in geno.iterrows():
    if idx % 20000 == 0:
        print(f"  {idx}/{len(geno)}...")
    af, n_alt, n_ref, n_miss = compute_af(row, founders)
    is_causal = (row['chr'], row['pos']) in causal_positions
    afs.append({
        'chr': row['chr'],
        'pos': row['pos'],
        'af': af,
        'maf': min(af, 1 - af) if not np.isnan(af) else np.nan,
        'n_alt': n_alt,
        'n_ref': n_ref,
        'n_missing': n_miss,
        'is_causal': is_causal
    })

af_df = pd.DataFrame(afs)
af_df = af_df.dropna(subset=['af'])
print(f"Valid AF computed: {len(af_df)}")

causal_af = af_df[af_df['is_causal']]
noncausal_af = af_df[~af_df['is_causal']]
print(f"Causal: {len(causal_af)}, Non-causal: {len(noncausal_af)}")

# ── 3. MAF comparison ────────────────────────────────────────────────────
print("\n--- MAF comparison ---")
mw_stat, mw_p = stats.mannwhitneyu(causal_af['maf'], noncausal_af['maf'], alternative='two-sided')
ks_stat, ks_p = stats.ks_2samp(causal_af['maf'], noncausal_af['maf'])

print(f"Causal MAF:     {causal_af['maf'].mean():.3f} ± {causal_af['maf'].std():.3f} (median {causal_af['maf'].median():.3f})")
print(f"Non-causal MAF: {noncausal_af['maf'].mean():.3f} ± {noncausal_af['maf'].std():.3f} (median {noncausal_af['maf'].median():.3f})")
print(f"Mann-Whitney p: {mw_p:.4e}")
print(f"KS test p:      {ks_p:.4e}")

# ── 4. Frequency spectrum bins ────────────────────────────────────────────
print("\n--- Frequency spectrum ---")
bins = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
causal_hist, _ = np.histogram(causal_af['maf'], bins=bins)
noncausal_hist, _ = np.histogram(noncausal_af['maf'], bins=bins)

causal_frac = causal_hist / causal_hist.sum()
noncausal_frac = noncausal_hist / noncausal_hist.sum()

print(f"{'MAF bin':<15} {'Causal':>10} {'Non-causal':>12} {'Enrichment':>12}")
for i in range(len(bins) - 1):
    label = f"{bins[i]:.2f}-{bins[i+1]:.2f}"
    enrich = causal_frac[i] / noncausal_frac[i] if noncausal_frac[i] > 0 else np.inf
    print(f"{label:<15} {causal_frac[i]:>10.3f} {noncausal_frac[i]:>12.3f} {enrich:>12.2f}×")

# ── 5. Ancestry group enrichment ──────────────────────────────────────────
print("\n--- Ancestry group enrichment at causal bQTL ---")

# For each causal bQTL, which ancestry groups carry the variant allele?
causal_with_geno = causal.merge(geno, on=['chr', 'pos'], how='inner')
print(f"Causal bQTL with genotype data: {len(causal_with_geno)}")

group_variant_counts = Counter()
group_total_counts = Counter()

for _, row in causal_with_geno.iterrows():
    for f in founders:
        val = str(row[f])
        if val in ('./.', 'nan', ''):
            continue
        group = FOUNDER_ANCESTRY.get(f, 'Unknown')
        group_total_counts[group] += 1
        if val != '0|0':
            group_variant_counts[group] += 1

print(f"\n{'Group':<20} {'Variant':>8} {'Total':>8} {'Rate':>8}")
for group in sorted(ANCESTRY.keys()):
    v = group_variant_counts[group]
    t = group_total_counts[group]
    rate = v / t if t > 0 else 0
    print(f"{group:<20} {v:>8} {t:>8} {rate:>8.3f}")

# Compare tropical vs temperate variant rates
tropical_founders = set(ANCESTRY['Tropical'])
temperate_founders = set(ANCESTRY['Stiff Stalk'] + ANCESTRY['Non-Stiff Stalk'] + ANCESTRY['Iodent'] + ANCESTRY['Lancaster'])

tropical_variant_per_site = []
temperate_variant_per_site = []

for _, row in causal_with_geno.iterrows():
    trop_v, trop_t = 0, 0
    temp_v, temp_t = 0, 0
    for f in founders:
        val = str(row[f])
        if val in ('./.', 'nan', ''):
            continue
        if f in tropical_founders:
            trop_t += 1
            if val != '0|0':
                trop_v += 1
        elif f in temperate_founders:
            temp_t += 1
            if val != '0|0':
                temp_v += 1
    if trop_t > 0 and temp_t > 0:
        tropical_variant_per_site.append(trop_v / trop_t)
        temperate_variant_per_site.append(temp_v / temp_t)

tropical_variant_per_site = np.array(tropical_variant_per_site)
temperate_variant_per_site = np.array(temperate_variant_per_site)

tt_stat, tt_p = stats.mannwhitneyu(tropical_variant_per_site, temperate_variant_per_site, alternative='two-sided')
print(f"\nTropical variant rate:  {tropical_variant_per_site.mean():.3f}")
print(f"Temperate variant rate: {temperate_variant_per_site.mean():.3f}")
print(f"Mann-Whitney p:         {tt_p:.4e}")

# ── 6. Effect size vs MAF ────────────────────────────────────────────────
print("\n--- Effect size vs MAF ---")
causal_merged = causal.merge(af_df[af_df['is_causal']][['chr', 'pos', 'maf', 'af']], on=['chr', 'pos'], how='inner')
rho, rho_p = stats.spearmanr(causal_merged['maf'], causal_merged['abs_d'])
print(f"MAF vs |Cohen's d|: Spearman rho={rho:.3f}, p={rho_p:.4e}")

# ── 7. Singleton/doubleton enrichment ─────────────────────────────────────
print("\n--- Rare variant enrichment ---")

def count_category(df, col='n_alt'):
    """Categorize by number of carriers."""
    cats = {
        'singleton (1)': (df[col] == 1).sum(),
        'doubleton (2)': (df[col] == 2).sum(),
        'rare (3-5)': ((df[col] >= 3) & (df[col] <= 5)).sum(),
        'common (6-12)': ((df[col] >= 6) & (df[col] <= 12)).sum(),
        'major (>12)': (df[col] > 12).sum(),
    }
    total = sum(cats.values())
    return {k: (v, v / total if total > 0 else 0) for k, v in cats.items()}

causal_cats = count_category(causal_af)
noncausal_cats = count_category(noncausal_af)

print(f"{'Category':<18} {'Causal':>10} {'Non-causal':>12} {'Enrichment':>12}")
for cat in causal_cats:
    c_n, c_f = causal_cats[cat]
    nc_n, nc_f = noncausal_cats[cat]
    enrich = c_f / nc_f if nc_f > 0 else np.inf
    print(f"{cat:<18} {c_f:>10.3f} {nc_f:>12.3f} {enrich:>12.2f}×")

# ── 8. Figure ─────────────────────────────────────────────────────────────
print("\n--- Generating figure ---")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: MAF distributions
ax = axes[0, 0]
ax.hist(causal_af['maf'], bins=30, alpha=0.6, density=True, color='crimson', label=f'Causal (n={len(causal_af)})', edgecolor='white')
ax.hist(noncausal_af['maf'], bins=30, alpha=0.4, density=True, color='steelblue', label=f'Non-causal (n={len(noncausal_af)})', edgecolor='white')
ax.set_xlabel('Minor allele frequency')
ax.set_ylabel('Density')
ax.set_title(f'A. MAF spectrum\n(KS p={ks_p:.2e})')
ax.legend(fontsize=8)

# Panel B: Frequency spectrum bars
ax = axes[0, 1]
x = np.arange(len(bins) - 1)
width = 0.35
labels = [f'{bins[i]:.0%}-{bins[i+1]:.0%}' for i in range(len(bins) - 1)]
ax.bar(x - width/2, causal_frac, width, label='Causal', color='crimson', alpha=0.7)
ax.bar(x + width/2, noncausal_frac, width, label='Non-causal', color='steelblue', alpha=0.7)
ax.set_xlabel('MAF bin')
ax.set_ylabel('Fraction')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_title('B. Allele frequency spectrum')
ax.legend(fontsize=8)

# Panel C: Effect size vs MAF
ax = axes[1, 0]
ax.scatter(causal_merged['maf'], causal_merged['abs_d'], alpha=0.3, s=8, color='crimson')
ax.set_xlabel('Minor allele frequency')
ax.set_ylabel("|Cohen's d|")
ax.set_title(f'C. Effect size vs MAF\n(ρ={rho:.3f}, p={rho_p:.2e})')

# Panel D: Ancestry group variant rates
ax = axes[1, 1]
groups = sorted(ANCESTRY.keys())
rates = [group_variant_counts[g] / group_total_counts[g] if group_total_counts[g] > 0 else 0 for g in groups]
colors = ['#e74c3c' if g == 'Tropical' else '#3498db' for g in groups]
bars = ax.barh(groups, rates, color=colors, alpha=0.7)
ax.set_xlabel('Variant allele rate')
ax.set_title('D. Variant rate by ancestry group')
ax.axvline(np.mean(rates), color='gray', linestyle='--', alpha=0.5, label='Mean')
ax.legend(fontsize=8)

plt.suptitle('Population Genetics of Causal bQTL', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / 'population_genetics.pdf', bbox_inches='tight', dpi=150)
print(f"Saved: {FIG_DIR / 'population_genetics.pdf'}")

# ── 9. Save results ──────────────────────────────────────────────────────
results = {
    'maf_comparison': {
        'causal_mean': float(causal_af['maf'].mean()),
        'causal_median': float(causal_af['maf'].median()),
        'noncausal_mean': float(noncausal_af['maf'].mean()),
        'noncausal_median': float(noncausal_af['maf'].median()),
        'mannwhitney_p': float(mw_p),
        'ks_p': float(ks_p),
    },
    'effect_vs_maf': {
        'spearman_rho': float(rho),
        'p_value': float(rho_p),
    },
    'tropical_vs_temperate': {
        'tropical_variant_rate': float(tropical_variant_per_site.mean()),
        'temperate_variant_rate': float(temperate_variant_per_site.mean()),
        'mannwhitney_p': float(tt_p),
    },
    'ancestry_rates': {g: float(group_variant_counts[g] / group_total_counts[g]) if group_total_counts[g] > 0 else 0 for g in groups},
    'rare_variant_enrichment': {
        cat: {
            'causal_frac': float(causal_cats[cat][1]),
            'noncausal_frac': float(noncausal_cats[cat][1]),
            'enrichment': float(causal_cats[cat][1] / noncausal_cats[cat][1]) if noncausal_cats[cat][1] > 0 else None
        } for cat in causal_cats
    },
    'n_causal': len(causal_af),
    'n_noncausal': len(noncausal_af),
}

with open(RESULTS_DIR / 'population_genetics.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {RESULTS_DIR / 'population_genetics.json'}")

# Save per-site data
af_df.to_csv(RESULTS_DIR / 'bqtl_allele_frequencies.csv', index=False)
print(f"Saved: {RESULTS_DIR / 'bqtl_allele_frequencies.csv'}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"MAF: causal {causal_af['maf'].mean():.3f} vs non-causal {noncausal_af['maf'].mean():.3f} (KS p={ks_p:.2e})")
print(f"Effect × MAF: rho={rho:.3f} (p={rho_p:.2e})")
print(f"Tropical vs temperate variant rate: {tropical_variant_per_site.mean():.3f} vs {temperate_variant_per_site.mean():.3f} (p={tt_p:.2e})")
