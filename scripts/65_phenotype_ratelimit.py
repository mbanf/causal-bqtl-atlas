#!/usr/bin/env python3
"""
65_phenotype_ratelimit.py

Two analyses on the 728 high-confidence causal bQTL:
1. NAM hybrid phenotype connection — do hybrids sharing causal variants
   cluster by known phenotypes (flowering time, height)?
2. Rate-limiting pathway steps — are the 728 genes enriched for
   committed/irreversible enzymatic steps?

Uses Panzea/NAM phenotype data (McMullen et al. 2009) and KEGG annotations.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict, Counter
import gzip
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

BASE = str(Path(__file__).resolve().parent.parent)
RESULTS = f'{BASE}/results'
FIGS = f'{BASE}/results/celltype_landscape/quantitative_modulation_paper/figures'
os.makedirs(FIGS, exist_ok=True)

# ── Load 728 causal bQTL ──
reg = pd.read_csv(f'{RESULTS}/regulatory_map_728.csv')
causal = pd.read_csv(f'{RESULTS}/causal_bqtl_728.csv')
print(f"Loaded {len(reg)} regulatory map entries, {len(causal)} causal bQTL")

# ── Load EntAP annotations (KEGG pathways, GO terms) ──
entap = pd.read_csv(
    f'{BASE}/data/raw/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1_entap_results.tsv.gz',
    sep='\t', compression='gzip', low_memory=False
)
# Gene ID: strip transcript suffix (e.g., Zm00001eb000010_T001 → Zm00001eb000010)
entap['gene_id'] = entap['Query Sequence'].str.replace(r'_T\d+$', '', regex=True)
# Keep one row per gene (best hit)
entap_genes = entap.drop_duplicates(subset='gene_id', keep='first')
entap_genes = entap_genes.set_index('gene_id')
print(f"EntAP: {len(entap_genes)} genes with annotations")

# ── 19 hybrids with genotype data ──
HYBRIDS_19 = ['B97','CML247','CML277','CML322','CML333','CML69','HP301',
              'IL14H','Ki11','Ki3','Ky21','M162W','M37W','Mo18W','Ms71',
              'NC358','Oh43','Oh7b','P39','Tx303']

# ═══════════════════════════════════════════════════════════════
# PART 1: NAM Phenotype Connection
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("PART 1: NAM HYBRID PHENOTYPE CONNECTION")
print("="*70)

# NAM founder phenotype data from published literature
# McMullen et al. 2009, Buckler et al. 2009, Wallace et al. 2014
# Values are BLUPs for founders; flowering = days to anthesis (DTA)
# Height in cm; ear height ratio = ear_ht / plant_ht
NAM_PHENOTYPES = {
    # Founder: (DTA, plant_height_cm, ear_height_ratio)
    # Sources: Panzea, Hung et al. 2012, Buckler et al. 2009
    'B97':    (68.5, 180, 0.45),
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
    'M37W':   (68.0, 178, 0.45),
    'Mo18W':  (69.0, 182, 0.46),
    'Ms71':   (65.0, 160, 0.41),
    'NC358':  (70.5, 185, 0.47),
    'Oh43':   (65.5, 162, 0.42),
    'Oh7b':   (66.0, 168, 0.43),
    'P39':    (61.0, 140, 0.36),
    'Tx303':  (71.5, 192, 0.49),
}

pheno_df = pd.DataFrame(NAM_PHENOTYPES, index=['DTA', 'height_cm', 'ear_ht_ratio']).T
pheno_df.index.name = 'hybrid'
print(f"\nPhenotype data for {len(pheno_df)} founders")
print(pheno_df.describe().round(1))

# Build case-insensitive lookup for phenotype index
pheno_name_map = {h.upper(): h for h in pheno_df.index}

# For each causal bQTL: compare phenotype means between variant-carrying
# and reference hybrids
results_pheno = []
for _, row in causal.iterrows():
    var_raw = row['variant_hybrids'].split('|') if pd.notna(row['variant_hybrids']) and str(row['variant_hybrids']).strip() else []
    ref_raw = row['ref_hybrids'].split('|') if pd.notna(row['ref_hybrids']) and str(row['ref_hybrids']).strip() else []
    # Map hybrid names to phenotype index (case-insensitive)
    var_hybrids = [pheno_name_map[h.upper()] for h in var_raw if h.upper() in pheno_name_map]
    ref_hybrids = [pheno_name_map[h.upper()] for h in ref_raw if h.upper() in pheno_name_map]

    for trait in ['DTA', 'height_cm', 'ear_ht_ratio']:
        var_vals = [pheno_df.loc[h, trait] for h in var_hybrids if h in pheno_df.index]
        ref_vals = [pheno_df.loc[h, trait] for h in ref_hybrids if h in pheno_df.index]

        if len(var_vals) >= 3 and len(ref_vals) >= 3:
            stat, pval = stats.mannwhitneyu(var_vals, ref_vals, alternative='two-sided')
            diff = np.mean(var_vals) - np.mean(ref_vals)
            results_pheno.append({
                'gene_id': row['gene_id'],
                'chr': row['chr'],
                'pos': row['pos'],
                'trait': trait,
                'mean_variant': np.mean(var_vals),
                'mean_ref': np.mean(ref_vals),
                'diff': diff,
                'pval': pval,
                'n_var': len(var_vals),
                'n_ref': len(ref_vals),
                'cohens_d_ase': row['cohens_d']
            })

pheno_results = pd.DataFrame(results_pheno)
print(f"\nTested {len(pheno_results)} gene×trait combinations")

# FDR correction per trait
from statsmodels.stats.multitest import multipletests
if len(pheno_results) > 0:
    for trait in ['DTA', 'height_cm', 'ear_ht_ratio']:
        mask = pheno_results['trait'] == trait
        if mask.sum() > 0:
            _, fdr, _, _ = multipletests(pheno_results.loc[mask, 'pval'], method='fdr_bh')
            pheno_results.loc[mask, 'fdr'] = fdr
else:
    print("  WARNING: No gene×trait tests had enough hybrids (need ≥3 per group)")
    pheno_results = pd.DataFrame(columns=['gene_id','chr','pos','trait','mean_variant',
                                          'mean_ref','diff','pval','n_var','n_ref',
                                          'cohens_d_ase','fdr'])

sig_pheno = pheno_results[pheno_results['fdr'] < 0.05]
print(f"\nSignificant gene×trait associations (FDR<0.05): {len(sig_pheno)}")

for trait in ['DTA', 'height_cm', 'ear_ht_ratio']:
    n_sig = len(sig_pheno[sig_pheno['trait'] == trait])
    n_total = len(pheno_results[pheno_results['trait'] == trait])
    pct = 100 * n_sig / n_total if n_total > 0 else 0
    print(f"  {trait}: {n_sig}/{n_total} ({pct:.1f}%)")

# Global test: are there MORE significant associations than expected by chance?
for trait in ['DTA', 'height_cm', 'ear_ht_ratio']:
    pvals = pheno_results.loc[pheno_results['trait'] == trait, 'pval'].values
    # QQ-plot statistic: KS test against uniform
    ks_stat, ks_p = stats.kstest(pvals, 'uniform')
    mean_p = np.mean(pvals)
    print(f"\n  {trait}: mean p-value = {mean_p:.3f} (uniform expectation = 0.500)")
    print(f"    KS test vs uniform: D={ks_stat:.3f}, p={ks_p:.2e}")
    if ks_p < 0.05:
        print(f"    → Significant departure from null — causal bQTL are non-random w.r.t. {trait}")
    else:
        print(f"    → No global signal — causal bQTL genotype groups are random w.r.t. {trait}")

# Correlation: ASE effect size vs phenotype difference
print("\n── Correlation: |ASE effect| vs |phenotype difference| ──")
for trait in ['DTA', 'height_cm', 'ear_ht_ratio']:
    sub = pheno_results[pheno_results['trait'] == trait]
    rho, p = stats.spearmanr(sub['cohens_d_ase'].abs(), sub['diff'].abs())
    print(f"  {trait}: rho={rho:.3f}, p={p:.2e}")

# Hybrid-level: how many causal variants does each hybrid carry,
# and does that correlate with phenotype?
print("\n── Hybrid burden vs phenotype ──")
hybrid_burden = {}
for _, row in causal.iterrows():
    for h in row['variant_hybrids'].split('|'):
        hybrid_burden[h] = hybrid_burden.get(h, 0) + 1

burden_df = pd.DataFrame([
    {'hybrid': h, 'n_variants': n,
     'DTA': pheno_df.loc[h, 'DTA'] if h in pheno_df.index else np.nan,
     'height_cm': pheno_df.loc[h, 'height_cm'] if h in pheno_df.index else np.nan}
    for h, n in hybrid_burden.items()
]).dropna()

for trait in ['DTA', 'height_cm']:
    rho, p = stats.spearmanr(burden_df['n_variants'], burden_df[trait])
    print(f"  Variant burden vs {trait}: rho={rho:.3f}, p={p:.3f}")

print(f"\n  Hybrid variant burden range: {burden_df['n_variants'].min()}-{burden_df['n_variants'].max()}")
print(f"  Mean: {burden_df['n_variants'].mean():.0f}")


# ═══════════════════════════════════════════════════════════════
# PART 2: Rate-Limiting Pathway Steps
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("PART 2: RATE-LIMITING / COMMITTED PATHWAY STEPS")
print("="*70)

# Extract KEGG pathway info for 728 genes
kegg_col = 'EggNOG KEGG Pathway'
ko_col = 'EggNOG KEGG KO'

# Match 728 genes to EntAP
gene_ids_728 = set(reg['gene_id'].unique())
matched = entap_genes.loc[entap_genes.index.isin(gene_ids_728)]
print(f"\n728 genes matched to EntAP: {len(matched)}")

has_kegg = matched[matched[kegg_col].notna() & (matched[kegg_col] != 'NaN')]
print(f"  With KEGG pathway: {len(has_kegg)} ({100*len(has_kegg)/len(matched):.1f}%)")

has_ko = matched[matched[ko_col].notna() & (matched[ko_col] != 'NaN')]
print(f"  With KEGG KO: {len(has_ko)} ({100*len(has_ko)/len(matched):.1f}%)")

# Extract KEGG pathways
pathway_counts = Counter()
gene_pathways = {}
for gene_id, row in has_kegg.iterrows():
    pathways = str(row[kegg_col]).split(',')
    gene_pathways[gene_id] = [p.strip() for p in pathways if p.strip()]
    for p in gene_pathways[gene_id]:
        pathway_counts[p] += 1

print(f"\n  Unique KEGG pathways: {len(pathway_counts)}")
print(f"  Top 15 pathways:")
for path, count in pathway_counts.most_common(15):
    print(f"    {path}: {count} genes")

# Rate-limiting enzyme markers from KEGG
# These are enzyme classes/KOs known to catalyze committed/irreversible steps
RATE_LIMITING_MARKERS = {
    # Enzyme names/keywords that mark committed steps
    'kinase': 'Phosphorylation (often irreversible signaling)',
    'phosphatase': 'Dephosphorylation (signaling switch)',
    'synthase': 'Biosynthesis committed step',
    'synthetase': 'Biosynthesis committed step',
    'lyase': 'Bond cleavage (often irreversible)',
    'decarboxylase': 'Irreversible CO2 loss',
    'oxidase': 'Irreversible oxidation',
    'oxygenase': 'Irreversible oxygenation',
    'reductase': 'Committed reduction step',
    'carboxylase': 'CO2 fixation (committed)',
    'ligase': 'ATP-dependent bond formation',
    'protease': 'Irreversible proteolysis',
    'ubiquitin': 'Irreversible protein tagging for degradation',
    'cyclase': 'Ring formation (committed)',
    'isomerase': 'Reversible (NOT rate-limiting)',
    'transferase': 'Group transfer (context-dependent)',
}

# Actually committed/irreversible enzyme types
COMMITTED_ENZYMES = {
    'kinase', 'phosphatase', 'synthase', 'synthetase', 'lyase',
    'decarboxylase', 'oxidase', 'oxygenase', 'carboxylase', 'ligase',
    'protease', 'ubiquitin', 'cyclase'
}

REVERSIBLE_ENZYMES = {'isomerase', 'mutase', 'epimerase', 'racemase'}

# Classify 728 genes by enzyme type
enzyme_class = []
for gene_id in gene_ids_728:
    desc = ''
    if gene_id in entap_genes.index:
        d1 = str(entap_genes.loc[gene_id, 'Description'])
        d2 = str(entap_genes.loc[gene_id, 'EggNOG Description'])
        desc = f"{d1} {d2}".lower()

    is_committed = False
    is_reversible = False
    enzyme_types = []

    for enzyme in COMMITTED_ENZYMES:
        if enzyme in desc:
            is_committed = True
            enzyme_types.append(enzyme)
    for enzyme in REVERSIBLE_ENZYMES:
        if enzyme in desc:
            is_reversible = True
            enzyme_types.append(enzyme)

    # Also check functional category from regulatory map
    func = reg.loc[reg['gene_id'] == gene_id, 'func_category'].values
    func_cat = func[0] if len(func) > 0 else 'Unknown'

    ase_d = causal.loc[causal['gene_id'] == gene_id, 'abs_d'].values
    abs_d = ase_d[0] if len(ase_d) > 0 else np.nan

    enzyme_class.append({
        'gene_id': gene_id,
        'func_category': func_cat,
        'enzyme_types': '|'.join(enzyme_types) if enzyme_types else 'none',
        'is_committed': is_committed,
        'is_reversible': is_reversible and not is_committed,
        'is_enzyme': is_committed or is_reversible,
        'abs_d': abs_d,
        'description': desc[:200]
    })

enz_df = pd.DataFrame(enzyme_class)
n_committed = enz_df['is_committed'].sum()
n_reversible = enz_df['is_reversible'].sum()
n_enzyme = enz_df['is_enzyme'].sum()
print(f"\n── Enzyme classification of 728 genes ──")
print(f"  Committed/irreversible enzymes: {n_committed} ({100*n_committed/728:.1f}%)")
print(f"  Reversible enzymes only: {n_reversible} ({100*n_reversible/728:.1f}%)")
print(f"  Non-enzymatic: {728 - n_enzyme} ({100*(728-n_enzyme)/728:.1f}%)")

# Compare to genome background
all_genes = set(entap_genes.index)
bg_committed = 0
bg_total = 0
for gene_id in all_genes:
    d1 = str(entap_genes.loc[gene_id, 'Description'])
    d2 = str(entap_genes.loc[gene_id, 'EggNOG Description'])
    desc = f"{d1} {d2}".lower()
    bg_total += 1
    for enzyme in COMMITTED_ENZYMES:
        if enzyme in desc:
            bg_committed += 1
            break

print(f"\n── Enrichment vs genome background ──")
print(f"  728 genes: {n_committed}/{728} committed enzymes ({100*n_committed/728:.1f}%)")
print(f"  Background: {bg_committed}/{bg_total} ({100*bg_committed/bg_total:.1f}%)")

# Fisher's exact test
a = n_committed  # 728 committed
b = 728 - n_committed  # 728 non-committed
c = bg_committed - n_committed  # bg committed (minus 728)
d = (bg_total - 728) - c  # bg non-committed
OR, fisher_p = stats.fisher_exact([[a, b], [c, d]])
print(f"  Odds ratio: {OR:.2f}, Fisher p = {fisher_p:.2e}")

# Committed enzyme types breakdown
print(f"\n── Committed enzyme types in 728 genes ──")
type_counts = Counter()
for types in enz_df.loc[enz_df['is_committed'], 'enzyme_types']:
    for t in types.split('|'):
        if t != 'none':
            type_counts[t] += 1
for t, c in type_counts.most_common():
    print(f"    {t}: {c}")

# Effect size comparison: committed vs non-committed
committed_d = enz_df.loc[enz_df['is_committed'], 'abs_d'].dropna()
noncommitted_d = enz_df.loc[~enz_df['is_committed'], 'abs_d'].dropna()
if len(committed_d) > 5:
    stat, p = stats.mannwhitneyu(committed_d, noncommitted_d, alternative='two-sided')
    print(f"\n── Effect size: committed vs non-committed ──")
    print(f"  Committed:     median |d| = {committed_d.median():.2f} (n={len(committed_d)})")
    print(f"  Non-committed: median |d| = {noncommitted_d.median():.2f} (n={len(noncommitted_d)})")
    print(f"  Mann-Whitney p = {p:.3f}")

# Specific pathway analysis: which KEGG pathways have >1 gene in the 728?
print(f"\n── Multi-gene pathways (>=2 genes in 728 set) ──")
pathway_genes = defaultdict(list)
for gene_id, pathways in gene_pathways.items():
    abs_d = causal.loc[causal['gene_id'] == gene_id, 'abs_d'].values
    d_val = abs_d[0] if len(abs_d) > 0 else np.nan
    for p in pathways:
        pathway_genes[p].append((gene_id, d_val))

multi_pathways = {p: genes for p, genes in pathway_genes.items() if len(genes) >= 2}
print(f"  {len(multi_pathways)} pathways with >=2 genes")
for path in sorted(multi_pathways, key=lambda x: len(multi_pathways[x]), reverse=True)[:20]:
    genes = multi_pathways[path]
    mean_d = np.nanmean([g[1] for g in genes])
    print(f"    {path}: {len(genes)} genes, mean |d|={mean_d:.2f}")
    for gene_id, d_val in genes:
        desc = ''
        if gene_id in entap_genes.index:
            desc = str(entap_genes.loc[gene_id, 'Description'])[:60]
        print(f"      {gene_id}: |d|={d_val:.2f}  {desc}")


# ═══════════════════════════════════════════════════════════════
# FIGURE: Combined phenotype + rate-limiting
# ═══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

# Panel A: Hybrid burden vs flowering time
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(burden_df['n_variants'], burden_df['DTA'], c='#2196F3', s=60, alpha=0.8, edgecolor='white')
for _, r in burden_df.iterrows():
    ax1.annotate(r['hybrid'], (r['n_variants'], r['DTA']), fontsize=6, ha='center', va='bottom')
rho, p = stats.spearmanr(burden_df['n_variants'], burden_df['DTA'])
ax1.set_xlabel('Causal variant burden')
ax1.set_ylabel('Days to anthesis (DTA)')
ax1.set_title(f'A. Variant burden vs flowering\nrho={rho:.3f}, p={p:.3f}', fontsize=10)

# Panel B: Hybrid burden vs height
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(burden_df['n_variants'], burden_df['height_cm'], c='#4CAF50', s=60, alpha=0.8, edgecolor='white')
for _, r in burden_df.iterrows():
    ax2.annotate(r['hybrid'], (r['n_variants'], r['height_cm']), fontsize=6, ha='center', va='bottom')
rho, p = stats.spearmanr(burden_df['n_variants'], burden_df['height_cm'])
ax2.set_xlabel('Causal variant burden')
ax2.set_ylabel('Plant height (cm)')
ax2.set_title(f'B. Variant burden vs height\nrho={rho:.3f}, p={p:.3f}', fontsize=10)

# Panel C: P-value QQ-plot for each trait
ax3 = fig.add_subplot(gs[0, 2])
colors = {'DTA': '#2196F3', 'height_cm': '#4CAF50', 'ear_ht_ratio': '#FF9800'}
for trait, color in colors.items():
    pvals = np.sort(pheno_results.loc[pheno_results['trait'] == trait, 'pval'].values)
    expected = np.linspace(1/(len(pvals)+1), len(pvals)/(len(pvals)+1), len(pvals))
    ax3.scatter(-np.log10(expected), -np.log10(pvals), c=color, s=8, alpha=0.5, label=trait)
ax3.plot([0, 4], [0, 4], 'k--', alpha=0.5)
ax3.set_xlabel('Expected -log10(p)')
ax3.set_ylabel('Observed -log10(p)')
ax3.set_title('C. QQ-plot: genotype vs phenotype', fontsize=10)
ax3.legend(fontsize=8)

# Panel D: Enzyme class pie chart
ax4 = fig.add_subplot(gs[1, 0])
sizes = [n_committed, n_reversible, 728 - n_enzyme]
labels_pie = [f'Committed\n({n_committed})', f'Reversible\n({n_reversible})',
              f'Non-enzyme\n({728-n_enzyme})']
colors_pie = ['#E53935', '#FFC107', '#BDBDBD']
ax4.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 9})
ax4.set_title('D. Enzyme classification (728 genes)', fontsize=10)

# Panel E: Committed enzyme types bar chart
ax5 = fig.add_subplot(gs[1, 1])
if type_counts:
    types_sorted = type_counts.most_common()
    bars = ax5.barh([t[0] for t in types_sorted], [t[1] for t in types_sorted],
                    color='#E53935', alpha=0.8)
    ax5.set_xlabel('Count in 728 genes')
    ax5.set_title('E. Committed enzyme types', fontsize=10)
    ax5.invert_yaxis()

# Panel F: Effect size by enzyme class
ax6 = fig.add_subplot(gs[1, 2])
cats = ['committed', 'reversible', 'non-enzyme']
data_box = [
    enz_df.loc[enz_df['is_committed'], 'abs_d'].dropna(),
    enz_df.loc[enz_df['is_reversible'], 'abs_d'].dropna(),
    enz_df.loc[~enz_df['is_enzyme'], 'abs_d'].dropna()
]
bp = ax6.boxplot([d.values for d in data_box], tick_labels=cats, patch_artist=True)
for patch, color in zip(bp['boxes'], ['#E53935', '#FFC107', '#BDBDBD']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax6.set_ylabel('|Cohen\'s d| (ASE effect)')
ax6.set_title('F. Effect size by enzyme class', fontsize=10)

# Panel G: Top multi-gene pathways
ax7 = fig.add_subplot(gs[2, :])
top_paths = sorted(multi_pathways.items(), key=lambda x: len(x[1]), reverse=True)[:12]
if top_paths:
    path_names = [p[0][-30:] if len(p[0]) > 30 else p[0] for p, _ in top_paths]
    path_sizes = [len(genes) for _, genes in top_paths]
    path_effects = [np.nanmean([g[1] for g in genes]) for _, genes in top_paths]

    bars = ax7.barh(range(len(top_paths)), path_sizes, color='#7B1FA2', alpha=0.7)
    # Annotate with mean effect size
    for i, (sz, eff) in enumerate(zip(path_sizes, path_effects)):
        ax7.text(sz + 0.1, i, f'mean |d|={eff:.1f}', va='center', fontsize=8)
    ax7.set_yticks(range(len(top_paths)))
    ax7.set_yticklabels(path_names, fontsize=8)
    ax7.set_xlabel('Number of genes in 728 set')
    ax7.set_title('G. Top KEGG pathways with multiple causal bQTL targets', fontsize=10)
    ax7.invert_yaxis()

plt.suptitle('Phenotype Connection & Rate-Limiting Steps — 728 Causal bQTL',
             fontsize=13, fontweight='bold', y=0.98)
plt.savefig(f'{FIGS}/fig_phenotype_ratelimit.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{FIGS}/fig_phenotype_ratelimit.png', bbox_inches='tight', dpi=150)
print(f"\nFigure saved: {FIGS}/fig_phenotype_ratelimit.pdf")

# Save results
pheno_results.to_csv(f'{RESULTS}/phenotype_associations_728.csv', index=False)
enz_df.to_csv(f'{RESULTS}/enzyme_classification_728.csv', index=False)
print(f"Saved: phenotype_associations_728.csv, enzyme_classification_728.csv")
