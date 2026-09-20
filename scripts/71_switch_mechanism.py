#!/usr/bin/env python3
"""
71_switch_mechanism.py

Two questions about TF motif switches:
1. Do switches systematically go from growth→defense or vice versa?
2. Does the direction of expression change correlate with TF family abundance?
   (i.e., does switching TO a more abundant TF family → higher expression?)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy import stats

BASE = str(Path(__file__).resolve().parent.parent)
RESULTS = f'{BASE}/results'

# ── Load data ──
rewire = pd.read_csv(f'{RESULTS}/rewiring_deep_analysis_728.csv')
causal = pd.read_csv(f'{RESULTS}/causal_bqtl_728.csv')
expr = pd.read_csv(f'{BASE}/data/processed/engelhorn_ww_vs_ds_expression.tsv', sep='\t')

# Load TF gene list with family assignments
import json
with open(f'{BASE}/data/processed/maize_tf_pwm_database.json') as f:
    pwm_db = json.load(f)

print(f"Loaded {len(rewire)} bQTL, {len(expr)} expression rows")

# ── QUESTION 1: Growth vs defense directionality ──
print("\n" + "="*70)
print("Q1: Do switches go systematically growth → defense or vice versa?")
print("="*70)

# Classify TF families by primary function
GROWTH_DEV = {'HD-ZIP', 'WOX', 'YABBY', 'TALE', 'MIKC_MADS', 'M-type_MADS',
              'G2-like', 'TCP', 'SBP', 'LBD', 'GRF', 'ARF', 'LFY'}
STRESS_DEF = {'WRKY', 'NAC', 'ERF', 'HSF', 'CAMTA', 'DREB'}
SIGNALING = {'C2H2', 'MYB', 'MYB_related', 'bHLH', 'Dof', 'bZIP', 'BES1',
             'EIL', 'Nin-like', 'CPP', 'Trihelix', 'GRAS', 'GATA',
             'BBR-BPC', 'AP2', 'RAV', 'ARR-B', 'B3', 'E2F/DP'}

def classify_family(fam):
    if fam in GROWTH_DEV:
        return 'growth/dev'
    elif fam in STRESS_DEF:
        return 'stress/defense'
    else:
        return 'signaling'

# Analyze transitions
rewired = rewire[rewire['category'] == 'rewired']
print(f"\nRewired bQTL: {len(rewired)}")

transition_types = Counter()
for _, row in rewired.iterrows():
    d_fams = [f for f in str(row['disrupted_families']).split(',') if f and f != 'nan']
    c_fams = [f for f in str(row['created_families']).split(',') if f and f != 'nan']
    for df in d_fams:
        for cf in c_fams:
            d_class = classify_family(df)
            c_class = classify_family(cf)
            transition_types[(d_class, c_class)] += 1

print("\nTransition type matrix (disrupted class → created class):")
classes = ['growth/dev', 'stress/defense', 'signaling']
print(f"{'':>20}", end='')
for c in classes:
    print(f"{c:>16}", end='')
print()
for d in classes:
    print(f"{d:>20}", end='')
    for c in classes:
        n = transition_types.get((d, c), 0)
        print(f"{n:>16}", end='')
    print()

# Key comparisons
growth_to_defense = transition_types.get(('growth/dev', 'stress/defense'), 0)
defense_to_growth = transition_types.get(('stress/defense', 'growth/dev'), 0)
growth_to_growth = transition_types.get(('growth/dev', 'growth/dev'), 0)
defense_to_defense = transition_types.get(('stress/defense', 'stress/defense'), 0)

print(f"\nGrowth→Defense: {growth_to_defense}")
print(f"Defense→Growth: {defense_to_growth}")
print(f"Growth→Growth: {growth_to_growth}")
print(f"Defense→Defense: {defense_to_defense}")

# Is there a directional bias?
total_cross = growth_to_defense + defense_to_growth
if total_cross > 0:
    print(f"\nCross-category switches: {total_cross}")
    print(f"  Growth→Defense: {growth_to_defense} ({100*growth_to_defense/total_cross:.1f}%)")
    print(f"  Defense→Growth: {defense_to_growth} ({100*defense_to_growth/total_cross:.1f}%)")
    # Binomial test for asymmetry
    from scipy.stats import binomtest
    result = binomtest(growth_to_defense, total_cross, 0.5, alternative='two-sided')
    print(f"  Binomial test for asymmetry: p = {result.pvalue:.4f}")

# Does expression change direction correlate with switch direction?
print("\n--- Expression direction by switch type ---")
for _, row in rewired.iterrows():
    d_fams = [f for f in str(row['disrupted_families']).split(',') if f and f != 'nan']
    c_fams = [f for f in str(row['created_families']).split(',') if f and f != 'nan']

# ── QUESTION 2: TF abundance mechanism ──
print(f"\n{'='*70}")
print("Q2: Does switching TO a more abundant TF → higher expression?")
print("="*70)

# Get TF expression levels per family
# Load the regulatory network TF gene list
try:
    reg_map = pd.read_csv(f'{RESULTS}/regulatory_map_728.csv')
except:
    reg_map = None

# Count expressed TF genes per family from PlantTFDB
# We need to know: for each TF family, how many TF genes are expressed?
# Use EntAP + expression data
entap = pd.read_csv(f'{BASE}/data/raw/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1_entap_results.tsv.gz',
                     sep='\t', low_memory=False)

# Get TF family from PlantTFDB annotations in our database
# Map gene_id to TF family
tf_genes = {}
for m in pwm_db['motifs']:
    gid = m.get('gene_id', '')
    fam = m.get('tf_family', '')
    if gid and fam:
        tf_genes[gid] = fam

print(f"TF genes in PWM database: {len(tf_genes)}")

# Check expression of TF genes
# Expression columns in Engelhorn data
expr_cols = [c for c in expr.columns if c not in ['gene_id', 'Unnamed: 0']]
print(f"Expression columns: {expr_cols[:5]}...")

# For each TF family: count how many TF genes, mean expression
# We need maize gene IDs matching the PWM database gene IDs
# PWM database has Zma_ prefixed IDs from PlantTFDB
# Let's look at what format they're in
sample_ids = list(tf_genes.keys())[:5]
print(f"Sample TF gene IDs from PWM DB: {sample_ids}")

# These are PlantTFDB IDs like 'Zma_GRMZM2G...' - need to map to Zm00001eb format
# Alternative: use the tripartite network TF list
try:
    network = pd.read_csv(f'{BASE}/results/regulatory_chains.csv')
    print(f"Loaded regulatory chains: {len(network)} rows")
    # Get unique TF families and their gene counts
    tf_family_genes = network.groupby('tf_family')['tf_gene_id'].nunique()
    print(f"\nTF genes per family (from regulatory network):")
    for fam, n in tf_family_genes.sort_values(ascending=False).head(20).items():
        print(f"  {fam}: {n} TF genes")
except Exception as e:
    print(f"Could not load regulatory chains: {e}")
    # Fall back: use expression data directly
    tf_family_genes = None

# Alternative approach: for each TF family, estimate total TF abundance
# from the number of expressed TF genes × their expression level
# Using the tripartite network data
try:
    chains = pd.read_csv(f'{BASE}/results/regulatory_chains.csv')

    # Get unique TF genes per family with expression
    tf_expr = chains[['tf_family', 'tf_gene_id']].drop_duplicates()

    # Merge with expression data
    # Expression data gene_id format?
    print(f"\nExpression data gene_id format: {expr.iloc[0]['gene_id'] if 'gene_id' in expr.columns else 'no gene_id'}")

    # Estimate family "abundance" as number of expressed TF genes
    family_abundance = tf_expr.groupby('tf_family').size().to_dict()
    print(f"\nFamily abundance (# expressed TF genes):")
    for fam in sorted(family_abundance, key=family_abundance.get, reverse=True)[:15]:
        print(f"  {fam}: {family_abundance[fam]} genes")

except Exception as e:
    print(f"Error: {e}")
    family_abundance = {}

# ── Key test: for rewired bQTL, does Cohen's d direction correlate
#    with the abundance difference between created and disrupted families? ──
print(f"\n{'='*70}")
print("KEY TEST: Does switching to a more abundant TF family → upregulation?")
print("="*70)

if family_abundance:
    results = []
    for _, row in rewired.iterrows():
        d_fams = [f for f in str(row['disrupted_families']).split(',') if f and f != 'nan']
        c_fams = [f for f in str(row['created_families']).split(',') if f and f != 'nan']

        if not d_fams or not c_fams:
            continue

        # Mean abundance of disrupted vs created families
        d_abund = np.mean([family_abundance.get(f, 0) for f in d_fams])
        c_abund = np.mean([family_abundance.get(f, 0) for f in c_fams])

        if d_abund == 0 and c_abund == 0:
            continue

        abundance_ratio = np.log2((c_abund + 1) / (d_abund + 1))  # positive = created more abundant

        results.append({
            'gene_id': row['gene_id'],
            'cohens_d': row['cohens_d'],
            'abundance_ratio': abundance_ratio,
            'd_abund': d_abund,
            'c_abund': c_abund,
            'disrupted': ','.join(d_fams),
            'created': ','.join(c_fams),
        })

    test_df = pd.DataFrame(results)
    print(f"Testable rewired bQTL: {len(test_df)}")

    # Correlation: abundance_ratio vs Cohen's d
    rho, p = stats.spearmanr(test_df['abundance_ratio'], test_df['cohens_d'])
    print(f"\nSpearman correlation (abundance ratio vs Cohen's d):")
    print(f"  rho = {rho:.3f}, p = {p:.4f}")

    # Split by direction
    to_more = test_df[test_df['abundance_ratio'] > 0]
    to_less = test_df[test_df['abundance_ratio'] < 0]
    print(f"\nSwitch to MORE abundant TF family: n={len(to_more)}")
    print(f"  Mean Cohen's d: {to_more['cohens_d'].mean():.3f}")
    print(f"  % positive d (B73 > NAM): {100*(to_more['cohens_d'] > 0).mean():.1f}%")

    print(f"\nSwitch to LESS abundant TF family: n={len(to_less)}")
    print(f"  Mean Cohen's d: {to_less['cohens_d'].mean():.3f}")
    print(f"  % positive d (B73 > NAM): {100*(to_less['cohens_d'] > 0).mean():.1f}%")

    # Mann-Whitney
    if len(to_more) > 5 and len(to_less) > 5:
        stat, p2 = stats.mannwhitneyu(to_more['cohens_d'], to_less['cohens_d'],
                                       alternative='two-sided')
        print(f"\n  Mann-Whitney (d of more-abundant vs less-abundant): p = {p2:.4f}")

# ── Also test: do activator vs repressor families differ? ──
print(f"\n{'='*70}")
print("ACTIVATOR vs REPRESSOR families")
print("="*70)

# Known activator families in plants
ACTIVATORS = {'ERF', 'bHLH', 'bZIP', 'MYB', 'NAC', 'WRKY', 'TCP', 'ARF',
              'G2-like', 'NF-Y', 'Nin-like', 'BES1', 'CAMTA'}
# Known repressor-enriched families
REPRESSORS = {'EIL', 'GRAS', 'Trihelix', 'RAV', 'LBD'}
# Mixed/context-dependent
MIXED = {'C2H2', 'HD-ZIP', 'HSF', 'TALE', 'Dof', 'MIKC_MADS', 'M-type_MADS',
         'MYB_related', 'SBP', 'CPP', 'GATA', 'WOX', 'YABBY', 'AP2',
         'ARR-B', 'B3', 'BBR-BPC'}

# For rewired bQTL: is disrupted family an activator and created a repressor?
act_to_rep = 0
rep_to_act = 0
act_d_vals = []
rep_d_vals = []

for _, row in rewired.iterrows():
    d_fams = [f for f in str(row['disrupted_families']).split(',') if f and f != 'nan']
    c_fams = [f for f in str(row['created_families']).split(',') if f and f != 'nan']

    for df in d_fams:
        for cf in c_fams:
            if df in ACTIVATORS and cf in REPRESSORS:
                act_to_rep += 1
                act_d_vals.append(row['cohens_d'])
            elif df in REPRESSORS and cf in ACTIVATORS:
                rep_to_act += 1
                rep_d_vals.append(row['cohens_d'])

print(f"Activator→Repressor switches: {act_to_rep}")
print(f"Repressor→Activator switches: {rep_to_act}")
if act_d_vals:
    print(f"  Act→Rep mean Cohen's d: {np.mean(act_d_vals):.3f}")
if rep_d_vals:
    print(f"  Rep→Act mean Cohen's d: {np.mean(rep_d_vals):.3f}")

# ── Summary: possible mechanisms ──
print(f"\n{'='*70}")
print("MECHANISTIC SUMMARY")
print("="*70)
print("""
Why does changing the TF family affect expression level?

Possible mechanisms (not mutually exclusive):
1. ACTIVATOR vs REPRESSOR: The new TF family may have opposite transcriptional
   activity. E.g., losing a NAC activator and gaining a Trihelix repressor
   would reduce expression.

2. TF ABUNDANCE: If the new TF family has more expressed members in this tissue,
   the promoter is occupied more frequently → higher expression. This is the
   "effective concentration" hypothesis.

3. BINDING AFFINITY: Even within the same peak, the new motif may have a
   different affinity than the old one, changing occupancy fraction.

4. COFACTOR RECRUITMENT: Different TF families recruit different coactivators/
   corepressors (Mediator subunits, chromatin remodelers), which affect
   transcription rate independently of occupancy.

5. COMPETITIVE DISPLACEMENT: The new TF may outcompete a previously bound
   factor, or the loss of the old TF may relieve competition for a shared
   cofactor, indirectly affecting expression.
""")
