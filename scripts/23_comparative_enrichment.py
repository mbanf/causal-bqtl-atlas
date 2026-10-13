"""
23_comparative_enrichment.py — Exhaustive comparative functional enrichment
of WW-only vs DS-only causal bQTL target genes.

Analyses:
  1. GO enrichment per condition (biological process, molecular function, cellular component)
  2. Direct WW vs DS Fisher's exact test per GO term
  3. Detailed functional category breakdown (metabolism subtypes, signaling, stress, etc.)
  4. Comparative figure: side-by-side enrichment + volcano of WW vs DS
"""

import sys, warnings, functools
print = functools.partial(print, flush=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import Counter, defaultdict
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_config import config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    causal_ww = pd.read_csv(config.results_dir / 'causal_bqtl_728.csv')
    causal_ds = pd.read_csv(config.results_dir / 'causal_bqtl_ds.csv')

    expr = pd.read_csv(config.expression, sep='\t')
    if 'log2fc_DS_vs_WW' in expr.columns:
        expr = expr.rename(columns={'log2fc_DS_vs_WW': 'log2fc'})

    gz_path = Path('/Users/michael/perelyn/research/cell_biology/shared_data/raw/'
                   'Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1_entap_results.tsv.gz')
    ann = pd.read_csv(gz_path, sep='\t', compression='gzip', low_memory=False)
    ann['gene_id'] = ann['Query Sequence'].str.replace(r'_T\d+$', '', regex=True)

    ww_genes = set(causal_ww['gene_id'].unique())
    ds_genes = set(causal_ds['gene_id'].unique())

    return {
        'ww_only': ww_genes - ds_genes,
        'ds_only': ds_genes - ww_genes,
        'shared': ww_genes & ds_genes,
        'all_ww': ww_genes,
        'all_ds': ds_genes,
        'ann': ann,
        'expr': expr,
    }


# ---------------------------------------------------------------------------
# GO term parsing and enrichment
# ---------------------------------------------------------------------------

def build_go_map(ann):
    """Build gene -> set of individual GO terms."""
    go_cols = [c for c in ann.columns if 'GO' in c.upper() and 'go_' not in c.lower()]
    gene_go = defaultdict(set)
    go_names = {}  # GO:xxxx -> description (if embedded in annotation)

    for _, row in ann.iterrows():
        gid = row.get('gene_id')
        if not gid:
            continue
        for col in go_cols:
            val = row.get(col)
            if not isinstance(val, str):
                continue
            for term in val.split(','):
                term = term.strip()
                if term.startswith('GO:'):
                    # Some have format "GO:0006468-protein phosphorylation"
                    if '-' in term and term.index('-') > 7:
                        go_id = term[:10]
                        go_desc = term[11:]
                        go_names[go_id] = go_desc
                    else:
                        go_id = term[:10] if len(term) >= 10 else term
                    gene_go[gid].add(go_id)

    return gene_go, go_names


def run_enrichment(test_genes, bg_genes, gene_go, min_genes=3, min_bg=10):
    """Fisher's exact enrichment of GO terms in test vs background."""
    test_set = test_genes & set(gene_go.keys())
    bg_set = bg_genes & set(gene_go.keys())

    # Collect all GO terms in test set
    all_terms = set()
    for g in test_set:
        all_terms.update(gene_go[g])

    results = []
    n_test = len(test_set)
    n_bg = len(bg_set)

    for term in all_terms:
        k = sum(1 for g in test_set if term in gene_go[g])
        if k < min_genes:
            continue
        K = sum(1 for g in bg_set if term in gene_go[g])
        if K < min_bg:
            continue

        # Fisher's exact
        table = [[k, n_test - k], [K - k, n_bg - n_test - (K - k)]]
        if any(v < 0 for row in table for v in row):
            continue
        odds, p = stats.fisher_exact(table, alternative='greater')
        fe = (k / n_test) / (K / n_bg) if K > 0 and n_bg > 0 else 0

        results.append({
            'go_term': term, 'k': k, 'K': K, 'n': n_test, 'N': n_bg,
            'odds_ratio': odds, 'p_value': p, 'fold_enrichment': fe,
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    _, fdr, _, _ = multipletests(df['p_value'], method='fdr_bh')
    df['fdr'] = fdr
    return df.sort_values('p_value')


def comparative_enrichment(ww_genes, ds_genes, gene_go, min_genes=3):
    """Direct comparison: which GO terms differ between WW and DS gene sets."""
    ww_set = ww_genes & set(gene_go.keys())
    ds_set = ds_genes & set(gene_go.keys())

    all_terms = set()
    for g in ww_set | ds_set:
        all_terms.update(gene_go[g])

    results = []
    n_ww = len(ww_set)
    n_ds = len(ds_set)

    for term in all_terms:
        ww_k = sum(1 for g in ww_set if term in gene_go[g])
        ds_k = sum(1 for g in ds_set if term in gene_go[g])

        if ww_k + ds_k < min_genes:
            continue

        # Fisher's exact: is term more in DS than WW?
        table = [[ds_k, n_ds - ds_k], [ww_k, n_ww - ww_k]]
        if any(v < 0 for row in table for v in row):
            continue
        odds, p = stats.fisher_exact(table, alternative='two-sided')

        ww_frac = ww_k / n_ww if n_ww > 0 else 0
        ds_frac = ds_k / n_ds if n_ds > 0 else 0
        log2fc = np.log2((ds_frac + 0.001) / (ww_frac + 0.001))

        results.append({
            'go_term': term, 'ww_k': ww_k, 'ds_k': ds_k,
            'ww_frac': ww_frac, 'ds_frac': ds_frac,
            'log2fc_ds_vs_ww': log2fc,
            'odds_ratio': odds, 'p_value': p,
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    _, fdr, _, _ = multipletests(df['p_value'], method='fdr_bh')
    df['fdr'] = fdr
    return df.sort_values('p_value')


# ---------------------------------------------------------------------------
# Detailed functional categories
# ---------------------------------------------------------------------------

def detailed_categories(gene_set, ann, expr, label):
    """Fine-grained functional categorization from descriptions."""
    sub = ann[ann['gene_id'].isin(gene_set)].copy()

    # Combine descriptions
    gene_descs = {}
    for _, row in sub.iterrows():
        gid = row['gene_id']
        desc = str(row.get('Description', '')).lower()
        eggnog = str(row.get('EggNOG Description', '')).lower()
        combined = f"{desc} {eggnog}"
        if gid not in gene_descs or len(combined) > len(gene_descs[gid]):
            gene_descs[gid] = combined

    categories = {
        # Metabolism subtypes
        'amino_acid_metabolism': ['amino acid biosynthesis', 'aminotransferase', 'glutamine synthetase',
                                   'asparagine', 'tryptophan', 'proline', 'arginine', 'lysine',
                                   'methionine', 'threonine', 'serine hydroxymethyltransferase'],
        'carbohydrate_metabolism': ['glycolysis', 'gluconeogenesis', 'fructose', 'glucose', 'sucrose',
                                     'starch', 'trehalose', 'hexokinase', 'pyruvate', 'aldolase',
                                     'glyceraldehyde', 'enolase'],
        'lipid_metabolism': ['fatty acid', 'lipase', 'acyltransferase', 'phospholipid', 'sphingolipid',
                              'wax', 'cutin', 'sterol', 'lipid biosynthesis'],
        'nucleotide_metabolism': ['nucleotide', 'purine', 'pyrimidine', 'dna repair', 'replication',
                                   'topoisomerase', 'helicase'],
        'secondary_metabolism': ['phenylpropanoid', 'flavonoid', 'anthocyanin', 'terpenoid', 'carotenoid',
                                  'alkaloid', 'lignin biosynthesis', 'chalcone', 'isoflavone'],
        'energy/redox': ['oxidoreductase', 'cytochrome', 'ferredoxin', 'thioredoxin', 'nadh', 'nadph',
                          'atp synthase', 'electron transport', 'respiratory'],
        # Signaling subtypes
        'kinase': ['protein kinase', 'serine/threonine kinase', 'receptor kinase', 'mapk',
                    'map kinase', 'cdpk', 'snrk', 'phosphorylation'],
        'phosphatase': ['phosphatase', 'dephosphorylation', 'pp2c', 'pp2a'],
        'calcium_signaling': ['calcium', 'calmodulin', 'calcineurin', 'cam-binding', 'cbl', 'cipk'],
        'receptor': ['receptor', 'receptor-like', 'lrr receptor', 'wall-associated kinase'],
        'GTPase/small_G': ['gtpase', 'ras-related', 'rab', 'arf', 'rho', 'rop', 'ran'],
        'hormone_signaling': ['auxin', 'ethylene', 'abscisic', 'aba', 'gibberellin', 'jasmonate',
                                'brassinosteroid', 'cytokinin', 'salicylic'],
        # Transport subtypes
        'water_transport': ['aquaporin', 'pip', 'tip', 'water channel'],
        'ion_transport': ['potassium', 'sodium', 'calcium transport', 'ion channel', 'cation',
                           'anion channel', 'exchanger'],
        'sugar_transport': ['sugar transport', 'sweet', 'hexose transporter', 'sut', 'sucrose transport'],
        'nutrient_transport': ['nitrate transport', 'phosphate transport', 'sulfate transport',
                                'ammonium transport', 'iron transport', 'zinc transport', 'boron'],
        'ABC_transport': ['abc transporter', 'multidrug', 'pleiotropic drug'],
        # Stress/defense
        'heat_stress': ['heat shock', 'hsp', 'hsf', 'chaperone', 'dnaj', 'clp protease'],
        'drought_stress': ['dehydrin', 'lea protein', 'drought', 'desiccation', 'osmotic'],
        'oxidative_stress': ['peroxidase', 'catalase', 'superoxide dismutase', 'glutathione',
                              'ascorbate', 'antioxidant', 'reactive oxygen'],
        'pathogen_defense': ['pathogenesis', 'disease resistance', 'nbs-lrr', 'chitinase',
                              'glucanase', 'defensin', 'pr protein'],
        # Transcription
        'TF_MYB': ['myb', 'r2r3-myb'],
        'TF_WRKY': ['wrky'],
        'TF_NAC': ['nac domain', 'nac transcription'],
        'TF_bZIP': ['bzip', 'basic leucine zipper'],
        'TF_ERF': ['erf', 'ap2/erf', 'ethylene response factor', 'dreb'],
        'TF_bHLH': ['bhlh', 'basic helix-loop-helix'],
        'TF_other': ['homeobox', 'homeodomain', 'zinc finger transcription', 'gata transcription',
                      'tcp', 'grf', 'dof', 'sbp'],
        # Chromatin/epigenetics
        'histone_modification': ['histone', 'acetyltransferase', 'deacetylase', 'methyltransferase',
                                   'demethylase', 'polycomb', 'trithorax'],
        'chromatin_remodeling': ['chromatin remodel', 'swi/snf', 'nucleosome', 'topoisomerase'],
        # Cell structure
        'cell_wall': ['cellulose synthase', 'pectin', 'xylan', 'expansin', 'extensin',
                       'cell wall', 'arabinogalactan', 'mannan'],
        'cytoskeleton': ['actin', 'tubulin', 'kinesin', 'myosin', 'microtubule'],
        # Protein processing
        'ubiquitin_proteasome': ['ubiquitin', 'proteasome', 'e3 ligase', 'f-box', 'ring finger',
                                   'u-box', 'skp', 'cullin'],
        'autophagy': ['autophagy', 'atg'],
        'protein_folding': ['chaperonin', 'hsp70', 'hsp90', 'dnak', 'groel', 'peptidyl-prolyl'],
        # Photosynthesis
        'photosynthesis': ['photosystem', 'rubisco', 'chlorophyll', 'light harvesting',
                            'thylakoid', 'carbon fixation', 'photorespiration'],
        # RNA processing
        'RNA_processing': ['splicing', 'spliceosome', 'rna binding', 'rna helicase', 'ribosome',
                            'rrna', 'trna', 'translation factor', 'elongation factor'],
        # Development
        'flowering': ['flowering', 'photoperiod', 'constans', 'ft protein', 'mads-box'],
        'root_development': ['root', 'lateral root', 'root hair'],
        'seed_development': ['seed', 'embryo', 'endosperm', 'grain', 'starch biosynthesis'],
    }

    # Count genes per category
    gene_cats = {}
    for gid, desc in gene_descs.items():
        gene_cats[gid] = set()
        for cat, keywords in categories.items():
            if any(kw in desc for kw in keywords):
                gene_cats[gid].add(cat)

    # Aggregate
    cat_counts = Counter()
    for gid, cats in gene_cats.items():
        for cat in cats:
            cat_counts[cat] += 1

    n_annotated = len(gene_descs)
    return cat_counts, n_annotated, categories


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  COMPARATIVE FUNCTIONAL ENRICHMENT: WW vs DS")
    print("=" * 70)

    data = load_data()
    ann = data['ann']
    expr = data['expr']

    print(f"  WW-only genes: {len(data['ww_only'])}")
    print(f"  DS-only genes: {len(data['ds_only'])}")
    print(f"  Shared genes: {len(data['shared'])}")

    # Build GO map
    gene_go, go_names = build_go_map(ann)
    all_annotated = set(gene_go.keys())
    print(f"  Genes with GO: {len(all_annotated)}")
    print(f"  Unique GO terms: {len(set(t for ts in gene_go.values() for t in ts))}")

    # ── 1. Individual enrichment ──
    print("\n" + "=" * 70)
    print("  1. GO ENRICHMENT PER CONDITION (vs all annotated genes)")
    print("=" * 70)

    for label, genes in [('WW-only', data['ww_only']), ('DS-only', data['ds_only'])]:
        enrich = run_enrichment(genes, all_annotated, gene_go)
        sig = enrich[enrich['fdr'] < 0.05] if len(enrich) > 0 else pd.DataFrame()
        print(f"\n  {label}: {len(sig)} enriched GO terms (FDR<0.05)")
        for _, row in sig.head(20).iterrows():
            name = go_names.get(row['go_term'], '')
            print(f"    {row['go_term']} {name:<40s} FE={row['fold_enrichment']:.1f}x "
                  f"({row['k']}/{row['n']}) FDR={row['fdr']:.3f}")

    # ── 2. Direct WW vs DS comparison ──
    print("\n" + "=" * 70)
    print("  2. DIRECT COMPARISON: GO TERMS DIFFERING BETWEEN WW AND DS")
    print("=" * 70)

    comp = comparative_enrichment(data['ww_only'], data['ds_only'], gene_go)
    if len(comp) > 0:
        sig_comp = comp[comp['fdr'] < 0.1]
        print(f"\n  Significantly different GO terms (FDR<0.1): {len(sig_comp)}")

        ds_enriched = sig_comp[sig_comp['log2fc_ds_vs_ww'] > 0].sort_values('fdr')
        ww_enriched = sig_comp[sig_comp['log2fc_ds_vs_ww'] < 0].sort_values('fdr')

        print(f"\n  DS-enriched ({len(ds_enriched)}):")
        for _, row in ds_enriched.head(15).iterrows():
            name = go_names.get(row['go_term'], '')
            print(f"    {row['go_term']} {name:<40s} DS={row['ds_k']} WW={row['ww_k']} "
                  f"log2FC={row['log2fc_ds_vs_ww']:+.2f} FDR={row['fdr']:.3f}")

        print(f"\n  WW-enriched ({len(ww_enriched)}):")
        for _, row in ww_enriched.head(15).iterrows():
            name = go_names.get(row['go_term'], '')
            print(f"    {row['go_term']} {name:<40s} DS={row['ds_k']} WW={row['ww_k']} "
                  f"log2FC={row['log2fc_ds_vs_ww']:+.2f} FDR={row['fdr']:.3f}")

    # ── 3. Detailed functional categories ──
    print("\n" + "=" * 70)
    print("  3. DETAILED FUNCTIONAL CATEGORIES")
    print("=" * 70)

    ww_cats, ww_n, cat_def = detailed_categories(data['ww_only'], ann, expr, 'WW-only')
    ds_cats, ds_n, _ = detailed_categories(data['ds_only'], ann, expr, 'DS-only')

    # Group categories
    groups = {
        'METABOLISM': ['amino_acid_metabolism', 'carbohydrate_metabolism', 'lipid_metabolism',
                        'nucleotide_metabolism', 'secondary_metabolism', 'energy/redox'],
        'SIGNALING': ['kinase', 'phosphatase', 'calcium_signaling', 'receptor',
                       'GTPase/small_G', 'hormone_signaling'],
        'TRANSPORT': ['water_transport', 'ion_transport', 'sugar_transport',
                       'nutrient_transport', 'ABC_transport'],
        'STRESS/DEFENSE': ['heat_stress', 'drought_stress', 'oxidative_stress', 'pathogen_defense'],
        'TRANSCRIPTION': ['TF_MYB', 'TF_WRKY', 'TF_NAC', 'TF_bZIP', 'TF_ERF', 'TF_bHLH', 'TF_other'],
        'CHROMATIN': ['histone_modification', 'chromatin_remodeling'],
        'CELL_STRUCTURE': ['cell_wall', 'cytoskeleton'],
        'PROTEIN_PROCESSING': ['ubiquitin_proteasome', 'autophagy', 'protein_folding'],
        'PHOTOSYNTHESIS': ['photosynthesis'],
        'RNA/TRANSLATION': ['RNA_processing'],
        'DEVELOPMENT': ['flowering', 'root_development', 'seed_development'],
    }

    print(f"\n  {'Category':<30s} {'WW-only':>8} {'%':>6} {'DS-only':>8} {'%':>6} {'DS/WW':>6} {'p':>10}")
    print("  " + "-" * 78)

    all_rows = []
    for group_name, subcats in groups.items():
        # Group total
        ww_total = sum(ww_cats.get(c, 0) for c in subcats)
        ds_total = sum(ds_cats.get(c, 0) for c in subcats)
        ww_pct = 100 * ww_total / max(ww_n, 1)
        ds_pct = 100 * ds_total / max(ds_n, 1)
        ratio = ds_pct / ww_pct if ww_pct > 0 else float('inf')

        # Fisher's test for group
        table = [[ds_total, ds_n - ds_total], [ww_total, ww_n - ww_total]]
        _, group_p = stats.fisher_exact(table, alternative='two-sided')

        marker = '***' if group_p < 0.001 else '**' if group_p < 0.01 else '*' if group_p < 0.05 else ''
        print(f"  {group_name:<30s} {ww_total:>8} {ww_pct:>5.1f}% {ds_total:>8} {ds_pct:>5.1f}% {ratio:>5.2f} {group_p:>9.2e} {marker}")

        all_rows.append({
            'group': group_name, 'category': group_name,
            'ww_count': ww_total, 'ds_count': ds_total,
            'ww_pct': ww_pct, 'ds_pct': ds_pct,
            'ratio': ratio, 'p_value': group_p, 'is_group': True,
        })

        # Subcategories
        for cat in subcats:
            ww_c = ww_cats.get(cat, 0)
            ds_c = ds_cats.get(cat, 0)
            if ww_c + ds_c == 0:
                continue
            ww_p = 100 * ww_c / max(ww_n, 1)
            ds_p = 100 * ds_c / max(ds_n, 1)
            r = ds_p / ww_p if ww_p > 0 else float('inf')

            table = [[ds_c, ds_n - ds_c], [ww_c, ww_n - ww_c]]
            _, sub_p = stats.fisher_exact(table, alternative='two-sided')
            sub_marker = '*' if sub_p < 0.05 else ''

            print(f"    {cat:<28s} {ww_c:>8} {ww_p:>5.1f}% {ds_c:>8} {ds_p:>5.1f}% {r:>5.2f} {sub_p:>9.2e} {sub_marker}")

            all_rows.append({
                'group': group_name, 'category': cat,
                'ww_count': ww_c, 'ds_count': ds_c,
                'ww_pct': ww_p, 'ds_pct': ds_p,
                'ratio': r, 'p_value': sub_p, 'is_group': False,
            })

    # ── 4. Expression context for each category ──
    print("\n" + "=" * 70)
    print("  4. DROUGHT EXPRESSION BY FUNCTIONAL CATEGORY")
    print("=" * 70)

    # For DS-only genes: what's the drought expression profile per category?
    ds_sub = ann[ann['gene_id'].isin(data['ds_only'])].copy()
    ds_descs = {}
    for _, row in ds_sub.iterrows():
        gid = row['gene_id']
        desc = str(row.get('Description', '')).lower()
        eggnog = str(row.get('EggNOG Description', '')).lower()
        ds_descs[gid] = f"{desc} {eggnog}"

    print(f"\n  {'Category':<30s} {'n_DS':>5} {'med_fc':>7} {'up':>4} {'dn':>4} {'stable':>6}")
    print("  " + "-" * 60)

    for group_name, subcats in groups.items():
        # Get genes in this group
        group_genes = set()
        for cat in subcats:
            for gid, desc in ds_descs.items():
                if any(kw in desc for kw in cat_def.get(cat, [])):
                    group_genes.add(gid)

        if not group_genes:
            continue

        sub_expr = expr[expr['gene_id'].isin(group_genes)]
        if len(sub_expr) == 0:
            continue

        fc = sub_expr['log2fc']
        n_up = (fc > 1).sum()
        n_dn = (fc < -1).sum()
        n_stable = len(fc) - n_up - n_dn

        print(f"  {group_name:<30s} {len(sub_expr):>5} {fc.median():>+6.2f} {n_up:>4} {n_dn:>4} {n_stable:>6}")

    # ── 5. Figure ──
    print("\n── Generating figure ──")

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.35)

    # Panel A: Grouped bar chart of major categories
    ax1 = fig.add_subplot(gs[0, :])
    group_df = pd.DataFrame([r for r in all_rows if r['is_group']])
    group_df = group_df.sort_values('ratio', ascending=True)

    y = np.arange(len(group_df))
    h = 0.35
    ax1.barh(y - h/2, group_df['ww_pct'], h, color='#4CAF50', label='WW-only', alpha=0.85)
    ax1.barh(y + h/2, group_df['ds_pct'], h, color='#FF9800', label='DS-only', alpha=0.85)

    # Mark significant differences
    for i, (_, row) in enumerate(group_df.iterrows()):
        if row['p_value'] < 0.05:
            ax1.text(max(row['ww_pct'], row['ds_pct']) + 0.3, i, '*',
                     fontsize=14, fontweight='bold', va='center', color='red')

    ax1.set_yticks(y)
    ax1.set_yticklabels(group_df['group'], fontsize=10)
    ax1.set_xlabel('% of annotated genes', fontsize=11)
    ax1.legend(fontsize=11, loc='lower right')
    ax1.set_title('A. Functional category comparison: WW-only vs DS-only causal bQTL targets',
                  fontsize=13, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: DS/WW ratio for subcategories (diverging bar)
    ax2 = fig.add_subplot(gs[1, 0])
    sub_df = pd.DataFrame([r for r in all_rows if not r['is_group']
                           and r['ww_count'] + r['ds_count'] >= 5])
    sub_df['log2_ratio'] = np.log2(sub_df['ratio'].clip(0.1, 10))
    sub_df = sub_df.sort_values('log2_ratio')

    colors = ['#FF9800' if lr > 0 else '#4CAF50' for lr in sub_df['log2_ratio']]
    sig_colors = ['#E53935' if p < 0.05 else c for p, c in zip(sub_df['p_value'], colors)]

    ax2.barh(range(len(sub_df)), sub_df['log2_ratio'], color=sig_colors, height=0.7)
    ax2.axvline(0, c='k', lw=0.5)
    ax2.set_yticks(range(len(sub_df)))
    ax2.set_yticklabels(sub_df['category'], fontsize=7)
    ax2.set_xlabel('log2(DS/WW ratio)', fontsize=10)
    ax2.set_title('B. Subcategory enrichment\n(red = significant p<0.05)',
                  fontsize=12, fontweight='bold')

    # Panel C: Drought expression by category (DS-only genes)
    ax3 = fig.add_subplot(gs[1, 1])
    expr_data = []
    for group_name, subcats in groups.items():
        group_genes = set()
        for cat in subcats:
            for gid, desc in ds_descs.items():
                if any(kw in desc for kw in cat_def.get(cat, [])):
                    group_genes.add(gid)
        sub_e = expr[expr['gene_id'].isin(group_genes)]['log2fc']
        if len(sub_e) >= 5:
            expr_data.append({'group': group_name, 'values': sub_e.values,
                              'median': sub_e.median(), 'n': len(sub_e)})

    if expr_data:
        expr_data.sort(key=lambda x: x['median'])
        positions = range(len(expr_data))
        bp = ax3.boxplot([d['values'] for d in expr_data], positions=list(positions),
                         vert=False, patch_artist=True, widths=0.6,
                         showfliers=False, medianprops={'color': 'black', 'linewidth': 2})
        for patch in bp['boxes']:
            patch.set_facecolor('#FF9800')
            patch.set_alpha(0.6)

        ax3.axvline(0, c='k', lw=0.5, ls='--')
        ax3.axvline(1, c='grey', lw=0.5, ls=':')
        ax3.axvline(-1, c='grey', lw=0.5, ls=':')
        ax3.set_yticks(list(positions))
        ax3.set_yticklabels([f"{d['group']} (n={d['n']})" for d in expr_data], fontsize=8)
        ax3.set_xlabel('Drought log2FC', fontsize=10)
        ax3.set_title('C. Drought expression of DS-only targets\nby functional category',
                      fontsize=12, fontweight='bold')

    fig_path = config.figures_dir / 'comparative_enrichment_ww_vs_ds.pdf'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  Figure saved: {fig_path}")

    # Save results
    cat_df = pd.DataFrame(all_rows)
    cat_path = config.results_dir / 'comparative_enrichment_categories.csv'
    cat_df.to_csv(cat_path, index=False)
    print(f"  Categories saved: {cat_path}")

    if len(comp) > 0:
        comp_path = config.results_dir / 'comparative_go_ww_vs_ds.csv'
        comp.to_csv(comp_path, index=False)
        print(f"  GO comparison saved: {comp_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
