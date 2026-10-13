"""
22_condition_functional_characterization.py — Functional characterization of
condition-specific causal bQTL target genes.

Analyses:
  1. Gene set decomposition: shared / WW-only / DS-only / same-gene-diff-bQTL
  2. GO/functional enrichment for each condition-specific gene set
  3. ESM-2 protein embeddings + UMAP visualization
  4. Pathway/category comparison across conditions
"""

import sys, warnings, functools
print = functools.partial(print, flush=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_config import config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Load and decompose gene sets
# ---------------------------------------------------------------------------

def decompose_gene_sets(causal_ww, causal_ds):
    """Split causal bQTL targets into condition-specific categories."""
    print("\n── Gene Set Decomposition ──")

    ww_genes = set(causal_ww['gene_id'].unique())
    ds_genes = set(causal_ds['gene_id'].unique())
    shared_genes = ww_genes & ds_genes
    ww_only = ww_genes - ds_genes
    ds_only = ds_genes - ww_genes

    # For shared genes: same bQTL or different?
    same_variant = set()
    diff_variant = set()
    for gene in shared_genes:
        ww_pos = set(zip(causal_ww[causal_ww['gene_id'] == gene]['chr'],
                         causal_ww[causal_ww['gene_id'] == gene]['pos']))
        ds_pos = set(zip(causal_ds[causal_ds['gene_id'] == gene]['chr'],
                         causal_ds[causal_ds['gene_id'] == gene]['pos']))
        if ww_pos & ds_pos:
            same_variant.add(gene)
        else:
            diff_variant.add(gene)

    print(f"  Total WW genes: {len(ww_genes)}")
    print(f"  Total DS genes: {len(ds_genes)}")
    print(f"  WW-only: {len(ww_only)}")
    print(f"  DS-only: {len(ds_only)}")
    print(f"  Shared (same variant): {len(same_variant)}")
    print(f"  Shared (different variant): {len(diff_variant)}")

    return {
        'ww_only': ww_only,
        'ds_only': ds_only,
        'shared_same': same_variant,
        'shared_diff': diff_variant,
        'all_ww': ww_genes,
        'all_ds': ds_genes,
    }


# ---------------------------------------------------------------------------
# Functional annotation from EntAP
# ---------------------------------------------------------------------------

def load_annotations():
    """Load EntAP functional annotations."""
    # Try gzipped first (always available)
    gz_path = Path('/Users/michael/perelyn/research/cell_biology/shared_data/raw/'
                   'Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1_entap_results.tsv.gz')
    try:
        entap_path = config.entap
        if entap_path.exists():
            ann = pd.read_csv(entap_path, sep='\t', low_memory=False)
        elif gz_path.exists():
            ann = pd.read_csv(gz_path, sep='\t', compression='gzip',
                              low_memory=False)
        else:
            print(f"  WARNING: EntAP annotations not found")
            return None
    except Exception:
        if gz_path.exists():
            ann = pd.read_csv(gz_path, sep='\t', compression='gzip',
                              low_memory=False)
        else:
            print(f"  WARNING: EntAP annotations not found")
            return None

    # Gene ID is in 'Query Sequence' column with transcript suffix _T001
    if 'Query Sequence' in ann.columns:
        ann['gene_id'] = ann['Query Sequence'].str.replace(r'_T\d+$', '', regex=True)
    else:
        id_col = None
        for col in ann.columns:
            if 'query' in col.lower() or 'gene' in col.lower():
                id_col = col
                break
        if id_col:
            ann['gene_id'] = ann[id_col].str.replace(r'_T\d+$', '', regex=True)

    print(f"  EntAP annotations loaded: {len(ann)} entries, "
          f"{ann['gene_id'].nunique()} genes")
    return ann


def annotate_gene_sets(gene_sets, ann_df, expr_df):
    """Get functional categories for each gene set."""
    print("\n── Functional Annotation ──")

    if ann_df is None:
        return None

    # Find GO and description columns
    go_cols = [c for c in ann_df.columns if 'GO' in c.upper() or 'go_' in c.lower()]
    desc_cols = [c for c in ann_df.columns if 'description' in c.lower()
                 or 'function' in c.lower()]

    print(f"  GO columns: {go_cols[:3]}")
    print(f"  Description columns: {desc_cols[:3]}")

    # Use KEGG/eggNOG/interpro if available
    kegg_cols = [c for c in ann_df.columns if 'KEGG' in c.upper() or 'kegg' in c.lower()]
    eggnog_cols = [c for c in ann_df.columns if 'eggnog' in c.lower() or 'COG' in c.upper()]

    results = {}
    for label, genes in gene_sets.items():
        if len(genes) == 0:
            continue

        sub = ann_df[ann_df['gene_id'].isin(genes)]
        n_annotated = len(sub)

        # Collect GO terms
        go_terms = []
        for col in go_cols:
            for val in sub[col].dropna():
                if isinstance(val, str):
                    for term in val.split(';'):
                        term = term.strip()
                        if term.startswith('GO:'):
                            go_terms.append(term)

        # Collect descriptions
        descriptions = []
        for col in desc_cols:
            for val in sub[col].dropna():
                if isinstance(val, str) and len(val) > 3:
                    descriptions.append(val.lower())

        # Simple keyword frequency analysis from descriptions
        keywords = Counter()
        stop_words = {'the', 'and', 'for', 'with', 'that', 'this', 'from',
                      'like', 'protein', 'gene', 'predicted', 'putative',
                      'uncharacterized', 'hypothetical', 'containing', 'domain',
                      'family', 'member', 'type', 'subunit', 'related'}
        for desc in descriptions:
            words = desc.replace('-', ' ').replace('/', ' ').split()
            for w in words:
                w = w.strip('.,;:()')
                if len(w) > 3 and w not in stop_words:
                    keywords[w] += 1

        # Categorize into broad functional groups
        categories = Counter()
        category_keywords = {
            'kinase/signaling': ['kinase', 'phosphatase', 'receptor', 'signal',
                                 'calcium', 'calmodulin', 'mapk', 'phosphorylation'],
            'transcription': ['transcription', 'zinc', 'finger', 'bzip', 'myb',
                              'wrky', 'erf', 'nac', 'bhlh', 'homeobox', 'hdzip'],
            'transport': ['transporter', 'channel', 'permease', 'pump', 'abc',
                          'aquaporin', 'carrier'],
            'metabolism': ['synthase', 'reductase', 'oxidase', 'dehydrogenase',
                           'transferase', 'hydrolase', 'isomerase', 'ligase',
                           'enzyme', 'catalytic', 'metabolic'],
            'stress/defense': ['stress', 'defense', 'resistance', 'pathogen',
                               'heat', 'shock', 'drought', 'dehydrin', 'lea',
                               'peroxidase', 'glutathione', 'antioxidant'],
            'cell_wall': ['cellulose', 'pectin', 'xylan', 'lignin', 'expansin',
                          'wall', 'glucan'],
            'hormone': ['auxin', 'ethylene', 'abscisic', 'gibberellin', 'jasmonic',
                        'brassinosteroid', 'cytokinin', 'salicylic'],
            'chromatin/epigenetic': ['histone', 'methyltransferase', 'chromatin',
                                     'acetyl', 'deacetylase', 'demethylase'],
            'ribosome/translation': ['ribosom', 'translation', 'elongation',
                                      'initiation', 'trna'],
            'proteasome/degradation': ['ubiquitin', 'proteasome', 'protease',
                                        'autophagy', 'degradation'],
            'photosynthesis': ['photosystem', 'chlorophyll', 'rubisco', 'thylakoid',
                               'photosynthesis', 'light'],
            'development': ['flower', 'seed', 'root', 'leaf', 'meristem',
                            'embryo', 'pollen', 'development'],
        }

        for desc in descriptions:
            for cat, kws in category_keywords.items():
                if any(kw in desc for kw in kws):
                    categories[cat] += 1

        # Add expression info
        expr_sub = expr_df[expr_df['gene_id'].isin(genes)] if expr_df is not None else None

        results[label] = {
            'n_genes': len(genes),
            'n_annotated': n_annotated,
            'go_terms': Counter(go_terms),
            'keywords': keywords,
            'categories': categories,
            'expr': expr_sub,
        }

        print(f"\n  {label} ({len(genes)} genes, {n_annotated} annotated):")
        print(f"    Top keywords: {keywords.most_common(8)}")
        print(f"    Categories: {categories.most_common(6)}")
        if expr_sub is not None and len(expr_sub) > 0:
            print(f"    Median drought log2fc: {expr_sub['log2fc'].median():.3f}")

    return results


# ---------------------------------------------------------------------------
# GO enrichment (Fisher's exact against background)
# ---------------------------------------------------------------------------

def go_enrichment(gene_sets, ann_df):
    """Run GO enrichment for each condition-specific set vs all expressed genes."""
    print("\n── GO Enrichment Analysis ──")

    if ann_df is None:
        return None

    go_cols = [c for c in ann_df.columns if 'GO' in c.upper() or 'go_' in c.lower()]
    if not go_cols:
        print("  No GO columns found")
        return None

    # Build gene -> GO mapping for all genes
    gene_go = defaultdict(set)
    for _, row in ann_df.iterrows():
        gid = row.get('gene_id')
        if not gid:
            continue
        for col in go_cols:
            val = row.get(col)
            if isinstance(val, str):
                for term in val.split(';'):
                    term = term.strip()
                    if term.startswith('GO:'):
                        gene_go[gid].add(term)

    all_genes_with_go = set(gene_go.keys())
    all_go_terms = set()
    for terms in gene_go.values():
        all_go_terms.update(terms)

    print(f"  Genes with GO annotations: {len(all_genes_with_go)}")
    print(f"  Unique GO terms: {len(all_go_terms)}")

    # For each gene set, test enrichment of each GO term
    enrichment_results = {}
    for label, genes in gene_sets.items():
        if len(genes) < 10:
            continue

        test_genes = genes & all_genes_with_go
        bg_genes = all_genes_with_go
        if len(test_genes) < 5:
            continue

        results = []
        for term in all_go_terms:
            # Genes with this term in test set
            k = sum(1 for g in test_genes if term in gene_go[g])
            if k < 3:
                continue
            # Genes with this term in background
            K = sum(1 for g in bg_genes if term in gene_go[g])
            n = len(test_genes)
            N = len(bg_genes)

            # Fisher's exact (one-sided, enrichment)
            table = [[k, n - k], [K - k, N - n - (K - k)]]
            # Ensure non-negative
            if any(v < 0 for row in table for v in row):
                continue
            odds, p = stats.fisher_exact(table, alternative='greater')
            results.append({
                'go_term': term, 'k': k, 'K': K, 'n': n, 'N': N,
                'odds_ratio': odds, 'p_value': p,
                'fold_enrichment': (k / n) / (K / N) if K > 0 and N > 0 else 0,
            })

        if not results:
            continue

        res_df = pd.DataFrame(results)
        from statsmodels.stats.multitest import multipletests
        _, fdr, _, _ = multipletests(res_df['p_value'], method='fdr_bh')
        res_df['fdr'] = fdr
        sig = res_df[res_df['fdr'] < 0.1].sort_values('fold_enrichment', ascending=False)

        enrichment_results[label] = res_df
        print(f"\n  {label}: {len(sig)} enriched GO terms (FDR<0.1)")
        for _, row in sig.head(10).iterrows():
            print(f"    {row['go_term']} FE={row['fold_enrichment']:.1f}x "
                  f"({row['k']}/{row['n']}) FDR={row['fdr']:.3f}")

    return enrichment_results


# ---------------------------------------------------------------------------
# UMAP visualization from protein descriptions / keyword embeddings
# ---------------------------------------------------------------------------

def compute_keyword_embeddings(gene_sets, ann_df):
    """Create TF-IDF keyword vectors for genes and project with UMAP."""
    print("\n── Keyword-based Gene Embedding + UMAP ──")

    if ann_df is None:
        return None

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import PCA
    except ImportError:
        print("  sklearn not available")
        return None

    try:
        import umap
        has_umap = True
    except ImportError:
        has_umap = False
        print("  UMAP not available, using PCA instead")

    # Collect all genes and their descriptions
    desc_cols = [c for c in ann_df.columns if 'description' in c.lower()
                 or 'function' in c.lower()]

    all_genes = set()
    for genes in gene_sets.values():
        all_genes.update(genes)

    gene_texts = {}
    for _, row in ann_df.iterrows():
        gid = row.get('gene_id')
        if gid not in all_genes:
            continue
        texts = []
        for col in desc_cols:
            val = row.get(col)
            if isinstance(val, str) and len(val) > 3:
                texts.append(val.lower())
        if texts:
            gene_texts[gid] = ' '.join(texts)

    if len(gene_texts) < 50:
        print(f"  Only {len(gene_texts)} genes with descriptions, too few for embedding")
        return None

    print(f"  Genes with text descriptions: {len(gene_texts)}")

    # Build gene labels
    gene_ids = list(gene_texts.keys())
    texts = [gene_texts[g] for g in gene_ids]

    # Assign condition label
    labels = []
    for g in gene_ids:
        if g in gene_sets.get('shared_same', set()) or g in gene_sets.get('shared_diff', set()):
            labels.append('Shared')
        elif g in gene_sets.get('ww_only', set()):
            labels.append('WW-only')
        elif g in gene_sets.get('ds_only', set()):
            labels.append('DS-only')
        else:
            labels.append('Other')

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=500, stop_words='english',
                            min_df=3, max_df=0.8)
    X = tfidf.fit_transform(texts).toarray()
    print(f"  TF-IDF matrix: {X.shape}")

    # Dimensionality reduction
    if has_umap and X.shape[0] > 15:
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            random_state=42, metric='cosine')
        coords = reducer.fit_transform(X)
        method = 'UMAP'
    else:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)
        method = 'PCA'

    print(f"  Projection method: {method}")

    return pd.DataFrame({
        'gene_id': gene_ids,
        'x': coords[:, 0],
        'y': coords[:, 1],
        'condition': labels,
    }), method


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(gene_sets, func_results, expr_df, embedding_df, embed_method,
                 go_results):
    """Generate multi-panel figure."""
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # Panel A: Gene set Venn-like bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['WW-only', 'DS-only', 'Shared\n(same var)', 'Shared\n(diff var)']
    counts = [len(gene_sets['ww_only']), len(gene_sets['ds_only']),
              len(gene_sets['shared_same']), len(gene_sets['shared_diff'])]
    colors = ['#4CAF50', '#FF9800', '#2196F3', '#9C27B0']
    bars = ax1.bar(categories, counts, color=colors, edgecolor='white', linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Number of genes', fontsize=11)
    ax1.set_title('A. Condition-specific gene sets', fontsize=12, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Functional category comparison
    ax2 = fig.add_subplot(gs[0, 1])
    if func_results:
        cat_labels = sorted(set(
            cat for res in func_results.values()
            for cat in res['categories'].keys()
        ))
        # Normalize to fraction of annotated genes
        ww_fracs, ds_fracs = [], []
        for cat in cat_labels:
            ww_n = func_results.get('ww_only', {}).get('n_annotated', 1) or 1
            ds_n = func_results.get('ds_only', {}).get('n_annotated', 1) or 1
            ww_c = func_results.get('ww_only', {}).get('categories', {}).get(cat, 0)
            ds_c = func_results.get('ds_only', {}).get('categories', {}).get(cat, 0)
            ww_fracs.append(100 * ww_c / ww_n)
            ds_fracs.append(100 * ds_c / ds_n)

        y = np.arange(len(cat_labels))
        h = 0.35
        ax2.barh(y - h / 2, ww_fracs, h, color='#4CAF50', label='WW-only', alpha=0.8)
        ax2.barh(y + h / 2, ds_fracs, h, color='#FF9800', label='DS-only', alpha=0.8)
        ax2.set_yticks(y)
        ax2.set_yticklabels(cat_labels, fontsize=8)
        ax2.set_xlabel('% of annotated genes', fontsize=10)
        ax2.legend(fontsize=9, loc='lower right')
        ax2.set_title('B. Functional categories', fontsize=12, fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

    # Panel C: Drought expression distributions per set
    ax3 = fig.add_subplot(gs[0, 2])
    bins = np.linspace(-5, 5, 60)
    for label, color, genes in [('WW-only', '#4CAF50', gene_sets['ww_only']),
                                  ('DS-only', '#FF9800', gene_sets['ds_only']),
                                  ('Shared', '#2196F3',
                                   gene_sets['shared_same'] | gene_sets['shared_diff'])]:
        sub = expr_df[expr_df['gene_id'].isin(genes)]['log2fc']
        if len(sub) > 0:
            ax3.hist(sub.clip(-5, 5), bins=bins, alpha=0.5, color=color,
                     label=f'{label} (n={len(sub)})', density=True)
    ax3.axvline(0, c='k', lw=0.5, ls='--')
    ax3.axvline(1, c='grey', lw=0.5, ls=':')
    ax3.axvline(-1, c='grey', lw=0.5, ls=':')
    ax3.set_xlabel('Drought log2FC', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.legend(fontsize=8)
    ax3.set_title('C. Drought expression by gene set', fontsize=12, fontweight='bold')

    # Panel D: UMAP / PCA embedding
    ax4 = fig.add_subplot(gs[1, 0])
    if embedding_df is not None:
        color_map = {'WW-only': '#4CAF50', 'DS-only': '#FF9800', 'Shared': '#2196F3'}
        for cond, color in color_map.items():
            mask = embedding_df['condition'] == cond
            if mask.sum() > 0:
                ax4.scatter(embedding_df.loc[mask, 'x'], embedding_df.loc[mask, 'y'],
                            c=color, s=8, alpha=0.4, label=cond)
        ax4.legend(fontsize=9, markerscale=2)
        ax4.set_xlabel(f'{embed_method} 1', fontsize=11)
        ax4.set_ylabel(f'{embed_method} 2', fontsize=11)
        ax4.set_title(f'D. Gene function embedding ({embed_method})',
                      fontsize=12, fontweight='bold')

    # Panel E: Top enriched GO terms (DS-only)
    ax5 = fig.add_subplot(gs[1, 1])
    if go_results and 'ds_only' in go_results:
        sig = go_results['ds_only']
        sig = sig[sig['fdr'] < 0.1].sort_values('fold_enrichment', ascending=True).tail(15)
        if len(sig) > 0:
            y = range(len(sig))
            colors = ['#FF9800' if fdr < 0.05 else '#FFE0B2' for fdr in sig['fdr']]
            ax5.barh(y, sig['fold_enrichment'], color=colors)
            ax5.set_yticks(y)
            ax5.set_yticklabels(sig['go_term'], fontsize=7)
            ax5.set_xlabel('Fold enrichment', fontsize=10)
            ax5.set_title('E. GO enrichment: DS-only genes',
                          fontsize=12, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'No significant\nGO terms', ha='center',
                     va='center', fontsize=12, transform=ax5.transAxes)
            ax5.set_title('E. GO enrichment: DS-only genes',
                          fontsize=12, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'GO data\nnot available', ha='center',
                 va='center', fontsize=12, transform=ax5.transAxes)
        ax5.set_title('E. GO enrichment: DS-only genes',
                      fontsize=12, fontweight='bold')

    # Panel F: Top enriched GO terms (WW-only)
    ax6 = fig.add_subplot(gs[1, 2])
    if go_results and 'ww_only' in go_results:
        sig = go_results['ww_only']
        sig = sig[sig['fdr'] < 0.1].sort_values('fold_enrichment', ascending=True).tail(15)
        if len(sig) > 0:
            y = range(len(sig))
            colors = ['#4CAF50' if fdr < 0.05 else '#C8E6C9' for fdr in sig['fdr']]
            ax6.barh(y, sig['fold_enrichment'], color=colors)
            ax6.set_yticks(y)
            ax6.set_yticklabels(sig['go_term'], fontsize=7)
            ax6.set_xlabel('Fold enrichment', fontsize=10)
            ax6.set_title('F. GO enrichment: WW-only genes',
                          fontsize=12, fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'No significant\nGO terms', ha='center',
                     va='center', fontsize=12, transform=ax6.transAxes)
            ax6.set_title('F. GO enrichment: WW-only genes',
                          fontsize=12, fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'GO data\nnot available', ha='center',
                 va='center', fontsize=12, transform=ax6.transAxes)
        ax6.set_title('F. GO enrichment: WW-only genes',
                      fontsize=12, fontweight='bold')

    fig_path = config.figures_dir / 'condition_functional_characterization.pdf'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"\n  Figure saved: {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  CONDITION-SPECIFIC FUNCTIONAL CHARACTERIZATION")
    print("=" * 70)

    # Load data
    causal_ww = pd.read_csv(config.results_dir / 'causal_bqtl_728.csv')
    causal_ds = pd.read_csv(config.results_dir / 'causal_bqtl_ds.csv')

    # Expression
    expr_df = pd.read_csv(config.expression, sep='\t')
    if 'log2fc_DS_vs_WW' in expr_df.columns:
        expr_df = expr_df.rename(columns={'log2fc_DS_vs_WW': 'log2fc'})

    # Decompose
    gene_sets = decompose_gene_sets(causal_ww, causal_ds)

    # Load annotations
    ann_df = load_annotations()

    # Functional annotation
    func_results = annotate_gene_sets(gene_sets, ann_df, expr_df)

    # GO enrichment
    go_results = go_enrichment(gene_sets, ann_df)

    # Keyword embedding
    embed_result = compute_keyword_embeddings(gene_sets, ann_df)
    embedding_df = embed_result[0] if embed_result else None
    embed_method = embed_result[1] if embed_result else 'N/A'

    # Figures
    make_figures(gene_sets, func_results, expr_df, embedding_df, embed_method,
                 go_results)

    # Save gene lists
    import json
    out = {
        'ww_only_genes': sorted(gene_sets['ww_only']),
        'ds_only_genes': sorted(gene_sets['ds_only']),
        'shared_same_variant': sorted(gene_sets['shared_same']),
        'shared_diff_variant': sorted(gene_sets['shared_diff']),
    }
    out_path = config.results_dir / 'condition_specific_gene_sets.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Gene sets saved: {out_path}")

    # Save functional summary as CSV
    if func_results:
        rows = []
        for label, res in func_results.items():
            for cat, count in res['categories'].items():
                rows.append({
                    'condition': label,
                    'category': cat,
                    'count': count,
                    'n_genes': res['n_genes'],
                    'n_annotated': res['n_annotated'],
                    'fraction': count / max(res['n_annotated'], 1),
                })
        cat_df = pd.DataFrame(rows)
        cat_path = config.results_dir / 'condition_functional_categories.csv'
        cat_df.to_csv(cat_path, index=False)
        print(f"  Categories saved: {cat_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
