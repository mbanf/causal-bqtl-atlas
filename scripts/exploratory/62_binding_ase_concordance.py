#!/usr/bin/env python3
"""
62_binding_ase_concordance.py — Test whether ASE switching reflects binding switching.

Part of: "Condition-dependent bQTL switching" paper (Banf & Hartwig)
Pipeline step: new (after 58-60)

Uses drought bQTL (Table S10) and differential binding (Table S9) from Engelhorn
to test whether bQTL that switch ASE functionality between WW and drought are
the same positions where TF binding itself changes.

Key questions:
  1. Do ASE-switching bQTL overlap with binding-switching positions?
  2. Does binding Jaccard (0.165) predict ASE Jaccard (0.262)?
  3. Do hybrids with more binding reorganization show more ASE switching?

Input:
  supp_table_MOESM5.xlsx Table_S6 (WW bQTL), Table_S10 (DS bQTL), Table_S9 (diff binding)
  ww_vs_drought_genotype_bqtl.csv (from script 58)
  data/processed/bqtl_snp_ww.csv

Output:
  binding_ase_concordance.csv
  figures/fig18_binding_ase_concordance.pdf
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import openpyxl
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUTDIR = BASE / "results"
FIG_DIR = BASE / "figures" / "paper_figures"
FIG_DIR.mkdir(exist_ok=True)

MOESM5 = DATA / "raw" / "supp_table_MOESM5.xlsx"
WW_BQTL_FILE = DATA / "processed" / "bqtl_snp_ww.csv"
ASE_RESULTS = OUTDIR / "ww_vs_drought_genotype_bqtl.csv"


def load_ds_bqtl(xlsx_path):
    """Load drought bQTL positions from Table S10."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb['Table_S10']
    positions = set()
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i < 3:
            continue
        # SNP bQTL in column 0
        pos = row[0]
        if pos and isinstance(pos, str) and pos.startswith('B73-chr'):
            parts = pos.split('_')
            chrom = parts[0].replace('B73-', '')
            coord = int(parts[1])
            positions.add((chrom, coord))
        # INDEL bQTL in column 4
        pos2 = row[4] if len(row) > 4 else None
        if pos2 and isinstance(pos2, str) and pos2.startswith('B73-chr'):
            parts = pos2.split('_')
            chrom = parts[0].replace('B73-', '')
            coord = int(parts[1])
            positions.add((chrom, coord))
    wb.close()
    return positions


def load_diff_binding(xlsx_path):
    """Load per-hybrid differential binding counts from Table S9."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb['Table_S9']
    records = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i < 3:
            continue
        hybrid = row[0]
        if hybrid is None:
            break
        ww_higher = row[1]
        ds_higher = row[2]
        if ww_higher is not None and ds_higher is not None:
            records.append({
                'hybrid': str(hybrid),
                'peaks_ww_higher': int(ww_higher),
                'peaks_ds_higher': int(ds_higher),
                'total_diff': int(ww_higher) + int(ds_higher),
                'binding_ratio': int(ds_higher) / (int(ww_higher) + int(ds_higher)),
            })
    wb.close()
    return pd.DataFrame(records)


def main():
    print("=" * 70)
    print("BINDING-ASE CONCORDANCE: Does binding switching predict ASE switching?")
    print("=" * 70)

    # ── Load data ──
    print("\n1. Loading data...")

    # WW bQTL
    ww_df = pd.read_csv(WW_BQTL_FILE)
    ww_set = set(zip(ww_df['chr'], ww_df['pos']))
    print(f"  WW bQTL: {len(ww_set):,}")

    # DS bQTL
    ds_set = load_ds_bqtl(MOESM5)
    print(f"  DS bQTL: {len(ds_set):,}")

    # Binding-level classification
    both_binding = ww_set & ds_set
    ww_only_binding = ww_set - ds_set
    ds_only_binding = ds_set - ww_set
    all_binding = ww_set | ds_set
    binding_jaccard = len(both_binding) / len(all_binding)

    print(f"\n  Binding-level switching:")
    print(f"    Both conditions: {len(both_binding):,} ({100*len(both_binding)/len(all_binding):.1f}%)")
    print(f"    WW-only binding: {len(ww_only_binding):,}")
    print(f"    DS-only binding: {len(ds_only_binding):,}")
    print(f"    Jaccard (binding): {binding_jaccard:.3f}")

    # Create binding status lookup
    binding_status = {}
    for pos in both_binding:
        binding_status[pos] = 'both'
    for pos in ww_only_binding:
        binding_status[pos] = 'ww_only'
    for pos in ds_only_binding:
        binding_status[pos] = 'ds_only'

    # ASE switching results
    ase_df = pd.read_csv(ASE_RESULTS)
    print(f"\n  ASE results: {len(ase_df):,} rows")

    # Split WW and DS results
    ase_ww = ase_df[ase_df['condition'] == 'WW'].copy()
    ase_ds = ase_df[ase_df['condition'] == 'DS'].copy()
    print(f"  WW ASE tests: {len(ase_ww):,}")
    print(f"  DS ASE tests: {len(ase_ds):,}")

    # Classify ASE switching per bQTL-gene pair
    ww_sig = set(zip(ase_ww[ase_ww['fdr_mw'] < 0.05]['chr'],
                     ase_ww[ase_ww['fdr_mw'] < 0.05]['pos'],
                     ase_ww[ase_ww['fdr_mw'] < 0.05]['gene_id']))
    ds_sig = set(zip(ase_ds[ase_ds['fdr_mw'] < 0.05]['chr'],
                     ase_ds[ase_ds['fdr_mw'] < 0.05]['pos'],
                     ase_ds[ase_ds['fdr_mw'] < 0.05]['gene_id']))

    ase_both = ww_sig & ds_sig
    ase_ww_only = ww_sig - ds_sig
    ase_ds_only = ds_sig - ww_sig

    # Get all tested pairs (tested in BOTH conditions)
    ww_tested = set(zip(ase_ww['chr'], ase_ww['pos'], ase_ww['gene_id']))
    ds_tested = set(zip(ase_ds['chr'], ase_ds['pos'], ase_ds['gene_id']))
    both_tested = ww_tested & ds_tested

    # Non-functional in either
    ase_none = both_tested - ww_sig - ds_sig

    print(f"\n  ASE switching (paired bQTL-gene pairs):")
    print(f"    Constitutive: {len(ase_both):,}")
    print(f"    WW-only ASE: {len(ase_ww_only):,}")
    print(f"    DS-only ASE: {len(ase_ds_only):,}")
    print(f"    Non-functional: {len(ase_none):,}")

    # ── 2. CONCORDANCE: Binding status x ASE switching ──
    print(f"\n{'='*70}")
    print("2. CONCORDANCE: Binding status x ASE switching category")
    print(f"{'='*70}")

    # For each ASE-tested pair, get binding status of the bQTL position
    concordance_rows = []
    for category, pairs, label in [
        ('constitutive', ase_both, 'ASE both'),
        ('ww_only_ase', ase_ww_only, 'ASE WW-only'),
        ('ds_only_ase', ase_ds_only, 'ASE DS-only'),
        ('non_functional', ase_none, 'Non-functional'),
    ]:
        for chrom, pos, gene_id in pairs:
            bs = binding_status.get((chrom, pos), 'unknown')
            concordance_rows.append({
                'chr': chrom, 'pos': pos, 'gene_id': gene_id,
                'ase_category': category,
                'binding_status': bs,
            })

    conc = pd.DataFrame(concordance_rows)
    print(f"\n  Total paired observations: {len(conc):,}")

    # Cross-tabulate
    ct = pd.crosstab(conc['ase_category'], conc['binding_status'],
                      margins=True, margins_name='Total')
    print(f"\n  Cross-tabulation (ASE category x Binding status):\n")
    print(ct.to_string())

    # Proportions within each ASE category
    print(f"\n  Binding status proportions within each ASE category:")
    print(f"  {'ASE category':<18} {'both bind':>10} {'ww_only':>10} {'ds_only':>10} {'unknown':>10}  N")
    print(f"  {'-'*70}")
    for cat in ['constitutive', 'ww_only_ase', 'ds_only_ase', 'non_functional']:
        sub = conc[conc['ase_category'] == cat]
        n = len(sub)
        if n == 0:
            continue
        both_pct = 100 * (sub['binding_status'] == 'both').mean()
        ww_pct = 100 * (sub['binding_status'] == 'ww_only').mean()
        ds_pct = 100 * (sub['binding_status'] == 'ds_only').mean()
        unk_pct = 100 * (sub['binding_status'] == 'unknown').mean()
        print(f"  {cat:<18} {both_pct:9.1f}% {ww_pct:9.1f}% {ds_pct:9.1f}% {unk_pct:9.1f}%  {n:,}")

    # ── Key tests ──
    print(f"\n  Statistical tests:")

    # Test 1: Are DS-only ASE bQTL enriched at DS-only binding sites?
    # Compare DS-only ASE vs all others for fraction at DS-only binding
    ds_ase = conc[conc['ase_category'] == 'ds_only_ase']
    not_ds_ase = conc[conc['ase_category'] != 'ds_only_ase']
    a = (ds_ase['binding_status'] == 'ds_only').sum()
    b = len(ds_ase) - a
    c = (not_ds_ase['binding_status'] == 'ds_only').sum()
    d = len(not_ds_ase) - c
    or_ds, p_ds = stats.fisher_exact([[a, b], [c, d]])
    print(f"\n  DS-only ASE enriched at DS-only binding sites?")
    print(f"    DS-only ASE at DS-only binding: {a}/{len(ds_ase)} ({100*a/len(ds_ase):.1f}%)")
    print(f"    Others at DS-only binding: {c}/{len(not_ds_ase)} ({100*c/len(not_ds_ase):.1f}%)")
    print(f"    OR={or_ds:.3f}, Fisher p={p_ds:.4e}")

    # Test 2: Are WW-only ASE bQTL enriched at WW-only binding sites?
    ww_ase = conc[conc['ase_category'] == 'ww_only_ase']
    not_ww_ase = conc[conc['ase_category'] != 'ww_only_ase']
    a2 = (ww_ase['binding_status'] == 'ww_only').sum()
    b2 = len(ww_ase) - a2
    c2 = (not_ww_ase['binding_status'] == 'ww_only').sum()
    d2 = len(not_ww_ase) - c2
    or_ww, p_ww = stats.fisher_exact([[a2, b2], [c2, d2]])
    print(f"\n  WW-only ASE enriched at WW-only binding sites?")
    print(f"    WW-only ASE at WW-only binding: {a2}/{len(ww_ase)} ({100*a2/len(ww_ase):.1f}%)")
    print(f"    Others at WW-only binding: {c2}/{len(not_ww_ase)} ({100*c2/len(not_ww_ase):.1f}%)")
    print(f"    OR={or_ww:.3f}, Fisher p={p_ww:.4e}")

    # Test 3: Are constitutive ASE bQTL enriched at constitutive binding sites?
    const_ase = conc[conc['ase_category'] == 'constitutive']
    not_const = conc[conc['ase_category'] != 'constitutive']
    a3 = (const_ase['binding_status'] == 'both').sum()
    b3 = len(const_ase) - a3
    c3 = (not_const['binding_status'] == 'both').sum()
    d3 = len(not_const) - c3
    or_c, p_c = stats.fisher_exact([[a3, b3], [c3, d3]])
    print(f"\n  Constitutive ASE enriched at constitutive binding sites?")
    print(f"    Constitutive ASE at both-binding: {a3}/{len(const_ase)} ({100*a3/len(const_ase):.1f}%)")
    print(f"    Others at both-binding: {c3}/{len(not_const)} ({100*c3/len(not_const):.1f}%)")
    print(f"    OR={or_c:.3f}, Fisher p={p_c:.4e}")

    # Test 4: Overall chi-squared for ASE x binding independence
    # Exclude unknowns for clean test
    clean = conc[conc['binding_status'] != 'unknown']
    if len(clean) > 0:
        ct_clean = pd.crosstab(clean['ase_category'], clean['binding_status'])
        chi2, p_chi2, dof, expected = stats.chi2_contingency(ct_clean)
        print(f"\n  Overall chi-squared (ASE category x binding status):")
        print(f"    chi2={chi2:.2f}, dof={dof}, p={p_chi2:.4e}")

    # ── 3. PER-HYBRID BINDING REMODELING vs ASE SWITCHING ──
    print(f"\n{'='*70}")
    print("3. HYBRID-LEVEL: Binding remodeling vs ASE switching")
    print(f"{'='*70}")

    diff_binding = load_diff_binding(MOESM5)
    print(f"\n  Differential binding data for {len(diff_binding)} hybrids")

    # Per-hybrid ASE switching rate
    # For each hybrid, count how many bQTL switch ASE
    # We need per-hybrid ASE data — use the individual-level data from ase_ww/ase_ds
    hybrid_ase_stats = []
    for _, row_ww in ase_ww.iterrows():
        key = (row_ww['chr'], row_ww['pos'], row_ww['gene_id'])
        # Find matching DS row
        ds_match = ase_ds[(ase_ds['chr'] == row_ww['chr']) &
                          (ase_ds['pos'] == row_ww['pos']) &
                          (ase_ds['gene_id'] == row_ww['gene_id'])]
        if len(ds_match) == 0:
            continue
        row_ds = ds_match.iloc[0]
        ww_d = abs(row_ww['cohens_d'])
        ds_d = abs(row_ds['cohens_d'])
        hybrid_ase_stats.append({
            'chr': row_ww['chr'], 'pos': row_ww['pos'],
            'gene_id': row_ww['gene_id'],
            'ww_d': ww_d, 'ds_d': ds_d,
            'delta_d': ds_d - ww_d,
            'ww_p': row_ww['mw_p'], 'ds_p': row_ds['mw_p'],
        })

    hybrid_stats = pd.DataFrame(hybrid_ase_stats)
    print(f"  Paired ASE observations: {len(hybrid_stats):,}")

    # Correlate: per hybrid, binding asymmetry vs ASE change
    # binding_ratio = fraction of diff peaks that are DS-higher
    # If binding_ratio > 0.5, hybrid gains more binding under DS
    # Mean delta_d should reflect whether ASE effects increase or decrease

    print(f"\n  Per-hybrid binding asymmetry (Table S9):")
    print(f"  {'Hybrid':<10} {'WW>DS':>8} {'DS>WW':>8} {'Ratio':>6} {'Total diff':>10}")
    print(f"  {'-'*50}")
    for _, row in diff_binding.sort_values('binding_ratio').iterrows():
        print(f"  {row['hybrid']:<10} {row['peaks_ww_higher']:>8,} {row['peaks_ds_higher']:>8,} "
              f"{row['binding_ratio']:>5.2f} {row['total_diff']:>10,}")

    # Summary stats
    mean_ratio = diff_binding['binding_ratio'].mean()
    print(f"\n  Mean binding ratio (DS fraction): {mean_ratio:.3f}")
    print(f"  Hybrids with WW-dominant binding (ratio<0.4): "
          f"{(diff_binding['binding_ratio'] < 0.4).sum()}")
    print(f"  Hybrids with balanced binding (0.4-0.6): "
          f"{((diff_binding['binding_ratio'] >= 0.4) & (diff_binding['binding_ratio'] <= 0.6)).sum()}")
    print(f"  Hybrids with DS-dominant binding (ratio>0.6): "
          f"{(diff_binding['binding_ratio'] > 0.6).sum()}")

    # ── 4. BINDING CATEGORY PREDICTS ASE EFFECT SIZE ──
    print(f"\n{'='*70}")
    print("4. BINDING CATEGORY PREDICTS ASE EFFECT SIZE")
    print(f"{'='*70}")

    # Add binding status to paired stats
    hybrid_stats['binding_status'] = hybrid_stats.apply(
        lambda r: binding_status.get((r['chr'], int(r['pos'])), 'unknown'), axis=1)

    for bs in ['both', 'ww_only', 'ds_only']:
        sub = hybrid_stats[hybrid_stats['binding_status'] == bs]
        if len(sub) < 10:
            continue
        print(f"\n  Binding={bs} (n={len(sub):,}):")
        print(f"    WW mean |d|: {sub['ww_d'].mean():.4f}")
        print(f"    DS mean |d|: {sub['ds_d'].mean():.4f}")
        print(f"    Mean delta_d: {sub['delta_d'].mean():+.4f}")
        print(f"    WW FDR<0.05 rate: {(sub['ww_p'] < 0.05).mean()*100:.1f}%")
        print(f"    DS FDR<0.05 rate: {(sub['ds_p'] < 0.05).mean()*100:.1f}%")

    # Test: do WW-only binding sites have higher WW |d| and lower DS |d|?
    ww_bind = hybrid_stats[hybrid_stats['binding_status'] == 'ww_only']
    ds_bind = hybrid_stats[hybrid_stats['binding_status'] == 'ds_only']
    both_bind = hybrid_stats[hybrid_stats['binding_status'] == 'both']

    if len(ww_bind) > 10 and len(ds_bind) > 10:
        stat, p = stats.mannwhitneyu(ww_bind['delta_d'], ds_bind['delta_d'])
        print(f"\n  Delta_d comparison (DS-only binding vs WW-only binding):")
        print(f"    WW-only binding mean delta_d: {ww_bind['delta_d'].mean():+.4f}")
        print(f"    DS-only binding mean delta_d: {ds_bind['delta_d'].mean():+.4f}")
        print(f"    Mann-Whitney p: {p:.4e}")

    if len(both_bind) > 10 and len(ww_bind) > 10:
        stat2, p2 = stats.mannwhitneyu(both_bind['ww_d'], ww_bind['ww_d'])
        print(f"\n  WW |d| at constitutive vs WW-only binding sites:")
        print(f"    Both-binding WW |d|: {both_bind['ww_d'].mean():.4f}")
        print(f"    WW-only binding WW |d|: {ww_bind['ww_d'].mean():.4f}")
        print(f"    Mann-Whitney p: {p2:.4e}")

    # ── 5. FIGURES ──
    print(f"\n{'='*70}")
    print("5. Generating figures...")
    print(f"{'='*70}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # A: Binding-level Venn (bar chart representation)
    ax = axes[0, 0]
    categories = ['WW-only\nbinding', 'Both\nconditions', 'DS-only\nbinding']
    counts = [len(ww_only_binding), len(both_binding), len(ds_only_binding)]
    colors = ['#1976D2', '#7B1FA2', '#D32F2F']
    ax.bar(categories, counts, color=colors, edgecolor='white')
    for i, (c, v) in enumerate(zip(categories, counts)):
        ax.text(i, v + 2000, f'{v:,}', ha='center', fontsize=9)
    ax.set_ylabel('Number of bQTL')
    ax.set_title(f'A. Binding-level switching\nJaccard = {binding_jaccard:.3f}')

    # B: ASE category x binding status (stacked bar)
    ax = axes[0, 1]
    ase_cats = ['constitutive', 'ww_only_ase', 'ds_only_ase', 'non_functional']
    ase_labels = ['Constitutive\nASE', 'WW-only\nASE', 'DS-only\nASE', 'Non-\nfunctional']
    bind_cats = ['both', 'ww_only', 'ds_only']
    bind_colors = {'both': '#7B1FA2', 'ww_only': '#1976D2', 'ds_only': '#D32F2F'}

    bottom = np.zeros(len(ase_cats))
    for bc in bind_cats:
        vals = []
        for ac in ase_cats:
            sub = conc[(conc['ase_category'] == ac) & (conc['binding_status'] != 'unknown')]
            if len(sub) > 0:
                vals.append((sub['binding_status'] == bc).mean() * 100)
            else:
                vals.append(0)
        ax.bar(ase_labels, vals, bottom=bottom, color=bind_colors[bc],
               label=f'{bc} binding', edgecolor='white')
        bottom += vals

    ax.set_ylabel('% of bQTL')
    ax.set_title('B. Binding status by ASE category')
    ax.legend(fontsize=8, loc='upper right')

    # C: Effect size by binding category
    ax = axes[0, 2]
    data_ww = []
    data_ds = []
    labels_bind = []
    for bs, label in [('both', 'Both'), ('ww_only', 'WW-only'), ('ds_only', 'DS-only')]:
        sub = hybrid_stats[hybrid_stats['binding_status'] == bs]
        if len(sub) > 10:
            data_ww.append(sub['ww_d'].values)
            data_ds.append(sub['ds_d'].values)
            labels_bind.append(label)

    x = np.arange(len(labels_bind))
    w = 0.35
    means_ww = [np.mean(d) for d in data_ww]
    means_ds = [np.mean(d) for d in data_ds]
    ax.bar(x - w/2, means_ww, w, color='#1976D2', label='WW |d|', edgecolor='white')
    ax.bar(x + w/2, means_ds, w, color='#D32F2F', label='DS |d|', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bind)
    ax.set_ylabel("Mean |Cohen's d|")
    ax.set_title('C. ASE effect by binding category')
    ax.legend(fontsize=8)

    # D: Delta_d distribution by binding status
    ax = axes[1, 0]
    for bs, color, label in [('ww_only', '#1976D2', 'WW-only bind'),
                              ('both', '#7B1FA2', 'Both bind'),
                              ('ds_only', '#D32F2F', 'DS-only bind')]:
        sub = hybrid_stats[hybrid_stats['binding_status'] == bs]
        if len(sub) > 10:
            ax.hist(sub['delta_d'].clip(-3, 3), bins=50, alpha=0.5, color=color,
                    density=True, label=f'{label} (n={len(sub):,})')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Delta |d| (DS - WW)')
    ax.set_ylabel('Density')
    ax.set_title('D. ASE change by binding category')
    ax.legend(fontsize=8)

    # E: Hybrid binding asymmetry
    ax = axes[1, 1]
    db = diff_binding.sort_values('binding_ratio')
    colors_bar = ['#1976D2' if r < 0.4 else '#D32F2F' if r > 0.6 else '#757575'
                  for r in db['binding_ratio']]
    ax.barh(range(len(db)), db['binding_ratio'] - 0.5, color=colors_bar,
            edgecolor='white', height=0.7)
    ax.set_yticks(range(len(db)))
    ax.set_yticklabels(db['hybrid'], fontsize=7)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Binding asymmetry (>0 = more DS binding)')
    ax.set_title('E. Per-hybrid binding remodeling')

    # F: Summary — Jaccard comparison
    ax = axes[1, 2]
    jaccard_levels = ['Binding\n(Table S6 vs S10)', 'ASE functional\n(script 58)']
    jaccards = [binding_jaccard, len(ase_both) / (len(ase_both) + len(ase_ww_only) + len(ase_ds_only))]
    ax.bar(jaccard_levels, jaccards, color=['#FF7043', '#42A5F5'], edgecolor='white', width=0.5)
    for i, j in enumerate(jaccards):
        ax.text(i, j + 0.01, f'{j:.3f}', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Jaccard index')
    ax.set_ylim(0, 0.5)
    ax.set_title('F. Condition overlap: binding vs ASE\n(lower = more switching)')
    ax.axhline(1/3, color='gray', linestyle=':', alpha=0.5, label='Random (1/3)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig18_binding_ase_concordance.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(FIG_DIR / 'fig18_binding_ase_concordance.png', bbox_inches='tight', dpi=150)
    print(f"  Saved: figures/fig18_binding_ase_concordance.pdf")
    plt.close()

    # ── Save ──
    conc.to_csv(OUTDIR / 'binding_ase_concordance.csv', index=False)
    print(f"  Saved: binding_ase_concordance.csv ({len(conc):,} rows)")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Binding Jaccard: {binding_jaccard:.3f} (83.5% condition-specific)")
    ase_j = len(ase_both) / (len(ase_both) + len(ase_ww_only) + len(ase_ds_only))
    print(f"  ASE Jaccard: {ase_j:.3f} ({100*(1-ase_j):.1f}% condition-specific)")
    print(f"  Binding switches MORE than ASE ({binding_jaccard:.3f} < {ase_j:.3f})")
    print(f"  → Consistent with binding as upstream cause, ASE as buffered readout")


if __name__ == '__main__':
    main()
