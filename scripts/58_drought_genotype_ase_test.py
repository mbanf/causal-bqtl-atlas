#!/usr/bin/env python3
"""
58_drought_genotype_ase_test.py — Paired WW vs drought genotype→ASE comparison.

Part of: "Condition-dependent bQTL switching" paper (Banf & Hartwig)
Paper section: "Condition-dependent bQTL switching"
Pipeline step: 6 of 8

This is the CENTRAL analysis script. Repeats the genotype-aware bQTL test from
script 54 under BOTH conditions using the raw MOESM5 Excel data, enabling a
direct paired comparison of bQTL functionality under WW vs drought.

Key results (paper numbers):
  WW: 8,040 pairs, 814 FDR hits (10.1%), 5.17x enrichment, pi1=0.376
  DS: 8,133 pairs, 742 FDR hits (9.1%), 4.95x enrichment, pi1=0.369
  Switching: 372 constitutive, 442 WW-only, 370 drought-specific (Jaccard=0.314)

Input:
  supp_table_MOESM5.xlsx Table_S13a (WW ASE) and Table_S13b (drought ASE)
  nam_founder_genotypes_at_bqtl.tsv (script 53)
  bqtl_snp_ww.csv
  GFF3

Output:
  ww_vs_drought_genotype_bqtl.csv — 16,173 paired WW+DS test results
  ww_vs_drought_tf_family_effects.csv — per-TF-family Δ|d|
  figures/fig15_ww_vs_drought_bqtl.pdf — Paper Fig. 3
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUTDIR = BASE / "results"
FIG_DIR = BASE / "figures" / "paper_figures"
FIG_DIR.mkdir(exist_ok=True)

MOESM5 = DATA / "raw" / "supp_table_MOESM5.xlsx"
GENOTYPE_FILE = DATA / "processed" / "nam_founder_genotypes_at_bqtl.tsv"
BQTL = DATA / "processed" / "bqtl_snp_ww.csv"
GFF3 = DATA / "raw" / "Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3"
WW_RESULTS = OUTDIR / "genotype_bqtl_results.csv"
MOTIF_DATA = OUTDIR / "bound_region_motifs.csv"

VALID_CHROMS = {f'chr{i}' for i in range(1, 11)}
PROMOTER_BP = 2000

FOUNDER_TO_HYBRID = {
    'B97': 'B97', 'CML247': 'CML247', 'CML277': 'CML277',
    'CML322': 'CML322', 'CML333': 'CML333', 'CML69': 'CML69',
    'HP301': 'HP301', 'Il14H': 'IL14H', 'Ki11': 'Ki11', 'Ki3': 'Ki3',
    'Ky21': 'Ky21', 'M162W': 'M162W', 'Mo18W': 'Mo18W', 'Ms71': 'Ms71',
    'NC358': 'NC358', 'Oh43': 'Oh43', 'Oh7B': 'Oh7b', 'P39': 'P39',
    'Tx303': 'Tx303',
}

HYBRIDS = ['A188', 'A619', 'B97', 'CML103', 'CML247', 'CML277', 'CML322',
           'CML333', 'CML69', 'HP301', 'IL14H', 'Ki11', 'Ki3', 'Ky21',
           'M162W', 'Mo17', 'Mo18W', 'Ms71', 'NC358', 'Oh43', 'Oh7b',
           'P39', 'Tx303', 'W22']


def parse_ase_table(xlsx_path, sheet_name):
    """Parse Table S13a or S13b from MOESM5.

    Format: Row 0 = description, Row 1 = hybrid names, Row 2 = column names, Row 3+ = data.
    Each hybrid has 6 columns: Rep1_B73, Rep2_B73, Rep3_B73, Rep1_NAM, Rep2_NAM, Rep3_NAM.

    Returns: DataFrame with gene_id, hybrid, b73_mean, nam_mean, log2_ratio
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    # Extract hybrid names from row 1
    hybrid_row = df.iloc[1].values
    col_names = df.iloc[2].values

    # Data starts at row 3
    data = df.iloc[3:].copy()
    data.columns = range(len(data.columns))

    # Gene ID is column 0
    gene_ids = data[0].values

    # Parse hybrids: find positions where hybrid name appears in row 1
    hybrid_positions = {}
    current_hybrid = None
    for i, val in enumerate(hybrid_row):
        if pd.notna(val) and str(val).startswith('B73x'):
            current_hybrid = str(val).replace('B73x', '')
            hybrid_positions[current_hybrid] = i

    # For each hybrid, extract B73 and NAM allele reads
    results = []
    for hybrid_name, start_col in hybrid_positions.items():
        # 3 B73 reps then 3 NAM reps
        b73_cols = [start_col, start_col + 1, start_col + 2]
        nam_cols = [start_col + 3, start_col + 4, start_col + 5]

        b73_data = data[b73_cols].apply(pd.to_numeric, errors='coerce')
        nam_data = data[nam_cols].apply(pd.to_numeric, errors='coerce')

        b73_mean = b73_data.mean(axis=1)
        nam_mean = nam_data.mean(axis=1)

        for gene_id, b73, nam in zip(gene_ids, b73_mean, nam_mean):
            if pd.isna(b73) or pd.isna(nam) or (b73 + nam) < 2:
                continue
            # Pseudocount to avoid log(0)
            log2_ratio = np.log2((b73 + 0.5) / (nam + 0.5))
            results.append({
                'gene_id': gene_id,
                'hybrid': f'B73x{hybrid_name}',
                'b73_mean': b73,
                'nam_mean': nam,
                'log2_ratio': log2_ratio,
                'total_reads': b73 + nam,
            })

    return pd.DataFrame(results)


def parse_gff3():
    genes = {}
    with open(GFF3) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'gene':
                continue
            chrom = parts[0]
            if chrom not in VALID_CHROMS:
                continue
            start, end = int(parts[3]), int(parts[4])
            strand = parts[6]
            gene_id = None
            for attr in parts[8].split(';'):
                if attr.startswith('ID=gene:'):
                    gene_id = attr.split(':')[1]
                    break
                elif attr.startswith('ID=Zm'):
                    gene_id = attr[3:]
                    break
            if gene_id:
                tss = start if strand == '+' else end
                genes[gene_id] = {'chr': chrom, 'start': start, 'end': end,
                                  'strand': strand, 'tss': tss}
    return genes


def load_genotype_matrix(genotype_file):
    """Load assembly-aligned genotypes. Values: '0|0'=ref, 'A/C/G/T'=SNP, 'DEL', './.'=missing."""
    df = pd.read_csv(genotype_file, sep='\t', dtype=str)
    founder_cols = [c for c in df.columns if c not in ('chr', 'pos', 'ref')]
    print(f"  Loaded genotypes: {len(df):,} variants × {len(founder_cols)} founders")

    # Vectorized binary conversion
    binary = pd.DataFrame(index=df.index)
    binary['chr'] = df['chr']
    binary['pos'] = df['pos'].astype(int)
    hybrid_cols = []
    for col in founder_cols:
        hybrid = FOUNDER_TO_HYBRID.get(col)
        if hybrid is None:
            continue
        vals = df[col]
        is_ref = vals == '0|0'
        is_miss = vals.isin(['./.', 'nan']) | vals.isna()
        binary[hybrid] = np.where(is_miss, np.nan, np.where(is_ref, 0, 1))
        hybrid_cols.append(hybrid)

    genotype_matrix = {}
    arr = binary[hybrid_cols].values
    for i in range(len(binary)):
        chrom = binary.iloc[i]['chr']
        pos = int(binary.iloc[i]['pos'])
        geno = {}
        for j, hybrid in enumerate(hybrid_cols):
            val = arr[i, j]
            if not np.isnan(val):
                geno[hybrid] = int(val)
        if geno:
            genotype_matrix[(chrom, pos)] = geno
    print(f"  Genotype matrix: {len(genotype_matrix):,} positions")
    return genotype_matrix


def map_bqtl_to_genes(bqtl_df, genes):
    mappings = []
    for _, bqtl in bqtl_df.iterrows():
        chrom = bqtl['chr']
        pos = bqtl['pos']
        for gene_id, ginfo in genes.items():
            if ginfo['chr'] != chrom:
                continue
            if abs(pos - ginfo['tss']) <= PROMOTER_BP:
                mappings.append({
                    'chr': chrom, 'pos': pos, 'gene_id': gene_id,
                    'dist_to_tss': abs(pos - ginfo['tss']),
                })
    return pd.DataFrame(mappings)


def run_genotype_test(genotype_matrix, bqtl_gene_df, ase_lookup, condition_label):
    results = []
    for _, row in bqtl_gene_df.iterrows():
        chrom, pos, gene_id = row['chr'], row['pos'], row['gene_id']
        geno = genotype_matrix.get((chrom, pos))
        if geno is None:
            continue
        ase_variant = []
        ase_reference = []
        for hybrid in HYBRIDS:
            g = geno.get(hybrid)
            if g is None:
                continue
            ase_val = ase_lookup.get((gene_id, f'B73x{hybrid}'))
            if ase_val is None or np.isnan(ase_val):
                continue
            if g == 1:
                ase_variant.append(ase_val)
            else:
                ase_reference.append(ase_val)

        if len(ase_variant) >= 3 and len(ase_reference) >= 3:
            mw_stat, mw_p = stats.mannwhitneyu(ase_variant, ase_reference,
                                                alternative='two-sided')
            t_stat, t_p = stats.ttest_ind(ase_variant, ase_reference, equal_var=False)
            mean_v = np.mean(ase_variant)
            mean_r = np.mean(ase_reference)
            sd_v = np.std(ase_variant, ddof=1)
            sd_r = np.std(ase_reference, ddof=1)
            pooled_sd = np.sqrt(((len(ase_variant)-1)*sd_v**2 + (len(ase_reference)-1)*sd_r**2) /
                                (len(ase_variant)+len(ase_reference)-2))
            cohens_d = (mean_v - mean_r) / pooled_sd if pooled_sd > 0 else 0

            results.append({
                'chr': chrom, 'pos': pos, 'gene_id': gene_id,
                'dist_to_tss': row.get('dist_to_tss', np.nan),
                'n_variant': len(ase_variant), 'n_reference': len(ase_reference),
                'mean_ase_variant': mean_v, 'mean_ase_reference': mean_r,
                'cohens_d': cohens_d,
                'abs_ase_variant': np.mean([abs(x) for x in ase_variant]),
                'abs_ase_reference': np.mean([abs(x) for x in ase_reference]),
                'mw_p': mw_p, 't_p': t_p,
                'condition': condition_label,
            })

    rdf = pd.DataFrame(results)
    if len(rdf) > 0:
        _, rdf['fdr_mw'], _, _ = multipletests(rdf['mw_p'], method='fdr_bh')
    return rdf


def main():
    print("=" * 70)
    print("GENOTYPE→ASE TEST: WELL-WATERED vs DROUGHT")
    print("=" * 70)

    # ── Load genotype matrix ──
    print("\n1. Loading genotype matrix...")
    genotype_matrix = load_genotype_matrix(GENOTYPE_FILE)
    print(f"  {len(genotype_matrix):,} bQTL positions with genotype data")

    # ── Parse ASE tables ──
    print("\n2. Parsing ASE tables from MOESM5...")
    print("  Parsing Table_S13a (well-watered)...")
    ase_ww = parse_ase_table(MOESM5, 'Table_S13a')
    print(f"  WW ASE: {len(ase_ww):,} gene-hybrid measurements, {ase_ww['gene_id'].nunique():,} genes")

    print("  Parsing Table_S13b (drought)...")
    ase_ds = parse_ase_table(MOESM5, 'Table_S13b')
    print(f"  DS ASE: {len(ase_ds):,} gene-hybrid measurements, {ase_ds['gene_id'].nunique():,} genes")

    # ── Map bQTL to genes ──
    print("\n3. Mapping bQTL to promoters...")
    bqtl_df = pd.read_csv(BQTL)
    genes = parse_gff3()
    bqtl_with_geno = bqtl_df[
        bqtl_df.apply(lambda r: (r['chr'], r['pos']) in genotype_matrix, axis=1)
    ].copy()
    bqtl_gene_df = map_bqtl_to_genes(bqtl_with_geno, genes)
    print(f"  bQTL-gene pairs: {len(bqtl_gene_df):,}")

    # ── Build ASE lookups ──
    ww_lookup = {(r['gene_id'], r['hybrid']): r['log2_ratio']
                 for _, r in ase_ww.iterrows()}
    ds_lookup = {(r['gene_id'], r['hybrid']): r['log2_ratio']
                 for _, r in ase_ds.iterrows()}
    print(f"  WW lookup: {len(ww_lookup):,}")
    print(f"  DS lookup: {len(ds_lookup):,}")

    # ── Run tests ──
    print("\n4. Running genotype→ASE tests...")
    print("  Well-watered...")
    rdf_ww = run_genotype_test(genotype_matrix, bqtl_gene_df, ww_lookup, 'WW')
    print(f"  WW: {len(rdf_ww):,} testable pairs")

    print("  Drought...")
    rdf_ds = run_genotype_test(genotype_matrix, bqtl_gene_df, ds_lookup, 'DS')
    print(f"  DS: {len(rdf_ds):,} testable pairs")

    # ── Compare results ──
    print(f"\n{'='*70}")
    print("RESULTS COMPARISON: WELL-WATERED vs DROUGHT")
    print(f"{'='*70}")

    for label, rdf in [('Well-watered', rdf_ww), ('Drought', rdf_ds)]:
        n_sig = (rdf['fdr_mw'] < 0.05).sum()
        n_nom = (rdf['mw_p'] < 0.05).sum()
        pi0 = min(1.0, 2 * (rdf['mw_p'] > 0.5).mean()) if len(rdf) > 100 else 1.0
        pi1 = 1 - pi0
        frac_higher = (rdf['abs_ase_variant'] > rdf['abs_ase_reference']).mean()
        print(f"\n  {label} (n={len(rdf):,}):")
        print(f"    FDR<0.05: {n_sig:,} ({100*n_sig/len(rdf):.1f}%)")
        print(f"    Nominal p<0.05: {n_nom:,} ({100*n_nom/len(rdf):.1f}%), expected: {0.05*len(rdf):.0f}")
        print(f"    Enrichment: {n_nom/(0.05*len(rdf)):.2f}x")
        print(f"    pi1: {pi1:.3f}")
        print(f"    Mean |d|: {rdf['cohens_d'].abs().mean():.4f}")
        print(f"    |ASE| variant > ref: {100*frac_higher:.1f}%")

    # ── Overlap analysis ──
    if len(rdf_ww) > 0 and len(rdf_ds) > 0:
        print(f"\n{'='*60}")
        print("OVERLAP: WW vs DROUGHT FUNCTIONAL bQTL")
        print(f"{'='*60}")

        ww_sig = set(zip(rdf_ww[rdf_ww['fdr_mw'] < 0.05]['chr'],
                         rdf_ww[rdf_ww['fdr_mw'] < 0.05]['pos'],
                         rdf_ww[rdf_ww['fdr_mw'] < 0.05]['gene_id']))
        ds_sig = set(zip(rdf_ds[rdf_ds['fdr_mw'] < 0.05]['chr'],
                         rdf_ds[rdf_ds['fdr_mw'] < 0.05]['pos'],
                         rdf_ds[rdf_ds['fdr_mw'] < 0.05]['gene_id']))

        both = ww_sig & ds_sig
        ww_only = ww_sig - ds_sig
        ds_only = ds_sig - ww_sig

        print(f"  WW significant: {len(ww_sig):,}")
        print(f"  DS significant: {len(ds_sig):,}")
        print(f"  Both: {len(both):,}")
        print(f"  WW only: {len(ww_only):,}")
        print(f"  DS only (drought-specific): {len(ds_only):,}")

        # ── Per-TF-family comparison ──
        print(f"\n{'='*60}")
        print("PER-TF-FAMILY: WW vs DROUGHT EFFECTS")
        print(f"{'='*60}")

        motifs = pd.read_csv(MOTIF_DATA)

        # Merge both results with motifs
        ww_motif = rdf_ww.merge(motifs[motifs['motifs_at_variant'] > 0],
                                 on=['chr', 'pos', 'gene_id'], how='inner')
        ds_motif = rdf_ds.merge(motifs[motifs['motifs_at_variant'] > 0],
                                 on=['chr', 'pos', 'gene_id'], how='inner')

        tf_comparison = []
        all_families = set(ww_motif['tf_family'].unique()) | set(ds_motif['tf_family'].unique())
        for fam in sorted(all_families):
            if fam == 'NONE':
                continue
            ww_fam = ww_motif[ww_motif['tf_family'] == fam].drop_duplicates(
                subset=['chr', 'pos', 'gene_id'])
            ds_fam = ds_motif[ds_motif['tf_family'] == fam].drop_duplicates(
                subset=['chr', 'pos', 'gene_id'])
            if len(ww_fam) < 20 or len(ds_fam) < 20:
                continue

            ww_abs_d = ww_fam['cohens_d'].abs().mean()
            ds_abs_d = ds_fam['cohens_d'].abs().mean()
            ww_sig_rate = (ww_fam['fdr_mw'] < 0.05).mean()
            ds_sig_rate = (ds_fam['fdr_mw'] < 0.05).mean()

            tf_comparison.append({
                'tf_family': fam,
                'ww_n': len(ww_fam), 'ds_n': len(ds_fam),
                'ww_abs_d': ww_abs_d, 'ds_abs_d': ds_abs_d,
                'd_change': ds_abs_d - ww_abs_d,
                'ww_sig_rate': ww_sig_rate, 'ds_sig_rate': ds_sig_rate,
                'sig_change': ds_sig_rate - ww_sig_rate,
            })

        tf_comp = pd.DataFrame(tf_comparison).sort_values('d_change', ascending=False)

        print(f"\n  {'Family':<15} {'WW |d|':>7} {'DS |d|':>7} {'Δ|d|':>7}  "
              f"{'WW sig%':>7} {'DS sig%':>7} {'Δsig':>7}")
        print("  " + "-" * 65)
        for _, row in tf_comp.iterrows():
            arrow = '↑' if row['d_change'] > 0.02 else '↓' if row['d_change'] < -0.02 else '='
            print(f"  {row['tf_family']:<15} {row['ww_abs_d']:7.3f} {row['ds_abs_d']:7.3f} "
                  f"{row['d_change']:+7.3f}{arrow} "
                  f"{100*row['ww_sig_rate']:6.1f}% {100*row['ds_sig_rate']:6.1f}% "
                  f"{100*row['sig_change']:+6.1f}%")

        # ── Save ──
        combined = pd.concat([rdf_ww, rdf_ds])
        combined.to_csv(OUTDIR / 'ww_vs_drought_genotype_bqtl.csv', index=False)
        tf_comp.to_csv(OUTDIR / 'ww_vs_drought_tf_family_effects.csv', index=False)
        print(f"\n  Saved: ww_vs_drought_genotype_bqtl.csv")
        print(f"  Saved: ww_vs_drought_tf_family_effects.csv")

        # ── Figures ──
        print("\n5. Generating figures...")
        make_figures(rdf_ww, rdf_ds, tf_comp)


def make_figures(rdf_ww, rdf_ds, tf_comp):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # A: P-value histograms comparison
    ax = axes[0, 0]
    ax.hist(rdf_ww['mw_p'], bins=50, alpha=0.6, color='#1976D2', density=True,
            edgecolor='white', label=f'Well-watered (n={len(rdf_ww):,})')
    ax.hist(rdf_ds['mw_p'], bins=50, alpha=0.6, color='#D32F2F', density=True,
            edgecolor='white', label=f'Drought (n={len(rdf_ds):,})')
    ax.axhline(1, color='gray', linestyle='--', alpha=0.5, label='Uniform')
    ax.set_xlabel('Mann-Whitney p-value')
    ax.set_ylabel('Density')
    ww_sig = (rdf_ww['fdr_mw'] < 0.05).sum()
    ds_sig = (rdf_ds['fdr_mw'] < 0.05).sum()
    ax.set_title(f'A. P-value distributions\nWW: {ww_sig:,} FDR sig, DS: {ds_sig:,}')
    ax.legend(fontsize=8)

    # B: Effect size comparison
    ax = axes[0, 1]
    ax.hist(rdf_ww['cohens_d'].abs().clip(0, 5), bins=40, alpha=0.6, color='#1976D2',
            density=True, edgecolor='white', label='WW')
    ax.hist(rdf_ds['cohens_d'].abs().clip(0, 5), bins=40, alpha=0.6, color='#D32F2F',
            density=True, edgecolor='white', label='DS')
    ax.set_xlabel("|Cohen's d|")
    ax.set_ylabel('Density')
    ax.set_title(f"B. Effect size distributions\n"
                 f"WW mean: {rdf_ww['cohens_d'].abs().mean():.3f}, "
                 f"DS mean: {rdf_ds['cohens_d'].abs().mean():.3f}")
    ax.legend(fontsize=8)

    # C: Paired comparison for shared bQTL
    ax = axes[0, 2]
    shared = rdf_ww.merge(rdf_ds, on=['chr', 'pos', 'gene_id'], suffixes=('_ww', '_ds'))
    if len(shared) > 0:
        ax.scatter(shared['cohens_d_ww'].abs().clip(0, 5),
                   shared['cohens_d_ds'].abs().clip(0, 5),
                   s=3, alpha=0.2, color='#7B1FA2', rasterized=True)
        lim = 5
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
        rho, p = stats.spearmanr(shared['cohens_d_ww'].abs(), shared['cohens_d_ds'].abs())
        ax.set_xlabel("|d| Well-watered")
        ax.set_ylabel("|d| Drought")
        ax.set_title(f'C. Paired effect sizes (n={len(shared):,})\nrho={rho:.3f}, p={p:.2e}')
    else:
        ax.text(0.5, 0.5, 'No shared pairs', transform=ax.transAxes, ha='center')

    # D: Per-TF-family change in effect size
    ax = axes[1, 0]
    tc = tf_comp.sort_values('d_change')
    colors = ['#D32F2F' if d > 0.02 else '#1976D2' if d < -0.02 else '#757575'
              for d in tc['d_change']]
    ax.barh(range(len(tc)), tc['d_change'], color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(tc)))
    ax.set_yticklabels(tc['tf_family'], fontsize=8)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Δ|d| (drought − well-watered)")
    ax.set_title('D. Change in ASE effect under drought')

    # E: Per-TF-family change in significance rate
    ax = axes[1, 1]
    tc2 = tf_comp.sort_values('sig_change')
    colors2 = ['#D32F2F' if d > 0.01 else '#1976D2' if d < -0.01 else '#757575'
               for d in tc2['sig_change']]
    ax.barh(range(len(tc2)), 100 * tc2['sig_change'], color=colors2, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(tc2)))
    ax.set_yticklabels(tc2['tf_family'], fontsize=8)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Δ FDR sig rate (drought − WW, %)")
    ax.set_title('E. Change in significance rate under drought')

    # F: Summary bar chart
    ax = axes[1, 2]
    metrics = ['FDR<0.05\nhits', 'Nominal\nenrichment', 'pi1', 'Mean |d|']
    ww_sig_n = (rdf_ww['fdr_mw'] < 0.05).sum()
    ds_sig_n = (rdf_ds['fdr_mw'] < 0.05).sum()
    ww_enr = (rdf_ww['mw_p'] < 0.05).sum() / (0.05 * len(rdf_ww))
    ds_enr = (rdf_ds['mw_p'] < 0.05).sum() / (0.05 * len(rdf_ds))
    ww_pi1 = 1 - min(1, 2*(rdf_ww['mw_p'] > 0.5).mean())
    ds_pi1 = 1 - min(1, 2*(rdf_ds['mw_p'] > 0.5).mean())
    ww_d = rdf_ww['cohens_d'].abs().mean()
    ds_d = rdf_ds['cohens_d'].abs().mean()

    # Normalize for display
    ww_vals = [ww_sig_n, ww_enr, ww_pi1, ww_d]
    ds_vals = [ds_sig_n, ds_enr, ds_pi1, ds_d]
    x = np.arange(len(metrics))
    w = 0.35
    # Use normalized scale (divide by max of each pair)
    for i in range(len(metrics)):
        mx = max(abs(ww_vals[i]), abs(ds_vals[i]), 0.001)
        ww_norm = ww_vals[i] / mx
        ds_norm = ds_vals[i] / mx
        ax.bar(x[i] - w/2, ww_norm, w, color='#1976D2', edgecolor='white')
        ax.bar(x[i] + w/2, ds_norm, w, color='#D32F2F', edgecolor='white')
        ax.text(x[i] - w/2, ww_norm + 0.05, f'{ww_vals[i]:.2f}' if i > 0 else f'{int(ww_vals[i])}',
                ha='center', fontsize=7, color='#1976D2')
        ax.text(x[i] + w/2, ds_norm + 0.05, f'{ds_vals[i]:.2f}' if i > 0 else f'{int(ds_vals[i])}',
                ha='center', fontsize=7, color='#D32F2F')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel('Normalized value')
    ax.set_title('F. WW vs Drought summary')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#1976D2', label='WW'),
                        Patch(color='#D32F2F', label='Drought')],
              fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig15_ww_vs_drought_bqtl.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(FIG_DIR / 'fig15_ww_vs_drought_bqtl.png', bbox_inches='tight', dpi=150)
    print(f"  Saved: figures/fig15_ww_vs_drought_bqtl.pdf")
    plt.close()


if __name__ == '__main__':
    main()
