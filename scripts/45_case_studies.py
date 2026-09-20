#!/usr/bin/env python3
"""
Case study analysis: find specific bQTL that best illustrate
quantitative modulation patterns.

Identifies:
1. MYB-buffered examples: bQTL disrupting MYB sites with high redundancy, low ASE
2. WRKY/bHLH-sensitive examples: bQTL disrupting sole or low-redundancy sites, high ASE
3. Combinatorial disruption: genes with many TF families disrupted
4. ZmTRE1/Dof from original paper — contextualize with redundancy data
5. HSFA6a from cell-type analysis — motif landscape
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
GFF3 = DATA / "raw" / "Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3"
GENOME = DATA / "raw" / "B73_NAM5.fa"
PWM_DB = DATA / "processed" / "maize_tf_pwm_database.json"
BQTL = DATA / "processed" / "bqtl_snp_ww.csv"
ASE = DATA / "processed" / "engelhorn_ase_ww_gene_summary.csv"
EXPR = DATA / "processed" / "engelhorn_ww_vs_ds_expression.tsv"
ENTAP = DATA / "processed" / "entap_annotations.tsv"
OUTDIR = BASE / "results"

# Key genes from the literature / original paper
KEY_GENES = {
    'Zm00001eb359620': 'ZmTRE1 (trehalase, drought/ABA signaling)',
    'Zm00001eb160080': 'HSFA6a (heat stress TF, 30-fold drought)',
    'Zm00001eb359950': 'NAC73 (cell wall master regulator)',
    'Zm00001eb157860': 'GBF3 (bZIP, ABA signaling)',
    'Zm00001eb379070': 'CslD2 (cellulose synthase-like)',
    'Zm00001eb166210': 'Polyol transporter 5',
}


def main():
    print("=" * 70)
    print("CASE STUDY ANALYSIS")
    print("=" * 70)

    # Load full-scale results
    full_df = pd.read_csv(OUTDIR / 'quantitative_modulation_full.csv')
    print(f"Loaded {len(full_df)} triples")

    # Load ASE
    ase_df = pd.read_csv(ASE)
    ase_dict = dict(zip(ase_df['gene_id'], ase_df['mean_abs_log2']))

    # Load expression / drought
    expr_df = pd.read_csv(EXPR, sep='\t')
    expr_dict = dict(zip(expr_df['gene_id'], expr_df['mean_expression']))
    drought_dict = {}
    if 'log2_fold_change' in expr_df.columns:
        drought_dict = dict(zip(expr_df['gene_id'], expr_df['log2_fold_change']))

    # Load annotations
    annot_dict = {}
    if ENTAP.exists():
        annot_df = pd.read_csv(ENTAP, sep='\t')
        for _, row in annot_df.iterrows():
            gene_id = str(row.get('Query Sequence', ''))
            desc = str(row.get('Description', ''))
            if gene_id.startswith('Zm') and desc != 'nan':
                annot_dict[gene_id] = desc

    # Gene-level summary
    gene_level = full_df.groupby('gene_id').agg(
        n_families=('tf_family', 'nunique'),
        n_bqtl=('pos', 'nunique'),
        mean_redundancy=('n_other_motifs', 'mean'),
        families_list=('tf_family', lambda x: ','.join(sorted(set(x)))),
        abs_log2_ase=('abs_log2_ase', 'first'),
    ).reset_index()

    # ========================================
    # 1. KEY GENES FROM THE LITERATURE
    # ========================================
    print("\n" + "=" * 70)
    print("1. KEY GENES FROM THE LITERATURE")
    print("=" * 70)

    for gene_id, name in KEY_GENES.items():
        gene_data = full_df[full_df['gene_id'] == gene_id]
        if len(gene_data) == 0:
            print(f"\n  {gene_id} ({name}): NOT in promoter bQTL set")
            continue

        print(f"\n  {gene_id}: {name}")
        ase = ase_dict.get(gene_id, np.nan)
        expr = expr_dict.get(gene_id, np.nan)
        drought = drought_dict.get(gene_id, np.nan)
        print(f"    Expression: {expr:.1f} CPM, Drought log2FC: {drought:+.2f}")
        print(f"    |log2(B73/NAM)|: {ase:.4f}")
        print(f"    bQTL positions: {gene_data['pos'].nunique()}")
        print(f"    TF families disrupted: {gene_data['tf_family'].nunique()}")

        for _, row in gene_data.drop_duplicates(['pos', 'tf_family']).iterrows():
            sole_str = "SOLE" if row['is_sole'] else f"+{int(row['n_other_motifs'])} redundant"
            print(f"      chr{row['chr'].replace('chr','')}:{row['pos']} | "
                  f"{row['tf_family']:12s} | {sole_str} | {row['bqtl_type']}")

    # ========================================
    # 2. BEST BUFFERED EXAMPLES (MYB/Dof)
    # ========================================
    print("\n" + "=" * 70)
    print("2. BEST BUFFERED EXAMPLES (high MYB redundancy, low ASE)")
    print("=" * 70)

    myb_data = full_df[full_df['tf_family'].isin(['MYB', 'MYB_related', 'Dof'])]
    myb_high_red = myb_data[myb_data['n_other_motifs'] >= 8]
    myb_low_ase = myb_high_red.nsmallest(20, 'abs_log2_ase')

    print(f"\n  Genes with ≥8 additional MYB/Dof motifs and lowest ASE:")
    for _, row in myb_low_ase.iterrows():
        desc = annot_dict.get(row['gene_id'], '')[:50]
        print(f"    {row['gene_id']} | {row['tf_family']:12s} | "
              f"+{int(row['n_other_motifs'])} motifs | "
              f"|ASE|={row['abs_log2_ase']:.3f} | {desc}")

    # ========================================
    # 3. BEST SENSITIVE EXAMPLES (sole motif, high ASE)
    # ========================================
    print("\n" + "=" * 70)
    print("3. BEST SENSITIVE EXAMPLES (sole motif, high ASE)")
    print("=" * 70)

    sensitive_fams = ['SBP', 'bHLH', 'bZIP', 'WRKY', 'NAC', 'ARF']
    sens_data = full_df[(full_df['tf_family'].isin(sensitive_fams)) & full_df['is_sole']]
    sens_high_ase = sens_data.nlargest(20, 'abs_log2_ase')

    print(f"\n  Genes with SOLE sensitive-family motif and highest ASE:")
    for _, row in sens_high_ase.iterrows():
        desc = annot_dict.get(row['gene_id'], '')[:50]
        drought = drought_dict.get(row['gene_id'], np.nan)
        dr_str = f"log2FC={drought:+.1f}" if not np.isnan(drought) else ""
        print(f"    {row['gene_id']} | {row['tf_family']:5s} SOLE | "
              f"|ASE|={row['abs_log2_ase']:.3f} | {dr_str:12s} | {desc}")

    # ========================================
    # 4. COMBINATORIAL DISRUPTION CHAMPIONS
    # ========================================
    print("\n" + "=" * 70)
    print("4. COMBINATORIAL DISRUPTION: genes with most TF families disrupted")
    print("=" * 70)

    top_comb = gene_level.nlargest(20, 'n_families')
    for _, row in top_comb.iterrows():
        desc = annot_dict.get(row['gene_id'], '')[:50]
        drought = drought_dict.get(row['gene_id'], np.nan)
        dr_str = f"log2FC={drought:+.1f}" if not np.isnan(drought) else ""
        print(f"  {row['gene_id']} | {row['n_families']:2d} families | "
              f"{row['n_bqtl']:2d} bQTL | |ASE|={row['abs_log2_ase']:.3f} | "
              f"{dr_str:12s} | {desc}")

    # ========================================
    # 5. CONTRAST PAIRS: same gene, buffered vs sensitive
    # ========================================
    print("\n" + "=" * 70)
    print("5. CONTRAST PAIRS: genes with both buffered AND sensitive family motifs")
    print("=" * 70)

    buff_fams = {'MYB', 'MYB_related', 'Dof'}
    sens_fams = {'SBP', 'bHLH', 'bZIP', 'NAC', 'ARF'}

    contrast_genes = []
    for gene_id in full_df['gene_id'].unique():
        gene_data = full_df[full_df['gene_id'] == gene_id]
        fams = set(gene_data['tf_family'])
        has_buff = fams & buff_fams
        has_sens = fams & sens_fams
        if has_buff and has_sens:
            buff_red = gene_data[gene_data['tf_family'].isin(buff_fams)]['n_other_motifs'].mean()
            sens_red = gene_data[gene_data['tf_family'].isin(sens_fams)]['n_other_motifs'].mean()
            contrast_genes.append({
                'gene_id': gene_id,
                'buff_families': ','.join(sorted(has_buff)),
                'sens_families': ','.join(sorted(has_sens)),
                'buff_redundancy': buff_red,
                'sens_redundancy': sens_red,
                'ase': ase_dict.get(gene_id, np.nan),
            })

    contrast_df = pd.DataFrame(contrast_genes)
    # Best contrasts: high buff redundancy, low sens redundancy
    contrast_df['contrast_score'] = contrast_df['buff_redundancy'] - contrast_df['sens_redundancy']
    top_contrast = contrast_df.nlargest(10, 'contrast_score')

    print(f"\n  {len(contrast_df)} genes have both buffered and sensitive family disruptions")
    print(f"  Top contrasts (high buffered redundancy, low sensitive redundancy):")
    for _, row in top_contrast.iterrows():
        desc = annot_dict.get(row['gene_id'], '')[:40]
        print(f"    {row['gene_id']} | buff: {row['buff_families']:20s} (red={row['buff_redundancy']:.1f}) | "
              f"sens: {row['sens_families']:15s} (red={row['sens_redundancy']:.1f}) | "
              f"|ASE|={row['ase']:.3f} | {desc}")

    # ========================================
    # 6. DROUGHT-RESPONSIVE WITH HIGH COMBINATORIAL DISRUPTION
    # ========================================
    print("\n" + "=" * 70)
    print("6. DROUGHT-RESPONSIVE GENES WITH HIGH COMBINATORIAL DISRUPTION")
    print("=" * 70)

    gene_level['drought_fc'] = gene_level['gene_id'].map(drought_dict)
    drought_genes = gene_level[gene_level['drought_fc'].abs() > 1].sort_values(
        'n_families', ascending=False).head(20)

    for _, row in drought_genes.iterrows():
        desc = annot_dict.get(row['gene_id'], '')[:50]
        print(f"  {row['gene_id']} | {row['n_families']:2d} fam | "
              f"|ASE|={row['abs_log2_ase']:.3f} | drought={row['drought_fc']:+.2f} | {desc}")

    # ========================================
    # 7. DETAILED MOTIF LANDSCAPE FOR TOP CASE STUDIES
    # ========================================
    print("\n" + "=" * 70)
    print("7. DETAILED MOTIF LANDSCAPES")
    print("=" * 70)

    # Pick 5 best case studies
    case_study_genes = []

    # 1. Highest combinatorial + drought
    if len(drought_genes) > 0:
        case_study_genes.append(drought_genes.iloc[0]['gene_id'])

    # 2. Best MYB-buffered
    best_myb = myb_data.sort_values('n_other_motifs', ascending=False).iloc[0]
    case_study_genes.append(best_myb['gene_id'])

    # 3. Best sole sensitive
    if len(sens_high_ase) > 0:
        case_study_genes.append(sens_high_ase.iloc[0]['gene_id'])

    # Add key genes
    for g in KEY_GENES:
        if g not in case_study_genes:
            case_study_genes.append(g)

    case_study_genes = case_study_genes[:8]

    for gene_id in case_study_genes:
        gene_data = full_df[full_df['gene_id'] == gene_id]
        if len(gene_data) == 0:
            continue

        desc = annot_dict.get(gene_id, KEY_GENES.get(gene_id, ''))
        ase = ase_dict.get(gene_id, np.nan)
        expr = expr_dict.get(gene_id, np.nan)
        drought = drought_dict.get(gene_id, np.nan)

        print(f"\n  === {gene_id} ===")
        print(f"  Description: {desc}")
        print(f"  Expression: {expr:.1f} CPM | Drought: {drought:+.2f} | |ASE|: {ase:.4f}")
        print(f"  bQTL: {gene_data['pos'].nunique()} positions, "
              f"{gene_data['tf_family'].nunique()} TF families")

        # Summary table
        for fam in sorted(gene_data['tf_family'].unique()):
            fam_data = gene_data[gene_data['tf_family'] == fam]
            for _, row in fam_data.iterrows():
                sole = "SOLE" if row['is_sole'] else f"+{int(row['n_other_motifs'])}"
                print(f"    {row['chr']}:{row['pos']} | {fam:12s} | {sole:>5s} | {row['bqtl_type']}")

    # Save case study summary
    case_df = pd.DataFrame([{
        'gene_id': g,
        'description': annot_dict.get(g, KEY_GENES.get(g, '')),
        'expression': expr_dict.get(g, np.nan),
        'drought_fc': drought_dict.get(g, np.nan),
        'ase': ase_dict.get(g, np.nan),
        'n_bqtl': full_df[full_df['gene_id'] == g]['pos'].nunique(),
        'n_families': full_df[full_df['gene_id'] == g]['tf_family'].nunique(),
        'families': ','.join(sorted(full_df[full_df['gene_id'] == g]['tf_family'].unique())),
    } for g in case_study_genes if len(full_df[full_df['gene_id'] == g]) > 0])

    case_df.to_csv(OUTDIR / 'case_studies.csv', index=False)
    print(f"\n  Saved: case_studies.csv")

    # Save contrast pairs
    if len(contrast_df) > 0:
        contrast_df.to_csv(OUTDIR / 'contrast_pairs.csv', index=False)
        print(f"  Saved: contrast_pairs.csv")

    print("\n" + "=" * 70)
    print("CASE STUDY SUMMARY")
    print("=" * 70)
    print(f"  Total genes in analysis: {gene_level['gene_id'].nunique()}")
    print(f"  Genes with drought response: {len(gene_level[gene_level['drought_fc'].abs() > 1])}")
    print(f"  Genes with ≥10 families disrupted: {len(gene_level[gene_level['n_families'] >= 10])}")
    print(f"  Genes with both buffered+sensitive: {len(contrast_df)}")
    print(f"  Key genes found in data: {sum(1 for g in KEY_GENES if len(full_df[full_df['gene_id'] == g]) > 0)}/{len(KEY_GENES)}")


if __name__ == '__main__':
    main()
