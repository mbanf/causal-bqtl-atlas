"""
29_ase_motif_validation.py — ASE × motif conservation validation.

Closes the causal loop: does disruption of a conserved motif actually
produce stronger allele-specific expression?

For each causal bQTL × hybrid:
  1. Does this hybrid carry the variant (alt) allele?
  2. What TF motifs are disrupted/created at this bQTL? (Layer 2)
  3. How conserved is each disrupted motif across the hybrid panel? (Layer 1 via step 28)
  4. What is the actual ASE magnitude at this gene in this hybrid?

Key tests:
  A. Alt-allele hybrids show stronger ASE than ref-allele hybrids at the same gene
  B. Disruption of highly conserved motifs produces stronger ASE
  C. Per-TF family: which family disruptions drive the strongest ASE?
  D. Multi-motif disruptions: does disrupting more motifs = stronger ASE?

Output:
  - ase_motif_validation.csv:        Per-bQTL × hybrid ASE vs motif data
  - ase_motif_family_effects.csv:    Per-TF family ASE impact summary
  - ase_motif_validation.pdf:        Summary figure
"""

import sys, warnings, functools
print = functools.partial(print, flush=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_config import config


def parse_sig_motifs(val):
    if not isinstance(val, str) or not val:
        return []
    entries = []
    for e in val.split(';'):
        if e and '(' in e:
            tf = e.split('(')[0]
            fam = e.split('(')[1].rstrip(')')
            entries.append((tf, fam))
    return entries


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='ASE × motif conservation validation')
    parser.add_argument('--condition', type=str, default='WW',
                        choices=['WW', 'DS'])
    args = parser.parse_args()

    print("=" * 70)
    print(f"  ASE × MOTIF CONSERVATION VALIDATION ({args.condition})")
    print("=" * 70)

    out_dir = config.results_dir
    condition = args.condition

    # ── Load data ──
    # ASE data: per gene × hybrid
    print("Loading ASE data...")
    ase = pd.read_csv(config.ase)
    # Extract hybrid name from 'B73xNAM' format
    ase['hybrid_name'] = ase['hybrid'].str.replace('B73x', '', regex=False)
    ase_lookup = {}
    for _, row in ase.iterrows():
        ase_lookup[(row['gene_id'], row['hybrid_name'])] = {
            'log2_ratio': row['log2_ratio'],
            'abs_log2_ratio': row['abs_log2_ratio'],
            'total_reads': row['total_reads'],
        }
    print(f"  ASE records: {len(ase):,} ({ase['gene_id'].nunique():,} genes)")

    # Layer 2 linkage: per bQTL, which motifs disrupted/created, which hybrids alt/ref
    suffix = '' if condition == 'WW' else '_ds'
    linkage_path = out_dir / f'hybrid_motif_linkage{suffix}.csv'
    linkage = pd.read_csv(linkage_path)
    print(f"  Linkage: {len(linkage):,} causal bQTL")

    # Conservation-impact scores from step 28
    impact_path = out_dir / 'cross_layer_conservation_impact.csv'
    impact = pd.read_csv(impact_path)
    # Build lookup: (bqtl, tf_family, effect) -> conservation score
    impact_lookup = {}
    for _, row in impact.iterrows():
        key = (row['bqtl'], row['tf_family'], row['effect'])
        impact_lookup[key] = row['motif_conservation']
    print(f"  Impact scores: {len(impact):,} entries")

    # ══════════════════════════════════════════════════════════════════════
    # MODULE A: Alt vs Ref ASE comparison at each bQTL
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  MODULE A: ALT vs REF ASE AT CAUSAL bQTL")
    print("=" * 70)

    bqtl_ase_rows = []
    alt_ase_all = []
    ref_ase_all = []

    for _, row in linkage.iterrows():
        gene = row['gene_id']
        bqtl_id = f"{row['chr']}:{row['pos']}"

        var_hybrids = set(row['variant_hybrids'].split('|')) if isinstance(row.get('variant_hybrids'), str) else set()
        ref_hybrids = set(row['ref_hybrids'].split('|')) if isinstance(row.get('ref_hybrids'), str) else set()

        # Motif info for this bQTL
        disrupted = parse_sig_motifs(row.get('all_disrupted_sig', ''))
        created = parse_sig_motifs(row.get('all_created_sig', ''))
        n_disrupted = len(disrupted)
        n_created = len(created)
        n_total_motifs = n_disrupted + n_created

        # Get disrupted families and their conservation scores
        disrupted_families = set(fam for _, fam in disrupted)
        max_conservation = 0.0
        mean_conservation = 0.0
        if disrupted:
            cons_scores = []
            for tf, fam in disrupted:
                c = impact_lookup.get((bqtl_id, fam, 'disrupted'), 0)
                cons_scores.append(c)
            max_conservation = max(cons_scores)
            mean_conservation = np.mean(cons_scores)

        # Collect ASE for alt and ref hybrids
        alt_ase_values = []
        ref_ase_values = []

        for hyb in var_hybrids:
            ase_data = ase_lookup.get((gene, hyb))
            if ase_data and ase_data['total_reads'] >= 10:
                alt_ase_values.append(ase_data['abs_log2_ratio'])
                alt_ase_all.append(ase_data['abs_log2_ratio'])
                bqtl_ase_rows.append({
                    'bqtl': bqtl_id, 'gene_id': gene, 'hybrid': hyb,
                    'allele': 'alt', 'abs_log2_ratio': ase_data['abs_log2_ratio'],
                    'log2_ratio': ase_data['log2_ratio'],
                    'total_reads': ase_data['total_reads'],
                    'n_disrupted': n_disrupted, 'n_created': n_created,
                    'n_total_motifs': n_total_motifs,
                    'max_conservation': max_conservation,
                    'mean_conservation': mean_conservation,
                    'disrupted_families': '|'.join(sorted(disrupted_families)),
                    'abs_d': row['abs_d'],
                })

        for hyb in ref_hybrids:
            ase_data = ase_lookup.get((gene, hyb))
            if ase_data and ase_data['total_reads'] >= 10:
                ref_ase_values.append(ase_data['abs_log2_ratio'])
                ref_ase_all.append(ase_data['abs_log2_ratio'])
                bqtl_ase_rows.append({
                    'bqtl': bqtl_id, 'gene_id': gene, 'hybrid': hyb,
                    'allele': 'ref', 'abs_log2_ratio': ase_data['abs_log2_ratio'],
                    'log2_ratio': ase_data['log2_ratio'],
                    'total_reads': ase_data['total_reads'],
                    'n_disrupted': n_disrupted, 'n_created': n_created,
                    'n_total_motifs': n_total_motifs,
                    'max_conservation': max_conservation,
                    'mean_conservation': mean_conservation,
                    'disrupted_families': '|'.join(sorted(disrupted_families)),
                    'abs_d': row['abs_d'],
                })

    bqtl_ase_df = pd.DataFrame(bqtl_ase_rows)
    print(f"  bQTL × hybrid records with ASE: {len(bqtl_ase_df):,}")

    alt_ase = bqtl_ase_df[bqtl_ase_df['allele'] == 'alt']['abs_log2_ratio']
    ref_ase = bqtl_ase_df[bqtl_ase_df['allele'] == 'ref']['abs_log2_ratio']

    print(f"\n  Alt-allele hybrids: n={len(alt_ase):,}, mean |log2 ASE| = {alt_ase.mean():.4f}")
    print(f"  Ref-allele hybrids: n={len(ref_ase):,}, mean |log2 ASE| = {ref_ase.mean():.4f}")

    if len(alt_ase) >= 10 and len(ref_ase) >= 10:
        u_stat, p_val = stats.mannwhitneyu(alt_ase, ref_ase, alternative='greater')
        cohens_d = (alt_ase.mean() - ref_ase.mean()) / np.sqrt(
            (alt_ase.std()**2 + ref_ase.std()**2) / 2)
        print(f"  Mann-Whitney U (alt > ref): p = {p_val:.2e}, Cohen's d = {cohens_d:.4f}")

    # Per-bQTL paired test
    print("\n  Per-bQTL paired comparison:")
    n_alt_higher = 0
    n_ref_higher = 0
    n_tested = 0
    bqtl_effects = []

    for bqtl in bqtl_ase_df['bqtl'].unique():
        bqtl_data = bqtl_ase_df[bqtl_ase_df['bqtl'] == bqtl]
        alt_vals = bqtl_data[bqtl_data['allele'] == 'alt']['abs_log2_ratio']
        ref_vals = bqtl_data[bqtl_data['allele'] == 'ref']['abs_log2_ratio']
        if len(alt_vals) >= 3 and len(ref_vals) >= 3:
            n_tested += 1
            if alt_vals.mean() > ref_vals.mean():
                n_alt_higher += 1
            else:
                n_ref_higher += 1
            bqtl_effects.append(alt_vals.mean() - ref_vals.mean())

    if n_tested > 0:
        print(f"  bQTL tested (≥3 hybrids each side): {n_tested}")
        print(f"  Alt > Ref: {n_alt_higher} ({100*n_alt_higher/n_tested:.1f}%)")
        print(f"  Ref > Alt: {n_ref_higher} ({100*n_ref_higher/n_tested:.1f}%)")
        t_stat, t_pval = stats.wilcoxon(bqtl_effects, alternative='greater')
        print(f"  Signed-rank test (alt > ref): p = {t_pval:.2e}")
        print(f"  Mean effect (alt - ref): {np.mean(bqtl_effects):.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # MODULE B: Conservation score vs ASE magnitude
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  MODULE B: CONSERVATION × ASE CORRELATION")
    print("=" * 70)

    alt_data = bqtl_ase_df[bqtl_ase_df['allele'] == 'alt'].copy()
    if len(alt_data) >= 20:
        # Correlation: mean conservation of disrupted motifs vs ASE
        rho, p = stats.spearmanr(alt_data['mean_conservation'], alt_data['abs_log2_ratio'])
        print(f"  Spearman: conservation vs |ASE| = {rho:.4f} (p = {p:.2e})")

        # Bin by conservation
        alt_data['cons_bin'] = pd.cut(alt_data['mean_conservation'],
                                       bins=[0, 0.5, 0.8, 0.95, 1.01],
                                       labels=['<0.5', '0.5-0.8', '0.8-0.95', '>0.95'])
        print("\n  ASE by conservation bin:")
        for bin_name, group in alt_data.groupby('cons_bin', observed=True):
            print(f"    {bin_name}: n={len(group)}, mean |ASE| = {group['abs_log2_ratio'].mean():.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # MODULE C: Per-TF family ASE effects
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  MODULE C: PER-TF FAMILY ASE EFFECTS")
    print("=" * 70)

    # For each TF family: collect ASE of alt-allele hybrids where that family
    # is disrupted, vs all ref-allele hybrids at same bQTL
    family_effects = []
    for fam in sorted(set(f for fams in alt_data['disrupted_families'].dropna()
                          for f in fams.split('|') if f)):
        # bQTL where this family is disrupted
        fam_bqtl = alt_data[alt_data['disrupted_families'].str.contains(fam, na=False)]
        if len(fam_bqtl) < 10:
            continue

        # Corresponding ref alleles at same bQTL
        fam_ref = bqtl_ase_df[
            (bqtl_ase_df['allele'] == 'ref') &
            (bqtl_ase_df['bqtl'].isin(fam_bqtl['bqtl'].unique()))
        ]

        alt_mean = fam_bqtl['abs_log2_ratio'].mean()
        ref_mean = fam_ref['abs_log2_ratio'].mean() if len(fam_ref) > 0 else 0

        if len(fam_ref) >= 10:
            u, p = stats.mannwhitneyu(fam_bqtl['abs_log2_ratio'],
                                       fam_ref['abs_log2_ratio'],
                                       alternative='greater')
        else:
            p = 1.0

        family_effects.append({
            'tf_family': fam,
            'n_alt': len(fam_bqtl),
            'n_ref': len(fam_ref),
            'mean_ase_alt': round(alt_mean, 4),
            'mean_ase_ref': round(ref_mean, 4),
            'delta_ase': round(alt_mean - ref_mean, 4),
            'mannwhitney_p': p,
        })

    family_df = pd.DataFrame(family_effects).sort_values('delta_ase', ascending=False)
    print(f"\n  {'Family':<15} {'n_alt':>6} {'n_ref':>6} {'ASE_alt':>8} {'ASE_ref':>8} {'Δ ASE':>8} {'p':>12}")
    for _, row in family_df.iterrows():
        sig = '*' if row['mannwhitney_p'] < 0.05 else ' '
        print(f"  {row['tf_family']:<15} {row['n_alt']:>6} {row['n_ref']:>6} "
              f"{row['mean_ase_alt']:>8.4f} {row['mean_ase_ref']:>8.4f} "
              f"{row['delta_ase']:>+8.4f} {row['mannwhitney_p']:>12.2e}{sig}")

    # ══════════════════════════════════════════════════════════════════════
    # MODULE D: Multi-motif disruption dose-response
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  MODULE D: MULTI-MOTIF DISRUPTION DOSE-RESPONSE")
    print("=" * 70)

    if len(alt_data) >= 20:
        rho_n, p_n = stats.spearmanr(alt_data['n_disrupted'], alt_data['abs_log2_ratio'])
        print(f"  Spearman: n_disrupted vs |ASE| = {rho_n:.4f} (p = {p_n:.2e})")

        rho_t, p_t = stats.spearmanr(alt_data['n_total_motifs'], alt_data['abs_log2_ratio'])
        print(f"  Spearman: n_total_motifs vs |ASE| = {rho_t:.4f} (p = {p_t:.2e})")

        # Bin by number of disrupted motifs
        print("\n  ASE by number of disrupted motifs:")
        for n in sorted(alt_data['n_disrupted'].unique()):
            group = alt_data[alt_data['n_disrupted'] == n]
            if len(group) >= 5:
                print(f"    n_disrupted={n}: n={len(group)}, mean |ASE| = {group['abs_log2_ratio'].mean():.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # MODULE E: Effect size validation (Cohen's d from bQTL vs ASE)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  MODULE E: COHEN'S D vs ASE MAGNITUDE")
    print("=" * 70)

    if len(alt_data) >= 20:
        rho_d, p_d = stats.spearmanr(alt_data['abs_d'], alt_data['abs_log2_ratio'])
        print(f"  Spearman: |Cohen's d| vs |ASE| = {rho_d:.4f} (p = {p_d:.2e})")

        # Combined: conservation × Cohen's d interaction
        alt_data['impact_x_d'] = alt_data['mean_conservation'] * alt_data['abs_d']
        rho_comb, p_comb = stats.spearmanr(alt_data['impact_x_d'], alt_data['abs_log2_ratio'])
        print(f"  Spearman: conservation × |d| vs |ASE| = {rho_comb:.4f} (p = {p_comb:.2e})")

    # ── Save outputs ──
    bqtl_ase_path = out_dir / 'ase_motif_validation.csv'
    bqtl_ase_df.to_csv(bqtl_ase_path, index=False)
    print(f"\n  Saved: {bqtl_ase_path}")

    family_path = out_dir / 'ase_motif_family_effects.csv'
    family_df.to_csv(family_path, index=False)
    print(f"  Saved: {family_path}")

    # ── Figure ──
    print("\nGenerating figure...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'ASE × Motif Conservation Validation ({condition})', fontsize=14)

        # A: Alt vs Ref ASE distribution
        ax = axes[0, 0]
        bins = np.linspace(0, 2, 41)
        ax.hist(alt_ase, bins=bins, alpha=0.6, color='red', density=True,
                label=f'Alt (n={len(alt_ase)})')
        ax.hist(ref_ase, bins=bins, alpha=0.6, color='blue', density=True,
                label=f'Ref (n={len(ref_ase)})')
        ax.set_xlabel('|log₂ ASE ratio|')
        ax.set_ylabel('Density')
        ax.set_title('A. ASE: alt vs ref allele')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 2)

        # B: Conservation vs ASE scatter
        ax = axes[0, 1]
        if len(alt_data) > 0:
            ax.scatter(alt_data['mean_conservation'], alt_data['abs_log2_ratio'],
                      alpha=0.1, s=5, c='red')
            ax.set_xlabel('Motif conservation (frac of bound hybrids)')
            ax.set_ylabel('|log₂ ASE ratio|')
            ax.set_title(f'B. Conservation vs ASE (ρ={rho:.3f})')

        # C: Per-family ASE delta
        ax = axes[0, 2]
        if len(family_df) > 0:
            top_fam = family_df.head(20)
            colors = ['darkred' if p < 0.05 else 'gray' for p in top_fam['mannwhitney_p']]
            ax.barh(range(len(top_fam)), top_fam['delta_ase'], color=colors)
            ax.set_yticks(range(len(top_fam)))
            ax.set_yticklabels(top_fam['tf_family'], fontsize=7)
            ax.set_xlabel('Δ|ASE| (alt − ref)')
            ax.set_title('C. TF family disruption → ASE effect')
            ax.axvline(0, color='black', linewidth=0.5)
            ax.invert_yaxis()

        # D: Dose-response (n_disrupted vs ASE)
        ax = axes[1, 0]
        if len(alt_data) > 0:
            dose = alt_data.groupby('n_disrupted')['abs_log2_ratio'].agg(['mean', 'sem', 'count'])
            dose = dose[dose['count'] >= 5]
            ax.bar(dose.index, dose['mean'], yerr=dose['sem'], color='coral',
                   capsize=3, edgecolor='black')
            ax.set_xlabel('Number of disrupted motifs')
            ax.set_ylabel('Mean |log₂ ASE ratio|')
            ax.set_title(f'D. Dose-response (ρ={rho_n:.3f})')

        # E: Cohen's d vs ASE
        ax = axes[1, 1]
        if len(alt_data) > 0:
            ax.scatter(alt_data['abs_d'], alt_data['abs_log2_ratio'],
                      alpha=0.1, s=5, c='purple')
            ax.set_xlabel("|Cohen's d| (binding effect)")
            ax.set_ylabel('|log₂ ASE ratio|')
            ax.set_title(f"E. Binding effect vs expression (ρ={rho_d:.3f})")

        # F: Per-bQTL effect distribution
        ax = axes[1, 2]
        if bqtl_effects:
            ax.hist(bqtl_effects, bins=50, color='teal', edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=1)
            ax.axvline(np.mean(bqtl_effects), color='orange', linestyle='-',
                      label=f'Mean: {np.mean(bqtl_effects):.4f}')
            ax.set_xlabel('Δ|ASE| per bQTL (alt − ref)')
            ax.set_ylabel('Number of bQTL')
            ax.set_title(f'F. Per-bQTL ASE difference (n={n_tested})')
            ax.legend(fontsize=8)

        plt.tight_layout()
        fig_path = config.figures_dir / 'ase_motif_validation.pdf'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")
    except ImportError:
        print("  matplotlib not available — skipping figure")

    # ── Final summary ──
    print("\n" + "=" * 70)
    print(f"  ASE × MOTIF VALIDATION COMPLETE")
    print(f"  bQTL × hybrid records: {len(bqtl_ase_df):,}")
    if len(alt_ase) >= 10 and len(ref_ase) >= 10:
        print(f"  Alt vs Ref ASE: alt={alt_ase.mean():.4f}, ref={ref_ase.mean():.4f}, p={p_val:.2e}")
    if n_tested > 0:
        print(f"  Per-bQTL: {n_alt_higher}/{n_tested} ({100*n_alt_higher/n_tested:.1f}%) alt > ref")
    print(f"  TF families tested: {len(family_df)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
