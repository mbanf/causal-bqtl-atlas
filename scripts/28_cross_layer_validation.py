"""
28_cross_layer_validation.py — Cross-layer motif validation and differential analysis.

Integrates data from all three layers to answer:

  Module A — Conservation-impact scoring:
    For each causal bQTL where Layer 2 says "motif X disrupted", look up Layer 1:
    how conserved is that motif across hybrids at that gene? A disruption of a
    highly conserved motif (present in 20/23 hybrids) is more impactful than
    disruption of an already-variable one.

  Module B — WW vs DS differential occupancy:
    Compare Layer 1 results between WW and DS conditions. Which TF families
    gain or lose motif occupancy under drought? Per-gene differential analysis.

  Module C — Hybrid-specific regulatory damage:
    For each hybrid: which bQTL carry the alt allele → which TFs are disrupted →
    compute a per-hybrid "regulatory damage score" (sum of conservation-weighted
    disruptions).

Output:
  - cross_layer_conservation_impact.csv:  Per-bQTL disruption impact scores
  - cross_layer_ww_vs_ds_differential.csv: Per-gene × family WW vs DS comparison
  - cross_layer_hybrid_damage.csv:        Per-hybrid regulatory damage scores
  - cross_layer_validation.pdf:           Summary figure
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
    """Parse 'TF1(Family1);TF2(Family2)' strings."""
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
        description='Cross-layer motif validation and differential analysis')
    parser.add_argument('--ww-occupancy', type=str, default=None)
    parser.add_argument('--ds-occupancy', type=str, default=None)
    parser.add_argument('--ww-linkage', type=str, default=None)
    parser.add_argument('--ds-linkage', type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  CROSS-LAYER MOTIF VALIDATION")
    print("=" * 70)

    out_dir = config.results_dir

    # ── Load Layer 1 occupancy data ──
    ww_occ_path = Path(args.ww_occupancy) if args.ww_occupancy else out_dir / 'population_motif_occupancy.csv'
    ds_occ_path = Path(args.ds_occupancy) if args.ds_occupancy else out_dir / 'population_motif_DS_occupancy.csv'

    print("Loading Layer 1 occupancy data...")
    ww_occ = pd.read_csv(ww_occ_path)
    print(f"  WW: {len(ww_occ):,} entries ({ww_occ['gene_id'].nunique():,} genes)")

    has_ds = ds_occ_path.exists()
    if has_ds:
        ds_occ = pd.read_csv(ds_occ_path)
        print(f"  DS: {len(ds_occ):,} entries ({ds_occ['gene_id'].nunique():,} genes)")
    else:
        print(f"  DS occupancy not found ({ds_occ_path}) — skipping Module B")

    # ── Load Layer 2 linkage data ──
    ww_link_path = Path(args.ww_linkage) if args.ww_linkage else out_dir / 'hybrid_motif_linkage.csv'
    ds_link_path = out_dir / 'hybrid_motif_linkage_ds.csv'

    print("Loading Layer 2 linkage data...")
    ww_link = pd.read_csv(ww_link_path)
    print(f"  WW linkage: {len(ww_link):,} causal bQTL")

    has_ds_link = ds_link_path.exists()
    if has_ds_link:
        ds_link = pd.read_csv(ds_link_path)
        print(f"  DS linkage: {len(ds_link):,} causal bQTL")

    # ══════════════════════════════════════════════════════════════════════
    # MODULE A: Conservation-impact scoring
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  MODULE A: CONSERVATION-IMPACT SCORING")
    print("=" * 70)

    # Build lookup: (gene_id, tf_family) -> occupancy stats from Layer 1
    occ_lookup = {}
    for _, row in ww_occ.iterrows():
        key = (row['gene_id'], row['tf_family'])
        occ_lookup[key] = {
            'n_hybrids_bound': row['n_hybrids_bound'],
            'n_hybrids_with_motif': row['n_hybrids_with_motif'],
            'frac_of_bound': row['frac_of_bound'],
            'entropy': row['entropy'],
            'variable': row['variable'],
        }

    impact_rows = []
    for _, row in ww_link.iterrows():
        gene = row['gene_id']
        bqtl_id = f"{row['chr']}:{row['pos']}"

        for effect_type, col in [('disrupted', 'all_disrupted_sig'),
                                  ('created', 'all_created_sig')]:
            for tf, fam in parse_sig_motifs(row.get(col, '')):
                occ = occ_lookup.get((gene, fam), {})
                n_bound = occ.get('n_hybrids_bound', 0)
                n_with_motif = occ.get('n_hybrids_with_motif', 0)
                frac = occ.get('frac_of_bound', 0)
                entropy = occ.get('entropy', 0)

                # Conservation-impact score:
                # High score = highly conserved motif being disrupted
                # For "created": high score = motif NOT present before (low frac) now created
                if effect_type == 'disrupted':
                    impact_score = frac  # higher conservation → higher impact
                else:
                    impact_score = 1.0 - frac  # less conserved before → more novel creation

                impact_rows.append({
                    'bqtl': bqtl_id,
                    'chr': row['chr'],
                    'pos': row['pos'],
                    'gene_id': gene,
                    'abs_d': row['abs_d'],
                    'effect': effect_type,
                    'tf_name': tf,
                    'tf_family': fam,
                    'n_hybrids_bound': n_bound,
                    'n_hybrids_with_motif': n_with_motif,
                    'motif_conservation': frac,
                    'motif_entropy': entropy,
                    'impact_score': round(impact_score, 4),
                })

    impact_df = pd.DataFrame(impact_rows)
    print(f"  Total disruption/creation events with occupancy data: {len(impact_df)}")

    # Summarize
    if len(impact_df) > 0:
        disrupted = impact_df[impact_df['effect'] == 'disrupted']
        created = impact_df[impact_df['effect'] == 'created']

        print(f"\n  Disruptions: {len(disrupted)}")
        if len(disrupted) > 0:
            print(f"    Mean conservation of disrupted motifs: {disrupted['motif_conservation'].mean():.3f}")
            print(f"    Median conservation: {disrupted['motif_conservation'].median():.3f}")
            # How many disrupt highly conserved motifs (>80% of hybrids)?
            highly_conserved = (disrupted['motif_conservation'] >= 0.8).sum()
            print(f"    Highly conserved disruptions (>80%): {highly_conserved} "
                  f"({100*highly_conserved/len(disrupted):.1f}%)")

        print(f"\n  Creations: {len(created)}")
        if len(created) > 0:
            print(f"    Mean conservation of created motifs: {created['motif_conservation'].mean():.3f}")
            # Truly novel creations (motif not present in >50% before)
            novel = (created['motif_conservation'] < 0.5).sum()
            print(f"    Novel creations (<50% prior occupancy): {novel} "
                  f"({100*novel/len(created):.1f}%)")

        # Per-family breakdown
        print(f"\n  {'Family':<15} {'Disrupted':>10} {'MeanCons':>10} {'Created':>10} {'MeanCons':>10}")
        for fam in sorted(impact_df['tf_family'].unique()):
            d = disrupted[disrupted['tf_family'] == fam]
            c = created[created['tf_family'] == fam]
            d_cons = d['motif_conservation'].mean() if len(d) > 0 else 0
            c_cons = c['motif_conservation'].mean() if len(c) > 0 else 0
            if len(d) + len(c) >= 5:
                print(f"  {fam:<15} {len(d):>10} {d_cons:>10.3f} {len(c):>10} {c_cons:>10.3f}")

    impact_path = out_dir / 'cross_layer_conservation_impact.csv'
    impact_df.to_csv(impact_path, index=False)
    print(f"\n  Saved: {impact_path}")

    # ══════════════════════════════════════════════════════════════════════
    # MODULE B: WW vs DS differential occupancy
    # ══════════════════════════════════════════════════════════════════════
    diff_df = pd.DataFrame()
    if has_ds:
        print("\n" + "=" * 70)
        print("  MODULE B: WW vs DS DIFFERENTIAL OCCUPANCY")
        print("=" * 70)

        # Merge WW and DS occupancy on gene × family
        ww_sub = ww_occ[['gene_id', 'tf_family', 'n_hybrids_bound', 'n_hybrids_with_motif',
                          'frac_of_bound', 'entropy', 'variable']].copy()
        ww_sub.columns = ['gene_id', 'tf_family', 'ww_n_bound', 'ww_n_motif',
                          'ww_frac', 'ww_entropy', 'ww_variable']

        ds_sub = ds_occ[['gene_id', 'tf_family', 'n_hybrids_bound', 'n_hybrids_with_motif',
                          'frac_of_bound', 'entropy', 'variable']].copy()
        ds_sub.columns = ['gene_id', 'tf_family', 'ds_n_bound', 'ds_n_motif',
                          'ds_frac', 'ds_entropy', 'ds_variable']

        diff_df = ww_sub.merge(ds_sub, on=['gene_id', 'tf_family'], how='outer')
        diff_df = diff_df.fillna(0)

        # Differential: DS fraction - WW fraction
        diff_df['delta_frac'] = diff_df['ds_frac'] - diff_df['ww_frac']
        diff_df['delta_n_motif'] = diff_df['ds_n_motif'] - diff_df['ww_n_motif']

        # Classify
        diff_df['category'] = 'unchanged'
        diff_df.loc[diff_df['delta_frac'] > 0.2, 'category'] = 'drought_gained'
        diff_df.loc[diff_df['delta_frac'] < -0.2, 'category'] = 'drought_lost'

        print(f"  Total gene × family pairs: {len(diff_df):,}")

        # Filter to genes with decent coverage in both conditions
        well_covered = diff_df[(diff_df['ww_n_bound'] >= 5) | (diff_df['ds_n_bound'] >= 5)]
        print(f"  Well-covered (≥5 hybrids in either): {len(well_covered):,}")

        # Per-family differential summary
        print(f"\n  {'Family':<15} {'Gained':>8} {'Lost':>8} {'Unchanged':>10} {'MeanΔ':>8} {'P-value':>12}")
        family_diff_rows = []
        for fam in sorted(well_covered['tf_family'].unique()):
            fam_data = well_covered[well_covered['tf_family'] == fam]
            n_gained = (fam_data['category'] == 'drought_gained').sum()
            n_lost = (fam_data['category'] == 'drought_lost').sum()
            n_unch = (fam_data['category'] == 'unchanged').sum()
            mean_delta = fam_data['delta_frac'].mean()

            # Paired test: is the mean delta significantly different from 0?
            delta_vals = fam_data['delta_frac']
            if len(delta_vals) >= 10:
                t_stat, p_val = stats.wilcoxon(delta_vals[delta_vals != 0])
            else:
                p_val = 1.0

            family_diff_rows.append({
                'tf_family': fam,
                'n_genes': len(fam_data),
                'n_drought_gained': n_gained,
                'n_drought_lost': n_lost,
                'n_unchanged': n_unch,
                'mean_delta_frac': round(mean_delta, 4),
                'wilcoxon_p': p_val,
            })

            if n_gained + n_lost >= 5:
                print(f"  {fam:<15} {n_gained:>8} {n_lost:>8} {n_unch:>10} {mean_delta:>+8.4f} {p_val:>12.2e}")

        family_diff_summary = pd.DataFrame(family_diff_rows)

        # Top genes with largest differential
        print("\n  Top genes with largest drought-induced motif changes:")
        gene_deltas = well_covered.groupby('gene_id')['delta_frac'].agg(['mean', 'sum', 'count'])
        gene_deltas = gene_deltas[gene_deltas['count'] >= 5]
        top_gained = gene_deltas.nlargest(10, 'mean')
        top_lost = gene_deltas.nsmallest(10, 'mean')

        print(f"\n  {'Gene':<20} {'MeanΔ':>8} {'N families':>12}")
        print("  --- DROUGHT GAINED ---")
        for gene, row in top_gained.iterrows():
            print(f"  {gene:<20} {row['mean']:>+8.4f} {int(row['count']):>12}")
        print("  --- DROUGHT LOST ---")
        for gene, row in top_lost.iterrows():
            print(f"  {gene:<20} {row['mean']:>+8.4f} {int(row['count']):>12}")

        diff_path = out_dir / 'cross_layer_ww_vs_ds_differential.csv'
        diff_df.to_csv(diff_path, index=False)
        print(f"\n  Saved: {diff_path}")

        family_diff_path = out_dir / 'cross_layer_ww_vs_ds_family_summary.csv'
        family_diff_summary.to_csv(family_diff_path, index=False)
        print(f"  Saved: {family_diff_path}")

    # ══════════════════════════════════════════════════════════════════════
    # MODULE C: Hybrid-specific regulatory damage
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  MODULE C: HYBRID-SPECIFIC REGULATORY DAMAGE")
    print("=" * 70)

    hybrid_damage = defaultdict(lambda: {
        'n_bqtl_alt': 0,
        'n_disruptions': 0,
        'n_creations': 0,
        'total_impact': 0.0,
        'disrupted_families': defaultdict(int),
        'created_families': defaultdict(int),
    })

    for _, row in ww_link.iterrows():
        # Parse which hybrids carry the variant allele
        var_hybrids = set()
        if isinstance(row.get('variant_hybrids', ''), str):
            var_hybrids = set(row['variant_hybrids'].split('|'))

        ref_hybrids = set()
        if isinstance(row.get('ref_hybrids', ''), str):
            ref_hybrids = set(row['ref_hybrids'].split('|'))

        gene = row['gene_id']

        # For each variant hybrid: motifs disrupted
        disrupted = parse_sig_motifs(row.get('all_disrupted_sig', ''))
        created = parse_sig_motifs(row.get('all_created_sig', ''))

        for hyb in var_hybrids:
            hybrid_damage[hyb]['n_bqtl_alt'] += 1

            for tf, fam in disrupted:
                hybrid_damage[hyb]['n_disruptions'] += 1
                hybrid_damage[hyb]['disrupted_families'][fam] += 1
                # Weight by conservation
                occ = occ_lookup.get((gene, fam), {})
                hybrid_damage[hyb]['total_impact'] += occ.get('frac_of_bound', 0)

            for tf, fam in created:
                hybrid_damage[hyb]['n_creations'] += 1
                hybrid_damage[hyb]['created_families'][fam] += 1

    damage_rows = []
    for hyb, d in sorted(hybrid_damage.items()):
        top_disrupted = sorted(d['disrupted_families'].items(), key=lambda x: -x[1])
        top_created = sorted(d['created_families'].items(), key=lambda x: -x[1])

        damage_rows.append({
            'hybrid': hyb,
            'n_bqtl_alt_allele': d['n_bqtl_alt'],
            'n_disruptions': d['n_disruptions'],
            'n_creations': d['n_creations'],
            'total_impact_score': round(d['total_impact'], 2),
            'impact_per_bqtl': round(d['total_impact'] / max(d['n_bqtl_alt'], 1), 4),
            'top_disrupted_families': '; '.join(f"{f}({n})" for f, n in top_disrupted[:5]),
            'top_created_families': '; '.join(f"{f}({n})" for f, n in top_created[:5]),
        })

    damage_df = pd.DataFrame(damage_rows).sort_values('total_impact_score', ascending=False)
    print(f"\n  {'Hybrid':<12} {'Alt bQTL':>10} {'Disrupt':>8} {'Create':>8} {'Impact':>8} {'Per bQTL':>10}")
    for _, row in damage_df.iterrows():
        print(f"  {row['hybrid']:<12} {row['n_bqtl_alt_allele']:>10} "
              f"{row['n_disruptions']:>8} {row['n_creations']:>8} "
              f"{row['total_impact_score']:>8.1f} {row['impact_per_bqtl']:>10.4f}")

    damage_path = out_dir / 'cross_layer_hybrid_damage.csv'
    damage_df.to_csv(damage_path, index=False)
    print(f"\n  Saved: {damage_path}")

    # ══════════════════════════════════════════════════════════════════════
    # FIGURE
    # ══════════════════════════════════════════════════════════════════════
    print("\nGenerating summary figure...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n_panels = 4 if has_ds else 3
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Cross-Layer Motif Validation', fontsize=14)

        # A: Conservation distribution of disrupted vs created motifs
        ax = axes[0, 0]
        if len(impact_df) > 0:
            disrupted = impact_df[impact_df['effect'] == 'disrupted']['motif_conservation']
            created = impact_df[impact_df['effect'] == 'created']['motif_conservation']
            bins = np.linspace(0, 1, 21)
            if len(disrupted) > 0:
                ax.hist(disrupted, bins=bins, alpha=0.6, color='red', label=f'Disrupted (n={len(disrupted)})')
            if len(created) > 0:
                ax.hist(created, bins=bins, alpha=0.6, color='blue', label=f'Created (n={len(created)})')
            ax.set_xlabel('Motif conservation (frac of bound hybrids)')
            ax.set_ylabel('Number of events')
            ax.set_title('A. Conservation of disrupted/created motifs')
            ax.legend(fontsize=8)

        # B: Impact score per family
        ax = axes[0, 1]
        if len(impact_df) > 0:
            fam_impact = impact_df[impact_df['effect'] == 'disrupted'].groupby('tf_family')['impact_score'].agg(['mean', 'count'])
            fam_impact = fam_impact[fam_impact['count'] >= 5].sort_values('mean', ascending=True)
            if len(fam_impact) > 0:
                top = fam_impact.tail(20)
                ax.barh(range(len(top)), top['mean'], color='darkred')
                ax.set_yticks(range(len(top)))
                ax.set_yticklabels(top.index, fontsize=7)
                ax.set_xlabel('Mean conservation of disrupted motifs')
                ax.set_title('B. Disruption impact by TF family')

        # C: Hybrid damage ranking
        ax = axes[1, 0]
        if len(damage_df) > 0:
            ax.barh(range(len(damage_df)), damage_df['total_impact_score'], color='purple')
            ax.set_yticks(range(len(damage_df)))
            ax.set_yticklabels(damage_df['hybrid'], fontsize=7)
            ax.set_xlabel('Total regulatory damage score')
            ax.set_title('C. Per-hybrid regulatory damage')
            ax.invert_yaxis()

        # D: WW vs DS differential (or disruption vs creation scatter)
        ax = axes[1, 1]
        if has_ds and len(diff_df) > 0:
            # Family-level mean delta
            fam_delta = diff_df.groupby('tf_family')['delta_frac'].mean().sort_values()
            colors = ['coral' if v > 0 else 'steelblue' for v in fam_delta]
            ax.barh(range(len(fam_delta)), fam_delta, color=colors)
            ax.set_yticks(range(len(fam_delta)))
            ax.set_yticklabels(fam_delta.index, fontsize=7)
            ax.set_xlabel('Mean Δ occupancy (DS − WW)')
            ax.set_title('D. Drought-induced occupancy change')
            ax.axvline(0, color='black', linewidth=0.5)
            ax.invert_yaxis()
        elif len(impact_df) > 0:
            # Scatter: conservation vs abs_d for disrupted motifs
            d = impact_df[impact_df['effect'] == 'disrupted']
            if len(d) > 0:
                ax.scatter(d['abs_d'], d['motif_conservation'], alpha=0.3, s=10, c='red')
                ax.set_xlabel('|Cohen\'s d| (effect size)')
                ax.set_ylabel('Motif conservation')
                ax.set_title('D. Effect size vs motif conservation')

        plt.tight_layout()
        fig_path = config.figures_dir / 'cross_layer_validation.pdf'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")
    except ImportError:
        print("  matplotlib not available — skipping figure")

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("  CROSS-LAYER VALIDATION COMPLETE")
    if len(impact_df) > 0:
        d = impact_df[impact_df['effect'] == 'disrupted']
        print(f"  Disruptions: {len(d)}, mean conservation: {d['motif_conservation'].mean():.3f}")
        c = impact_df[impact_df['effect'] == 'created']
        print(f"  Creations: {len(c)}, mean conservation: {c['motif_conservation'].mean():.3f}")
    print(f"  Hybrids scored: {len(damage_df)}")
    if has_ds:
        wc = diff_df[(diff_df['ww_n_bound'] >= 5) | (diff_df['ds_n_bound'] >= 5)]
        n_gained = (wc['category'] == 'drought_gained').sum()
        n_lost = (wc['category'] == 'drought_lost').sum()
        print(f"  WW vs DS: {n_gained:,} gained, {n_lost:,} lost (Δfrac > 0.2)")
    print("=" * 70)


if __name__ == '__main__':
    main()
