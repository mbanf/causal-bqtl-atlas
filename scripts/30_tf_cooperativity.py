"""
30_tf_cooperativity.py — TF cooperativity network and cis-regulatory logic.

Uses Layer 1 population-level occupancy to build TF co-occupancy networks
and test for cooperative/competitive binding patterns across hybrids.

Modules:
  A. Co-occupancy network: which TF families co-occur in same promoters?
     Jaccard index + Fisher's exact test for each pair.

  B. Hybrid-variable cooperativity: pairs where co-occurrence varies across
     hybrids — some hybrids have both TFs, others have only one.
     Tests enhanceosome vs billboard: does losing one TF's motif predict
     loss of the other's binding (enhanceosome = both or neither)?

  C. Condition-specific partnerships: TF pairs that co-occur in WW but
     not DS, or vice versa.

  D. ASE-linked cooperativity: co-disruption effects. When TWO TF motifs
     are disrupted at the same bQTL, is the ASE stronger than single
     disruptions? (synergistic = enhanceosome; additive = billboard)

Output:
  - tf_cooperativity_network.csv:   All TF pair co-occurrence stats
  - tf_cooperativity_variable.csv:  Pairs with variable co-occurrence
  - tf_cooperativity_condition.csv: WW vs DS differential pairs
  - tf_cooperativity.pdf:           Summary figure
"""

import sys, warnings, functools
print = functools.partial(print, flush=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_config import config


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='TF cooperativity network and cis-regulatory logic')
    parser.add_argument('--min-genes', type=int, default=100,
                        help='Minimum genes for a TF family to be included')
    args = parser.parse_args()

    print("=" * 70)
    print("  TF COOPERATIVITY NETWORK")
    print("=" * 70)

    out_dir = config.results_dir

    # ── Load Layer 1 occupancy (WW) ──
    print("Loading Layer 1 occupancy data...")
    ww_occ = pd.read_csv(out_dir / 'population_motif_occupancy.csv')
    print(f"  WW: {len(ww_occ):,} entries")

    # Load DS if available
    ds_occ_path = out_dir / 'population_motif_DS_occupancy.csv'
    has_ds = ds_occ_path.exists()
    if has_ds:
        ds_occ = pd.read_csv(ds_occ_path)
        print(f"  DS: {len(ds_occ):,} entries")

    # Build gene → set of families with motifs (for WW)
    print("\nBuilding gene-family occupancy sets...")
    gene_families_ww = defaultdict(set)  # gene -> families present in ≥1 hybrid
    gene_family_hybrids_ww = defaultdict(lambda: defaultdict(set))  # gene -> family -> hybrids
    all_genes = set()

    for _, row in ww_occ.iterrows():
        gene = row['gene_id']
        fam = row['tf_family']
        all_genes.add(gene)
        if row['n_hybrids_with_motif'] > 0:
            gene_families_ww[gene].add(fam)
            if isinstance(row.get('hybrids_with_motif', ''), str) and row['hybrids_with_motif']:
                for hyb in row['hybrids_with_motif'].split('|'):
                    gene_family_hybrids_ww[gene][fam].add(hyb)

    # Get families with enough genes
    family_gene_counts = defaultdict(int)
    for gene, fams in gene_families_ww.items():
        for f in fams:
            family_gene_counts[f] += 1

    valid_families = sorted(f for f, c in family_gene_counts.items() if c >= args.min_genes)
    print(f"  Genes with any motif: {len(gene_families_ww):,}")
    print(f"  Families with ≥{args.min_genes} genes: {len(valid_families)}")

    n_total = len(all_genes)

    # ══════════════════════════════════════════════════════════════════════
    # MODULE A: Co-occupancy network
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  MODULE A: TF CO-OCCUPANCY NETWORK")
    print("=" * 70)

    # Build family -> gene set
    family_genes = defaultdict(set)
    for gene, fams in gene_families_ww.items():
        for f in fams:
            if f in valid_families:
                family_genes[f].add(gene)

    cooccurrence_rows = []
    for fam_a, fam_b in combinations(valid_families, 2):
        genes_a = family_genes[fam_a]
        genes_b = family_genes[fam_b]

        both = len(genes_a & genes_b)
        only_a = len(genes_a - genes_b)
        only_b = len(genes_b - genes_a)
        neither = n_total - len(genes_a | genes_b)

        # Jaccard index
        union = len(genes_a | genes_b)
        jaccard = both / union if union > 0 else 0

        # Fisher's exact test for association
        table = [[both, only_a], [only_b, neither]]
        odds_ratio, p_val = stats.fisher_exact(table)

        # Expected co-occurrence under independence
        expected = len(genes_a) * len(genes_b) / n_total
        obs_exp_ratio = both / expected if expected > 0 else 0

        cooccurrence_rows.append({
            'family_a': fam_a,
            'family_b': fam_b,
            'n_both': both,
            'n_only_a': only_a,
            'n_only_b': only_b,
            'n_neither': neither,
            'jaccard': round(jaccard, 4),
            'odds_ratio': round(odds_ratio, 4),
            'obs_exp_ratio': round(obs_exp_ratio, 4),
            'fisher_p': p_val,
        })

    cooc_df = pd.DataFrame(cooccurrence_rows)
    cooc_df['fdr'] = np.minimum(cooc_df['fisher_p'] * len(cooc_df), 1.0)
    cooc_df = cooc_df.sort_values('odds_ratio', ascending=False)

    # Top co-occurring pairs
    sig = cooc_df[cooc_df['fdr'] < 0.05]
    enriched = sig[sig['odds_ratio'] > 1].sort_values('odds_ratio', ascending=False)
    depleted = sig[sig['odds_ratio'] < 1].sort_values('odds_ratio', ascending=True)

    print(f"\n  Significant pairs (FDR < 0.05): {len(sig)} / {len(cooc_df)}")
    print(f"  Enriched (co-occur more than expected): {len(enriched)}")
    print(f"  Depleted (avoid each other): {len(depleted)}")

    print(f"\n  Top co-occurring pairs:")
    print(f"  {'Family A':<15} {'Family B':<15} {'Both':>6} {'Jaccard':>8} {'OR':>8} {'O/E':>6}")
    for _, row in enriched.head(15).iterrows():
        print(f"  {row['family_a']:<15} {row['family_b']:<15} "
              f"{row['n_both']:>6} {row['jaccard']:>8.3f} "
              f"{row['odds_ratio']:>8.2f} {row['obs_exp_ratio']:>6.2f}")

    if len(depleted) > 0:
        print(f"\n  Top avoided pairs:")
        for _, row in depleted.head(10).iterrows():
            print(f"  {row['family_a']:<15} {row['family_b']:<15} "
                  f"{row['n_both']:>6} {row['jaccard']:>8.3f} "
                  f"{row['odds_ratio']:>8.2f} {row['obs_exp_ratio']:>6.2f}")

    # ══════════════════════════════════════════════════════════════════════
    # MODULE B: Hybrid-variable cooperativity (enhanceosome vs billboard)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  MODULE B: ENHANCEOSOME vs BILLBOARD CLASSIFICATION")
    print("=" * 70)

    # For each gene with co-occurring TF pairs: across hybrids, do both TFs
    # appear/disappear together (enhanceosome) or independently (billboard)?
    variable_rows = []

    for fam_a, fam_b in combinations(valid_families, 2):
        # Genes where BOTH families have motifs in at least 1 hybrid
        shared_genes = family_genes[fam_a] & family_genes[fam_b]
        if len(shared_genes) < 50:
            continue

        # For each shared gene, compute correlation of A and B occupancy across hybrids
        concordant = 0
        discordant = 0
        total_tested = 0

        for gene in shared_genes:
            hybs_a = gene_family_hybrids_ww[gene].get(fam_a, set())
            hybs_b = gene_family_hybrids_ww[gene].get(fam_b, set())

            if not hybs_a or not hybs_b:
                continue

            # Both present in same hybrids?
            both = len(hybs_a & hybs_b)
            only_a = len(hybs_a - hybs_b)
            only_b = len(hybs_b - hybs_a)

            if both + only_a + only_b == 0:
                continue

            total_tested += 1

            # Concordance: fraction where both appear together
            concordance = both / (both + only_a + only_b)
            if concordance > 0.8:
                concordant += 1
            elif concordance < 0.2:
                discordant += 1

        if total_tested >= 20:
            pct_concordant = 100 * concordant / total_tested
            pct_discordant = 100 * discordant / total_tested

            # Classification:
            # Enhanceosome: high concordance (both or neither)
            # Billboard: low concordance (independent)
            if pct_concordant > 50:
                logic = 'enhanceosome'
            elif pct_discordant > 30:
                logic = 'competitive'
            else:
                logic = 'billboard'

            variable_rows.append({
                'family_a': fam_a,
                'family_b': fam_b,
                'n_shared_genes': len(shared_genes),
                'n_tested': total_tested,
                'n_concordant': concordant,
                'n_discordant': discordant,
                'pct_concordant': round(pct_concordant, 1),
                'pct_discordant': round(pct_discordant, 1),
                'regulatory_logic': logic,
            })

    var_df = pd.DataFrame(variable_rows)
    if len(var_df) > 0:
        logic_counts = var_df['regulatory_logic'].value_counts()
        print(f"\n  Pairs classified: {len(var_df)}")
        for logic, count in logic_counts.items():
            print(f"    {logic}: {count} ({100*count/len(var_df):.1f}%)")

        # Top enhanceosome pairs
        enhanceosome = var_df[var_df['regulatory_logic'] == 'enhanceosome'].sort_values(
            'pct_concordant', ascending=False)
        if len(enhanceosome) > 0:
            print(f"\n  Top enhanceosome pairs (both TFs appear/disappear together):")
            print(f"  {'Family A':<15} {'Family B':<15} {'Genes':>6} {'Concordant':>11}")
            for _, row in enhanceosome.head(15).iterrows():
                print(f"  {row['family_a']:<15} {row['family_b']:<15} "
                      f"{row['n_tested']:>6} {row['pct_concordant']:>10.1f}%")

        # Competitive pairs
        competitive = var_df[var_df['regulatory_logic'] == 'competitive'].sort_values(
            'pct_discordant', ascending=False)
        if len(competitive) > 0:
            print(f"\n  Top competitive pairs (one excludes the other):")
            for _, row in competitive.head(10).iterrows():
                print(f"  {row['family_a']:<15} {row['family_b']:<15} "
                      f"{row['n_tested']:>6} {row['pct_discordant']:>10.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # MODULE C: WW vs DS differential partnerships
    # ══════════════════════════════════════════════════════════════════════
    if has_ds:
        print("\n" + "=" * 70)
        print("  MODULE C: CONDITION-SPECIFIC TF PARTNERSHIPS")
        print("=" * 70)

        # Build gene-family sets for DS
        gene_families_ds = defaultdict(set)
        for _, row in ds_occ.iterrows():
            if row['n_hybrids_with_motif'] > 0:
                gene_families_ds[row['gene_id']].add(row['tf_family'])

        family_genes_ds = defaultdict(set)
        for gene, fams in gene_families_ds.items():
            for f in fams:
                if f in valid_families:
                    family_genes_ds[f].add(gene)

        # Compare co-occurrence: WW Jaccard vs DS Jaccard for each pair
        cond_rows = []
        for fam_a, fam_b in combinations(valid_families, 2):
            ww_both = len(family_genes[fam_a] & family_genes[fam_b])
            ww_union = len(family_genes[fam_a] | family_genes[fam_b])
            ww_j = ww_both / ww_union if ww_union > 0 else 0

            ds_both = len(family_genes_ds.get(fam_a, set()) & family_genes_ds.get(fam_b, set()))
            ds_union = len(family_genes_ds.get(fam_a, set()) | family_genes_ds.get(fam_b, set()))
            ds_j = ds_both / ds_union if ds_union > 0 else 0

            delta_j = ds_j - ww_j

            cond_rows.append({
                'family_a': fam_a,
                'family_b': fam_b,
                'ww_jaccard': round(ww_j, 4),
                'ds_jaccard': round(ds_j, 4),
                'delta_jaccard': round(delta_j, 4),
                'ww_n_both': ww_both,
                'ds_n_both': ds_both,
            })

        cond_df = pd.DataFrame(cond_rows).sort_values('delta_jaccard')

        # Drought-lost partnerships (WW high, DS low)
        lost = cond_df[cond_df['delta_jaccard'] < -0.01].head(15)
        gained = cond_df[cond_df['delta_jaccard'] > 0.01].sort_values('delta_jaccard', ascending=False).head(15)

        if len(lost) > 0:
            print(f"\n  Drought-LOST partnerships (co-occur in WW, not DS):")
            print(f"  {'Family A':<15} {'Family B':<15} {'WW_J':>6} {'DS_J':>6} {'ΔJ':>8}")
            for _, row in lost.iterrows():
                print(f"  {row['family_a']:<15} {row['family_b']:<15} "
                      f"{row['ww_jaccard']:>6.3f} {row['ds_jaccard']:>6.3f} "
                      f"{row['delta_jaccard']:>+8.4f}")

        if len(gained) > 0:
            print(f"\n  Drought-GAINED partnerships (co-occur in DS, not WW):")
            for _, row in gained.iterrows():
                print(f"  {row['family_a']:<15} {row['family_b']:<15} "
                      f"{row['ww_jaccard']:>6.3f} {row['ds_jaccard']:>6.3f} "
                      f"{row['delta_jaccard']:>+8.4f}")

        cond_path = out_dir / 'tf_cooperativity_condition.csv'
        cond_df.to_csv(cond_path, index=False)
        print(f"\n  Saved: {cond_path}")

    # ══════════════════════════════════════════════════════════════════════
    # MODULE D: Co-disruption synergy (multi-TF disruption ASE effects)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  MODULE D: CO-DISRUPTION SYNERGY")
    print("=" * 70)

    ase_path = out_dir / 'ase_motif_validation.csv'
    if ase_path.exists():
        ase_data = pd.read_csv(ase_path)
        alt_data = ase_data[ase_data['allele'] == 'alt'].copy()

        if len(alt_data) >= 20:
            # Single vs multi-TF disruption
            single = alt_data[alt_data['n_disrupted'] == 1]['abs_log2_ratio']
            multi = alt_data[alt_data['n_disrupted'] >= 2]['abs_log2_ratio']

            print(f"  Single-TF disruptions: n={len(single)}, mean |ASE| = {single.mean():.4f}")
            print(f"  Multi-TF disruptions:  n={len(multi)}, mean |ASE| = {multi.mean():.4f}")

            if len(single) >= 10 and len(multi) >= 10:
                u, p = stats.mannwhitneyu(multi, single, alternative='greater')
                print(f"  Multi > Single: p = {p:.2e}")

                if multi.mean() > single.mean() * 1.3:
                    print(f"  → SYNERGISTIC: multi-TF disruption is >30% stronger (enhanceosome-like)")
                elif multi.mean() > single.mean() * 1.05:
                    print(f"  → ADDITIVE: multi-TF disruption is modestly stronger (billboard-like)")
                else:
                    print(f"  → NO DOSE EFFECT: multi-TF disruption is not stronger")

            # Which family pairs show synergy when co-disrupted?
            print("\n  Co-disrupted family pairs and their ASE effects:")
            pair_effects = defaultdict(list)
            for _, row in alt_data[alt_data['n_disrupted'] >= 2].iterrows():
                fams = row['disrupted_families'].split('|') if isinstance(row.get('disrupted_families'), str) else []
                for fa, fb in combinations(sorted(set(fams)), 2):
                    pair_effects[(fa, fb)].append(row['abs_log2_ratio'])

            pair_rows = []
            for (fa, fb), ase_vals in sorted(pair_effects.items(), key=lambda x: -np.mean(x[1])):
                if len(ase_vals) >= 5:
                    pair_rows.append({
                        'family_a': fa, 'family_b': fb,
                        'n': len(ase_vals),
                        'mean_ase': round(np.mean(ase_vals), 4),
                    })

            if pair_rows:
                print(f"  {'Family A':<15} {'Family B':<15} {'N':>5} {'Mean |ASE|':>12}")
                for row in pair_rows[:15]:
                    print(f"  {row['family_a']:<15} {row['family_b']:<15} "
                          f"{row['n']:>5} {row['mean_ase']:>12.4f}")
    else:
        print("  ASE validation not found — run step 29 first")

    # ── Save outputs ──
    cooc_path = out_dir / 'tf_cooperativity_network.csv'
    cooc_df.to_csv(cooc_path, index=False)
    print(f"\nSaved: {cooc_path}")

    if len(var_df) > 0:
        var_path = out_dir / 'tf_cooperativity_variable.csv'
        var_df.to_csv(var_path, index=False)
        print(f"Saved: {var_path}")

    # ── Figure ──
    print("\nGenerating figure...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('TF Cooperativity Network', fontsize=14)

        # A: Co-occurrence heatmap (top families)
        ax = axes[0, 0]
        top_fams = sorted(valid_families)[:20]
        n_fam = len(top_fams)
        mat = np.zeros((n_fam, n_fam))
        for _, row in cooc_df.iterrows():
            if row['family_a'] in top_fams and row['family_b'] in top_fams:
                i = top_fams.index(row['family_a'])
                j = top_fams.index(row['family_b'])
                val = np.log2(row['odds_ratio']) if row['odds_ratio'] > 0 else 0
                mat[i, j] = val
                mat[j, i] = val
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-2, vmax=2)
        ax.set_xticks(range(n_fam))
        ax.set_yticks(range(n_fam))
        ax.set_xticklabels(top_fams, rotation=90, fontsize=5)
        ax.set_yticklabels(top_fams, fontsize=5)
        ax.set_title('A. TF co-occurrence (log₂ OR)')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # B: Regulatory logic classification
        ax = axes[0, 1]
        if len(var_df) > 0:
            logic_counts = var_df['regulatory_logic'].value_counts()
            ax.bar(logic_counts.index, logic_counts.values,
                   color=['forestgreen', 'orange', 'red'])
            ax.set_ylabel('Number of TF pairs')
            ax.set_title('B. Regulatory logic classification')

        # C: WW vs DS Jaccard scatter
        ax = axes[1, 0]
        if has_ds:
            ax.scatter(cond_df['ww_jaccard'], cond_df['ds_jaccard'],
                      alpha=0.3, s=10, c='teal')
            lim = max(cond_df['ww_jaccard'].max(), cond_df['ds_jaccard'].max())
            ax.plot([0, lim], [0, lim], 'r--', linewidth=0.5)
            ax.set_xlabel('WW Jaccard')
            ax.set_ylabel('DS Jaccard')
            ax.set_title('C. Co-occurrence: WW vs DS')

        # D: Concordance distribution
        ax = axes[1, 1]
        if len(var_df) > 0:
            ax.hist(var_df['pct_concordant'], bins=30, color='steelblue',
                    edgecolor='black')
            ax.set_xlabel('% genes with concordant occupancy')
            ax.set_ylabel('Number of TF pairs')
            ax.set_title('D. Concordance distribution')
            ax.axvline(50, color='red', linestyle='--',
                      label='Enhanceosome threshold')
            ax.legend(fontsize=8)

        plt.tight_layout()
        fig_path = config.figures_dir / 'tf_cooperativity.pdf'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fig_path}")
    except ImportError:
        print("  matplotlib not available — skipping figure")

    print("\n" + "=" * 70)
    print("  TF COOPERATIVITY COMPLETE")
    print(f"  Pairs tested: {len(cooc_df)}")
    if len(var_df) > 0:
        print(f"  Logic classified: {len(var_df)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
