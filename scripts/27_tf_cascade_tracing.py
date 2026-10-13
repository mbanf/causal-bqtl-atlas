"""
27_tf_cascade_tracing.py — Layer 3: TF cascade tracing from causal bQTL.

Traces the downstream effects of causal bQTL through TF regulatory cascades:

  Level 0: Causal bQTL disrupts/creates TF_A motif at target gene promoter
  Level 1: If target gene IS a TF (TF_B), count all genes where TF_B has
           motifs in their promoters (from Layer 1 occupancy)
  Level 2+: Further cascade depth (summary counts only, not individual edges)

This builds a cascade graph: bQTL → TF_A motif disrupted → gene = TF_B →
TF_B targets → ...

The cascade reveals:
  - Amplification: a single bQTL can affect hundreds of downstream genes
  - Hub vulnerability: TF genes that are targets of many causal bQTL
  - Cascade depth: how deep the regulatory effects propagate
  - Cross-condition cascades: WW vs DS specific cascade paths

Strategy: Level 0 edges are stored individually. Deeper cascade levels store
only per-bQTL summary counts (n_targets, amplification) to avoid combinatorial
explosion (78M+ edges at depth 2 with ~34K genes per TF family).

Output:
  - tf_cascade_edges.csv:        Level 0 edges (direct bQTL → TF → gene)
  - tf_cascade_summary.csv:      Per-bQTL cascade statistics (all depths)
  - tf_cascade_hubs.csv:         TF hub genes (most downstream targets)
  - tf_cascade_network.pdf:      Visualization
"""

import sys, json, warnings, functools
print = functools.partial(print, flush=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_config import config


def parse_sig_motifs(val):
    """Parse 'TF1(Family1);TF2(Family2)' strings from linkage data."""
    if not isinstance(val, str) or not val:
        return []
    entries = []
    for e in val.split(';'):
        if e and '(' in e:
            tf = e.split('(')[0]
            fam = e.split('(')[1].rstrip(')')
            entries.append((tf, fam))
    return entries


def load_tf_gene_mapping(config):
    """Identify which genes are TFs using EntAP annotations.
    Returns dict of gene_id -> tf_family (or None)."""
    try:
        entap = pd.read_csv(config.entap_gz, sep='\t', compression='gzip')
    except FileNotFoundError:
        entap = pd.read_csv(config.entap, sep='\t')
    entap['gene_id'] = entap['Query Sequence'].str.replace(r'_T\d+', '', regex=True)

    tf_keywords = {
        'WRKY': ['wrky'],
        'MYB': ['myb domain', 'myb-related', 'myb transcription'],
        'NAC': ['nac domain', 'nac transcription'],
        'bZIP': ['bzip', 'basic leucine zipper'],
        'bHLH': ['bhlh', 'basic helix-loop-helix'],
        'ERF': ['erf', 'ethylene-responsive', 'ap2/erf'],
        'C2H2': ['c2h2', 'zinc finger protein'],
        'HD-ZIP': ['hd-zip', 'homeobox-leucine'],
        'Dof': ['dof zinc', 'dof domain'],
        'TCP': ['tcp'],
        'HSF': ['heat shock factor'],
        'GATA': ['gata'],
        'MADS': ['mads'],
        'Trihelix': ['trihelix'],
        'SBP': ['sbp', 'squamosa'],
        'G2-like': ['g2-like', 'glk', 'golden2'],
        'B3': ['b3 domain'],
        'ARF': ['auxin response factor'],
    }

    desc_cols = ['Description', 'EggNOG Description']
    gene_tf_family = {}

    for _, row in entap.iterrows():
        gene_id = row['gene_id']
        if gene_id in gene_tf_family:
            continue
        desc = ' '.join(str(row.get(c, '')) for c in desc_cols if c in row.index).lower()
        for fam, kws in tf_keywords.items():
            if any(kw in desc for kw in kws):
                gene_tf_family[gene_id] = fam
                break

    return gene_tf_family


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Layer 3: TF cascade tracing from causal bQTL')
    parser.add_argument('--linkage', type=str, default=None)
    parser.add_argument('--occupancy', type=str, default=None,
                        help='Layer 1 occupancy CSV (for downstream target lookup)')
    parser.add_argument('--max-depth', type=int, default=3,
                        help='Maximum cascade depth (default: 3)')
    parser.add_argument('--label', type=str, default='WW')
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()

    suffix = args.suffix

    print("=" * 70)
    print(f"  LAYER 3: TF CASCADE TRACING ({args.label})")
    print("=" * 70)

    # Load hybrid motif linkage (Level 0 edges)
    linkage_path = Path(args.linkage) if args.linkage else config.results_dir / 'hybrid_motif_linkage.csv'
    linkage = pd.read_csv(linkage_path)
    print(f"Linkage data: {len(linkage)} causal bQTL")

    # Load TF gene mapping
    print("Identifying TF genes from annotations...")
    gene_tf_family = load_tf_gene_mapping(config)
    print(f"  TF genes identified: {len(gene_tf_family)}")

    # Load Layer 1 occupancy data (for downstream target lookup)
    occ_path = Path(args.occupancy) if args.occupancy else config.results_dir / 'population_motif_occupancy.csv'
    if occ_path.exists():
        occ = pd.read_csv(occ_path)
        # Build TF family -> set of target genes (genes with that family's motif in promoter)
        family_targets = defaultdict(set)
        for _, row in occ.iterrows():
            if row['n_hybrids_with_motif'] > 0:
                family_targets[row['tf_family']].add(row['gene_id'])
        print(f"  Layer 1 occupancy: {len(occ)} entries, {len(family_targets)} families")
        has_occupancy = True
    else:
        print(f"  Layer 1 occupancy not found ({occ_path}) — using linkage data only")
        family_targets = defaultdict(set)
        has_occupancy = False

    # If no occupancy data, build family_targets from linkage data itself
    # (which genes have which TF family motifs in their promoter, from the
    # motif scan done in 18b)
    if not has_occupancy:
        # From linkage: all_disrupted_sig and all_created_sig tell us which
        # TF families have motifs at each gene
        for _, row in linkage.iterrows():
            gene = row['gene_id']
            for _, fam in parse_sig_motifs(row.get('all_disrupted_sig', '')):
                family_targets[fam].add(gene)
            for _, fam in parse_sig_motifs(row.get('all_created_sig', '')):
                family_targets[fam].add(gene)

    # Also load the motif content data from step 01 for broader coverage
    motif_content_path = config.results_dir / 'bound_region_motifs_147942.csv'
    if motif_content_path.exists():
        mc = pd.read_csv(motif_content_path, usecols=['gene_id', 'tf_family'])
        mc = mc.dropna(subset=['gene_id', 'tf_family'])
        for _, row in mc.iterrows():
            family_targets[row['tf_family']].add(row['gene_id'])
        print(f"  Motif content data: {len(mc)} entries")

    for fam, targets in sorted(family_targets.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"    {fam}: {len(targets)} target genes")

    # ── Build Level 0 edges ──
    print("\nBuilding Level 0 edges (bQTL → disrupted TF → target gene)...")

    level0_edges = []
    for _, row in linkage.iterrows():
        gene = row['gene_id']
        bqtl_id = f"{row['chr']}:{row['pos']}"

        for tf, fam in parse_sig_motifs(row.get('all_disrupted_sig', '')):
            level0_edges.append({
                'bqtl': bqtl_id, 'chr': row['chr'], 'pos': row['pos'],
                'source_tf': tf, 'source_family': fam,
                'target_gene': gene, 'effect': 'disrupted',
                'level': 0, 'abs_d': row['abs_d'],
            })
        for tf, fam in parse_sig_motifs(row.get('all_created_sig', '')):
            level0_edges.append({
                'bqtl': bqtl_id, 'chr': row['chr'], 'pos': row['pos'],
                'source_tf': tf, 'source_family': fam,
                'target_gene': gene, 'effect': 'created',
                'level': 0, 'abs_d': row['abs_d'],
            })

    print(f"  Level 0 edges: {len(level0_edges)}")

    # ── Cascade: trace downstream through TF genes (summary-only for depth > 0) ──
    # Level 0 edges stored individually. Deeper levels: we count unique downstream
    # targets per bQTL without materializing all edges (avoids 78M+ row explosion).

    edges_df = pd.DataFrame(level0_edges)

    # For each bQTL, trace the cascade and compute summary stats
    print("\nTracing cascades (summary counts, max depth={})...".format(args.max_depth))

    cascade_stats = []
    level_edge_counts = Counter()  # level -> total edges across all bQTL
    level_edge_counts[0] = len(level0_edges)

    for bqtl, group in edges_df.groupby('bqtl'):
        n_direct = len(group)

        # Collect all unique targets across cascade levels
        all_targets = set(group['target_gene'])
        current_genes = set(group['target_gene'])
        visited_tf = set()
        n_tf_intermediaries = 0
        max_depth_reached = 0

        for depth in range(1, args.max_depth + 1):
            # Find TF genes in current frontier
            next_targets = set()
            for gene in current_genes:
                if gene in gene_tf_family:
                    tf_fam = gene_tf_family[gene]
                    key = (gene, tf_fam)
                    if key not in visited_tf:
                        visited_tf.add(key)
                        n_tf_intermediaries += 1
                        downstream = family_targets.get(tf_fam, set()) - {gene}
                        new = downstream - all_targets
                        next_targets |= new
                        level_edge_counts[depth] += len(downstream)

            if not next_targets:
                break

            all_targets |= next_targets
            current_genes = next_targets
            max_depth_reached = depth

        cascade_stats.append({
            'bqtl': bqtl,
            'chr': group['chr'].iloc[0],
            'pos': group['pos'].iloc[0],
            'abs_d': group['abs_d'].iloc[0],
            'n_direct_targets': n_direct,
            'n_cascade_targets': len(all_targets) - n_direct,
            'n_total_unique_targets': len(all_targets),
            'max_cascade_depth': max_depth_reached,
            'n_tf_intermediaries': n_tf_intermediaries,
            'amplification_factor': len(all_targets) / max(n_direct, 1),
        })

    stats_df = pd.DataFrame(cascade_stats).sort_values('n_total_unique_targets', ascending=False)

    for level, count in sorted(level_edge_counts.items()):
        print(f"  Level {level}: {count:,} edges")

    print(f"\n{'bQTL':<20} {'Direct':>8} {'Cascade':>10} {'Total':>8} {'Depth':>6} {'Amplif':>8}")
    for _, row in stats_df.head(20).iterrows():
        print(f"{row['bqtl']:<20} {row['n_direct_targets']:>8} "
              f"{row['n_cascade_targets']:>10} {row['n_total_unique_targets']:>8} "
              f"{row['max_cascade_depth']:>6} {row['amplification_factor']:>8.1f}x")

    # ── TF hub analysis ──
    print("\nTF hub analysis (genes that are both TF targets of bQTL AND TFs themselves):")
    hub_rows = []
    tf_gene_edges = edges_df[edges_df['target_gene'].isin(gene_tf_family)]

    for gene in tf_gene_edges['target_gene'].unique():
        tf_fam = gene_tf_family[gene]
        gene_edges = tf_gene_edges[tf_gene_edges['target_gene'] == gene]
        n_bqtl_targeting = gene_edges['bqtl'].nunique()
        n_downstream = len(family_targets.get(tf_fam, set()))
        disrupting_families = set()
        for _, e in gene_edges.iterrows():
            disrupting_families.add(e['source_family'])

        hub_rows.append({
            'gene_id': gene,
            'tf_family': tf_fam,
            'n_bqtl_targeting': n_bqtl_targeting,
            'n_downstream_targets': n_downstream,
            'disrupting_families': '|'.join(sorted(disrupting_families)),
            'cascade_reach': n_bqtl_targeting * n_downstream,
        })

    hubs_df = pd.DataFrame(hub_rows).sort_values('cascade_reach', ascending=False)
    print(f"\nTF hubs: {len(hubs_df)} TF genes targeted by causal bQTL")
    print(f"\n{'Gene':<20} {'Family':<12} {'bQTL':>6} {'Downstream':>12} {'Reach':>10}")
    for _, row in hubs_df.head(20).iterrows():
        print(f"{row['gene_id']:<20} {row['tf_family']:<12} "
              f"{row['n_bqtl_targeting']:>6} {row['n_downstream_targets']:>12} "
              f"{row['cascade_reach']:>10}")

    # ── Autoregulatory loops ──
    print("\nAutoregulatory loops (TF motif disrupted at its own gene's promoter):")
    auto_loops = []
    seen_auto = set()
    for _, edge in edges_df.iterrows():
        target = edge['target_gene']
        if target in gene_tf_family:
            target_fam = gene_tf_family[target]
            if target_fam == edge['source_family']:
                key = (edge['bqtl'], target, edge['effect'])
                if key not in seen_auto:
                    seen_auto.add(key)
                    auto_loops.append({
                        'bqtl': edge['bqtl'],
                        'gene_id': target,
                        'tf_family': target_fam,
                        'effect': edge['effect'],
                        'abs_d': edge['abs_d'],
                    })

    if auto_loops:
        auto_df = pd.DataFrame(auto_loops)
        print(f"  {len(auto_df)} autoregulatory loops found")
        for _, row in auto_df.iterrows():
            print(f"    {row['bqtl']}: {row['gene_id']} ({row['tf_family']}) — {row['effect']}")
    else:
        print("  None found")

    # ── Save outputs ──
    out_dir = config.results_dir
    prefix = 'tf_cascade'

    edges_path = out_dir / f'{prefix}_edges{suffix}.csv'
    edges_df.to_csv(edges_path, index=False)
    print(f"\nSaved: {edges_path}")

    stats_path = out_dir / f'{prefix}_summary{suffix}.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved: {stats_path}")

    hubs_path = out_dir / f'{prefix}_hubs{suffix}.csv'
    hubs_df.to_csv(hubs_path, index=False)
    print(f"Saved: {hubs_path}")

    # ── Summary figure ──
    print("\nGenerating cascade figure...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'TF Cascade Tracing from Causal bQTL ({args.label})', fontsize=14)

        # A: Cascade depth distribution
        ax = axes[0, 0]
        depth_counts = stats_df['max_cascade_depth'].value_counts().sort_index()
        ax.bar(depth_counts.index, depth_counts.values, color='steelblue')
        ax.set_xlabel('Maximum cascade depth')
        ax.set_ylabel('Number of bQTL')
        ax.set_title('A. Cascade depth distribution')

        # B: Amplification factor distribution
        ax = axes[0, 1]
        amp = stats_df['amplification_factor']
        ax.hist(amp[amp > 1], bins=50, color='coral', edgecolor='black')
        ax.set_xlabel('Amplification factor (total / direct targets)')
        ax.set_ylabel('Number of bQTL')
        ax.set_title('B. Cascade amplification')
        ax.axvline(amp.median(), color='red', linestyle='--',
                   label=f'Median: {amp.median():.1f}x')
        ax.legend()

        # C: Top TF hubs
        ax = axes[1, 0]
        top_hubs = hubs_df.head(15)
        if len(top_hubs) > 0:
            labels = [f"{r['gene_id'][:15]}\n({r['tf_family']})" for _, r in top_hubs.iterrows()]
            ax.barh(range(len(top_hubs)), top_hubs['cascade_reach'], color='purple')
            ax.set_yticks(range(len(top_hubs)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel('Cascade reach (bQTL × downstream)')
            ax.set_title('C. Top TF hubs')
            ax.invert_yaxis()

        # D: Edge counts by level
        ax = axes[1, 1]
        levels = sorted(level_edge_counts.keys())
        counts = [level_edge_counts[l] for l in levels]
        ax.bar(levels, counts, color='teal')
        ax.set_xlabel('Cascade level')
        ax.set_ylabel('Number of edges')
        ax.set_title('D. Edges per cascade level')
        ax.set_yscale('log')

        plt.tight_layout()
        fig_path = config.figures_dir / f'{prefix}_network{suffix}.pdf'
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig_path}")
    except ImportError:
        print("  matplotlib not available — skipping figure")

    # ── Final summary ──
    print("\n" + "=" * 70)
    print(f"  LAYER 3 COMPLETE")
    for level, count in sorted(level_edge_counts.items()):
        label = 'direct' if level == 0 else 'cascade'
        print(f"  Level {level} ({label}): {count:,} edges")
    print(f"  Unique bQTL with cascades: {stats_df[stats_df['max_cascade_depth'] > 0]['bqtl'].nunique()}")
    print(f"  TF hubs: {len(hubs_df)}")
    if auto_loops:
        print(f"  Autoregulatory loops: {len(auto_loops)}")
    cascade_bqtl = stats_df[stats_df['max_cascade_depth'] > 0]
    if len(cascade_bqtl) > 0:
        print(f"  Median amplification (cascade bQTL only): {cascade_bqtl['amplification_factor'].median():.1f}x")
    print("=" * 70)


if __name__ == '__main__':
    main()
