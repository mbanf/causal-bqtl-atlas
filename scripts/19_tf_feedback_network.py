"""
19_tf_feedback_network.py — TF enrichment, feedback loops, and network visualization
at causal bQTL.

Analyses:
  1. Per-TF and per-family enrichment: which TFs are over-represented as drivers
     at causal bQTL (disrupted + created), compared to null expectation from PWM database?
  2. Autoregulatory feedback: are TF genes themselves targets of causal bQTL?
     Identify loops: TF_A motif disrupted/created → target gene = TF_B (or TF_A itself)
  3. Network visualization: rewiring diagram, feedback loops, TF hierarchy
"""

import sys, json, warnings, gzip
import functools
print = functools.partial(print, flush=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_config import config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey


def parse_sig(val):
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


def load_tf_genes(config):
    """Load TF gene annotations from EntAP."""
    entap = pd.read_csv(config.entap_gz, sep='\t', compression='gzip')
    entap['gene_id'] = entap['Query Sequence'].str.replace(r'_T\d+', '', regex=True)

    # Identify TF genes by keyword in descriptions
    tf_keywords = [
        'transcription factor', 'zinc finger', 'MYB', 'WRKY', 'NAC',
        'bZIP', 'bHLH', 'ERF', 'ethylene-responsive', 'AP2',
        'homeobox', 'HD-ZIP', 'GATA', 'TCP', 'HSF', 'heat shock factor',
        'MADS', 'C2H2', 'Dof', 'trihelix', 'SBP', 'G2-like', 'GLK',
        'B3 domain', 'auxin response factor', 'ARF',
    ]

    desc_cols = ['Description', 'EggNOG Description']
    mask = pd.Series(False, index=entap.index)
    for col in desc_cols:
        if col in entap.columns:
            for kw in tf_keywords:
                mask |= entap[col].str.contains(kw, case=False, na=False)

    tf_entap = entap[mask][['gene_id'] + [c for c in desc_cols if c in entap.columns]].drop_duplicates('gene_id')
    return tf_entap


def assign_tf_family(description):
    """Assign a TF family from EntAP description text."""
    desc = str(description).lower()
    family_keywords = {
        'WRKY': ['wrky'],
        'MYB': ['myb'],
        'NAC': ['nac domain', 'nac transcription', 'nam,'],
        'bZIP': ['bzip', 'basic leucine zipper'],
        'bHLH': ['bhlh', 'basic helix-loop-helix'],
        'ERF': ['erf', 'ethylene-responsive', 'ap2/erf'],
        'AP2': ['ap2 domain', 'apetala2'],
        'C2H2': ['c2h2', 'zinc finger protein'],
        'HD-ZIP': ['hd-zip', 'homeobox-leucine', 'homeodomain-leucine'],
        'GATA': ['gata'],
        'TCP': ['tcp'],
        'HSF': ['heat shock factor', 'hsf'],
        'MADS': ['mads'],
        'Dof': ['dof zinc', 'dof domain'],
        'Trihelix': ['trihelix'],
        'SBP': ['sbp', 'squamosa'],
        'G2-like': ['g2-like', 'glk', 'golden2'],
        'B3': ['b3 domain'],
        'ARF': ['auxin response factor', 'arf'],
        'HB-other': ['homeobox'],
    }
    for fam, keywords in family_keywords.items():
        for kw in keywords:
            if kw in desc:
                return fam
    return 'Unknown'


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--linkage', type=str, default=None,
                        help='Motif linkage CSV (default: hybrid_motif_linkage.csv)')
    parser.add_argument('--causal', type=str, default=None,
                        help='Causal bQTL CSV (default: causal_bqtl_728.csv)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for output files (e.g. _ds)')
    parser.add_argument('--label', type=str, default='WW')
    args = parser.parse_args()

    suffix = args.suffix

    print("=" * 70)
    print(f"  TF ENRICHMENT, FEEDBACK LOOPS & NETWORK VISUALIZATION ({args.label})")
    print("=" * 70)

    # Load data
    linkage_path = Path(args.linkage) if args.linkage else config.results_dir / 'hybrid_motif_linkage.csv'
    causal_path = Path(args.causal) if args.causal else config.results_dir / 'causal_bqtl_728.csv'
    linkage = pd.read_csv(linkage_path)
    expr = pd.read_csv(config.expression, sep='\t')
    causal = pd.read_csv(causal_path)

    print(f"Causal bQTL: {len(linkage)}")
    print(f"Expression data: {len(expr)} genes")

    # Load PWM database for null expectation
    db = json.load(open(config.pwm_db))
    pwm_tfs = [(m['tf_name'], m['tf_family']) for m in db['motifs']]
    n_pwms = len(pwm_tfs)
    pwm_family_counts = Counter(f for _, f in pwm_tfs)
    print(f"PWM database: {n_pwms} motifs, {len(pwm_family_counts)} families")

    # ── Build edges ──
    edges = []
    for _, r in linkage.iterrows():
        gene = r['gene_id']
        for tf, fam in parse_sig(r.get('all_disrupted_sig', '')):
            edges.append({'tf_name': tf, 'tf_family': fam, 'target_gene': gene,
                          'effect_type': 'disrupted', 'abs_d': r['abs_d'],
                          'chr': r['chr'], 'pos': r['pos']})
        for tf, fam in parse_sig(r.get('all_created_sig', '')):
            edges.append({'tf_name': tf, 'tf_family': fam, 'target_gene': gene,
                          'effect_type': 'created', 'abs_d': r['abs_d'],
                          'chr': r['chr'], 'pos': r['pos']})

    edges_df = pd.DataFrame(edges)
    n_total_edges = len(edges_df)
    print(f"Total edges: {n_total_edges}")

    # ══════════════════════════════════════════════════════════════════
    # 1. TF ENRICHMENT ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  1. TF FAMILY ENRICHMENT (observed vs expected from PWM database)")
    print("=" * 70)

    # Expected: if all PWMs were equally likely to be disrupted/created,
    # each family's share = n_pwms_in_family / total_pwms
    # Observed: actual counts from edges

    fam_obs = edges_df.groupby('tf_family').size()
    total_obs = fam_obs.sum()

    print(f"\n{'Family':<18} {'Observed':>10} {'Expected':>10} {'Obs/Exp':>10} {'p (binom)':>12} {'Sig':>5}")
    enrichment_results = []
    for fam in sorted(pwm_family_counts.keys()):
        n_pwm = pwm_family_counts[fam]
        expected_frac = n_pwm / n_pwms
        expected_count = expected_frac * total_obs
        observed = fam_obs.get(fam, 0)
        # Binomial test
        p = stats.binomtest(observed, total_obs, expected_frac).pvalue
        ratio = observed / max(expected_count, 1)
        sig = '**' if p < 0.001 else ('*' if p < 0.05 else '')
        print(f"{fam:<18} {observed:>10} {expected_count:>10.1f} {ratio:>10.2f} {p:>12.2e} {sig:>5}")
        enrichment_results.append({
            'family': fam, 'observed': observed, 'expected': expected_count,
            'obs_exp_ratio': ratio, 'p_binomial': p,
            'n_pwms': n_pwm, 'direction': 'enriched' if ratio > 1 else 'depleted'
        })

    # Per-TF enrichment (top individuals)
    print(f"\n── Top individual TFs by target count ──")
    tf_obs = edges_df.groupby(['tf_name', 'tf_family']).agg(
        n_targets=('target_gene', 'nunique'),
        n_disrupted=('effect_type', lambda x: (x == 'disrupted').sum()),
        n_created=('effect_type', lambda x: (x == 'created').sum()),
    ).sort_values('n_targets', ascending=False)

    print(f"{'TF':<20} {'Family':<15} {'Targets':>8} {'Disrupted':>10} {'Created':>10} {'D/C ratio':>10}")
    for (tf, fam), row in tf_obs.head(25).iterrows():
        dc_ratio = row['n_disrupted'] / max(row['n_created'], 1)
        print(f"{tf:<20} {fam:<15} {row['n_targets']:>8} {row['n_disrupted']:>10} {row['n_created']:>10} {dc_ratio:>10.2f}")

    # Disruption vs creation bias per family
    print(f"\n── Disruption/Creation bias per family ──")
    fam_dc = edges_df.groupby(['tf_family', 'effect_type']).size().unstack(fill_value=0)
    if 'disrupted' in fam_dc.columns and 'created' in fam_dc.columns:
        fam_dc['total'] = fam_dc['disrupted'] + fam_dc['created']
        fam_dc['disrupt_frac'] = fam_dc['disrupted'] / fam_dc['total']
        fam_dc = fam_dc.sort_values('total', ascending=False)
        print(f"{'Family':<18} {'Disrupted':>10} {'Created':>10} {'D/(D+C)':>10} {'Bias':>15}")
        for fam, row in fam_dc.iterrows():
            bias = 'DISRUPT-biased' if row['disrupt_frac'] > 0.55 else (
                'CREATE-biased' if row['disrupt_frac'] < 0.45 else 'balanced')
            print(f"{fam:<18} {int(row['disrupted']):>10} {int(row['created']):>10} {row['disrupt_frac']:>10.2f} {bias:>15}")

    # ══════════════════════════════════════════════════════════════════
    # 2. AUTOREGULATORY FEEDBACK LOOPS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  2. AUTOREGULATORY FEEDBACK LOOPS")
    print("=" * 70)

    # Load TF gene annotations
    tf_genes = load_tf_genes(config)
    tf_gene_set = set(tf_genes['gene_id'])
    print(f"TF genes identified from EntAP: {len(tf_gene_set)}")

    # Cross-reference with expression
    tf_expressed = tf_gene_set & set(expr['gene_id'])
    print(f"TF genes with expression data: {len(tf_expressed)}")

    # Which target genes of causal bQTL are themselves TFs?
    target_genes = set(edges_df['target_gene'])
    tf_targets = target_genes & tf_gene_set
    print(f"\nTarget genes that ARE TFs: {len(tf_targets)} / {len(target_genes)} "
          f"({100*len(tf_targets)/len(target_genes):.1f}%)")

    # Get details for TF targets
    tf_target_details = edges_df[edges_df['target_gene'].isin(tf_targets)]
    print(f"Edges pointing to TF genes: {len(tf_target_details)} / {n_total_edges} "
          f"({100*len(tf_target_details)/n_total_edges:.1f}%)")

    # Assign families to target TF genes
    tf_gene_families = {}
    for _, row in tf_genes.iterrows():
        gene = row['gene_id']
        desc = row.get('Description', '') or row.get('EggNOG Description', '')
        tf_gene_families[gene] = assign_tf_family(desc)

    # Find feedback loops: TF_A disrupted → gene = TF_B, and TF_B disrupted → gene with TF_A motif
    print(f"\n── TF genes regulated by causal bQTL ──")
    tf_as_targets = defaultdict(list)
    for _, r in tf_target_details.iterrows():
        target = r['target_gene']
        target_fam = tf_gene_families.get(target, 'Unknown')
        tf_as_targets[target].append({
            'regulator_tf': r['tf_name'],
            'regulator_family': r['tf_family'],
            'effect': r['effect_type'],
            'target_family': target_fam,
            'abs_d': r['abs_d'],
        })

    # Count: which TF families are regulated by which other families
    reg_matrix = Counter()  # (regulator_family, target_family, effect_type) -> count
    for target, regulators in tf_as_targets.items():
        target_fam = tf_gene_families.get(target, 'Unknown')
        for reg in regulators:
            reg_matrix[(reg['regulator_family'], target_fam, reg['effect'])] += 1

    print(f"\n── Cross-family regulatory connections (TF family regulates TF family) ──")
    print(f"{'Regulator':<18} {'→ Target TF':<18} {'Disrupted':>10} {'Created':>10}")
    cross_fam = defaultdict(lambda: {'disrupted': 0, 'created': 0})
    for (reg_fam, tgt_fam, effect), count in reg_matrix.items():
        cross_fam[(reg_fam, tgt_fam)][effect] += count

    sorted_cross = sorted(cross_fam.items(), key=lambda x: -(x[1]['disrupted'] + x[1]['created']))
    for (reg_fam, tgt_fam), counts in sorted_cross[:25]:
        print(f"{reg_fam:<18} → {tgt_fam:<18} {counts['disrupted']:>10} {counts['created']:>10}")

    # Self-regulation: same family regulates itself
    print(f"\n── Self-regulatory loops (same family regulates itself) ──")
    for (reg_fam, tgt_fam), counts in sorted_cross:
        if reg_fam == tgt_fam:
            total = counts['disrupted'] + counts['created']
            if total >= 2:
                print(f"  {reg_fam}: {counts['disrupted']} disrupted, {counts['created']} created ({total} total)")

    # Specific feedback examples
    print(f"\n── Top TF target genes (most regulatory edges incoming) ──")
    tf_target_counts = tf_target_details.groupby('target_gene').agg(
        n_edges=('tf_name', 'size'),
        n_disrupt=('effect_type', lambda x: (x == 'disrupted').sum()),
        n_create=('effect_type', lambda x: (x == 'created').sum()),
        families=('tf_family', lambda x: ','.join(sorted(set(x)))),
    ).sort_values('n_edges', ascending=False)

    # Merge with expression
    tf_target_counts = tf_target_counts.merge(
        expr[['gene_id', 'log2fc_DS_vs_WW', 'mean_expression']],
        left_index=True, right_on='gene_id', how='left'
    )
    tf_target_counts = tf_target_counts.merge(
        tf_genes[['gene_id', 'Description']].drop_duplicates('gene_id'),
        on='gene_id', how='left'
    )

    print(f"{'Gene':<20} {'Edges':>6} {'D':>4} {'C':>4} {'log2fc':>8} {'Family':<12} {'Description'}")
    for _, r in tf_target_counts.head(20).iterrows():
        fam = tf_gene_families.get(r['gene_id'], '?')
        desc = str(r.get('Description', ''))[:50]
        fc = f"{r['log2fc_DS_vs_WW']:.2f}" if pd.notna(r.get('log2fc_DS_vs_WW')) else 'NA'
        print(f"{r['gene_id']:<20} {r['n_edges']:>6} {r['n_disrupt']:>4} {r['n_create']:>4} {fc:>8} {fam:<12} {desc}")

    # ── Is TF-targeting enriched? ──
    # Compare: fraction of TF genes among causal bQTL targets vs all genes
    n_all_genes = len(expr)
    n_all_tf = len(tf_expressed)
    n_targets = len(target_genes)
    n_tf_targets = len(tf_targets)

    # Fisher's exact
    a = n_tf_targets
    b = n_targets - n_tf_targets
    c = n_all_tf - n_tf_targets
    d = n_all_genes - n_all_tf - b
    OR, p = stats.fisher_exact([[a, b], [c, d]])
    print(f"\n── TF targeting enrichment ──")
    print(f"  TF genes among causal bQTL targets: {n_tf_targets}/{n_targets} ({100*n_tf_targets/n_targets:.1f}%)")
    print(f"  TF genes among all expressed genes: {n_all_tf}/{n_all_genes} ({100*n_all_tf/n_all_genes:.1f}%)")
    print(f"  Fisher OR = {OR:.2f}, p = {p:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # 3. VISUALIZATIONS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  3. GENERATING VISUALIZATIONS")
    print("=" * 70)

    fig_dir = config.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: TF family enrichment bar chart ──
    enrich_df = pd.DataFrame(enrichment_results).sort_values('obs_exp_ratio', ascending=True)
    enrich_df = enrich_df[enrich_df['observed'] >= 5]  # filter low counts

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#d32f2f' if r > 1.3 else '#1565c0' if r < 0.7 else '#757575'
              for r in enrich_df['obs_exp_ratio']]
    bars = ax.barh(range(len(enrich_df)), enrich_df['obs_exp_ratio'], color=colors, edgecolor='white')
    ax.axvline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_yticks(range(len(enrich_df)))
    ax.set_yticklabels(enrich_df['family'], fontsize=9)
    ax.set_xlabel('Observed / Expected ratio', fontsize=11)
    ax.set_title('TF Family Enrichment at Causal bQTL\n(vs PWM database expectation)', fontsize=13)

    # Add significance stars
    for i, (_, row) in enumerate(enrich_df.iterrows()):
        sig = '***' if row['p_binomial'] < 0.001 else ('**' if row['p_binomial'] < 0.01 else ('*' if row['p_binomial'] < 0.05 else ''))
        if sig:
            ax.text(row['obs_exp_ratio'] + 0.02, i, sig, va='center', fontsize=9, fontweight='bold')

    ax.legend(handles=[
        mpatches.Patch(color='#d32f2f', label='Enriched (>1.3×)'),
        mpatches.Patch(color='#1565c0', label='Depleted (<0.7×)'),
        mpatches.Patch(color='#757575', label='Not significant'),
    ], loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / f'causal_tf_enrichment{suffix}.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / f'causal_tf_enrichment{suffix}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: causal_tf_enrichment.pdf")

    # ── Figure 2: Disruption vs Creation balance per family ──
    if 'disrupted' in fam_dc.columns and 'created' in fam_dc.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        fam_plot = fam_dc.head(20)
        x = range(len(fam_plot))
        w = 0.35
        ax.bar([i - w/2 for i in x], fam_plot['disrupted'], w, color='#d32f2f', label='Disrupted', alpha=0.85)
        ax.bar([i + w/2 for i in x], fam_plot['created'], w, color='#1565c0', label='Created', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(fam_plot.index, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Number of edges', fontsize=11)
        ax.set_title('Motif Disruption vs Creation at Causal bQTL by TF Family', fontsize=13)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(fig_dir / f'causal_disruption_vs_creation{suffix}.pdf', dpi=150, bbox_inches='tight')
        plt.savefig(fig_dir / 'causal_disruption_vs_creation.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: causal_disruption_vs_creation.pdf")

    # ── Figure 3: Regulatory rewiring heatmap ──
    # Build matrix: rows = disrupted family, cols = created family (at same bQTL)
    rewired = linkage[(linkage['n_disrupted_sig'] > 0) & (linkage['n_created_sig'] > 0)]
    rewire_matrix = defaultdict(lambda: defaultdict(int))
    for _, r in rewired.iterrows():
        d_fams = set(f for _, f in parse_sig(r.get('all_disrupted_sig', '')))
        c_fams = set(f for _, f in parse_sig(r.get('all_created_sig', '')))
        for d in d_fams:
            for c in c_fams:
                rewire_matrix[d][c] += 1

    top_fams = [f for f, _ in Counter({f: fam_obs.get(f, 0) for f in pwm_family_counts}).most_common(15)]
    mat = np.zeros((len(top_fams), len(top_fams)))
    for i, d in enumerate(top_fams):
        for j, c in enumerate(top_fams):
            mat[i, j] = rewire_matrix[d][c]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(top_fams)))
    ax.set_xticklabels(top_fams, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(top_fams)))
    ax.set_yticklabels(top_fams, fontsize=9)
    ax.set_xlabel('Created (gained motif)', fontsize=11)
    ax.set_ylabel('Disrupted (lost motif)', fontsize=11)
    ax.set_title('Regulatory Rewiring at Causal bQTL\n(rows = disrupted, cols = created)', fontsize=13)
    plt.colorbar(im, ax=ax, label='Number of bQTL', shrink=0.8)

    # Add counts in cells
    for i in range(len(top_fams)):
        for j in range(len(top_fams)):
            val = int(mat[i, j])
            if val > 0:
                color = 'white' if val > mat.max() * 0.6 else 'black'
                ax.text(j, i, str(val), ha='center', va='center', fontsize=7, color=color)

    # Highlight diagonal (same-family switches)
    for i in range(len(top_fams)):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor='blue', linewidth=2, linestyle='--'))

    plt.tight_layout()
    plt.savefig(fig_dir / f'causal_rewiring_heatmap{suffix}.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / 'causal_rewiring_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: causal_rewiring_heatmap.pdf")

    # ── Figure 4: Feedback loop network diagram ──
    # Show TF families as nodes, edges = regulation of TF genes by other TF families
    fig, ax = plt.subplots(figsize=(14, 10))

    # Get top families involved in feedback
    feedback_fams = Counter()
    for (reg, tgt), counts in cross_fam.items():
        if reg != 'Unknown' and tgt != 'Unknown':
            total = counts['disrupted'] + counts['created']
            if total >= 2:
                feedback_fams[reg] += total
                feedback_fams[tgt] += total

    top_feedback = [f for f, _ in feedback_fams.most_common(12)]
    if not top_feedback:
        top_feedback = top_fams[:12]

    # Position nodes in a circle
    n_nodes = len(top_feedback)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    radius = 4
    positions = {f: (radius * np.cos(a), radius * np.sin(a)) for f, a in zip(top_feedback, angles)}

    # Draw edges (curved arrows)
    for (reg, tgt), counts in sorted_cross:
        if reg in positions and tgt in positions:
            total = counts['disrupted'] + counts['created']
            if total < 2:
                continue
            x1, y1 = positions[reg]
            x2, y2 = positions[tgt]

            if reg == tgt:
                # Self-loop
                loop_r = 0.6
                theta = np.arctan2(y1, x1)
                ax.annotate('', xy=(x1 + loop_r * np.cos(theta + 0.3), y1 + loop_r * np.sin(theta + 0.3)),
                           xytext=(x1 + loop_r * np.cos(theta - 0.3), y1 + loop_r * np.sin(theta - 0.3)),
                           arrowprops=dict(arrowstyle='->', color='#d32f2f',
                                          connectionstyle='arc3,rad=2.0',
                                          linewidth=min(total / 3, 3), alpha=0.7))
            else:
                # Curved edge
                color = '#d32f2f' if counts['disrupted'] > counts['created'] else '#1565c0'
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color=color,
                                          connectionstyle='arc3,rad=0.2',
                                          linewidth=min(total / 3, 3), alpha=0.6))

    # Draw nodes
    node_sizes = []
    for f in top_feedback:
        n_incoming = sum(counts['disrupted'] + counts['created']
                        for (_, tgt), counts in cross_fam.items() if tgt == f)
        node_sizes.append(max(n_incoming * 30, 300))

    for f, (x, y) in positions.items():
        n_incoming = sum(counts['disrupted'] + counts['created']
                        for (_, tgt), counts in cross_fam.items() if tgt == f)
        n_outgoing = sum(counts['disrupted'] + counts['created']
                        for (reg, _), counts in cross_fam.items() if reg == f)
        size = max(n_incoming * 30, 300)

        # Color: orange if net target (more incoming), green if net regulator
        if n_incoming > n_outgoing:
            node_color = '#ff9800'  # orange = net target
        else:
            node_color = '#4caf50'  # green = net regulator
        circle = plt.Circle((x, y), 0.5, color=node_color, alpha=0.8, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, f, ha='center', va='center', fontsize=8, fontweight='bold', zorder=6)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('TF Feedback Network at Causal bQTL\n'
                 'Red arrows = disruption-dominant, Blue = creation-dominant\n'
                 'Orange nodes = net targets, Green = net regulators',
                 fontsize=12)

    ax.legend(handles=[
        mpatches.Patch(color='#d32f2f', label='Disruption-dominant edge'),
        mpatches.Patch(color='#1565c0', label='Creation-dominant edge'),
        mpatches.Patch(color='#ff9800', label='Net target (more incoming)'),
        mpatches.Patch(color='#4caf50', label='Net regulator (more outgoing)'),
    ], loc='lower left', fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_dir / f'causal_tf_feedback_network{suffix}.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / 'causal_tf_feedback_network.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: causal_tf_feedback_network.pdf")

    # ── Figure 5: Sankey diagram of top rewiring flows ──
    fig, ax = plt.subplots(figsize=(14, 8))

    # Top rewiring flows
    top_flows = sorted_cross[:15]
    left_fams = sorted(set(d for (d, _), _ in top_flows))
    right_fams = sorted(set(c for (_, c), _ in top_flows))

    # Position left and right nodes
    n_left = len(left_fams)
    n_right = len(right_fams)
    left_y = {f: i for i, f in enumerate(left_fams)}
    right_y = {f: i for i, f in enumerate(right_fams)}

    # Normalize y positions
    max_y = max(n_left, n_right)
    for f in left_y:
        left_y[f] = left_y[f] * (max_y / n_left)
    for f in right_y:
        right_y[f] = right_y[f] * (max_y / n_right)

    # Draw flows
    max_flow = max(c['disrupted'] + c['created'] for _, c in top_flows)
    for (d_fam, c_fam), counts in top_flows:
        total = counts['disrupted'] + counts['created']
        width = total / max_flow * 0.8
        y1 = left_y.get(d_fam)
        y2 = right_y.get(c_fam)
        if y1 is not None and y2 is not None:
            color = '#d32f2f' if counts['disrupted'] > counts['created'] else '#1565c0'
            ax.fill_between([0, 1], [y1 - width/2, y2 - width/2], [y1 + width/2, y2 + width/2],
                           alpha=0.4, color=color)
            ax.plot([0, 1], [y1, y2], color=color, alpha=0.6, linewidth=0.5)
            # Label the flow
            mid_x = 0.5
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y, str(total), ha='center', va='center', fontsize=7, alpha=0.8)

    # Draw labels
    for f, y in left_y.items():
        ax.text(-0.05, y, f, ha='right', va='center', fontsize=10, fontweight='bold')
    for f, y in right_y.items():
        ax.text(1.05, y, f, ha='left', va='center', fontsize=10, fontweight='bold')

    ax.text(-0.05, max_y + 0.5, 'DISRUPTED', ha='right', va='bottom', fontsize=12,
            fontweight='bold', color='#d32f2f')
    ax.text(1.05, max_y + 0.5, 'CREATED', ha='left', va='bottom', fontsize=12,
            fontweight='bold', color='#1565c0')

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.5, max_y + 1)
    ax.axis('off')
    ax.set_title('Regulatory Rewiring Flows at Causal bQTL\n(Top 15 disrupted → created TF family pairs)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_dir / f'causal_rewiring_flows{suffix}.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / 'causal_rewiring_flows.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: causal_rewiring_flows.pdf")

    # ── Save results ──
    pd.DataFrame(enrichment_results).to_csv(
        config.results_dir / f'causal_tf_enrichment{suffix}.csv', index=False)
    edges_df.to_csv(
        config.results_dir / f'causal_regulatory_network{suffix}.csv', index=False)

    # Save feedback summary
    feedback_rows = []
    for target, regulators in tf_as_targets.items():
        target_fam = tf_gene_families.get(target, 'Unknown')
        for reg in regulators:
            feedback_rows.append({
                'target_tf_gene': target,
                'target_tf_family': target_fam,
                'regulator_tf': reg['regulator_tf'],
                'regulator_family': reg['regulator_family'],
                'effect': reg['effect'],
                'abs_d': reg['abs_d'],
            })
    if feedback_rows:
        pd.DataFrame(feedback_rows).to_csv(
            config.results_dir / f'causal_tf_feedback_loops{suffix}.csv', index=False)
        print(f"  Saved: causal_tf_feedback_loops.csv ({len(feedback_rows)} edges)")

    print("\nDone.")


if __name__ == '__main__':
    main()
