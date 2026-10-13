"""
21_drought_expression_integration.py — Integrate drought causal bQTL with
population-level drought expression to dissect cis vs trans regulation.

Analyses:
  1. Cis-driven vs trans-driven drought response
  2. Cohen's d vs drought log2FC correlation
  3. Condition-specific rewiring directionality (WW-only / DS-only)
  4. TF family enrichment at drought-specific vs WW-specific causal bQTL
"""

import sys, warnings, functools
print = functools.partial(print, flush=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_config import config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_gff3_genes(gff3_path):
    """Load gene TSS positions from GFF3."""
    genes = {}
    with open(gff3_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'gene':
                continue
            chrom = parts[0]
            if chrom not in config.valid_chroms:
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
                genes[gene_id] = {'chr': chrom, 'tss': tss, 'strand': strand,
                                  'start': start, 'end': end}
    return genes


def map_bqtl_to_genes(causal_df, genes, promoter_size=2000):
    """Map causal bQTL to nearby genes (within promoter_size of TSS)."""
    # Build lookup by chromosome
    by_chr = {}
    for gid, info in genes.items():
        by_chr.setdefault(info['chr'], []).append((gid, info['tss']))

    mappings = []
    for _, row in causal_df.iterrows():
        chrom, pos = row['chr'], row['pos']
        for gid, tss in by_chr.get(chrom, []):
            if abs(pos - tss) <= promoter_size:
                mappings.append({
                    'chr': chrom, 'pos': pos, 'gene_id': gid,
                    'dist_to_tss': abs(pos - tss),
                    'cohens_d': row.get('cohens_d', np.nan),
                    'abs_d': row.get('abs_d', np.nan),
                })
    return pd.DataFrame(mappings)


def load_pwm_database():
    """Load PlantTFDB PWMs for motif scanning."""
    import json
    pwm_path = config.pwm_db
    with open(pwm_path) as f:
        db = json.load(f)
    return db


def scan_variant_for_motifs(seq_ref, seq_alt, pwms, threshold=6.0):
    """Scan ref and alt sequences for PWM matches, return disrupted/created families."""
    disrupted = set()
    created = set()

    for pwm_entry in pwms:
        family = pwm_entry['family']
        matrix = np.array(pwm_entry['pwm'])
        w = len(matrix)

        # Scan both strands
        for seq, label in [(seq_ref, 'ref'), (seq_alt, 'alt')]:
            if len(seq) < w:
                continue
            best_score = -999
            for strand_seq in [seq, complement_reverse(seq)]:
                for i in range(len(strand_seq) - w + 1):
                    kmer = strand_seq[i:i+w]
                    score = sum(matrix[j].get(kmer[j], -2) for j in range(w)
                                if j < len(kmer))
                    best_score = max(best_score, score)

            if label == 'ref':
                ref_score = best_score
            else:
                alt_score = best_score

        if ref_score >= threshold and alt_score < threshold:
            disrupted.add(family)
        elif alt_score >= threshold and ref_score < threshold:
            created.add(family)

    return disrupted, created


def complement_reverse(seq):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq))


# ---------------------------------------------------------------------------
# Module 1: Cis vs Trans drought response
# ---------------------------------------------------------------------------

def module1_cis_vs_trans(causal_ds, causal_ww, expr_df, genes):
    """Classify drought-responsive genes as cis-driven (have causal bQTL) or trans."""
    print("\n── Module 1: Cis-driven vs Trans-driven Drought Response ──")

    # Drought-responsive genes
    drought_up = set(expr_df[expr_df['log2fc'] > 1]['gene_id'])
    drought_dn = set(expr_df[expr_df['log2fc'] < -1]['gene_id'])
    drought_all = drought_up | drought_dn
    print(f"  Drought-responsive genes (|log2fc|>1): {len(drought_all)}")
    print(f"    Up-regulated: {len(drought_up)}")
    print(f"    Down-regulated: {len(drought_dn)}")

    # Map drought-causal bQTL to genes
    ds_mapped = map_bqtl_to_genes(causal_ds, genes)
    ds_genes = set(ds_mapped['gene_id'].unique())

    # Map WW-causal bQTL to genes
    ww_mapped = map_bqtl_to_genes(causal_ww, genes)
    ww_genes = set(ww_mapped['gene_id'].unique())

    # Cis-driven: drought-responsive AND drought-causal bQTL
    cis_driven = drought_all & ds_genes
    cis_up = drought_up & ds_genes
    cis_dn = drought_dn & ds_genes
    trans_driven = drought_all - ds_genes

    print(f"\n  Cis-driven (drought-responsive + DS-causal bQTL): {len(cis_driven)}")
    print(f"    Cis up-regulated: {len(cis_up)}")
    print(f"    Cis down-regulated: {len(cis_dn)}")
    print(f"  Trans-driven (drought-responsive, no DS-causal bQTL): {len(trans_driven)}")
    print(f"  Fraction cis: {len(cis_driven)/len(drought_all):.1%}")

    # Also check: WW-causal at drought genes
    ww_at_drought = drought_all & ww_genes
    print(f"\n  WW-causal bQTL at drought genes: {len(ww_at_drought)}")
    ds_only_cis = cis_driven - ww_genes
    print(f"  Drought-ONLY cis-driven (no WW-causal): {len(ds_only_cis)}")

    # Expression magnitude comparison
    expr_cis = expr_df[expr_df['gene_id'].isin(cis_driven)]['log2fc'].abs()
    expr_trans = expr_df[expr_df['gene_id'].isin(trans_driven)]['log2fc'].abs()
    if len(expr_cis) > 0 and len(expr_trans) > 0:
        u_stat, u_p = stats.mannwhitneyu(expr_cis, expr_trans, alternative='two-sided')
        print(f"\n  Expression magnitude (|log2fc|):")
        print(f"    Cis-driven median: {expr_cis.median():.3f}")
        print(f"    Trans-driven median: {expr_trans.median():.3f}")
        print(f"    Mann-Whitney p = {u_p:.2e}")

    return {
        'cis_driven': cis_driven, 'trans_driven': trans_driven,
        'cis_up': cis_up, 'cis_dn': cis_dn,
        'ds_mapped': ds_mapped, 'ww_mapped': ww_mapped,
        'ds_only_cis': ds_only_cis,
    }


# ---------------------------------------------------------------------------
# Module 2: Cohen's d vs drought log2FC correlation
# ---------------------------------------------------------------------------

def module2_effect_correlation(causal_ds, expr_df, genes):
    """Correlate allelic imbalance (Cohen's d) with population drought expression."""
    print("\n── Module 2: Allelic Imbalance vs Drought Expression ──")

    ds_mapped = map_bqtl_to_genes(causal_ds, genes)

    # Merge with expression
    merged = ds_mapped.merge(expr_df[['gene_id', 'log2fc']], on='gene_id', how='inner')
    print(f"  bQTL-gene pairs with expression data: {len(merged)}")
    print(f"  Unique genes: {merged['gene_id'].nunique()}")

    if len(merged) < 10:
        print("  Too few pairs for correlation")
        return None

    # Correlation: Cohen's d vs log2FC
    rho_signed, p_signed = stats.spearmanr(merged['cohens_d'], merged['log2fc'])
    rho_abs, p_abs = stats.spearmanr(merged['abs_d'], merged['log2fc'].abs())

    print(f"\n  Signed correlation (d vs log2fc):")
    print(f"    Spearman rho = {rho_signed:.3f}, p = {p_signed:.2e}")
    print(f"  Absolute correlation (|d| vs |log2fc|):")
    print(f"    Spearman rho = {rho_abs:.3f}, p = {p_abs:.2e}")

    # Stratify by drought-responsive
    drought_mask = merged['log2fc'].abs() > 1
    if drought_mask.sum() > 10:
        sub = merged[drought_mask]
        rho_dr, p_dr = stats.spearmanr(sub['abs_d'], sub['log2fc'].abs())
        print(f"\n  Drought-responsive subset ({drought_mask.sum()} pairs):")
        print(f"    Spearman rho = {rho_dr:.3f}, p = {p_dr:.2e}")

    return merged


# ---------------------------------------------------------------------------
# Module 3: Condition-specific rewiring directionality
# ---------------------------------------------------------------------------

def module3_condition_specificity(causal_ds, causal_ww, expr_df, genes):
    """Analyze WW-only and DS-only causal bQTL in context of drought expression."""
    print("\n── Module 3: Condition-Specific Regulatory Rewiring ──")

    # Identify condition-specific sets
    ww_positions = set(zip(causal_ww['chr'], causal_ww['pos']))
    ds_positions = set(zip(causal_ds['chr'], causal_ds['pos']))
    shared = ww_positions & ds_positions
    ww_only = ww_positions - ds_positions
    ds_only = ds_positions - ww_positions

    print(f"  Shared causal: {len(shared)}")
    print(f"  WW-only causal: {len(ww_only)}")
    print(f"  DS-only causal: {len(ds_only)}")

    # Map each set to genes and get expression
    ww_only_df = causal_ww[causal_ww.apply(lambda r: (r['chr'], r['pos']) in ww_only, axis=1)]
    ds_only_df = causal_ds[causal_ds.apply(lambda r: (r['chr'], r['pos']) in ds_only, axis=1)]
    shared_ww = causal_ww[causal_ww.apply(lambda r: (r['chr'], r['pos']) in shared, axis=1)]

    results = {}
    for label, df in [('WW-only', ww_only_df), ('DS-only', ds_only_df), ('Shared', shared_ww)]:
        mapped = map_bqtl_to_genes(df, genes)
        if len(mapped) == 0:
            print(f"\n  {label}: no gene mappings")
            continue
        merged = mapped.merge(expr_df[['gene_id', 'log2fc']], on='gene_id', how='inner')
        n_genes = merged['gene_id'].nunique()
        n_drought = (merged.drop_duplicates('gene_id')['log2fc'].abs() > 1).sum()

        print(f"\n  {label} causal bQTL:")
        print(f"    Mapped to {n_genes} genes")
        print(f"    Drought-responsive: {n_drought} ({100*n_drought/max(n_genes,1):.1f}%)")
        if n_genes > 0:
            fc = merged.drop_duplicates('gene_id')['log2fc']
            print(f"    Median log2fc: {fc.median():.3f}")
            print(f"    Mean |log2fc|: {fc.abs().mean():.3f}")
            # Direction breakdown
            n_up = (fc > 1).sum()
            n_dn = (fc < -1).sum()
            n_stable = ((fc >= -1) & (fc <= 1)).sum()
            print(f"    Up / Down / Stable: {n_up} / {n_dn} / {n_stable}")

        results[label] = merged

    # Key question: are WW-only targets drought-repressed?
    if 'WW-only' in results and 'DS-only' in results:
        ww_fc = results['WW-only'].drop_duplicates('gene_id')['log2fc']
        ds_fc = results['DS-only'].drop_duplicates('gene_id')['log2fc']
        u, p = stats.mannwhitneyu(ww_fc.abs(), ds_fc.abs(), alternative='two-sided')
        print(f"\n  |log2fc| comparison:")
        print(f"    WW-only median |log2fc|: {ww_fc.abs().median():.3f}")
        print(f"    DS-only median |log2fc|: {ds_fc.abs().median():.3f}")
        print(f"    Mann-Whitney p = {p:.2e}")

    return results


# ---------------------------------------------------------------------------
# Module 4: TF family enrichment at condition-specific causal bQTL
# ---------------------------------------------------------------------------

def module4_tf_enrichment(causal_ds, causal_ww, genes):
    """Compare TF family enrichment at WW-only vs DS-only causal bQTL using
    pre-computed hybrid motif linkage results if available."""
    print("\n── Module 4: TF Family Enrichment by Condition ──")

    # Try to load pre-computed motif linkage results
    linkage_ww_path = config.results_dir / 'hybrid_motif_linkage.csv'

    # For now, use the variant-overlap scan approach from script 26
    # Load genotypes and reference genome for sequence extraction
    try:
        import pyfaidx
        genome = pyfaidx.Fasta(str(config.genome))
    except Exception as e:
        print(f"  Cannot load genome: {e}")
        print("  Skipping TF enrichment (requires genome for motif scanning)")
        return None

    # Load PWM database
    import json
    pwm_path = config.pwm_db
    with open(pwm_path) as f:
        pwm_db = json.load(f)

    # Build PWM lookup: family -> list of PWMs
    # PWM format: list of [p_A, p_C, p_G, p_T] per position
    NUC_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    family_pwms = {}
    for entry in pwm_db['motifs']:
        fam = entry['tf_family']
        # Pre-convert PWM to log-odds
        pwm_raw = entry['pwm']
        log_odds = []
        for pos_probs in pwm_raw:
            row = {}
            for nuc, idx in NUC_IDX.items():
                p = max(pos_probs[idx], 1e-4)
                row[nuc] = np.log2(p / 0.25)
            log_odds.append(row)
        family_pwms.setdefault(fam, []).append({
            'pwm': log_odds, 'width': len(log_odds),
        })

    # Load genotype data
    geno = pd.read_csv(config.genotypes, sep='\t')
    geno_lookup = {}
    for _, row in geno.iterrows():
        geno_lookup[(row['chr'], row['pos'])] = row

    # Condition-specific sets
    ww_positions = set(zip(causal_ww['chr'], causal_ww['pos']))
    ds_positions = set(zip(causal_ds['chr'], causal_ds['pos']))
    ww_only = ww_positions - ds_positions
    ds_only = ds_positions - ww_positions

    def scan_bqtl_set(positions, label):
        """Scan a set of bQTL for disrupted TF motifs at variant position."""
        family_counts = Counter()
        n_scanned = 0
        window = 28  # max PWM width

        for chrom, pos in positions:
            if (chrom, pos) not in geno_lookup:
                continue

            row = geno_lookup[(chrom, pos)]
            ref_base = str(row['ref'])

            # Get any alt allele
            alt_base = None
            for col in geno.columns[3:]:
                val = str(row[col])
                if val not in ('0|0', ref_base, 'nan', 'None'):
                    # Extract first alt allele
                    if '|' in val:
                        alleles = val.split('|')
                        for a in alleles:
                            if a != '0' and a != ref_base:
                                alt_base = a
                                break
                    else:
                        alt_base = val
                    if alt_base:
                        break

            if not alt_base or len(alt_base) != 1 or len(ref_base) != 1:
                continue

            # Extract ±window around variant
            try:
                start = max(0, pos - 1 - window)
                end = pos - 1 + window + 1
                seq = str(genome[chrom][start:end]).upper()
            except Exception:
                continue

            var_idx = pos - 1 - start
            if var_idx < 0 or var_idx >= len(seq):
                continue

            seq_ref = seq
            seq_alt = seq[:var_idx] + alt_base + seq[var_idx+1:]
            n_scanned += 1

            # Scan each family — count if any PWM in family is disrupted
            for fam, pwms in family_pwms.items():
                found = False
                for pwm_entry in pwms:
                    matrix = pwm_entry['pwm']
                    w = pwm_entry['width']

                    # Check if motif overlaps variant position
                    for offset in range(max(0, var_idx - w + 1),
                                        min(len(seq) - w + 1, var_idx + 1)):
                        kmer_ref = seq_ref[offset:offset+w]
                        kmer_alt = seq_alt[offset:offset+w]

                        if len(kmer_ref) < w or len(kmer_alt) < w:
                            continue

                        score_ref = sum(matrix[j].get(kmer_ref[j], -2)
                                        for j in range(w))
                        score_alt = sum(matrix[j].get(kmer_alt[j], -2)
                                        for j in range(w))

                        if score_ref >= 6.0 and score_alt < 6.0:
                            family_counts[fam] += 1
                            found = True
                            break
                    if found:
                        break

        print(f"  {label}: scanned {n_scanned} bQTL")
        return family_counts, n_scanned

    ds_counts, ds_n = scan_bqtl_set(ds_only, "DS-only")
    ww_counts, ww_n = scan_bqtl_set(ww_only, "WW-only")

    # Compare enrichment
    all_families = sorted(set(list(ds_counts.keys()) + list(ww_counts.keys())))
    print(f"\n  {'Family':<12} {'DS-only':>10} {'WW-only':>10} {'DS%':>8} {'WW%':>8} {'Ratio':>8}")
    print(f"  {'-'*58}")

    enrichment_data = []
    for fam in all_families:
        ds_c = ds_counts.get(fam, 0)
        ww_c = ww_counts.get(fam, 0)
        ds_pct = 100 * ds_c / max(ds_n, 1)
        ww_pct = 100 * ww_c / max(ww_n, 1)
        ratio = (ds_c / max(ds_n, 1)) / (ww_c / max(ww_n, 1)) if ww_c > 0 else float('inf')

        # Fisher's exact: is this family more disrupted in DS-only vs WW-only?
        table = [[ds_c, ds_n - ds_c], [ww_c, ww_n - ww_c]]
        odds, fisher_p = stats.fisher_exact(table)
        enrichment_data.append({
            'family': fam, 'ds_count': ds_c, 'ww_count': ww_c,
            'ds_pct': ds_pct, 'ww_pct': ww_pct, 'ds_ww_ratio': ratio,
            'fisher_or': odds, 'fisher_p': fisher_p,
        })
        sig = '*' if fisher_p < 0.05 else ''
        print(f"  {fam:<12} {ds_c:>10} {ww_c:>10} {ds_pct:>7.1f}% {ww_pct:>7.1f}% {ratio:>7.2f} {sig}")

    enrich_df = pd.DataFrame(enrichment_data)

    # FDR correction
    from statsmodels.stats.multitest import multipletests
    if len(enrich_df) > 0:
        _, fdr, _, _ = multipletests(enrich_df['fisher_p'], method='fdr_bh')
        enrich_df['fdr'] = fdr
        sig_fams = enrich_df[enrich_df['fdr'] < 0.05].sort_values('fisher_or', ascending=False)
        print(f"\n  Significant families (FDR<0.05): {len(sig_fams)}")
        for _, row in sig_fams.iterrows():
            direction = "DS-enriched" if row['fisher_or'] > 1 else "WW-enriched"
            print(f"    {row['family']:<12} OR={row['fisher_or']:.2f} FDR={row['fdr']:.3f} ({direction})")

    return enrich_df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(mod1, mod2_merged, mod3_results, mod4_enrich, expr_df):
    """Generate 4-panel figure."""
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Cis vs Trans pie + expression distributions
    ax1 = fig.add_subplot(gs[0, 0])
    n_cis = len(mod1['cis_driven'])
    n_trans = len(mod1['trans_driven'])
    colors = ['#2196F3', '#E0E0E0']
    wedges, texts, autotexts = ax1.pie(
        [n_cis, n_trans], labels=['Cis-driven', 'Trans-driven'],
        autopct='%1.1f%%', colors=colors, startangle=90,
        textprops={'fontsize': 11})
    ax1.set_title(f'A. Drought response: cis vs trans\n'
                  f'({n_cis} cis, {n_trans} trans)', fontsize=12, fontweight='bold')

    # Panel B: Cohen's d vs log2FC scatter
    ax2 = fig.add_subplot(gs[0, 1])
    if mod2_merged is not None and len(mod2_merged) > 0:
        drought_mask = mod2_merged['log2fc'].abs() > 1
        ax2.scatter(mod2_merged.loc[~drought_mask, 'cohens_d'],
                    mod2_merged.loc[~drought_mask, 'log2fc'],
                    alpha=0.2, s=8, c='grey', label='Stable')
        ax2.scatter(mod2_merged.loc[drought_mask, 'cohens_d'],
                    mod2_merged.loc[drought_mask, 'log2fc'],
                    alpha=0.4, s=12, c='#E53935', label='Drought-responsive')
        rho, p = stats.spearmanr(mod2_merged['cohens_d'], mod2_merged['log2fc'])
        ax2.set_xlabel("Cohen's d (allelic imbalance)", fontsize=11)
        ax2.set_ylabel("Drought log2FC", fontsize=11)
        ax2.axhline(0, c='k', lw=0.5, ls='--')
        ax2.axvline(0, c='k', lw=0.5, ls='--')
        ax2.legend(fontsize=9)
        ax2.set_title(f'B. Allelic imbalance vs drought expression\n'
                      f'ρ = {rho:.3f}, p = {p:.1e}', fontsize=12, fontweight='bold')

    # Panel C: Expression of WW-only vs DS-only targets
    ax3 = fig.add_subplot(gs[1, 0])
    if 'WW-only' in mod3_results and 'DS-only' in mod3_results:
        ww_fc = mod3_results['WW-only'].drop_duplicates('gene_id')['log2fc']
        ds_fc = mod3_results['DS-only'].drop_duplicates('gene_id')['log2fc']
        bins = np.linspace(-4, 4, 50)
        ax3.hist(ww_fc.clip(-4, 4), bins=bins, alpha=0.6, color='#4CAF50',
                 label=f'WW-only (n={len(ww_fc)})', density=True)
        ax3.hist(ds_fc.clip(-4, 4), bins=bins, alpha=0.6, color='#FF9800',
                 label=f'DS-only (n={len(ds_fc)})', density=True)
        ax3.axvline(0, c='k', lw=0.5, ls='--')
        ax3.set_xlabel('Drought log2FC', fontsize=11)
        ax3.set_ylabel('Density', fontsize=11)
        ax3.legend(fontsize=9)
        u, p = stats.mannwhitneyu(ww_fc.abs(), ds_fc.abs(), alternative='two-sided')
        ax3.set_title(f'C. Drought expression of condition-specific targets\n'
                      f'MW p = {p:.1e}', fontsize=12, fontweight='bold')

    # Panel D: TF family enrichment DS vs WW
    ax4 = fig.add_subplot(gs[1, 1])
    if mod4_enrich is not None and len(mod4_enrich) > 0:
        # Plot top families by absolute difference
        plot_df = mod4_enrich.copy()
        plot_df['diff'] = plot_df['ds_pct'] - plot_df['ww_pct']
        plot_df = plot_df.sort_values('diff')

        # Color by significance
        colors = ['#E53935' if row['fdr'] < 0.05 and row['fisher_or'] > 1
                  else '#2196F3' if row['fdr'] < 0.05 and row['fisher_or'] < 1
                  else '#BDBDBD' for _, row in plot_df.iterrows()]

        y_pos = range(len(plot_df))
        ax4.barh(y_pos, plot_df['diff'], color=colors, height=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(plot_df['family'], fontsize=8)
        ax4.axvline(0, c='k', lw=0.5)
        ax4.set_xlabel('Disruption rate difference (DS% − WW%)', fontsize=10)
        ax4.set_title('D. TF family enrichment: drought vs well-watered\n'
                      '(red = DS-enriched, blue = WW-enriched)',
                      fontsize=12, fontweight='bold')

    fig_path = config.figures_dir / 'drought_expression_integration.pdf'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"\n  Figure saved: {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  DROUGHT EXPRESSION INTEGRATION")
    print("=" * 70)

    # Load data
    causal_ds = pd.read_csv(config.results_dir / 'causal_bqtl_ds.csv')
    causal_ww = pd.read_csv(config.results_dir / 'causal_bqtl_728.csv')

    # Load population-level drought expression
    expr_df = pd.read_csv(config.expression, sep='\t')
    # Standardize column names
    if 'log2fc_DS_vs_WW' in expr_df.columns:
        expr_df = expr_df.rename(columns={'log2fc_DS_vs_WW': 'log2fc'})
    elif 'log2FoldChange' in expr_df.columns:
        expr_df = expr_df.rename(columns={'log2FoldChange': 'log2fc'})
    if 'GeneID' in expr_df.columns:
        expr_df = expr_df.rename(columns={'GeneID': 'gene_id'})
    elif 'gene' in expr_df.columns:
        expr_df = expr_df.rename(columns={'gene': 'gene_id'})
    print(f"  Drought expression data: {len(expr_df)} genes")

    # Load GFF3 genes
    genes = load_gff3_genes(config.gff3)
    print(f"  GFF3 genes: {len(genes)}")

    print(f"  WW causal bQTL: {len(causal_ww)}")
    print(f"  DS causal bQTL: {len(causal_ds)}")

    # Run modules
    mod1 = module1_cis_vs_trans(causal_ds, causal_ww, expr_df, genes)
    mod2_merged = module2_effect_correlation(causal_ds, expr_df, genes)
    mod3_results = module3_condition_specificity(causal_ds, causal_ww, expr_df, genes)
    mod4_enrich = module4_tf_enrichment(causal_ds, causal_ww, genes)

    # Generate figures
    make_figures(mod1, mod2_merged, mod3_results, mod4_enrich, expr_df)

    # Save summary
    summary = {
        'n_cis_driven': len(mod1['cis_driven']),
        'n_trans_driven': len(mod1['trans_driven']),
        'cis_fraction': len(mod1['cis_driven']) / (len(mod1['cis_driven']) + len(mod1['trans_driven'])),
        'cis_up': len(mod1['cis_up']),
        'cis_dn': len(mod1['cis_dn']),
        'ds_only_cis': len(mod1['ds_only_cis']),
        'cis_genes': sorted(mod1['cis_driven']),
        'ds_only_cis_genes': sorted(mod1['ds_only_cis']),
    }
    import json
    out_path = config.results_dir / 'drought_expression_integration.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {out_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
