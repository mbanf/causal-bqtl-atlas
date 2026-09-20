#!/usr/bin/env python3
"""
Per-hybrid binding landscape analysis and ASE prediction.

For each of 24 hybrids with both MOA-seq peaks and ASE data:
1. Extract B73 peaks in each gene's promoter
2. Identify which peaks contain bQTL (disrupted binding)
3. Build feature matrix: peak count, disruption count, fraction, signal, etc.
4. Train ASE predictor from binding features
5. Cross-hybrid leave-one-out validation

Key question: can we predict allele-specific expression from the
binding landscape of a gene's promoter?
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import bisect
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
GFF3 = DATA / "raw" / "Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3"
PEAKS_DIR = DATA / "raw" / "engelhorn_peaks"
BQTL = DATA / "processed" / "bqtl_snp_ww.csv"
ASE_HYBRID = DATA / "processed" / "engelhorn_ase_ww.csv"
ASE_GENE = DATA / "processed" / "engelhorn_ase_ww_gene_summary.csv"
EXPR = DATA / "processed" / "engelhorn_ww_vs_ds_expression.tsv"
OUTDIR = BASE / "results"

PROMOTER_BP = 2000
VALID_CHROMS = {f'chr{i}' for i in range(1, 11)}

# Hybrids with both peaks and ASE
HYBRIDS = ['A188', 'A619', 'B97', 'CML103', 'CML247', 'CML277', 'CML322',
           'CML333', 'CML69', 'HP301', 'IL14H', 'Ki11', 'Ki3', 'Ky21',
           'M162W', 'Mo17', 'Mo18W', 'Ms71', 'NC358', 'Oh43', 'Oh7b',
           'P39', 'Tx303', 'W22']


def parse_gff3():
    """Parse GFF3 for gene coordinates on main chromosomes."""
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
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attrs = parts[8]
            gene_id = None
            for attr in attrs.split(';'):
                if attr.startswith('ID=gene:'):
                    gene_id = attr.split(':')[1]
                    break
                elif attr.startswith('ID=Zm'):
                    gene_id = attr[3:]
                    break
            if gene_id:
                tss = start if strand == '+' else end
                genes[gene_id] = {
                    'chr': chrom, 'start': start, 'end': end,
                    'strand': strand, 'tss': tss
                }
    return genes


def load_hybrid_peaks(hybrid):
    """Load B73 peaks for a single hybrid (WW condition)."""
    peak_file = PEAKS_DIR / f"{hybrid}.w2_4.q3_peaks.narrowPeak"
    peaks = []
    with open(peak_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            chrom_raw = parts[0]
            if not chrom_raw.startswith('B73-chr'):
                continue
            chrom = chrom_raw.replace('B73-', '')
            if chrom not in VALID_CHROMS:
                continue
            peaks.append({
                'chr': chrom,
                'start': int(parts[1]),
                'end': int(parts[2]),
                'score': int(parts[4]),
                'signal': float(parts[6]),
                'pval': float(parts[7]),
                'qval': float(parts[8]),
                'summit_offset': int(parts[9]),
            })
    return pd.DataFrame(peaks)


def build_promoter_index(gene_coords):
    """Build sorted promoter intervals per chromosome for fast lookup."""
    promoters_by_chr = defaultdict(list)
    for gene_id, info in gene_coords.items():
        chrom = info['chr']
        if info['strand'] == '+':
            prom_start = max(0, info['tss'] - PROMOTER_BP)
            prom_end = info['tss']
        else:
            prom_start = info['tss']
            prom_end = info['tss'] + PROMOTER_BP
        promoters_by_chr[chrom].append((prom_start, prom_end, gene_id))
    for chrom in promoters_by_chr:
        promoters_by_chr[chrom].sort()
    return promoters_by_chr


def map_peaks_to_genes(peaks_df, promoters_by_chr):
    """Map peaks to gene promoters. Returns dict: gene_id -> list of peak dicts."""
    gene_peaks = defaultdict(list)

    for chrom in peaks_df['chr'].unique():
        chr_peaks = peaks_df[peaks_df['chr'] == chrom].sort_values('start')
        promoters = promoters_by_chr.get(chrom, [])
        if not promoters:
            continue

        prom_starts = np.array([p[0] for p in promoters])
        prom_ends = np.array([p[1] for p in promoters])
        prom_genes = [p[2] for p in promoters]

        # Pre-extract peak arrays (avoid pandas in inner loop)
        pk_starts = chr_peaks['start'].values
        pk_ends = chr_peaks['end'].values
        pk_signals = chr_peaks['signal'].values
        pk_scores = chr_peaks['score'].values
        pk_summits = chr_peaks['summit_offset'].values
        n_peaks = len(pk_starts)

        for pi in range(n_peaks):
            ps, pe = pk_starts[pi], pk_ends[pi]
            # Find promoters overlapping this peak
            right = bisect.bisect_left(prom_starts, pe)
            for j in range(max(0, right - 50), right):  # promoters near this position
                if prom_ends[j] > ps and prom_starts[j] < pe:
                    gene_peaks[prom_genes[j]].append({
                        'chr': chrom,
                        'start': int(ps),
                        'end': int(pe),
                        'signal': float(pk_signals[pi]),
                        'score': int(pk_scores[pi]),
                        'summit_offset': int(pk_summits[pi]),
                    })

    return gene_peaks


def build_bqtl_index(bqtl_df):
    """Build sorted bQTL position index per chromosome."""
    bqtl_by_chr = {}
    for chrom in VALID_CHROMS:
        positions = sorted(bqtl_df[bqtl_df['chr'] == chrom]['pos'].values)
        bqtl_by_chr[chrom] = np.array(positions)
    return bqtl_by_chr


def count_bqtl_in_peak(peak_chr, peak_start, peak_end, bqtl_by_chr):
    """Count bQTL falling within a peak."""
    positions = bqtl_by_chr.get(peak_chr, np.array([]))
    if len(positions) == 0:
        return 0
    left = bisect.bisect_left(positions, peak_start)
    right = bisect.bisect_right(positions, peak_end)
    return right - left


def extract_gene_features(gene_peaks_list, bqtl_by_chr, gene_info):
    """Extract binding features for one gene from its peaks."""
    n_peaks = len(gene_peaks_list)
    if n_peaks == 0:
        return {
            'n_peaks': 0,
            'n_disrupted': 0,
            'n_intact': 0,
            'fraction_disrupted': 0.0,
            'total_bqtl': 0,
            'mean_signal': 0.0,
            'max_signal': 0.0,
            'mean_score': 0.0,
            'mean_peak_width': 0.0,
            'total_peak_coverage': 0.0,
            'mean_dist_to_tss': PROMOTER_BP,
            'min_dist_to_tss': PROMOTER_BP,
            'has_proximal_peak': 0,  # within 200bp of TSS
            'has_distal_peak': 0,    # >1000bp from TSS
        }

    tss = gene_info['tss']
    disrupted = 0
    total_bqtl = 0
    signals = []
    scores = []
    widths = []
    dists = []
    coverage = 0

    for peak in gene_peaks_list:
        n_bqtl = count_bqtl_in_peak(peak['chr'], peak['start'], peak['end'], bqtl_by_chr)
        total_bqtl += n_bqtl
        if n_bqtl > 0:
            disrupted += 1
        signals.append(peak['signal'])
        scores.append(peak['score'])
        w = peak['end'] - peak['start']
        widths.append(w)
        coverage += w
        summit = peak['start'] + peak['summit_offset']
        dists.append(abs(summit - tss))

    return {
        'n_peaks': n_peaks,
        'n_disrupted': disrupted,
        'n_intact': n_peaks - disrupted,
        'fraction_disrupted': disrupted / n_peaks,
        'total_bqtl': total_bqtl,
        'mean_signal': np.mean(signals),
        'max_signal': np.max(signals),
        'mean_score': np.mean(scores),
        'mean_peak_width': np.mean(widths),
        'total_peak_coverage': coverage,
        'mean_dist_to_tss': np.mean(dists),
        'min_dist_to_tss': np.min(dists),
        'has_proximal_peak': int(np.min(dists) < 200),
        'has_distal_peak': int(np.max(dists) > 1000),
    }


def build_hybrid_feature_matrix():
    """Build feature matrix: gene × hybrid × binding features."""
    print("=" * 70)
    print("BUILDING PER-HYBRID FEATURE MATRIX")
    print("=" * 70)

    gene_coords = parse_gff3()
    promoters_by_chr = build_promoter_index(gene_coords)
    bqtl_df = pd.read_csv(BQTL)
    bqtl_by_chr = build_bqtl_index(bqtl_df)

    # Load ASE data (per hybrid)
    ase_df = pd.read_csv(ASE_HYBRID)
    # Create lookup: (gene_id, parent) -> abs_log2_ratio
    ase_dict = {}
    for _, row in ase_df.iterrows():
        parent = row['hybrid'].replace('B73x', '')
        ase_dict[(row['gene_id'], parent)] = row['abs_log2_ratio']

    print(f"Genes: {len(gene_coords):,}")
    print(f"bQTL: {len(bqtl_df):,}")
    print(f"ASE entries: {len(ase_df):,}")
    print(f"Hybrids: {len(HYBRIDS)}")

    all_records = []
    for hi, hybrid in enumerate(HYBRIDS):
        peaks_df = load_hybrid_peaks(hybrid)
        gene_peaks = map_peaks_to_genes(peaks_df, promoters_by_chr)

        n_genes_with_peaks = len(gene_peaks)
        hybrid_records = 0

        for gene_id, info in gene_coords.items():
            peaks_list = gene_peaks.get(gene_id, [])
            features = extract_gene_features(peaks_list, bqtl_by_chr, info)
            features['gene_id'] = gene_id
            features['hybrid'] = hybrid
            features['abs_log2_ase'] = ase_dict.get((gene_id, hybrid), np.nan)
            all_records.append(features)
            hybrid_records += 1

        print(f"  [{hi+1:2d}/24] {hybrid:8s}: {len(peaks_df):6d} B73 peaks, "
              f"{n_genes_with_peaks:5d} genes with peaks, "
              f"{hybrid_records} records")

    df = pd.DataFrame(all_records)
    print(f"\nFeature matrix: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Genes: {df['gene_id'].nunique():,}")
    print(f"Records with ASE: {df['abs_log2_ase'].notna().sum():,}")

    return df


def analyze_features(df):
    """Analyze which features predict ASE."""
    from scipy import stats

    print(f"\n{'='*70}")
    print("FEATURE ANALYSIS")
    print(f"{'='*70}")

    has_ase = df[df['abs_log2_ase'].notna()].copy()
    print(f"Records with ASE: {len(has_ase):,}")

    feature_cols = [c for c in df.columns if c not in ('gene_id', 'hybrid', 'abs_log2_ase')]

    print(f"\n--- Feature correlations with |ASE| ---")
    correlations = []
    for col in feature_cols:
        rho, p = stats.spearmanr(has_ase[col], has_ase['abs_log2_ase'])
        correlations.append({'feature': col, 'rho': rho, 'p': p, 'abs_rho': abs(rho)})
        print(f"  {col:25s}: rho={rho:+.4f}, p={p:.2e}")

    corr_df = pd.DataFrame(correlations).sort_values('abs_rho', ascending=False)
    print(f"\nRanked by |rho|:")
    for _, row in corr_df.iterrows():
        sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
        print(f"  {row['feature']:25s}: rho={row['rho']:+.4f} {sig}")

    return has_ase, feature_cols


def train_ase_predictor(has_ase, feature_cols):
    """Train and evaluate ASE predictor using binding features."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    import sklearn.model_selection as ms

    print(f"\n{'='*70}")
    print("ASE PREDICTION FROM BINDING FEATURES")
    print(f"{'='*70}")

    X = has_ase[feature_cols].values
    y = has_ase['abs_log2_ase'].values
    hybrids = has_ase['hybrid'].values

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"y stats: mean={y.mean():.4f}, std={y.std():.4f}")

    # --- 1. Standard 5-fold CV (random split) ---
    print(f"\n--- 1. Random 5-fold CV ---")
    for name, model in [
        ('Ridge', Ridge(alpha=1.0)),
        ('RF', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ('GBR', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
    ]:
        scores_r2 = ms.cross_val_score(model, X, y, cv=5, scoring='r2')
        scores_mae = -ms.cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        print(f"  {name:5s}: R²={scores_r2.mean():.4f}±{scores_r2.std():.4f}, "
              f"MAE={scores_mae.mean():.4f}±{scores_mae.std():.4f}")

    # --- 2. Leave-one-hybrid-out CV ---
    print(f"\n--- 2. Leave-one-hybrid-out CV ---")
    unique_hybrids = sorted(set(hybrids))
    print(f"  {len(unique_hybrids)} hybrids")

    for name, model_fn in [
        ('Ridge', lambda: Ridge(alpha=1.0)),
        ('RF', lambda: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ('GBR', lambda: GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
    ]:
        hybrid_r2s = []
        hybrid_maes = []
        all_preds = []
        all_true = []

        for test_hybrid in unique_hybrids:
            train_mask = hybrids != test_hybrid
            test_mask = hybrids == test_hybrid

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            if len(y_test) < 10:
                continue

            model = model_fn()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            hybrid_r2s.append(r2)
            hybrid_maes.append(mae)
            all_preds.extend(y_pred)
            all_true.extend(y_test)

        overall_r2 = r2_score(all_true, all_preds)
        overall_mae = mean_absolute_error(all_true, all_preds)
        print(f"  {name:5s}: R²={overall_r2:.4f} (per-hybrid mean={np.mean(hybrid_r2s):.4f}), "
              f"MAE={overall_mae:.4f}")

    # --- 3. Feature importance (RF) ---
    print(f"\n--- 3. Feature importance (Random Forest) ---")
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        print(f"  {feature_cols[i]:25s}: {importances[i]:.4f}")

    # --- 4. Binary classification: high vs low ASE ---
    print(f"\n--- 4. Binary classification (|ASE| > 1 vs ≤ 1) ---")
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    y_binary = (y > 1.0).astype(int)
    n_pos = y_binary.sum()
    n_neg = len(y_binary) - n_pos
    print(f"  Positive (|ASE|>1): {n_pos:,} ({100*n_pos/len(y_binary):.1f}%)")
    print(f"  Negative (|ASE|≤1): {n_neg:,}")

    for name, model_fn in [
        ('RF', lambda: RandomForestClassifier(n_estimators=100, max_depth=10,
                                               random_state=42, n_jobs=-1, class_weight='balanced')),
        ('GBR', lambda: GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ]:
        # Leave-one-hybrid-out
        all_preds = []
        all_true = []
        all_proba = []

        for test_hybrid in unique_hybrids:
            train_mask = hybrids != test_hybrid
            test_mask = hybrids == test_hybrid

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y_binary[train_mask], y_binary[test_mask]

            if len(y_test) < 10 or y_test.sum() < 2:
                continue

            model = model_fn()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            all_preds.extend(y_pred)
            all_true.extend(y_test)
            all_proba.extend(y_prob)

        auc = roc_auc_score(all_true, all_proba)
        bacc = balanced_accuracy_score(all_true, all_preds)
        print(f"  {name:5s} (LOHO): AUC={auc:.4f}, Balanced Acc={bacc:.4f}")

    return rf, feature_cols


def generate_figures(df, has_ase, feature_cols, rf_model):
    """Generate publication figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy import stats

    fig_dir = OUTDIR / 'figures'
    fig_dir.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Panel A: Peaks per gene per hybrid
    ax1 = fig.add_subplot(gs[0, 0])
    hybrid_means = df.groupby('hybrid')['n_peaks'].mean().sort_values()
    ax1.barh(range(len(hybrid_means)), hybrid_means.values, color='#2196F3', edgecolor='white')
    ax1.set_yticks(range(len(hybrid_means)))
    ax1.set_yticklabels(hybrid_means.index, fontsize=6)
    ax1.set_xlabel('Mean peaks per gene promoter')
    ax1.set_title('A. Binding events per hybrid')

    # Panel B: Disruption fraction per hybrid
    ax2 = fig.add_subplot(gs[0, 1])
    has_peaks = df[df['n_peaks'] > 0]
    hybrid_frac = has_peaks.groupby('hybrid')['fraction_disrupted'].mean().sort_values()
    ax2.barh(range(len(hybrid_frac)), hybrid_frac.values, color='#FF5722', edgecolor='white')
    ax2.set_yticks(range(len(hybrid_frac)))
    ax2.set_yticklabels(hybrid_frac.index, fontsize=6)
    ax2.set_xlabel('Mean fraction peaks disrupted')
    ax2.set_title('B. Disruption fraction per hybrid')

    # Panel C: Feature importance
    ax3 = fig.add_subplot(gs[0, 2])
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)
    top_n = min(12, len(feature_cols))
    top_idx = sorted_idx[-top_n:]
    ax3.barh(range(top_n), importances[top_idx], color='#4CAF50', edgecolor='white')
    ax3.set_yticks(range(top_n))
    ax3.set_yticklabels([feature_cols[i] for i in top_idx], fontsize=7)
    ax3.set_xlabel('Feature importance')
    ax3.set_title('C. RF feature importance for |ASE|')

    # Panel D: n_peaks vs ASE by hybrid (line plot)
    ax4 = fig.add_subplot(gs[1, 0])
    for hybrid in ['Mo17', 'Oh43', 'CML247', 'B97']:
        hdata = has_ase[has_ase['hybrid'] == hybrid]
        bins = [0, 1, 2, 3, 4, 100]
        labels = ['0', '1', '2', '3', '4+']
        hdata = hdata.copy()
        hdata['pk_bin'] = pd.cut(hdata['n_peaks'], bins=bins, labels=labels, right=False)
        means = hdata.groupby('pk_bin', observed=True)['abs_log2_ase'].mean()
        ax4.plot(means.index, means.values, 'o-', label=hybrid, alpha=0.7, markersize=4)
    ax4.set_xlabel('Peaks in promoter')
    ax4.set_ylabel('Mean |log₂(B73/NAM)|')
    ax4.set_title('D. Peaks vs ASE (select hybrids)')
    ax4.legend(fontsize=7)

    # Panel E: Disrupted vs intact ASE comparison
    ax5 = fig.add_subplot(gs[1, 1])
    has_ase_copy = has_ase.copy()
    has_ase_copy['category'] = 'No peaks'
    has_ase_copy.loc[
        (has_ase_copy['n_peaks'] > 0) & (has_ase_copy['n_disrupted'] == 0), 'category'
    ] = 'Peaks, none disrupted'
    has_ase_copy.loc[has_ase_copy['n_disrupted'] > 0, 'category'] = 'Has disrupted peak'
    cats = ['No peaks', 'Peaks, none disrupted', 'Has disrupted peak']
    cat_stats = []
    for cat in cats:
        cdata = has_ase_copy[has_ase_copy['category'] == cat]['abs_log2_ase']
        cat_stats.append({'cat': cat, 'mean': cdata.mean(), 'se': cdata.sem(), 'n': len(cdata)})
    cat_df = pd.DataFrame(cat_stats)
    ax5.bar(range(len(cats)), cat_df['mean'], yerr=cat_df['se'],
            color=['#9E9E9E', '#4CAF50', '#FF5722'], edgecolor='white', capsize=3)
    ax5.set_xticks(range(len(cats)))
    ax5.set_xticklabels([c.replace(', ', '\n') for c in cats], fontsize=7)
    ax5.set_ylabel('Mean |log₂(B73/NAM)|')
    ax5.set_title('E. ASE by disruption status')
    for i, row in cat_df.iterrows():
        ax5.text(i, row['mean'] + row['se'] + 0.01, f'n={row["n"]:,}', ha='center', fontsize=6)

    # Panel F: Cross-hybrid variation in peak count
    ax6 = fig.add_subplot(gs[1, 2])
    # How variable is peak count across hybrids for the same gene?
    gene_var = df.groupby('gene_id')['n_peaks'].agg(['mean', 'std', 'min', 'max'])
    gene_var['range'] = gene_var['max'] - gene_var['min']
    ax6.hist(gene_var['range'], bins=range(0, 10), color='#9C27B0',
             edgecolor='white', alpha=0.8)
    ax6.set_xlabel('Range of peak count across 24 hybrids')
    ax6.set_ylabel('Genes')
    ax6.set_title(f'F. Cross-hybrid peak variation\n'
                  f'(mean range={gene_var["range"].mean():.1f})')

    plt.suptitle('Per-Hybrid Binding Landscape and ASE Prediction',
                 fontsize=13, fontweight='bold')
    plt.savefig(fig_dir / 'fig7_hybrid_binding_predictor.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(fig_dir / 'fig7_hybrid_binding_predictor.png', bbox_inches='tight', dpi=150)
    print(f"\nSaved: figures/fig7_hybrid_binding_predictor.pdf")
    plt.close()


def main():
    # Build feature matrix
    df = build_hybrid_feature_matrix()

    # Save feature matrix
    df.to_csv(OUTDIR / 'hybrid_binding_features.csv', index=False)
    print(f"Saved: hybrid_binding_features.csv")

    # Analyze features
    has_ase, feature_cols = analyze_features(df)

    # Train predictor
    rf_model, feature_cols = train_ase_predictor(has_ase, feature_cols)

    # Generate figures
    generate_figures(df, has_ase, feature_cols, rf_model)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Feature matrix: {df.shape[0]:,} gene×hybrid records")
    print(f"Hybrids: {df['hybrid'].nunique()}")
    print(f"Genes: {df['gene_id'].nunique():,}")
    print(f"Records with ASE: {has_ase.shape[0]:,}")


if __name__ == '__main__':
    main()
