#!/usr/bin/env python3
"""
17_causality_prediction.py — Predict which bQTL are causal (drive ASE).

Two-phase model:
  Phase 1: Motif + distance features (available for all 147K bQTL)
  Phase 2: Agro-NT embeddings (requires extraction for negative samples)

Positive class: 1,075 causal bQTL (FDR<0.05 in colocalisation test)
Negative class: subsampled non-causal bQTL from the same 147K set

Features (Phase 1):
  - dist_to_tss: distance from bQTL to nearest gene TSS
  - n_variant_families: number of TF families with motif at variant
  - motifs_at_variant: total motif hits overlapping variant
  - total_motifs_in_region: motifs in the bound region
  - n_families_in_region: TF families in bound region
  - region_width: bound region width (bp)
  - per-family binary indicators: whether each TF family has motif at variant
  - motif_density: motifs_at_variant / region_width
  - family_diversity: n_variant_families / n_families_in_region

Features (Phase 2 — appended when embeddings available):
  - 1500-dim frozen Agro-NT embedding
  - 1500-dim LoRA Agro-NT embedding (or PCA-reduced)
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_config import config

RESULTS = config.results_dir
FIGURES = config.figures_dir
SEED = 42


def load_motif_features(motifs_path, causal_path):
    """Build feature matrix from bound_region_motifs.csv."""
    print("  Loading motif data...")
    motifs = pd.read_csv(motifs_path)
    causal = pd.read_csv(causal_path)
    causal_set = set(zip(causal.chr, causal.pos))

    # Aggregate per bQTL: take max motifs_at_variant, collect families
    print(f"  Raw motif rows: {len(motifs):,}")

    # Get unique bQTL positions with their features
    grouped = motifs.groupby(['chr', 'pos']).agg({
        'dist_to_tss': 'min',  # closest gene
        'motifs_at_variant': 'max',
        'total_motifs_in_region': 'max',
        'n_families_in_region': 'max',
        'n_variant_families': 'max',
        'region_width': 'max',
        'variant_families': lambda x: ','.join(
            set(f for fam_str in x.dropna() for f in str(fam_str).split(',') if f.strip())),
    }).reset_index()

    # Label causal vs non-causal
    causal_keys = grouped[['chr', 'pos']].apply(tuple, axis=1)
    grouped['causal'] = causal_keys.isin(causal_set).astype(int)

    print(f"  Unique bQTL: {len(grouped):,}")
    print(f"  Causal: {grouped['causal'].sum():,}")
    print(f"  Non-causal: {(grouped['causal'] == 0).sum():,}")

    return grouped, causal_set


def build_feature_matrix(grouped):
    """Convert aggregated motif data to numeric feature matrix."""
    # Collect all TF families
    all_families = set()
    for fam_str in grouped['variant_families']:
        if pd.notna(fam_str) and fam_str:
            all_families.update(f.strip() for f in fam_str.split(',') if f.strip())
    all_families.discard('')
    all_families = sorted(all_families)
    print(f"  TF families for indicators: {len(all_families)}")

    # Numeric features
    numeric_cols = [
        'dist_to_tss', 'motifs_at_variant', 'total_motifs_in_region',
        'n_families_in_region', 'n_variant_families', 'region_width'
    ]

    X = grouped[numeric_cols].copy()

    # Handle dist_to_tss = -1 (no gene nearby)
    X['dist_to_tss'] = X['dist_to_tss'].replace(-1, 100000)
    X['log_dist'] = np.log1p(X['dist_to_tss'].abs())

    # Derived features
    X['motif_density'] = X['motifs_at_variant'] / X['region_width'].clip(lower=1)
    X['family_diversity'] = (X['n_variant_families'] /
                              X['n_families_in_region'].clip(lower=1))
    X['has_any_motif'] = (X['motifs_at_variant'] > 0).astype(int)

    # Per-family binary indicators (vectorized via MultiLabelBinarizer pattern)
    family_lists = grouped['variant_families'].fillna('').str.split(',')
    fam_df = pd.DataFrame(
        [{fam: 1 for fam in fams if fam} for fams in family_lists],
        index=grouped.index
    ).fillna(0).astype(np.int8)
    # Only keep families that appear
    fam_df = fam_df.reindex(columns=all_families, fill_value=0)
    fam_df.columns = [f'fam_{c}' for c in fam_df.columns]

    X = pd.concat([X, fam_df], axis=1)
    X = X.fillna(0)
    feature_names = list(X.columns)

    return X.values.astype(np.float32), feature_names


def balanced_subsample(X, y, ratio=3, seed=42):
    """Subsample negatives to ratio * n_positives."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_neg = min(len(neg_idx), ratio * len(pos_idx))
    neg_sub = rng.choice(neg_idx, size=n_neg, replace=False)
    idx = np.concatenate([pos_idx, neg_sub])
    rng.shuffle(idx)
    return X[idx], y[idx], idx


def evaluate_models(X, y, feature_names, n_splits=5):
    """Run stratified CV with multiple classifiers."""
    models = {
        'RF': RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=5,
            class_weight='balanced', random_state=SEED, n_jobs=-1),
        'GBM': GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=SEED),
        'LogReg': LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, random_state=SEED),
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    results = {}

    for name, model in models.items():
        print(f"\n  --- {name} ---")
        aucs, aps = [], []
        all_y_true, all_y_prob = [], []
        importances = np.zeros(X.shape[1])

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            if name == 'LogReg':
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)

            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_te)[:, 1]

            auc = roc_auc_score(y_te, y_prob)
            ap = average_precision_score(y_te, y_prob)
            aucs.append(auc)
            aps.append(ap)
            all_y_true.extend(y_te)
            all_y_prob.extend(y_prob)

            if hasattr(model, 'feature_importances_'):
                importances += model.feature_importances_

            print(f"    Fold {fold+1}: AUC={auc:.4f}, AUPRC={ap:.4f}")

        importances /= n_splits

        print(f"  Mean AUC:   {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
        print(f"  Mean AUPRC: {np.mean(aps):.4f} +/- {np.std(aps):.4f}")

        # Confusion matrix at 0.5
        y_pred = (np.array(all_y_prob) >= 0.5).astype(int)
        cm = confusion_matrix(all_y_true, y_pred)
        print(f"  Confusion matrix:")
        print(f"    TP={cm[1,1]}, FP={cm[0,1]}, TN={cm[0,0]}, FN={cm[1,0]}")

        # Top features
        if importances.sum() > 0:
            top_idx = np.argsort(importances)[::-1][:15]
            print(f"  Top 15 features:")
            for i in top_idx:
                print(f"    {feature_names[i]:30s} {importances[i]:.4f}")

        results[name] = {
            'auc_mean': float(np.mean(aucs)),
            'auc_std': float(np.std(aucs)),
            'auprc_mean': float(np.mean(aps)),
            'auprc_std': float(np.std(aps)),
            'aucs': [float(a) for a in aucs],
            'confusion_matrix': cm.tolist(),
        }

    return results


def phase2_embeddings(grouped, causal_set, feature_names_phase1, X_phase1, y):
    """Add Agro-NT embeddings for causal bQTL + matched negatives."""
    frozen_path = RESULTS / 'agro_nt_embeddings_frozen.npy'
    lora_path = RESULTS / 'agro_nt_embeddings_lora.npy'
    pos_path = RESULTS / 'agro_nt_embedding_positions.csv'

    if not frozen_path.exists() or not pos_path.exists():
        print("\n  Phase 2 skipped: embeddings not found")
        return None

    print("\n" + "=" * 70)
    print("  PHASE 2: EMBEDDING-BASED PREDICTION (causal-only subset)")
    print("=" * 70)

    emb_frozen = np.load(frozen_path)
    pos_df = pd.read_csv(pos_path)

    has_lora = lora_path.exists()
    if has_lora:
        emb_lora = np.load(lora_path)

    print(f"  Frozen embeddings: {emb_frozen.shape}")
    if has_lora:
        print(f"  LoRA embeddings: {emb_lora.shape}")

    # PCA reduce embeddings
    n_components = 50
    pca_frozen = PCA(n_components=n_components, random_state=SEED)
    emb_frozen_pca = pca_frozen.fit_transform(emb_frozen)
    print(f"  Frozen PCA: {n_components} components, "
          f"explained variance = {pca_frozen.explained_variance_ratio_.sum():.3f}")

    if has_lora:
        pca_lora = PCA(n_components=n_components, random_state=SEED)
        emb_lora_pca = pca_lora.fit_transform(emb_lora)
        print(f"  LoRA PCA: {n_components} components, "
              f"explained variance = {pca_lora.explained_variance_ratio_.sum():.3f}")

    # We only have embeddings for causal bQTL — use a within-causal
    # analysis to see if embeddings predict effect size (|Cohen's d|)
    causal_df = pd.read_csv(RESULTS / 'causal_bqtl_728.csv')

    # Merge embeddings with causal data
    print(f"\n  --- Embedding → Effect Size Prediction (within causal) ---")
    abs_d = causal_df['abs_d'].values

    # Median split: strong vs weak causal bQTL
    median_d = np.median(abs_d)
    y_strong = (abs_d >= median_d).astype(int)
    print(f"  Median |Cohen's d|: {median_d:.2f}")
    print(f"  Strong (>= median): {y_strong.sum()}, Weak: {(y_strong == 0).sum()}")

    for emb_name, emb_pca in [('Frozen', emb_frozen_pca)] + \
            ([('LoRA', emb_lora_pca)] if has_lora else []):
        print(f"\n  {emb_name} embeddings → strong/weak classification:")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        aucs = []
        for fold, (tr, te) in enumerate(skf.split(emb_pca, y_strong)):
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=8, class_weight='balanced',
                random_state=SEED, n_jobs=-1)
            rf.fit(emb_pca[tr], y_strong[tr])
            auc = roc_auc_score(y_strong[te], rf.predict_proba(emb_pca[te])[:, 1])
            aucs.append(auc)
        print(f"    AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    # Also: can motif features (for causal bQTL only) predict effect size?
    print(f"\n  --- Motif features → strong/weak (within causal) ---")
    # Match causal bQTL to their motif features
    causal_positions = list(zip(causal_df.chr, causal_df.pos))
    grouped_idx = {(r.chr, r.pos): i for i, r in grouped.iterrows()}
    causal_motif_idx = [grouped_idx.get(p) for p in causal_positions]
    valid = [i for i, idx in enumerate(causal_motif_idx) if idx is not None]
    X_causal_motif = X_phase1[[causal_motif_idx[i] for i in valid]]
    y_strong_valid = y_strong[valid]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    for fold, (tr, te) in enumerate(skf.split(X_causal_motif, y_strong_valid)):
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight='balanced',
            random_state=SEED, n_jobs=-1)
        rf.fit(X_causal_motif[tr], y_strong_valid[tr])
        auc = roc_auc_score(y_strong_valid[te],
                            rf.predict_proba(X_causal_motif[te])[:, 1])
        aucs.append(auc)
    print(f"    AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    # Combined: motif + embeddings
    print(f"\n  --- Combined (motif + frozen + LoRA) → strong/weak ---")
    parts = [X_causal_motif, emb_frozen_pca[valid]]
    if has_lora:
        parts.append(emb_lora_pca[valid])
    X_combined = np.hstack(parts)
    aucs = []
    for fold, (tr, te) in enumerate(skf.split(X_combined, y_strong_valid)):
        rf = RandomForestClassifier(
            n_estimators=500, max_depth=10, class_weight='balanced',
            random_state=SEED, n_jobs=-1)
        rf.fit(X_combined[tr], y_strong_valid[tr])
        auc = roc_auc_score(y_strong_valid[te],
                            rf.predict_proba(X_combined[te])[:, 1])
        aucs.append(auc)
    print(f"    AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    return {'median_cohens_d': float(median_d)}


def main():
    print("=" * 70)
    print("  CAUSALITY PREDICTION MODEL FOR bQTL")
    print("=" * 70)

    motifs_path = RESULTS / 'bound_region_motifs.csv'
    causal_path = RESULTS / 'causal_bqtl_728.csv'

    # Phase 1: Motif + distance features
    print("\n" + "=" * 70)
    print("  PHASE 1: MOTIF + DISTANCE FEATURES")
    print("=" * 70)

    grouped, causal_set = load_motif_features(motifs_path, causal_path)
    X_full, feature_names = build_feature_matrix(grouped)
    y_full = grouped['causal'].values

    print(f"\n  Feature matrix: {X_full.shape}")
    print(f"  Class balance: {y_full.sum()} causal / {(y_full==0).sum()} non-causal")

    # Subsample negatives (3:1 ratio)
    print("\n  Subsampling negatives (3:1 ratio)...")
    X_sub, y_sub, sub_idx = balanced_subsample(X_full, y_full, ratio=3, seed=SEED)
    print(f"  Subsampled: {X_sub.shape[0]} ({y_sub.sum()} pos, "
          f"{(y_sub==0).sum()} neg)")

    results_phase1 = evaluate_models(X_sub, y_sub, feature_names)

    # Also test with 1:1 ratio
    print("\n\n  === Balanced 1:1 ratio ===")
    X_bal, y_bal, _ = balanced_subsample(X_full, y_full, ratio=1, seed=SEED)
    print(f"  Balanced: {X_bal.shape[0]} ({y_bal.sum()} pos, {(y_bal==0).sum()} neg)")
    results_balanced = evaluate_models(X_bal, y_bal, feature_names)

    # Also test on full imbalanced data
    print("\n\n  === Full imbalanced data ===")
    print(f"  Full: {X_full.shape[0]} ({y_full.sum()} pos, {(y_full==0).sum()} neg)")
    results_full = evaluate_models(X_full, y_full, feature_names)

    # Phase 2: Embeddings (within-causal effect size prediction)
    emb_results = phase2_embeddings(
        grouped, causal_set, feature_names, X_full, y_full)

    # Save results
    all_results = {
        'phase1_3to1': results_phase1,
        'phase1_1to1': results_balanced,
        'phase1_full': results_full,
    }
    if emb_results:
        all_results['phase2_embeddings'] = emb_results

    out_path = RESULTS / 'causality_prediction_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
