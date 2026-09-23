#!/usr/bin/env python3
"""
exp_01_multi_bqtl_sole_motif.py — Can we recover causal bQTL from multi-bQTL genes?

Thomas's comment (L62/63): For genes with >1 significant bQTL, check if only
one bQTL disrupts a sole-copy (non-redundant) motif. If so, that bQTL is the
likely causal variant — the others may be in LD or affect redundant sites.

Analysis:
1. Take the 631 genes dropped at Step 5 (multi-bQTL per gene)
2. For each bQTL, check motif content in the surrounding peak
3. Identify bQTL where only ONE disrupts a sole-copy motif
4. Report how many genes could be recovered

Input:
  results/genotype_bqtl_results.csv
  results/bound_region_motifs.csv
  results/pan_cistrome_peaks.csv

Output:
  results/experimental/multi_bqtl_sole_motif_recovery.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
OUT = RESULTS / "experimental"
OUT.mkdir(exist_ok=True)

print("=" * 70)
print("  EXPERIMENT 1: Multi-bQTL gene recovery via sole-copy motif filter")
print("=" * 70)

# ── Load data ──
geno = pd.read_csv(RESULTS / "genotype_bqtl_results.csv")
brm = pd.read_csv(RESULTS / "bound_region_motifs.csv")
peaks = pd.read_csv(RESULTS / "pan_cistrome_peaks.csv")

sig = geno[geno["fdr_mw"] < 0.05].copy()
print(f"\n  Significant bQTL-gene pairs: {len(sig):,}")

# ── Identify multi-bQTL genes ──
bqtl_per_gene = sig.groupby("gene_id").size()
multi_genes = bqtl_per_gene[bqtl_per_gene > 1].index
single_genes = bqtl_per_gene[bqtl_per_gene == 1].index

multi = sig[sig["gene_id"].isin(multi_genes)].copy()
print(f"  Single-bQTL genes: {len(single_genes):,}")
print(f"  Multi-bQTL genes:  {len(multi_genes):,} ({len(multi):,} bQTL total)")
print(f"  bQTL per multi-gene: {bqtl_per_gene[multi_genes].describe().to_dict()}")

# ── Build peak lookup ──
peak_dict = {}
for _, p in peaks.iterrows():
    peak_dict.setdefault(p["chr"], []).append((p["start"], p["end"]))

def in_peak(chrom, pos):
    for start, end in peak_dict.get(chrom, []):
        if start <= pos <= end:
            return True
    return False

# ── For each multi-bQTL gene, check motif content ──
# bound_region_motifs has: chr, pos, tf_family, motif_id, score, etc.
# We need to check if a motif at each bQTL position is sole-copy in the peak

# First, understand the structure
print(f"\n  bound_region_motifs columns: {list(brm.columns)}")
print(f"  bound_region_motifs rows: {len(brm):,}")

# Check which columns indicate sole-copy status
sole_cols = [c for c in brm.columns if "sole" in c.lower() or "count" in c.lower() or "copy" in c.lower()]
print(f"  Sole-related columns: {sole_cols}")

# Get motif info per bQTL position
# Group by (chr, pos) to get families at each bQTL
if "tf_family" in brm.columns:
    motif_at_pos = brm.groupby(["chr", "pos"]).agg(
        n_families=("tf_family", "nunique"),
        families=("tf_family", lambda x: "|".join(sorted(x.unique()))),
    ).reset_index()
else:
    print("  ERROR: tf_family column not found in bound_region_motifs.csv")
    exit(1)

# Check if we have sole-copy info from script 68
sole_path = RESULTS / "sole_motif_verification_728.csv"
if sole_path.exists():
    sole_728 = pd.read_csv(sole_path)
    print(f"\n  Sole motif verification (728): {len(sole_728)} rows")
    sole_cols_728 = [c for c in sole_728.columns if "sole" in c.lower()]
    print(f"  Sole columns: {sole_cols_728}")

# ── Analyze motif content at each multi-bQTL position ──
# For each bQTL in a multi-gene, count how many TF families it disrupts
multi_with_motifs = multi.merge(
    motif_at_pos, on=["chr", "pos"], how="left"
)
multi_with_motifs["has_motif"] = multi_with_motifs["n_families"].notna()
multi_with_motifs["n_families"] = multi_with_motifs["n_families"].fillna(0).astype(int)

print(f"\n  Multi-bQTL with motif data: {multi_with_motifs['has_motif'].sum():,} / {len(multi_with_motifs):,}")

# ── Check if bQTL is in a peak ──
multi_with_motifs["in_peak"] = multi_with_motifs.apply(
    lambda r: in_peak(r["chr"], r["pos"]), axis=1
)
print(f"  Multi-bQTL in MOA-seq peak: {multi_with_motifs['in_peak'].sum():,} / {len(multi_with_motifs):,}")

# ── For each gene, check if exactly ONE bQTL has motifs and is in peak ──
results = []
for gene_id, grp in multi_with_motifs.groupby("gene_id"):
    n_bqtl = len(grp)
    n_in_peak = grp["in_peak"].sum()
    n_with_motif = (grp["n_families"] > 0).sum()

    # Strategy 1: only one bQTL is in a peak
    sole_peak = n_in_peak == 1
    # Strategy 2: only one bQTL has motif content
    sole_motif = n_with_motif == 1
    # Strategy 3: both — one bQTL in peak AND with motif
    sole_both = False
    candidate_idx = None
    in_peak_and_motif = grp[(grp["in_peak"]) & (grp["n_families"] > 0)]
    if len(in_peak_and_motif) == 1:
        sole_both = True
        candidate_idx = in_peak_and_motif.index[0]

    # Strategy 4: one bQTL has unique/rare motif families not shared by others
    # (more complex — check if one bQTL has families that no other bQTL in the gene has)
    unique_family_bqtl = None
    if n_with_motif > 1:
        family_sets = {}
        for idx, row in grp.iterrows():
            if row["n_families"] > 0:
                family_sets[idx] = set(row["families"].split("|"))
        # Check if any bQTL has families not present in any other
        for idx, fams in family_sets.items():
            other_fams = set()
            for other_idx, other_f in family_sets.items():
                if other_idx != idx:
                    other_fams |= other_f
            unique = fams - other_fams
            if len(unique) > 0 and unique_family_bqtl is None:
                unique_family_bqtl = idx

    best = grp.loc[candidate_idx] if candidate_idx is not None else None

    results.append({
        "gene_id": gene_id,
        "n_bqtl": n_bqtl,
        "n_in_peak": n_in_peak,
        "n_with_motif": n_with_motif,
        "recoverable_peak_only": sole_peak,
        "recoverable_motif_only": sole_motif,
        "recoverable_peak_and_motif": sole_both,
        "has_unique_family": unique_family_bqtl is not None,
        "best_chr": best["chr"] if best is not None else None,
        "best_pos": best["pos"] if best is not None else None,
        "best_cohens_d": best["cohens_d"] if best is not None else None,
        "best_fdr": best["fdr_mw"] if best is not None else None,
        "best_families": best["families"] if best is not None else None,
    })

rdf = pd.DataFrame(results)

# ── Summary ──
print(f"\n{'=' * 70}")
print(f"  RESULTS: Recovery of multi-bQTL genes")
print(f"{'=' * 70}")
print(f"  Total multi-bQTL genes:                     {len(rdf):>6,}")
print(f"")
print(f"  Strategy 1 — Only 1 bQTL in MOA-seq peak:   {rdf['recoverable_peak_only'].sum():>6,}  ({100*rdf['recoverable_peak_only'].mean():.1f}%)")
print(f"  Strategy 2 — Only 1 bQTL with TF motif:     {rdf['recoverable_motif_only'].sum():>6,}  ({100*rdf['recoverable_motif_only'].mean():.1f}%)")
print(f"  Strategy 3 — Only 1 in peak AND with motif:  {rdf['recoverable_peak_and_motif'].sum():>6,}  ({100*rdf['recoverable_peak_and_motif'].mean():.1f}%)")
print(f"  Strategy 4 — Has unique TF family:           {rdf['has_unique_family'].sum():>6,}  ({100*rdf['has_unique_family'].mean():.1f}%)")
print(f"")

# Breakdown by number of bQTL
print(f"  Recovery rate by bQTL count:")
for n in sorted(rdf["n_bqtl"].unique()):
    sub = rdf[rdf["n_bqtl"] == n]
    rec = sub["recoverable_peak_and_motif"].sum()
    print(f"    {n} bQTL: {rec}/{len(sub)} recoverable ({100*rec/len(sub):.1f}%)")

# Effect sizes of recoverable vs non-recoverable
recoverable = rdf[rdf["recoverable_peak_and_motif"]]
if len(recoverable) > 0:
    print(f"\n  Recovered bQTL effect sizes:")
    print(f"    Median |Cohen's d|: {recoverable['best_cohens_d'].abs().median():.2f}")
    print(f"    Median FDR:         {recoverable['best_fdr'].median():.4f}")

    # Compare with the 728
    causal = pd.read_csv(RESULTS / "causal_bqtl_728.csv")
    print(f"\n  For comparison — 728 causal bQTL:")
    print(f"    Median |Cohen's d|: {causal['abs_d'].median():.2f}")
    print(f"    Median FDR:         {causal['fdr'].median():.4f}")

print(f"\n  Conclusion: {rdf['recoverable_peak_and_motif'].sum()} genes could potentially")
print(f"  be added to the 728 atlas (educated guesses, as Thomas notes).")
print(f"{'=' * 70}")

# ── Save ──
rdf.to_csv(OUT / "multi_bqtl_sole_motif_recovery.csv", index=False)
print(f"\n  Saved: results/experimental/multi_bqtl_sole_motif_recovery.csv")

# Also save the actual recovered bQTL for potential inclusion
if len(recoverable) > 0:
    recovered_bqtl = multi_with_motifs[
        multi_with_motifs.apply(
            lambda r: any(
                (r["gene_id"] == row["gene_id"]) and
                (r["chr"] == row["best_chr"]) and
                (r["pos"] == row["best_pos"])
                for _, row in recoverable.iterrows()
            ), axis=1
        )
    ] if len(recoverable) < 50 else pd.DataFrame()  # avoid O(n²) for large sets

    # Simpler approach: merge
    rec_keys = recoverable[["gene_id", "best_chr", "best_pos"]].rename(
        columns={"best_chr": "chr", "best_pos": "pos"}
    )
    recovered_detail = multi_with_motifs.merge(rec_keys, on=["gene_id", "chr", "pos"])
    recovered_detail.to_csv(OUT / "recovered_bqtl_candidates.csv", index=False)
    print(f"  Saved: results/experimental/recovered_bqtl_candidates.csv ({len(recovered_detail)} rows)")
