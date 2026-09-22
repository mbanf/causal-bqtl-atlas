#!/usr/bin/env python3
"""
63_select_causal_728.py — Select 728 high-confidence causal bQTL.

Part of: "A high-confidence causal bQTL atlas" paper (Banf & Hartwig)
Paper section: "A six-step pipeline identifies 728 high-confidence causal bQTL"
Pipeline step: after 54 (genotype-aware test), before 65-72 (characterization)

Filtering pipeline (Table 1 in paper):
  Step 1: ~237,000 MOA-seq binding peaks per hybrid
  Step 2: 147,942 bQTL positions (allelic binding differences)
  Step 3: 19,368 bQTL-gene pairs (bQTL within 2kb of TSS + ASE data available)
  Step 4: 1,403 significant pairs (genotype→ASE, FDR < 0.05, Mann-Whitney)
  Step 5: 772 genes with exactly one significant bQTL (no LD ambiguity)
  Step 6: 728 genes where bQTL falls within at least one hybrid's MOA-seq peak

Steps 1-3 are handled by data preparation.
Step 4 is performed by script 54 (genotype_bqtl_results.csv).
Steps 5-6 are performed here.

Input:
  results/genotype_bqtl_results.csv   (from script 54)
  results/pan_cistrome_peaks.csv       (from script 48)
  data/processed/engelhorn_ww_vs_ds_expression.tsv
  data/raw/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1_entap_results.tsv.gz
  results/bound_region_motifs.csv      (from script 47, for TF family annotation)

Output:
  results/causal_bqtl_728.csv          (728 rows: core causal bQTL)
  results/regulatory_map_728.csv       (728 rows: annotated with TF families, function, drought)
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results"

print("=" * 70)
print("  STEP 63: Select 728 high-confidence causal bQTL")
print("=" * 70)

# ── Step 4 output: genotype→ASE results ──
geno = pd.read_csv(RESULTS / "genotype_bqtl_results.csv")
print(f"\n  Genotype→ASE results: {len(geno):,} bQTL-gene pairs")

# Filter to significant (FDR < 0.05)
sig = geno[geno["fdr_mw"] < 0.05].copy()
print(f"  Step 4 — FDR < 0.05: {len(sig):,} significant pairs")
print(f"           Unique genes: {sig['gene_id'].nunique():,}")

# ── Step 5: Single-bQTL per gene (no LD ambiguity) ──
bqtl_per_gene = sig.groupby("gene_id").size()
single_bqtl_genes = bqtl_per_gene[bqtl_per_gene == 1].index
multi_bqtl_genes = bqtl_per_gene[bqtl_per_gene > 1].index

single = sig[sig["gene_id"].isin(single_bqtl_genes)].copy()
print(f"  Step 5 — Single-bQTL per gene: {len(single):,} genes")
print(f"           Dropped (multi-bQTL): {len(multi_bqtl_genes):,} genes")

# ── Step 6: bQTL must fall within MOA-seq peak ──
peaks = pd.read_csv(RESULTS / "pan_cistrome_peaks.csv")
print(f"\n  Pan-cistrome peaks: {len(peaks):,}")

# Build chromosome-indexed lookup for speed
peak_dict = {}
for _, p in peaks.iterrows():
    peak_dict.setdefault(p["chr"], []).append((p["start"], p["end"]))

def in_peak(chrom, pos):
    for start, end in peak_dict.get(chrom, []):
        if start <= pos <= end:
            return True
    return False

single["in_peak"] = single.apply(lambda r: in_peak(r["chr"], r["pos"]), axis=1)
causal = single[single["in_peak"]].copy()
not_in_peak = len(single) - len(causal)
print(f"  Step 6 — In MOA-seq peak: {len(causal):,} genes")
print(f"           Dropped (not in peak): {not_in_peak:,} genes")

# ── Build causal_bqtl_728.csv ──
# Standardize columns to match expected format
causal_out = causal[["chr", "pos", "gene_id"]].copy()
causal_out["cohens_d"] = causal["cohens_d"]
causal_out["abs_d"] = causal["cohens_d"].abs()
causal_out["fdr"] = causal["fdr_mw"]
causal_out["n_variant"] = causal["n_variant"]
causal_out["n_ref"] = causal["n_reference"]
causal_out["n_missing"] = 19 - causal["n_variant"] - causal["n_reference"]
causal_out["variant_hybrids"] = causal["variant_hybrids"] if "variant_hybrids" in causal.columns else ""
causal_out["ref_hybrids"] = causal["ref_hybrids"] if "ref_hybrids" in causal.columns else ""

causal_out.to_csv(RESULTS / "causal_bqtl_728.csv", index=False)
print(f"\n  Saved: results/causal_bqtl_728.csv ({len(causal_out)} genes)")

# ── Build regulatory_map_728.csv (annotated version) ──
reg = causal_out.copy()

# Add TF families at variant (from bound_region_motifs if available)
brm_path = RESULTS / "bound_region_motifs.csv"
if brm_path.exists():
    brm = pd.read_csv(brm_path)
    # Get families per bQTL
    if "tf_family" in brm.columns:
        fam = brm.groupby(["chr", "pos"])["tf_family"].apply(
            lambda x: "|".join(sorted(x.unique()))
        ).reset_index()
        fam.columns = ["chr", "pos", "families_at_variant"]
        reg = reg.merge(fam, on=["chr", "pos"], how="left")
    else:
        reg["families_at_variant"] = ""
else:
    reg["families_at_variant"] = ""
    print("  Warning: bound_region_motifs.csv not found, skipping TF family annotation")

# Add functional annotation (EntAP)
entap_path = DATA / "raw" / "Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1_entap_results.tsv.gz"
if entap_path.exists():
    entap = pd.read_csv(entap_path, sep="\t", compression="gzip", low_memory=False)
    # Try common column names
    for id_col in ["Query Sequence", "query_sequence", "Gene ID"]:
        if id_col in entap.columns:
            break
    desc_col = next((c for c in entap.columns if "Description" in c), None)
    cog_col = next((c for c in entap.columns if "EggNOG" in c and "Description" in c), None)

    if desc_col:
        ann = entap[[id_col, desc_col]].copy()
        if cog_col:
            ann[cog_col] = entap[cog_col]
        ann.columns = ["gene_id", "Description"] + (["EggNOG COG Description"] if cog_col else [])
        # Gene IDs in EntAP may have transcript suffix
        ann["gene_id"] = ann["gene_id"].str.replace(r"_T\d+$", "", regex=True)
        ann = ann.drop_duplicates("gene_id")
        reg = reg.merge(ann, on="gene_id", how="left")
    else:
        reg["Description"] = ""
else:
    reg["Description"] = ""
    print("  Warning: EntAP annotations not found")

# Add drought expression
expr_path = DATA / "processed" / "engelhorn_ww_vs_ds_expression.tsv"
if expr_path.exists():
    expr = pd.read_csv(expr_path, sep="\t")
    fc_col = next((c for c in expr.columns if "log2" in c.lower() or "fc" in c.lower()), None)
    if fc_col:
        expr_sub = expr[["gene_id", fc_col]].drop_duplicates("gene_id")
        expr_sub.columns = ["gene_id", "log2fc_DS_vs_WW"]
        reg = reg.merge(expr_sub, on="gene_id", how="left")
    else:
        reg["log2fc_DS_vs_WW"] = np.nan
else:
    reg["log2fc_DS_vs_WW"] = np.nan

# Add functional category (simplified)
# This is a placeholder — the full categorization was done manually/semi-automatically
reg["func_category"] = "Unknown"

# Add sharing category
reg["sharing"] = pd.cut(
    reg["n_variant"],
    bins=[0, 4, 8, 12, 19],
    labels=["Rare (1-4)", "Medium (5-8)", "Common (9-12)", "Widespread (13+)"]
)

# Add drought category
if "log2fc_DS_vs_WW" in reg.columns:
    reg["drought_cat"] = pd.cut(
        reg["log2fc_DS_vs_WW"].fillna(0),
        bins=[-np.inf, -1, 1, np.inf],
        labels=["Strong down", "Stable", "Strong up"]
    )
else:
    reg["drought_cat"] = "Unknown"

reg.to_csv(RESULTS / "regulatory_map_728.csv", index=False)
print(f"  Saved: results/regulatory_map_728.csv ({len(reg)} genes)")

# ── Summary ──
print(f"\n{'=' * 70}")
print(f"  SUMMARY")
print(f"{'=' * 70}")
print(f"  Total bQTL-gene pairs tested:    {len(geno):>8,}")
print(f"  Significant (FDR < 0.05):        {len(sig):>8,}")
print(f"  Single-bQTL per gene:            {len(single):>8,}")
print(f"  In MOA-seq peak (final):         {len(causal):>8,}")
print(f"  Median |Cohen's d|:              {causal_out['abs_d'].median():>8.2f}")
print(f"  Median variant hybrids:          {causal_out['n_variant'].median():>8.0f}")
print(f"  Effect size range:               {causal_out['abs_d'].min():.2f} — {causal_out['abs_d'].max():.2f}")
print(f"{'=' * 70}")
