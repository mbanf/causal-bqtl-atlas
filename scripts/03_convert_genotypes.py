#!/usr/bin/env python3
"""
53c_convert_julia_genotypes.py — Convert Julia's genotype data to pipeline format.

Julia provided per-chromosome genotype files for all 25 hybrids at all SNP
positions in MOA peaks (1.49M positions). This script:
1. Reads all 10 chromosome files + the ref/alt allele file
2. Filters to our 147,942 bQTL positions
3. Converts encoding: -1 → 0|0 (ref), 0 → actual alt allele, NaN → ./.
4. Outputs in pipeline format: chr, pos, ref, <hybrid1>, ..., <hybridN>
5. Excludes M37W (no RNA-seq data)

Julia's encoding (from her message):
  -1 = no SNP present → same allele as B73 (reference)
   0 = SNP present → variant allele
  NaN = missing data

Input:
  NAM_genotypes_2FP_WW/genotypes_divided_2FPs_{1..10}.csv
  NAM_genotypes_2FP_WW/All_FPs_WW_include_Ref_Alt_uniq.csv
  data/processed/bqtl_snp_ww.csv

Output:
  data/processed/nam_founder_genotypes_at_bqtl.tsv (replaces 53b output)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
BQTL_FILE = DATA / "processed" / "bqtl_snp_ww.csv"
OUTPUT = DATA / "processed" / "nam_founder_genotypes_at_bqtl.tsv"

# Julia's data location
JULIA_DIR = BASE.parent / "NAM_genotypes_2FP_WW"

# Hybrids to exclude (no RNA-seq)
EXCLUDE_HYBRIDS = {"M37W"}

print("=" * 70)
print("  53c: Convert Julia's genotype data to pipeline format")
print("=" * 70)

# ── Check inputs ──
if not JULIA_DIR.exists():
    print(f"ERROR: Julia's genotype directory not found: {JULIA_DIR}")
    print("Expected: NAM_genotypes_2FP_WW/ in parent directory")
    sys.exit(1)

# ── Load bQTL positions ──
bqtl = pd.read_csv(BQTL_FILE)
bqtl["julia_chrom"] = "B73-" + bqtl["chr"]
bqtl_keys = set(zip(bqtl["julia_chrom"], bqtl["pos"]))
print(f"\n  bQTL positions: {len(bqtl_keys):,}")

# ── Load ref/alt allele file ──
print("  Loading ref/alt alleles...")
refalt = pd.read_csv(
    JULIA_DIR / "All_FPs_WW_include_Ref_Alt_uniq.csv",
    sep="\t", header=None, names=["chrom", "pos", "ref", "alt"]
)
# Strip quotes from ref/alt
refalt["ref"] = refalt["ref"].str.strip('"')
refalt["alt"] = refalt["alt"].str.strip('"')
refalt_lookup = {}
for _, row in refalt.iterrows():
    refalt_lookup[(row["chrom"], row["pos"])] = (row["ref"], row["alt"])
print(f"  Ref/alt entries: {len(refalt_lookup):,}")

# ── Process chromosome files ──
all_rows = []
n_found = 0
n_missing_refalt = 0

for chrom_num in range(1, 11):
    filepath = JULIA_DIR / f"genotypes_divided_2FPs_{chrom_num}.csv"
    print(f"\n  Processing chromosome {chrom_num}...")
    df = pd.read_csv(filepath)

    # Identify hybrid columns
    hybrid_cols = [c for c in df.columns if c not in ["#CHROM", "POS"]]
    # Exclude M37W
    hybrid_cols = [c for c in hybrid_cols if c not in EXCLUDE_HYBRIDS]

    # Filter to bQTL positions
    df["_key"] = list(zip(df["#CHROM"], df["POS"]))
    bqtl_mask = df["_key"].isin(bqtl_keys)
    df_bqtl = df[bqtl_mask].copy()
    n_found += len(df_bqtl)
    print(f"    Total positions: {len(df):,}, bQTL matches: {len(df_bqtl):,}")

    for _, row in df_bqtl.iterrows():
        chrom_julia = row["#CHROM"]  # e.g., "B73-chr1"
        pos = row["POS"]
        chrom_clean = chrom_julia.replace("B73-", "")  # e.g., "chr1"

        # Look up ref/alt alleles
        ra = refalt_lookup.get((chrom_julia, pos))
        if ra is None:
            n_missing_refalt += 1
            ref_allele = "N"
            alt_allele = "N"
        else:
            ref_allele, alt_allele = ra

        out_row = {"chr": chrom_clean, "pos": pos, "ref": ref_allele}

        for hybrid in hybrid_cols:
            val = row[hybrid]
            if pd.isna(val):
                out_row[hybrid] = "./."
            elif val == -1:
                out_row[hybrid] = "0|0"  # Same as B73 reference
            elif val == 0:
                out_row[hybrid] = alt_allele  # Has the variant allele
            else:
                out_row[hybrid] = "./."  # Unexpected value

        all_rows.append(out_row)

result = pd.DataFrame(all_rows)
# Sort by chromosome and position
chrom_order = {f"chr{i}": i for i in range(1, 11)}
result["_chrom_sort"] = result["chr"].map(chrom_order)
result = result.sort_values(["_chrom_sort", "pos"]).drop(columns=["_chrom_sort"])

print(f"\n{'=' * 70}")
print(f"  RESULTS")
print(f"{'=' * 70}")
print(f"  bQTL positions found: {n_found:,} / {len(bqtl_keys):,} ({100*n_found/len(bqtl_keys):.1f}%)")
print(f"  Missing ref/alt:      {n_missing_refalt:,}")
print(f"  Hybrids:              {len([c for c in result.columns if c not in ['chr','pos','ref']])}")
print(f"  Hybrid names:         {[c for c in result.columns if c not in ['chr','pos','ref']]}")

# Per-hybrid stats
hybrid_cols_out = [c for c in result.columns if c not in ["chr", "pos", "ref"]]
print(f"\n  Per-hybrid genotype stats:")
for h in hybrid_cols_out:
    n_ref = (result[h] == "0|0").sum()
    n_var = result[h].isin(list("ACGT")).sum()
    n_miss = (result[h] == "./.").sum()
    pct_covered = 100 * (n_ref + n_var) / len(result)
    print(f"    {h:8s}: ref={n_ref:>7,}  var={n_var:>7,}  miss={n_miss:>6,}  ({pct_covered:.1f}% covered)")

# ── Save ──
# Back up old file if it exists
old_file = OUTPUT
if old_file.exists():
    backup = old_file.with_suffix(".tsv.bak_19hybrids")
    if not backup.exists():
        old_file.rename(backup)
        print(f"\n  Backed up old 19-hybrid file to: {backup.name}")

result.to_csv(OUTPUT, sep="\t", index=False)
print(f"\n  Saved: {OUTPUT}")
print(f"  Shape: {result.shape[0]:,} positions × {result.shape[1]} columns")
print(f"{'=' * 70}")
