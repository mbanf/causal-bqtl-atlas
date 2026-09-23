#!/usr/bin/env python3
"""
exp_02_peak_motif_conservation.py — TF family conservation at bQTL vs non-bQTL peaks

Thomas's comment (L176/177): The 728 causal bQTL disrupt 2-5 TF families at the
variant position. How does this compare to other peaks? Are non-bQTL peaks more
or less conserved in their motif content?

Analysis:
1. Take all pan-cistrome peaks (291K)
2. Split into: peaks containing a causal bQTL vs peaks without any bQTL
3. Compare motif density and TF family diversity between groups
4. Test whether bQTL peaks are more "promiscuous" (more families) or
   whether the 2-5 families is typical for any peak

Input:
  results/pan_cistrome_peaks.csv
  results/causal_bqtl_728.csv
  results/bound_region_motifs.csv
  data/processed/bqtl_snp_ww.csv
  data/processed/maize_tf_pwm_database.json

Output:
  results/experimental/peak_motif_conservation.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
DATA = BASE / "data"
OUT = RESULTS / "experimental"
OUT.mkdir(exist_ok=True)

print("=" * 70)
print("  EXPERIMENT 2: TF motif conservation at bQTL vs non-bQTL peaks")
print("=" * 70)

# ── Load data ──
peaks = pd.read_csv(RESULTS / "pan_cistrome_peaks.csv")
causal = pd.read_csv(RESULTS / "causal_bqtl_728.csv")
brm = pd.read_csv(RESULTS / "bound_region_motifs.csv")
bqtl_all = pd.read_csv(DATA / "processed" / "bqtl_snp_ww.csv")

print(f"\n  Pan-cistrome peaks: {len(peaks):,}")
print(f"  Causal bQTL: {len(causal):,}")
print(f"  All bQTL: {len(bqtl_all):,}")

# ── Load PWM database for scanning non-bQTL peaks ──
with open(DATA / "processed" / "maize_tf_pwm_database.json") as f:
    pwm_db = json.load(f)
print(f"  PWM database: {len(pwm_db)} motifs")

# ── Classify peaks ──
# For each peak, check if it contains a causal bQTL, any bQTL, or none
def classify_peaks(peaks_df, causal_df, bqtl_df):
    """Label each peak as 'causal_bqtl', 'other_bqtl', or 'no_bqtl'."""
    # Build position sets
    causal_set = set(zip(causal_df["chr"], causal_df["pos"]))

    # Get bQTL positions — need chr and pos columns
    bqtl_chr_col = next((c for c in bqtl_df.columns if c.lower() in ["chr", "chrom", "chromosome"]), None)
    bqtl_pos_col = next((c for c in bqtl_df.columns if c.lower() in ["pos", "position", "bp"]), None)

    if bqtl_chr_col is None or bqtl_pos_col is None:
        print(f"  bQTL columns: {list(bqtl_df.columns)}")
        # Try to infer
        bqtl_chr_col = bqtl_df.columns[0]
        bqtl_pos_col = bqtl_df.columns[1]

    bqtl_set = set(zip(bqtl_df[bqtl_chr_col], bqtl_df[bqtl_pos_col]))

    labels = []
    for _, peak in peaks_df.iterrows():
        has_causal = False
        has_bqtl = False
        for pos_set, label in [(causal_set, "causal"), (bqtl_set, "bqtl")]:
            for chrom, pos in pos_set:
                if chrom == peak["chr"] and peak["start"] <= pos <= peak["end"]:
                    if label == "causal":
                        has_causal = True
                    has_bqtl = True
        if has_causal:
            labels.append("causal_bqtl")
        elif has_bqtl:
            labels.append("other_bqtl")
        else:
            labels.append("no_bqtl")
    return labels

# This is too slow for 291K peaks × 147K positions. Use interval approach.
print("\n  Classifying peaks (interval lookup)...")

# Build sorted position arrays per chromosome for fast lookup
from bisect import bisect_left, bisect_right

causal_by_chr = {}
for _, r in causal.iterrows():
    causal_by_chr.setdefault(r["chr"], []).append(r["pos"])
for k in causal_by_chr:
    causal_by_chr[k].sort()

bqtl_chr_col = "chr"
bqtl_pos_col = "pos"

bqtl_by_chr = {}
for _, r in bqtl_all.iterrows():
    bqtl_by_chr.setdefault(r[bqtl_chr_col], []).append(r[bqtl_pos_col])
for k in bqtl_by_chr:
    bqtl_by_chr[k].sort()

def count_in_range(sorted_arr, start, end):
    """Count positions in sorted array that fall within [start, end]."""
    left = bisect_left(sorted_arr, start)
    right = bisect_right(sorted_arr, end)
    return right - left

peak_labels = []
peak_n_causal = []
peak_n_bqtl = []

for _, peak in peaks.iterrows():
    chrom = peak["chr"]
    n_c = count_in_range(causal_by_chr.get(chrom, []), peak["start"], peak["end"])
    n_b = count_in_range(bqtl_by_chr.get(chrom, []), peak["start"], peak["end"])
    peak_n_causal.append(n_c)
    peak_n_bqtl.append(n_b)
    if n_c > 0:
        peak_labels.append("causal_bqtl")
    elif n_b > 0:
        peak_labels.append("other_bqtl")
    else:
        peak_labels.append("no_bqtl")

peaks["label"] = peak_labels
peaks["n_causal_bqtl"] = peak_n_causal
peaks["n_any_bqtl"] = peak_n_bqtl

print(f"\n  Peak classification:")
for label, count in peaks["label"].value_counts().items():
    print(f"    {label}: {count:,} ({100*count/len(peaks):.1f}%)")

# ── Get motif content from bound_region_motifs ──
# brm has motif data per (chr, pos) for bQTL positions
# For non-bQTL peaks, we need to use the peak-level summary

# bound_region_motifs already has per-position data within peaks
# columns: n_families_in_region, total_motifs_in_region, families_in_region, region_width

# Get one row per (chr, pos) with region-level stats
region_stats = brm.drop_duplicates(subset=["chr", "pos"])[
    ["chr", "pos", "n_families_in_region", "total_motifs_in_region", "region_width"]
]
print(f"\n  Region stats: {len(region_stats):,} unique bQTL positions with motif data")

# ── Compare motif diversity at variant positions ──
# For the 728 causal bQTL: how many TF families at the variant?
causal_motifs = brm[brm.apply(
    lambda r: (r["chr"], r["pos"]) in set(zip(causal["chr"], causal["pos"])),
    axis=1
)]

# More efficient
causal_set = set(zip(causal["chr"], causal["pos"]))
brm["is_causal"] = brm.apply(lambda r: (r["chr"], r["pos"]) in causal_set, axis=1)

# Families disrupted at variant position
fam_at_variant = brm.groupby(["chr", "pos", "is_causal"]).agg(
    n_families_at_variant=("tf_family", "nunique"),
    n_families_in_region=("n_families_in_region", "first"),
    total_motifs_in_region=("total_motifs_in_region", "first"),
    region_width=("region_width", "first"),
).reset_index()

causal_stats = fam_at_variant[fam_at_variant["is_causal"]]
noncausal_stats = fam_at_variant[~fam_at_variant["is_causal"]]

print(f"\n{'=' * 70}")
print(f"  RESULTS: Motif content at bQTL positions")
print(f"{'=' * 70}")

print(f"\n  A) TF families disrupted at variant position:")
print(f"     Causal 728:     median={causal_stats['n_families_at_variant'].median():.0f}, "
      f"mean={causal_stats['n_families_at_variant'].mean():.1f}, "
      f"IQR=[{causal_stats['n_families_at_variant'].quantile(0.25):.0f}-"
      f"{causal_stats['n_families_at_variant'].quantile(0.75):.0f}]")
print(f"     Other bQTL:     median={noncausal_stats['n_families_at_variant'].median():.0f}, "
      f"mean={noncausal_stats['n_families_at_variant'].mean():.1f}, "
      f"IQR=[{noncausal_stats['n_families_at_variant'].quantile(0.25):.0f}-"
      f"{noncausal_stats['n_families_at_variant'].quantile(0.75):.0f}]")
mw_stat, mw_p = stats.mannwhitneyu(
    causal_stats["n_families_at_variant"],
    noncausal_stats["n_families_at_variant"],
    alternative="two-sided"
)
print(f"     Mann-Whitney p: {mw_p:.2e}")

print(f"\n  B) Total TF families in surrounding peak/region:")
print(f"     Causal 728:     median={causal_stats['n_families_in_region'].median():.0f}, "
      f"mean={causal_stats['n_families_in_region'].mean():.1f}")
print(f"     Other bQTL:     median={noncausal_stats['n_families_in_region'].median():.0f}, "
      f"mean={noncausal_stats['n_families_in_region'].mean():.1f}")
mw2, p2 = stats.mannwhitneyu(
    causal_stats["n_families_in_region"],
    noncausal_stats["n_families_in_region"],
    alternative="two-sided"
)
print(f"     Mann-Whitney p: {p2:.2e}")

print(f"\n  C) Total motif instances in surrounding peak/region:")
print(f"     Causal 728:     median={causal_stats['total_motifs_in_region'].median():.0f}, "
      f"mean={causal_stats['total_motifs_in_region'].mean():.1f}")
print(f"     Other bQTL:     median={noncausal_stats['total_motifs_in_region'].median():.0f}, "
      f"mean={noncausal_stats['total_motifs_in_region'].mean():.1f}")

# ── Motif density (motifs per bp) ──
causal_stats_c = causal_stats.copy()
noncausal_stats_c = noncausal_stats.copy()
causal_stats_c["motif_density"] = causal_stats_c["total_motifs_in_region"] / causal_stats_c["region_width"]
noncausal_stats_c["motif_density"] = noncausal_stats_c["total_motifs_in_region"] / noncausal_stats_c["region_width"]

print(f"\n  D) Motif density (motifs/bp):")
print(f"     Causal 728:     median={causal_stats_c['motif_density'].median():.3f}")
print(f"     Other bQTL:     median={noncausal_stats_c['motif_density'].median():.3f}")
mw3, p3 = stats.mannwhitneyu(
    causal_stats_c["motif_density"].dropna(),
    noncausal_stats_c["motif_density"].dropna(),
    alternative="two-sided"
)
print(f"     Mann-Whitney p: {p3:.2e}")

# ── Peak width comparison ──
causal_peaks = peaks[peaks["label"] == "causal_bqtl"]
other_peaks = peaks[peaks["label"] == "other_bqtl"]
no_bqtl_peaks = peaks[peaks["label"] == "no_bqtl"]

print(f"\n  E) Peak width by category:")
print(f"     Causal bQTL peaks:  median={causal_peaks['width'].median():.0f}bp  (n={len(causal_peaks):,})")
print(f"     Other bQTL peaks:   median={other_peaks['width'].median():.0f}bp  (n={len(other_peaks):,})")
print(f"     No-bQTL peaks:      median={no_bqtl_peaks['width'].median():.0f}bp  (n={len(no_bqtl_peaks):,})")

# ── bQTL density per peak ──
bqtl_peaks = peaks[peaks["label"].isin(["causal_bqtl", "other_bqtl"])]
bqtl_peaks = bqtl_peaks.copy()
bqtl_peaks["bqtl_density"] = bqtl_peaks["n_any_bqtl"] / bqtl_peaks["width"] * 1000  # per kb

print(f"\n  F) bQTL density in peaks (per kb):")
print(f"     Causal peaks:  median={bqtl_peaks.loc[bqtl_peaks['label']=='causal_bqtl', 'bqtl_density'].median():.2f}")
print(f"     Other peaks:   median={bqtl_peaks.loc[bqtl_peaks['label']=='other_bqtl', 'bqtl_density'].median():.2f}")

# ── Fraction of families disrupted (at variant / in region) ──
fam_at_variant["frac_disrupted"] = fam_at_variant["n_families_at_variant"] / fam_at_variant["n_families_in_region"]

print(f"\n  G) Fraction of region's TF families disrupted at variant:")
print(f"     Causal 728:     median={fam_at_variant.loc[fam_at_variant['is_causal'], 'frac_disrupted'].median():.3f}")
print(f"     Other bQTL:     median={fam_at_variant.loc[~fam_at_variant['is_causal'], 'frac_disrupted'].median():.3f}")

print(f"\n{'=' * 70}")

# ── Figure ──
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Panel A: families at variant
ax = axes[0]
data_a = [causal_stats["n_families_at_variant"].values,
          noncausal_stats["n_families_at_variant"].values]
bp = ax.boxplot(data_a, tick_labels=["Causal\n(n=728)", f"Other bQTL\n(n={len(noncausal_stats):,})"],
                patch_artist=True, showfliers=False)
bp["boxes"][0].set_facecolor("#e74c3c")
bp["boxes"][1].set_facecolor("#95a5a6")
ax.set_ylabel("TF families disrupted at variant")
ax.set_title(f"A. Families at variant\n(p={mw_p:.1e})")

# Panel B: families in region
ax = axes[1]
data_b = [causal_stats["n_families_in_region"].values,
          noncausal_stats["n_families_in_region"].values]
bp = ax.boxplot(data_b, tick_labels=["Causal", "Other"],
                patch_artist=True, showfliers=False)
bp["boxes"][0].set_facecolor("#e74c3c")
bp["boxes"][1].set_facecolor("#95a5a6")
ax.set_ylabel("TF families in surrounding peak")
ax.set_title(f"B. Families in peak\n(p={p2:.1e})")

# Panel C: fraction disrupted
ax = axes[2]
data_c = [fam_at_variant.loc[fam_at_variant["is_causal"], "frac_disrupted"].dropna().values,
          fam_at_variant.loc[~fam_at_variant["is_causal"], "frac_disrupted"].dropna().values]
bp = ax.boxplot(data_c, tick_labels=["Causal", "Other"],
                patch_artist=True, showfliers=False)
bp["boxes"][0].set_facecolor("#e74c3c")
bp["boxes"][1].set_facecolor("#95a5a6")
ax.set_ylabel("Fraction of peak families disrupted")
ax.set_title("C. Disruption fraction")

plt.tight_layout()
fig.savefig(OUT / "peak_motif_conservation.pdf", bbox_inches="tight")
fig.savefig(OUT / "peak_motif_conservation.png", dpi=150, bbox_inches="tight")
print(f"\n  Saved: results/experimental/peak_motif_conservation.pdf")

# ── Save data ──
fam_at_variant.to_csv(OUT / "peak_motif_conservation.csv", index=False)
print(f"  Saved: results/experimental/peak_motif_conservation.csv ({len(fam_at_variant):,} rows)")
