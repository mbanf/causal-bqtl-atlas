#!/usr/bin/env python3
"""
exp_03_drought_allele_specificity.py — Which allele changes under drought?

Thomas's comment (L202 / drought section): Which allele changes expression
under drought — B73, NAM, or both? Differentiating could reveal whether
drought-responsive expression is allele-specific.

The ASE metric log2(B73/NAM) captures the *ratio* between alleles in each
F1 hybrid. If drought changes this ratio, it means one allele responded
more than the other. We can decompose:

- If ASE shifts toward B73 under drought → NAM allele was repressed more
  (or B73 induced more)
- If ASE shifts toward NAM → B73 allele was repressed more
- If ASE stays the same → both alleles respond equally (trans effect)

Combined with total expression (log2FC WW→DS), we can infer:
- Gene UP + ASE unchanged → both alleles induced equally (trans)
- Gene UP + ASE shifts → one allele drives the induction (cis × environment)
- Gene DOWN + ASE unchanged → both repressed equally (trans)
- Gene DOWN + ASE shifts → allele-specific repression (cis × environment)

Input:
  results/condition_switching_deepdive.csv (WW + DS ASE per bQTL)
  results/causal_bqtl_728.csv
  data/processed/engelhorn_ww_vs_ds_expression.tsv

Output:
  results/experimental/drought_allele_specificity.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
DATA = BASE / "data"
OUT = RESULTS / "experimental"
OUT.mkdir(exist_ok=True)

print("=" * 70)
print("  EXPERIMENT 3: Drought allele specificity analysis")
print("=" * 70)

# ── Load data ──
cs = pd.read_csv(RESULTS / "condition_switching_deepdive.csv")
causal = pd.read_csv(RESULTS / "causal_bqtl_728.csv")
expr = pd.read_csv(DATA / "processed" / "engelhorn_ww_vs_ds_expression.tsv", sep="\t")

print(f"\n  Condition switching data: {len(cs):,} bQTL-gene pairs")
print(f"  Causal bQTL: {len(causal):,}")
print(f"  Expression data: {len(expr):,} genes")

# ── Filter to 728 causal bQTL ──
causal_set = set(zip(causal["chr"], causal["pos"]))
cs_causal = cs[cs.apply(lambda r: (r["chr"], r["pos"]) in causal_set, axis=1)].copy()
print(f"  Causal bQTL with both WW+DS data: {len(cs_causal):,}")

# ── ASE change under drought ──
# mean_ase_variant/reference = mean log2(B73/NAM) across hybrids in that genotype group
# Positive ASE = B73-biased; Negative ASE = NAM-biased
# cohens_d = effect size of genotype on ASE

# Key metric: ASE shift = cohens_d_ds - cohens_d_ww
# This captures whether the genotype→ASE relationship changes under drought
cs_causal["d_shift"] = cs_causal["cohens_d_ds"] - cs_causal["cohens_d_ww"]
cs_causal["abs_d_shift"] = cs_causal["d_shift"].abs()

# Alternative: look at mean ASE change per genotype group
# variant hybrids: ASE_ds - ASE_ww
cs_causal["ase_change_variant"] = cs_causal["mean_ase_variant_ds"] - cs_causal["mean_ase_variant_ww"]
# reference hybrids: ASE_ds - ASE_ww
cs_causal["ase_change_reference"] = cs_causal["mean_ase_reference_ds"] - cs_causal["mean_ase_reference_ww"]

# Add total expression change
cs_causal = cs_causal.merge(
    expr[["gene_id", "log2fc_DS_vs_WW"]], on="gene_id", how="left"
)

print(f"\n{'=' * 70}")
print(f"  RESULTS")
print(f"{'=' * 70}")

# ── 1) Overall ASE shift statistics ──
print(f"\n  1) ASE shift under drought (Cohen's d change):")
print(f"     Median |d_shift|: {cs_causal['abs_d_shift'].median():.2f}")
print(f"     Mean |d_shift|:   {cs_causal['abs_d_shift'].mean():.2f}")
print(f"     d_shift > 0.5:    {(cs_causal['abs_d_shift'] > 0.5).sum()} ({100*(cs_causal['abs_d_shift'] > 0.5).mean():.1f}%)")
print(f"     d_shift > 1.0:    {(cs_causal['abs_d_shift'] > 1.0).sum()} ({100*(cs_causal['abs_d_shift'] > 1.0).mean():.1f}%)")

# ── 2) ASE change in variant vs reference hybrids ──
print(f"\n  2) ASE change (log2 B73/NAM) under drought by genotype group:")
print(f"     Variant hybrids:   median change = {cs_causal['ase_change_variant'].median():.4f}")
print(f"     Reference hybrids: median change = {cs_causal['ase_change_reference'].median():.4f}")

# Test if variant hybrids show different ASE shift than reference
mw_stat, mw_p = stats.mannwhitneyu(
    cs_causal["ase_change_variant"].dropna(),
    cs_causal["ase_change_reference"].dropna(),
    alternative="two-sided"
)
print(f"     Mann-Whitney p: {mw_p:.2e}")

# ── 3) Decompose into cis × environment interaction categories ──
# Threshold for meaningful change
ASE_THRESH = 0.3  # log2 units (~23% change in allelic ratio)
FC_THRESH = 1.0   # log2 fold-change for drought response

cs_causal["drought_cat"] = "stable"
cs_causal.loc[cs_causal["log2fc_DS_vs_WW"] > FC_THRESH, "drought_cat"] = "up"
cs_causal.loc[cs_causal["log2fc_DS_vs_WW"] < -FC_THRESH, "drought_cat"] = "down"

# ASE shift categories
cs_causal["ase_shift_cat"] = "unchanged"
# In variant hybrids: does ASE shift?
cs_causal.loc[cs_causal["ase_change_variant"].abs() > ASE_THRESH, "ase_shift_cat"] = "shifted"

# Determine which allele drives the shift
# Positive ASE change = B73 becomes relatively more expressed (or NAM less)
# Negative ASE change = NAM becomes relatively more expressed (or B73 less)
cs_causal["allele_driver"] = "equal"
mask_variant_shift = cs_causal["ase_change_variant"].abs() > ASE_THRESH
cs_causal.loc[mask_variant_shift & (cs_causal["ase_change_variant"] > 0), "allele_driver"] = "B73_up_or_NAM_down"
cs_causal.loc[mask_variant_shift & (cs_causal["ase_change_variant"] < 0), "allele_driver"] = "NAM_up_or_B73_down"

print(f"\n  3) Cis × environment interaction (variant hybrids, threshold={ASE_THRESH}):")
print(f"     ASE unchanged under drought: {(cs_causal['ase_shift_cat']=='unchanged').sum()} ({100*(cs_causal['ase_shift_cat']=='unchanged').mean():.1f}%)")
print(f"     ASE shifted under drought:   {(cs_causal['ase_shift_cat']=='shifted').sum()} ({100*(cs_causal['ase_shift_cat']=='shifted').mean():.1f}%)")
print(f"       → B73 up / NAM down: {(cs_causal['allele_driver']=='B73_up_or_NAM_down').sum()}")
print(f"       → NAM up / B73 down: {(cs_causal['allele_driver']=='NAM_up_or_B73_down').sum()}")

# ── 4) Cross-tabulate drought response × ASE shift ──
print(f"\n  4) Cross-tabulation: total expression change × allelic shift")
print(f"     (variant hybrid ASE, threshold ASE={ASE_THRESH}, FC={FC_THRESH})")
ct = pd.crosstab(cs_causal["drought_cat"], cs_causal["allele_driver"], margins=True)
print(ct.to_string())

# ── 5) Interpretation per drought category ──
print(f"\n  5) Interpretation:")
for cat in ["up", "down", "stable"]:
    sub = cs_causal[cs_causal["drought_cat"] == cat]
    if len(sub) == 0:
        continue
    n_shifted = (sub["ase_shift_cat"] == "shifted").sum()
    pct = 100 * n_shifted / len(sub)
    b73_drive = (sub["allele_driver"] == "B73_up_or_NAM_down").sum()
    nam_drive = (sub["allele_driver"] == "NAM_up_or_B73_down").sum()
    print(f"     Drought {cat:>6}: {len(sub):>4} genes, {n_shifted:>3} shifted ({pct:.1f}%)")
    if n_shifted > 0:
        print(f"                     B73-driven: {b73_drive}, NAM-driven: {nam_drive}")

# ── 6) Also look at reference hybrids (B73 allele only) ──
print(f"\n  6) Reference hybrids (B73 homozygous) ASE change under drought:")
ref_change = cs_causal["ase_change_reference"].dropna()
print(f"     Median change: {ref_change.median():.4f}")
print(f"     |change| > 0.3: {(ref_change.abs() > 0.3).sum()} ({100*(ref_change.abs() > 0.3).mean():.1f}%)")
print(f"     → B73 responds to drought in ref hybrids too (trans effect)")

# ── 7) Key insight: does the bQTL genotype predict drought response? ──
# Compare ASE change in variant vs reference hybrids for drought-responsive genes
drought_genes = cs_causal[cs_causal["drought_cat"].isin(["up", "down"])]
if len(drought_genes) > 10:
    corr, corr_p = stats.spearmanr(
        drought_genes["ase_change_variant"].dropna(),
        drought_genes["log2fc_DS_vs_WW"].dropna()
    )
    print(f"\n  7) Correlation: ASE change in variant hybrids vs total drought FC")
    print(f"     Drought-responsive genes only (n={len(drought_genes)}):")
    print(f"     Spearman rho={corr:.3f}, p={corr_p:.2e}")

    corr2, corr2_p = stats.spearmanr(
        drought_genes["ase_change_reference"].dropna(),
        drought_genes["log2fc_DS_vs_WW"].dropna()
    )
    print(f"     Reference hybrids: rho={corr2:.3f}, p={corr2_p:.2e}")

# ── 8) Specific examples ──
print(f"\n  8) Notable examples:")
# Find genes where variant and reference hybrids respond differently
cs_causal["differential_response"] = (cs_causal["ase_change_variant"] - cs_causal["ase_change_reference"]).abs()

top_diff = cs_causal.nlargest(10, "differential_response")
for _, row in top_diff.iterrows():
    direction = "UP" if row["log2fc_DS_vs_WW"] > 1 else ("DOWN" if row["log2fc_DS_vs_WW"] < -1 else "stable")
    print(f"     {row['gene_id']}: drought {direction} (FC={row['log2fc_DS_vs_WW']:.2f})")
    print(f"       Variant hybrids ASE shift: {row['ase_change_variant']:+.3f}")
    print(f"       Reference hybrids ASE shift: {row['ase_change_reference']:+.3f}")
    print(f"       → {'NAM' if row['ase_change_variant'] < row['ase_change_reference'] else 'B73'} allele responds more in variant background")

print(f"\n{'=' * 70}")

# ── Figure ──
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# A: ASE change distribution by genotype group
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(cs_causal["ase_change_variant"].dropna(), bins=50, alpha=0.6, color="#e74c3c", label="Variant hybrids")
ax1.hist(cs_causal["ase_change_reference"].dropna(), bins=50, alpha=0.6, color="#3498db", label="Reference hybrids")
ax1.axvline(0, color="black", linestyle="--", alpha=0.5)
ax1.set_xlabel("ASE change (DS − WW)")
ax1.set_ylabel("Count")
ax1.set_title("A. Allelic shift under drought")
ax1.legend(fontsize=8)

# B: Scatter - total FC vs ASE change in variant hybrids
ax2 = fig.add_subplot(gs[0, 1])
mask = cs_causal["log2fc_DS_vs_WW"].notna()
colors = cs_causal.loc[mask, "drought_cat"].map({"up": "#e74c3c", "down": "#3498db", "stable": "#95a5a6"})
ax2.scatter(cs_causal.loc[mask, "log2fc_DS_vs_WW"],
           cs_causal.loc[mask, "ase_change_variant"],
           c=colors, alpha=0.4, s=15)
ax2.axhline(0, color="black", linestyle="--", alpha=0.3)
ax2.axvline(0, color="black", linestyle="--", alpha=0.3)
ax2.set_xlabel("Total expression log2FC (DS/WW)")
ax2.set_ylabel("ASE shift in variant hybrids")
ax2.set_title("B. Expression change vs allelic shift")

# C: ASE shift comparison between variant and reference
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(cs_causal["ase_change_reference"], cs_causal["ase_change_variant"],
           alpha=0.3, s=10, c="#2c3e50")
lims = [-2, 2]
ax3.plot(lims, lims, "--", color="red", alpha=0.5, label="Equal response")
ax3.set_xlim(lims)
ax3.set_ylim(lims)
ax3.set_xlabel("ASE change (reference hybrids)")
ax3.set_ylabel("ASE change (variant hybrids)")
ax3.set_title("C. Variant vs reference response")
ax3.legend(fontsize=8)

# D: Cohen's d under WW vs DS
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(cs_causal["cohens_d_ww"], cs_causal["cohens_d_ds"],
           alpha=0.3, s=10, c="#2c3e50")
lims_d = [-10, 10]
ax4.plot(lims_d, lims_d, "--", color="red", alpha=0.5)
ax4.set_xlabel("Cohen's d (WW)")
ax4.set_ylabel("Cohen's d (DS)")
ax4.set_title("D. Genotype effect: WW vs drought")

# E: Bar chart of allele driver categories by drought response
ax5 = fig.add_subplot(gs[1, 1])
ct_plot = pd.crosstab(cs_causal["drought_cat"], cs_causal["allele_driver"])
ct_plot = ct_plot.reindex(["down", "stable", "up"])
ct_plot = ct_plot.reindex(columns=["B73_up_or_NAM_down", "equal", "NAM_up_or_B73_down"])
ct_plot.plot(kind="bar", ax=ax5, color=["#3498db", "#95a5a6", "#e74c3c"], stacked=True)
ax5.set_xlabel("Drought response")
ax5.set_ylabel("Number of bQTL")
ax5.set_title("E. Which allele responds?")
ax5.legend(["B73↑/NAM↓", "Equal", "NAM↑/B73↓"], fontsize=7)
ax5.tick_params(axis='x', rotation=0)

# F: Differential response (variant - reference ASE change)
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(cs_causal["differential_response"].dropna(), bins=50, color="#8e44ad", alpha=0.7)
ax6.axvline(cs_causal["differential_response"].median(), color="red", linestyle="--",
           label=f"Median={cs_causal['differential_response'].median():.2f}")
ax6.set_xlabel("|Variant − Reference| ASE change")
ax6.set_ylabel("Count")
ax6.set_title("F. Genotype-dependent drought response")
ax6.legend(fontsize=8)

plt.suptitle("Drought allele specificity analysis — 728 causal bQTL", fontsize=13, y=1.01)
fig.savefig(OUT / "drought_allele_specificity.pdf", bbox_inches="tight")
fig.savefig(OUT / "drought_allele_specificity.png", dpi=150, bbox_inches="tight")
print(f"  Saved: results/experimental/drought_allele_specificity.pdf")

# ── Save ──
cs_causal.to_csv(OUT / "drought_allele_specificity.csv", index=False)
print(f"  Saved: results/experimental/drought_allele_specificity.csv ({len(cs_causal)} rows)")
