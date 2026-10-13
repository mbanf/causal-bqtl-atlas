#!/usr/bin/env python3
"""
exp_04_core_vs_facultative.py — Pan-cistrome core vs facultative gene analysis

Julia's suggestion: Classify genes by how many hybrids they're bound in,
analogous to pangenome core/dispensable/private classification.

- Core: gene promoter bound in all/most hybrids (>=20 of 25)
- Shared: bound in many (10-19)
- Dispensable: bound in few (2-9)
- Private: bound in only 1 hybrid

Then ask: are causal bQTL enriched in core or facultative genes?

Input:
  results/pan_cistrome_peaks.csv (291K peaks with n_hybrids)
  results/causal_bqtl_728.csv
  data/raw/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3

Output:
  results/experimental/core_vs_facultative.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from bisect import bisect_left, bisect_right
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
print("  EXPERIMENT 4: Pan-cistrome core vs facultative gene classification")
print("=" * 70)

# ── Load data ──
peaks = pd.read_csv(RESULTS / "pan_cistrome_peaks.csv")
causal = pd.read_csv(RESULTS / "causal_bqtl_728.csv")
print(f"\n  Peaks: {len(peaks):,}")
print(f"  Causal bQTL: {len(causal):,}")

# ── Load gene coordinates from GFF3 ──
gff_path = DATA / "raw" / "Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3"
print(f"  Loading gene coordinates from GFF3...")

genes = []
with open(gff_path) as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split("\t")
        if len(parts) < 9 or parts[2] != "gene":
            continue
        chrom = parts[0]
        start = int(parts[3])
        end = int(parts[4])
        strand = parts[6]
        attrs = parts[8]
        gene_id = None
        for attr in attrs.split(";"):
            if attr.startswith("ID=gene:"):
                gene_id = attr.replace("ID=gene:", "")
            elif attr.startswith("ID="):
                gene_id = attr.replace("ID=", "")
        if gene_id:
            tss = start if strand == "+" else end
            genes.append({
                "gene_id": gene_id,
                "chr": chrom,
                "start": start,
                "end": end,
                "strand": strand,
                "tss": tss,
            })

gene_df = pd.DataFrame(genes)
print(f"  Genes in GFF3: {len(gene_df):,}")

# ── For each gene, find peaks in its promoter (2kb upstream of TSS) ──
PROMOTER_SIZE = 2000

# Build sorted peak arrays per chromosome
peak_by_chr = {}
for _, p in peaks.iterrows():
    peak_by_chr.setdefault(p["chr"], []).append((p["start"], p["end"], p["n_hybrids"], p["width"]))

for k in peak_by_chr:
    peak_by_chr[k].sort()

def get_promoter_peaks(chrom, tss, strand):
    """Find all peaks overlapping the promoter region."""
    if strand == "+":
        prom_start = max(0, tss - PROMOTER_SIZE)
        prom_end = tss
    else:
        prom_start = tss
        prom_end = tss + PROMOTER_SIZE

    hits = []
    for (ps, pe, nh, pw) in peak_by_chr.get(chrom, []):
        if ps > prom_end:
            break
        if pe >= prom_start and ps <= prom_end:
            hits.append(nh)
    return hits

print(f"\n  Mapping peaks to gene promoters (2kb upstream)...")

results = []
for _, gene in gene_df.iterrows():
    peak_hybrids = get_promoter_peaks(gene["chr"], gene["tss"], gene["strand"])
    if len(peak_hybrids) == 0:
        max_hybrids = 0
        mean_hybrids = 0
        n_peaks = 0
    else:
        max_hybrids = max(peak_hybrids)
        mean_hybrids = np.mean(peak_hybrids)
        n_peaks = len(peak_hybrids)

    results.append({
        "gene_id": gene["gene_id"],
        "chr": gene["chr"],
        "tss": gene["tss"],
        "n_promoter_peaks": n_peaks,
        "max_hybrid_breadth": max_hybrids,
        "mean_hybrid_breadth": mean_hybrids,
    })

res_df = pd.DataFrame(results)

# ── Classify genes ──
def classify(n_hybrids):
    if n_hybrids == 0:
        return "unbound"
    elif n_hybrids == 1:
        return "private"
    elif n_hybrids <= 9:
        return "dispensable"
    elif n_hybrids <= 19:
        return "shared"
    else:
        return "core"

res_df["category"] = res_df["max_hybrid_breadth"].apply(classify)

# Category order
cat_order = ["unbound", "private", "dispensable", "shared", "core"]

print(f"\n{'=' * 70}")
print(f"  GENE CLASSIFICATION (by max hybrid breadth in promoter)")
print(f"{'=' * 70}")
for cat in cat_order:
    n = (res_df["category"] == cat).sum()
    pct = 100 * n / len(res_df)
    print(f"  {cat:>12}: {n:>6,} genes ({pct:.1f}%)")

# ── Cross with causal bQTL ──
causal_genes = set(causal["gene_id"])
res_df["is_causal"] = res_df["gene_id"].isin(causal_genes)

print(f"\n{'=' * 70}")
print(f"  CAUSAL bQTL ENRICHMENT BY CATEGORY")
print(f"{'=' * 70}")

# Overall rate
overall_rate = res_df["is_causal"].mean()
print(f"\n  Overall causal rate: {res_df['is_causal'].sum()}/{len(res_df)} ({100*overall_rate:.2f}%)")

print(f"\n  {'Category':>12}  {'All genes':>10}  {'Causal':>8}  {'Rate':>8}  {'Enrichment':>12}  {'p-value':>10}")
print(f"  {'-'*72}")

enrichments = []
for cat in cat_order:
    mask = res_df["category"] == cat
    n_total = mask.sum()
    n_causal = (mask & res_df["is_causal"]).sum()
    rate = n_causal / n_total if n_total > 0 else 0
    enrichment = rate / overall_rate if overall_rate > 0 else 0

    # Fisher's exact test
    a = n_causal
    b = n_total - n_causal
    c = res_df["is_causal"].sum() - n_causal
    d = len(res_df) - n_total - c
    odds, pval = stats.fisher_exact([[a, b], [c, d]])

    print(f"  {cat:>12}  {n_total:>10,}  {n_causal:>8,}  {100*rate:>7.2f}%  {enrichment:>11.2f}x  {pval:>10.2e}")
    enrichments.append({"category": cat, "n_genes": n_total, "n_causal": n_causal,
                        "rate": rate, "enrichment": enrichment, "pval": pval})

# ── Hybrid breadth distribution for causal vs non-causal ──
causal_breadth = res_df.loc[res_df["is_causal"], "max_hybrid_breadth"]
noncausal_breadth = res_df.loc[~res_df["is_causal"] & (res_df["max_hybrid_breadth"] > 0), "max_hybrid_breadth"]

print(f"\n  Hybrid breadth (max in promoter):")
print(f"    Causal 728:   median={causal_breadth.median():.0f}, mean={causal_breadth.mean():.1f}")
print(f"    Non-causal:   median={noncausal_breadth.median():.0f}, mean={noncausal_breadth.mean():.1f}")
mw, p = stats.mannwhitneyu(causal_breadth, noncausal_breadth, alternative="two-sided")
print(f"    Mann-Whitney p: {p:.2e}")

# ── Effect size by category ──
causal_with_cat = causal.merge(res_df[["gene_id", "category", "max_hybrid_breadth"]], on="gene_id", how="left")
print(f"\n  Effect size by category (causal bQTL only):")
for cat in ["private", "dispensable", "shared", "core"]:
    sub = causal_with_cat[causal_with_cat["category"] == cat]
    if len(sub) > 0:
        print(f"    {cat:>12}: n={len(sub):>4}, median |d|={sub['abs_d'].median():.2f}, "
              f"median hybrids={sub['max_hybrid_breadth'].median():.0f}")

# ── Correlation: hybrid breadth vs effect size ──
valid = causal_with_cat.dropna(subset=["max_hybrid_breadth", "abs_d"])
if len(valid) > 10:
    rho, rho_p = stats.spearmanr(valid["max_hybrid_breadth"], valid["abs_d"])
    print(f"\n  Correlation: hybrid breadth vs |d|")
    print(f"    Spearman rho={rho:.3f}, p={rho_p:.2e}")

# ── Variant sharing vs binding breadth ──
if "n_variant" in causal_with_cat.columns:
    rho2, rho2_p = stats.spearmanr(
        causal_with_cat["max_hybrid_breadth"].dropna(),
        causal_with_cat["n_variant"].dropna()
    )
    print(f"\n  Correlation: binding breadth vs variant sharing (n_variant)")
    print(f"    Spearman rho={rho2:.3f}, p={rho2_p:.2e}")

print(f"\n{'=' * 70}")

# ── Figure ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# A: Distribution of hybrid breadth
ax = axes[0]
ax.hist(res_df.loc[res_df["max_hybrid_breadth"] > 0, "max_hybrid_breadth"],
        bins=25, color="#3498db", alpha=0.7, edgecolor="white")
ax.set_xlabel("Max hybrid breadth in promoter")
ax.set_ylabel("Number of genes")
ax.set_title("A. Pan-cistrome binding breadth")
ax.axvline(20, color="red", linestyle="--", alpha=0.5, label="Core threshold")
ax.legend(fontsize=8)

# B: Causal rate by category
ax = axes[1]
enr_df = pd.DataFrame(enrichments)
enr_df = enr_df[enr_df["category"] != "unbound"]
colors = {"private": "#e74c3c", "dispensable": "#f39c12", "shared": "#2ecc71", "core": "#3498db"}
bars = ax.bar(enr_df["category"], enr_df["enrichment"],
              color=[colors.get(c, "#95a5a6") for c in enr_df["category"]])
ax.axhline(1, color="black", linestyle="--", alpha=0.5, label="Expected")
ax.set_ylabel("Enrichment (obs/expected)")
ax.set_title("B. Causal bQTL enrichment")
ax.legend(fontsize=8)
# Add p-values
for i, row in enr_df.iterrows():
    if row["pval"] < 0.05:
        ax.text(i, row["enrichment"] + 0.05, f"p={row['pval']:.1e}", ha="center", fontsize=7)

# C: Hybrid breadth causal vs non-causal
ax = axes[2]
ax.hist(noncausal_breadth, bins=25, alpha=0.5, color="#95a5a6",
        label="Non-causal", density=True)
ax.hist(causal_breadth, bins=25, alpha=0.7, color="#e74c3c",
        label="Causal 728", density=True)
ax.set_xlabel("Max hybrid breadth")
ax.set_ylabel("Density")
ax.set_title(f"C. Binding breadth (p={p:.1e})")
ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT / "core_vs_facultative.pdf", bbox_inches="tight")
fig.savefig(OUT / "core_vs_facultative.png", dpi=150, bbox_inches="tight")
print(f"\n  Saved: results/experimental/core_vs_facultative.pdf")

# ── Save ──
res_df.to_csv(OUT / "core_vs_facultative.csv", index=False)
causal_with_cat.to_csv(OUT / "causal_728_with_binding_breadth.csv", index=False)
print(f"  Saved: results/experimental/core_vs_facultative.csv ({len(res_df):,} genes)")
print(f"  Saved: results/experimental/causal_728_with_binding_breadth.csv")
