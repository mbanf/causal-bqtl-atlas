#!/usr/bin/env python3
"""
verify_pipeline.py — Verify that pipeline outputs match expected results.

Run after `bash run_pipeline.sh --skip-download` to confirm reproducibility.
Checks row counts, column presence, key statistics, and gene set hashes.

Usage:
    python scripts/verify_pipeline.py
    python scripts/verify_pipeline.py --verbose
"""

import sys
import hashlib
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas and numpy required. Install with: pip install pandas numpy")
    sys.exit(1)

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"

VERBOSE = "--verbose" in sys.argv

# ── Expected results (ground truth from original run) ──
EXPECTED = {
    "results/genotype_bqtl_results.csv": {
        "rows": 19368,
        "cols": 19,
        "gene_hash": "9496f706",
        "description": "Genotype→ASE causal test (script 54)",
    },
    "results/causal_bqtl_728.csv": {
        "rows": 728,
        "cols": 11,
        "gene_hash": "2b0f6b6c",
        "description": "728 high-confidence causal bQTL (script 63)",
    },
    "results/regulatory_map_728.csv": {
        "rows": 728,
        "gene_hash": "2b0f6b6c",
        "description": "Annotated regulatory map (script 63)",
    },
    "results/bound_region_motifs.csv": {
        "rows": 388400,
        "description": "Motif content in MOA-seq peaks (script 47)",
    },
    "results/pan_cistrome_peaks.csv": {
        "rows": 291878,
        "cols": 6,
        "description": "Merged pan-cistrome peaks (script 48)",
    },
    "results/bidirectional_motif_scan.csv": {
        "rows": 120000,
        "description": "Bidirectional motif disruption/creation (script 46)",
    },
    "results/gene_peak_disruption.csv": {
        "rows": 32082,
        "description": "Peak-level disruption analysis (script 48)",
    },
    "results/sole_motif_verification_728.csv": {
        "rows": 728,
        "gene_hash": "2b0f6b6c",
        "description": "Sole-copy motif verification (script 68)",
    },
    "results/motif_creation_vs_disruption_728.csv": {
        "rows": 728,
        "gene_hash": "2b0f6b6c",
        "description": "Bidirectional motif analysis for 728 (script 69)",
    },
    "results/rewiring_deep_analysis_728.csv": {
        "rows": 716,
        "gene_hash": "13ac58a1",
        "description": "TF motif rewiring analysis (script 70)",
    },
    "results/gene_atlas_728.csv": {
        "rows": 728,
        "gene_hash": "2b0f6b6c",
        "description": "Full gene atlas characterization (script 66)",
    },
    "results/ww_vs_drought_genotype_bqtl.csv": {
        "rows": 40243,
        "description": "WW vs drought paired test (script 58)",
    },
    "results/condition_switching_deepdive.csv": {
        "rows": 19554,
        "description": "Condition switching deep dive (script 60)",
    },
}

# ── Key paper statistics to verify ──
PAPER_STATS = [
    {
        "name": "Significant bQTL-gene pairs (FDR<0.05)",
        "file": "results/genotype_bqtl_results.csv",
        "check": lambda df: len(df[df["fdr_mw"] < 0.05]),
        "expected": 1403,
        "tolerance": 0,
    },
    {
        "name": "Single-bQTL genes",
        "file": "results/genotype_bqtl_results.csv",
        "check": lambda df: len(df[df["fdr_mw"] < 0.05].groupby("gene_id").filter(lambda x: len(x) == 1)["gene_id"].unique()),
        "expected": 772,
        "tolerance": 0,
    },
    {
        "name": "Final causal bQTL count",
        "file": "results/causal_bqtl_728.csv",
        "check": lambda df: len(df),
        "expected": 728,
        "tolerance": 0,
    },
    {
        "name": "Median |Cohen's d|",
        "file": "results/causal_bqtl_728.csv",
        "check": lambda df: round(df["abs_d"].median(), 1),
        "expected": 2.4,
        "tolerance": 0.2,
    },
    {
        "name": "Rewired bQTL (motif switch)",
        "file": "results/rewiring_deep_analysis_728.csv",
        "check": lambda df: len(df[df.get("category", pd.Series()) == "rewired"]) if "category" in df.columns else -1,
        "expected": 233,
        "tolerance": 5,
    },
    {
        "name": "Sole-copy fraction",
        "file": "results/sole_motif_verification_728.csv",
        "check": lambda df: round(df["has_any_sole"].mean() * 100, 1) if "has_any_sole" in df.columns else -1,
        "expected": 39.4,
        "tolerance": 2.0,
    },
]


def gene_hash(df):
    """MD5 hash of sorted unique gene_ids for reproducibility check."""
    if "gene_id" not in df.columns:
        return "n/a"
    genes = "|".join(sorted(df["gene_id"].dropna().unique()))
    return hashlib.md5(genes.encode()).hexdigest()[:8]


def main():
    print("=" * 70)
    print("  CAUSAL bQTL ATLAS — PIPELINE VERIFICATION")
    print("=" * 70)

    passed = 0
    failed = 0
    missing = 0

    # ── Check result files ──
    print("\n  FILE CHECKS")
    print("  " + "-" * 66)

    for relpath, exp in EXPECTED.items():
        fpath = BASE / relpath
        if not fpath.exists():
            print(f"  MISSING  {relpath}")
            missing += 1
            continue

        df = pd.read_csv(fpath)
        errors = []

        # Row count
        if "rows" in exp and len(df) != exp["rows"]:
            errors.append(f"rows: {len(df)} (expected {exp['rows']})")

        # Column count
        if "cols" in exp and len(df.columns) != exp["cols"]:
            errors.append(f"cols: {len(df.columns)} (expected {exp['cols']})")

        # Gene hash
        if "gene_hash" in exp:
            gh = gene_hash(df)
            if gh != exp["gene_hash"]:
                errors.append(f"gene_hash: {gh} (expected {exp['gene_hash']})")

        if errors:
            print(f"  FAIL     {relpath}")
            for e in errors:
                print(f"           {e}")
            failed += 1
        else:
            status = f"{len(df):,} rows"
            if VERBOSE:
                status += f", {len(df.columns)} cols"
                gh = gene_hash(df)
                if gh != "n/a":
                    status += f", hash={gh}"
            print(f"  OK       {relpath} ({status})")
            passed += 1

    # ── Check paper statistics ──
    print(f"\n  PAPER STATISTICS")
    print("  " + "-" * 66)

    stats_pass = 0
    stats_fail = 0

    for stat in PAPER_STATS:
        fpath = BASE / stat["file"]
        if not fpath.exists():
            print(f"  SKIP     {stat['name']} (file missing)")
            continue

        df = pd.read_csv(fpath)
        actual = stat["check"](df)

        if actual == -1:
            print(f"  SKIP     {stat['name']} (column missing)")
            continue

        diff = abs(actual - stat["expected"])
        if diff <= stat["tolerance"]:
            print(f"  OK       {stat['name']}: {actual} (expected {stat['expected']})")
            stats_pass += 1
        else:
            print(f"  FAIL     {stat['name']}: {actual} (expected {stat['expected']}, diff={diff})")
            stats_fail += 1

    # ── Check interactive atlas ──
    print(f"\n  INTERACTIVE ATLAS")
    print("  " + "-" * 66)
    html_path = BASE / "interactive" / "gene_atlas_728_interactive.html"
    if html_path.exists():
        size_kb = html_path.stat().st_size / 1024
        print(f"  OK       gene_atlas_728_interactive.html ({size_kb:.0f} KB)")
        passed += 1
    else:
        print(f"  MISSING  interactive/gene_atlas_728_interactive.html")
        missing += 1

    # ── Summary ──
    total = passed + failed + missing
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {passed}/{total} files OK, "
          f"{stats_pass}/{stats_pass + stats_fail} stats OK")
    if failed > 0 or missing > 0:
        print(f"  {failed} FAILED, {missing} MISSING")
    if failed == 0 and missing == 0 and stats_fail == 0:
        print(f"  ALL CHECKS PASSED — pipeline reproduces original results")
    print(f"{'=' * 70}")

    sys.exit(1 if (failed > 0 or stats_fail > 0) else 0)


if __name__ == "__main__":
    main()
