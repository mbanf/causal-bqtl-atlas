#!/bin/bash
# =============================================================================
# Reproduction pipeline for:
# "Condition-dependent binding QTL switching reveals the transcription factor
#  concentration landscape as a gatekeeper of functional genetic variation
#  in maize"
#
# Banf & Hartwig, 2026
#
# Usage:
#   cd bqtl_predict
#   bash paper_regulatory_config/run_pipeline.sh [--skip-download]
#
# Options:
#   --skip-download   Skip script 53 (NAM genotype download, ~4-6 hours)
#                     Use this if data/processed/nam_founder_genotypes_at_bqtl.tsv
#                     already exists.
#
# Prerequisites:
#   - Python venv at bqtl_predict/.venv with all dependencies
#   - bcftools installed (for script 53 only)
#   - Input data files (see README.md)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ANALYSIS_DIR="$PROJECT_DIR/results/celltype_landscape/quantitative_modulation_paper"

# Parse arguments
SKIP_DOWNLOAD=false
for arg in "$@"; do
    case $arg in
        --skip-download) SKIP_DOWNLOAD=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# Activate venv
cd "$PROJECT_DIR"
source .venv/bin/activate

echo "=================================================================="
echo " REGULATORY CONFIGURATION PAPER — FULL PIPELINE"
echo "=================================================================="
echo " Project: $PROJECT_DIR"
echo " Scripts: $ANALYSIS_DIR"
echo " Python:  $(python --version)"
echo " Date:    $(date)"
echo "=================================================================="

LOGDIR="$ANALYSIS_DIR/logs"
mkdir -p "$LOGDIR"

run_script() {
    local num=$1
    local name=$2
    local script="$ANALYSIS_DIR/${num}_${name}.py"
    local log="$LOGDIR/${num}_${name}.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP ${num}: ${name}"
    echo "  Script: ${script}"
    echo "  Log:    ${log}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local start=$(date +%s)
    python "$script" 2>&1 | tee "$log"
    local status=$?
    local end=$(date +%s)
    local duration=$((end - start))

    if [ $status -eq 0 ]; then
        echo "  ✓ Completed in ${duration}s"
    else
        echo "  ✗ FAILED (exit code $status) after ${duration}s"
        echo "  Check log: $log"
        exit 1
    fi
}

# ── Step 53: Download NAM founder genotypes ──
GENOTYPE_FILE="$PROJECT_DIR/data/processed/nam_founder_genotypes_at_bqtl.tsv"
if [ "$SKIP_DOWNLOAD" = true ]; then
    if [ -f "$GENOTYPE_FILE" ]; then
        echo ""
        echo "  Skipping script 53 (--skip-download). Using existing genotype file."
        echo "  File: $GENOTYPE_FILE"
        echo "  Size: $(du -h "$GENOTYPE_FILE" | cut -f1)"
    else
        echo ""
        echo "  ERROR: --skip-download specified but genotype file not found:"
        echo "  $GENOTYPE_FILE"
        echo "  Run without --skip-download to download genotypes first."
        exit 1
    fi
else
    run_script "53" "download_nam_genotypes"
fi

# ── Step 54: Genotype→ASE causal test (WW) ──
run_script "54" "genotype_aware_bqtl_test"

# ── Step 55: TF motif → genotype → ASE chain ──
run_script "55" "motif_genotype_ase_chain"

# ── Step 56: Regulatory grammar of functional bQTL ──
run_script "56" "functional_bqtl_grammar"

# ── Step 57: Drought TF regulatory configuration ──
run_script "57" "drought_tf_bqtl_configuration"

# ── Step 58: Drought genotype→ASE test ──
run_script "58" "drought_genotype_ase_test"

# ── Step 59: Regulatory configuration figure ──
run_script "59" "regulatory_configuration_figure"

# ── Step 60: Condition switching deep dive ──
run_script "60" "condition_switching_deepdive"

echo ""
echo "=================================================================="
echo " PIPELINE COMPLETE"
echo "=================================================================="
echo ""
echo " Output files:"
echo "   CSV:     $ANALYSIS_DIR/*.csv"
echo "   Figures: $ANALYSIS_DIR/figures/fig1[1-7]*.pdf"
echo "   Logs:    $LOGDIR/"
echo ""
echo " Key results files:"
ls -lh "$ANALYSIS_DIR"/genotype_bqtl_results.csv \
       "$ANALYSIS_DIR"/ww_vs_drought_genotype_bqtl.csv \
       "$ANALYSIS_DIR"/condition_switching_deepdive.csv \
       "$ANALYSIS_DIR"/functional_bqtl_grammar.csv 2>/dev/null
echo ""
echo " Figures:"
ls "$ANALYSIS_DIR"/figures/fig1[1-7]*.pdf 2>/dev/null
echo ""
echo "=================================================================="
