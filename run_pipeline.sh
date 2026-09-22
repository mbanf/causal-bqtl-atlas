#!/bin/bash
# =============================================================================
# Reproduction pipeline for:
# "A high-confidence causal bQTL atlas reveals the regulatory architecture
#  of natural gene expression variation in maize"
#
# Banf & Hartwig, 2026
#
# Usage:
#   cd causal_bqtl_atlas
#   bash run_pipeline.sh [--skip-download] [--paper-only]
#
# Options:
#   --skip-download   Skip script 53b (NAM genotype extraction, ~4-6 hours)
#                     Use if data/processed/nam_founder_genotypes_at_bqtl.tsv exists.
#   --paper-only      Run only the core paper pipeline (skip exploratory scripts)
#
# Prerequisites:
#   - Python 3.10+ with venv (see requirements.txt)
#   - Data files in data/raw/ and data/processed/ (see DATA_MANIFEST.md)
#   - minimap2, samtools, bcftools (for script 53b only)
#   - ~3.5 GB disk space for raw data, ~500 MB for results
#
# Estimated runtime (Apple M-series, single core):
#   Phase 1 (foundational):     ~45 min
#   Phase 2 (genotype+causal):  ~30 min (+ 4-6h for genotype download)
#   Phase 3 (728 atlas):        ~20 min
#   Total (skip-download):      ~1.5 hours
#   Total (with download):      ~6 hours
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/scripts" && pwd)"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$PROJECT_DIR/results"
FIG_DIR="$PROJECT_DIR/figures/paper_figures"
LOG_DIR="$RESULTS_DIR/logs"

# Parse arguments
SKIP_DOWNLOAD=false
PAPER_ONLY=false
for arg in "$@"; do
    case $arg in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --paper-only) PAPER_ONLY=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# Find Python (prefer project venv, fall back to system)
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
elif [ -f "$PROJECT_DIR/../bqtl_predict/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/../bqtl_predict/.venv/bin/activate"
fi

mkdir -p "$RESULTS_DIR" "$RESULTS_DIR/figures" "$FIG_DIR" "$LOG_DIR" "$PROJECT_DIR/interactive"

echo "=================================================================="
echo " CAUSAL bQTL ATLAS — FULL REPRODUCTION PIPELINE"
echo "=================================================================="
echo " Project:  $PROJECT_DIR"
echo " Scripts:  $SCRIPT_DIR"
echo " Results:  $RESULTS_DIR"
echo " Figures:  $FIG_DIR"
echo " Python:   $(python3 --version)"
echo " Date:     $(date)"
echo " Options:  skip-download=$SKIP_DOWNLOAD  paper-only=$PAPER_ONLY"
echo "=================================================================="

# ── Verify required data ──
echo ""
echo "Checking required input data..."
MISSING=0
for f in \
    data/processed/bqtl_snp_ww.csv \
    data/processed/engelhorn_ase_ww.csv \
    data/processed/engelhorn_ase_ww_gene_summary.csv \
    data/processed/engelhorn_ww_vs_ds_expression.tsv \
    data/processed/maize_tf_pwm_database.json \
    data/raw/B73_NAM5.fa \
    data/raw/B73_NAM5.fa.fai \
    data/raw/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3 \
    data/raw/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1_entap_results.tsv.gz \
    data/raw/supp_table_MOESM5.xlsx; do
    if [ ! -e "$PROJECT_DIR/$f" ]; then
        echo "  MISSING: $f"
        MISSING=$((MISSING + 1))
    fi
done
# Check engelhorn_peaks directory
if [ ! -d "$PROJECT_DIR/data/raw/engelhorn_peaks" ] || [ -z "$(ls "$PROJECT_DIR/data/raw/engelhorn_peaks/" 2>/dev/null)" ]; then
    echo "  MISSING: data/raw/engelhorn_peaks/ (narrowPeak files)"
    MISSING=$((MISSING + 1))
fi
if [ $MISSING -gt 0 ]; then
    echo ""
    echo "  ERROR: $MISSING required data file(s) missing."
    echo "  See DATA_MANIFEST.md for download instructions."
    exit 1
fi
echo "  All required data files present."

# ── Helper function ──
run_script() {
    local script="$1"
    local name=$(basename "$script" .py)
    local log="$LOG_DIR/${name}.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ${name}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local start=$(date +%s)
    python3 "$script" 2>&1 | tee "$log"
    local status=$?
    local end=$(date +%s)
    local duration=$((end - start))

    if [ $status -eq 0 ]; then
        echo "  ✓ ${name} completed in ${duration}s"
    else
        echo "  ✗ ${name} FAILED (exit code $status) after ${duration}s"
        echo "  Check log: $log"
        exit 1
    fi
}

TOTAL_START=$(date +%s)

# ══════════════════════════════════════════════════════════════════════
# PHASE 1: Foundational analysis (independent scripts)
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 1: Foundational analysis                                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# These produce core intermediate files used by later scripts
run_script "$SCRIPT_DIR/44_full_quantitative_modulation.py"    # → quantitative_modulation_full.csv
run_script "$SCRIPT_DIR/46_bidirectional_motif_scan.py"        # → bidirectional_motif_scan.csv
run_script "$SCRIPT_DIR/47_moa_bound_region_motifs.py"         # → bound_region_motifs.csv (CRITICAL)
run_script "$SCRIPT_DIR/48_moa_peak_disruption.py"             # → pan_cistrome_peaks.csv (CRITICAL)

if [ "$PAPER_ONLY" = false ]; then
    run_script "$SCRIPT_DIR/49_hybrid_binding_predictor.py"    # → hybrid_binding_features.csv
    run_script "$SCRIPT_DIR/50_functional_bqtl.py"             # → bqtl_functional_test.csv
    run_script "$SCRIPT_DIR/52_ase_direction_proxy.py"         # → bqtl_ase_direction_concordance.csv
fi

run_script "$SCRIPT_DIR/45_case_studies.py"                    # → case_studies.csv

# ══════════════════════════════════════════════════════════════════════
# PHASE 2: Genotype-centric causal analysis
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 2: Genotype-centric causal analysis                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# Step 53b: Extract NAM founder genotypes
GENOTYPE_FILE="$PROJECT_DIR/data/processed/nam_founder_genotypes_at_bqtl.tsv"
if [ "$SKIP_DOWNLOAD" = true ]; then
    if [ -f "$GENOTYPE_FILE" ]; then
        echo ""
        echo "  Skipping genotype extraction (--skip-download)."
        echo "  Using: $GENOTYPE_FILE ($(du -h "$GENOTYPE_FILE" | cut -f1))"
    else
        echo "  ERROR: --skip-download but genotype file not found."
        echo "  Run without --skip-download first."
        exit 1
    fi
else
    run_script "$SCRIPT_DIR/53b_comprehensive_genotypes.py"
fi

# Core causal test
run_script "$SCRIPT_DIR/54_genotype_aware_bqtl_test.py"        # → genotype_bqtl_results.csv

# Select 728 causal bQTL (THE pivotal step)
run_script "$SCRIPT_DIR/63_select_causal_728.py"               # → causal_bqtl_728.csv, regulatory_map_728.csv

# Downstream analyses using genotype data
run_script "$SCRIPT_DIR/55_motif_genotype_ase_chain.py"        # → motif_genotype_ase_chain.csv
run_script "$SCRIPT_DIR/56_functional_bqtl_grammar.py"         # → functional_bqtl_grammar.csv
run_script "$SCRIPT_DIR/57_drought_tf_bqtl_configuration.py"   # → drought_tf_bqtl_config.csv
run_script "$SCRIPT_DIR/58_drought_genotype_ase_test.py"        # → ww_vs_drought_genotype_bqtl.csv
run_script "$SCRIPT_DIR/59_regulatory_configuration_figure.py"  # → tf_expression_vs_bqtl_effect.csv
run_script "$SCRIPT_DIR/60_condition_switching_deepdive.py"      # → condition_switching_deepdive.csv
run_script "$SCRIPT_DIR/61_tf_saturation_model.py"              # → tf_saturation_results.csv
run_script "$SCRIPT_DIR/62_binding_ase_concordance.py"          # → binding_ase_concordance.csv

# ══════════════════════════════════════════════════════════════════════
# PHASE 3: 728 causal bQTL characterization
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 3: 728 causal bQTL characterization                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

run_script "$SCRIPT_DIR/65_phenotype_ratelimit.py"              # phenotype/pathway analysis
run_script "$SCRIPT_DIR/66_gene_atlas_728.py"                   # → gene_atlas_728.csv
run_script "$SCRIPT_DIR/68_verify_sole_motif.py"                # → sole_motif_verification_728.csv
run_script "$SCRIPT_DIR/69_motif_creation_vs_disruption.py"     # → motif_creation_vs_disruption_728.csv
run_script "$SCRIPT_DIR/70_rewiring_deep_analysis.py"           # → rewiring_deep_analysis_728.csv
run_script "$SCRIPT_DIR/71_switch_mechanism.py"                 # switch directionality
run_script "$SCRIPT_DIR/72_threshold_sensitivity.py"            # → threshold_sensitivity_sweep.csv

# ══════════════════════════════════════════════════════════════════════
# PHASE 4: Interactive atlas
# ══════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 4: Interactive atlas                                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

run_script "$SCRIPT_DIR/67_build_interactive.py"                # → interactive/gene_atlas_728_interactive.html
run_script "$SCRIPT_DIR/update_atlas_data.py"                   # update HTML with rewiring data

# ══════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════
TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( (TOTAL_END - TOTAL_START) / 60 ))

echo ""
echo "=================================================================="
echo " PIPELINE COMPLETE (${TOTAL_DURATION} minutes)"
echo "=================================================================="
echo ""
echo " Core outputs:"
echo "   results/causal_bqtl_728.csv         (728 causal variants)"
echo "   results/regulatory_map_728.csv      (annotated atlas)"
echo "   results/gene_atlas_728.csv          (full gene characterization)"
echo "   interactive/gene_atlas_728_interactive.html"
echo ""
echo " Paper figures:"
ls "$FIG_DIR"/*.pdf 2>/dev/null | wc -l | xargs echo "   PDF figures:"
echo ""
echo " Logs: $LOG_DIR/"
echo "=================================================================="
