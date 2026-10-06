#!/usr/bin/env nextflow

/*
 * Causal bQTL Atlas Pipeline
 * ==========================
 * Identifies 728 high-confidence causal binding QTL from maize pan-cistrome data.
 *
 * Banf & Hartwig (2026)
 *
 * Usage:
 *   nextflow run main.nf --data_dir /path/to/data
 *   nextflow run main.nf --data_dir /path/to/data --skip_genotype_download
 *   nextflow run main.nf --data_dir /path/to/data --paper_only
 *   nextflow run main.nf -resume   # resume from last checkpoint
 */

nextflow.enable.dsl=2

// ── Parameters ──
params.data_dir       = "${projectDir}/data"
params.results_dir    = "${projectDir}/results"
params.scripts_dir    = "${projectDir}/scripts"
params.skip_genotype_download = true  // Julia's genotypes are pre-converted
params.paper_only     = false         // skip exploratory scripts (49, 50, 52)

log.info """
╔══════════════════════════════════════════════════════════════╗
║           CAUSAL bQTL ATLAS PIPELINE                        ║
║           Banf & Hartwig (2026)                              ║
╠══════════════════════════════════════════════════════════════╣
║  Data directory : ${params.data_dir}
║  Results        : ${params.results_dir}
║  Scripts        : ${params.scripts_dir}
║  Skip download  : ${params.skip_genotype_download}
║  Paper only     : ${params.paper_only}
╚══════════════════════════════════════════════════════════════╝
"""

// ── Phase 1: Foundational Analysis ──

process QUANTITATIVE_MODULATION {
    tag "script_44"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path data_dir

    output:
    path "quantitative_modulation_full.csv", emit: quant_mod
    path "figures/*", emit: figures, optional: true

    script:
    """
    mkdir -p figures
    python3 ${params.scripts_dir}/44_quantitative_modulation.py
    """
}

process REDUNDANCY_OVERVIEW {
    tag "script_45"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path quant_mod

    output:
    path "per_family_redundancy.csv", emit: redundancy

    script:
    """
    python3 ${params.scripts_dir}/45_redundancy_overview.py
    """
}

process COMBINATORIAL_GRAMMAR {
    tag "script_46"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path redundancy

    output:
    path "functional_bqtl_grammar.csv", emit: grammar

    script:
    """
    python3 ${params.scripts_dir}/46_combinatorial_grammar.py
    """
}

process BOUND_REGION_MOTIFS {
    tag "script_47"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path data_dir

    output:
    path "bound_region_motifs.csv", emit: brm

    script:
    """
    python3 ${params.scripts_dir}/47_bound_region_motifs.py
    """
}

process PAN_CISTROME_PEAKS {
    tag "script_48"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path data_dir

    output:
    path "pan_cistrome_peaks.csv", emit: peaks

    script:
    """
    python3 ${params.scripts_dir}/48_pan_cistrome_peaks.py
    """
}

process BIDIRECTIONAL_MOTIF_SCAN {
    tag "script_51"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path data_dir

    output:
    path "bidirectional_motif_scan.csv", emit: bidir

    script:
    """
    python3 ${params.scripts_dir}/51_bidirectional_motif_scan.py
    """
}

process GENE_PEAK_DISRUPTION {
    tag "script_52"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path peaks
    path brm

    output:
    path "gene_peak_disruption.csv", emit: gpd

    script:
    """
    python3 ${params.scripts_dir}/52_gene_peak_disruption.py
    """
}

// ── Phase 2: Genotype-Causal Analysis ──

process CONVERT_GENOTYPES {
    tag "script_53c"
    publishDir "${params.data_dir}/processed", mode: 'copy'

    input:
    path data_dir

    output:
    path "nam_founder_genotypes_at_bqtl.tsv", emit: genotypes

    when:
    params.skip_genotype_download

    script:
    """
    python3 ${params.scripts_dir}/53c_convert_julia_genotypes.py
    """
}

process GENOTYPE_BQTL_TEST {
    tag "script_54"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path genotypes
    path data_dir

    output:
    path "genotype_bqtl_results.csv", emit: geno_results

    script:
    """
    python3 ${params.scripts_dir}/54_genotype_bqtl_test.py
    """
}

process MOTIF_GENOTYPE_ASE {
    tag "script_55"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path geno_results
    path brm

    output:
    path "motif_genotype_ase_chain.csv", emit: motif_geno

    script:
    """
    python3 ${params.scripts_dir}/55_motif_genotype_ase.py
    """
}

process WW_VS_DROUGHT {
    tag "script_58"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path genotypes
    path data_dir

    output:
    path "ww_vs_drought_genotype_bqtl.csv", emit: ww_ds

    script:
    """
    python3 ${params.scripts_dir}/58_ww_vs_drought_bqtl.py
    """
}

process CONDITION_SWITCHING {
    tag "script_60"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path ww_ds
    path grammar

    output:
    path "condition_switching_deepdive.csv", emit: switching

    script:
    """
    python3 ${params.scripts_dir}/60_condition_switching_deepdive.py
    """
}

process TF_SATURATION {
    tag "script_61"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path brm
    path genotypes

    output:
    path "tf_saturation_results.csv", emit: saturation

    script:
    """
    python3 ${params.scripts_dir}/61_tf_saturation_model.py
    """
}

process BINDING_ASE_CONCORDANCE {
    tag "script_62"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path ww_ds

    output:
    path "binding_ase_concordance.csv", emit: concordance

    script:
    """
    python3 ${params.scripts_dir}/62_binding_ase_concordance.py
    """
}

// ── Phase 3: 728 Atlas ──

process SELECT_CAUSAL_728 {
    tag "script_63"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path geno_results
    path peaks
    path genotypes

    output:
    path "causal_bqtl_728.csv", emit: causal
    path "regulatory_map_728.csv", emit: reg_map

    script:
    """
    python3 ${params.scripts_dir}/63_select_causal_728.py
    """
}

process PHENOTYPE_RATELIMIT {
    tag "script_65"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path causal
    path reg_map

    output:
    path "phenotype_associations_728.csv", emit: pheno
    path "enzyme_classification_728.csv", emit: enzyme

    script:
    """
    python3 ${params.scripts_dir}/65_phenotype_ratelimit.py
    """
}

process GENE_ATLAS {
    tag "script_66"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path causal
    path brm

    output:
    path "gene_atlas_728.csv", emit: atlas

    script:
    """
    python3 ${params.scripts_dir}/66_gene_atlas_728.py
    """
}

process VERIFY_SOLE_MOTIF {
    tag "script_68"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path causal
    path brm
    path peaks

    output:
    path "sole_motif_verification_728.csv", emit: sole

    script:
    """
    python3 ${params.scripts_dir}/68_verify_sole_motif.py
    """
}

process MOTIF_CREATION_DISRUPTION {
    tag "script_69"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path causal
    path data_dir

    output:
    path "motif_creation_vs_disruption_728.csv", emit: creation

    script:
    """
    python3 ${params.scripts_dir}/69_motif_creation_vs_disruption.py
    """
}

process REWIRING_ANALYSIS {
    tag "script_70"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path creation

    output:
    path "rewiring_deep_analysis_728.csv", emit: rewiring

    script:
    """
    python3 ${params.scripts_dir}/70_rewiring_deep_analysis.py
    """
}

process THRESHOLD_SENSITIVITY {
    tag "script_72"
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path causal
    path data_dir

    output:
    path "threshold_sensitivity_sweep.csv", emit: threshold

    script:
    """
    python3 ${params.scripts_dir}/72_threshold_sensitivity.py
    """
}

// ── Phase 4: Interactive Atlas ──

process BUILD_INTERACTIVE {
    tag "script_67"
    publishDir "${projectDir}/interactive", mode: 'copy'

    input:
    path atlas
    path creation
    path rewiring
    path sole

    output:
    path "gene_atlas_728_interactive.html", emit: html

    script:
    """
    python3 ${params.scripts_dir}/67_build_interactive.py
    python3 ${params.scripts_dir}/update_atlas_data.py
    """
}

// ── Verification ──

process VERIFY_PIPELINE {
    tag "verify"

    input:
    path html
    path results_dir

    output:
    stdout

    script:
    """
    python3 ${params.scripts_dir}/verify_pipeline.py
    """
}

// ── Main workflow ──

workflow {
    // Input channels
    data_dir = Channel.fromPath(params.data_dir, type: 'dir')

    // Phase 1: Foundational (independent processes can run in parallel)
    QUANTITATIVE_MODULATION(data_dir)
    BOUND_REGION_MOTIFS(data_dir)
    PAN_CISTROME_PEAKS(data_dir)
    BIDIRECTIONAL_MOTIF_SCAN(data_dir)

    REDUNDANCY_OVERVIEW(QUANTITATIVE_MODULATION.out.quant_mod)
    COMBINATORIAL_GRAMMAR(REDUNDANCY_OVERVIEW.out.redundancy)
    GENE_PEAK_DISRUPTION(PAN_CISTROME_PEAKS.out.peaks, BOUND_REGION_MOTIFS.out.brm)

    // Phase 2: Genotype-causal
    CONVERT_GENOTYPES(data_dir)
    GENOTYPE_BQTL_TEST(CONVERT_GENOTYPES.out.genotypes, data_dir)
    MOTIF_GENOTYPE_ASE(GENOTYPE_BQTL_TEST.out.geno_results, BOUND_REGION_MOTIFS.out.brm)
    WW_VS_DROUGHT(CONVERT_GENOTYPES.out.genotypes, data_dir)
    CONDITION_SWITCHING(WW_VS_DROUGHT.out.ww_ds, COMBINATORIAL_GRAMMAR.out.grammar)
    TF_SATURATION(BOUND_REGION_MOTIFS.out.brm, CONVERT_GENOTYPES.out.genotypes)
    BINDING_ASE_CONCORDANCE(WW_VS_DROUGHT.out.ww_ds)

    // Phase 3: 728 Atlas
    SELECT_CAUSAL_728(
        GENOTYPE_BQTL_TEST.out.geno_results,
        PAN_CISTROME_PEAKS.out.peaks,
        CONVERT_GENOTYPES.out.genotypes
    )
    PHENOTYPE_RATELIMIT(SELECT_CAUSAL_728.out.causal, SELECT_CAUSAL_728.out.reg_map)
    GENE_ATLAS(SELECT_CAUSAL_728.out.causal, BOUND_REGION_MOTIFS.out.brm)
    VERIFY_SOLE_MOTIF(SELECT_CAUSAL_728.out.causal, BOUND_REGION_MOTIFS.out.brm, PAN_CISTROME_PEAKS.out.peaks)
    MOTIF_CREATION_DISRUPTION(SELECT_CAUSAL_728.out.causal, data_dir)
    REWIRING_ANALYSIS(MOTIF_CREATION_DISRUPTION.out.creation)
    THRESHOLD_SENSITIVITY(SELECT_CAUSAL_728.out.causal, data_dir)

    // Phase 4: Interactive
    BUILD_INTERACTIVE(
        GENE_ATLAS.out.atlas,
        MOTIF_CREATION_DISRUPTION.out.creation,
        REWIRING_ANALYSIS.out.rewiring,
        VERIFY_SOLE_MOTIF.out.sole
    )
}
