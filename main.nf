#!/usr/bin/env nextflow

/*
 * Causal bQTL Atlas Pipeline
 *
 * A six-step pipeline to identify high-confidence causal binding QTL
 * from plant pan-cistrome data.
 *
 * Reference: Banf & Hartwig (2026)
 *
 * Usage:
 *   nextflow run main.nf                             # local
 *   nextflow run main.nf -profile slurm,conda        # HPC + conda
 *   nextflow run main.nf -profile slurm,singularity  # HPC + container
 *   nextflow run main.nf --paper_only                # skip exploratory analyses
 */

nextflow.enable.dsl=2


// ══════════════════════════════════════════════════════════════════
// PHASE 1: Data Preparation
// ══════════════════════════════════════════════════════════════════

process BOUND_REGION_MOTIFS {
    /*
     * Script 47: Scan MOA-seq binding peaks for TF motifs (PWM database).
     * Produces the motif content map used by downstream characterization.
     */
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path peaks_dir
    path pwm_database
    path genome_fasta

    output:
    path "bound_region_motifs.csv", emit: motifs

    script:
    """
    python3 ${projectDir}/scripts/47_moa_bound_region_motifs.py \
        --peaks-dir ${peaks_dir} \
        --pwm-db ${pwm_database} \
        --genome ${genome_fasta} \
        --output bound_region_motifs.csv \
        --threshold ${params.motif_score_threshold}
    """
}

process PAN_CISTROME_PEAKS {
    /*
     * Script 48: Merge MOA-seq peaks across all hybrids into a
     * pan-cistrome peak set. Also computes per-gene peak disruption.
     */
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path peaks_dir
    path gff3

    output:
    path "pan_cistrome_peaks.csv",   emit: peaks
    path "gene_peak_disruption.csv", emit: disruption

    script:
    """
    python3 ${projectDir}/scripts/48_moa_peak_disruption.py \
        --peaks-dir ${peaks_dir} \
        --gff3 ${gff3} \
        --output-peaks pan_cistrome_peaks.csv \
        --output-disruption gene_peak_disruption.csv \
        --promoter-size ${params.promoter_size}
    """
}

process CONVERT_GENOTYPES {
    /*
     * Script 53c: Convert Julia's per-chromosome genotype files
     * to pipeline format (VCF-like TSV with hybrid columns).
     * Only runs if --genotype_dir is provided.
     */
    publishDir "${params.data_dir}/processed", mode: 'copy'

    input:
    path genotype_dir
    path bqtl_file

    output:
    path "nam_founder_genotypes_at_bqtl.tsv", emit: genotypes

    when:
    params.genotype_dir != null

    script:
    """
    python3 ${projectDir}/scripts/53c_convert_julia_genotypes.py \
        --genotype-dir ${genotype_dir} \
        --bqtl ${bqtl_file} \
        --output nam_founder_genotypes_at_bqtl.tsv
    """
}


// ══════════════════════════════════════════════════════════════════
// PHASE 2: Core Causal Pipeline (Six-Step Filtering)
// ══════════════════════════════════════════════════════════════════

process FUNCTIONAL_BQTL {
    /*
     * Script 50: Map bQTL to gene promoters (2kb upstream of TSS).
     * Produces bQTL-gene pairs with ASE data for testing.
     */
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path bqtl_file
    path ase_file
    path gff3
    path peaks_dir

    output:
    path "bqtl_functional_test.csv", emit: functional

    script:
    """
    python3 ${projectDir}/scripts/50_functional_bqtl.py \
        --bqtl ${bqtl_file} \
        --ase ${ase_file} \
        --gff3 ${gff3} \
        --peaks-dir ${peaks_dir} \
        --output bqtl_functional_test.csv \
        --promoter-size ${params.promoter_size}
    """
}

process GENOTYPE_BQTL_TEST {
    /*
     * Script 54: Genotype-aware Mann-Whitney test.
     * For each bQTL-gene pair: do hybrids carrying the variant allele
     * show different ASE than reference hybrids? (FDR < 0.05)
     */
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path functional
    path genotypes
    path ase_file
    path gff3

    output:
    path "genotype_bqtl_results.csv", emit: results

    script:
    """
    python3 ${projectDir}/scripts/54_genotype_aware_bqtl_test.py \
        --functional ${functional} \
        --genotypes ${genotypes} \
        --ase ${ase_file} \
        --gff3 ${gff3} \
        --output genotype_bqtl_results.csv \
        --fdr-threshold ${params.fdr_threshold}
    """
}

process SELECT_CAUSAL {
    /*
     * Script 63: Apply Steps 5-6 of the filtering pipeline.
     *   Step 5: Keep only genes with exactly one significant bQTL
     *   Step 6: bQTL must fall within an MOA-seq peak
     * Produces the final causal bQTL set.
     */
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path geno_results
    path pan_peaks
    path genotypes
    path entap_annot
    path expression
    path motifs

    output:
    path "causal_bqtl_728.csv",     emit: causal
    path "regulatory_map_728.csv",   emit: reg_map

    script:
    """
    python3 ${projectDir}/scripts/63_select_causal_728.py \
        --results ${geno_results} \
        --peaks ${pan_peaks} \
        --genotypes ${genotypes} \
        --entap ${entap_annot} \
        --expression ${expression} \
        --motifs ${motifs} \
        --output-causal causal_bqtl_728.csv \
        --output-regmap regulatory_map_728.csv
    """
}


// ══════════════════════════════════════════════════════════════════
// PHASE 3: Characterization
// ══════════════════════════════════════════════════════════════════

process GENE_ATLAS {
    /*
     * Script 66: Deep characterization of all causal bQTL target genes.
     * Signaling layers, TF families, GO/KEGG, drought response.
     */
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path causal
    path reg_map
    path motifs
    path entap_annot
    path expression

    output:
    path "gene_atlas_728.csv", emit: atlas

    script:
    """
    python3 ${projectDir}/scripts/66_gene_atlas_728.py
    """
}

process SOLE_MOTIF_VERIFY {
    /*
     * Script 68: For each causal bQTL, verify whether the disrupted
     * TF motif is the sole copy of that family within the MOA-seq peak.
     */
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path causal
    path peaks_dir
    path pwm_database
    path genome_fasta

    output:
    path "sole_motif_verification_728.csv", emit: sole_motif

    script:
    """
    python3 ${projectDir}/scripts/68_verify_sole_motif.py
    """
}

process MOTIF_CREATION_DISRUPTION {
    /*
     * Script 69: Bidirectional motif analysis — compare B73 ref vs
     * NAM alt sequence at each bQTL to classify motif changes as
     * disrupted, created, or rewired.
     */
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path causal
    path genotypes
    path pwm_database
    path genome_fasta

    output:
    path "motif_creation_vs_disruption_728.csv", emit: creation

    script:
    """
    python3 ${projectDir}/scripts/69_motif_creation_vs_disruption.py
    """
}

process REWIRING_ANALYSIS {
    /*
     * Script 70: Deep analysis of TF motif rewiring — transition
     * matrices, sole-copy switches, signaling layer integration.
     */
    publishDir "${params.results_dir}", mode: 'copy'
    publishDir "${params.figures_dir}/paper_figures", mode: 'copy', pattern: '*.{pdf,png}'

    input:
    path causal
    path creation
    path sole_motif
    path genotypes
    path atlas
    path peaks_dir
    path pwm_database
    path genome_fasta

    output:
    path "rewiring_deep_analysis_728.csv",  emit: rewiring
    path "tf_family_transition_matrix.csv",  emit: transitions
    path "*.pdf",                            optional: true
    path "*.png",                            optional: true

    script:
    """
    python3 ${projectDir}/scripts/70_rewiring_deep_analysis.py
    """
}

process PHENOTYPE_RATELIMIT {
    /*
     * Script 65: NAM hybrid phenotype connection (flowering, height)
     * and rate-limiting pathway step enrichment analysis.
     */
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path causal
    path reg_map
    path entap_annot

    output:
    path "phenotype_associations_728.csv",  emit: phenotype
    path "enzyme_classification_728.csv",   emit: enzyme

    script:
    """
    python3 ${projectDir}/scripts/65_phenotype_ratelimit.py
    """
}

process THRESHOLD_SENSITIVITY {
    /*
     * Script 72: Sweep PWM threshold 4.0–8.0 to verify stability
     * of motif disruption/creation/rewiring findings.
     */
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path causal
    path genotypes
    path pwm_database
    path genome_fasta

    output:
    path "threshold_sensitivity_sweep.csv", emit: sweep

    script:
    """
    python3 ${projectDir}/scripts/72_threshold_sensitivity.py
    """
}

process INTERACTIVE_ATLAS {
    /*
     * Script 67: Generate interactive HTML visualization of all
     * causal bQTL target genes with filters and AI summaries.
     */
    publishDir "${projectDir}/interactive", mode: 'copy'

    input:
    path atlas
    path causal

    output:
    path "gene_atlas_728_interactive.html", emit: html

    script:
    """
    python3 ${projectDir}/scripts/67_build_interactive.py
    """
}


// ══════════════════════════════════════════════════════════════════
// PHASE 4: Drought / Condition Analysis
// ══════════════════════════════════════════════════════════════════

process WW_VS_DROUGHT {
    /*
     * Script 58: Repeat genotype→ASE test under drought stress.
     * Compare WW vs DS allele-specific expression patterns.
     */
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path genotypes
    path bqtl_file
    path gff3
    path ww_results
    path motifs

    output:
    path "ww_vs_drought_genotype_bqtl.csv", emit: drought_results

    script:
    """
    python3 ${projectDir}/scripts/58_drought_genotype_ase_test.py
    """
}

process CONDITION_SWITCHING {
    /*
     * Script 60: Deep dive into condition-dependent (WW↔DS) bQTL
     * behavior — which bQTL switch from significant to non-significant?
     */
    publishDir "${params.results_dir}", mode: 'copy'

    input:
    path drought_results
    path grammar
    path expression

    output:
    path "condition_switching_deepdive.csv", emit: switching

    script:
    """
    python3 ${projectDir}/scripts/60_condition_switching_deepdive.py
    """
}


// ══════════════════════════════════════════════════════════════════
// PHASE 5: Validation
// ══════════════════════════════════════════════════════════════════

process VERIFY_PIPELINE {
    /*
     * Run verification checks on all pipeline outputs.
     */
    input:
    path results_dir

    output:
    stdout

    script:
    """
    python3 ${projectDir}/scripts/verify_pipeline.py
    """
}


// ══════════════════════════════════════════════════════════════════
// WORKFLOW
// ══════════════════════════════════════════════════════════════════

workflow {

    // ── Input channels ──
    peaks_dir      = Channel.fromPath("${params.data_dir}/raw/engelhorn_peaks", type: 'dir')
    bqtl_file      = Channel.fromPath("${params.data_dir}/processed/bqtl_snp_ww.csv")
    ase_file       = Channel.fromPath("${params.data_dir}/processed/engelhorn_ase_ww.csv")
    gff3           = Channel.fromPath(params.gff3)
    genome_fasta   = Channel.fromPath(params.genome_fasta)
    pwm_database   = Channel.fromPath(params.pwm_database)
    entap_annot    = Channel.fromPath("${params.data_dir}/raw/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1_entap_results.tsv.gz")
    expression     = Channel.fromPath("${params.data_dir}/processed/engelhorn_ww_vs_ds_expression.tsv")

    // Genotypes: use pre-converted file or convert from Julia's format
    if (params.skip_genotype_download) {
        genotypes = Channel.fromPath("${params.data_dir}/processed/nam_founder_genotypes_at_bqtl.tsv")
    } else {
        genotype_dir = Channel.fromPath(params.genotype_dir, type: 'dir')
        CONVERT_GENOTYPES(genotype_dir, bqtl_file)
        genotypes = CONVERT_GENOTYPES.out.genotypes
    }

    // ── Phase 1: Data preparation (parallel) ──
    BOUND_REGION_MOTIFS(peaks_dir, pwm_database, genome_fasta)
    PAN_CISTROME_PEAKS(peaks_dir, gff3)

    // ── Phase 2: Core pipeline (sequential) ──
    FUNCTIONAL_BQTL(bqtl_file, ase_file, gff3, peaks_dir)

    GENOTYPE_BQTL_TEST(
        FUNCTIONAL_BQTL.out.functional,
        genotypes,
        ase_file,
        gff3
    )

    SELECT_CAUSAL(
        GENOTYPE_BQTL_TEST.out.results,
        PAN_CISTROME_PEAKS.out.peaks,
        genotypes,
        entap_annot,
        expression,
        BOUND_REGION_MOTIFS.out.motifs
    )

    // ── Phase 3: Characterization (parallel after SELECT_CAUSAL) ──
    GENE_ATLAS(
        SELECT_CAUSAL.out.causal,
        SELECT_CAUSAL.out.reg_map,
        BOUND_REGION_MOTIFS.out.motifs,
        entap_annot,
        expression
    )

    SOLE_MOTIF_VERIFY(
        SELECT_CAUSAL.out.causal,
        peaks_dir,
        pwm_database,
        genome_fasta
    )

    MOTIF_CREATION_DISRUPTION(
        SELECT_CAUSAL.out.causal,
        genotypes,
        pwm_database,
        genome_fasta
    )

    PHENOTYPE_RATELIMIT(
        SELECT_CAUSAL.out.causal,
        SELECT_CAUSAL.out.reg_map,
        entap_annot
    )

    THRESHOLD_SENSITIVITY(
        SELECT_CAUSAL.out.causal,
        genotypes,
        pwm_database,
        genome_fasta
    )

    // ── Phase 3b: Depends on Phase 3 results ──
    REWIRING_ANALYSIS(
        SELECT_CAUSAL.out.causal,
        MOTIF_CREATION_DISRUPTION.out.creation,
        SOLE_MOTIF_VERIFY.out.sole_motif,
        genotypes,
        GENE_ATLAS.out.atlas,
        peaks_dir,
        pwm_database,
        genome_fasta
    )

    INTERACTIVE_ATLAS(
        GENE_ATLAS.out.atlas,
        SELECT_CAUSAL.out.causal
    )

    // ── Phase 4: Drought analysis ──
    if (!params.paper_only) {
        WW_VS_DROUGHT(
            genotypes,
            bqtl_file,
            gff3,
            GENOTYPE_BQTL_TEST.out.results,
            BOUND_REGION_MOTIFS.out.motifs
        )

        CONDITION_SWITCHING(
            WW_VS_DROUGHT.out.drought_results,
            Channel.fromPath("${params.results_dir}/functional_bqtl_grammar.csv"),
            expression
        )
    }
}
