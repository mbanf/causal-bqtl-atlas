#!/usr/bin/env nextflow

/*
 * Causal bQTL Atlas Pipeline
 *
 * A 26-step pipeline to identify and characterize high-confidence
 * causal binding QTL from plant pan-cistrome data.
 *
 * Scripts use hardcoded project-relative paths (BASE = project root).
 * Nextflow manages execution order, parallelism, and monitoring.
 *
 * Reference: Banf & Hartwig (2026)
 *
 * Usage:
 *   nextflow run main.nf                             # local
 *   nextflow run main.nf -profile slurm,conda        # HPC + conda
 *   nextflow run main.nf -with-tower                 # with Tower monitoring
 *   nextflow run main.nf --skip "01,02"              # skip steps with existing outputs
 */

nextflow.enable.dsl=2

// Parse skip list into a set for easy lookup
def skipSet = "${params.skip ?: ''}".tokenize(',').collect { it.trim() }.toSet()


// ══════════════════════════════════════════════════════════════════
// PHASE 1: Data Preparation
// ══════════════════════════════════════════════════════════════════

process BOUND_REGION_MOTIFS {
    /*
     * Step 01: Scan MOA-seq binding peaks for TF motifs (PWM database).
     * Produces the motif content map used by downstream characterization.
     */

    output:
    val true, emit: done

    script:
    if ("01" in skipSet)
        "echo 'SKIP: Step 01 — using existing bound_region_motifs.csv'"
    else
        "python3 ${projectDir}/scripts/01_bound_region_motifs.py"
}

process PAN_CISTROME_PEAKS {
    /*
     * Step 02: Merge MOA-seq peaks across all hybrids into a
     * pan-cistrome peak set. Also computes per-gene peak disruption.
     */

    output:
    val true, emit: done

    script:
    if ("02" in skipSet)
        "echo 'SKIP: Step 02 — using existing pan_cistrome_peaks.csv'"
    else
        "python3 ${projectDir}/scripts/02_pan_cistrome_peaks.py"
}

process CONVERT_GENOTYPES {
    /*
     * Step 03: Convert Julia's per-chromosome genotype files
     * to pipeline format (VCF-like TSV with hybrid columns).
     * Only runs if --genotype_dir is provided.
     */

    when:
    params.genotype_dir != null

    output:
    val true, emit: done

    script:
    """
    python3 ${projectDir}/scripts/03_convert_genotypes.py \
        --genotype-dir ${params.genotype_dir} \
        --bqtl ${params.data_dir}/processed/bqtl_snp_ww.csv \
        --output ${params.data_dir}/processed/nam_founder_genotypes_at_bqtl.tsv
    """
}


// ══════════════════════════════════════════════════════════════════
// PHASE 2: Core Causal Pipeline (Six-Step Filtering)
// ══════════════════════════════════════════════════════════════════

process FUNCTIONAL_BQTL {
    /*
     * Step 04: Map bQTL to gene promoters (2kb upstream of TSS).
     * Produces bQTL-gene pairs with ASE data for testing.
     */

    input:
    val ready  // ensures Phase 1 data prep is done

    output:
    val true, emit: done

    script:
    if ("04" in skipSet)
        "echo 'SKIP: Step 04 — using existing functional_bqtl.csv'"
    else
        "python3 ${projectDir}/scripts/04_functional_bqtl.py"
}

process GENOTYPE_BQTL_TEST {
    /*
     * Step 05: Genotype-aware Mann-Whitney test.
     * For each bQTL-gene pair: do hybrids carrying the variant allele
     * show different ASE than reference hybrids? (FDR < 0.05)
     */

    input:
    val ready  // waits for FUNCTIONAL_BQTL

    output:
    val true, emit: done

    script:
    if ("05" in skipSet)
        "echo 'SKIP: Step 05 — using existing genotype_bqtl_results.csv'"
    else
        "python3 ${projectDir}/scripts/05_genotype_bqtl_test.py"
}

process SELECT_CAUSAL {
    /*
     * Step 06: Apply the final filtering steps.
     *   - Keep only genes with exactly one significant bQTL
     *   - bQTL must fall within an MOA-seq peak
     * Produces the final causal bQTL set.
     */

    input:
    val geno_ready   // waits for GENOTYPE_BQTL_TEST
    val peaks_ready  // waits for PAN_CISTROME_PEAKS
    val motifs_ready // waits for BOUND_REGION_MOTIFS

    output:
    val true, emit: done

    script:
    if ("06" in skipSet)
        "echo 'SKIP: Step 06 — using existing causal_bqtl_atlas.csv'"
    else
        "python3 ${projectDir}/scripts/06_select_causal.py"
}


// ══════════════════════════════════════════════════════════════════
// PHASE 3: Characterization
// ══════════════════════════════════════════════════════════════════

process GENE_ATLAS {
    /*
     * Step 07: Deep characterization of all causal bQTL target genes.
     * Signaling layers, TF families, GO/KEGG, drought response.
     */

    input:
    val ready  // waits for SELECT_CAUSAL

    output:
    val true, emit: done

    script:
    if ("07" in skipSet)
        "echo 'SKIP: Step 07 — using existing gene_atlas.csv'"
    else
        "python3 ${projectDir}/scripts/07_gene_atlas.py"
}

process SOLE_MOTIF_VERIFY {
    /*
     * Step 08: For each causal bQTL, verify whether the disrupted
     * TF motif is the sole copy of that family within the MOA-seq peak.
     */

    input:
    val ready  // waits for SELECT_CAUSAL

    output:
    val true, emit: done

    script:
    if ("08" in skipSet)
        "echo 'SKIP: Step 08 — using existing sole_motif_verify.csv'"
    else
        "python3 ${projectDir}/scripts/08_sole_motif_verify.py"
}

process MOTIF_CREATION_DISRUPTION {
    /*
     * Step 09: Bidirectional motif analysis — compare B73 ref vs
     * NAM alt sequence at each bQTL to classify motif changes as
     * disrupted, created, or rewired.
     */

    input:
    val ready  // waits for SELECT_CAUSAL

    output:
    val true, emit: done

    script:
    if ("09" in skipSet)
        "echo 'SKIP: Step 09 — using existing motif_creation_disruption.csv'"
    else
        "python3 ${projectDir}/scripts/09_motif_creation_disruption.py"
}

process PHENOTYPE_RATELIMIT {
    /*
     * Step 10: NAM hybrid phenotype connection (flowering, height)
     * and rate-limiting pathway step enrichment analysis.
     */

    input:
    val ready  // waits for SELECT_CAUSAL

    output:
    val true, emit: done

    script:
    if ("10" in skipSet)
        "echo 'SKIP: Step 10 — using existing phenotype_ratelimit results'"
    else
        "python3 ${projectDir}/scripts/10_phenotype_ratelimit.py"
}

process THRESHOLD_SENSITIVITY {
    /*
     * Step 11: Sweep PWM threshold 4.0–8.0 to verify stability
     * of motif disruption/creation/rewiring findings.
     */

    input:
    val ready  // waits for SELECT_CAUSAL

    output:
    val true, emit: done

    script:
    if ("11" in skipSet)
        "echo 'SKIP: Step 11 — using existing threshold_sensitivity results'"
    else
        "python3 ${projectDir}/scripts/11_threshold_sensitivity.py"
}

process REWIRING_ANALYSIS {
    /*
     * Step 12: Deep analysis of TF motif rewiring — transition
     * matrices, sole-copy switches, signaling layer integration.
     */

    input:
    val atlas_ready     // waits for GENE_ATLAS
    val sole_ready      // waits for SOLE_MOTIF_VERIFY
    val creation_ready  // waits for MOTIF_CREATION_DISRUPTION

    output:
    val true, emit: done

    script:
    if ("12" in skipSet)
        "echo 'SKIP: Step 12 — using existing rewiring_analysis results'"
    else
        "python3 ${projectDir}/scripts/12_rewiring_analysis.py"
}

process INTERACTIVE_ATLAS {
    /*
     * Step 13: Generate interactive HTML visualization of all
     * causal bQTL target genes with filters and AI summaries.
     */

    input:
    val ready  // waits for GENE_ATLAS

    output:
    val true, emit: done

    script:
    if ("13" in skipSet)
        "echo 'SKIP: Step 13 — using existing interactive atlas'"
    else
        "python3 ${projectDir}/scripts/13_build_interactive.py"
}


// ══════════════════════════════════════════════════════════════════
// PHASE 4: Per-Hybrid Linkage & TF Network
// ══════════════════════════════════════════════════════════════════

process HYBRID_MOTIF_LINKAGE {
    /*
     * Step 18b: Per-hybrid haplotype-aware motif scanning with 405
     * DAP-seq PWMs. Tests whether each motif's binding score separates
     * variant vs reference hybrid groups.
     */

    input:
    val ready  // waits for SELECT_CAUSAL

    output:
    val true, emit: done

    script:
    if ("18b" in skipSet)
        "echo 'SKIP: Step 18b — using existing hybrid_motif_linkage.csv'"
    else
        "python3 ${projectDir}/scripts/18b_hybrid_motif_linkage.py"
}

process TF_FEEDBACK_NETWORK {
    /*
     * Step 19: TF family enrichment among causal targets (binomial test).
     * Constructs directed regulatory network: autoregulatory loops and
     * cross-family rewiring.
     */

    input:
    val ready  // waits for HYBRID_MOTIF_LINKAGE

    output:
    val true, emit: done

    script:
    if ("19" in skipSet)
        "echo 'SKIP: Step 19 — using existing tf_feedback_network results'"
    else
        "python3 ${projectDir}/scripts/19_tf_feedback_network.py"
}

process CAUSALITY_PREDICTION {
    /*
     * Step 17: Train classifiers to predict which bQTL are causal
     * from motif features. Validates that causality is learnable (AUC=0.895).
     */

    input:
    val ready  // waits for SELECT_CAUSAL

    output:
    val true, emit: done

    script:
    if ("17" in skipSet)
        "echo 'SKIP: Step 17 — using existing causality_prediction results'"
    else
        "python3 ${projectDir}/scripts/17_causality_prediction.py"
}


process POPULATION_MOTIF_OCCUPANCY {
    /*
     * Step 26: Layer 1 — Population-level TF motif occupancy.
     * Scans MOA-seq peaks across all hybrids per gene for TF motif
     * enrichment, independent of bQTL.
     */

    input:
    val ready  // waits for SELECT_CAUSAL (needs causal bQTL for cross-ref)

    output:
    val true, emit: done

    script:
    if ("26" in skipSet)
        "echo 'SKIP: Step 26 — using existing population_motif_occupancy results'"
    else
        "python3 ${projectDir}/scripts/26_population_motif_occupancy.py"
}

process TF_CASCADE_TRACING {
    /*
     * Step 27: Layer 3 — TF cascade tracing from causal bQTL.
     * Traces disrupted TF → TF gene target → downstream targets.
     * Requires Layer 1 occupancy and Layer 2 linkage.
     */

    input:
    val linkage_ready    // waits for HYBRID_MOTIF_LINKAGE
    val occupancy_ready  // waits for POPULATION_MOTIF_OCCUPANCY

    output:
    val true, emit: done

    script:
    if ("27" in skipSet)
        "echo 'SKIP: Step 27 — using existing tf_cascade results'"
    else
        "python3 ${projectDir}/scripts/27_tf_cascade_tracing.py"
}


// ══════════════════════════════════════════════════════════════════
// PHASE 5: Drought / Condition Analysis
// ══════════════════════════════════════════════════════════════════

process DROUGHT_CAUSAL_BQTL {
    /*
     * Step 20: Repeat causal bQTL pipeline on drought-stress ASE data.
     * Identifies 1,267 drought-causal bQTL (12% overlap with WW).
     */

    input:
    val ready  // waits for SELECT_CAUSAL

    output:
    val true, emit: done

    script:
    if ("20" in skipSet)
        "echo 'SKIP: Step 20 — using existing drought causal bQTL results'"
    else
        "python3 ${projectDir}/scripts/20_drought_causal_bqtl.py"
}

process DROUGHT_EXPRESSION {
    /*
     * Step 21: Integrate drought causal bQTL with population-level
     * expression data. Cis vs trans, effect correlation, condition specificity.
     */

    input:
    val ready  // waits for DROUGHT_CAUSAL_BQTL

    output:
    val true, emit: done

    script:
    if ("21" in skipSet)
        "echo 'SKIP: Step 21 — using existing drought expression results'"
    else
        "python3 ${projectDir}/scripts/21_drought_expression_integration.py"
}

process FUNCTIONAL_CHARACTERIZATION {
    /*
     * Step 22: Gene set decomposition (871 WW-only, 821 DS-only,
     * 129 shared-same, 75 shared-different). EntAP, GO enrichment, UMAP.
     */

    input:
    val ready  // waits for DROUGHT_CAUSAL_BQTL

    output:
    val true, emit: done

    script:
    if ("22" in skipSet)
        "echo 'SKIP: Step 22 — using existing functional characterization results'"
    else
        "python3 ${projectDir}/scripts/22_condition_functional_characterization.py"
}

process COMPARATIVE_ENRICHMENT {
    /*
     * Step 23: Exhaustive functional comparison of WW vs DS gene sets.
     * GO enrichment, detailed category breakdown, Fisher's exact tests.
     */

    input:
    val ready  // waits for FUNCTIONAL_CHARACTERIZATION

    output:
    val true, emit: done

    script:
    if ("23" in skipSet)
        "echo 'SKIP: Step 23 — using existing comparative enrichment results'"
    else
        "python3 ${projectDir}/scripts/23_comparative_enrichment.py"
}


process CROSS_LAYER_VALIDATION {
    /*
     * Step 28: Cross-layer motif validation and differential analysis.
     * Module A: Conservation-impact scoring (Layer 1 × Layer 2)
     * Module B: WW vs DS differential occupancy (requires DS Layer 1)
     * Module C: Hybrid-specific regulatory damage scoring
     */

    input:
    val cascade_ready     // waits for TF_CASCADE_TRACING
    val drought_ready     // waits for DROUGHT_CAUSAL_BQTL (DS linkage)

    output:
    val true, emit: done

    script:
    if ("28" in skipSet)
        "echo 'SKIP: Step 28 — using existing cross-layer validation results'"
    else
        "python3 ${projectDir}/scripts/28_cross_layer_validation.py"
}

process ASE_MOTIF_VALIDATION {
    /*
     * Step 29: ASE × motif conservation validation.
     * Closes the causal loop: disruption of conserved motifs → stronger ASE.
     * Tests alt vs ref allele ASE, per-family effects, dose-response.
     */

    input:
    val ready  // waits for CROSS_LAYER_VALIDATION

    output:
    val true, emit: done

    script:
    if ("29" in skipSet)
        "echo 'SKIP: Step 29 — using existing ASE-motif validation results'"
    else
        "python3 ${projectDir}/scripts/29_ase_motif_validation.py"
}

process TF_COOPERATIVITY {
    /*
     * Step 30: TF cooperativity network and cis-regulatory logic.
     * Co-occupancy analysis, enhanceosome vs billboard classification,
     * WW vs DS differential partnerships, co-disruption synergy.
     */

    input:
    val ase_ready      // waits for ASE_MOTIF_VALIDATION (needs its output)
    val occupancy_ready // waits for POPULATION_MOTIF_OCCUPANCY

    output:
    val true, emit: done

    script:
    if ("30" in skipSet)
        "echo 'SKIP: Step 30 — using existing TF cooperativity results'"
    else
        "python3 ${projectDir}/scripts/30_tf_cooperativity.py"
}


process FLOW_MATCHING_DESIGN {
    /*
     * Step 31: Conditional flow matching for regulatory sequence design.
     * Trains a generative model that maps desired TF binding changes to
     * minimal sequence edits, guided by Agro-NT LoRA scoring.
     * Includes ASE prediction head and CRISPR base-edit design.
     */

    input:
    val coop_ready  // waits for TF_COOPERATIVITY (needs motif grammar)

    output:
    val true, emit: done

    script:
    if ("31" in skipSet)
        "echo 'SKIP: Step 31 — using existing flow matching model'"
    else
        "python3 ${projectDir}/scripts/31_flow_matching_design.py train"
}


// ══════════════════════════════════════════════════════════════════
// PHASE 5: Validation
// ══════════════════════════════════════════════════════════════════

process VERIFY_PIPELINE {
    /*
     * Run verification checks on all pipeline outputs.
     */

    input:
    val ready  // waits for everything

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



// ══════════════════════════════════════════════════════════════════
// DYNAMICALLY ADDED PROCESSES
// ══════════════════════════════════════════════════════════════════

process POPULATION_GENETICS {
    /*
     * Step 31b: Population Genetics
     * Allele frequency spectrum, effect-frequency relationship, and selection signatures at causal bQTL. Tests whether causal 
     */

    input:
    val ready

    output:
    val true, emit: done

    script:
    if ("31b" in skipSet)
        "echo 'SKIP: Step 31b — using existing Population Genetics results'"
    else
        "python3 ${projectDir}/scripts/31_population_genetics.py"
}

process HYBRID_PHENOTYPE_SIMILARITY {
    /*
     * Step 32: Hybrid Phenotype Similarity
     * Cross-hybrid phenotype similarity from shared causal bQTL burden. Tests whether hybrids sharing more causal variants hav
     */

    input:
    val ready

    output:
    val true, emit: done

    script:
    if ("32" in skipSet)
        "echo 'SKIP: Step 32 — using existing Hybrid Phenotype Similarity results'"
    else
        "python3 ${projectDir}/scripts/32_hybrid_phenotype_similarity.py"
}

process GWAS_OVERLAP {
    /*
     * Step 33: GWAS Overlap
     * Overlap between causal bQTL and published GWAS hits. Tests enrichment of causal variants near trait-associated loci from
     */

    input:
    val ready

    output:
    val true, emit: done

    script:
    if ("33" in skipSet)
        "echo 'SKIP: Step 33 — using existing GWAS Overlap results'"
    else
        "python3 ${projectDir}/scripts/33_gwas_overlap.py"
}

process SCATAC_SEQ_INTEGRATION {
    /*
     * Step 34: scATAC-seq Integration
     * Integrate causal bQTL with single-cell ATAC-seq accessible chromatin regions (Marand et al. 2021). Tests cell-type enric
     */

    input:
    val ready

    output:
    val true, emit: done

    script:
    if ("34" in skipSet)
        "echo 'SKIP: Step 34 — using existing scATAC-seq Integration results'"
    else
        "python3 ${projectDir}/scripts/34_scatac_integration.py"
}

process NETWORK_TOPOLOGY {
    /*
     * Step 35: Network Topology
     * Test whether causal bQTL target genes occupy different positions in the TF regulatory network (degree, entropy, hub stat
     */

    input:
    val ready

    output:
    val true, emit: done

    script:
    if ("35" in skipSet)
        "echo 'SKIP: Step 35 — using existing Network Topology results'"
    else
        "python3 ${projectDir}/scripts/35_network_topology.py"
}

process BQTL_EPISTASIS {
    /*
     * Step 36: bQTL Epistasis
     * Test additive vs epistatic combination of multiple bQTL per gene. Dose-response: each variant allele adds ~0.056 |ASE|. 
     */

    input:
    val ready

    output:
    val true, emit: done

    script:
    if ("36" in skipSet)
        "echo 'SKIP: Step 36 — using existing bQTL Epistasis results'"
    else
        "python3 ${projectDir}/scripts/36_bqtl_epistasis.py"
}

process TE_CONTEXT {
    /*
     * Step 37: TE Context
     * Classify all 147,942 bQTL by transposable element superfamily using EDTA annotations. Causal bQTL are depleted in TEs (1
     */

    input:
    val ready

    output:
    val true, emit: done

    script:
    if ("37" in skipSet)
        "echo 'SKIP: Step 37 — using existing TE Context results'"
    else
        "python3 ${projectDir}/scripts/37_te_context.py"
}

process CHROMATIN_CONTEXT {
    /*
     * Step 38: Chromatin Context
     * Classify bQTL by gene-relative position and Marand ACR overlap as chromatin state proxy. 3.51x enrichment at proximal pr
     */

    input:
    val ready

    output:
    val true, emit: done

    script:
    if ("38" in skipSet)
        "echo 'SKIP: Step 38 — using existing Chromatin Context results'"
    else
        "python3 ${projectDir}/scripts/38_chromatin_context.py"
}

process BINDING_EXPRESSION_CAUSAL_CHAIN {
    /*
     * Step 39: Binding-Expression Causal Chain
     * Test binding-to-expression causal chain using ASE as eQTL proxy. 73.7% concordance (830/1126 genes), effect size predict
     */

    input:
    val ready

    output:
    val true, emit: done

    script:
    if ("39" in skipSet)
        "echo 'SKIP: Step 39 — using existing Binding-Expression Causal Chain results'"
    else
        "python3 ${projectDir}/scripts/39_eqtl_proxy.py"
}

workflow {

    // ── Phase 1: Data preparation (parallel) ──
    BOUND_REGION_MOTIFS()
    PAN_CISTROME_PEAKS()

    if (params.genotype_dir != null) {
        CONVERT_GENOTYPES()
    }

    // ── Phase 2: Core pipeline (sequential) ──
    FUNCTIONAL_BQTL(PAN_CISTROME_PEAKS.out.done)

    GENOTYPE_BQTL_TEST(FUNCTIONAL_BQTL.out.done)

    SELECT_CAUSAL(
        GENOTYPE_BQTL_TEST.out.done,
        PAN_CISTROME_PEAKS.out.done,
        BOUND_REGION_MOTIFS.out.done
    )

    // ── Phase 3: Characterization (parallel after SELECT_CAUSAL) ──
    GENE_ATLAS(SELECT_CAUSAL.out.done)
    SOLE_MOTIF_VERIFY(SELECT_CAUSAL.out.done)
    MOTIF_CREATION_DISRUPTION(SELECT_CAUSAL.out.done)
    PHENOTYPE_RATELIMIT(SELECT_CAUSAL.out.done)
    THRESHOLD_SENSITIVITY(SELECT_CAUSAL.out.done)

    // Depends on Phase 3 results
    REWIRING_ANALYSIS(
        GENE_ATLAS.out.done,
        SOLE_MOTIF_VERIFY.out.done,
        MOTIF_CREATION_DISRUPTION.out.done
    )

    INTERACTIVE_ATLAS(GENE_ATLAS.out.done)

    // ── Phase 4: Per-hybrid linkage, population occupancy & TF network ──
    HYBRID_MOTIF_LINKAGE(SELECT_CAUSAL.out.done)
    TF_FEEDBACK_NETWORK(HYBRID_MOTIF_LINKAGE.out.done)
    CAUSALITY_PREDICTION(SELECT_CAUSAL.out.done)
    POPULATION_MOTIF_OCCUPANCY(SELECT_CAUSAL.out.done)
    TF_CASCADE_TRACING(
        HYBRID_MOTIF_LINKAGE.out.done,
        POPULATION_MOTIF_OCCUPANCY.out.done
    )

    // ── Phase 5: Drought analysis ──
    DROUGHT_CAUSAL_BQTL(SELECT_CAUSAL.out.done)
    DROUGHT_EXPRESSION(DROUGHT_CAUSAL_BQTL.out.done)
    FUNCTIONAL_CHARACTERIZATION(DROUGHT_CAUSAL_BQTL.out.done)
    COMPARATIVE_ENRICHMENT(FUNCTIONAL_CHARACTERIZATION.out.done)

    // ── Cross-layer integration (needs both cascade + drought) ──
    CROSS_LAYER_VALIDATION(
        TF_CASCADE_TRACING.out.done,
        DROUGHT_CAUSAL_BQTL.out.done
    )

    // ── ASE validation + TF cooperativity ──
    ASE_MOTIF_VALIDATION(CROSS_LAYER_VALIDATION.out.done)
    TF_COOPERATIVITY(
        ASE_MOTIF_VALIDATION.out.done,
        POPULATION_MOTIF_OCCUPANCY.out.done
    )

    // ── Generative design (needs cooperativity grammar) ──
    FLOW_MATCHING_DESIGN(TF_COOPERATIVITY.out.done)

    // ── Extended analyses (dynamically added) ──
    POPULATION_GENETICS(SELECT_CAUSAL.out.done)
    HYBRID_PHENOTYPE_SIMILARITY(SELECT_CAUSAL.out.done)
    GWAS_OVERLAP(SELECT_CAUSAL.out.done)
    SCATAC_SEQ_INTEGRATION(SELECT_CAUSAL.out.done)
    NETWORK_TOPOLOGY(TF_FEEDBACK_NETWORK.out.done)
    BQTL_EPISTASIS(SELECT_CAUSAL.out.done)
    TE_CONTEXT(SELECT_CAUSAL.out.done)
    CHROMATIN_CONTEXT(SELECT_CAUSAL.out.done)
    BINDING_EXPRESSION_CAUSAL_CHAIN(DROUGHT_CAUSAL_BQTL.out.done)

    // ── Phase 6: Verification ──
    VERIFY_PIPELINE(
        REWIRING_ANALYSIS.out.done
            .mix(INTERACTIVE_ATLAS.out.done)
            .mix(THRESHOLD_SENSITIVITY.out.done)
            .mix(PHENOTYPE_RATELIMIT.out.done)
            .mix(TF_FEEDBACK_NETWORK.out.done)
            .mix(CAUSALITY_PREDICTION.out.done)
            .mix(TF_CASCADE_TRACING.out.done)
            .mix(COMPARATIVE_ENRICHMENT.out.done)
            .mix(DROUGHT_EXPRESSION.out.done)
            .mix(FLOW_MATCHING_DESIGN.out.done)
            .collect()
    )
}
