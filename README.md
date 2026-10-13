# Causal bQTL Atlas Pipeline

A Nextflow DSL2 pipeline and AI agent for identifying high-confidence causal binding QTL (bQTL) from plant pan-cistrome data.

**Reference:** Banf, M. and Hartwig, T. (2026). *A high-confidence causal bQTL atlas reveals the regulatory architecture of natural gene expression variation in maize.*

---

## Pipeline overview

The pipeline implements a four-step filtering strategy that takes 147,942 bQTL from the maize pan-cistrome (Engelhorn et al. 2025, Nature Genetics) and identifies **1,126 high-confidence causal variants** using 24 NAM hybrids. It then performs deep characterization across 35 analysis steps organized in 6 phases.

Each causal bQTL must:
1. Fall within 3 kb of a gene's TSS (empirically determined distance threshold)
2. Show allele-specific expression across NAM hybrids
3. Pass genotype-aware Mann–Whitney test (FDR < 0.05)
4. Be the **only** significant bQTL in its target gene's promoter

```
Phase 1 — Data Preparation (parallel)
┌──────────────────────────────────┐
│ 01  Bound region motifs          │
│ 02  Pan-cistrome peaks           │
│ 03  Convert genotypes (optional) │
└───────────────┬──────────────────┘
                │
Phase 2 — Core Causal Pipeline (sequential)
┌───────────────┴──────────────────┐
│ 04  Functional bQTL mapping      │
│ 05  Genotype → ASE causal test   │
│ 06  Select causal set            │
└───────────────┬──────────────────┘
                │
Phase 3 — Characterization (parallel)
┌───────────────┴──────────────────┐
│ 07  Gene atlas                   │
│ 08  Sole motif verification      │
│ 09  Motif creation / disruption  │
│ 10  Phenotype & rate-limiting    │
│ 11  Threshold sensitivity        │
│ 12  Rewiring analysis            │
│ 13  Interactive HTML atlas       │
└───────────────┬──────────────────┘
                │
Phase 4 — Per-Hybrid Linkage & TF Network
┌───────────────┴──────────────────┐
│ 18b Per-hybrid motif linkage     │
│ 19  TF feedback network          │
│ 17  Causality prediction (ML)    │
│ 26  Population motif occupancy   │
│ 27  TF cascade tracing           │
└───────────────┬──────────────────┘
                │
Phase 5 — Drought & Cross-Layer Validation
┌───────────────┴──────────────────┐
│ 20  Drought causal bQTL          │
│ 21  Drought expression           │
│ 22  Functional characterization  │
│ 23  Comparative enrichment       │
│ 28  Cross-layer validation       │
│ 29  ASE × motif validation       │
│ 30  TF cooperativity             │
│ 31  Flow matching design         │
└───────────────┬──────────────────┘
                │
Phase 6 — Extended Analyses (parallel)
┌───────────────┴──────────────────┐
│ 31b Population genetics          │
│ 32  Hybrid phenotype similarity  │
│ 33  GWAS overlap                 │
│ 34  scATAC-seq integration       │
│ 35  Network topology             │
│ 36  bQTL epistasis               │
│ 37  TE context                   │
│ 38  Chromatin context            │
│ 39  Binding–expression chain     │
└──────────────────────────────────┘
```

---

## Quick start

### Prerequisites

- [Nextflow](https://www.nextflow.io/) >= 23.04
- Python 3.10+ with dependencies: `pip install -e .` or `uv sync`
- Data files (~3.4 GB) — see [DATA_MANIFEST.md](DATA_MANIFEST.md)

### Option A: Run the Nextflow pipeline directly

```bash
git clone https://github.com/mbanf/causal-bqtl-atlas.git
cd causal-bqtl-atlas

# Download data (see DATA_MANIFEST.md)
ln -s /path/to/data/raw data/raw
ln -s /path/to/data/processed data/processed

# Run the full pipeline
nextflow run main.nf

# Skip steps that already have outputs
nextflow run main.nf --skip "01,02,03"

# Resume after interruption
nextflow run main.nf -resume
```

### Option B: Use the Atlas Agent

The Atlas Agent is an AI-powered CLI that can run, validate, explain, and dynamically extend the pipeline.

```bash
# Install agent dependencies
pip install pydantic-ai rich

# Quick validation — check which steps have completed
python atlas_agent.py --validate

# Print a results summary
python atlas_agent.py --explain

# Run specific steps
python atlas_agent.py --run 37 38 39

# Interactive chat mode (requires API key)
export OPENAI_API_KEY=sk-...
python atlas_agent.py

# Or with Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python atlas_agent.py --model anthropic:claude-sonnet-4-20250514
```

### Option C: Run on HPC

```bash
# SLURM + conda
nextflow run main.nf -profile slurm,conda

# SLURM + Singularity (no root required)
nextflow run main.nf -profile slurm,singularity

# PBS/Torque
nextflow run main.nf -profile pbs,conda
```

---

## Atlas Agent tutorial

The agent has three levels of pipeline control:

### Level 1: Run individual scripts

```
you> Run the TE context analysis
agent> Running step 37... Step 37 completed successfully.
       Causal bQTL in TE: 12.2% vs non-causal 20.6% (LTR 7× depleted)
```

Or from the CLI:
```bash
python atlas_agent.py --run 37
```

### Level 2: Batch run with dependency resolution

```
you> Run steps 35 through 39
agent> Resolving dependencies...
       Batch run order: 35 -> 36 -> 37 -> 38 -> 39
       [35] Network Topology — SUCCESS
       [36] bQTL Epistasis — SUCCESS
       [37] TE Context — SUCCESS
       [38] Chromatin Context — SUCCESS
       [39] Binding-Expression Chain — SUCCESS
```

The agent automatically resolves transitive dependencies — if step 35 depends on step 19 and step 19 hasn't run, it runs 19 first. Steps with existing outputs are skipped unless you say "force rerun".

### Level 3: Dynamically extend the Nextflow pipeline

```
you> Show me the pipeline DAG
agent> Pipeline DAG:
         [done] [01] Bound Region Motifs
         [done] [02] Pan-Cistrome Peaks
         ...
         [done] [37] TE Context
         [done] [38] Chromatin Context
         [done] [39] Binding-Expression Causal Chain

you> Preview a Nextflow process for the TE context step
agent> process TE_CONTEXT {
           input:
           val ready
           output:
           val true, emit: done
           script:
           if ("37" in skipSet)
               "echo 'SKIP: Step 37'"
           else
               "python3 ${projectDir}/scripts/37_te_context.py"
       }

you> Add steps 35 through 39 to the Nextflow pipeline
agent> Successfully added 5 processes to main.nf:
         - NETWORK_TOPOLOGY (step 35)
         - BQTL_EPISTASIS (step 36)
         - TE_CONTEXT (step 37)
         - CHROMATIN_CONTEXT (step 38)
         - BINDING_EXPRESSION_CAUSAL_CHAIN (step 39)
       Run with: nextflow run main.nf -resume
```

### Other agent capabilities

```
you> Validate the pipeline
you> What analyses haven't been run yet?
you> Explain the TF cooperativity results
you> Check all paper figures — are any stale?
you> Verify the statistics in the paper match the results
you> Show me the gene atlas for step 07
you> Regenerate the rewiring figure
```

---

## Pipeline parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--skip` | `""` | Comma-separated step IDs to skip |
| `--skip_genotype_download` | `true` | Use pre-converted genotypes |
| `--genotype_dir` | `null` | Path to per-chromosome genotype files |
| `--promoter_size` | `3000` | Promoter window (bp upstream of TSS) |
| `--fdr_threshold` | `0.05` | FDR cutoff for genotype→ASE test |
| `--motif_score_threshold` | `6.0` | PWM log-odds score threshold |

## Key outputs

| File | Description |
|------|-------------|
| `results/causal_bqtl_728.csv` | 1,126 high-confidence causal bQTL (WW) |
| `results/causal_bqtl_ds.csv` | 1,267 drought-stress causal bQTL |
| `results/gene_atlas_728.csv` | Full gene characterization |
| `results/causal_regulatory_network.csv` | 6,787-edge TF regulatory network |
| `results/allele_specific_motifs.csv` | Per-hybrid motif linkage |
| `results/ase_motif_validation.csv` | ASE × motif causal validation |
| `results/tf_cooperativity_network.csv` | TF co-occupancy network |
| `results/bqtl_allele_frequencies.csv` | Population genetics |
| `results/te_context.json` | TE superfamily enrichment |
| `results/chromatin_context.json` | Chromatin state classification |
| `results/eqtl_proxy.json` | Binding→expression causal chain |
| `figures/paper_figures/*.pdf` | All paper figures |

## Scripts

| Step | Script | Description |
|------|--------|-------------|
| 01 | `01_bound_region_motifs.py` | Scan MOA-seq peaks for TF motifs (405 DAP-seq PWMs) |
| 02 | `02_pan_cistrome_peaks.py` | Merge peaks into pan-cistrome set |
| 03 | `03_convert_genotypes.py` | Convert per-chromosome genotypes |
| 04 | `04_functional_bqtl.py` | Map bQTL to gene promoters (3 kb) |
| 05 | `05_genotype_bqtl_test.py` | Genotype-aware Mann–Whitney causal test |
| 06 | `06_select_causal.py` | Apply uniqueness filter → 1,126 causal bQTL |
| 07 | `07_gene_atlas.py` | Gene × TF heatmap, functional categories |
| 08 | `08_sole_motif_verify.py` | Sole-copy motif verification within peaks |
| 09 | `09_motif_creation_disruption.py` | Bidirectional motif analysis (disruption/creation/rewiring) |
| 10 | `10_phenotype_ratelimit.py` | Phenotype associations & rate-limiting |
| 11 | `11_threshold_sensitivity.py` | PWM threshold sweep (4.0–8.0) |
| 12 | `12_rewiring_analysis.py` | TF motif rewiring deep analysis |
| 13 | `13_build_interactive.py` | Interactive D3.js HTML atlas |
| 17 | `17_causality_prediction.py` | ML prediction of causality (AUC = 0.895) |
| 18b | `18b_hybrid_motif_linkage.py` | Per-hybrid haplotype-aware motif scanning |
| 19 | `19_tf_feedback_network.py` | TF feedback loops & regulatory network |
| 20 | `20_drought_causal_bqtl.py` | Drought-stress causal bQTL (1,267 variants) |
| 21 | `21_drought_expression_integration.py` | Drought × expression integration |
| 22 | `22_condition_functional_characterization.py` | WW vs DS gene set decomposition |
| 23 | `23_comparative_enrichment.py` | GO/functional enrichment comparison |
| 26 | `26_population_motif_occupancy.py` | Population-level TF motif occupancy |
| 27 | `27_tf_cascade_tracing.py` | TF cascade tracing (bQTL → TF → targets) |
| 28 | `28_cross_layer_validation.py` | Conservation-impact scoring |
| 29 | `29_ase_motif_validation.py` | ASE × motif causal validation |
| 30 | `30_tf_cooperativity.py` | TF cooperativity (enhanceosome vs billboard) |
| 31 | `31_flow_matching_design.py` | Conditional flow matching sequence design |
| 31b | `31_population_genetics.py` | Allele frequency & selection signatures |
| 32 | `32_hybrid_phenotype_similarity.py` | Cross-hybrid phenotype similarity |
| 33 | `33_gwas_overlap.py` | GWAS overlap analysis |
| 34 | `34_scatac_integration.py` | scATAC-seq chromatin integration |
| 35 | `35_network_topology.py` | Regulatory network topology |
| 36 | `36_bqtl_epistasis.py` | Additive vs epistatic bQTL effects |
| 37 | `37_te_context.py` | Transposable element context |
| 38 | `38_chromatin_context.py` | Chromatin context classification |
| 39 | `39_eqtl_proxy.py` | Binding→expression causal chain |

## Monitoring

Nextflow generates execution reports in `results/logs/`:

- **DAG** (`pipeline_dag.html`): directed acyclic graph of all processes
- **Timeline** (`pipeline_timeline.html`): Gantt chart of process execution
- **Report** (`pipeline_report.html`): resource usage per process
- **Trace** (`pipeline_trace.tsv`): machine-readable execution metrics

For real-time monitoring with [Seqera Platform](https://tower.nf):
```bash
export TOWER_ACCESS_TOKEN=<your_token>
nextflow run main.nf -with-tower
```

## Verification

```bash
python scripts/verify_pipeline.py          # basic checks
python scripts/verify_pipeline.py --verbose # with hashes
```

## Execution profiles

| Profile | Description |
|---------|-------------|
| `standard` | Local execution (default) |
| `slurm` | SLURM cluster |
| `pbs` | PBS/Torque cluster |
| `sge` | SGE/UGE cluster |
| `conda` | Auto-create conda env from `environment.yml` |
| `docker` | Docker container |
| `singularity` | Singularity container (HPC, no root) |
| `aws` | AWS Batch |

Combine profiles: `-profile slurm,singularity`

## Data

All data derive from Engelhorn et al. (2025), Nature Genetics (s41588-025-02246-7).

**Download (~3.4 GB):** [OneDrive shared folder](https://1drv.ms/f/c/6346f45e78d9bcf9/IgAsn1H2gq1DRKfj93mRTpiDAVPJG4lIeGd6gWDM9s2NNgY?e=oZLf0c)

See [DATA_MANIFEST.md](DATA_MANIFEST.md) for the complete list of required files, sizes, and sources.

## Citation

```
Banf, M. and Hartwig, T. (2026). A high-confidence causal bQTL atlas reveals
the regulatory architecture of natural gene expression variation in maize.
```

## License

MIT
