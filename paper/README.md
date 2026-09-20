# Condition-dependent bQTL switching paper

## Reproduction walkthrough

All analysis scripts are in:
```
bqtl_predict/results/celltype_landscape/quantitative_modulation_paper/
```

### Prerequisites

```bash
cd bqtl_predict
source .venv/bin/activate
```

Required packages: pandas, numpy, scipy, scikit-learn, matplotlib, pyfaidx, xgboost, torch, transformers, peft.
All installed in `.venv/`.

External tools: `bcftools` (for genotype extraction in script 53).

### Input data

| File | Source | Description |
|------|--------|-------------|
| `data/processed/bqtl_snp_ww.csv` | Engelhorn 2025, Table S6 | 147,942 bQTL positions |
| `data/raw/supp_table_MOESM5.xlsx` (S13a, S13b) | Engelhorn 2025 | Per-hybrid ASE (WW and drought) |
| `data/processed/engelhorn_ase_ww.csv` | Preprocessed from S13a | Per-hybrid signed ASE |
| `data/processed/engelhorn_ww_vs_ds_expression.tsv` | Engelhorn 2025, Table S7 | Gene-level WW vs drought expression |
| `data/raw/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3` | MaizeGDB | B73 NAM5 gene models |
| `data/raw/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1_entap_results.tsv.gz` | EntAP pipeline | Gene annotations (TF family IDs) |
| `data/processed/maize_tf_pwm_database.json` | PlantTFDB 5.0 | 259 TF motif PWMs (36 families) |
| `bound_region_motifs.csv` | Script 47 (previous pipeline) | Per-bQTL motif scan (388K rows) |
| `results/regulatory_network/regulatory_chains.csv` | Script 41 (previous pipeline) | TF→bQTL→target chains |
| `pan_cistrome_peaks.csv` | Previous pipeline | 291,878 peak coordinates |
| `gene_peak_disruption.csv` | Previous pipeline | Per-gene peak stats |

### Pipeline execution order

Scripts must run in order because each depends on outputs of earlier scripts.
Use `run_pipeline.sh` (below) or run individually:

```
53 → 54 → 55 → 56 → 57 → 58 → 59 → 60
```

#### Script 53: Download NAM founder genotypes
```bash
python 53_download_nam_genotypes.py
```
- **Duration**: ~4-6 hours (downloads ~100 GB of VCFs from CyVerse iRODS, extracts and deletes)
- **Requires**: `python-irodsclient`, `bcftools`, internet access to CyVerse (port 1247)
- **Output**: `data/processed/nam_founder_genotypes_at_bqtl.tsv` (43,900 variants × 24 founders)
- **Paper section**: Methods ("NAM founder genotypes")
- **Note**: Only needs to run once. If output exists, skip.

#### Script 54: Genotype→ASE causal test (well-watered)
```bash
python 54_genotype_aware_bqtl_test.py
```
- **Duration**: ~2-5 min
- **Input**: genotype matrix (script 53), ASE data, GFF3
- **Output**: `genotype_bqtl_results.csv`, `figures/fig11_genotype_bqtl_test.pdf`
- **Paper section**: "Variant genotype causally determines allele-specific expression"
- **Key numbers**: 8,040 testable pairs, 785 FDR<0.05, 5.2x enrichment, pi1=0.376, median |d|=0.855

#### Script 55: TF motif → genotype → ASE chain
```bash
python 55_motif_genotype_ase_chain.py
```
- **Duration**: ~1-2 min
- **Input**: genotype results (script 54), motif scan data
- **Output**: `motif_genotype_ase_chain.csv`, `figures/fig12_motif_genotype_ase.pdf`
- **Paper section**: "TF motif identity is irrelevant under baseline conditions"
- **Key numbers**: 92% have motifs at variant, |d| 0.830 vs 0.800 (p=0.25), CV=4.6% across 18 families

#### Script 56: Regulatory grammar of functional bQTL
```bash
python 56_functional_bqtl_grammar.py
```
- **Duration**: ~2-3 min
- **Input**: genotype results (script 54), motif data, regulatory chains, peaks
- **Output**: `functional_bqtl_grammar.csv`, `gene_level_grammar.csv`, `figures/fig13_regulatory_grammar.pdf`
- **Paper section**: "Regulatory grammar of functional bQTL"
- **Key numbers**: TSS distance 521 vs 649bp (p=1.2e-6), expression 698 vs 1000 CPM (p=0.007)

#### Script 57: Drought TF regulatory configuration
```bash
python 57_drought_tf_bqtl_configuration.py
```
- **Duration**: ~2-3 min
- **Input**: genotype results (script 54), motif data, expression data, EntAP annotations
- **Output**: `drought_tf_bqtl_config.csv`, `family_drought_ase_effects.csv`, `tf_family_expression_stats.csv`, `figures/fig14_drought_tf_configuration.pdf`
- **Paper section**: "The TF concentration landscape gates bQTL functionality" (WW baseline analysis)
- **Key numbers**: TF CV=0.580, NAC log2FC=+4.0, HSF +1.15, ERF +0.94

#### Script 58: Drought genotype→ASE test
```bash
python 58_drought_genotype_ase_test.py
```
- **Duration**: ~5-10 min (parses large Excel files)
- **Input**: genotype matrix (script 53), MOESM5 Excel (Tables S13a, S13b), GFF3
- **Output**: `ww_vs_drought_genotype_bqtl.csv`, `ww_vs_drought_tf_family_effects.csv`, `figures/fig15_ww_vs_drought_bqtl.pdf`
- **Paper section**: "Condition-dependent bQTL switching"
- **Key numbers**: WW 814 FDR hits, DS 742 FDR hits, 372 constitutive, 442 WW-only, 370 DS-only

#### Script 59: Regulatory configuration figure (saturation model)
```bash
python 59_regulatory_configuration_figure.py
```
- **Duration**: ~1-2 min
- **Input**: WW vs drought results (script 58), TF expression, EntAP
- **Output**: `tf_expression_vs_bqtl_effect.csv`, `figures/fig16_regulatory_configuration.pdf`
- **Paper section**: "The TF concentration landscape gates bQTL functionality" (saturation test)
- **Key numbers**: WW rho=-0.830 (p=0.0008), DS rho=-0.067 (p=0.84), NAC +5.25 log2FC

#### Script 60: Condition switching deep dive
```bash
python 60_condition_switching_deepdive.py
```
- **Duration**: ~2-3 min
- **Input**: WW vs drought results (script 58), functional grammar (script 56), expression data
- **Output**: `condition_switching_deepdive.csv`, `figures/fig17_switching_deepdive.pdf`
- **Paper section**: "Switching is a threshold effect", "Target gene drought response predicts switching"
- **Key numbers**: WW-only |d| drops 50%, DS-only gains 83%, gene log2FC +0.413 vs +0.219 (p=4.1e-4), NAC OR=1.30, TCP OR=1.51

### Output summary

All outputs in `results/celltype_landscape/quantitative_modulation_paper/`:

**Key CSV files:**
- `genotype_bqtl_results.csv` — 8,040 genotype→ASE tests (WW)
- `ww_vs_drought_genotype_bqtl.csv` — 16,173 paired WW+DS tests
- `condition_switching_deepdive.csv` — 7,888 paired bQTL classified by switching category
- `functional_bqtl_grammar.csv` — regulatory features at each bQTL-gene pair
- `motif_genotype_ase_chain.csv` — TF motif × genotype × ASE chain
- `tf_expression_vs_bqtl_effect.csv` — TF family expression vs bQTL effect

**Figures (in `figures/`):**
- `fig11_genotype_bqtl_test.pdf` → Paper Fig. 1
- `fig12_motif_genotype_ase.pdf` → Paper Fig. 2
- `fig13_regulatory_grammar.pdf` → Paper Fig. 3 (partial)
- `fig15_ww_vs_drought_bqtl.pdf` → Paper Fig. 3 (switching)
- `fig16_regulatory_configuration.pdf` → Paper Fig. 4 (saturation)
- `fig17_switching_deepdive.pdf` → Paper Fig. 3 (deep dive panels)

### Verifying numbers

Each script prints key statistics to stdout. To verify any number in the paper,
re-run the relevant script and check the printed output against the paper claims.
All statistical tests use scipy (Mann-Whitney U, Spearman correlation, Fisher's exact).
FDR correction uses Benjamini-Hochberg via `statsmodels.stats.multitest.multipletests`.
