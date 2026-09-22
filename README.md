# Causal bQTL Atlas

**A high-confidence causal bQTL atlas reveals the regulatory architecture of natural gene expression variation in maize**

Michael Banf and Thomas Hartwig

---

## Overview

This repository contains the analysis pipeline and interactive atlas for identifying 728 high-confidence causal binding quantitative trait loci (bQTL) from the maize pan-cistrome (Engelhorn et al. 2025, Nature Genetics).

Each causal bQTL passes three criteria:
1. It is the **only** significant bQTL in its target gene's promoter (no LD ambiguity)
2. It falls within an actual **MOA-seq binding peak** (actively bound chromatin)
3. Its genotype **causally predicts** allele-specific expression across 19 NAM hybrids (FDR < 0.05)

## Interactive Atlas

Browse all 728 causal bQTL at: **https://mbanf.github.io/causal-bqtl-atlas/**

## Quick Start

```bash
git clone https://github.com/mbanf/causal-bqtl-atlas.git
cd causal-bqtl-atlas

# 1. Set up Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy scipy matplotlib seaborn pyfaidx openpyxl

# 2. Download data (~3.4 GB) — see DATA_MANIFEST.md for sources
#    Symlink into the project:
ln -s /path/to/downloaded/raw data/raw
ln -s /path/to/downloaded/processed data/processed

# 3. Run the pipeline (~1.5 hours with pre-computed genotypes)
bash run_pipeline.sh --skip-download

# 4. Verify results match the paper
python scripts/verify_pipeline.py
```

## Pipeline

The pipeline has four phases (29 scripts total):

| Phase | Scripts | Description | Time |
|-------|---------|-------------|------|
| 1. Foundational | 44-52 | Motif scanning, peak analysis, binding features | ~45 min |
| 2. Genotype-causal | 53-63 | NAM genotypes, causal test, 728 selection | ~30 min |
| 3. 728 Atlas | 65-72 | Sole-copy, rewiring, cascades, drought | ~20 min |
| 4. Interactive | 67, update | HTML atlas generation | ~2 min |

### Key outputs

| File | Description |
|------|-------------|
| `results/causal_bqtl_728.csv` | 728 causal variants (core result) |
| `results/regulatory_map_728.csv` | Annotated with TF families, function, drought |
| `results/gene_atlas_728.csv` | Full gene characterization |
| `interactive/gene_atlas_728_interactive.html` | Interactive D3.js visualization |

### Options

```bash
bash run_pipeline.sh --skip-download    # Skip 4-6h genotype extraction (use pre-computed)
bash run_pipeline.sh --paper-only       # Skip exploratory scripts (faster)
```

## Data

All data derive from Engelhorn et al. (2025): SRA accession PRJNA1101486, GEO GSE294039.

See [DATA_MANIFEST.md](DATA_MANIFEST.md) for the complete list of required files, sizes, download URLs, and hosting instructions.

**Total data size: ~3.4 GB** (2.1 GB genome + 923 MB peaks + 300 MB processed files)

## Verification

After running the pipeline, verify reproducibility:

```bash
python scripts/verify_pipeline.py
```

This checks:
- 13 result files (row counts, column counts, gene set hashes)
- 6 paper statistics (1,403 significant pairs, 728 final set, median |d|=2.4, 233 rewired, etc.)
- Interactive atlas presence

## Citation

```
Banf, M. and Hartwig, T. (2026). A high-confidence causal bQTL atlas reveals
the regulatory architecture of natural gene expression variation in maize.
```

## License

MIT
