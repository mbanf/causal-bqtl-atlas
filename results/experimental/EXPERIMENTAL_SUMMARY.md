# Experimental Analyses — Thomas's Comments

Analyses addressing reviewer/collaborator comments on the causal bQTL atlas manuscript.
Branch: `experimental`

---

## Experiment 1: Multi-bQTL gene recovery via sole-copy motif (L62/63)

**Question:** For genes with >1 significant bQTL, can we identify the causal one by checking if only one disrupts a sole-copy (non-redundant) motif?

**Result:** 57 of 257 multi-bQTL genes (22%) are recoverable.

| Strategy | Genes recovered | % |
|----------|:-:|:-:|
| Only 1 bQTL in MOA-seq peak | 14 | 5.4% |
| Only 1 bQTL with sole-copy motif | 61 | 23.7% |
| Only 1 with sole-copy + in peak | 57 | 22.2% |
| Union (any strategy) | 71 | 27.6% |

- Recovered bQTL have strong effects: median |d|=2.88 (vs 2.41 for the 728)
- Top sole-copy families at recovered sites: Nin-like, BES1, ARR-B, CAMTA
- These remain "educated guesses" (Thomas's term) — lower confidence than the 728

**Scripts:** `exp_01_multi_bqtl_sole_motif.py`
**Output:** `multi_bqtl_sole_motif_recovery.csv`, `recovered_bqtl_candidates.csv`

---

## Experiment 2: TF family conservation at bQTL vs non-bQTL peaks (L176/177)

**Question:** The 728 causal bQTL disrupt 2-5 TF families. Is this unusual compared to other peaks?

**Result:** No — 2-5 disrupted families is typical for any bQTL position.

| Metric | Causal 728 | Other bQTL | p-value |
|--------|:----------:|:----------:|:-------:|
| Families disrupted at variant | median 2 | median 2 | 0.11 (n.s.) |
| Families in surrounding peak | median 27 | median 27 | 0.06 (n.s.) |
| Motif density (motifs/bp) | 0.455 | 0.415 | 5.6e-22 |
| Peak width | 682 bp | 619 bp | — |

- Non-bQTL peaks are much smaller (median 166 bp) — essentially minor binding sites
- What distinguishes causal bQTL is not motif architecture but the genotype→ASE causal link
- The motif diversity is a property of regulatory DNA in general, not specific to causal variants

**Scripts:** `exp_02_peak_motif_conservation.py`
**Output:** `peak_motif_conservation.csv`, `peak_motif_conservation.pdf`

---

## Experiment 3: Drought allele specificity (L202 / drought section)

**Question:** Which allele (B73 or NAM) changes expression under drought? Can we differentiate allele-specific drought responses?

**Paper text fix:** Added introductory paragraph to drought section (L546) explaining the WW/DS experimental design.

**Result:** 46% of causal bQTL show allele-specific drought response.

### Overall ASE shift under drought
- Median |d_shift| = 0.90 (substantial genotype × environment interaction)
- 71% of bQTL show |d_shift| > 0.5
- 46% show |d_shift| > 1.0

### Which allele responds? (cross-tabulation)

| Drought response | B73 drives | Equal | NAM drives | Total |
|-----------------|:----------:|:-----:|:----------:|:-----:|
| Upregulated | **56** | 49 | 24 | 129 |
| Downregulated | 13 | 35 | **32** | 80 |
| Stable | 98 | 321 | 128 | 547 |

**Key asymmetry:**
- Drought-**upregulated** genes: B73 allele drives induction (56 vs 24, 2.3× ratio)
- Drought-**downregulated** genes: NAM allele is preferentially repressed (32 vs 13, 2.5× ratio)
- This suggests B73 has stronger drought-inducible cis-regulatory elements, while NAM founders carry variants that make certain alleles more susceptible to drought-mediated repression

### Correlation
- ASE shift in variant hybrids correlates with total drought FC (Spearman rho=0.40, p=1.5e-9)
- Reference hybrids also show this (rho=0.39), confirming trans effects alongside cis × environment

### Notable examples
- **Zm00001eb157820** (drought UP, FC=+4.18): B73 allele drives — variant hybrids ASE shift +3.59 vs reference +1.60
- **Zm00001eb351960** (drought DOWN, FC=-1.23): NAM allele repressed — variant hybrids ASE shift -2.36 vs reference -0.05
- **Zm00001eb370490** (stable total, FC=-0.71): Dramatic allele-specific response hidden by total expression — variant hybrids ASE shift -2.56 vs reference +0.07

**Scripts:** `exp_03_drought_allele_specificity.py`
**Output:** `drought_allele_specificity.csv`, `drought_allele_specificity.pdf`
