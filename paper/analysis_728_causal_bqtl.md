# Analysis: 728 High-Confidence Causal bQTL

## Filtering Pipeline

| Step | Filter | N remaining | Drop |
|------|--------|-------------|------|
| 1 | All MOA binding peaks per hybrid | ~237,000/hybrid | — |
| 2 | bQTL positions (allelic binding differences) | 147,942 | Peaks without allelic variation |
| 3 | bQTL within 2kb of gene TSS + ASE data available | 19,368 pairs | bQTL far from genes or missing ASE |
| 4 | Genotype→ASE causal test FDR<0.05 | 1,403 pairs | Non-significant: either too weak or underpowered (N=19 hybrids) |
| 5 | Single bQTL per gene (avoid LD confounding) | 772 genes | 257 multi-bQTL genes (25%) where causal variant is ambiguous |
| 6 | bQTL falls within a MOA peak | 728 genes | 44 bQTL not in any hybrid's peak file |

**Interpretation**: These 728 represent cases where (a) a specific DNA variant (b) disrupts
TF binding at a single site, (c) is the only significant variant near that gene, and (d) causally
shifts allele-specific expression across hybrids. This is the highest-confidence set for
mechanistic interpretation.

**Validation**: The 4.85x enrichment over chance (1,403 observed vs ~290 expected under null)
confirms the genotype→ASE signal is real, not noise. The 7.2% hit rate is a lower bound on
the true functional fraction — many real effects are too small to detect with N=19 hybrids.


## Linkage and the Single-bQTL Design

75% of significant genes have exactly 1 bQTL — no linkage concern. For the 25% with multiple
bQTL, 62% of within-gene pairs have identical genotype vectors across 19 founders (complete LD).
The single-bQTL filter removes this ambiguity at the cost of excluding some real signals.


## Motif Redundancy: Why One Variant Matters

Scanning with 259 PWMs (36 TF families) at different window sizes:

| Window | Size | Total motif hits | Disrupted motif is sole copy |
|--------|------|-----------------|------------------------------|
| Full 2kb promoter | 2,000bp | ~220 | 4.5% |
| Merged union peaks | ~614bp | 269 | 30.4% |
| Individual hybrid peak | ~160bp | 68 | **68.3%** |

**Key finding**: Within the actual ~160bp binding peak, the disrupted TF family's motif is
the ONLY copy in 68% of cases. The other ~67 motif hits belong to different families.
There is no redundancy for that specific family within the bound region.

This explains why a single nucleotide change can matter: it eliminates the only binding
site for a specific TF family within the accessible chromatin window. The ~150 other
motifs in the 2kb promoter are outside the open chromatin and not actually bound.


## Co-occurrence in Binding Peaks

Most TF family pairs co-occur at expected rates (lift ~1.0). Non-random combinations:

| Pair | Lift | Known biology |
|------|------|---------------|
| BBR-BPC + GRAS | 1.90 | Novel |
| BES1 + CAMTA | 1.80 | Both signaling-responsive |
| BES1 + bHLH | 1.76 | Known heterodimer (BES1-PIF4) |
| BES1 + bZIP | 1.63 | Known cooperative binding |
| HD-ZIP + YABBY | 1.59 | Both leaf polarity |

BES1 appears in 5/20 top non-random pairs, consistent with its role as a
brassinosteroid signaling hub that cooperates with multiple TF partners.


## Variant Distribution Across Hybrids

- Median 7 hybrids carry each variant (out of 19 genotyped)
- Range: 3-16 (no variant reaches all 19)
- Distribution peaks at 5-6 hybrids (intermediate frequency)
- Each hybrid carries 264-308 causal variants (36-42%) — remarkably uniform
- Weak negative correlation: rare variants have slightly larger effects
  (rho=-0.11, p=0.003), consistent with purifying selection


## Functional Characterization

### Target gene categories (728 genes)
- Kinase/Phosphorylation: 76 (10.4%)
- Chaperone/HSP: 63 (8.7%)
- Biosynthesis (enzymes): 39 (5.4%)
- Transcription factors: 32 (4.4%) — including MYB30, WRKY50/51, NAC67/82, bHLH68/128
- Translation: 29 (4.0%)
- Signal transduction: 26 (3.6%)
- Transporters: 17 (2.3%)
- Redox/Metabolism: 17 (2.3%)
- Chromatin/Epigenetic: 14 (1.9%)
- Hormone signaling: 6 (0.8%) — auxin (IAA5, ARF7), ABA (PYR1)
- Cell wall: 6 (0.8%)

### Enriched pathways (Fisher's exact vs genome background)
- Pentose phosphate pathway (OR=3.37, p=0.012)
- Steroid biosynthesis (OR=3.60, p=0.016)
- Amino sugar/nucleotide sugar metabolism (OR=2.11, p=0.021)
- Ubiquitin-mediated proteolysis (OR=2.14, p=0.015)
- Pantothenate/CoA biosynthesis (OR=4.32, p=0.018)

### Enriched GO terms
- Pentose metabolic process (OR=19.2, p=1.6e-4)
- Plant-type cell wall biogenesis (OR=6.4, p=0.005)
- Regulation of hormone biosynthesis (OR=5.1, p=0.011)
- Aromatic amino acid biosynthesis (OR=5.4, p=0.009)
- Phosphorus metabolic process (54 genes, OR=1.45, p=0.008)

### Metabolic gene cluster cross-reference
- BX (benzoxazinoid): only Bx9 (chr10, d=3.54) in the 728 set
- Terpene synthase genes: not tested (no bQTL in promoters)
- Flavonoid pathway: 1 gene (flavonoid 3'-monooxygenase, d=2.48)
- No large metabolic gene clusters enriched

**Interpretation**: Causal bQTL predominantly target the REGULATORY LAYER — signaling
kinases, TFs, transporters — not metabolic enzymes directly. The variant doesn't change
an enzyme; it modulates how much a regulatory gene is expressed from one allele, which
cascades to downstream metabolic effects.


## Disrupted TF Families → Target Gene Regulatory Map

Top disrupted families and their target gene profiles:

| TF Family | N targets | Top target functions |
|-----------|-----------|---------------------|
| MYB | 96 | Kinases (5), TFs (2), Transporters (2) |
| C2H2 | 91 | Kinases (9), TFs (4) |
| NAC | 68 | Kinases (4), TFs (2) |
| MIKC_MADS | 66 | Kinases (6), TFs (4) |
| ERF | 66 | Kinases (5), TFs (4), Transporters (2) |
| TALE | 60 | Kinases (6), TFs (5) |
| Dof | 53 | Kinases (6), TFs (3) |
| Trihelix | 50 | Kinases (4), TFs (2), Transporters (2) |

All TF families target similar functional profiles (dominated by signaling genes).
This is consistent with the earlier finding that TF identity is irrelevant for
effect size (CV=5.5%) — the gate is chromatin accessibility, not which TF is disrupted.


## Notable Individual Genes

### Highest effect sizes
- Zm00001eb047770 (chr1, dioxygenase): |d|=7.18, 9 hybrids
- Zm00001eb352310 (chr8, H3K9 methyltransferase): |d|=5.15, 4 hybrids
- Zm00001eb105750 (chr2, SH3 domain protein): |d|=3.98, 3 hybrids
- Zm00001eb398660 (Bx9, UDP-glucosyltransferase): |d|=3.54, benzoxazinoid pathway

### Hormone signaling targets
- Zm00001eb360250: Auxin-responsive IAA5 (chr8, |d|=2.11)
- Zm00001eb182260: Auxin response factor ARF7 (chr4, |d|=2.25)
- Zm00001eb390480: ABA receptor PYR1 (chr9, |d|=2.44)

### Physical cluster: chr8:154-155Mb
- IAA5 (auxin-responsive) + ATAF2 (NAC TF) + ribosomal protein + unknown
- 4 causal genes in ~460kb — auxin/stress signaling hub


## Drought Response

- 16.5% of target genes are drought-upregulated (log2FC > 1)
- 10.7% are drought-downregulated (log2FC < -1)
- Mean log2FC = +0.206 (slight drought induction)
- Hormone signaling genes show the widest drought response range


## CRISPR/Design Implications

Each of the 728 bQTL represents a single nucleotide position where:
1. The variant disrupts the sole copy of a TF family's motif within the binding peak (68%)
2. Changing this nucleotide causally shifts allelic expression
3. The effect is shared across multiple hybrids (median 7)

This makes them candidate CRISPR targets for tuning gene expression:
- Introduce the variant → reduce TF binding → reduce expression from that allele
- Restore the reference → restore TF binding → restore expression
- The predicted effect size (Cohen's d) provides a quantitative expectation

The MaizeDesigner tool (biodesign_agent.py) can already design sequences at these
positions and predict the bQTL probability change using the LoRA model.


## Figures

- fig_causal_bqtl_hybrid_distribution.pdf — 4-panel: variant frequency, effect vs frequency, per-hybrid counts, cumulative
- fig_regulatory_map_728.pdf — 4-panel: TF×function heatmap, sharing by function, effect by TF, drought by function
- fig_regulatory_network_map.pdf — 3-layer network: TF family → gene function → hybrid sharing
