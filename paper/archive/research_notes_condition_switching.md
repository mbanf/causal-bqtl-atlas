# Research Notes: Condition-Dependent bQTL Switching

## 1. What input data do we have?

| Data layer | Source | Dimensions | What it tells us |
|------------|--------|------------|------------------|
| **bQTL positions** | Engelhorn 2025, Table S6 | 147,942 variants | Where in the genome TF binding differs between hybrids |
| **Genotypes at bQTL** | Assembly alignment (script 53b) | 147,323 positions x 19 founders | Which NAM founder carries variant vs reference at each bQTL |
| **Allele-specific expression (ASE)** | Engelhorn 2025, Tables S13a/S13b | ~25,000 genes x 25 hybrids x 2 conditions (WW + drought) | Per-hybrid, per-condition allelic imbalance (log2 B73/NAM) |
| **Gene expression** | Engelhorn 2025, Table S7 | 24,796 genes, WW vs drought | Bulk expression level and drought fold-change per gene |
| **TF motif scan** | PlantTFDB 5.0, 259 PWMs, 36 families | Scanned at +/-28bp around each bQTL | Which TF family's motif is disrupted by each variant |
| **TF gene identities** | PlantTFDB + EntAP annotations | ~1,100 TF genes across ~30 families | Which genes encode TFs, and which family they belong to |
| **bQTL = binding events** | MOA cistrome (Engelhorn) | Per hybrid, per condition | bQTL are defined as positions where binding DIFFERS between hybrids — binding itself is measured |

### Critical design feature: ASE as within-individual control

Each F1 hybrid has TWO alleles in the SAME nucleus. Both alleles see the same TF concentrations, the same chromatin environment, the same signaling state. Any difference in allele-specific expression is therefore **purely cis-encoded** — it can only come from the sequence variant itself and how the local regulatory context reads it.

This means: when we split hybrids by genotype at a bQTL and compare ASE, we're measuring the **cis-effect of the variant**, controlled for all trans factors.


## 2. What can we learn by combining these data?

### Chain 1: Variant genotype -> ASE (causal test)

**Combination**: genotypes + ASE + gene positions

For each bQTL near a gene: split 19 hybrids by who carries the variant -> compare their ASE values. If significant, the variant causally determines allelic expression.

**Result**: 1,453 FDR<0.05 pairs (WW), 1,089 (drought), 4.8x enrichment over chance.

### Chain 2: Which TF motif is disrupted?

**Combination**: bQTL positions + motif scan

For each bQTL, identify which TF family's binding motif overlaps the variant. This tells us WHOSE binding site is disrupted.

**Result**: 92% of testable bQTL disrupt at least one TF family motif.

### Chain 3: Does disrupted TF identity predict effect size?

**Combination**: chain 1 + chain 2

Group bQTL by which TF family is disrupted, compare mean effect sizes.

**Result**: NO. All families produce nearly identical effects (CV=5.5%). Under baseline, it doesn't matter WHICH TF's site you disrupt — just THAT you disrupt one.

### Chain 4: What DOES predict functional bQTL?

**Combination**: chain 1 + gene positions + expression

Compare functional (FDR<0.05) vs non-functional bQTL across features.

**Result**:
- Distance to TSS: 510bp (functional) vs 626bp (non-functional), p=3.9e-11
- Target gene expression: 803 CPM vs 968 CPM, p=1.9e-7
- NOT: TF identity, motif count, redundancy, regulatory chain presence

### Chain 5: Condition-dependent switching

**Combination**: chain 1 under WW + chain 1 under drought (same variants, same genotypes)

Run the causal test under both conditions, compare which bQTL are functional.

**Result**: 74% of functional bQTL are condition-specific (Jaccard=0.262). Only 26% are constitutive. The same variant, in the same genotype, switches between functional and silent depending on condition.

### Chain 6: What predicts switching?

**Combination**: chain 5 + gene expression changes

Compare drought response of target genes across switching categories.

**Result**: Drought-specific bQTL target more drought-responsive genes (median log2FC=+0.542 vs +0.239 for WW-only, p=6.4e-13). The gene's transcriptional response predicts whether its bQTL switch on or off.


## 3. What the saturation model claimed (and why it collapsed)

### The original argument

**Combination**: chain 3 + TF gene expression

For each TF family, compute mean mRNA expression (proxy for protein concentration). Correlate with mean bQTL effect size per family.

**Old result (29.7% genotype coverage)**: rho=-0.830, p=0.0008. Interpretation: abundant TFs saturate both alleles, masking allelic differences. Scarce TFs only occupy the strong allele, creating large effects.

**New result (99.6% coverage)**: rho=-0.203, p=0.53. The correlation dissolves.

### Why it collapsed

With 3.4x more data per family, family-level mean |d| estimates converge (all ~0.87-1.07, CV=5.5%). The old correlation was driven by sampling noise in which specific bQTL had VCF genotypes. With comprehensive coverage, there's no variance left for expression to correlate with.

### What this means

TF mRNA expression (our proxy for protein concentration) does NOT predict which bQTL are functional at the family level. The "TF concentration landscape as gatekeeper" framing is not supported at genome-wide scale.


## 4. The revised model: chromatin accessibility as the gate

### What IS supported

The data consistently point to **locus-level regulatory context** as the gate:

1. **TSS distance gradient** (p=1.2e-17): proximal variants are always functional (constitutive 214bp), distal variants need additional context (non-functional 697bp). This is a chromatin accessibility signature — promoter-proximal regions are constitutively open.

2. **Target gene expression** (p=1.9e-7): genes with lower expression have more functional bQTL. Consistent with: moderately transcribed genes have intermediate accessibility where allelic differences matter; highly transcribed genes have fully open chromatin where both alleles are saturated.

3. **Gene drought response predicts switching** (p=6.4e-13): DS-specific bQTL target genes whose expression changes under drought (median log2FC=+0.542). The gene's expression change reflects chromatin reorganization at that locus.

4. **TF identity is irrelevant** (CV=5.5%): if the gate were TF-specific (concentration), you'd expect family-level differences. The uniformity means the gate is upstream of TF identity — it's about whether the site is accessible at all, not which TF tries to bind.

5. **Dimmer switch, not binary** (direction consistency 91-99%): WW-only bQTL don't lose their effect under drought — it decreases by 48%. DS-only bQTL don't appear from nothing — they increase by 104%. The variant always does the same thing; what changes is the magnitude. This is consistent with graded chromatin accessibility changes.

### The revised framework

The gate is **effective site availability**, which depends on:

```
Functional effect = f(sequence variant) x g(chromatin accessibility) x h(transcriptional activity)
```

- `f(sequence variant)`: fixed property of the DNA — does the variant disrupt a motif? How much affinity change?
- `g(chromatin accessibility)`: condition-dependent — is the region open? This changes under drought.
- `h(transcriptional activity)`: condition-dependent — is the gene being actively transcribed? Polymerase passage maintains open chromatin.

Under WW baseline, most regulatory regions have similar accessibility -> all TF families see similar effective availability -> uniform effects (CV=5.5%).

Under drought, accessibility is reorganized at specific loci -> different bQTL cross the detection threshold -> switching.

This is STILL thermodynamic: binding = [TF] x [accessible sites] x K_d(variant). But the rate-limiting variable is [accessible sites], not [TF].


## 5. Open questions: what more can we learn from existing data?

### Question A: Is TF expression response consistent across hybrids?

We have per-hybrid expression under WW and drought. For each TF family, we can ask:
- Do all 25 hybrids upregulate NAC under drought, or do some not respond?
- Is the cross-hybrid CV of TF drought response similar to non-TF genes?
- Do hybrids with stronger TF drought response show different bQTL switching patterns?

**What this tests**: whether the TF concentration landscape varies meaningfully across hybrids (it might not — drought response may be conserved).

**Data needed**: per-hybrid, per-gene expression under WW and DS (we have this from Tables S13a/S13b raw counts, and from Table S7 for bulk).

**Current evidence**: Script 57 showed TF expression CV across hybrids = 0.580 (similar to non-TF CV = 0.582). NAC has the highest hybrid CV (0.784). This suggests TF expression varies across hybrids about as much as any gene — there IS cross-hybrid variation to exploit.

### Question B: Does per-hybrid TF expression predict per-hybrid ASE?

This is the INDIVIDUAL-LEVEL version of the saturation test (instead of family-level ecological correlation). For each bQTL-gene pair where we know the disrupted TF family:
- Get TF family expression in each hybrid
- Get ASE in each hybrid
- Test: do hybrids with higher TF expression show LESS allelic effect (|ASE|)?

**What this tests**: the thermodynamic saturation prediction at the individual hybrid level. This has N=19 per bQTL (not N=12 families), so it's better powered for detecting real effects.

**Status**: Script 61 (TF saturation model) tested a version of this and found NO significant interaction for any family (all FDR>0.75, global Wilcoxon p=0.99, mean beta_interaction=-0.003). This is consistent with the saturation model failing — individual-level TF expression variation doesn't modulate the bQTL effect.

### Question C: Does binding itself change between WW and drought?

The bQTL are defined from WW binding data. But Engelhorn also collected drought data. Key question: do the same positions show binding under both conditions, or does binding change?

**If binding changes**: this directly measures chromatin accessibility changes. A bQTL where binding disappears under drought = site closed = bQTL silenced. A position that gains binding under drought = site opened = new bQTL.

**What we'd need**: drought-condition binding data (MOA peaks under drought). This may exist in the Engelhorn supplementary but we haven't extracted it.

**Why it matters**: this would directly test the chromatin gate hypothesis instead of inferring it from ASE patterns.

### Question D: Hybrid-specific switching patterns

Do certain hybrids drive the switching? If hybrid X has a particularly strong drought response for TF family Y, do bQTL disrupting family Y motifs switch more in hybrid X than in other hybrids?

**What this tests**: whether switching is genotype-specific (some hybrids switch more than others) or condition-universal (all hybrids switch similarly).

**Current evidence**: Script 60 showed mean |delta n_variant| ~ 0.09-0.10 between conditions (the number of variant-carrying hybrids barely changes). This confirms switching is NOT driven by genotype composition changes — it's driven by the regulatory environment changing.


## 6. KEY DISCOVERY: Binding itself massively reorganizes under drought

### Available data (already in MOESM5)

- **Table S6**: WW bQTL positions (147,942) — what we've been using
- **Table S10**: DS bQTL positions (124,504) — drought-specific binding variants!
- **Table S9**: Per-hybrid differential binding peaks (WW vs DS)
- **Table S1**: Total peak counts per hybrid per condition

### Binding-level switching (direct evidence for chromatin gate)

| Metric | Value |
|--------|-------|
| WW bQTL | 147,942 |
| DS bQTL | 124,504 |
| Shared | 38,614 (16.5%) |
| WW-only binding | 109,328 |
| DS-only binding | 85,890 |
| **Jaccard (binding)** | **0.165** |

This is even more extreme than ASE switching (Jaccard=0.262). 83.5% of binding variants are condition-specific. The binding sites themselves reorganize under drought — this is DIRECT measurement of chromatin accessibility changes, not an inference from downstream expression.

### Hybrid-specific chromatin remodeling (Table S9)

Different hybrids reorganize binding dramatically differently:

| Hybrid | Peaks higher in WW | Peaks higher in DS | Pattern |
|--------|--------------------|--------------------|---------|
| IL14H | 90,556 | 12,838 | Massive WW-dominant loss |
| B97 | 66,484 | 15,972 | WW-dominant |
| CML247 | 37,867 | 40,895 | Balanced |
| Oh7b | 21,146 | 29,765 | DS-dominant gain |
| M162W | 25,426 | 36,717 | DS-dominant gain |
| NC358 | 26,423 | 33,968 | DS-dominant gain |

This is crucial: chromatin remodeling is GENOTYPE-DEPENDENT. Different genetic backgrounds reorganize their binding landscapes differently under the same stress. This creates genotype-specific windows of vulnerability — different variants become functional in different hybrids under drought.

### Testable predictions (new analyses)

1. **ASE-binding concordance**: Do bQTL that switch ASE (WW-only or DS-only from script 58) sit in regions where binding changes? DS-only ASE bQTL should overlap with DS-only or DS-gained binding positions.

2. **Hybrid-specific binding x ASE**: Do hybrids with more differential binding (e.g., IL14H with 90K WW-lost peaks) show more ASE switching than hybrids with balanced binding (e.g., CML247)?

3. **Binding change predicts ASE change**: For individual bQTL, does loss of binding under drought (Table S9) predict loss of ASE significance, and vice versa?

4. **The causal chain closes**: variant genotype -> binding change (WW vs DS) -> ASE change (WW vs DS) -> gene expression change (drought response). All three layers now have condition-dependent data.


### Script 62 results: ASE-binding concordance (tested)

**Setup**: Cross-tabulated ASE switching categories (from script 58) with binding status at each bQTL (WW bQTL from Table S6, DS bQTL from Table S10).

**Binding-level switching** (revised numbers using Table S10):
- WW bQTL: 147,942; DS bQTL: 147,896
- Shared: 38,686 (15.0%); Jaccard = 0.150
- WW-only binding: 109,256; DS-only binding: 109,210

**Key results**:

| Finding | Value | p-value |
|---------|-------|---------|
| Overall association (chi-squared) | chi2=36.38 | p=6.2e-8 |
| Constitutive ASE at constitutive binding | OR=1.444 | p=1.1e-4 |
| Constitutive binding → larger WW effect | 0.986 vs 0.921 | p=1.2e-3 |
| Binding switches MORE than ASE | Jaccard 0.150 < 0.262 | — |

**Interpretation**: bQTL with stable binding (both conditions) are significantly enriched for stable ASE effects and produce larger allelic effects. The fact that binding switches more than ASE (Jaccard 0.150 vs 0.262) is consistent with binding being the upstream cause and ASE being a buffered downstream readout — some binding changes don't propagate to detectable expression changes.

**Important limitation**: Since all ASE tests start from the WW bQTL list (147,942), DS-only binding positions (109,210) are invisible — they have no ASE test paired to them. This means we CANNOT test the prediction that DS-only ASE maps to DS-only binding sites. A future analysis starting from DS bQTL (Table S10) would be needed to close that gap.

**Hybrid binding remodeling** (all 25 hybrids from Table S9):
- IL14H: most extreme WW-dominant (90K WW > 13K DS, ratio 0.12)
- M37W: most DS-dominant (27K WW < 40K DS, ratio 0.60)
- Mean DS fraction: 0.397 (overall WW-dominant remodeling)
- 16 hybrids WW-dominant, 8 balanced, 1 DS-dominant


## 7. Summary: what the paper should say

### Core findings (robust, supported by full data)

1. Variant genotype causally determines ASE (1,453 FDR hits, 4.8x enrichment)
2. TF motif identity is irrelevant under baseline (CV=5.5%)
3. Functional bQTL are distinguished by TSS proximity and target gene expression, not TF identity
4. 74% of functional bQTL are condition-specific (Jaccard=0.262)
5. Switching is continuous (dimmer, not binary switch) with 91-99% direction consistency
6. Target gene drought response predicts switching direction (p=6.4e-13)

### Revised mechanistic interpretation

The gate is chromatin/regulatory context, not TF concentration. The condition-dependent switching reflects reorganization of the chromatin accessibility landscape under drought, which changes which binding sites are available for TF occupancy — and therefore which sequence variants produce detectable allelic effects.

### What to drop

- The TF expression vs bQTL effect correlation (rho=-0.830) — this was a sampling artifact
- The "TF concentration landscape as gatekeeper" framing
- The specific saturation model predictions about rare vs abundant TFs

### What to reframe

- The switching mechanism: from "TF concentration changes" to "regulatory context reorganization"
- The title: needs to reflect condition-dependent switching as the main finding, with chromatin/context as the mechanism
- The thermodynamic framework can still be mentioned as background theory, but the data show the dominant variable is site accessibility, not TF concentration
