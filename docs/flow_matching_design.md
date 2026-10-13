# Conditional Flow Matching for Regulatory Sequence Design

## Motivation

Steps 26-30 of the Causal bQTL Atlas established that:

- bQTL disrupt specific TF motifs (C2H2, NAC, MYB dominate; 33 families from 405 DAP-seq PWMs)
- Disruption causes measurable allele-specific expression (p=2e-242, Cohen's d=0.37)
- TF pairs co-bind cooperatively (enhanceosome at promoter level) but contribute additively to expression (billboard at variant level)
- 81% of causal bQTL are condition-specific (WW vs DS)
- Motif conservation across 23 hybrids validates functional importance (98.9% highly conserved)
- ALL 30 TF families show significant disruption-to-ASE effects

This means we can learn a GENERATIVE MODEL that maps desired TF binding changes to minimal sequence edits. The question "increase WRKY binding while maintaining NAC binding under drought" becomes a conditioning input, not a post-hoc filter.

## Why Flow Matching

| Property | Flow Matching | Diffusion | MLM (MaizeDesigner) |
|---|---|---|---|
| Trajectory | Deterministic ODE | Stochastic SDE | Discrete sampling |
| Edit distance | Naturally minimal (OT) | Noisy, needs many steps | Random, rejection-sampled |
| Conditioning | First-class input | Classifier guidance | Emergent from scoring |
| Steps needed | ~50 Euler steps | ~1000 denoising steps | ~10 mask-fill rounds |
| Directionality | Learns ref->alt grammar | Learns noise->signal | No directional grammar |

Flow matching (Lipman et al. 2023) learns continuous interpolation paths via optimal transport. The velocity field v_theta(x_t, t, c) tells us HOW to edit, not just WHAT to edit toward.

## Architecture

```
                    +-----------------------------+
                    |    Conditioning Vector c     |
                    |  [TF_target, ASE_target,     |
                    |   hybrid_context, condition] |
                    +-------------+---------------+
                                  |
                    +-------------v---------------+
     x_0 -------->  |  Flow Matching Network       |  --------> x_1
  (source seq)      |  1D U-Net, 7.9M params       |        (edited seq)
                    |  v_theta(x_t, t, c)          |
                    +-------------+---------------+
                                  |
                    +-------------v---------------+
                    |  Scoring / Validation         |
                    |  - Agro-NT LoRA (bQTL score)  |
                    |  - CNN IG (attribution)       |
                    |  - PWM scan (motif check)     |
                    |  - ASE predictor (expression) |
                    +-------------------------------+
```

### Flow Matching Network (1D U-Net)

- Input: x_t in R^{4 x 1001} (interpolated one-hot sequence at time t)
- Condition: c in R^73 (35 TF disrupt + 35 TF create + ASE + WW/DS + conservation)
- Output: v_theta in R^{4 x 1001} (velocity field)
- Encoder: 3 levels of ConvBlocks (kernel=9) with downsampling, base channels=64
- Bottleneck: ConvBlock at lowest resolution (256 channels)
- Decoder: 3 levels with skip connections
- Time embedding: sinusoidal (64-dim)
- Condition embedding: MLP (73 -> 128 -> 64)
- Total parameters: ~7.9M

### Conditioning Vector (73-dim)

| Dimensions | Content | Source |
|---|---|---|
| 0-34 | TF families to disrupt (binary) | User specification / bQTL annotation |
| 35-69 | TF families to create (binary) | User specification / bQTL annotation |
| 70 | Target ASE log2FC (continuous) | Measured from hybrid expression |
| 71 | Condition: 0=WW, 1=DS | Experimental condition |
| 72 | Conservation score (0-1) | Population motif occupancy (Layer 1) |

## Training

### Data

- 147,942 bQTL positions with ref/alt alleles and TF motif annotations
- x 2 bidirectional (ref->alt AND alt->ref) = 295,884 edit pairs
- 2,342 causal bQTL with measured ASE effects across 23 hybrids
- Sequence: 1001bp centered on variant, one-hot encoded (4 x 1001)

### Loss Function

Optimal Transport Conditional Flow Matching:

```
x_t = (1 - t) * x_0 + t * x_1       # OT interpolation
u_t = x_1 - x_0                       # target velocity
L = E_t ||v_theta(x_t, t, c) - u_t||^2   # MSE loss
```

Key property: the OT path is the SHORTEST path between source and target, which naturally encourages minimal edits.

### Training Estimate

- ~295K pairs x 100 epochs = ~30M forward passes
- GPU (A100): ~4-6 hours
- Apple MPS: ~12-18 hours
- Small enough for rapid iteration

## Inference: LoRA-Guided ODE Integration

At inference time, we integrate the learned ODE:

```
x_{t+dt} = x_t + dt * v_theta(x_t, t, c)    # Euler step
```

with optional Agro-NT LoRA gradient guidance:

```
v_guided = v_theta(x_t, t, c) + w * grad_x[LoRA_score(x_t)]
```

- LoRA guidance applied only after t > 0.3 (first 30% unguided for stable trajectory)
- LoRA (AUC 0.865) provides quality signal for what makes good regulatory edits
- 50 Euler steps sufficient (flow matching paths are straighter than diffusion)

### Two Roles

1. **Flow model** learns the EDITING GRAMMAR from 147K natural bQTL variants
2. **LoRA model** provides the QUALITY SIGNAL for regulatory function

This separation means the flow model can generate edits that follow natural variation patterns, while LoRA ensures the results are functionally meaningful.

## ASE Prediction Head

Separate network predicting allele-specific expression from sequence pairs:

```
f(ref_seq, alt_seq, hybrid_embedding) -> log2(alt/ref) ASE
```

- Sequence difference encoder: Conv1D (4 -> 32 -> 64) + global pooling
- 23 learnable hybrid embeddings (16-dim each)
- Prediction head: MLP (80 -> 64 -> 1)

### Hybrid Generalization

For new (unseen) F1 hybrids:

1. **Mean embedding**: Use average of all 23 training hybrid embeddings (zero-shot)
2. **Nearest hybrid**: Match by NAM parent genome k-mer similarity
3. **Few-shot calibration**: Optimize a new embedding from a handful of measured ASE values

The flow model itself needs NO hybrid-specific features — TF binding grammar is conserved. The hybrid embedding only matters for ASE prediction.

## CRISPR Integration

The framework filters flow-designed edits for base-editing compatibility:

| Edit | Editor | Window |
|---|---|---|
| C->T (G->A) | CBE (cytosine base editor) | Positions 4-8 of protospacer |
| A->G (T->C) | ABE (adenine base editor) | Positions 4-8 of protospacer |
| Other SNPs | HDR (homology-directed repair) | Requires donor template |

Pipeline:
1. Flow matching generates 20 candidate edits
2. Filter for <=3 base changes (base-editing compatible)
3. Design guide RNAs: find NGG PAM sites within 20bp
4. Score off-target activity via genome-wide alignment
5. Rank by composite score: motif achievement x LoRA score - edit penalty

## Example Use Cases

### 1. Drought-Responsive Promoter Engineering

```
design(
    source_seq="ZmDREB2A promoter",
    target_tf_change={"disrupt": ["WRKY"], "maintain": ["NAC", "MYB"]},
    target_ase=-0.5,
    condition="DS"
)
```

Goal: Reduce WRKY-mediated drought over-response while maintaining basal expression.

### 2. Condition-Specific Binding Switch

```
design(
    source_seq="ZmHSF promoter",
    target_tf_change={"disrupt": ["HSF"], "create": ["ERF"]},
    condition="WW"
)
```

Goal: Convert heat-stress-specific regulation to general stress regulation.

### 3. Enhancing Natural Variation

Based on our finding that HOX disruption has the strongest ASE effect (delta=0.775):

```
design(
    source_seq="candidate promoter",
    target_tf_change={"disrupt": ["HOX"]},
    target_ase=0.8,
    condition="WW"
)
```

## Connection to Existing Pipeline

| Component | Role | Where |
|---|---|---|
| Step 26 (regulatory grammar) | Identifies which TFs are enriched/depleted at bQTL | Informs conditioning |
| Step 28 (cross-layer validation) | Conservation scores across hybrid panel | Conservation dimension |
| Step 29 (ASE validation) | Per-TF family ASE effects | ASE prediction training |
| Step 30 (cooperativity) | TF pair grammar, enhanceosome/billboard | Multi-TF conditioning |
| Agro-NT LoRA (bqtl_predict) | bQTL classifier (AUC 0.865) | Gradient guidance |
| BQTLNet CNN (bqtl_predict) | Single-nucleotide attribution | Post-design validation |
| MaizeDesigner (bqtl_predict) | MLM-based design agent | Complementary approach |

## Open Questions for Discussion

1. **Sample efficiency**: With 147K pairs (mostly single-SNP edits), can the model learn multi-base editing patterns? May need data augmentation or curriculum learning.

2. **Discrete vs continuous**: DNA is discrete (ACGT) but flow matching operates in continuous space. The argmax decoding at the end is lossy. Alternative: Riemannian flow matching on the simplex, or discrete flow matching (Campbell et al. 2024).

3. **Multi-objective conditioning**: Currently the conditioning is a flat vector. A hierarchical conditioning (TF-level goals -> motif-level constraints -> base-level edits) might learn faster.

4. **Evaluation metrics**: Beyond LoRA score and motif verification, how do we evaluate design quality without wet-lab validation? Cross-validation on held-out bQTL where we know the "right answer" (the natural alt allele).

5. **Scaling**: Can we pre-train on all plant bQTL data (not just maize) and fine-tune? The TF families are largely conserved across angiosperms.

## Implementation Status

- [x] Script: `scripts/31_flow_matching_design.py` (architecture, training loop, inference, CRISPR)
- [x] Nextflow: `FLOW_MATCHING_DESIGN` process in `main.nf` (depends on TF_COOPERATIVITY)
- [x] Agent: Step 31 registered in `atlas_agent.py`
- [ ] Data loader: Connect `load_training_data()` to actual bQTL tables
- [ ] Training: Run on GPU (A100 recommended)
- [ ] Evaluation: Held-out bQTL reconstruction accuracy
- [ ] LoRA integration: Wire actual Agro-NT LoRA for gradient guidance
- [ ] Wet-lab validation: CRISPR designs for experimental testing

## References

- Lipman et al. (2023). Flow Matching for Generative Modeling. ICLR.
- Tong et al. (2024). Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport. TMLR.
- Campbell et al. (2024). A Continuous Time Framework for Discrete Denoising Models. NeurIPS.
- Stark et al. (2024). Dirichlet Flow Matching with Applications to DNA Sequence Design. ICML.
