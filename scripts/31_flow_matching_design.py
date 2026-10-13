#!/usr/bin/env python3
"""
31_flow_matching_design.py — Conditional Flow Matching for Regulatory Sequence Design

Framework for learning DIRECTED sequence edits that achieve specific TF binding
changes while predicting allele-specific expression effects.

=== BIOLOGICAL MOTIVATION ===

We have established (steps 26-30) that:
  - bQTL disrupt specific TF motifs (C2H2, NAC, MYB dominate)
  - Disruption → measurable ASE (p=2e-242, Cohen's d=0.37)
  - TF pairs co-bind cooperatively (enhanceosome) but contribute additively to expression
  - 81% of causal bQTL are condition-specific (WW vs DS)
  - Motif conservation across 23 hybrids validates functional importance

This means: we can learn a GENERATIVE MODEL that maps desired TF binding changes
to the minimal sequence edits needed. Flow matching is ideal because:
  1. It learns continuous interpolation paths (not discrete jumps)
  2. Conditional on the desired outcome (which TFs to gain/lose)
  3. Naturally handles the constraint that most of the sequence should stay fixed
  4. The velocity field can be steered by Agro-NT LoRA gradients

=== ARCHITECTURE ===

                    ┌─────────────────────────────┐
                    │    Conditioning Vector c     │
                    │  [TF_target, ASE_target,     │
                    │   hybrid_context, condition] │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
     x_0 ───────►  │  Flow Matching Network       │  ───────► x_1
  (source seq)      │  U-Net 1D / Transformer      │        (edited seq)
                    │  v_θ(x_t, t, c)              │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │  Scoring / Validation         │
                    │  • Agro-NT LoRA (bQTL score)  │
                    │  • CNN IG (attribution)        │
                    │  • PWM scan (motif check)      │
                    │  • ASE predictor (expression)  │
                    └───────────────────────────────┘

=== TRAINING DATA ===

From our atlas:
  - 147,942 bQTL positions with ref/alt alleles
  - 1,075 WW causal + 1,267 DS causal bQTL with ASE measurements
  - Per-hybrid haplotype calls across 23 F1 hybrids
  - TF motif disruption/creation annotations per bQTL (405 PWMs, 33 families)
  - Population motif occupancy (conservation) scores

Training pairs: (ref_seq, alt_seq, TF_disrupted, TF_created, ASE_effect, condition)
  - Source: ref allele sequence (±500bp around bQTL)
  - Target: alt allele sequence
  - Condition: which TFs lost/gained, ASE magnitude, WW/DS

=== IMPLEMENTATION ===
"""

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Defer torch import for documentation/planning mode
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    import types as _t
    # Stubs so class definitions parse without torch
    class _Stub:
        pass
    Dataset = _Stub
    torch = _t.ModuleType("torch")
    torch.no_grad = lambda: (lambda f: f)  # identity decorator
    nn = _t.ModuleType("nn")
    nn.Module = _Stub
    nn.ModuleList = list
    nn.Sequential = list
    nn.Conv1d = _Stub
    nn.ConvTranspose1d = _Stub
    nn.Linear = _Stub
    nn.GroupNorm = _Stub
    nn.Embedding = _Stub
    nn.SiLU = _Stub
    nn.AdaptiveAvgPool1d = _Stub
    F = _t.ModuleType("F")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
ATLAS_DIR = SCRIPT_DIR.parent
BQTL_DIR = ATLAS_DIR.parent / "bqtl_predict"
RESULTS_DIR = ATLAS_DIR / "results"


# ============================================================================
# 1. DATA: bQTL Edit Pairs
# ============================================================================

@dataclass
class EditPair:
    """A single ref→alt edit with biological annotations."""
    chrom: str
    pos: int
    ref_seq: str          # 1001bp centered on variant
    alt_seq: str          # same but with alt allele
    tf_disrupted: list    # TF families whose motifs are lost
    tf_created: list      # TF families whose motifs are gained
    ase_log2fc: float     # log2(alt/ref) expression ratio
    condition: str        # "WW" or "DS"
    hybrid: str           # which hybrid
    conservation: float   # fraction of hybrids with this motif (0-1)


class BQTLEditDataset(Dataset):
    """
    Dataset of (source_seq, target_seq, conditioning) for flow matching.

    Each sample is a bQTL where we know:
      - The ref and alt sequences
      - Which TF motifs are disrupted/created
      - The ASE effect
      - The condition (WW/DS)

    For training, we learn to flow from ref → alt (disruption direction)
    AND from alt → ref (creation/restoration direction), doubling the data.
    """

    # 33 TF families from DAP-seq PWM database
    TF_FAMILIES = [
        "AP2", "B3", "BBR-BPC", "BES1", "bHLH", "bZIP", "C2H2", "C3H",
        "CAMTA", "CPP", "Dof", "E2F/DP", "ERF", "G2-like", "GATA",
        "GeBP", "GRAS", "GRF", "HD-ZIP", "HOX", "HSF", "LBD", "MIKC_MADS",
        "MYB", "MYB-related", "NAC", "Nin-like", "RAV", "SBP", "SRS",
        "TCP", "Trihelix", "WRKY", "WOX", "YABBY"
    ]
    N_TF = len(TF_FAMILIES)
    TF_TO_IDX = {f: i for i, f in enumerate(TF_FAMILIES)}

    SEQ_LEN = 1001       # ±500bp around variant
    N_BASES = 4           # one-hot ACGT
    COND_DIM = 2 * N_TF + 3  # disrupted (35) + created (35) + ASE + condition + conservation

    def __init__(self, edit_pairs: list[EditPair], bidirectional: bool = True):
        self.pairs = edit_pairs
        self.bidirectional = bidirectional

    def __len__(self):
        return len(self.pairs) * (2 if self.bidirectional else 1)

    def _encode_seq(self, seq: str) -> np.ndarray:
        """One-hot encode DNA sequence → (4, L)."""
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        arr = np.zeros((4, len(seq)), dtype=np.float32)
        for i, base in enumerate(seq.upper()):
            if base in mapping:
                arr[mapping[base], i] = 1.0
        return arr

    def _encode_condition(self, pair: EditPair, reverse: bool = False) -> np.ndarray:
        """Encode the conditioning vector."""
        c = np.zeros(self.COND_DIM, dtype=np.float32)

        # TF disruption targets (what we WANT to disrupt)
        disrupted = pair.tf_created if reverse else pair.tf_disrupted
        created = pair.tf_disrupted if reverse else pair.tf_created
        for tf in disrupted:
            if tf in self.TF_TO_IDX:
                c[self.TF_TO_IDX[tf]] = 1.0
        for tf in created:
            if tf in self.TF_TO_IDX:
                c[self.N_TF + self.TF_TO_IDX[tf]] = 1.0

        # ASE target (signed: positive = increase expression)
        c[2 * self.N_TF] = -pair.ase_log2fc if reverse else pair.ase_log2fc
        # Condition: 0 = WW, 1 = DS
        c[2 * self.N_TF + 1] = 1.0 if pair.condition == "DS" else 0.0
        # Conservation score
        c[2 * self.N_TF + 2] = pair.conservation
        return c

    def __getitem__(self, idx):
        reverse = idx >= len(self.pairs)
        pair = self.pairs[idx % len(self.pairs)]

        if reverse:
            x0 = self._encode_seq(pair.alt_seq)
            x1 = self._encode_seq(pair.ref_seq)
        else:
            x0 = self._encode_seq(pair.ref_seq)
            x1 = self._encode_seq(pair.alt_seq)

        c = self._encode_condition(pair, reverse=reverse)
        return {
            "x0": x0,   # source sequence (4, L)
            "x1": x1,   # target sequence (4, L)
            "c": c,      # conditioning vector
        }


# ============================================================================
# 2. MODEL: Conditional Flow Matching Network
# ============================================================================

class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal embedding for flow time t ∈ [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freq[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ConvBlock(nn.Module):
    """1D convolution block with residual connection."""

    def __init__(self, channels: int, kernel_size: int = 9, cond_dim: int = 128):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.cond_proj = nn.Linear(cond_dim, channels)

    def forward(self, x, c_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        # Add conditioning
        h = h + self.cond_proj(c_emb).unsqueeze(-1)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class FlowMatchingUNet1D(nn.Module):
    """
    1D U-Net for conditional flow matching on DNA sequences.

    Input: x_t ∈ R^{4×L} (interpolated one-hot at time t)
    Condition: c ∈ R^{COND_DIM} (TF targets + ASE + condition)
    Output: v_θ(x_t, t, c) ∈ R^{4×L} (velocity field)

    Architecture:
      - Encoder: 3 levels of ConvBlocks with downsampling
      - Bottleneck: ConvBlock at lowest resolution
      - Decoder: 3 levels with upsampling and skip connections
      - Time t embedded via sinusoidal encoding
      - Condition c projected and added at each level
    """

    def __init__(
        self,
        seq_len: int = 1001,
        in_channels: int = 4,
        base_channels: int = 64,
        n_levels: int = 3,
        blocks_per_level: int = 2,
        kernel_size: int = 9,
        cond_dim: int = 73,   # 2*35 + 3
        time_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cond_dim = cond_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbed(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Condition embedding (shared across levels)
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Combined embedding dimension
        emb_dim = time_dim  # time + condition summed

        # Input projection
        self.in_proj = nn.Conv1d(in_channels, base_channels, 1)

        # Encoder
        self.encoder = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        self.skip_channels = []
        for level in range(n_levels):
            blocks = nn.ModuleList()
            for _ in range(blocks_per_level):
                blocks.append(ConvBlock(ch, kernel_size, emb_dim))
            self.encoder.append(blocks)
            self.skip_channels.append(ch)
            if level < n_levels - 1:
                next_ch = ch * 2
                self.downsamples.append(nn.Conv1d(ch, next_ch, 4, stride=2, padding=1))
                ch = next_ch

        # Bottleneck
        self.bottleneck = ConvBlock(ch, kernel_size, emb_dim)

        # Decoder
        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        for level in reversed(range(n_levels)):
            if level < n_levels - 1:
                prev_ch = ch
                ch = self.skip_channels[level]
                self.upsamples.append(nn.ConvTranspose1d(prev_ch, ch, 4, stride=2, padding=1))
                self.skip_projs.append(nn.Conv1d(ch * 2, ch, 1))
            blocks = nn.ModuleList()
            for _ in range(blocks_per_level):
                blocks.append(ConvBlock(ch, kernel_size, emb_dim))
            self.decoder.append(blocks)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv1d(ch, in_channels, 1),
        )

    def forward(self, x_t, t, c):
        """
        Args:
            x_t: (B, 4, L) interpolated sequence
            t:   (B,) flow time in [0, 1]
            c:   (B, COND_DIM) conditioning vector
        Returns:
            v:   (B, 4, L) velocity field
        """
        # Embeddings
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(c)
        emb = t_emb + c_emb  # (B, emb_dim)

        # Input
        h = self.in_proj(x_t)

        # Encoder (save skip connections)
        skips = []
        for level, blocks in enumerate(self.encoder):
            for block in blocks:
                h = block(h, emb)
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        # Bottleneck
        h = self.bottleneck(h, emb)

        # Decoder
        for level, blocks in enumerate(self.decoder):
            if level > 0:
                h = self.upsamples[level - 1](h)
                skip = skips[-(level + 1)]
                # Handle size mismatch from odd lengths
                if h.shape[-1] != skip.shape[-1]:
                    h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
                h = self.skip_projs[level - 1](torch.cat([h, skip], dim=1))
            for block in blocks:
                h = block(h, emb)

        return self.out_proj(h)


# ============================================================================
# 3. TRAINING: Optimal Transport Conditional Flow Matching
# ============================================================================

@dataclass
class FlowMatchingConfig:
    """Training configuration."""
    # Model
    base_channels: int = 64
    n_levels: int = 3
    blocks_per_level: int = 2
    kernel_size: int = 9

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    n_epochs: int = 100
    warmup_steps: int = 500

    # Flow matching
    sigma_min: float = 1e-4   # minimum noise for OT path

    # Agro-NT LoRA guidance
    guidance_weight: float = 0.5    # classifier-free guidance weight
    lora_guidance_start: float = 0.3  # start LoRA guidance after 30% of flow

    # Data
    seq_len: int = 1001
    bidirectional: bool = True

    # ASE prediction head
    predict_ase: bool = True
    ase_loss_weight: float = 0.1


def ot_flow_matching_loss(model, x0, x1, c, sigma_min=1e-4):
    """
    Optimal Transport Conditional Flow Matching loss (Lipman et al. 2023).

    The OT path between x0 and x1 is:
        x_t = (1 - t) * x0 + t * x1
    with velocity:
        u_t = x1 - x0

    We train the model to predict this velocity:
        L = E_t ||v_θ(x_t, t, c) - u_t||^2
    """
    B = x0.shape[0]
    device = x0.device

    # Sample random time t ~ U(0, 1)
    t = torch.rand(B, device=device)

    # Add small noise for numerical stability
    noise = torch.randn_like(x0) * sigma_min

    # OT interpolation: x_t = (1-t)*x0 + t*x1 + σ*noise
    t_expand = t[:, None, None]  # (B, 1, 1)
    x_t = (1 - t_expand) * x0 + t_expand * x1 + noise

    # Target velocity: u_t = x1 - x0
    u_t = x1 - x0

    # Predicted velocity
    v_t = model(x_t, t, c)

    # MSE loss
    loss = F.mse_loss(v_t, u_t)
    return loss


def train_epoch(model, dataloader, optimizer, scheduler, config, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        x0 = batch["x0"].to(device)
        x1 = batch["x1"].to(device)
        c = batch["c"].to(device)

        optimizer.zero_grad()
        loss = ot_flow_matching_loss(model, x0, x1, c, config.sigma_min)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================================
# 4. INFERENCE: ODE Integration with LoRA Guidance
# ============================================================================

@torch.no_grad()
def generate_edit(
    model,
    source_seq: str,
    condition: np.ndarray,
    n_steps: int = 50,
    lora_scorer=None,
    guidance_weight: float = 0.5,
    device: str = "cpu",
) -> tuple[str, list[dict]]:
    """
    Generate a sequence edit by integrating the learned flow ODE.

    Uses Euler integration: x_{t+dt} = x_t + dt * v_θ(x_t, t, c)

    Optionally guided by Agro-NT LoRA gradient to steer toward high bQTL score.

    Args:
        model: Trained FlowMatchingUNet1D
        source_seq: Input DNA sequence (1001bp)
        condition: Conditioning vector (TF targets, ASE, etc.)
        n_steps: Number of Euler steps
        lora_scorer: Optional Agro-NT LoRA model for gradient guidance
        guidance_weight: Strength of LoRA gradient guidance
        device: Compute device

    Returns:
        edited_seq: The designed sequence
        trajectory: List of intermediate states with scores
    """
    model.eval()

    # Encode source sequence
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    x = torch.zeros(1, 4, len(source_seq), device=device)
    for i, base in enumerate(source_seq.upper()):
        if base in mapping:
            x[0, mapping[base], i] = 1.0

    c = torch.tensor(condition, device=device).unsqueeze(0)
    dt = 1.0 / n_steps
    trajectory = []

    for step in range(n_steps):
        t = torch.tensor([step * dt], device=device)

        # Predicted velocity from flow matching
        v = model(x, t, c)

        # Optional: LoRA gradient guidance (classifier guidance)
        if lora_scorer is not None and step > n_steps * 0.3:
            x_guided = x.detach().requires_grad_(True)
            score = lora_scorer(x_guided)  # bQTL probability
            grad = torch.autograd.grad(score.sum(), x_guided)[0]
            v = v + guidance_weight * grad

        # Euler step
        x = x + dt * v

        # Record trajectory
        if step % (n_steps // 10) == 0:
            trajectory.append({
                "step": step,
                "t": step * dt,
                "x_norm": x.norm().item(),
            })

    # Decode: argmax over base dimension → sequence
    indices = x.squeeze(0).argmax(dim=0).cpu().numpy()
    decode = {0: "A", 1: "C", 2: "G", 3: "T"}
    edited_seq = "".join(decode[i] for i in indices)

    return edited_seq, trajectory


# ============================================================================
# 5. AGRO-NT LORA STEERING
# ============================================================================

class LoRAGuidedFlowMatcher:
    """
    Combines flow matching with Agro-NT LoRA for guided sequence design.

    The key insight: Agro-NT LoRA (AUC 0.865) already knows what makes a
    sequence a bQTL. We use its gradients to STEER the flow matching ODE
    toward sequences that score highly while respecting the learned
    TF-specific editing patterns.

    Two guidance modes:
    1. Score guidance: push toward higher/lower bQTL probability
    2. Motif guidance: push toward specific TF motif gain/loss patterns

    The flow model learns the EDITING GRAMMAR from our 147K bQTL pairs.
    The LoRA model provides the QUALITY SIGNAL for what makes good edits.
    """

    def __init__(self, flow_model, lora_model, cnn_model=None, pwm_scanner=None):
        self.flow = flow_model
        self.lora = lora_model
        self.cnn = cnn_model          # for single-nucleotide attribution
        self.scanner = pwm_scanner    # for motif verification

    def design(
        self,
        source_seq: str,
        target_tf_change: dict,      # {"disrupt": ["WRKY"], "maintain": ["NAC"]}
        target_ase: float = None,     # desired log2FC
        condition: str = "WW",
        n_candidates: int = 10,
        n_steps: int = 50,
        device: str = "cpu",
    ) -> list[dict]:
        """
        Design sequence edits to achieve specific TF binding changes.

        Example:
            design(
                source_seq="ACGT...",
                target_tf_change={"disrupt": ["WRKY"], "maintain": ["NAC", "MYB"]},
                target_ase=-0.5,  # reduce expression
                condition="DS",   # under drought
            )

        Returns ranked candidates with:
          - edited sequence
          - edit distance (number of changes)
          - bQTL score (from LoRA)
          - motif scan (before/after)
          - predicted ASE effect
        """
        # Build conditioning vector
        c = self._build_condition(target_tf_change, target_ase, condition)

        candidates = []
        for i in range(n_candidates):
            # Add small noise to conditioning for diversity
            c_noisy = c + np.random.randn(*c.shape) * 0.05

            edited, traj = generate_edit(
                self.flow, source_seq, c_noisy,
                n_steps=n_steps,
                lora_scorer=self.lora if i > 0 else None,  # first = unguided
                device=device,
            )

            # Score the candidate
            result = self._score_candidate(source_seq, edited, target_tf_change)
            result["trajectory"] = traj
            candidates.append(result)

        # Rank by composite score
        candidates.sort(key=lambda x: x["composite_score"], reverse=True)
        return candidates

    def _build_condition(self, target_tf_change, target_ase, condition):
        """Build conditioning vector from human-readable specification."""
        c = np.zeros(BQTLEditDataset.COND_DIM, dtype=np.float32)

        for tf in target_tf_change.get("disrupt", []):
            if tf in BQTLEditDataset.TF_TO_IDX:
                c[BQTLEditDataset.TF_TO_IDX[tf]] = 1.0

        for tf in target_tf_change.get("create", []):
            if tf in BQTLEditDataset.TF_TO_IDX:
                c[BQTLEditDataset.N_TF + BQTLEditDataset.TF_TO_IDX[tf]] = 1.0

        if target_ase is not None:
            c[2 * BQTLEditDataset.N_TF] = target_ase
        c[2 * BQTLEditDataset.N_TF + 1] = 1.0 if condition == "DS" else 0.0
        c[2 * BQTLEditDataset.N_TF + 2] = 0.9  # assume conserved target
        return c

    def _score_candidate(self, source, edited, target_tf_change):
        """Score an edited sequence against design objectives."""
        # Edit distance
        n_edits = sum(1 for a, b in zip(source, edited) if a != b)

        # Motif check (placeholder — would use PWM scanner)
        motif_score = 1.0  # fraction of targets achieved

        # LoRA score (placeholder — would use actual model)
        lora_score = 0.5

        # Composite: minimize edits while maximizing target achievement
        edit_penalty = max(0, n_edits - 5) * 0.01  # penalize >5 edits
        composite = motif_score * 0.4 + lora_score * 0.4 - edit_penalty + 0.2

        return {
            "edited_seq": edited,
            "n_edits": n_edits,
            "edit_positions": [i for i, (a, b) in enumerate(zip(source, edited)) if a != b],
            "lora_score": lora_score,
            "motif_score": motif_score,
            "composite_score": composite,
        }


# ============================================================================
# 6. ASE PREDICTION HEAD
# ============================================================================

class ASEPredictor(nn.Module):
    """
    Predict allele-specific expression from sequence pair + hybrid context.

    This is a SEPARATE head that can be trained jointly with the flow model
    or independently. It learns:

        f(ref_seq, alt_seq, hybrid_embedding) → log2(alt/ref) ASE

    Key insight from step 29: ALL 30 TF families show significant
    disruption → ASE effects. HOX is strongest (Δ=0.775), but the
    relationship is additive (billboard), meaning we can sum contributions.

    Training data: 2,342 causal bQTL × 23 hybrids with measured ASE values.
    """

    def __init__(self, seq_channels: int = 64, n_hybrids: int = 23):
        super().__init__()
        # Sequence difference encoder
        self.seq_encoder = nn.Sequential(
            nn.Conv1d(4, 32, 9, padding=4),
            nn.SiLU(),
            nn.Conv1d(32, seq_channels, 9, padding=4),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Hybrid embedding (learnable)
        self.hybrid_embed = nn.Embedding(n_hybrids, 16)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(seq_channels + 16, 64),
            nn.SiLU(),
            nn.Linear(64, 1),  # log2FC
        )

    def forward(self, ref_seq, alt_seq, hybrid_idx):
        """
        Args:
            ref_seq: (B, 4, L) one-hot reference
            alt_seq: (B, 4, L) one-hot alternate
            hybrid_idx: (B,) hybrid index
        Returns:
            ase_pred: (B, 1) predicted log2FC
        """
        diff = alt_seq - ref_seq  # difference encoding
        h = self.seq_encoder(diff).squeeze(-1)  # (B, seq_channels)
        hybrid_emb = self.hybrid_embed(hybrid_idx)  # (B, 16)
        combined = torch.cat([h, hybrid_emb], dim=-1)
        return self.head(combined)


# ============================================================================
# 7. HYBRID GENERALIZATION
# ============================================================================

class HybridGeneralizer:
    """
    Generalize to new (unseen) F1 hybrids.

    Strategy: Learn a hybrid embedding space from the 23 training hybrids.
    For a new hybrid, we can:
    1. Use the MEAN embedding (average regulatory behavior)
    2. Use k-mer composition of the NAM parent to find nearest training hybrid
    3. Use a few ASE measurements to fine-tune the embedding (few-shot)

    The key property enabling this: our flow model is conditioned on
    TF-level targets (33 families), not hybrid-specific features. The
    hybrid embedding only matters for the ASE prediction head.

    For the flow model itself, a new hybrid just needs:
    - Its genome sequence (for source sequences)
    - The desired TF binding change specification

    This is because TF binding grammar is CONSERVED — the same motifs
    mean the same thing regardless of genetic background. What varies
    is which motifs are present/absent in each haplotype.
    """

    def __init__(self, ase_predictor, known_hybrids: list[str]):
        self.ase_predictor = ase_predictor
        self.known_hybrids = known_hybrids
        self.hybrid_to_idx = {h: i for i, h in enumerate(known_hybrids)}

    def embed_new_hybrid(
        self,
        hybrid_name: str,
        calibration_data: list[tuple] = None,  # [(ref_seq, alt_seq, measured_ase)]
        method: str = "mean",
    ) -> int:
        """
        Create embedding for a new hybrid.

        Methods:
            "mean": Use average of all known hybrid embeddings
            "nearest": Find nearest known hybrid by genome similarity
            "finetune": Use calibration ASE data to optimize embedding
        """
        if method == "mean":
            with torch.no_grad():
                all_embs = self.ase_predictor.hybrid_embed.weight.data
                mean_emb = all_embs.mean(dim=0)
                # Add as new embedding
                new_idx = len(self.known_hybrids)
                new_weight = torch.cat([all_embs, mean_emb.unsqueeze(0)])
                self.ase_predictor.hybrid_embed = nn.Embedding.from_pretrained(
                    new_weight, freeze=False
                )
                return new_idx

        elif method == "finetune" and calibration_data is not None:
            # Few-shot: optimize just the new embedding
            new_idx = self.embed_new_hybrid(hybrid_name, method="mean")
            # Freeze everything except the new embedding
            for p in self.ase_predictor.parameters():
                p.requires_grad_(False)
            self.ase_predictor.hybrid_embed.weight.requires_grad_(True)

            optimizer = torch.optim.Adam(
                [self.ase_predictor.hybrid_embed.weight], lr=1e-3
            )

            for epoch in range(50):
                total_loss = 0
                for ref_seq, alt_seq, measured_ase in calibration_data:
                    pred = self.ase_predictor(ref_seq, alt_seq,
                                              torch.tensor([new_idx]))
                    loss = F.mse_loss(pred, torch.tensor([[measured_ase]]))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            # Unfreeze
            for p in self.ase_predictor.parameters():
                p.requires_grad_(True)
            return new_idx


# ============================================================================
# 8. PRECISION BREEDING / CRISPR APPLICATION
# ============================================================================

@dataclass
class CRISPRDesign:
    """A CRISPR edit design for precision breeding."""
    target_gene: str
    target_chrom: str
    target_pos: int
    ref_allele: str
    designed_allele: str
    edit_type: str           # "SNP", "small_indel", "base_edit"
    guide_rna: str           # 20bp guide RNA sequence
    pam_site: str            # PAM site (NGG for SpCas9)
    off_target_score: float  # predicted off-target activity

    # Biological rationale
    tf_disrupted: list
    tf_maintained: list
    predicted_ase: float
    condition_specificity: str  # "WW", "DS", "both"

    # Confidence
    lora_score: float
    motif_verification: bool
    conservation_score: float


def design_crispr_edit(
    flow_matcher: LoRAGuidedFlowMatcher,
    gene_name: str,
    promoter_seq: str,
    target_tf_change: dict,
    condition: str = "WW",
) -> list[CRISPRDesign]:
    """
    Design CRISPR base edits for precision breeding.

    Pipeline:
    1. Flow matching generates candidate edits
    2. Filter for minimal edits (≤3 bases) compatible with base editing
    3. Design guide RNAs for each candidate
    4. Score off-target activity
    5. Rank by composite score

    Base editing compatibility:
    - C→T (or G→A on complement): CBE (cytosine base editor)
    - A→G (or T→C on complement): ABE (adenine base editor)
    - Within editing window: positions 4-8 of the protospacer

    Example use case:
        "Increase WRKY binding at ZmDREB2A promoter under drought
         while maintaining NAC binding for baseline expression"
    """
    candidates = flow_matcher.design(
        source_seq=promoter_seq,
        target_tf_change=target_tf_change,
        condition=condition,
        n_candidates=20,
    )

    crispr_designs = []
    for cand in candidates:
        if cand["n_edits"] > 3:
            continue  # too many edits for base editing

        for pos in cand["edit_positions"]:
            ref_base = promoter_seq[pos]
            alt_base = cand["edited_seq"][pos]

            # Check base editing compatibility
            if (ref_base, alt_base) in [("C", "T"), ("A", "G")]:
                edit_type = "base_edit"
            elif (ref_base, alt_base) in [("G", "A"), ("T", "C")]:
                edit_type = "base_edit"  # complement strand
            else:
                edit_type = "SNP"  # needs HDR

            # Find PAM sites near edit position
            # SpCas9: NGG on either strand within ~20bp
            pam_candidates = []
            for offset in range(-20, 20):
                pam_pos = pos + offset
                if 0 <= pam_pos + 2 < len(promoter_seq):
                    if promoter_seq[pam_pos + 1:pam_pos + 3] == "GG":
                        pam_candidates.append(pam_pos)

            if not pam_candidates:
                continue

            best_pam = min(pam_candidates, key=lambda p: abs(p - pos))
            guide_start = best_pam - 20
            if guide_start < 0 or guide_start + 20 > len(promoter_seq):
                continue

            guide_rna = promoter_seq[guide_start:guide_start + 20]

            design = CRISPRDesign(
                target_gene="",  # filled by caller
                target_chrom="",
                target_pos=pos,
                ref_allele=ref_base,
                designed_allele=alt_base,
                edit_type=edit_type,
                guide_rna=guide_rna,
                pam_site=promoter_seq[best_pam:best_pam + 3],
                off_target_score=0.0,  # requires genome-wide search
                tf_disrupted=list(target_tf_change.get("disrupt", [])),
                tf_maintained=list(target_tf_change.get("maintain", [])),
                predicted_ase=0.0,
                condition_specificity=condition,
                lora_score=cand["lora_score"],
                motif_verification=cand["motif_score"] > 0.8,
                conservation_score=0.0,
            )
            crispr_designs.append(design)

    return crispr_designs


# ============================================================================
# 9. FULL PIPELINE: Data → Train → Design → Validate
# ============================================================================

def load_training_data(atlas_dir: Path) -> list[EditPair]:
    """
    Load bQTL edit pairs from the causal atlas results.

    Sources:
      - causal bQTL: results/causal_bqtl_WW.tsv, results/causal_bqtl_DS.tsv
      - motif annotations: results/allele_specific_motifs/
      - ASE data: from hybrid-specific expression tables
      - sequences: extracted from B73 NAM5 reference genome

    Returns list of EditPair objects ready for flow matching training.
    """
    results_dir = atlas_dir / "results"

    # This would load the actual data files
    # For now, document the expected format
    log.info("Loading training data from %s", results_dir)
    log.info("Expected files:")
    log.info("  - causal_bqtl_WW.tsv (1,075 causal bQTL)")
    log.info("  - causal_bqtl_DS.tsv (1,267 causal bQTL)")
    log.info("  - allele_specific_motifs_summary.json (disruption/creation counts)")
    log.info("  - population_motif_summary.csv (conservation scores)")
    log.info("  - Full bQTL table: 147,942 positions with ref/alt alleles")

    return []  # placeholder


def main():
    parser = argparse.ArgumentParser(
        description="Conditional Flow Matching for Regulatory Sequence Design"
    )
    sub = parser.add_subparsers(dest="command")

    # Train
    train_p = sub.add_parser("train", help="Train flow matching model")
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--batch-size", type=int, default=32)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--device", default="auto")
    train_p.add_argument("--output", default=str(ATLAS_DIR / "models" / "flow_matching"))

    # Design
    design_p = sub.add_parser("design", help="Design sequence edits")
    design_p.add_argument("--model", required=True, help="Path to trained model")
    design_p.add_argument("--seq", required=True, help="Source sequence or chr:pos")
    design_p.add_argument("--disrupt", nargs="+", default=[], help="TF families to disrupt")
    design_p.add_argument("--maintain", nargs="+", default=[], help="TF families to maintain")
    design_p.add_argument("--create", nargs="+", default=[], help="TF families to create")
    design_p.add_argument("--ase", type=float, default=None, help="Target ASE log2FC")
    design_p.add_argument("--condition", choices=["WW", "DS"], default="WW")
    design_p.add_argument("--n-candidates", type=int, default=10)
    design_p.add_argument("--lora-adapter", default=None, help="Path to LoRA adapter for guidance")

    # Evaluate
    eval_p = sub.add_parser("evaluate", help="Evaluate on held-out bQTL")
    eval_p.add_argument("--model", required=True)
    eval_p.add_argument("--test-set", required=True)

    # CRISPR
    crispr_p = sub.add_parser("crispr", help="Design CRISPR edits for a target gene")
    crispr_p.add_argument("--model", required=True)
    crispr_p.add_argument("--gene", required=True, help="Gene name or ID")
    crispr_p.add_argument("--disrupt", nargs="+", default=[])
    crispr_p.add_argument("--maintain", nargs="+", default=[])
    crispr_p.add_argument("--condition", choices=["WW", "DS"], default="WW")

    args = parser.parse_args()

    if args.command == "train":
        if not HAS_TORCH:
            log.error("PyTorch required for training. pip install torch")
            sys.exit(1)

        device = args.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        log.info("Device: %s", device)

        # Load data
        pairs = load_training_data(ATLAS_DIR)
        if not pairs:
            log.warning("No training data loaded — running in demo mode")
            log.info("")
            log.info("=" * 70)
            log.info("FLOW MATCHING DESIGN FRAMEWORK — ARCHITECTURE SUMMARY")
            log.info("=" * 70)
            log.info("")
            log.info("Training data dimensions:")
            log.info("  147,942 bQTL × 2 (bidirectional) = 295,884 edit pairs")
            log.info("  + 2,342 causal bQTL with measured ASE effects")
            log.info("  + 23 hybrid contexts per causal bQTL")
            log.info("  Sequence: 1001bp one-hot (4 × 1001)")
            log.info("  Condition: 73-dim (35 disrupt + 35 create + ASE + WW/DS + conservation)")
            log.info("")
            log.info("Model: 1D U-Net with 3 levels, ~7.9M parameters")
            log.info("  + Sinusoidal time embedding (64-dim)")
            log.info("  + Condition embedding (73 → 64-dim)")
            log.info("  + OT flow matching loss (Lipman et al. 2023)")
            log.info("")
            log.info("Guidance: Agro-NT LoRA (AUC 0.865) classifier gradients")
            log.info("  Applied after t > 0.3 for stable initial trajectory")
            log.info("")
            log.info("ASE prediction: Separate head trained on 2,342 × 23 pairs")
            log.info("  23 learnable hybrid embeddings for generalization")
            log.info("")
            log.info("Training estimate:")
            log.info("  ~295K pairs × 100 epochs = ~30M forward passes")
            log.info("  GPU (A100): ~4-6 hours")
            log.info("  Apple MPS: ~12-18 hours")
            log.info("")
            log.info("CRISPR integration:")
            log.info("  Filter edits for base-editing compatibility (C→T, A→G)")
            log.info("  Design guide RNAs with PAM site proximity")
            log.info("  Off-target scoring via genome-wide alignment")
            return

        config = FlowMatchingConfig(
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

        dataset = BQTLEditDataset(pairs, bidirectional=config.bidirectional)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        model = FlowMatchingUNet1D(
            seq_len=config.seq_len,
            base_channels=config.base_channels,
            n_levels=config.n_levels,
            cond_dim=BQTLEditDataset.COND_DIM,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        log.info("Model parameters: %s", f"{n_params:,}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.n_epochs * len(dataloader)
        )

        # Training loop
        best_loss = float("inf")
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(config.n_epochs):
            loss = train_epoch(model, dataloader, optimizer, scheduler, config, device)
            log.info("Epoch %d/%d  loss=%.6f", epoch + 1, config.n_epochs, loss)

            if loss < best_loss:
                best_loss = loss
                torch.save({
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "loss": loss,
                }, output_dir / "best_model.pt")

        log.info("Training complete. Best loss: %.6f", best_loss)

    elif args.command == "design":
        log.info("Design mode — loading model from %s", args.model)
        target = {
            "disrupt": args.disrupt,
            "maintain": args.maintain,
            "create": args.create,
        }
        log.info("Target TF changes: %s", target)
        log.info("Target ASE: %s", args.ase)
        log.info("Condition: %s", args.condition)

    elif args.command == "evaluate":
        log.info("Evaluate mode — not yet implemented")

    elif args.command == "crispr":
        log.info("CRISPR design mode — not yet implemented")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
