#!/usr/bin/env python3
"""
16_agro_nt_embeddings.py — Extract Agro-NT embeddings for causal bQTL.

Extracts 1500-dim hidden state embeddings from the Agro Nucleotide Transformer
(1B) for each causal bQTL position. Uses 1001bp windows centered on each bQTL.

Two modes:
  1. Untrained (frozen) backbone — captures pretrained plant regulatory grammar
  2. LoRA fine-tuned adapter — captures bQTL-specific regulatory features

Output:
  results/agro_nt_embeddings_frozen.npy    (N × 1500 float32)
  results/agro_nt_embeddings_lora.npy      (N × 1500 float32)
  results/agro_nt_embedding_positions.csv  (chr, pos, gene_id for each row)

Usage:
  python scripts/16_agro_nt_embeddings.py --mode frozen
  python scripts/16_agro_nt_embeddings.py --mode lora
  python scripts/16_agro_nt_embeddings.py --mode both
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from pipeline_config import config

RESULTS = config.results_dir
GENOME = config.genome

BQTL_PREDICT_DIR = Path(__file__).resolve().parent.parent.parent / "bqtl_predict"
ADAPTER_DIR = BQTL_PREDICT_DIR / "models" / "agro_nt_lora_adapter_hardneg"
MODEL_NAME = "InstaDeepAI/agro-nucleotide-transformer-1b"
WINDOW = 500  # ±500bp = 1001bp total
BATCH_SIZE = 4

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


class SeqDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.sequences[idx],
            return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length,
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }


def load_sequences(causal_df, genome_path):
    """Extract 1001bp sequences centered on each bQTL position."""
    from pyfaidx import Fasta
    genome = Fasta(str(genome_path))

    sequences = []
    valid_mask = []
    for _, row in causal_df.iterrows():
        chrom, pos = row["chr"], int(row["pos"])
        start = max(0, pos - WINDOW)
        end = pos + WINDOW + 1
        try:
            seq = str(genome[chrom][start:end]).upper()
            if len(seq) == 2 * WINDOW + 1:
                sequences.append(seq)
                valid_mask.append(True)
            else:
                sequences.append("N" * (2 * WINDOW + 1))
                valid_mask.append(False)
        except (KeyError, ValueError):
            sequences.append("N" * (2 * WINDOW + 1))
            valid_mask.append(False)

    n_valid = sum(valid_mask)
    print(f"  Extracted {n_valid}/{len(causal_df)} sequences ({2*WINDOW+1}bp each)")
    return sequences, valid_mask


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """Extract mean-pooled hidden state embeddings."""
    model.eval()
    all_embeddings = []

    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # Last layer: (B, T, 1500)

        # Mean pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        all_embeddings.append(pooled.cpu().numpy())

        if (i + 1) % 50 == 0:
            print(f"    Batch {i+1}/{len(dataloader)}", flush=True)

    return np.concatenate(all_embeddings, axis=0)


def run_frozen(sequences, tokenizer, max_length, batch_size=4):
    """Extract embeddings from frozen (pretrained) Agro-NT backbone."""
    from transformers import AutoModelForMaskedLM

    print("\n  Loading frozen Agro-NT backbone...")
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    dataset = SeqDataset(sequences, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Extracting embeddings ({len(sequences)} sequences, batch_size={batch_size})...")
    t0 = time.time()
    embeddings = extract_embeddings(model, loader, DEVICE)
    elapsed = time.time() - t0

    print(f"  Done: {embeddings.shape} in {elapsed:.0f}s ({elapsed/len(sequences):.1f}s/seq)")

    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return embeddings


def run_lora(sequences, tokenizer, max_length, batch_size=4):
    """Extract embeddings from LoRA fine-tuned Agro-NT."""
    from transformers import AutoModelForMaskedLM
    from peft import PeftModel
    import json

    if not ADAPTER_DIR.exists():
        print(f"  ERROR: LoRA adapter not found at {ADAPTER_DIR}")
        return None

    print("\n  Loading Agro-NT + LoRA adapter...")
    with open(ADAPTER_DIR / "adapter_config.json") as f:
        adapter_config = json.load(f)

    backbone = AutoModelForMaskedLM.from_pretrained(
        adapter_config["base_model"], trust_remote_code=True
    )
    model = PeftModel.from_pretrained(backbone, ADAPTER_DIR / "lora_adapter")
    model = model.to(DEVICE)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} ({trainable:,} LoRA)")

    dataset = SeqDataset(sequences, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Extracting embeddings ({len(sequences)} sequences, batch_size={batch_size})...")
    t0 = time.time()
    embeddings = extract_embeddings(model, loader, DEVICE)
    elapsed = time.time() - t0

    print(f"  Done: {embeddings.shape} in {elapsed:.0f}s ({elapsed/len(sequences):.1f}s/seq)")

    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Extract Agro-NT embeddings for causal bQTL")
    parser.add_argument("--mode", choices=["frozen", "lora", "both"],
                        default="both", help="Which model(s) to use")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    batch_size = args.batch_size

    print("=" * 70)
    print("  AGRO-NT EMBEDDING EXTRACTION FOR CAUSAL bQTL")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Mode: {args.mode}")

    # Load causal bQTL
    causal_path = RESULTS / "causal_bqtl_728.csv"
    if not causal_path.exists():
        print(f"  ERROR: {causal_path} not found. Run steps 05-06 first.")
        return
    causal = pd.read_csv(causal_path)
    print(f"  Causal bQTL: {len(causal)}")

    # Extract sequences
    print("\n  Extracting DNA sequences...")
    sequences, valid_mask = load_sequences(causal, GENOME)

    # Save position index
    pos_df = causal[["chr", "pos", "gene_id"]].copy()
    pos_df["valid_seq"] = valid_mask
    pos_df.to_csv(RESULTS / "agro_nt_embedding_positions.csv", index=False)
    print(f"  Saved: agro_nt_embedding_positions.csv")

    # Tokenizer
    from transformers import AutoTokenizer
    print(f"\n  Loading tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    test_tokens = tokenizer("A" * 1001, return_tensors="pt")
    max_length = test_tokens["input_ids"].shape[1]
    print(f"  1001bp → {max_length} tokens")

    # Frozen embeddings
    if args.mode in ("frozen", "both"):
        print("\n" + "=" * 70)
        print("  FROZEN BACKBONE EMBEDDINGS")
        print("=" * 70)
        emb_frozen = run_frozen(sequences, tokenizer, max_length, batch_size)
        np.save(RESULTS / "agro_nt_embeddings_frozen.npy", emb_frozen.astype(np.float32))
        print(f"  Saved: agro_nt_embeddings_frozen.npy ({emb_frozen.shape})")

    # LoRA embeddings
    if args.mode in ("lora", "both"):
        print("\n" + "=" * 70)
        print("  LORA FINE-TUNED EMBEDDINGS")
        print("=" * 70)
        emb_lora = run_lora(sequences, tokenizer, max_length, batch_size)
        if emb_lora is not None:
            np.save(RESULTS / "agro_nt_embeddings_lora.npy", emb_lora.astype(np.float32))
            print(f"  Saved: agro_nt_embeddings_lora.npy ({emb_lora.shape})")

    # Quick summary if both done
    if args.mode == "both" and emb_lora is not None:
        from scipy.spatial.distance import cosine
        cos_sims = []
        for i in range(len(emb_frozen)):
            cos_sims.append(1 - cosine(emb_frozen[i], emb_lora[i]))
        cos_sims = np.array(cos_sims)
        print(f"\n  Frozen vs LoRA cosine similarity:")
        print(f"    Mean: {cos_sims.mean():.4f}")
        print(f"    Std:  {cos_sims.std():.4f}")
        print(f"    Min:  {cos_sims.min():.4f}")
        print(f"    Max:  {cos_sims.max():.4f}")

    print(f"\n{'=' * 70}")
    print(f"  DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
