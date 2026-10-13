#!/usr/bin/env python3
"""
53_download_nam_genotypes.py — Download NAM founder genotypes at bQTL positions.

Part of: "Condition-dependent bQTL switching" paper (Banf & Hartwig)
Paper section: Methods ("NAM founder genotypes")
Pipeline step: 1 of 8 (53 → 54 → 55 → 56 → 57 → 58 → 59 → 60)

Data source: Grzybowski et al. 2023 VCFs via CyVerse iRODS (anonymous access)
  /iplant/home/shared/Grzybowski_MaizeSNPset_2022/imputed/

Strategy: Download each chromosome VCF + tabix index via iRODS,
extract NAM founder genotypes at bQTL positions with bcftools,
then delete the full VCF to save disk space.

Runtime: ~4-6 hours (downloads ~100 GB total, processes sequentially)
Requires: python-irodsclient, bcftools, internet access (CyVerse port 1247)

Input:
  data/processed/bqtl_snp_ww.csv — 147,942 bQTL positions

Output:
  data/processed/nam_founder_genotypes_at_bqtl.tsv — 43,900 variants × 24 founders
    Columns: chr, pos, ref, alt, <founder1_gt>, <founder2_gt>, ...
    Coverage: 29.7% of bQTL positions (limited by VCF content)
"""

import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from irods.session import iRODSSession
import os
import sys
import time

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUTDIR = BASE / "results"
BQTL = DATA / "processed" / "bqtl_snp_ww.csv"
TMPDIR = OUTDIR / "tmp_vcf"
TMPDIR.mkdir(exist_ok=True)

# Output
GENOTYPE_OUT = DATA / "processed" / "nam_founder_genotypes_at_bqtl.tsv"

# iRODS path
IRODS_BASE = "/iplant/home/shared/Grzybowski_MaizeSNPset_2022/imputed"

# NAM founder sample names in the VCF (verified from header)
NAM_FOUNDERS = {
    'A188': 'A188', 'A619': 'A619', 'B73': 'B73', 'B97': 'B97',
    'CML247': 'CML_247', 'CML277': 'CML_277', 'CML322': 'CML_322',
    'CML333': 'CML333', 'CML69': 'CML69',
    'HP301': 'HP301', 'IL14H': 'Il14H', 'Ki11': 'Ki11', 'Ki3': 'Ki3',
    'Ky21': 'Ky21', 'M162W': 'M162W', 'Mo17': 'Mo17', 'Mo18W': 'MO18W',
    'Ms71': 'MS71', 'NC358': 'NC358', 'Oh43': 'Oh43', 'Oh7b': 'Oh7B',
    'P39': 'P39', 'Tx303': 'Tx303', 'W22': 'W22',
    # CML103 not in this dataset
}


def download_irods_file(irods_path, local_path, session):
    """Download a file from iRODS to local path."""
    obj = session.data_objects.get(irods_path)
    file_size = obj.size
    print(f"  Downloading {irods_path} ({file_size/1e9:.2f} GB)...")

    chunk_size = 8 * 1024 * 1024  # 8MB chunks
    downloaded = 0
    start = time.time()

    with obj.open('r') as src, open(local_path, 'wb') as dst:
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break
            dst.write(chunk)
            downloaded += len(chunk)
            elapsed = time.time() - start
            speed = downloaded / elapsed / 1e6 if elapsed > 0 else 0
            pct = 100 * downloaded / file_size if file_size > 0 else 0
            print(f"\r    {pct:.1f}% ({downloaded/1e9:.2f}/{file_size/1e9:.2f} GB, {speed:.1f} MB/s)", end='', flush=True)
    print()
    return local_path


def extract_genotypes_at_positions(vcf_path, positions_bed, samples_str, output_path):
    """Use bcftools to extract genotypes at specific positions for specific samples."""
    cmd = [
        'bcftools', 'view',
        '-R', positions_bed,         # regions from BED file
        '-s', samples_str,           # subset samples
        '--force-samples',           # don't error on missing samples
        '-O', 'v',                   # output VCF text
        str(vcf_path)
    ]
    print(f"  Extracting genotypes with bcftools...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  bcftools error: {result.stderr[:500]}")
        return None

    # Parse VCF output
    rows = []
    sample_names = []
    for line in result.stdout.split('\n'):
        if line.startswith('#CHROM'):
            parts = line.strip().split('\t')
            sample_names = parts[9:]
        elif line and not line.startswith('#'):
            parts = line.strip().split('\t')
            if len(parts) < 10:
                continue
            chrom, pos, _, ref, alt = parts[0], int(parts[1]), parts[2], parts[3], parts[4]
            genotypes = parts[9:]
            row = {'chr': chrom, 'pos': pos, 'ref': ref, 'alt': alt}
            for sname, gt_field in zip(sample_names, genotypes):
                gt = gt_field.split(':')[0]  # GT is first field
                row[sname] = gt
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, sep='\t', index=False, mode='a',
                  header=not os.path.exists(output_path))
        print(f"  Extracted {len(df):,} variants")
        return df
    else:
        print(f"  No variants found at bQTL positions")
        return None


def main():
    print("=" * 70)
    print("DOWNLOAD NAM FOUNDER GENOTYPES AT bQTL POSITIONS")
    print("=" * 70)

    # Load bQTL positions
    bqtl_df = pd.read_csv(BQTL)
    print(f"Total bQTL: {len(bqtl_df):,}")

    # Sample string for bcftools
    samples_str = ','.join(NAM_FOUNDERS.values())
    print(f"NAM founders to extract: {len(NAM_FOUNDERS)} (CML103 not in dataset)")

    # Remove old output
    if GENOTYPE_OUT.exists():
        GENOTYPE_OUT.unlink()

    # Connect to iRODS
    print("\nConnecting to CyVerse iRODS...")
    session = iRODSSession(
        host='data.cyverse.org', port=1247,
        user='anonymous', password='', zone='iplant'
    )

    total_variants = 0
    for chrom_num in range(1, 11):
        chrom = f"chr{chrom_num}"
        print(f"\n{'='*50}")
        print(f"CHROMOSOME {chrom_num}")
        print(f"{'='*50}")

        # bQTL on this chromosome
        chrom_bqtl = bqtl_df[bqtl_df['chr'] == chrom]
        print(f"  bQTL on {chrom}: {len(chrom_bqtl):,}")
        if len(chrom_bqtl) == 0:
            continue

        # Write positions BED file for bcftools -R
        bed_path = TMPDIR / f"bqtl_{chrom}.bed"
        with open(bed_path, 'w') as f:
            for _, row in chrom_bqtl.iterrows():
                # BED is 0-based; bcftools -R with BED is 0-based half-open
                f.write(f"{chrom}\t{row['pos']-1}\t{row['pos']}\n")

        # Download VCF + tabix index
        vcf_name = f"chr_{chrom_num}_imputed.vcf.gz"
        tbi_name = f"chr_{chrom_num}_imputed.vcf.gz.tbi"
        vcf_local = TMPDIR / vcf_name
        tbi_local = TMPDIR / tbi_name

        try:
            if not vcf_local.exists():
                download_irods_file(f"{IRODS_BASE}/{vcf_name}", vcf_local, session)
            if not tbi_local.exists():
                download_irods_file(f"{IRODS_BASE}/{tbi_name}", tbi_local, session)

            # Extract genotypes
            chrom_out = TMPDIR / f"genotypes_{chrom}.tsv"
            df = extract_genotypes_at_positions(
                vcf_local, bed_path, samples_str, GENOTYPE_OUT
            )
            if df is not None:
                total_variants += len(df)

            # Clean up VCF to save space (keep BED and output)
            print(f"  Cleaning up VCF ({vcf_local.stat().st_size/1e9:.2f} GB)...")
            vcf_local.unlink()
            tbi_local.unlink()

        except Exception as e:
            print(f"  ERROR on {chrom}: {e}")
            # Clean up on error too
            if vcf_local.exists():
                vcf_local.unlink()
            if tbi_local.exists():
                tbi_local.unlink()
            continue

    session.cleanup()

    # Clean up BED files
    for f in TMPDIR.glob("*.bed"):
        f.unlink()
    try:
        TMPDIR.rmdir()
    except:
        pass

    print(f"\n{'='*70}")
    print(f"DONE: {total_variants:,} variants extracted at bQTL positions")
    print(f"Output: {GENOTYPE_OUT}")
    print(f"{'='*70}")

    if GENOTYPE_OUT.exists():
        result = pd.read_csv(GENOTYPE_OUT, sep='\t')
        print(f"\nGenotype matrix: {len(result):,} variants x {len(NAM_FOUNDERS)} founders")

        # Quick stats
        n_bqtl = len(bqtl_df)
        coverage = len(result) / n_bqtl
        print(f"bQTL coverage: {len(result):,}/{n_bqtl:,} ({100*coverage:.1f}%)")

        # For each founder, count how many bQTL have the ALT allele
        print(f"\nPer-founder variant counts:")
        for our_name, vcf_name in sorted(NAM_FOUNDERS.items()):
            if vcf_name in result.columns:
                gt_col = result[vcf_name]
                n_het = ((gt_col == '0/1') | (gt_col == '0|1')).sum()
                n_hom_alt = ((gt_col == '1/1') | (gt_col == '1|1')).sum()
                n_variant = n_het + n_hom_alt
                print(f"  {our_name:8s} ({vcf_name:8s}): {n_variant:6,} variant ({100*n_variant/len(result):.1f}%)")


if __name__ == '__main__':
    main()
