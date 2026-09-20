#!/usr/bin/env python3
"""
53b_comprehensive_genotypes.py — Extract NAM founder genotypes at ALL bQTL positions.

Part of: "Condition-dependent bQTL switching" paper (Banf & Hartwig)
Paper section: Methods ("NAM founder genotypes")
Pipeline step: 1 of 8 (53b -> 54 -> 55 -> 56 -> 57 -> 58 -> 59 -> 60)

Improvement over script 53:
  Script 53 uses Grzybowski 2023 IMPUTED VCFs (~46M markers) -> 29.7% coverage.
  This script maps short windows around each bQTL from B73 to founder assemblies,
  extracting alleles at ALL bQTL positions -> ~78% coverage per founder.

Strategy (targeted window-mapping):
  For each NAM founder:
    1. Download the founder genome assembly (~630 MB compressed)
    2. Decompress and index with samtools faidx
    3. For each chromosome:
       a. Extract 201bp windows (+-100bp) around each bQTL from B73
       b. Map windows to founder chromosome with minimap2 -cx sr --cs
       c. Parse cs short tags to determine founder allele at each bQTL position
    4. Delete assembly files

Runtime: ~4-5 hours (download+decompress ~10 min/founder, mapping ~2 min/founder)
Requires: minimap2, samtools, pyfaidx, internet access (MaizeGDB)

Input:
  data/raw/B73_NAM5.fa — B73 reference genome (NAM5, with .fai index)
  data/processed/bqtl_snp_ww.csv — 147,942 bQTL positions

Output:
  data/processed/nam_founder_genotypes_at_bqtl.tsv — genotype matrix
    Columns: chr, pos, ref, <founder1>, <founder2>, ...
"""

import subprocess
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import os
import sys
import time
import gzip
import shutil
import tempfile

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RAW = DATA / "raw"
BQTL = DATA / "processed" / "bqtl_snp_ww.csv"
GENOTYPE_OUT = DATA / "processed" / "nam_founder_genotypes_at_bqtl.tsv"
B73_REF = RAW / "B73_NAM5.fa"
TMPDIR = Path(__file__).resolve().parent / "tmp_asm"

# MaizeGDB download URL template
MAIZEGDB_URL = "https://download.maizegdb.org/Zm-{name}-REFERENCE-NAM-1.0/Zm-{name}-REFERENCE-NAM-1.0.fa.gz"

# Mapping window: extract +-FLANK bp around each bQTL position
FLANK = 100

# minimap2 mapping quality threshold
MIN_MAPQ = 30

# NAM founders: our_name -> (MaizeGDB assembly name, Grzybowski VCF name)
NAM_FOUNDERS = {
    'B97':    ('B97',    'B97'),
    'CML103': ('CML103', None),
    'CML228': ('CML228', None),
    'CML247': ('CML247', 'CML_247'),
    'CML277': ('CML277', 'CML_277'),
    'CML322': ('CML322', 'CML_322'),
    'CML333': ('CML333', 'CML333'),
    'CML52':  ('CML52',  None),
    'CML69':  ('CML69',  'CML69'),
    'HP301':  ('HP301',  'HP301'),
    'Il14H':  ('Il14H',  'Il14H'),
    'Ki11':   ('Ki11',   'Ki11'),
    'Ki3':    ('Ki3',    'Ki3'),
    'Ky21':   ('Ky21',   'Ky21'),
    'M162W':  ('M162W',  'M162W'),
    'M37W':   ('M37W',   None),
    'Mo18W':  ('Mo18W',  'MO18W'),
    'Ms71':   ('Ms71',   'MS71'),
    'NC350':  ('NC350',  None),
    'NC358':  ('NC358',  'NC358'),
    'Oh43':   ('Oh43',   'Oh43'),
    'Oh7B':   ('Oh7B',   'Oh7B'),
    'P39':    ('P39',    'P39'),
    'Tx303':  ('Tx303',  'Tx303'),
    'Tzi8':   ('Tzi8',   None),
}

# Founders used in Engelhorn 2025 hybrids (B73 x founder)
ENGELHORN_FOUNDERS = [
    'B97', 'CML247', 'CML277', 'CML322', 'CML333', 'CML69',
    'HP301', 'Il14H', 'Ki11', 'Ki3', 'Ky21', 'M162W',
    'Mo18W', 'Ms71', 'NC358', 'Oh43', 'Oh7B', 'P39', 'Tx303',
]

# Chromosomes in B73 NAM5
CHROMOSOMES = [f'chr{i}' for i in range(1, 11)]


def download_assembly(founder_name, dest_path):
    """Download a NAM founder assembly from MaizeGDB."""
    url = MAIZEGDB_URL.format(name=founder_name)
    print(f"  Downloading {founder_name} assembly...")
    print(f"  URL: {url}")
    result = subprocess.run(
        ['curl', '-#', '-L', '-o', str(dest_path), url],
        capture_output=False, timeout=1800
    )
    if result.returncode != 0 or not dest_path.exists():
        raise RuntimeError(f"Download failed for {founder_name}")
    size_gb = dest_path.stat().st_size / 1e9
    print(f"  Downloaded: {size_gb:.2f} GB")


def decompress_assembly(gz_path, fa_path):
    """Decompress a gzipped FASTA file."""
    print(f"  Decompressing...")
    with gzip.open(gz_path, 'rb') as f_in, open(fa_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    size_gb = fa_path.stat().st_size / 1e9
    print(f"  Decompressed: {size_gb:.2f} GB")


def index_fasta(fa_path):
    """Create samtools faidx index for a FASTA file."""
    fai_path = Path(str(fa_path) + '.fai')
    if not fai_path.exists():
        print(f"  Indexing {fa_path.name}...")
        subprocess.run(['samtools', 'faidx', str(fa_path)], check=True,
                        capture_output=True)
    return fai_path


def get_chrom_lengths(fai_path):
    """Parse .fai to get chromosome lengths."""
    lengths = {}
    with open(fai_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            lengths[parts[0]] = int(parts[1])
    return lengths


def extract_windows(b73_fa, chrom, positions, out_fa):
    """Extract 201bp windows around bQTL positions from B73 reference.

    Window header encodes: >{chrom}_{pos}_{var_offset}
    where var_offset is the 0-based position of the variant within the window.
    """
    from pyfaidx import Fasta
    ref = Fasta(str(b73_fa))
    chrom_len = len(ref[chrom])

    n_written = 0
    with open(out_fa, 'w') as f:
        for pos in positions:
            start = max(0, pos - FLANK - 1)       # 0-based inclusive
            end = min(chrom_len, pos + FLANK)       # 0-based exclusive
            seq = str(ref[chrom][start:end])
            var_offset = pos - 1 - start            # 0-based offset within window
            f.write(f">{chrom}_{pos}_{var_offset}\n{seq}\n")
            n_written += 1

    ref.close()
    return n_written


def extract_founder_chrom(founder_fa, chrom, out_fa):
    """Extract a single chromosome from founder assembly using samtools faidx."""
    result = subprocess.run(
        ['samtools', 'faidx', str(founder_fa), chrom],
        capture_output=True, timeout=120
    )
    if result.returncode != 0:
        return False
    with open(out_fa, 'wb') as f:
        f.write(result.stdout)
    # Index the extracted chromosome
    subprocess.run(['samtools', 'faidx', str(out_fa)], capture_output=True)
    return True


def run_minimap2_windows(founder_chrom_fa, windows_fa):
    """Map bQTL windows to founder chromosome with minimap2 short-read mode.

    Returns PAF output as string.
    """
    cmd = [
        'minimap2',
        '-cx', 'sr',        # short-read mapping (201bp windows)
        '--cs',              # short cs tag
        '-t', '4',           # threads
        '--secondary=no',    # no secondary alignments
        str(founder_chrom_fa),
        str(windows_fa),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        stderr = result.stderr.decode()[:500]
        print(f"    minimap2 warning: {stderr}")
        return ""
    return result.stdout.decode()


def parse_paf_cs(paf_text):
    """Parse minimap2 PAF+cs output to extract founder alleles at bQTL positions.

    Each query name encodes: {chrom}_{pos}_{var_offset}
    The cs tag tells us what the founder allele is at that offset.

    For substitutions in cs short format: *XY where X=ref(founder), Y=query(B73)
    So the founder allele is X (group 3), B73 allele is Y (group 4).

    Returns: dict of {(chrom, pos): founder_allele}
    """
    cs_pattern = re.compile(r'(:(\d+)|\*([a-z])([a-z])|\+([a-z]+)|-([a-z]+))')
    results = {}

    for line in paf_text.strip().split('\n'):
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 12:
            continue

        # Parse query name: {chrom}_{pos}_{var_offset}
        qname = parts[0]
        fields = qname.rsplit('_', 2)
        if len(fields) != 3:
            continue
        chrom, pos_str, offset_str = fields
        pos = int(pos_str)
        var_offset = int(offset_str)

        # Check mapping quality
        mapq = int(parts[11])
        if mapq < MIN_MAPQ:
            continue

        # Query (B73 window) start position — var_offset is relative to full window
        qstart = int(parts[2])

        # Find cs tag
        cs_tag = None
        for tag in parts[12:]:
            if tag.startswith('cs:Z:'):
                cs_tag = tag[5:]
                break
        if cs_tag is None:
            continue

        # Walk through cs tag tracking query position
        qpos = qstart  # 0-based position in query (B73 window)
        found = False

        for match in cs_pattern.finditer(cs_tag):
            if found:
                break

            if match.group(2) is not None:
                # :N — N identical bases
                n = int(match.group(2))
                if qpos <= var_offset < qpos + n:
                    # Position is identical — founder has same allele as B73
                    results[(chrom, pos)] = 'REF'
                    found = True
                qpos += n

            elif match.group(3) is not None:
                # *XY — substitution, X=ref(founder), Y=query(B73)
                if qpos == var_offset:
                    founder_allele = match.group(3).upper()
                    results[(chrom, pos)] = founder_allele
                    found = True
                qpos += 1

            elif match.group(5) is not None:
                # +seq — insertion in query (extra bases in B73 window)
                ins_len = len(match.group(5))
                if qpos <= var_offset < qpos + ins_len:
                    results[(chrom, pos)] = 'DEL'
                    found = True
                qpos += ins_len

            elif match.group(6) is not None:
                # -seq — deletion from reference (bases in founder, not in B73 window)
                # No query position change
                pass

    return results


def process_founder(founder, asm_name, bqtl_by_chrom, b73_ref, tmpdir):
    """Process one founder: download, decompress, map per-chromosome, extract alleles.

    Returns: dict of {(chrom, pos): allele_info}
    """
    gz_path = tmpdir / f"{asm_name}.fa.gz"
    fa_path = tmpdir / f"{asm_name}.fa"

    # Step 1: Download assembly
    if not gz_path.exists():
        download_assembly(asm_name, gz_path)
    else:
        print(f"  Using cached assembly: {gz_path.name}")

    # Step 2: Decompress
    if not fa_path.exists():
        decompress_assembly(gz_path, fa_path)
    else:
        print(f"  Using cached decompressed: {fa_path.name}")

    # Step 3: Index
    fai_path = index_fasta(fa_path)
    founder_chroms = get_chrom_lengths(fai_path)

    # Step 4: Per-chromosome window mapping
    all_results = {}
    total_mapped = 0
    total_bqtl = 0

    for chrom in CHROMOSOMES:
        if chrom not in bqtl_by_chrom:
            continue
        positions = bqtl_by_chrom[chrom]
        total_bqtl += len(positions)

        if chrom not in founder_chroms:
            print(f"    {chrom}: not in founder assembly, skipping")
            continue

        # Create temp files for this chromosome
        windows_fa = tmpdir / f"windows_{asm_name}_{chrom}.fa"
        founder_chrom_fa = tmpdir / f"{asm_name}_{chrom}.fa"

        try:
            # Extract B73 windows
            n_windows = extract_windows(b73_ref, chrom, positions, windows_fa)

            # Extract founder chromosome
            if not extract_founder_chrom(fa_path, chrom, founder_chrom_fa):
                print(f"    {chrom}: extraction failed, skipping")
                continue

            # Map windows to founder chromosome
            paf_text = run_minimap2_windows(founder_chrom_fa, windows_fa)

            # Parse results
            chrom_results = parse_paf_cs(paf_text)
            all_results.update(chrom_results)
            total_mapped += len(chrom_results)

            n_snp = sum(1 for v in chrom_results.values() if v not in ('REF', 'DEL'))
            n_del = sum(1 for v in chrom_results.values() if v == 'DEL')
            n_ref = sum(1 for v in chrom_results.values() if v == 'REF')
            print(f"    {chrom}: {len(chrom_results):,}/{len(positions):,} mapped "
                  f"(REF={n_ref:,}, SNP={n_snp:,}, DEL={n_del:,})")

        finally:
            # Clean up per-chromosome temp files
            for p in [windows_fa, founder_chrom_fa,
                      Path(str(founder_chrom_fa) + '.fai')]:
                if p.exists():
                    p.unlink()

    print(f"  Total: {total_mapped:,}/{total_bqtl:,} ({100*total_mapped/total_bqtl:.1f}%)")
    return all_results


def get_b73_alleles(b73_ref, positions_by_chrom):
    """Get B73 reference alleles at all bQTL positions."""
    from pyfaidx import Fasta
    ref = Fasta(str(b73_ref))
    b73_alleles = {}
    for chrom, positions in positions_by_chrom.items():
        for pos in positions:
            base = str(ref[chrom][pos - 1:pos])  # 1-based to 0-based
            b73_alleles[(chrom, pos)] = base.upper()
    ref.close()
    return b73_alleles


def main():
    print("=" * 70)
    print("COMPREHENSIVE NAM FOUNDER GENOTYPES VIA WINDOW MAPPING")
    print("Method: minimap2 short-read mapping of B73 windows to founder assemblies")
    print("=" * 70)

    # Verify prerequisites
    if not B73_REF.exists():
        print(f"ERROR: B73 reference not found: {B73_REF}")
        sys.exit(1)

    for tool in ['minimap2', 'samtools']:
        try:
            subprocess.run([tool, '--version'], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print(f"ERROR: {tool} not installed")
            sys.exit(1)

    # Load bQTL positions
    bqtl_df = pd.read_csv(BQTL)
    print(f"Total bQTL: {len(bqtl_df):,}")

    # Organize positions by chromosome (sorted)
    bqtl_by_chrom = {}
    for chrom, group in bqtl_df.groupby('chr'):
        bqtl_by_chrom[chrom] = sorted(group['pos'].tolist())
    for chrom in CHROMOSOMES:
        n = len(bqtl_by_chrom.get(chrom, []))
        print(f"  {chrom}: {n:,} bQTL")

    TMPDIR.mkdir(exist_ok=True)

    # Get B73 reference alleles at all positions
    print(f"\nExtracting B73 reference alleles...")
    b73_alleles = get_b73_alleles(B73_REF, bqtl_by_chrom)
    print(f"  Got {len(b73_alleles):,} reference alleles")

    # Process each founder
    all_genotypes = defaultdict(dict)  # {(chrom, pos): {founder: allele}}
    overall_start = time.time()

    founders_to_process = [f for f in ENGELHORN_FOUNDERS if f in NAM_FOUNDERS]
    print(f"\nProcessing {len(founders_to_process)} founders: {', '.join(founders_to_process)}")

    for i, founder in enumerate(founders_to_process):
        asm_name = NAM_FOUNDERS[founder][0]
        print(f"\n{'=' * 55}")
        print(f"FOUNDER {i+1}/{len(founders_to_process)}: {founder} ({asm_name})")
        print(f"{'=' * 55}")

        try:
            start = time.time()
            results = process_founder(founder, asm_name, bqtl_by_chrom,
                                       B73_REF, TMPDIR)
            elapsed = time.time() - start
            print(f"  Time: {elapsed:.0f}s")

            for (chrom, pos), allele in results.items():
                all_genotypes[(chrom, pos)][founder] = allele

            # Clean up large files (keep .gz cached)
            fa_path = TMPDIR / f"{asm_name}.fa"
            fai_path = TMPDIR / f"{asm_name}.fa.fai"
            for p in [fa_path, fai_path]:
                if p.exists():
                    p.unlink()
                    print(f"  Cleaned: {p.name}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            # Clean up on error
            for ext in ['.fa', '.fa.fai']:
                p = TMPDIR / f"{asm_name}{ext}"
                if p.exists():
                    p.unlink()
            continue

    # Build output table
    print(f"\n{'=' * 70}")
    print(f"BUILDING GENOTYPE TABLE")
    print(f"{'=' * 70}")

    rows = []
    for chrom in CHROMOSOMES:
        for pos in bqtl_by_chrom.get(chrom, []):
            ref_allele = b73_alleles.get((chrom, pos), 'N')
            row = {'chr': chrom, 'pos': pos, 'ref': ref_allele}

            for founder in founders_to_process:
                allele = all_genotypes.get((chrom, pos), {}).get(founder)
                if allele is None:
                    row[founder] = './.'       # not mapped
                elif allele == 'REF':
                    row[founder] = '0|0'       # same as B73
                elif allele == 'DEL':
                    row[founder] = 'DEL'       # deletion
                else:
                    row[founder] = allele      # actual alt allele (A/C/G/T)

            rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(GENOTYPE_OUT, sep='\t', index=False)

    elapsed_total = time.time() - overall_start

    # Summary stats
    print(f"\nGenotype matrix: {len(result_df):,} positions x {len(founders_to_process)} founders")
    print(f"Output: {GENOTYPE_OUT}")
    print(f"Total time: {elapsed_total/60:.1f} minutes")

    # Per-founder coverage
    print(f"\nPer-founder stats:")
    for founder in founders_to_process:
        col = result_df[founder]
        n_ref = (col == '0|0').sum()
        n_snp = col.isin(list('ACGT')).sum()
        n_del = (col == 'DEL').sum()
        n_miss = (col == './.').sum()
        n_covered = n_ref + n_snp + n_del
        pct = 100 * n_covered / len(result_df)
        print(f"  {founder:8s}: {n_covered:6,} covered ({pct:5.1f}%), "
              f"REF={n_ref:,}, SNP={n_snp:,}, DEL={n_del:,}, miss={n_miss:,}")

    # Overall coverage
    any_covered = 0
    for _, row in result_df.iterrows():
        for founder in founders_to_process:
            if row[founder] != './.':
                any_covered += 1
                break
    print(f"\nPositions covered by at least 1 founder: {any_covered:,}/{len(result_df):,} "
          f"({100*any_covered/len(result_df):.1f}%)")

    # Clean up compressed assemblies
    print(f"\nCleaning up cached assemblies...")
    for gz in TMPDIR.glob("*.fa.gz"):
        gz.unlink()
        print(f"  Removed: {gz.name}")
    try:
        TMPDIR.rmdir()
    except OSError:
        pass


if __name__ == '__main__':
    main()
