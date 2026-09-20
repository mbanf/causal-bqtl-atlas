#!/usr/bin/env python3
"""
Genotype-aware functional bQTL analysis.

The correct test for whether a bQTL is functional:
  For each bQTL position, group hybrids by GENOTYPE (variant vs reference),
  then compare ASE at the target gene between the two groups.

This is fundamentally different from the peak-presence test in script 50:
  - Peak presence = "is there TF binding?" (noisy, confounded)
  - Genotype = "does this hybrid carry the causal variant?" (deterministic)

Phase 1: Use Ensembl REST API to get allele frequencies at bQTL positions.
         This tells us how many hybrids are expected to be informative.
Phase 2: When per-hybrid genotype matrix is available (from Hartwig),
         plug into the framework and run the actual test.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import defaultdict
import time
import json
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUTDIR = BASE / "results"
FIG_DIR = BASE / "figures" / "paper_figures"
FIG_DIR.mkdir(exist_ok=True)

BQTL = DATA / "processed" / "bqtl_snp_ww.csv"
ASE_HYBRID = DATA / "processed" / "engelhorn_ase_ww.csv"
GFF3 = DATA / "raw" / "Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3"

VALID_CHROMS = {f'chr{i}' for i in range(1, 11)}
PROMOTER_BP = 2000
HYBRIDS = ['A188', 'A619', 'B97', 'CML103', 'CML247', 'CML277', 'CML322',
           'CML333', 'CML69', 'HP301', 'IL14H', 'Ki11', 'Ki3', 'Ky21',
           'M162W', 'Mo17', 'Mo18W', 'Ms71', 'NC358', 'Oh43', 'Oh7b',
           'P39', 'Tx303', 'W22']

# Cache for Ensembl API results
ENSEMBL_CACHE = OUTDIR / "ensembl_variant_cache.json"


def parse_gff3():
    genes = {}
    with open(GFF3) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'gene':
                continue
            chrom = parts[0]
            if chrom not in VALID_CHROMS:
                continue
            start, end = int(parts[3]), int(parts[4])
            strand = parts[6]
            gene_id = None
            for attr in parts[8].split(';'):
                if attr.startswith('ID=gene:'):
                    gene_id = attr.split(':')[1]
                    break
                elif attr.startswith('ID=Zm'):
                    gene_id = attr[3:]
                    break
            if gene_id:
                tss = start if strand == '+' else end
                genes[gene_id] = {'chr': chrom, 'start': start, 'end': end,
                                  'strand': strand, 'tss': tss}
    return genes


# ============================================================
# PHASE 1: Ensembl REST API — allele frequencies at bQTL positions
# ============================================================

def query_ensembl_variants(positions, batch_size=200, max_queries=500):
    """Query Ensembl REST API for variant info at bQTL positions.

    Uses the POST /overlap/region endpoint for batch queries.
    Rate-limited to ~15 requests/second per Ensembl guidelines.

    Args:
        positions: list of (chr, pos) tuples
        batch_size: region size for each query (bp)
        max_queries: max number of API calls
    Returns:
        dict: {(chr,pos): variant_info} for positions with known variants
    """
    import requests

    # Load cache
    cache = {}
    if ENSEMBL_CACHE.exists():
        with open(ENSEMBL_CACHE) as f:
            raw = json.load(f)
            cache = {tuple(k.split(',')): v for k, v in raw.items()}
        print(f"  Loaded {len(cache):,} cached variants")

    # Find positions not in cache
    uncached = [(c, p) for c, p in positions if (c, str(p)) not in cache]
    print(f"  {len(uncached):,} positions to query ({len(cache):,} cached)")

    if not uncached:
        return cache

    # Group by chromosome and sort
    by_chr = defaultdict(list)
    for c, p in uncached:
        by_chr[c].append(p)
    for c in by_chr:
        by_chr[c].sort()

    # Create non-overlapping query windows
    windows = []
    for chrom in sorted(by_chr.keys()):
        positions_sorted = by_chr[chrom]
        chr_num = chrom.replace('chr', '')

        i = 0
        while i < len(positions_sorted):
            start = positions_sorted[i]
            end = start
            # Extend window to include nearby positions
            while i + 1 < len(positions_sorted) and positions_sorted[i + 1] - start < 50000:
                i += 1
                end = positions_sorted[i]
            windows.append((chr_num, start, end))
            i += 1

    print(f"  {len(windows):,} query windows across {len(by_chr)} chromosomes")

    if len(windows) > max_queries:
        print(f"  Limiting to {max_queries} queries (sampling uniformly)")
        step = len(windows) // max_queries
        windows = windows[::step][:max_queries]

    # Query Ensembl
    base_url = "https://rest.ensembl.org"
    n_found = 0
    n_queried = 0

    for wi, (chr_num, start, end) in enumerate(windows):
        region = f"{chr_num}:{start}-{end}"
        url = f"{base_url}/overlap/region/zea_mays/{region}?feature=variation;content-type=application/json"

        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 429:
                # Rate limited — wait and retry
                wait = float(r.headers.get('Retry-After', 5))
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                r = requests.get(url, timeout=10)

            if r.status_code == 200:
                variants = r.json()
                for v in variants:
                    key = (f"chr{chr_num}", str(v['start']))
                    cache[key] = {
                        'id': v.get('id', ''),
                        'alleles': v.get('alleles', []),
                        'source': v.get('source', ''),
                    }
                    n_found += 1
                n_queried += 1
            else:
                n_queried += 1

        except Exception as e:
            n_queried += 1

        # Rate limiting: max 15/s
        if (wi + 1) % 14 == 0:
            time.sleep(1.1)

        if (wi + 1) % 100 == 0:
            print(f"  Queried {wi+1}/{len(windows)}, found {n_found:,} variants")

    print(f"  Done: {n_queried} queries, {n_found:,} new variants")

    # Save cache
    cache_out = {f"{k[0]},{k[1]}": v for k, v in cache.items()}
    with open(ENSEMBL_CACHE, 'w') as f:
        json.dump(cache_out, f)
    print(f"  Cache saved: {len(cache):,} total variants")

    return cache


def query_variant_genotypes(variant_ids, batch_size=50):
    """Query Ensembl for population genotype frequencies of specific variants.

    Args:
        variant_ids: list of Ensembl variant IDs (e.g., PZE0100427148)
    Returns:
        dict: {variant_id: {'maf': float, 'alleles': str, 'pop_geno': [...]}}
    """
    import requests

    results = {}
    for i in range(0, len(variant_ids), batch_size):
        batch = variant_ids[i:i + batch_size]

        # Use POST endpoint for batch lookup
        url = "https://rest.ensembl.org/variation/zea_mays"
        payload = {"ids": batch}
        headers = {"Content-Type": "application/json",
                    "Accept": "application/json"}

        try:
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                data = r.json()
                for vid, info in data.items():
                    maf = info.get('MAF')
                    minor = info.get('minor_allele', '')
                    pop_genos = info.get('population_genotypes', [])
                    results[vid] = {
                        'maf': maf,
                        'minor_allele': minor,
                        'pop_genotypes': pop_genos,
                    }
            elif r.status_code == 429:
                wait = float(r.headers.get('Retry-After', 5))
                time.sleep(wait)
                # retry
                r = requests.post(url, json=payload, headers=headers, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    for vid, info in data.items():
                        results[vid] = {
                            'maf': info.get('MAF'),
                            'minor_allele': info.get('minor_allele', ''),
                            'pop_genotypes': info.get('population_genotypes', []),
                        }
        except Exception as e:
            pass

        if (i + batch_size) % 200 == 0:
            time.sleep(1.1)

    return results


# ============================================================
# PHASE 2: Framework for genotype-aware test (uses genotype matrix)
# ============================================================

def run_genotype_test(genotype_matrix, bqtl_gene_df, ase_signed):
    """Run the genotype-aware functional bQTL test.

    For each bQTL-gene pair:
      1. Split hybrids into VARIANT (alt allele) vs REFERENCE (ref allele)
      2. Compare signed ASE (log2 B73/NAM) between the two groups
      3. Test for significance

    Args:
        genotype_matrix: dict {(chr, pos): {hybrid: 0/1}} where 1=variant
        bqtl_gene_df: DataFrame with columns [pos, chr, gene_id, dist_to_tss]
        ase_signed: dict {(gene_id, hybrid): signed_log2_ratio}
    Returns:
        DataFrame with test results per bQTL-gene pair
    """
    from statsmodels.stats.multitest import multipletests

    results = []
    for _, row in bqtl_gene_df.iterrows():
        chrom, pos, gene_id = row['chr'], row['pos'], row['gene_id']
        geno = genotype_matrix.get((chrom, pos), {})

        if not geno:
            continue

        # Split hybrids by genotype
        ase_variant = []   # hybrids with NAM variant
        ase_reference = [] # hybrids with B73-like allele

        for hybrid in HYBRIDS:
            g = geno.get(hybrid)
            if g is None:
                continue  # missing genotype
            ase = ase_signed.get((gene_id, hybrid))
            if ase is None or np.isnan(ase):
                continue

            if g == 1:
                ase_variant.append(ase)
            else:
                ase_reference.append(ase)

        if len(ase_variant) >= 3 and len(ase_reference) >= 3:
            # Mann-Whitney test (non-parametric)
            mw_stat, mw_p = stats.mannwhitneyu(ase_variant, ase_reference,
                                                alternative='two-sided')
            # t-test on signed values
            t_stat, t_p = stats.ttest_ind(ase_variant, ase_reference)

            mean_v = np.mean(ase_variant)
            mean_r = np.mean(ase_reference)
            pooled_std = np.sqrt(
                ((len(ase_variant)-1)*np.var(ase_variant, ddof=1) +
                 (len(ase_reference)-1)*np.var(ase_reference, ddof=1)) /
                (len(ase_variant) + len(ase_reference) - 2)
            )
            cohens_d = (mean_v - mean_r) / pooled_std if pooled_std > 0 else 0

            results.append({
                'pos': pos, 'chr': chrom, 'gene_id': gene_id,
                'dist_to_tss': row.get('dist_to_tss', np.nan),
                'n_variant': len(ase_variant),
                'n_reference': len(ase_reference),
                'mean_ase_variant': mean_v,
                'mean_ase_reference': mean_r,
                'ase_diff': mean_v - mean_r,
                'cohens_d': cohens_d,
                'mw_p': mw_p, 't_p': t_p,
            })

    rdf = pd.DataFrame(results)
    if len(rdf) > 0:
        _, rdf['fdr_mw'], _, _ = multipletests(rdf['mw_p'], method='fdr_bh')
        _, rdf['fdr_t'], _, _ = multipletests(rdf['t_p'], method='fdr_bh')
        rdf['functional'] = rdf['fdr_mw'] < 0.05

    return rdf


# ============================================================
# PHASE 1 execution: Ensembl allele frequency survey
# ============================================================

def phase1_ensembl_survey():
    """Survey bQTL positions for known variants and allele frequencies."""
    print("=" * 70)
    print("PHASE 1: ENSEMBL VARIANT SURVEY AT bQTL POSITIONS")
    print("=" * 70)

    bqtl_df = pd.read_csv(BQTL)
    print(f"Total bQTL: {len(bqtl_df):,}")

    # Sample 2000 bQTL across all chromosomes for the survey
    sample_size = min(2000, len(bqtl_df))
    sampled = bqtl_df.sample(n=sample_size, random_state=42)
    positions = list(zip(sampled['chr'], sampled['pos']))

    print(f"\nSurveying {sample_size:,} sampled bQTL positions...")
    variant_cache = query_ensembl_variants(positions, max_queries=500)

    # Check how many bQTL positions have known Ensembl variants
    n_found = 0
    n_with_maf = 0
    variant_ids = []

    for c, p in positions:
        key = (c, str(p))
        if key in variant_cache:
            n_found += 1
            vid = variant_cache[key].get('id', '')
            if vid:
                variant_ids.append(vid)

    print(f"\n  bQTL positions with known Ensembl variants: {n_found}/{sample_size} "
          f"({100*n_found/sample_size:.1f}%)")

    # Query allele frequencies for found variants
    if variant_ids:
        print(f"\n  Querying MAF for {len(variant_ids)} variants...")
        geno_results = query_variant_genotypes(variant_ids[:200])

        mafs = []
        for vid, info in geno_results.items():
            if info['maf'] is not None:
                mafs.append(info['maf'])

        if mafs:
            mafs = np.array(mafs)
            print(f"\n  MAF distribution ({len(mafs)} variants):")
            print(f"    Mean: {mafs.mean():.4f}")
            print(f"    Median: {np.median(mafs):.4f}")
            print(f"    Q25: {np.percentile(mafs, 25):.4f}")
            print(f"    Q75: {np.percentile(mafs, 75):.4f}")

            # Estimate informative hybrids per bQTL
            # In 25 hybrids, if MAF=p, expected variant carriers ≈ 25*p
            expected_carriers = 25 * mafs
            print(f"\n  Expected variant-carrying hybrids (out of 25):")
            print(f"    Mean: {expected_carriers.mean():.1f}")
            print(f"    Median: {np.median(expected_carriers):.1f}")
            print(f"    <3 carriers: {(expected_carriers < 3).sum()} "
                  f"({100*(expected_carriers<3).mean():.1f}%)")
            print(f"    3-10 carriers: {((expected_carriers>=3)&(expected_carriers<=10)).sum()} "
                  f"({100*((expected_carriers>=3)&(expected_carriers<=10)).mean():.1f}%)")
            print(f"    >10 carriers: {(expected_carriers > 10).sum()} "
                  f"({100*(expected_carriers>10).mean():.1f}%)")

            # Power analysis
            print(f"\n  Power implications:")
            for n_carriers in [3, 5, 8, 12]:
                n_ref = 25 - n_carriers
                # For effect size d=0.5 (moderate), what's the power?
                # Using approximation: power ≈ 1 - beta
                # t-test with n1=n_carriers, n2=n_ref, d=0.5
                from scipy.stats import norm
                se = np.sqrt(1/n_carriers + 1/n_ref)
                ncp = 0.5 / se  # non-centrality parameter for d=0.5
                alpha = 0.05
                z_crit = norm.ppf(1 - alpha/2)
                power = 1 - norm.cdf(z_crit - ncp) + norm.cdf(-z_crit - ncp)
                print(f"    {n_carriers} variant / {n_ref} ref: "
                      f"power={power:.3f} for d=0.5")

            return mafs, geno_results

    return None, None


def phase1_figures(mafs):
    """Generate figures for Phase 1 survey."""
    if mafs is None:
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # A: MAF distribution
    axes[0].hist(mafs, bins=30, color='#1976D2', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Minor Allele Frequency')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'A. MAF at bQTL positions\n(n={len(mafs)})')
    axes[0].axvline(np.median(mafs), color='red', linestyle='--',
                     label=f'median={np.median(mafs):.3f}')
    axes[0].legend(fontsize=8)

    # B: Expected informative hybrids
    expected = 25 * mafs
    axes[1].hist(expected, bins=range(0, 26), color='#FF7043',
                  edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('Expected variant-carrying hybrids (of 25)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('B. Expected informative hybrids per bQTL')
    axes[1].axvline(3, color='red', linestyle='--', alpha=0.7,
                     label='minimum for test (n=3)')
    axes[1].legend(fontsize=8)

    # C: Power curve
    from scipy.stats import norm
    n_carriers_range = np.arange(2, 15)
    for d in [0.3, 0.5, 0.8, 1.0]:
        powers = []
        for nc in n_carriers_range:
            nr = 25 - nc
            se = np.sqrt(1/nc + 1/nr)
            ncp = d / se
            z_crit = norm.ppf(0.975)
            power = 1 - norm.cdf(z_crit - ncp) + norm.cdf(-z_crit - ncp)
            powers.append(power)
        axes[2].plot(n_carriers_range, powers, 'o-', label=f'd={d}', markersize=4)
    axes[2].axhline(0.8, color='gray', linestyle=':', alpha=0.5)
    axes[2].set_xlabel('Number of variant-carrying hybrids')
    axes[2].set_ylabel('Statistical power')
    axes[2].set_title('C. Power to detect bQTL→ASE effect')
    axes[2].legend(fontsize=7)
    axes[2].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig10_genotype_bqtl_survey.pdf', bbox_inches='tight', dpi=150)
    plt.savefig(FIG_DIR / 'fig10_genotype_bqtl_survey.png', bbox_inches='tight', dpi=150)
    print(f"\nSaved: figures/fig10_genotype_bqtl_survey.pdf")
    plt.close()


def main():
    # Phase 1: Ensembl survey
    mafs, geno_results = phase1_ensembl_survey()
    phase1_figures(mafs)

    # Phase 2: Framework ready for when genotype matrix is available
    print(f"\n{'='*70}")
    print("PHASE 2: GENOTYPE-AWARE TEST FRAMEWORK")
    print(f"{'='*70}")
    print("""
The genotype-aware test framework is ready (run_genotype_test()).
It requires a genotype matrix: {(chr, pos): {hybrid: 0/1}}

To obtain this matrix, either:
  1. Ask Thomas Hartwig for per-hybrid SNP positions at bQTL loci
  2. Download MaizeGDB unified VCF (235 GB) from:
     https://ars-usda.app.box.com/v/maizegdb-public/folder/255390517505
     Then extract NAM founder genotypes at 148K bQTL positions using bcftools
  3. Use SNPversity 2.1 (https://wgs.maizegdb.org/) manually for targeted regions

Once the genotype matrix is available, save as JSON:
  {
    "chr1,457532": {"A188": 1, "B97": 0, "Mo17": 1, ...},
    "chr1,457918": {"A188": 0, "B97": 1, "Mo17": 0, ...},
    ...
  }
and load with:
  with open('bqtl_genotype_matrix.json') as f:
      raw = json.load(f)
  genotype_matrix = {tuple(k.split(',')): v for k,v in raw.items()}
""")


if __name__ == '__main__':
    main()
