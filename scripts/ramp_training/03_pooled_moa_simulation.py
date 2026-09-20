#!/usr/bin/env python3
"""
03_pooled_moa_simulation.py — Simulate a pooled MOA-seq experiment.

Thomas's idea: Instead of running MOA on each hybrid separately, pool hundreds
of lines into ONE experiment. The allele frequency at each position within
MOA peaks will deviate from genome-wide frequency, revealing which allele
gives stronger binding.

Simulation:
  We have 19 hybrids with BOTH genotype data AND B73-mapped MOA peaks.
  For each bQTL position:
    1. Genome-wide variant allele frequency (AF_genome) = # carriers / # hybrids
    2. Peak allele frequency (AF_peak) = # carriers with peak / # with peak
    3. Signal-weighted AF = sum(signal × carrier) / sum(signal)
    4. Allele Frequency Deviation (AFD) = AF_peak - AF_genome

  If bQTL is functional: AFD ≠ 0 (variant enriched or depleted in binding)
  At non-bQTL positions: AFD ≈ 0 (no allele-specific effect)

  Power analysis: What fraction of known bQTLs show significant AFD?
  Saturation: How many hybrids needed (5, 10, 15, 19)?
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import bisect
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results"
OUTDIR = Path(__file__).resolve().parent
FIG_DIR = OUTDIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

PEAKS_DIR = DATA / "raw" / "engelhorn_peaks"

print("=" * 70)
print("POOLED MOA-seq SIMULATION")
print("=" * 70)

# ── Load genotype data ────────────────────────────────────────────────────────

print("\n── Loading genotype data ──")
geno = pd.read_csv(DATA / "processed" / "nam_founder_genotypes_at_bqtl.tsv", sep='\t')
founder_cols = [c for c in geno.columns if c not in ('chr', 'pos', 'ref')]
print(f"  {len(geno):,} bQTL positions, {len(founder_cols)} NAM founders")
print(f"  Founders: {', '.join(founder_cols)}")

# ── Identify hybrids with both genotype AND B73-mapped peaks ──────────────────

print("\n── Loading per-hybrid MOA peaks ──")

# Map founder names to peak file names (case variations)
peak_files = list(PEAKS_DIR.glob("*.w2_4.q3_peaks.narrowPeak"))
peak_name_map = {}
for pf in peak_files:
    name = pf.name.split('.')[0]  # e.g., "B97"
    # Check if B73-mapped
    with open(pf) as f:
        first_chr = f.readline().split('\t')[0]
    if first_chr.startswith('B73-'):
        peak_name_map[name] = pf

# Match founder names to peak names
founder_to_peak = {}
for fc in founder_cols:
    fc_lower = fc.lower()
    for pn, pf in peak_name_map.items():
        if pn.lower() == fc_lower:
            founder_to_peak[fc] = pf
            break

matched_founders = sorted(founder_to_peak.keys())
print(f"  {len(peak_name_map)} B73-mapped hybrid peak files")
print(f"  {len(matched_founders)} matched to genotype data")
print(f"  Matched: {', '.join(matched_founders)}")

# ── Build interval index per hybrid ───────────────────────────────────────────

print("\n── Building peak interval indices ──")

# Structure: hybrid_peaks[founder][chrom] = [(start, end, signal), ...]
hybrid_peaks = {}

for founder, peak_file in sorted(founder_to_peak.items()):
    peaks = defaultdict(list)
    with open(peak_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            # Convert B73-chr1 → chr1
            chrom = parts[0].replace('B73-', '')
            start = int(parts[1])
            end = int(parts[2])
            signal = float(parts[6])  # fold enrichment
            peaks[chrom].append((start, end, signal))

    # Sort by start for binary search
    for chrom in peaks:
        peaks[chrom].sort()

    hybrid_peaks[founder] = dict(peaks)
    n_peaks = sum(len(v) for v in peaks.values())
    print(f"  {founder:8s}: {n_peaks:>7,} peaks loaded")

# ── Peak lookup function ──────────────────────────────────────────────────────

def lookup_peak(chrom, pos, peak_list):
    """Check if position falls in a peak. Returns signal or 0."""
    if not peak_list:
        return 0.0
    # Binary search
    lo, hi = 0, len(peak_list) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end, signal = peak_list[mid]
        if pos < start:
            hi = mid - 1
        elif pos > end:
            lo = mid + 1
        else:
            return signal
    return 0.0

# ── Simulate pooled experiment at bQTL positions ──────────────────────────────

print("\n── Simulating pooled MOA at bQTL positions ──")

valid_chroms = {f'chr{i}' for i in range(1, 11)}
results = []
n_processed = 0

for _, row in geno.iterrows():
    chrom, pos = row['chr'], row['pos']
    if chrom not in valid_chroms:
        continue

    # For each founder: determine genotype and peak signal
    n_variant = 0       # total hybrids carrying variant allele
    n_ref = 0           # total hybrids carrying ref allele
    n_variant_peak = 0  # variant carriers WITH peak
    n_ref_peak = 0      # ref carriers WITH peak
    signal_variant = 0.0  # sum of signal in variant carriers
    signal_ref = 0.0      # sum of signal in ref carriers
    n_with_peak = 0
    n_total = 0
    signals = []        # all signals for this position

    for founder in matched_founders:
        gt = str(row[founder])
        # Determine if variant or reference
        if gt in ('0|0', '0/0'):
            is_variant = False
        elif gt in ('./.', '.|.', '.'):
            continue  # missing genotype
        else:
            is_variant = True  # has alt allele

        # Look up peak signal
        peak_list = hybrid_peaks[founder].get(chrom, [])
        signal = lookup_peak(chrom, pos, peak_list)
        has_peak = signal > 0

        n_total += 1
        if is_variant:
            n_variant += 1
            if has_peak:
                n_variant_peak += 1
                signal_variant += signal
        else:
            n_ref += 1
            if has_peak:
                n_ref_peak += 1
                signal_ref += signal

        if has_peak:
            n_with_peak += 1
            signals.append(signal)

    if n_total < 5 or n_variant == 0 or n_ref == 0:
        continue  # need both alleles represented

    # Allele frequencies
    af_genome = n_variant / n_total
    af_peak = n_variant_peak / n_with_peak if n_with_peak > 0 else np.nan

    # Signal-weighted allele frequency
    total_signal = signal_variant + signal_ref
    af_signal = signal_variant / total_signal if total_signal > 0 else np.nan

    # Allele frequency deviation
    afd = af_peak - af_genome if not np.isnan(af_peak) else np.nan
    afd_signal = af_signal - af_genome if not np.isnan(af_signal) else np.nan

    # Binding rate per allele
    bind_rate_variant = n_variant_peak / n_variant if n_variant > 0 else 0
    bind_rate_ref = n_ref_peak / n_ref if n_ref > 0 else 0
    bind_rate_diff = bind_rate_variant - bind_rate_ref

    results.append({
        'chr': chrom, 'pos': pos,
        'n_total': n_total, 'n_variant': n_variant, 'n_ref': n_ref,
        'n_with_peak': n_with_peak,
        'n_variant_peak': n_variant_peak, 'n_ref_peak': n_ref_peak,
        'af_genome': af_genome, 'af_peak': af_peak,
        'af_signal': af_signal,
        'afd': afd, 'afd_signal': afd_signal,
        'bind_rate_variant': bind_rate_variant,
        'bind_rate_ref': bind_rate_ref,
        'bind_rate_diff': bind_rate_diff,
        'mean_signal': np.mean(signals) if signals else 0,
        'signal_variant': signal_variant,
        'signal_ref': signal_ref,
    })

    n_processed += 1
    if n_processed % 20000 == 0:
        print(f"  Processed {n_processed:,} bQTLs...")

df = pd.DataFrame(results)
print(f"\n  Total processed: {len(df):,} bQTLs with valid genotype + peak data")

# ── Generate matched control positions ────────────────────────────────────────

print("\n── Generating control positions (random genomic) ──")

from pyfaidx import Fasta
genome = Fasta(str(DATA / "raw" / "B73_NAM5.fa"))
chrom_sizes = {name: len(genome[name]) for name in genome.keys()
               if name.startswith('chr') and name[3:].isdigit()}

np.random.seed(42)
n_controls = min(len(df), 50000)
controls = []
bqtl_set = set(zip(geno['chr'], geno['pos']))

n_attempts = 0
while len(controls) < n_controls and n_attempts < n_controls * 20:
    chrom = np.random.choice(list(valid_chroms))
    pos = np.random.randint(1000, chrom_sizes[chrom] - 1000)
    if (chrom, pos) in bqtl_set:
        n_attempts += 1
        continue

    n_variant_ctrl = 0
    n_ref_ctrl = 0
    n_variant_peak_ctrl = 0
    n_ref_peak_ctrl = 0
    signal_variant_ctrl = 0.0
    signal_ref_ctrl = 0.0
    n_with_peak_ctrl = 0
    n_total_ctrl = 0
    signals_ctrl = []

    for founder in matched_founders:
        # Simulate random genotype (match overall AF distribution)
        is_variant = np.random.random() < 0.3  # ~30% minor allele freq
        peak_list = hybrid_peaks[founder].get(chrom, [])
        signal = lookup_peak(chrom, pos, peak_list)
        has_peak = signal > 0

        n_total_ctrl += 1
        if is_variant:
            n_variant_ctrl += 1
            if has_peak:
                n_variant_peak_ctrl += 1
                signal_variant_ctrl += signal
        else:
            n_ref_ctrl += 1
            if has_peak:
                n_ref_peak_ctrl += 1
                signal_ref_ctrl += signal

        if has_peak:
            n_with_peak_ctrl += 1
            signals_ctrl.append(signal)

    if n_total_ctrl < 5 or n_variant_ctrl == 0 or n_ref_ctrl == 0:
        n_attempts += 1
        continue

    af_genome_ctrl = n_variant_ctrl / n_total_ctrl
    af_peak_ctrl = (n_variant_peak_ctrl / n_with_peak_ctrl
                    if n_with_peak_ctrl > 0 else np.nan)
    total_sig = signal_variant_ctrl + signal_ref_ctrl
    af_signal_ctrl = signal_variant_ctrl / total_sig if total_sig > 0 else np.nan

    controls.append({
        'chr': chrom, 'pos': pos,
        'n_with_peak': n_with_peak_ctrl,
        'af_genome': af_genome_ctrl,
        'af_peak': af_peak_ctrl,
        'af_signal': af_signal_ctrl,
        'afd': af_peak_ctrl - af_genome_ctrl if not np.isnan(af_peak_ctrl) else np.nan,
        'afd_signal': af_signal_ctrl - af_genome_ctrl if not np.isnan(af_signal_ctrl) else np.nan,
        'bind_rate_diff': (n_variant_peak_ctrl / n_variant_ctrl -
                           n_ref_peak_ctrl / n_ref_ctrl),
        'mean_signal': np.mean(signals_ctrl) if signals_ctrl else 0,
    })
    n_attempts += 1

ctrl_df = pd.DataFrame(controls)
print(f"  Generated {len(ctrl_df):,} control positions")

# ── Analysis ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

# Filter to positions where at least some hybrids have peaks
bqtl_in_peak = df[df['n_with_peak'] >= 3].copy()
ctrl_in_peak = ctrl_df[ctrl_df['n_with_peak'] >= 3].copy()
print(f"\nbQTLs with ≥3 hybrids having peak: {len(bqtl_in_peak):,} / {len(df):,}")
print(f"Controls with ≥3 hybrids having peak: {len(ctrl_in_peak):,} / {len(ctrl_df):,}")

# AFD statistics
from scipy import stats

print(f"\n── Allele Frequency Deviation (AFD) ──")
bqtl_afd = bqtl_in_peak['afd'].dropna()
ctrl_afd = ctrl_in_peak['afd'].dropna()

print(f"  bQTL AFD:    mean={bqtl_afd.mean():.4f}, median={bqtl_afd.median():.4f}, "
      f"std={bqtl_afd.std():.4f}")
print(f"  Control AFD: mean={ctrl_afd.mean():.4f}, median={ctrl_afd.median():.4f}, "
      f"std={ctrl_afd.std():.4f}")

# Test: is bQTL |AFD| larger than control |AFD|?
bqtl_abs_afd = bqtl_afd.abs()
ctrl_abs_afd = ctrl_afd.abs()
u_stat, u_p = stats.mannwhitneyu(bqtl_abs_afd, ctrl_abs_afd, alternative='greater')
print(f"\n  |AFD| bQTL vs control (Mann-Whitney U, one-sided):")
print(f"    bQTL |AFD| mean: {bqtl_abs_afd.mean():.4f}")
print(f"    Control |AFD| mean: {ctrl_abs_afd.mean():.4f}")
print(f"    U={u_stat:.0f}, p={u_p:.2e}")
print(f"    Enrichment: {bqtl_abs_afd.mean() / ctrl_abs_afd.mean():.2f}x")

# Binding rate difference
print(f"\n── Binding Rate Difference (variant - reference) ──")
bqtl_brd = bqtl_in_peak['bind_rate_diff'].dropna()
ctrl_brd = ctrl_in_peak['bind_rate_diff'].dropna()
print(f"  bQTL:    mean={bqtl_brd.mean():.4f}, |mean|={bqtl_brd.abs().mean():.4f}")
print(f"  Control: mean={ctrl_brd.mean():.4f}, |mean|={ctrl_brd.abs().mean():.4f}")

u2, p2 = stats.mannwhitneyu(bqtl_brd.abs(), ctrl_brd.abs(), alternative='greater')
print(f"  |BRD| Mann-Whitney: U={u2:.0f}, p={p2:.2e}")

# ── Detection power ───────────────────────────────────────────────────────────

print(f"\n── Detection Power ──")

# What fraction of bQTLs have |AFD| above the 95th percentile of controls?
if len(ctrl_afd) > 0:
    threshold_95 = ctrl_abs_afd.quantile(0.95)
    threshold_99 = ctrl_abs_afd.quantile(0.99)
    detected_95 = (bqtl_abs_afd > threshold_95).sum()
    detected_99 = (bqtl_abs_afd > threshold_99).sum()
    print(f"  Control |AFD| 95th percentile: {threshold_95:.4f}")
    print(f"  Control |AFD| 99th percentile: {threshold_99:.4f}")
    print(f"  bQTLs detected at 5% FDR:  {detected_95:,} / {len(bqtl_abs_afd):,} "
          f"({100*detected_95/len(bqtl_abs_afd):.1f}%)")
    print(f"  bQTLs detected at 1% FDR:  {detected_99:,} / {len(bqtl_abs_afd):,} "
          f"({100*detected_99/len(bqtl_abs_afd):.1f}%)")

# ── Saturation analysis ──────────────────────────────────────────────────────

print(f"\n── Saturation Analysis (sub-pool sizes) ──")

pool_sizes = [5, 8, 10, 13, 15, 19]
saturation_results = []

# Use a subset for speed
bqtl_sample = bqtl_in_peak.sample(min(10000, len(bqtl_in_peak)), random_state=42)

for n_pool in pool_sizes:
    # Subsample hybrids
    sub_founders = np.random.choice(matched_founders, size=n_pool, replace=False)

    abs_afds = []
    for _, row in bqtl_sample.iterrows():
        chrom, pos = row['chr'], row['pos']
        geno_row = geno[(geno['chr'] == chrom) & (geno['pos'] == pos)]
        if len(geno_row) == 0:
            continue
        geno_row = geno_row.iloc[0]

        n_var, n_ref_sub, n_var_pk, n_ref_pk = 0, 0, 0, 0
        for f in sub_founders:
            gt = str(geno_row.get(f, './.'))
            if gt in ('0|0', '0/0'):
                is_var = False
            elif gt in ('./.', '.|.', '.'):
                continue
            else:
                is_var = True

            signal = lookup_peak(chrom, pos,
                                  hybrid_peaks[f].get(chrom, []))
            has_pk = signal > 0

            if is_var:
                n_var += 1
                if has_pk:
                    n_var_pk += 1
            else:
                n_ref_sub += 1
                if has_pk:
                    n_ref_pk += 1

        n_total_sub = n_var + n_ref_sub
        n_pk_sub = n_var_pk + n_ref_pk
        if n_total_sub >= 3 and n_pk_sub >= 2 and n_var > 0 and n_ref_sub > 0:
            af_g = n_var / n_total_sub
            af_p = n_var_pk / n_pk_sub
            abs_afds.append(abs(af_p - af_g))

    mean_afd = np.mean(abs_afds) if abs_afds else 0
    # Detection at control 95th percentile
    det_rate = np.mean([a > threshold_95 for a in abs_afds]) if abs_afds else 0

    saturation_results.append({
        'n_pool': n_pool, 'n_testable': len(abs_afds),
        'mean_abs_afd': mean_afd, 'detection_rate_5pct': det_rate
    })
    print(f"  Pool={n_pool:2d}: testable={len(abs_afds):>5,}, "
          f"|AFD|={mean_afd:.4f}, detection@5%={100*det_rate:.1f}%")

sat_df = pd.DataFrame(saturation_results)

# ── Visualization ─────────────────────────────────────────────────────────────

print("\n── Generating figures ──")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# A: AFD distribution — bQTL vs control
ax = axes[0, 0]
bins = np.linspace(-0.6, 0.6, 80)
ax.hist(ctrl_afd, bins=bins, alpha=0.5, color='gray', label='Control', density=True)
ax.hist(bqtl_afd, bins=bins, alpha=0.6, color='#2E7D32', label='bQTL', density=True)
ax.axvline(0, color='black', ls=':', alpha=0.5)
ax.set_xlabel('Allele Frequency Deviation (AFD)')
ax.set_ylabel('Density')
ax.set_title('A. AFD distribution')
ax.legend()

# B: |AFD| cumulative
ax = axes[0, 1]
bqtl_sorted = np.sort(bqtl_abs_afd)
ctrl_sorted = np.sort(ctrl_abs_afd)
ax.plot(bqtl_sorted, np.linspace(1, 0, len(bqtl_sorted)),
        color='#2E7D32', label='bQTL', linewidth=2)
ax.plot(ctrl_sorted, np.linspace(1, 0, len(ctrl_sorted)),
        color='gray', label='Control', linewidth=2)
ax.axvline(threshold_95, color='red', ls='--', alpha=0.7,
           label=f'5% FDR ({threshold_95:.3f})')
ax.set_xlabel('|AFD| threshold')
ax.set_ylabel('Fraction above threshold')
ax.set_title('B. Detection power (survival curve)')
ax.legend(fontsize=9)
ax.set_xlim(0, 0.5)

# C: Binding rate by allele
ax = axes[0, 2]
ax.scatter(bqtl_in_peak['bind_rate_ref'], bqtl_in_peak['bind_rate_variant'],
           alpha=0.02, s=5, color='#2E7D32')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('Binding rate (reference allele)')
ax.set_ylabel('Binding rate (variant allele)')
ax.set_title('C. Per-allele binding rate')
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)

# D: Saturation curve
ax = axes[1, 0]
ax.plot(sat_df['n_pool'], sat_df['detection_rate_5pct'] * 100,
        'o-', color='#1565C0', linewidth=2, markersize=8)
ax.set_xlabel('Number of hybrids in pool')
ax.set_ylabel('Detection rate at 5% FDR (%)')
ax.set_title('D. Saturation: pool size vs detection')
ax.set_ylim(0, max(sat_df['detection_rate_5pct'] * 100) * 1.3)

# E: |AFD| vs number of hybrids with peak
ax = axes[1, 1]
ax.scatter(bqtl_in_peak['n_with_peak'], bqtl_in_peak['afd'].abs(),
           alpha=0.02, s=5, color='#E53935')
ax.set_xlabel('Number of hybrids with peak at position')
ax.set_ylabel('|AFD|')
ax.set_title('E. |AFD| vs peak support')

# F: Signal difference (variant vs ref)
ax = axes[1, 2]
bqtl_sig = bqtl_in_peak[(bqtl_in_peak['signal_variant'] > 0) &
                          (bqtl_in_peak['signal_ref'] > 0)].copy()
if len(bqtl_sig) > 0:
    log_ratio = np.log2((bqtl_sig['signal_variant'] / bqtl_sig['n_variant']) /
                         (bqtl_sig['signal_ref'] / bqtl_sig['n_ref']))
    log_ratio = log_ratio.replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(log_ratio.clip(-3, 3), bins=60, color='#7B1FA2', alpha=0.7,
            edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='black', ls=':', alpha=0.5)
    ax.set_xlabel('log2(signal variant / signal reference)')
    ax.set_ylabel('Count')
    ax.set_title('F. Signal ratio per allele')

plt.suptitle("Pooled MOA-seq Simulation\n"
             "Can a single pooled experiment detect bQTLs via allele frequency deviation?",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'pooled_moa_simulation.pdf', dpi=150, bbox_inches='tight')
plt.savefig(FIG_DIR / 'pooled_moa_simulation.png', dpi=150, bbox_inches='tight')
print(f"  Saved: {FIG_DIR / 'pooled_moa_simulation.pdf'}")

# ── Save results ──────────────────────────────────────────────────────────────

df.to_csv(OUTDIR / 'pooled_moa_bqtl_results.csv', index=False)
sat_df.to_csv(OUTDIR / 'pooled_moa_saturation.csv', index=False)

print(f"\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  bQTLs analyzed:         {len(df):>10,}")
print(f"  With peak support ≥3:   {len(bqtl_in_peak):>10,}")
print(f"  Mean |AFD| (bQTL):      {bqtl_abs_afd.mean():>10.4f}")
print(f"  Mean |AFD| (control):   {ctrl_abs_afd.mean():>10.4f}")
print(f"  Enrichment:             {bqtl_abs_afd.mean()/ctrl_abs_afd.mean():>10.2f}x")
print(f"  bQTL detection @ 5%:    {100*detected_95/len(bqtl_abs_afd):>10.1f}%")
print(f"  bQTL detection @ 1%:    {100*detected_99/len(bqtl_abs_afd):>10.1f}%")
print(f"\n  Conclusion: ", end="")
if bqtl_abs_afd.mean() / ctrl_abs_afd.mean() > 1.2 and u_p < 0.01:
    print("Pooled MOA CAN detect bQTLs via allele frequency deviation")
    print(f"  at {100*detected_95/len(bqtl_abs_afd):.0f}% power (5% FDR) with 19 hybrids.")
else:
    print("Pooled MOA shows limited power for bQTL detection with 19 hybrids.")
    print("  More lines needed, or signal-weighted approaches required.")
