"""
18_allele_specific_motifs.py — Allele-specific TF motif disruption/creation at causal bQTL.

For each causal bQTL:
  1. Get ref/alt alleles from NAM founder genotype data
  2. Extract ±14bp window (max PWM width) centered on variant from B73 reference
  3. Swap SNP to create alt sequence
  4. Scan both with all 405 maize DAP-seq PWMs
  5. Identify motifs DISRUPTED (ref>thr, alt<thr) or CREATED (alt>thr, ref<thr)
  6. Aggregate: which TF families are consistently disrupted/created at causal bQTL?
  7. Compare to non-causal bQTL as control
"""

import sys, json, warnings
import functools
print = functools.partial(print, flush=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_config import config


# ── PWM scanning ──

def load_pwms():
    """Load PWM database, return list of (tf_name, tf_family, pwm_array, width)."""
    db = json.load(open(config.pwm_db))
    pwms = []
    for m in db['motifs']:
        pwm = np.array(m['pwm'])  # shape (width, 4) — [A, C, G, T]
        pwms.append((m['tf_name'], m['tf_family'], pwm, m['width']))
    return pwms


BASE_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def score_sequence(seq, pwm):
    """Score a sequence against a PWM using log-odds (uniform background)."""
    seq = seq.upper()
    if len(seq) != pwm.shape[0]:
        return -np.inf
    score = 0.0
    for i, base in enumerate(seq):
        idx = BASE_IDX.get(base)
        if idx is None:
            return -np.inf
        p = pwm[i, idx]
        if p <= 0:
            return -np.inf
        score += np.log2(p / 0.25)
    return score


def scan_window(seq, pwms, threshold=6.0):
    """Scan a sequence for all PWM hits above threshold.
    Returns set of (tf_name, tf_family) tuples that have a hit overlapping the center.
    """
    hits = set()
    center = len(seq) // 2  # variant position
    for tf_name, tf_family, pwm, width in pwms:
        # Scan all positions where the PWM overlaps the center base
        for start in range(max(0, center - width + 1), min(len(seq) - width + 1, center + 1)):
            subseq = seq[start:start + width]
            s = score_sequence(subseq, pwm)
            if s >= threshold:
                hits.add((tf_name, tf_family))
                break  # one hit per motif is enough
            # Also check reverse complement
            rc = subseq[::-1].translate(str.maketrans('ACGT', 'TGCA'))
            s_rc = score_sequence(rc, pwm)
            if s_rc >= threshold:
                hits.add((tf_name, tf_family))
                break
    return hits


# ── Data loading ──

def load_data():
    """Load causal bQTL, genotypes, and genome."""
    from pyfaidx import Fasta

    causal = pd.read_csv(config.results_dir / 'causal_bqtl_728.csv')
    print(f"Causal bQTL: {len(causal)}")

    geno = pd.read_csv(config.genotypes, sep='\t')
    print(f"Genotypes: {len(geno)} positions x {geno.shape[1]-3} hybrids")

    genome = Fasta(str(config.genome))

    return causal, geno, genome


def get_alleles(row, geno):
    """Get ref and most common alt allele for a bQTL position."""
    match = geno[(geno['chr'] == row['chr']) & (geno['pos'] == row['pos'])]
    if len(match) == 0:
        return None, None
    match = match.iloc[0]
    ref = match['ref']
    # Get alt alleles from hybrid columns
    hybrid_cols = [c for c in geno.columns if c not in ('chr', 'pos', 'ref')]
    alts = []
    for h in hybrid_cols:
        val = str(match[h])
        if val != '0|0' and val != 'nan' and val != ref:
            alts.append(val)
    if not alts:
        return ref, None
    # Most common alt
    alt = Counter(alts).most_common(1)[0][0]
    return ref, alt


def extract_window(genome, chrom, pos, flank=28):
    """Extract ±flank bp around position (0-based internally, 1-based input)."""
    start = pos - 1 - flank
    end = pos - 1 + flank + 1
    seq = str(genome[chrom][start:end]).upper()
    return seq


# ── Main analysis ──

def main():
    print("=" * 70)
    print("  ALLELE-SPECIFIC MOTIF ANALYSIS AT CAUSAL bQTL")
    print("=" * 70)

    pwms = load_pwms()
    print(f"PWMs loaded: {len(pwms)} motifs, {len(set(f for _,f,_,_ in pwms))} families")

    causal, geno, genome = load_data()

    # Merge genotype info with causal bQTL
    threshold = 6.0
    flank = 28  # max PWM width

    results = []
    n_skip = 0

    for i, row in causal.iterrows():
        ref_base, alt_base = get_alleles(row, geno)
        if ref_base is None or alt_base is None:
            n_skip += 1
            continue

        # Extract window and create alt sequence
        seq = extract_window(genome, row['chr'], row['pos'], flank)
        if len(seq) != 2 * flank + 1:
            n_skip += 1
            continue

        center = flank
        # Verify ref matches
        if seq[center] != ref_base:
            n_skip += 1
            continue

        alt_seq = seq[:center] + alt_base + seq[center+1:]

        # Scan both
        ref_hits = scan_window(seq, pwms, threshold)
        alt_hits = scan_window(alt_seq, pwms, threshold)

        # Classify
        disrupted = ref_hits - alt_hits  # in ref, not in alt
        created = alt_hits - ref_hits    # in alt, not in ref
        maintained = ref_hits & alt_hits

        results.append({
            'chr': row['chr'],
            'pos': row['pos'],
            'gene_id': row['gene_id'],
            'cohens_d': row['cohens_d'],
            'abs_d': row['abs_d'],
            'ref': ref_base,
            'alt': alt_base,
            'n_ref_motifs': len(ref_hits),
            'n_alt_motifs': len(alt_hits),
            'n_disrupted': len(disrupted),
            'n_created': len(created),
            'n_maintained': len(maintained),
            'disrupted_tfs': ','.join(sorted(n for n, _ in disrupted)) if disrupted else '',
            'created_tfs': ','.join(sorted(n for n, _ in created)) if created else '',
            'disrupted_families': ','.join(sorted(set(f for _, f in disrupted))) if disrupted else '',
            'created_families': ','.join(sorted(set(f for _, f in created))) if created else '',
            'variant_hybrids': row['variant_hybrids'],
            'ref_hybrids': row['ref_hybrids'],
        })

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(causal)} causal bQTL...")

    print(f"\nProcessed: {len(results)}, Skipped: {n_skip}")
    df = pd.DataFrame(results)

    # ── Summary statistics ──
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    has_disrupted = df['n_disrupted'] > 0
    has_created = df['n_created'] > 0
    print(f"\nbQTL with motif DISRUPTED: {has_disrupted.sum()} ({100*has_disrupted.mean():.1f}%)")
    print(f"bQTL with motif CREATED:   {has_created.sum()} ({100*has_created.mean():.1f}%)")
    print(f"bQTL with both:            {(has_disrupted & has_created).sum()}")
    print(f"bQTL with neither:         {(~has_disrupted & ~has_created).sum()} ({100*(~has_disrupted & ~has_created).mean():.1f}%)")

    # Family-level disruption counts
    print("\n── Top disrupted TF families ──")
    disrupt_fams = Counter()
    create_fams = Counter()
    for _, r in df.iterrows():
        if r['disrupted_families']:
            for f in r['disrupted_families'].split(','):
                disrupt_fams[f] += 1
        if r['created_families']:
            for f in r['created_families'].split(','):
                create_fams[f] += 1

    print(f"{'Family':<18} {'Disrupted':>10} {'Created':>10} {'Net':>10}")
    all_fams = sorted(set(list(disrupt_fams.keys()) + list(create_fams.keys())))
    rows = []
    for f in all_fams:
        d = disrupt_fams.get(f, 0)
        c = create_fams.get(f, 0)
        rows.append((f, d, c, d - c))
    rows.sort(key=lambda x: -x[1])
    for f, d, c, net in rows:
        print(f"{f:<18} {d:>10} {c:>10} {net:>10}")

    # ── Effect size comparison: disrupted vs not ──
    print("\n── Effect size by disruption status ──")
    if has_disrupted.sum() > 0 and (~has_disrupted).sum() > 0:
        d_yes = df.loc[has_disrupted, 'abs_d']
        d_no = df.loc[~has_disrupted, 'abs_d']
        t, p = stats.mannwhitneyu(d_yes, d_no, alternative='two-sided')
        print(f"  Disrupted:     median |d| = {d_yes.median():.3f} (n={len(d_yes)})")
        print(f"  Not disrupted: median |d| = {d_no.median():.3f} (n={len(d_no)})")
        print(f"  Mann-Whitney p = {p:.4f}")

    # ── Per-family: does disruption of family X correlate with higher |d|? ──
    print("\n── Per-family effect size (disrupted vs not, Mann-Whitney) ──")
    print(f"{'Family':<18} {'n_disrupted':>12} {'med_d_yes':>10} {'med_d_no':>10} {'p':>10}")
    for f, d_count in sorted(disrupt_fams.items(), key=lambda x: -x[1]):
        if d_count < 10:
            continue
        mask = df['disrupted_families'].str.contains(f, na=False)
        d_yes = df.loc[mask, 'abs_d']
        d_no = df.loc[~mask, 'abs_d']
        if len(d_yes) > 0 and len(d_no) > 0:
            _, p = stats.mannwhitneyu(d_yes, d_no, alternative='two-sided')
            print(f"{f:<18} {d_count:>12} {d_yes.median():>10.3f} {d_no.median():>10.3f} {p:>10.4f}")

    # ── Now compare to non-causal bQTL (sample) ──
    print("\n" + "=" * 70)
    print("  CONTROL: NON-CAUSAL bQTL (random sample)")
    print("=" * 70)

    # Load motif data to get non-causal bQTL
    motif_df = pd.read_csv(config.results_dir / 'bound_region_motifs.csv')
    causal_keys = set(zip(causal['chr'], causal['pos']))

    # Get unique non-causal bQTL positions
    all_keys = set(zip(motif_df['chr'], motif_df['pos']))
    noncausal_keys = all_keys - causal_keys
    noncausal_list = list(noncausal_keys)

    # Sample same number as causal
    np.random.seed(42)
    sample_idx = np.random.choice(len(noncausal_list), size=min(len(results), len(noncausal_list)), replace=False)
    noncausal_sample = [noncausal_list[i] for i in sample_idx]
    print(f"Sampling {len(noncausal_sample)} non-causal bQTL as control...")

    ctrl_results = []
    n_ctrl_skip = 0
    for j, (chrom, pos) in enumerate(noncausal_sample):
        match = geno[(geno['chr'] == chrom) & (geno['pos'] == pos)]
        if len(match) == 0:
            n_ctrl_skip += 1
            continue
        match = match.iloc[0]
        ref_base = match['ref']
        hybrid_cols = [c for c in geno.columns if c not in ('chr', 'pos', 'ref')]
        alts = [str(match[h]) for h in hybrid_cols if str(match[h]) not in ('0|0', 'nan', ref_base)]
        if not alts:
            n_ctrl_skip += 1
            continue
        alt_base = Counter(alts).most_common(1)[0][0]

        seq = extract_window(genome, chrom, pos, flank)
        if len(seq) != 2 * flank + 1 or seq[flank] != ref_base:
            n_ctrl_skip += 1
            continue

        alt_seq = seq[:flank] + alt_base + seq[flank+1:]
        ref_hits = scan_window(seq, pwms, threshold)
        alt_hits = scan_window(alt_seq, pwms, threshold)

        disrupted = ref_hits - alt_hits
        created = alt_hits - ref_hits

        ctrl_results.append({
            'n_disrupted': len(disrupted),
            'n_created': len(created),
            'disrupted_families': ','.join(sorted(set(f for _, f in disrupted))) if disrupted else '',
            'created_families': ','.join(sorted(set(f for _, f in created))) if created else '',
        })

        if (j + 1) % 200 == 0:
            print(f"  Processed {j+1}/{len(noncausal_sample)} control bQTL...")

    ctrl_df = pd.DataFrame(ctrl_results)
    print(f"Control processed: {len(ctrl_results)}, skipped: {n_ctrl_skip}")

    ctrl_has_disrupt = ctrl_df['n_disrupted'] > 0
    ctrl_has_create = ctrl_df['n_created'] > 0
    print(f"\nControl bQTL with motif DISRUPTED: {ctrl_has_disrupt.sum()} ({100*ctrl_has_disrupt.mean():.1f}%)")
    print(f"Control bQTL with motif CREATED:   {ctrl_has_create.sum()} ({100*ctrl_has_create.mean():.1f}%)")

    # Fisher's test: disruption enriched in causal?
    a = has_disrupted.sum()
    b = (~has_disrupted).sum()
    c = ctrl_has_disrupt.sum()
    d = (~ctrl_has_disrupt).sum()
    OR, p = stats.fisher_exact([[a, b], [c, d]])
    print(f"\nDisruption enrichment (causal vs non-causal):")
    print(f"  OR = {OR:.3f}, Fisher p = {p:.4f}")

    # Per-family: disruption enrichment in causal vs control
    print("\n── Per-family disruption enrichment (causal vs control) ──")
    print(f"{'Family':<18} {'Causal':>8} {'Control':>8} {'OR':>8} {'p':>10}")

    ctrl_disrupt_fams = Counter()
    for _, r in ctrl_df.iterrows():
        if r['disrupted_families']:
            for f in r['disrupted_families'].split(','):
                ctrl_disrupt_fams[f] += 1

    n_causal = len(df)
    n_ctrl = len(ctrl_df)
    fam_enrichments = []
    for f in all_fams:
        a = disrupt_fams.get(f, 0)
        b = n_causal - a
        c = ctrl_disrupt_fams.get(f, 0)
        d_val = n_ctrl - c
        if a + c < 5:
            continue
        OR, p = stats.fisher_exact([[a, b], [c, d_val]])
        fam_enrichments.append((f, a, c, OR, p))

    fam_enrichments.sort(key=lambda x: x[4])
    for f, a, c, OR, p in fam_enrichments:
        sig = '*' if p < 0.05 else ''
        print(f"{f:<18} {a:>8} {c:>8} {OR:>8.2f} {p:>10.4f} {sig}")

    # ── Save results ──
    out_path = config.results_dir / 'allele_specific_motifs.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Save summary
    summary = {
        'n_causal': len(df),
        'n_disrupted': int(has_disrupted.sum()),
        'n_created': int(has_created.sum()),
        'pct_disrupted': float(100 * has_disrupted.mean()),
        'pct_created': float(100 * has_created.mean()),
        'n_control': len(ctrl_df),
        'ctrl_pct_disrupted': float(100 * ctrl_has_disrupt.mean()),
        'disruption_enrichment_OR': float(OR),
        'disruption_enrichment_p': float(p),
        'top_disrupted_families': dict(disrupt_fams.most_common(10)),
        'top_created_families': dict(create_fams.most_common(10)),
    }
    json.dump(summary, open(config.results_dir / 'allele_specific_motifs_summary.json', 'w'), indent=2)
    print("Done.")


if __name__ == '__main__':
    main()
