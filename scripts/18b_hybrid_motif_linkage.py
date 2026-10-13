"""
18b_hybrid_motif_linkage.py — Per-hybrid haplotype-aware motif analysis at causal bQTL.

For each causal bQTL:
  1. Get the actual local haplotype for each of the 24 hybrids
     (bQTL SNP + any nearby SNPs within ±28bp from genotype data)
  2. Construct each hybrid's local sequence by applying its SNPs to the B73 reference
  3. Scan each hybrid's sequence for TF motifs
  4. Compare variant-group vs reference-group motifs:
     - Which motifs are consistently present in ref hybrids but absent in variant hybrids (disrupted)?
     - Which are consistently present in variant hybrids but absent in ref hybrids (created)?
  5. Statistical test: is there a motif whose presence/absence perfectly or near-perfectly
     separates the two hybrid groups?
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


# ── PWM scanning (same as 18a) ──

BASE_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def load_pwms():
    db = json.load(open(config.pwm_db))
    pwms = []
    for m in db['motifs']:
        pwm = np.array(m['pwm'])
        pwms.append((m['tf_name'], m['tf_family'], pwm, m['width']))
    return pwms


def score_sequence(seq, pwm):
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
    """Return dict of {(tf_name, tf_family): max_score} for hits overlapping center."""
    hits = {}
    center = len(seq) // 2
    for tf_name, tf_family, pwm, width in pwms:
        best = -np.inf
        for start in range(max(0, center - width + 1), min(len(seq) - width + 1, center + 1)):
            subseq = seq[start:start + width]
            s = score_sequence(subseq, pwm)
            best = max(best, s)
            rc = subseq[::-1].translate(str.maketrans('ACGT', 'TGCA'))
            s_rc = score_sequence(rc, pwm)
            best = max(best, s_rc)
        if best >= threshold:
            hits[(tf_name, tf_family)] = best
    return hits


# ── Main analysis ──

def main():
    import argparse
    from pyfaidx import Fasta

    parser = argparse.ArgumentParser()
    parser.add_argument('--causal', type=str, default=None,
                        help='Path to causal bQTL CSV (default: causal_bqtl_728.csv)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV filename (default: hybrid_motif_linkage.csv)')
    parser.add_argument('--label', type=str, default='WW',
                        help='Condition label for display')
    args = parser.parse_args()

    print("=" * 70)
    print(f"  PER-HYBRID MOTIF LINKAGE AT CAUSAL bQTL ({args.label})")
    print("=" * 70)

    pwms = load_pwms()
    print(f"PWMs: {len(pwms)} motifs, {len(set(f for _,f,_,_ in pwms))} families")

    causal_path = Path(args.causal) if args.causal else config.results_dir / 'causal_bqtl_728.csv'
    causal = pd.read_csv(causal_path)
    geno = pd.read_csv(config.genotypes, sep='\t')
    genome = Fasta(str(config.genome))
    print(f"Causal bQTL: {len(causal)}")

    hybrid_cols = [c for c in geno.columns if c not in ('chr', 'pos', 'ref')]
    flank = 28
    threshold = 6.0

    results = []
    n_skip = 0
    n_with_nearby = 0

    for idx, row in causal.iterrows():
        chrom, pos = row['chr'], row['pos']
        variant_hybrids = set(row['variant_hybrids'].split('|'))
        ref_hybrids = set(row['ref_hybrids'].split('|'))

        # Get the bQTL SNP itself
        bqtl_match = geno[(geno['chr'] == chrom) & (geno['pos'] == pos)]
        if len(bqtl_match) == 0:
            n_skip += 1
            continue
        bqtl_row = bqtl_match.iloc[0]
        ref_base = bqtl_row['ref']

        # Get nearby SNPs (within ±flank bp)
        nearby_mask = (geno['chr'] == chrom) & \
                      (geno['pos'] >= pos - flank) & \
                      (geno['pos'] <= pos + flank) & \
                      (geno['pos'] != pos)
        nearby_snps = geno[nearby_mask]
        has_nearby = len(nearby_snps) > 0
        if has_nearby:
            n_with_nearby += 1

        # Extract B73 reference window
        start_coord = pos - 1 - flank
        end_coord = pos - 1 + flank + 1
        ref_seq = str(genome[chrom][start_coord:end_coord]).upper()
        if len(ref_seq) != 2 * flank + 1:
            n_skip += 1
            continue
        if ref_seq[flank] != ref_base:
            n_skip += 1
            continue

        # Build per-hybrid sequences
        hybrid_seqs = {}
        for h in hybrid_cols:
            seq = list(ref_seq)  # start from B73

            # Apply the bQTL SNP for this hybrid
            alt_val = str(bqtl_row[h])
            if alt_val != '0|0' and alt_val != 'nan':
                seq[flank] = alt_val

            # Apply any nearby SNPs for this hybrid
            for _, nearby_row in nearby_snps.iterrows():
                offset = nearby_row['pos'] - pos + flank
                alt_nearby = str(nearby_row[h])
                if alt_nearby != '0|0' and alt_nearby != 'nan':
                    seq[offset] = alt_nearby

            hybrid_seqs[h] = ''.join(seq)

        # Scan each hybrid's sequence for motifs
        hybrid_motifs = {}
        for h, seq in hybrid_seqs.items():
            hybrid_motifs[h] = scan_window(seq, pwms, threshold)

        # Collect all motifs seen in any hybrid
        all_motif_keys = set()
        for motifs in hybrid_motifs.values():
            all_motif_keys.update(motifs.keys())

        if not all_motif_keys:
            results.append({
                'chr': chrom, 'pos': pos, 'gene_id': row['gene_id'],
                'cohens_d': row['cohens_d'], 'abs_d': row['abs_d'],
                'has_nearby_snps': has_nearby,
                'n_motifs_tested': 0,
                'n_perfectly_separating': 0,
                'best_motif_tf': '', 'best_motif_family': '',
                'best_motif_type': '', 'best_motif_fisher_p': 1.0,
                'all_disrupted': '', 'all_created': '',
                'variant_hybrids': row['variant_hybrids'],
                'ref_hybrids': row['ref_hybrids'],
            })
            continue

        # For each motif, test if it separates variant vs ref hybrids
        motif_tests = []
        for (tf_name, tf_family) in all_motif_keys:
            # Count: how many variant hybrids have this motif vs ref hybrids
            var_has = sum(1 for h in variant_hybrids
                         if h in hybrid_motifs and (tf_name, tf_family) in hybrid_motifs[h])
            var_lacks = len(variant_hybrids) - var_has
            ref_has = sum(1 for h in ref_hybrids
                         if h in hybrid_motifs and (tf_name, tf_family) in hybrid_motifs[h])
            ref_lacks = len(ref_hybrids) - ref_has

            # Fisher's exact test
            OR, p = stats.fisher_exact([[var_has, var_lacks], [ref_has, ref_lacks]])

            # Classify: disrupted (ref has, variant doesn't) or created (variant has, ref doesn't)
            if ref_has > var_has:
                mtype = 'disrupted'
            elif var_has > ref_has:
                mtype = 'created'
            else:
                mtype = 'unchanged'

            # Consistency score: what fraction of variant hybrids agree?
            if mtype == 'disrupted':
                consistency = var_lacks / len(variant_hybrids)  # should lack motif
            elif mtype == 'created':
                consistency = var_has / len(variant_hybrids)  # should have motif
            else:
                consistency = 0

            motif_tests.append({
                'tf_name': tf_name, 'tf_family': tf_family,
                'var_has': var_has, 'var_lacks': var_lacks,
                'ref_has': ref_has, 'ref_lacks': ref_lacks,
                'OR': OR, 'p': p, 'type': mtype,
                'consistency': consistency,
            })

        # Sort by p-value
        motif_tests.sort(key=lambda x: x['p'])

        # Count perfectly separating motifs (p < 0.05 AND consistency >= 0.8)
        sig_motifs = [m for m in motif_tests if m['p'] < 0.05 and m['consistency'] >= 0.8]
        disrupted_sig = [m for m in sig_motifs if m['type'] == 'disrupted']
        created_sig = [m for m in sig_motifs if m['type'] == 'created']

        best = motif_tests[0] if motif_tests else None

        results.append({
            'chr': chrom, 'pos': pos, 'gene_id': row['gene_id'],
            'cohens_d': row['cohens_d'], 'abs_d': row['abs_d'],
            'has_nearby_snps': has_nearby,
            'n_motifs_tested': len(all_motif_keys),
            'n_sig_separating': len(sig_motifs),
            'n_disrupted_sig': len(disrupted_sig),
            'n_created_sig': len(created_sig),
            'best_motif_tf': best['tf_name'] if best else '',
            'best_motif_family': best['tf_family'] if best else '',
            'best_motif_type': best['type'] if best else '',
            'best_motif_fisher_p': best['p'] if best else 1.0,
            'best_motif_consistency': best['consistency'] if best else 0.0,
            'best_motif_var_has': best['var_has'] if best else 0,
            'best_motif_ref_has': best['ref_has'] if best else 0,
            'all_disrupted_sig': ';'.join(f"{m['tf_name']}({m['tf_family']})" for m in disrupted_sig),
            'all_created_sig': ';'.join(f"{m['tf_name']}({m['tf_family']})" for m in created_sig),
            'variant_hybrids': row['variant_hybrids'],
            'ref_hybrids': row['ref_hybrids'],
        })

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx+1}/{len(causal)}...")

    df = pd.DataFrame(results)
    print(f"\nProcessed: {len(df)}, Skipped: {n_skip}, With nearby SNPs: {n_with_nearby}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    has_sig = df['n_sig_separating'] > 0
    print(f"\nbQTL with ≥1 significantly separating motif (p<0.05, consistency≥80%):")
    print(f"  {has_sig.sum()}/{len(df)} ({100*has_sig.mean():.1f}%)")

    has_disrupt = df['n_disrupted_sig'] > 0
    has_create = df['n_created_sig'] > 0
    print(f"  With disrupted motif: {has_disrupt.sum()} ({100*has_disrupt.mean():.1f}%)")
    print(f"  With created motif:   {has_create.sum()} ({100*has_create.mean():.1f}%)")

    print(f"\nDistribution of # separating motifs per bQTL:")
    for n in sorted(df['n_sig_separating'].unique()):
        count = (df['n_sig_separating'] == n).sum()
        if count > 0:
            print(f"  {n}: {count} bQTL")

    # Top TF families as drivers
    print("\n── TF families most often the 'driver' (significantly separating) ──")
    fam_disrupt = Counter()
    fam_create = Counter()
    tf_disrupt = Counter()
    tf_create = Counter()
    for _, r in df.iterrows():
        val = r.get('all_disrupted_sig', '')
        if isinstance(val, str) and val:
            for entry in val.split(';'):
                if not entry:
                    continue
                tf = entry.split('(')[0]
                fam = entry.split('(')[1].rstrip(')')
                fam_disrupt[fam] += 1
                tf_disrupt[tf] += 1
        val = r.get('all_created_sig', '')
        if isinstance(val, str) and val:
            for entry in val.split(';'):
                if not entry:
                    continue
                tf = entry.split('(')[0]
                fam = entry.split('(')[1].rstrip(')')
                fam_create[fam] += 1
                tf_create[tf] += 1

    print(f"\n{'Family':<18} {'Disrupted':>10} {'Created':>10} {'Total':>10}")
    all_fams = sorted(set(list(fam_disrupt.keys()) + list(fam_create.keys())),
                       key=lambda f: -(fam_disrupt.get(f, 0) + fam_create.get(f, 0)))
    for f in all_fams:
        d = fam_disrupt.get(f, 0)
        c = fam_create.get(f, 0)
        print(f"{f:<18} {d:>10} {c:>10} {d+c:>10}")

    # Top individual TFs
    print(f"\n── Top 20 individual TFs as drivers ──")
    print(f"{'TF':<20} {'Disrupted':>10} {'Created':>10}")
    all_tfs = Counter()
    all_tfs.update(tf_disrupt)
    all_tfs.update(tf_create)
    for tf, total in all_tfs.most_common(20):
        d = tf_disrupt.get(tf, 0)
        c = tf_create.get(tf, 0)
        print(f"{tf:<20} {d:>10} {c:>10}")

    # Co-occurring driver motifs (multiple TFs at same bQTL)
    print(f"\n── bQTL with multiple driver TF families ──")
    multi = df[df['n_sig_separating'] >= 2]
    print(f"bQTL with ≥2 separating motifs: {len(multi)}")
    if len(multi) > 0:
        # Count co-occurring family pairs
        pair_counts = Counter()
        for _, r in multi.iterrows():
            fams = set()
            for col in ['all_disrupted_sig', 'all_created_sig']:
                val = r.get(col, '')
                if isinstance(val, str) and val:
                    for entry in val.split(';'):
                        if entry:
                            fam = entry.split('(')[1].rstrip(')')
                            fams.add(fam)
            fams = sorted(fams)
            for i in range(len(fams)):
                for j in range(i + 1, len(fams)):
                    pair_counts[(fams[i], fams[j])] += 1

        print(f"\nTop TF family co-occurrences at same bQTL:")
        for (f1, f2), count in pair_counts.most_common(15):
            print(f"  {f1} × {f2}: {count}")

    # Effect size: bQTL with driver motif vs without
    print(f"\n── Effect size by driver motif presence ──")
    d_with = df.loc[has_sig, 'abs_d']
    d_without = df.loc[~has_sig, 'abs_d']
    if len(d_with) > 0 and len(d_without) > 0:
        _, p = stats.mannwhitneyu(d_with, d_without, alternative='two-sided')
        print(f"  With driver motif:    median |d| = {d_with.median():.3f} (n={len(d_with)})")
        print(f"  Without driver motif: median |d| = {d_without.median():.3f} (n={len(d_without)})")
        print(f"  Mann-Whitney p = {p:.4f}")

    # Example: show top 10 bQTL with strongest motif separation
    print(f"\n── Top 10 bQTL with strongest driver motif signal ──")
    top = df[df['best_motif_fisher_p'] < 1.0].nsmallest(10, 'best_motif_fisher_p')
    for _, r in top.iterrows():
        print(f"  {r['chr']}:{r['pos']} gene={r['gene_id']} |d|={r['abs_d']:.2f}")
        print(f"    Best: {r['best_motif_tf']} ({r['best_motif_family']}) "
              f"{r['best_motif_type']} p={r['best_motif_fisher_p']:.2e} "
              f"consistency={r['best_motif_consistency']:.0%} "
              f"var_has={r['best_motif_var_has']} ref_has={r['best_motif_ref_has']}")
        val = r.get('all_disrupted_sig', '')
        if isinstance(val, str) and val:
            print(f"    Disrupted: {val}")
        val = r.get('all_created_sig', '')
        if isinstance(val, str) and val:
            print(f"    Created: {val}")

    # Save
    out_name = args.output if args.output else 'hybrid_motif_linkage.csv'
    out_path = config.results_dir / out_name
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
