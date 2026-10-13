# Causal bQTL Atlas — Pipeline Revision Log

Tracks all modifications to the pipeline, their rationale, and impact on results.
Use this to communicate changes to the team and update the paper.

---

## 2026-10-07: Switch from q3 to q255 peak files + remove fallback + fix controls

**Scripts:** `scripts/01_bound_region_motifs.py`, `scripts/02_pan_cistrome_peaks.py`,
`scripts/04_functional_bqtl.py`, `scripts/08_sole_motif_verify.py`,
`scripts/12_rewiring_analysis.py`, `scripts/pipeline_config.py`

**Problem:**
Three issues with the peak data used throughout the pipeline:
1. **Wrong peak set** — We used q3 narrowPeak files (MAPQ≥3, multi-mapping allowed), but
   bQTL were identified using q255 peaks (MAPQ=255, unique mapping only) per Julia Engelhorn.
   q3 peaks are a stricter subset due to MACS2 background model behavior, so 15.9% of bQTL
   fell outside all q3 peaks. With q255, coverage rises from 84.1% to 90.1%.
2. **Unnecessary ±250bp fallback** — bQTL are by definition within binding peaks. The fallback
   created artificial scan windows for positions where we simply lacked the correct peak file.
3. **Inflated control window** — The control analysis used ±250bp (500bp total) but the actual
   median peak width is ~138bp (q255), making the null expectation ~3.6× too wide.

**Data source:**
q255 peaks downloaded from GEO supplement GSE294039:
`GSE294039_q255.peaks.names.sorted.bed.tar.gz` (208MB, 50 gzipped BED files)
→ extracted to `data/raw/engelhorn_peaks_q255/`

**Change:**
- Default peaks directory changed from `engelhorn_peaks` (q3) to `engelhorn_peaks_q255`
- All peak-loading functions updated to support gzipped q255 BED format, with q3 fallback
- Removed `FALLBACK_REGION_BP = 250` — bQTL without peaks are now skipped, not approximated
- Removed `used_peak` column from output (all rows now use real peaks)
- Control analysis uses actual median peak width from real data instead of fixed ±250bp
- Added terminology note: "disrupted"/"created" are relative to B73, not evolutionary direction

**Coverage:**

| Peak set | bQTL covered | Coverage |
|----------|-------------|----------|
| q3 WW (old) | 124,374 | 84.1% |
| **q255 WW (new)** | **133,260** | **90.1%** |
| q255 WW+DS | 137,339 | 92.8% |

The remaining ~10% are in NAM-coordinate-only peaks (confirmed by J. Engelhorn).
These are recovered via NAM coordinate liftover (see below).
Median peak width: 138bp (q255) vs 183bp (q3).

**NAM coordinate liftover (recovering the remaining ~10%):**
J. Engelhorn provided per-hybrid liftover files mapping B73→NAM coordinates for all
SNPs tested in bQTL analysis (`NAM_coordinates_2FP/B73.{FOUNDER}.WW.2FPs.csv.gz`).
For bQTL not in any B73-coordinate q255 peak, the script:
1. Looks up NAM coordinates from the liftover file
2. Finds the tightest NAM-coordinate peak containing that position
3. Projects peak boundaries back to B73 using position offsets
4. Scans the B73 reference sequence within those projected boundaries

This recovers 14,678 of 14,682 missing bQTL (only 4 still unresolvable, 0.003%).

| Peak source | bQTL covered | Coverage |
|-------------|-------------|----------|
| B73 q255 WW | 133,260 | 90.1% |
| NAM liftover | 14,678 | 9.9% |
| **Total** | **147,938** | **~100%** |

**Files changed:**
- `scripts/pipeline_config.py` — default peaks_dir → engelhorn_peaks_q255
- `scripts/01_bound_region_motifs.py` — q255 support, NAM liftover, fallback removed, control fix
- `scripts/02_pan_cistrome_peaks.py` — q255 support
- `scripts/04_functional_bqtl.py` — q255 support
- `scripts/08_sole_motif_verify.py` — q255 support
- `scripts/12_rewiring_analysis.py` — q255 support
- `data/raw/engelhorn_peaks_q255/` — new q255 peak files from GEO
- `data/raw/NAM_coordinates_2FP/` — liftover files from J. Engelhorn

---

## 2026-10-07: Step 01 — Use real WW peak boundaries instead of fixed proxy

*Superseded by the q255 + NAM liftover entry above. This documents the original
change from ±250bp proxy to real peaks; the entry above documents the subsequent
switch from q3 to q255 peaks and NAM liftover integration.*

**Final results (q255 + NAM liftover):**

| Metric | Original (±250bp proxy) | Final (q255 + NAM) | Change |
|--------|------------------------|-------------------|--------|
| bQTL analyzed | 147,942 (100%, proxy) | **147,938 (~100%, real peaks)** | Real coverage |
| Peak source | None (fixed window) | B73 90.1% + NAM 9.9% | Experimentally defined |
| Region width (median) | 500bp (fixed) | **158bp** (range 57–1737bp) | -68% |
| Mean motifs per region | 215 | **83.2** | **-61%** |
| bQTL with motif at variant | 91.5% | **86.3%** | -5.2pp |
| Sole-copy motifs | 6.3% | **29.0%** | **+23pp (4.6×)** |
| bQTL vs control ratio | >1x (inflated) | **1.35× (p=1.7e-6)** | Corrected |
| Control window | ±250bp (500bp) | **±79bp (158bp)** | Matched to data |
| Redundancy vs ASE | — | rho=0.009, p=0.016 | Weak positive |

**Key biological implications:**
1. **Sole-copy motifs at 29%** — nearly a third of disrupted motifs have no backup
   copy of the same TF family in the binding region, meaning complete loss of that
   TF's binding site.
2. **bQTL regions are 1.35× enriched** over random promoter regions (p=1.7×10⁻⁶).
   The old ±250bp control window was 3.6× too wide, masking this real enrichment.
3. **Per-family sole-copy rates:** E2F/DP 95%, LFY 93%, AP2 81%, RAV 81%, HSF 69%,
   YABBY 69% — these families rarely have redundant copies. ERF (14%) and Dof (14%)
   remain highly redundant.

**Files changed:**
- `scripts/01_bound_region_motifs.py` — q255 peaks, NAM liftover, no fallback, control fix
- `results/bound_region_motifs.csv` — regenerated with `peak_source`, `region_width` columns
- `figures/paper_figures/fig5_bound_region_motifs.pdf` — regenerated

---

## 2026-10-07: Pipeline infrastructure — shared config module

**Scripts:** All 14 pipeline scripts + `nextflow.config`

**Problem:**
All scripts had hardcoded paths relative to `__file__`, making it impossible to run
the pipeline with different data directories or on different organisms without editing
every script.

**Change:**
- Created `scripts/pipeline_config.py` — shared config module with lazy-initialized
  path properties and three-level override: CLI args > environment variables > defaults
- Converted all 14 pipeline scripts to `from pipeline_config import config`
- Updated `nextflow.config` to export `ATLAS_*` environment variables in `beforeScript`
- Created `params.yaml` documenting all input data files and parameters

**Impact:** No change to results — purely infrastructure. Enables running on different
organisms/datasets by changing `params.yaml` or CLI arguments.

---

## 2026-10-07: Step 14 — Fix drought genotype test (WW_VS_DROUGHT)

**Script:** `scripts/14_drought_genotype_test.py`

**Problem:**
Three bugs caused the drought analysis to produce 0 testable pairs:
1. Founder-to-hybrid mapping failed — `B73xA188.upper()` looked up `B73XA188` but
   founders are `A188` → no matches in the mapping dict
2. Double `B73x` prefix in ASE lookup — `f'B73x{hybrid}'` but `hybrid` already
   contained the prefix (e.g., `B73xA188`)
3. No guard for empty DataFrame — KeyError on `fdr_mw` column when 0 pairs tested

**Change:**
- Strip `B73x` prefix before matching: `h.replace('B73x', '').upper()` for lookup
- Use `hybrid` directly in ASE lookup (already contains the prefix)
- Added `if len(rdf) == 0: continue` guard before accessing FDR columns

**Impact on results:**

| Metric | Before (broken) | After (fixed) |
|--------|-----------------|---------------|
| WW testable pairs | 0 | 34,006 |
| DS testable pairs | 0 | 34,367 |
| WW FDR hits | 0 | 4,400 (12.9%) |
| DS FDR hits | 0 | 3,911 (11.4%) |
| Constitutive (both) | — | 2,051 |
| WW-only | — | 2,315 |
| DS-only | — | 1,799 |
| Functional fraction | — | 18.5% |

**Paper sections to update:**
- Results: drought/condition analysis section (all numbers)
- Step 15 (condition switching) results also changed downstream

---

## 2026-10-07: Verify pipeline — wire into workflow

**Script:** `main.nf`

**Problem:**
The `VERIFY_PIPELINE` process was defined but never called in the workflow block.

**Change:**
Added `VERIFY_PIPELINE()` call collecting `.done` signals from all terminal steps
via `.mix().collect()`.

**Impact:** No change to analysis results. Pipeline now runs verification automatically.

---

## 2026-10-07: Step 05 — Fix three bugs in genotype-aware bQTL test

**Script:** `scripts/05_genotype_bqtl_test.py`

**Problem:**
Three bugs prevented the genotype-aware functional test from producing valid results:
1. **Undefined `DATA` variable** — line 58 referenced `DATA` (from pre-`pipeline_config` era)
   which caused a `NameError` when `config.ase` path didn't contain "genotype". The script
   could not run at all with the current config module.
2. **Founder-to-hybrid mapping failed** — `FOUNDER_TO_HYBRID` matched `A188.upper()` against
   `B73XA188` (full hybrid name including prefix). Since `A188 ≠ B73XA188`, no founders
   mapped to hybrids, producing 0 genotype lookups. Same class of bug as step 14.
3. **Double `B73x` prefix in ASE lookup** — `f'B73x{hybrid}'` but `hybrid` already contained
   the prefix (e.g., `B73xA188` → `B73xB73xA188`). Same class of bug as step 14.

Additionally:
4. **Column name case mismatch** — `bqtl.get('type', '')` used lowercase but CSV column is
   `Type`. Similarly `fdr` → `FDRp`. These silently returned defaults instead of real values.
5. **O(n×m) performance** — `map_bqtl_to_genes` iterated all 40K genes for each of 147K bQTL.
   Replaced with chromosome-indexed promoter lookup.

**Change:**
- Fixed ASE file loading to use `config.ase` directly (removed broken fallback)
- Strip `B73x` prefix before matching: `h.replace('B73x', '').upper()` for founder lookup
- Use `hybrid` directly in ASE lookup (already contains the prefix)
- Fixed column names: `Type`, `FDRp` (matching bqtl_snp_ww.csv header)
- Refactored `map_bqtl_to_genes` with per-chromosome sorted promoter index

**Impact on results:**
Existing `genotype_bqtl_results.csv` (from Oct 6) was generated by a pre-`pipeline_config`
version. The script needs to be re-run to produce correct results with all hybrids properly
mapped. Expected: significant increase in testable pairs (from broken → working mapping).

**Paper sections to update:**
- Results: genotype-aware causal test section (all numbers need regeneration)
- Figure 11: will be regenerated with correct data

**Files changed:**
- `scripts/05_genotype_bqtl_test.py` — fixed three bugs + performance improvement

---

## Template for future entries

```
## YYYY-MM-DD: Step XX — Short description

**Script:** `scripts/XX_name.py`

**Problem:**
What was wrong and why it matters.

**Change:**
What was changed technically.

**Impact on results:**
Table or bullet points showing before/after numbers.

**Paper sections to update:**
Which sections/figures need revision.

**Files changed:**
List of modified files.
```
