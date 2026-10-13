"""
pipeline_config.py — Shared path configuration for all pipeline scripts.

Centralizes all file paths so they can be overridden via environment variables
(set by Nextflow) or CLI arguments, while preserving standalone defaults.

Usage in scripts:
    from pipeline_config import config
    # Then use config.gff3, config.genome, config.bqtl, etc.

Override via environment (Nextflow sets these automatically):
    ATLAS_DATA_DIR=/custom/data python scripts/01_bound_region_motifs.py

Override via CLI:
    python scripts/01_bound_region_motifs.py --data-dir /custom/data
"""

import os
import argparse
from pathlib import Path


class PipelineConfig:
    """Lazy-initialized pipeline configuration with env/CLI override support."""

    def __init__(self):
        self._initialized = False
        self._base = Path(__file__).resolve().parent.parent

    def _init(self):
        if self._initialized:
            return
        self._initialized = True

        # Parse CLI args (only recognized ones; ignore script-specific args)
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--data-dir', type=str, default=None)
        parser.add_argument('--results-dir', type=str, default=None)
        parser.add_argument('--figures-dir', type=str, default=None)
        parser.add_argument('--genome-fasta', type=str, default=None)
        parser.add_argument('--gff3', type=str, default=None)
        parser.add_argument('--pwm-database', type=str, default=None)
        parser.add_argument('--peaks-dir', type=str, default=None)
        parser.add_argument('--bqtl-file', type=str, default=None)
        parser.add_argument('--genotype-file', type=str, default=None)
        parser.add_argument('--ase-file', type=str, default=None)
        parser.add_argument('--ase-summary-file', type=str, default=None)
        parser.add_argument('--expression-file', type=str, default=None)
        parser.add_argument('--entap-file', type=str, default=None)
        parser.add_argument('--entap-gz-file', type=str, default=None)
        parser.add_argument('--moesm5-file', type=str, default=None)
        parser.add_argument('--moa-peaks-file', type=str, default=None)
        parser.add_argument('--promoter-size', type=int, default=None)
        parser.add_argument('--fdr-threshold', type=float, default=None)
        parser.add_argument('--motif-threshold', type=float, default=None)
        args, _ = parser.parse_known_args()

        def resolve(cli_val, env_key, default):
            """Priority: CLI arg > env var > default."""
            if cli_val is not None:
                return Path(cli_val)
            env_val = os.environ.get(env_key)
            if env_val is not None:
                return Path(env_val)
            return Path(default)

        def resolve_val(cli_val, env_key, default):
            """Same but for non-path values."""
            if cli_val is not None:
                return cli_val
            env_val = os.environ.get(env_key)
            if env_val is not None:
                return type(default)(env_val)
            return default

        # ── Directories ──
        self._data_dir = resolve(args.data_dir, 'ATLAS_DATA_DIR',
                                 self._base / 'data')
        self._results_dir = resolve(args.results_dir, 'ATLAS_RESULTS_DIR',
                                    self._base / 'results')
        self._figures_dir = resolve(args.figures_dir, 'ATLAS_FIGURES_DIR',
                                    self._base / 'figures' / 'paper_figures')

        # ── Reference files ──
        raw = self._data_dir / 'raw'
        proc = self._data_dir / 'processed'

        self._genome = resolve(args.genome_fasta, 'ATLAS_GENOME',
                               raw / 'B73_NAM5.fa')
        self._gff3 = resolve(args.gff3, 'ATLAS_GFF3',
                             raw / 'Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3')
        self._pwm_db = resolve(args.pwm_database, 'ATLAS_PWM_DB',
                               proc / 'maize_tf_pwm_database.json')
        self._peaks_dir = resolve(args.peaks_dir, 'ATLAS_PEAKS_DIR',
                                  raw / 'engelhorn_peaks_q255')
        self._moa_peaks = resolve(args.moa_peaks_file, 'ATLAS_MOA_PEAKS',
                                  raw / 'savadel2021_mf_bc7_nam5.bed')

        # ── Processed data ──
        self._bqtl = resolve(args.bqtl_file, 'ATLAS_BQTL',
                             proc / 'bqtl_snp_ww.csv')
        self._genotypes = resolve(args.genotype_file, 'ATLAS_GENOTYPES',
                                  proc / 'nam_founder_genotypes_at_bqtl.tsv')
        self._ase = resolve(args.ase_file, 'ATLAS_ASE',
                            proc / 'engelhorn_ase_ww.csv')
        self._ase_summary = resolve(args.ase_summary_file, 'ATLAS_ASE_SUMMARY',
                                    proc / 'engelhorn_ase_ww_gene_summary.csv')
        self._expression = resolve(args.expression_file, 'ATLAS_EXPRESSION',
                                   proc / 'engelhorn_ww_vs_ds_expression.tsv')
        self._entap = resolve(args.entap_file, 'ATLAS_ENTAP',
                              proc / 'entap_annotations.tsv')
        self._entap_gz = resolve(args.entap_gz_file, 'ATLAS_ENTAP_GZ',
                                 raw / 'Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1_entap_results.tsv.gz')
        self._moesm5 = resolve(args.moesm5_file, 'ATLAS_MOESM5',
                               raw / 'supp_table_MOESM5.xlsx')

        # ── Analysis parameters ──
        self._promoter_size = resolve_val(args.promoter_size, 'ATLAS_PROMOTER_SIZE', 3000)
        self._fdr_threshold = resolve_val(args.fdr_threshold, 'ATLAS_FDR_THRESHOLD', 0.05)
        self._motif_threshold = resolve_val(args.motif_threshold, 'ATLAS_MOTIF_THRESHOLD', 6.0)

        # Ensure output directories exist
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Properties (lazy init on first access) ──

    @property
    def base(self):
        return self._base

    @property
    def data_dir(self):
        self._init()
        return self._data_dir

    @property
    def results_dir(self):
        self._init()
        return self._results_dir

    @property
    def figures_dir(self):
        self._init()
        return self._figures_dir

    @property
    def genome(self):
        self._init()
        return self._genome

    @property
    def gff3(self):
        self._init()
        return self._gff3

    @property
    def pwm_db(self):
        self._init()
        return self._pwm_db

    @property
    def peaks_dir(self):
        self._init()
        return self._peaks_dir

    @property
    def moa_peaks(self):
        self._init()
        return self._moa_peaks

    @property
    def bqtl(self):
        self._init()
        return self._bqtl

    @property
    def genotypes(self):
        self._init()
        return self._genotypes

    @property
    def ase(self):
        self._init()
        return self._ase

    @property
    def ase_summary(self):
        self._init()
        return self._ase_summary

    @property
    def expression(self):
        self._init()
        return self._expression

    @property
    def entap(self):
        self._init()
        return self._entap

    @property
    def entap_gz(self):
        self._init()
        return self._entap_gz

    @property
    def moesm5(self):
        self._init()
        return self._moesm5

    @property
    def promoter_size(self):
        self._init()
        return self._promoter_size

    @property
    def fdr_threshold(self):
        self._init()
        return self._fdr_threshold

    @property
    def motif_threshold(self):
        self._init()
        return self._motif_threshold

    # ── Chromosomes ──
    valid_chroms = {f'chr{i}' for i in range(1, 11)}


# Singleton instance — import this in scripts
config = PipelineConfig()
