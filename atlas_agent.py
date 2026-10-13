#!/usr/bin/env python3
"""
atlas_agent.py — Pydantic AI agent for the Causal bQTL Atlas pipeline.

An LLM-powered agent that orchestrates, validates, explains, and
troubleshoots the causal bQTL atlas pipeline.

Usage:
  # Interactive chat (default)
  python atlas_agent.py

  # With specific provider/model
  python atlas_agent.py --model openai:gpt-4o
  python atlas_agent.py --model anthropic:claude-sonnet-4-20250514

  # Non-interactive modes
  python atlas_agent.py --validate
  python atlas_agent.py --explain
  python atlas_agent.py --run
  python atlas_agent.py --run 06 07 20
"""

import os
import re
import sys
import csv
import json
import asyncio
import subprocess
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext


# ── Pipeline step definitions ────────────────────────────────────────

@dataclass
class PipelineStep:
    """Definition of a single pipeline step."""
    id: str
    script: str
    name: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    optional: bool = False


PIPELINE_STEPS = [
    PipelineStep(
        id="01", script="01_bound_region_motifs.py",
        name="Bound Region Motifs",
        description="Scan MOA-seq binding peaks for TF motifs using 405 DAP-seq PWMs. "
                    "Maps each bQTL to its surrounding binding peak and identifies which "
                    "TF family motifs are present at the variant position.",
        inputs=["data/raw/engelhorn_peaks_q255/",
                "data/processed/bqtl_snp_ww.csv",
                "data/processed/maize_tf_pwm_database.json"],
        outputs=["results/bound_region_motifs.csv"],
    ),
    PipelineStep(
        id="02", script="02_pan_cistrome_peaks.py",
        name="Pan-Cistrome Peaks",
        description="Merge peaks across all 25 hybrids to create a unified peak catalog. "
                    "Counts disrupted peaks per gene promoter.",
        inputs=["data/raw/engelhorn_peaks_q255/"],
        outputs=["results/pan_cistrome_peaks.csv"],
    ),
    PipelineStep(
        id="03", script="03_convert_genotypes.py",
        name="Convert Genotypes",
        description="Convert per-chromosome NAM founder genotype files into pipeline TSV format.",
        outputs=["data/processed/nam_founder_genotypes_at_bqtl.tsv"],
        optional=True,
    ),
    PipelineStep(
        id="04", script="04_functional_bqtl.py",
        name="Functional bQTL Test",
        description="Map bQTL to gene promoters (2kb upstream TSS) and test whether peak "
                    "presence associates with allele-specific expression.",
        depends_on=["01", "02"],
        inputs=["results/bound_region_motifs.csv", "results/pan_cistrome_peaks.csv"],
        outputs=["results/bqtl_functional_test.csv"],
    ),
    PipelineStep(
        id="05", script="05_genotype_bqtl_test.py",
        name="Genotype-ASE Causal Test",
        description="For each bQTL-gene pair, split NAM hybrids by genotype and compare "
                    "ASE distributions (Mann-Whitney U). Core causal test: does the variant "
                    "genotype predict allele-specific expression?",
        depends_on=["04"],
        inputs=["results/bqtl_functional_test.csv",
                "data/processed/nam_founder_genotypes_at_bqtl.tsv"],
        outputs=["results/bqtl_signed_ase_test.csv"],
    ),
    PipelineStep(
        id="06", script="06_select_causal.py",
        name="Select Causal bQTL",
        description="Apply 6-step filtering: promoter -> ASE -> FDR<0.05 -> unique gene -> "
                    "peak overlap -> high-confidence causal bQTL set.",
        depends_on=["05"],
        inputs=["results/bqtl_signed_ase_test.csv"],
        outputs=["results/causal_bqtl_728.csv"],
    ),
    PipelineStep(
        id="07", script="07_gene_atlas.py",
        name="Gene Atlas",
        description="Deep characterization of causal targets: TF x gene heatmap, "
                    "co-regulation patterns, master regulators, signaling layers.",
        depends_on=["06"],
        outputs=["results/gene_atlas_728.csv"],
    ),
    PipelineStep(
        id="08", script="08_sole_motif_verify.py",
        name="Sole Motif Verification",
        description="Check whether the disrupted TF family's motif is the sole copy within "
                    "the MOA-seq binding peak.",
        depends_on=["06"],
        outputs=["results/sole_motif_verification_728.csv"],
    ),
    PipelineStep(
        id="09", script="09_motif_creation_disruption.py",
        name="Bidirectional Motif Analysis",
        description="Compare B73 ref vs NAM alt at each bQTL: which motifs are "
                    "disrupted, created, or rewired (both at same position)?",
        depends_on=["06"],
        outputs=["results/allele_specific_motifs.csv"],
    ),
    PipelineStep(
        id="10", script="10_phenotype_ratelimit.py",
        name="Phenotype Rate-Limiting",
        description="NAM hybrid phenotype clustering and KEGG rate-limiting step enrichment.",
        depends_on=["06"],
        outputs=["results/phenotype_associations_728.csv"],
    ),
    PipelineStep(
        id="11", script="11_threshold_sensitivity.py",
        name="Threshold Sensitivity",
        description="PWM threshold sweep (4.0-8.0) to verify stability of disrupted/created/"
                    "rewired fractions across thresholds.",
        depends_on=["06"],
        outputs=["results/threshold_sensitivity_sweep.csv"],
    ),
    PipelineStep(
        id="12", script="12_rewiring_analysis.py",
        name="Rewiring Analysis",
        description="Deep rewiring analysis: sole-copy checks, family-to-family transition "
                    "matrix, signaling integration.",
        depends_on=["06", "08", "09"],
        outputs=["results/rewiring_deep_analysis_728.csv"],
    ),
    PipelineStep(
        id="13", script="13_build_interactive.py",
        name="Interactive Atlas",
        description="Build the interactive HTML visualization with D3.js cell diagram, "
                    "filters, tooltips, and optional OpenAI summaries.",
        depends_on=["06", "09"],
        outputs=["results/gene_atlas_728_interactive.html"],
        optional=True,
    ),
    PipelineStep(
        id="18b", script="18b_hybrid_motif_linkage.py",
        name="Per-Hybrid Motif Linkage",
        description="Per-hybrid haplotype-aware motif scanning with 405 DAP-seq PWMs. "
                    "Tests whether each motif's binding score separates variant vs reference "
                    "hybrid groups. Identifies driver motifs at 93% of causal bQTL.",
        depends_on=["06"],
        outputs=["results/hybrid_motif_linkage.csv"],
    ),
    PipelineStep(
        id="19", script="19_tf_feedback_network.py",
        name="TF Feedback Network",
        description="TF family enrichment among causal targets (binomial test). Constructs "
                    "directed regulatory network: autoregulatory loops and cross-family rewiring.",
        depends_on=["18b"],
        outputs=["results/causal_tf_enrichment.csv",
                 "results/causal_regulatory_network.csv"],
    ),
    PipelineStep(
        id="20", script="20_drought_causal_bqtl.py",
        name="Drought Causal bQTL",
        description="Repeat causal bQTL pipeline on drought-stress ASE data. "
                    "Identifies 1,267 drought-causal bQTL (12% overlap with WW).",
        depends_on=["06"],
        outputs=["results/causal_bqtl_ds.csv", "results/ww_vs_ds_comparison.json"],
    ),
    PipelineStep(
        id="21", script="21_drought_expression_integration.py",
        name="Drought Expression Integration",
        description="Integrate drought causal bQTL with population-level expression data. "
                    "Cis vs trans, effect correlation, condition specificity, TF enrichment.",
        depends_on=["20"],
        outputs=["results/drought_expression_integration.json"],
    ),
    PipelineStep(
        id="22", script="22_condition_functional_characterization.py",
        name="Functional Characterization",
        description="Gene set decomposition (871 WW-only, 821 DS-only, 129 shared-same, "
                    "75 shared-different). EntAP annotation, GO enrichment, UMAP.",
        depends_on=["20"],
        outputs=["results/condition_specific_gene_sets.json",
                 "results/condition_functional_categories.csv"],
    ),
    PipelineStep(
        id="23", script="23_comparative_enrichment.py",
        name="Comparative Enrichment",
        description="Exhaustive functional comparison of WW vs DS gene sets. GO enrichment, "
                    "detailed category breakdown, Fisher's exact tests.",
        depends_on=["22"],
        outputs=["results/comparative_enrichment_categories.csv"],
    ),
    PipelineStep(
        id="17", script="17_causality_prediction.py",
        name="Causality Prediction",
        description="Train classifiers to predict which bQTL are causal from motif features. "
                    "Achieves AUC=0.895 — validates that causal atlas is learnable.",
        depends_on=["06"],
        outputs=["results/causality_prediction_results.json"],
    ),
    PipelineStep(
        id="26", script="26_population_motif_occupancy.py",
        name="Population Motif Occupancy (Layer 1)",
        description="Layer 1 of the three-layer motif analysis. For each gene, scans MOA-seq peaks "
                    "across all 23 WW hybrids for TF motif enrichment using 405 DAP-seq PWMs. "
                    "Builds a gene × TF family × hybrid occupancy matrix, independent of bQTL.",
        depends_on=["06"],
        outputs=["results/population_motif_occupancy.csv",
                 "results/population_motif_summary.csv",
                 "results/population_motif_peak_counts.csv",
                 "results/population_motif_causal_crossref.csv"],
    ),
    PipelineStep(
        id="27", script="27_tf_cascade_tracing.py",
        name="TF Cascade Tracing (Layer 3)",
        description="Layer 3 of the three-layer motif analysis. Traces causal bQTL through TF "
                    "regulatory cascades: bQTL disrupts TF_A motif → target gene = TF_B → "
                    "TF_B's downstream targets. Reveals amplification, hub vulnerability, "
                    "and cascade depth.",
        depends_on=["18b", "26"],
        outputs=["results/tf_cascade_edges.csv",
                 "results/tf_cascade_summary.csv",
                 "results/tf_cascade_hubs.csv"],
    ),
    PipelineStep(
        id="28", script="28_cross_layer_validation.py",
        name="Cross-Layer Validation",
        description="Integrates all three motif layers: (A) conservation-impact scoring — how "
                    "conserved across hybrids is each disrupted motif, (B) WW vs DS differential "
                    "occupancy — which TF families gain/lose under drought, (C) hybrid-specific "
                    "regulatory damage — per-hybrid disruption score weighted by conservation.",
        depends_on=["27", "20"],
        outputs=["results/cross_layer_conservation_impact.csv",
                 "results/cross_layer_ww_vs_ds_differential.csv",
                 "results/cross_layer_hybrid_damage.csv"],
    ),
    PipelineStep(
        id="29", script="29_ase_motif_validation.py",
        name="ASE × Motif Validation",
        description="Closes the causal loop: tests whether disruption of conserved motifs "
                    "produces stronger allele-specific expression. Alt vs ref ASE, per-TF family "
                    "effects, dose-response, Cohen's d correlation.",
        depends_on=["28"],
        outputs=["results/ase_motif_validation.csv",
                 "results/ase_motif_family_effects.csv"],
    ),
    PipelineStep(
        id="30", script="30_tf_cooperativity.py",
        name="TF Cooperativity Network",
        description="TF co-occupancy network from Layer 1 data. Classifies TF pair regulatory "
                    "logic (enhanceosome vs billboard), WW vs DS differential partnerships, "
                    "and co-disruption synergy effects on ASE.",
        depends_on=["29", "26"],
        outputs=["results/tf_cooperativity_network.csv",
                 "results/tf_cooperativity_variable.csv",
                 "results/tf_cooperativity_condition.csv"],
    ),
    PipelineStep(
        id="31", script="31_flow_matching_design.py",
        name="Flow Matching Sequence Design",
        description="Conditional flow matching generative model for regulatory sequence design. "
                    "Learns directed sequence edits from 147K bQTL ref/alt pairs conditioned on "
                    "TF binding targets, ASE effects, and drought/control context. Guided by "
                    "Agro-NT LoRA gradients. Includes ASE prediction head and CRISPR base-edit design.",
        depends_on=["30"],
        outputs=["models/flow_matching/best_model.pt"],
    ),
    PipelineStep(
        id="31b", script="31_population_genetics.py",
        name="Population Genetics",
        description="Allele frequency spectrum, effect-frequency relationship, and selection "
                    "signatures at causal bQTL. Tests whether causal variants show different MAF "
                    "distributions (0.312 vs 0.280) consistent with selection-mutation balance.",
        depends_on=["06"],
        outputs=["results/bqtl_allele_frequencies.csv",
                 "results/population_genetics.json"],
    ),
    PipelineStep(
        id="32", script="32_hybrid_phenotype_similarity.py",
        name="Hybrid Phenotype Similarity",
        description="Cross-hybrid phenotype similarity from shared causal bQTL burden. "
                    "Tests whether hybrids sharing more causal variants have more similar "
                    "phenotypes (flowering, height).",
        depends_on=["06"],
        outputs=["results/hybrid_phenotype_similarity.json"],
    ),
    PipelineStep(
        id="33", script="33_gwas_overlap.py",
        name="GWAS Overlap",
        description="Overlap between causal bQTL and published GWAS hits. Tests enrichment "
                    "of causal variants near trait-associated loci from Wallace et al.",
        depends_on=["06"],
        outputs=["results/causal_gwas_overlap.csv"],
    ),
    PipelineStep(
        id="34", script="34_scatac_integration.py",
        name="scATAC-seq Integration",
        description="Integrate causal bQTL with single-cell ATAC-seq accessible chromatin "
                    "regions (Marand et al. 2021). Tests cell-type enrichment of causal variants.",
        depends_on=["06"],
        outputs=["results/scatac_integration.json"],
        optional=True,
    ),
    PipelineStep(
        id="35", script="35_network_topology.py",
        name="Network Topology",
        description="Test whether causal bQTL target genes occupy different positions in the "
                    "TF regulatory network (degree, entropy, hub status, Jaccard similarity). "
                    "Result: no significant differences — supports distributed OS model.",
        depends_on=["19"],
        outputs=["results/network_topology.json",
                 "results/network_topology_metrics.csv"],
    ),
    PipelineStep(
        id="36", script="36_bqtl_epistasis.py",
        name="bQTL Epistasis",
        description="Test additive vs epistatic combination of multiple bQTL per gene. "
                    "Dose-response: each variant allele adds ~0.056 |ASE|. 2.6:1 additive:saturating "
                    "ratio validates the billboard model of cis-regulatory evolution.",
        depends_on=["06"],
        outputs=["results/bqtl_epistasis.json",
                 "results/bqtl_epistasis_genes.csv"],
    ),
    PipelineStep(
        id="37", script="37_te_context.py",
        name="TE Context",
        description="Classify all 147,942 bQTL by transposable element superfamily using EDTA "
                    "annotations. Causal bQTL are depleted in TEs (12.2% vs 20.6%), LTR 7x depleted.",
        depends_on=["06"],
        inputs=["data/raw/B73_NAM5_TE.gff3.gz"],
        outputs=["results/te_context.json",
                 "results/bqtl_te_context.csv"],
    ),
    PipelineStep(
        id="38", script="38_chromatin_context.py",
        name="Chromatin Context",
        description="Classify bQTL by gene-relative position and Marand ACR overlap as chromatin "
                    "state proxy. 3.51x enrichment at proximal promoters (63.4% vs 18.1%).",
        depends_on=["06"],
        outputs=["results/chromatin_context.json"],
    ),
    PipelineStep(
        id="39", script="39_eqtl_proxy.py",
        name="Binding-Expression Causal Chain",
        description="Test binding-to-expression causal chain using ASE as eQTL proxy. "
                    "73.7% concordance (830/1126 genes), effect size predicts ASE (rho=0.107).",
        depends_on=["20"],
        outputs=["results/eqtl_proxy.json"],
    ),
]

STEP_MAP = {s.id: s for s in PIPELINE_STEPS}


# ── Pipeline runner (agent dependency) ────────────────────────────────

class PipelineRunner:
    """Runs and monitors pipeline scripts. Injected as agent dependency."""

    def __init__(self, base_dir: Path, python: str = "python3"):
        self.base = base_dir
        self.scripts_dir = base_dir / "scripts"
        self.results_dir = base_dir / "results"
        self.figures_dir = base_dir / "figures" / "paper_figures"
        self.python = python
        self.paper = PaperCompanion(base_dir)

    def check_inputs(self, step: PipelineStep) -> dict:
        missing, found = [], []
        for inp in step.inputs:
            (found if (self.base / inp).exists() else missing).append(inp)
        return {"found": found, "missing": missing, "ready": len(missing) == 0}

    def check_outputs(self, step: PipelineStep) -> dict:
        existing, missing = [], []
        for out in step.outputs:
            (existing if (self.base / out).exists() else missing).append(out)
        return {"existing": existing, "missing": missing,
                "complete": len(missing) == 0 and len(existing) > 0}

    def run_step(self, step_id: str) -> dict:
        step = STEP_MAP.get(step_id)
        if not step:
            return {"success": False, "error": f"Unknown step: {step_id}"}
        script_path = self.scripts_dir / step.script
        if not script_path.exists():
            return {"success": False, "error": f"Script not found: {script_path}"}
        for dep in step.depends_on:
            dep_step = STEP_MAP.get(dep)
            if dep_step and dep_step.outputs:
                if not self.check_outputs(dep_step)["complete"]:
                    return {"success": False,
                            "error": f"Dependency {dep} ({dep_step.name}) not complete."}
        try:
            result = subprocess.run(
                [self.python, str(script_path)],
                capture_output=True, text=True,
                cwd=str(self.base), timeout=3600
            )
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-2000:] if result.stdout else "",
                "stderr_tail": result.stderr[-1000:] if result.stderr else "",
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout (1 hour)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def validate_all(self) -> dict:
        status = {}
        for step in PIPELINE_STEPS:
            outputs = self.check_outputs(step)
            inputs = self.check_inputs(step)
            status[step.id] = {
                "name": step.name,
                "inputs_ready": inputs["ready"],
                "outputs_complete": outputs["complete"],
                "missing_inputs": inputs["missing"],
                "missing_outputs": outputs["missing"],
            }
        return status

    def get_next_steps(self) -> list[str]:
        runnable = []
        for step in PIPELINE_STEPS:
            if self.check_outputs(step)["complete"]:
                continue
            if not self.check_inputs(step)["ready"]:
                continue
            deps_met = all(
                self.check_outputs(STEP_MAP[d])["complete"]
                for d in step.depends_on if d in STEP_MAP
            )
            if deps_met:
                runnable.append(step.id)
        return runnable

    def load_result(self, path: str) -> Optional[dict]:
        full_path = self.base / path
        if not full_path.exists():
            return None
        try:
            if full_path.suffix == '.json':
                with open(full_path) as f:
                    return json.load(f)
            elif full_path.suffix == '.csv':
                try:
                    import pandas as pd
                    df = pd.read_csv(full_path)
                    return {
                        "rows": len(df),
                        "columns": list(df.columns),
                        "head": df.head(5).to_dict(orient="records"),
                    }
                except ImportError:
                    with open(full_path) as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        rows = list(reader)
                    return {
                        "rows": len(rows),
                        "columns": header,
                        "head": [dict(zip(header, r)) for r in rows[:5]],
                    }
        except Exception as e:
            return {"error": str(e)}

    def list_results(self) -> list[dict]:
        if not self.results_dir.exists():
            return []
        files = []
        for f in sorted(self.results_dir.glob("*")):
            if f.is_file():
                files.append({"name": f.name, "bytes": f.stat().st_size})
        return files

    def list_figures(self) -> list[str]:
        if not self.figures_dir.exists():
            return []
        return sorted([f.name for f in self.figures_dir.glob("*.pdf")])

    def read_config(self) -> str:
        config_path = self.scripts_dir / "pipeline_config.py"
        if config_path.exists():
            return config_path.read_text()
        return "pipeline_config.py not found"

    # ── Batch run with dependency resolution ──

    def _topo_sort(self, step_ids: list[str]) -> list[str]:
        """Topological sort of step IDs respecting dependencies."""
        # Collect all required steps (including transitive deps)
        required = set()
        def _collect(sid):
            if sid in required:
                return
            required.add(sid)
            step = STEP_MAP.get(sid)
            if step:
                for dep in step.depends_on:
                    if dep in STEP_MAP and not self.check_outputs(STEP_MAP[dep])["complete"]:
                        _collect(dep)
        for sid in step_ids:
            _collect(sid)

        # Kahn's algorithm
        in_degree = {s: 0 for s in required}
        for s in required:
            step = STEP_MAP.get(s)
            if step:
                for dep in step.depends_on:
                    if dep in required:
                        in_degree[s] = in_degree.get(s, 0) + 1

        queue = sorted([s for s, d in in_degree.items() if d == 0])
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for s in sorted(required):
                step = STEP_MAP.get(s)
                if step and node in step.depends_on:
                    in_degree[s] -= 1
                    if in_degree[s] == 0:
                        queue.append(s)
        return order

    def batch_run(self, step_ids: list[str], force: bool = False) -> dict:
        """Run multiple steps in dependency order. Skips steps with existing outputs
        unless force=True. Returns per-step results."""
        order = self._topo_sort(step_ids)
        results = {}
        for sid in order:
            step = STEP_MAP.get(sid)
            if not step:
                results[sid] = {"success": False, "error": f"Unknown step: {sid}"}
                continue
            if not force and self.check_outputs(step)["complete"]:
                results[sid] = {"success": True, "skipped": True,
                                "message": f"Outputs already exist for {step.name}"}
                continue
            result = self.run_step(sid)
            results[sid] = result
            if not result["success"]:
                results["_stopped_at"] = sid
                break
        return {
            "order": order,
            "results": results,
            "all_success": all(r.get("success", False) for r in results.values()
                              if not isinstance(r, str)),
        }

    # ── Dynamic Nextflow generation ──

    # Map step IDs to their existing Nextflow process names
    _EXISTING_PROCESS_NAMES = {
        "01": "BOUND_REGION_MOTIFS", "02": "PAN_CISTROME_PEAKS",
        "03": "CONVERT_GENOTYPES", "04": "FUNCTIONAL_BQTL",
        "05": "GENOTYPE_BQTL_TEST", "06": "SELECT_CAUSAL",
        "07": "GENE_ATLAS", "08": "SOLE_MOTIF_VERIFY",
        "09": "MOTIF_CREATION_DISRUPTION", "10": "PHENOTYPE_RATELIMIT",
        "11": "THRESHOLD_SENSITIVITY", "12": "REWIRING_ANALYSIS",
        "13": "INTERACTIVE_ATLAS", "18b": "HYBRID_MOTIF_LINKAGE",
        "19": "TF_FEEDBACK_NETWORK", "20": "DROUGHT_CAUSAL_BQTL",
        "21": "DROUGHT_EXPRESSION", "22": "FUNCTIONAL_CHARACTERIZATION",
        "23": "COMPARATIVE_ENRICHMENT", "17": "CAUSALITY_PREDICTION",
        "26": "POPULATION_MOTIF_OCCUPANCY", "27": "TF_CASCADE_TRACING",
        "28": "CROSS_LAYER_VALIDATION", "29": "ASE_MOTIF_VALIDATION",
        "30": "TF_COOPERATIVITY", "31": "FLOW_MATCHING_DESIGN",
    }

    def _step_to_process_name(self, step_id: str) -> str:
        """Convert step ID to Nextflow process name (e.g., '37' -> 'TE_CONTEXT')."""
        if step_id in self._EXISTING_PROCESS_NAMES:
            return self._EXISTING_PROCESS_NAMES[step_id]
        step = STEP_MAP.get(step_id)
        if not step:
            return f"STEP_{step_id}"
        return step.name.upper().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")

    def generate_nextflow_process(self, step_id: str) -> str:
        """Generate a Nextflow DSL2 process block for a pipeline step."""
        step = STEP_MAP.get(step_id)
        if not step:
            return f"// Unknown step: {step_id}"

        process_name = self._step_to_process_name(step_id)

        # Build input channels from dependencies
        input_lines = []
        if step.depends_on:
            for i, dep in enumerate(step.depends_on):
                var_name = f"ready_{i}" if i > 0 else "ready"
                input_lines.append(f"    val {var_name}")
        else:
            input_lines.append("    // No dependencies — can run immediately")

        lines = [
            f"process {process_name} {{",
            f"    /*",
            f"     * Step {step.id}: {step.name}",
            f"     * {step.description[:120]}",
            f"     */",
            f"",
        ]

        if input_lines and step.depends_on:
            lines.append("    input:")
            lines.extend(input_lines)
            lines.append("")

        lines.extend([
            "    output:",
            "    val true, emit: done",
            "",
            "    script:",
            f'    if ("{step.id}" in skipSet)',
            f'        "echo \'SKIP: Step {step.id} — using existing {step.name} results\'"',
            "    else",
            f'        "python3 ${{projectDir}}/scripts/{step.script}"',
            "}",
        ])

        return "\n".join(lines)

    def generate_workflow_wiring(self, step_ids: list[str]) -> str:
        """Generate Nextflow workflow wiring for a set of steps."""
        order = self._topo_sort(step_ids)
        lines = ["    // ── Dynamically added steps ──"]

        for sid in order:
            step = STEP_MAP.get(sid)
            if not step:
                continue
            process_name = self._step_to_process_name(sid)
            if step.depends_on:
                dep_args = []
                for dep in step.depends_on:
                    dep_name = self._step_to_process_name(dep)
                    dep_args.append(f"{dep_name}.out.done")
                if len(dep_args) == 1:
                    lines.append(f"    {process_name}({dep_args[0]})")
                else:
                    args = ",\n        ".join(dep_args)
                    lines.append(f"    {process_name}(\n        {args}\n    )")
            else:
                lines.append(f"    {process_name}()")

        return "\n".join(lines)

    def add_to_nextflow(self, step_ids: list[str]) -> dict:
        """Add new processes to main.nf and wire them into the workflow."""
        nf_path = self.base / "main.nf"
        if not nf_path.exists():
            return {"success": False, "error": "main.nf not found"}

        content = nf_path.read_text()

        # Check which steps already have processes
        new_ids = []
        for sid in step_ids:
            step = STEP_MAP.get(sid)
            if not step:
                continue
            process_name = self._step_to_process_name(sid)
            if f"process {process_name}" in content:
                continue  # already exists
            new_ids.append(sid)

        if not new_ids:
            return {"success": True, "message": "All requested steps already in main.nf",
                    "added": []}

        # Generate process blocks
        process_blocks = []
        for sid in new_ids:
            process_blocks.append(self.generate_nextflow_process(sid))

        # Insert processes before the workflow block
        workflow_marker = "workflow {"
        if workflow_marker not in content:
            return {"success": False, "error": "Could not find 'workflow {' in main.nf"}

        new_section = (
            "\n\n// ══════════════════════════════════════════════════════════════════\n"
            "// DYNAMICALLY ADDED PROCESSES\n"
            "// ══════════════════════════════════════════════════════════════════\n\n"
            + "\n\n".join(process_blocks)
            + "\n\n"
        )

        # Only add if section doesn't exist yet
        if "// DYNAMICALLY ADDED PROCESSES" in content:
            # Append to existing dynamic section
            marker = "// ══════════════════════════════════════════════════════════════════\n// DYNAMICALLY ADDED PROCESSES"
            # Find end of dynamic section (next // ══ line or workflow {)
            idx = content.index(marker)
            # Find the workflow { after
            wf_idx = content.index(workflow_marker, idx)
            content = (content[:wf_idx]
                       + "\n".join(self.generate_nextflow_process(sid) for sid in new_ids)
                       + "\n\n\n"
                       + content[wf_idx:])
        else:
            content = content.replace(
                workflow_marker,
                new_section + workflow_marker
            )

        # Add wiring to workflow
        wiring = self.generate_workflow_wiring(new_ids)

        # Insert before VERIFY_PIPELINE or at end of workflow
        verify_marker = "    // ── Phase 6: Verification ──"
        if verify_marker in content:
            content = content.replace(
                verify_marker,
                f"    // ── Dynamically wired steps ──\n{wiring}\n\n{verify_marker}"
            )
        else:
            # Insert before closing }
            last_brace = content.rindex("}")
            content = content[:last_brace] + f"\n{wiring}\n" + content[last_brace:]

        # Write back
        nf_path.write_text(content)

        return {
            "success": True,
            "added": new_ids,
            "process_names": [self._step_to_process_name(sid) for sid in new_ids],
            "message": f"Added {len(new_ids)} processes to main.nf: "
                       + ", ".join(self._step_to_process_name(sid) for sid in new_ids),
        }

    def show_pipeline_dag(self) -> str:
        """Generate an ASCII DAG of all registered pipeline steps."""
        lines = ["\nPipeline DAG:\n"]
        for step in PIPELINE_STEPS:
            deps = " <- " + ", ".join(step.depends_on) if step.depends_on else ""
            complete = self.check_outputs(step)["complete"]
            icon = "[done]" if complete else "[    ]"
            opt = " (optional)" if step.optional else ""
            lines.append(f"  {icon} [{step.id:3s}] {step.name:40s}{deps}{opt}")
        return "\n".join(lines)

    def run_nextflow(self, skip: str = "") -> dict:
        nf_path = self.base / "main.nf"
        if not nf_path.exists():
            return {"success": False, "error": "main.nf not found"}
        cmd = ["nextflow", "run", str(nf_path)]
        if skip:
            cmd.extend(["--skip", skip])
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=str(self.base), timeout=7200
            )
            return {
                "success": result.returncode == 0,
                "stdout_tail": result.stdout[-3000:] if result.stdout else "",
                "stderr_tail": result.stderr[-1000:] if result.stderr else "",
            }
        except FileNotFoundError:
            return {"success": False, "error": "nextflow not found in PATH"}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout (2 hours)"}

    def run_verification(self) -> str:
        script = self.scripts_dir / "verify_pipeline.py"
        if not script.exists():
            return "verify_pipeline.py not found."
        try:
            result = subprocess.run(
                [self.python, str(script)],
                capture_output=True, text=True,
                cwd=str(self.base), timeout=60
            )
            return result.stdout + result.stderr
        except Exception as e:
            return f"Verification failed: {e}"


# ── Paper companion ───────────────────────────────────────────────────

# Map paper figures to generating scripts and result dependencies
FIGURE_MAP = {
    "fig_gene_motif_heatmap.pdf": {
        "script": "07_gene_atlas.py", "step": "07",
        "results": ["gene_atlas_728.csv"],
        "desc": "Gene × TF family heatmap showing which TF families are disrupted at each target gene",
    },
    "fig_signaling_layers.pdf": {
        "script": "07_gene_atlas.py", "step": "07",
        "results": ["gene_atlas_728.csv"],
        "desc": "Functional architecture of causal bQTL targets: signal transduction, TFs, metabolism, transport",
    },
    "fig_coregulation_network.pdf": {
        "script": "07_gene_atlas.py", "step": "07",
        "results": ["gene_atlas_728.csv"],
        "desc": "Co-regulation patterns: genes sharing disrupted TF families form regulatory modules",
    },
    "fig_motif_rewiring.pdf": {
        "script": "25b_rewiring_figures_only.py", "step": "12",
        "results": ["rewiring_deep_analysis_728.csv", "tf_family_transition_matrix.csv"],
        "desc": "TF family switching heatmap + category breakdown (rewired/disrupted/created/neither)",
    },
    "fig_rewiring_effects.pdf": {
        "script": "25b_rewiring_figures_only.py", "step": "12",
        "results": ["rewiring_deep_analysis_728.csv"],
        "desc": "Effect sizes by motif change category and clean switch status",
    },
    "fig_causal_bqtl_hybrid_distribution.pdf": {
        "script": "25_paper_extra_figures.py", "step": "06",
        "results": ["causal_bqtl_728.csv"],
        "desc": "4-panel: carrier histogram, effect vs frequency, per-hybrid counts, cumulative distribution",
    },
    "fig_tf_cascade_network.pdf": {
        "script": "25_paper_extra_figures.py", "step": "07",
        "results": ["causal_bqtl_728.csv", "gene_atlas_728.csv"],
        "desc": "TF cascade: disrupted motifs at TF bQTL, cascade reach, drought response pie",
    },
}

# Map paper sections to pipeline steps and key statistics
PAPER_SECTIONS = [
    {"title": "Introduction", "line": 88, "steps": [],
     "summary": "Motivates causal variant identification from pan-cistrome data"},
    {"title": "Six-step pipeline identifies 1,075 causal bQTL", "line": 124, "steps": ["04", "05", "06"],
     "summary": "Core filtering: promoter→ASE→FDR→unique→peak overlap",
     "key_stats": ["1,075 causal bQTL", "FDR<0.05", "24 hybrids"]},
    {"title": "Linkage disequilibrium and single-bQTL design", "line": 169, "steps": ["06"],
     "summary": "Each gene has exactly one causal bQTL — avoids LD confounding"},
    {"title": "Disrupted motifs are non-redundant", "line": 182, "steps": ["08"],
     "summary": "44.5% sole-copy disruptions within MOA-seq peaks",
     "key_stats": ["44.5% sole-copy", "478/1,075 have any sole", "120 all-sole"]},
    {"title": "Variants create new motifs too", "line": 254, "steps": ["09", "12"],
     "summary": "38.0% rewired (disrupt+create), cross-family switches dominate",
     "key_stats": ["37.3% rewired", "27 clean switches", "13.3:1 cross:same ratio"]},
    {"title": "Variant distribution across NAM hybrids", "line": 366, "steps": ["06"],
     "summary": "Median 8 carriers per variant, 372-438 variants per hybrid",
     "figures": ["fig_causal_bqtl_hybrid_distribution.pdf"]},
    {"title": "Target genes encode regulatory layer", "line": 399, "steps": ["07"],
     "summary": "Signal transduction (173), metabolism (143), TF (75), transport (56)",
     "figures": ["fig_signaling_layers.pdf"]},
    {"title": "Gene × TF family heatmap", "line": 454, "steps": ["07"],
     "summary": "Heatmap reveals co-regulation modules and master regulators",
     "figures": ["fig_gene_motif_heatmap.pdf"]},
    {"title": "Co-regulation patterns", "line": 489, "steps": ["07"],
     "summary": "TF family convergence and regulatory network structure",
     "figures": ["fig_coregulation_network.pdf"]},
    {"title": "TF cascade network", "line": 518, "steps": ["07", "19"],
     "summary": "75 TF targets with 6,787 directed regulatory edges",
     "figures": ["fig_tf_cascade_network.pdf"]},
    {"title": "Drought integration", "line": 545, "steps": ["20", "21", "22"],
     "summary": "27% drought-responsive, WRKY +34% under drought, aquaporin TIP3-2"},
    {"title": "Known signaling modules", "line": 576, "steps": ["07"],
     "summary": "BES1-BZR1, WRKY cascades, TCP-auxin confirm biological validity"},
    {"title": "Per-hybrid motif linkage", "line": 603, "steps": ["18b"],
     "summary": "93% of causal bQTL have ≥1 significantly separating motif",
     "key_stats": ["997/1,075 (92.7%) linked", "405 DAP-seq PWMs"]},
    {"title": "TF feedback network", "line": 627, "steps": ["19"],
     "summary": "Autoregulatory loops and cross-family rewiring network",
     "key_stats": ["397 TFs", "997 targets", "6,787 edges"]},
    {"title": "Drought unmasks different causal variants", "line": 658, "steps": ["20", "22", "23"],
     "summary": "1,267 DS bQTL, 12% overlap, 81% condition-specific genes",
     "key_stats": ["1,267 DS", "129 overlap", "871 WW-only", "821 DS-only"]},
    {"title": "Causality predictable from motif architecture", "line": 742, "steps": ["17"],
     "summary": "RF AUC=0.895 — motif features alone predict causality",
     "key_stats": ["AUC=0.895±0.005"]},
    {"title": "Discussion", "line": 776, "steps": [],
     "summary": "Implications for breeding, cis-regulatory evolution, cross-species potential"},
    {"title": "Methods", "line": 891, "steps": [],
     "summary": "Detailed methods for all pipeline steps"},
]


class PaperCompanion:
    """Read and cross-reference the LaTeX paper with pipeline results."""

    def __init__(self, base_dir: Path):
        self.base = base_dir
        self.paper_path = base_dir / "paper" / "causal_bqtl_atlas.tex"
        self.results_dir = base_dir / "results"
        self.figures_dir = base_dir / "figures" / "paper_figures"
        self.scripts_dir = base_dir / "scripts"
        self._paper_text = None

    @property
    def paper_text(self) -> str:
        if self._paper_text is None:
            if self.paper_path.exists():
                self._paper_text = self.paper_path.read_text()
            else:
                self._paper_text = ""
        return self._paper_text

    def list_sections(self) -> list[dict]:
        return PAPER_SECTIONS

    def read_section(self, section_idx: int) -> dict:
        if section_idx < 0 or section_idx >= len(PAPER_SECTIONS):
            return {"error": f"Invalid section index. Use 0-{len(PAPER_SECTIONS)-1}"}
        sec = PAPER_SECTIONS[section_idx]
        lines = self.paper_text.split('\n')
        start = sec["line"] - 1
        # Find next section
        end = len(lines)
        for other in PAPER_SECTIONS:
            if other["line"] > sec["line"]:
                end = other["line"] - 1
                break
        text = '\n'.join(lines[start:min(start + 80, end)])
        return {
            "title": sec["title"],
            "summary": sec["summary"],
            "steps": sec.get("steps", []),
            "figures": sec.get("figures", []),
            "key_stats": sec.get("key_stats", []),
            "text_preview": text,
        }

    def verify_statistics(self) -> list[dict]:
        """Cross-check key paper claims against result files."""
        checks = []

        # Check 1: Causal bQTL count
        causal_path = self.results_dir / "causal_bqtl_728.csv"
        if causal_path.exists():
            with open(causal_path) as f:
                n = sum(1 for _ in f) - 1
            in_paper = "1,075" in self.paper_text or "1075" in self.paper_text
            checks.append({
                "claim": f"1,075 causal bQTL",
                "actual": n,
                "match": n == 1075,
                "in_paper": in_paper,
            })

        # Check 2: Drought overlap
        comp_path = self.results_dir / "ww_vs_ds_comparison.json"
        if comp_path.exists():
            with open(comp_path) as f:
                comp = json.load(f)
            n_ds = comp.get("n_ds_causal", 0)
            n_overlap = comp.get("n_overlap", 0)
            checks.append({
                "claim": f"1,267 drought causal bQTL",
                "actual": n_ds,
                "match": n_ds == 1267,
                "in_paper": "1,267" in self.paper_text,
            })
            checks.append({
                "claim": f"129 overlap positions",
                "actual": n_overlap,
                "match": n_overlap == 129,
                "in_paper": "129" in self.paper_text,
            })

        # Check 3: Rewiring percentage
        rw_path = self.results_dir / "rewiring_deep_analysis_728.csv"
        if rw_path.exists():
            with open(rw_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            rewired = sum(1 for r in rows if r.get("category") == "rewired")
            pct = 100 * rewired / len(rows) if rows else 0
            checks.append({
                "claim": "37.3% rewired",
                "actual": f"{pct:.1f}% ({rewired}/{len(rows)})",
                "match": abs(pct - 37.3) < 0.5,
                "in_paper": "37.3" in self.paper_text,
            })

        # Check 4: Causality prediction AUC
        pred_path = self.results_dir / "causality_prediction_results.json"
        if pred_path.exists():
            with open(pred_path) as f:
                pred = json.load(f)
            auc = pred.get("phase1_3to1", {}).get("RF", {}).get("auc_mean", 0)
            checks.append({
                "claim": "AUC = 0.895",
                "actual": f"{auc:.3f}",
                "match": abs(auc - 0.895) < 0.005,
                "in_paper": "0.895" in self.paper_text,
            })

        # Check 5: Hybrid motif linkage
        link_path = self.results_dir / "hybrid_motif_linkage.csv"
        if link_path.exists():
            with open(link_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            n_linked = sum(1 for r in rows
                         if r.get("n_sig_separating", "0").strip()
                         and float(r["n_sig_separating"]) > 0)
            pct = 100 * n_linked / len(rows) if rows else 0
            checks.append({
                "claim": "93% have ≥1 sig motif",
                "actual": f"{pct:.1f}% ({n_linked}/{len(rows)})",
                "match": abs(pct - 92.7) < 1.0,
                "in_paper": "92.7" in self.paper_text or "93" in self.paper_text,
            })

        # Check 6: Gene set sizes
        gs_path = self.results_dir / "condition_specific_gene_sets.json"
        if gs_path.exists():
            with open(gs_path) as f:
                gs = json.load(f)
            for key, expected, label in [
                ("ww_only_genes", 871, "871 WW-only"),
                ("ds_only_genes", 821, "821 DS-only"),
            ]:
                actual = len(gs.get(key, []))
                checks.append({
                    "claim": label,
                    "actual": actual,
                    "match": actual == expected,
                    "in_paper": str(expected) in self.paper_text,
                })

        return checks

    def figure_info(self, fig_name: str) -> dict:
        info = FIGURE_MAP.get(fig_name)
        if not info:
            return {"error": f"Unknown figure. Available: {', '.join(FIGURE_MAP.keys())}"}
        exists = (self.figures_dir / fig_name).exists()
        result = {**info, "exists": exists}
        if exists:
            stat = (self.figures_dir / fig_name).stat()
            result["size_kb"] = stat.st_size / 1000
        return result

    def regenerate_figure(self, fig_name: str, python: str = "python3") -> dict:
        info = FIGURE_MAP.get(fig_name)
        if not info:
            return {"success": False, "error": f"Unknown figure: {fig_name}"}
        script_path = self.scripts_dir / info["script"]
        if not script_path.exists():
            return {"success": False, "error": f"Script not found: {script_path}"}
        try:
            result = subprocess.run(
                [python, str(script_path)],
                capture_output=True, text=True,
                cwd=str(self.base), timeout=600,
            )
            return {
                "success": result.returncode == 0,
                "stdout_tail": result.stdout[-1000:] if result.stdout else "",
                "stderr_tail": result.stderr[-500:] if result.stderr else "",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def check_figures(self) -> list[dict]:
        """Check all paper figures exist and are up to date."""
        results = []
        for fig_name, info in FIGURE_MAP.items():
            fig_path = self.figures_dir / fig_name
            entry = {"figure": fig_name, "exists": fig_path.exists()}
            if fig_path.exists():
                entry["size_kb"] = fig_path.stat().st_size / 1000
                # Check if results are newer than figure
                for res in info.get("results", []):
                    res_path = self.results_dir / res
                    if res_path.exists() and res_path.stat().st_mtime > fig_path.stat().st_mtime:
                        entry["stale"] = True
                        entry["newer_result"] = res
                        break
            results.append(entry)
        return results


# ── Pydantic AI Agent ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Causal bQTL Atlas Agent — an expert assistant for the maize causal bQTL analysis pipeline AND its companion paper.

You help researchers:
1. VALIDATE their data and pipeline state
2. RUN pipeline steps in the correct order (individual scripts or full Nextflow)
3. BATCH RUN multiple steps with automatic dependency resolution
4. EXPLAIN results at each step in biological context
5. TROUBLESHOOT errors and suggest fixes
6. INTERPRET the final atlas findings
7. CONFIGURE pipeline paths and parameters
8. WALK THROUGH the paper — explain sections, cross-reference results, verify claims
9. MANAGE FIGURES — check, explain, and regenerate paper figures
10. DYNAMICALLY EXTEND the Nextflow pipeline — generate process blocks, wire dependencies, add new analyses

Pipeline overview:
- Input: 147,942 maize bQTL from a 24-hybrid pan-cistrome (Engelhorn et al. 2025, q255 peaks)
- Core pipeline (steps 01-06): filters to 1,075 high-confidence causal bQTL
- Characterization (steps 07-13): motif analysis, rewiring, interactive atlas
- Three-layer motif analysis:
  Layer 1 (step 26): Population-level TF motif occupancy across all hybrids per gene
  Layer 2 (step 18b): Per-hybrid haplotype-aware motif linkage at causal bQTL
  Layer 3 (step 27): TF cascade tracing — bQTL → disrupted TF → TF gene → downstream targets
  Cross-layer (step 28): conservation-impact scoring, WW vs DS differential, hybrid damage scores
  ASE validation (step 29): motif disruption → allele-specific expression causality
  Cooperativity (step 30): TF co-occupancy, enhanceosome/billboard, condition-specific partnerships
  Flow matching design (step 31): conditional generative model for TF-directed sequence edits, LoRA-guided
- TF feedback network (19): autoregulatory loops and cross-family rewiring
- Drought comparison (20-23): condition-specific regulatory rewiring
- Prediction (17): ML validation that causality is learnable from motif features
- Extended analyses (31b-39): population genetics, GWAS overlap, TE context, network topology,
  bQTL epistasis, chromatin context, binding-expression causal chain

Dynamic Nextflow capabilities:
- Use show_pipeline_dag to visualize all steps and their status
- Use batch_run_steps to run multiple steps with automatic dependency resolution
- Use preview_nextflow_process to see what a Nextflow process block would look like
- Use add_steps_to_nextflow to dynamically add new analyses to main.nf

Key biological findings:
- 1,075 causal bQTL under well-watered, 1,267 under drought
- Only 12% overlap between conditions; 81% of target genes are condition-specific
- 37.3% of variants rewire TF occupancy (disrupt one motif, create another)
- Cross-family switches outnumber same-family 13.3:1
- Causality predictable from motif architecture alone (AUC=0.895)

When explaining results, connect findings to plant biology and breeding implications.
Be concise but precise with numbers. Use tools to check pipeline state before making claims."""

atlas_agent = Agent(
    deps_type=PipelineRunner,
    system_prompt=SYSTEM_PROMPT,
    name="atlas-agent",
)


@atlas_agent.tool
async def validate_pipeline(ctx: RunContext[PipelineRunner]) -> str:
    """Check the status of all pipeline steps: which have completed,
    which are ready to run, which are missing inputs."""
    status = ctx.deps.validate_all()
    complete = sum(1 for s in status.values() if s["outputs_complete"])
    total = len(status)
    lines = [f"Pipeline status: {complete}/{total} steps complete\n"]
    for sid, s in status.items():
        icon = "done" if s["outputs_complete"] else "pending"
        lines.append(f"  [{icon}] [{sid}] {s['name']}")
        if s["missing_inputs"]:
            lines.append(f"      Missing inputs: {', '.join(s['missing_inputs'])}")
    return "\n".join(lines)


@atlas_agent.tool
async def get_next_steps(ctx: RunContext[PipelineRunner]) -> str:
    """Find pipeline steps that are ready to run (dependencies met, outputs missing)."""
    step_ids = ctx.deps.get_next_steps()
    if not step_ids:
        return "All steps are complete, or no steps have their dependencies met."
    lines = ["Steps ready to run:"]
    for sid in step_ids:
        step = STEP_MAP[sid]
        lines.append(f"  [{sid}] {step.name}: {step.description[:120]}...")
    return "\n".join(lines)


@atlas_agent.tool
async def describe_step(ctx: RunContext[PipelineRunner], step_id: str) -> str:
    """Get detailed description of a pipeline step including its
    inputs, outputs, dependencies, and current status."""
    step = STEP_MAP.get(step_id)
    if not step:
        return f"Unknown step: {step_id}. Valid IDs: {', '.join(STEP_MAP.keys())}"
    inputs = ctx.deps.check_inputs(step)
    outputs = ctx.deps.check_outputs(step)
    return "\n".join([
        f"Step [{step.id}] {step.name}",
        f"Script: {step.script}",
        f"Description: {step.description}",
        f"Dependencies: {step.depends_on or 'none'}",
        f"Inputs: {step.inputs or 'from previous steps'}",
        f"Outputs: {step.outputs or 'figures only'}",
        f"Optional: {step.optional}",
        f"\nCurrent status:",
        f"  Inputs ready: {inputs['ready']}",
        f"  Outputs complete: {outputs['complete']}",
    ])


@atlas_agent.tool
async def run_step(ctx: RunContext[PipelineRunner], step_id: str) -> str:
    """Run a specific pipeline step by its ID (e.g., '01', '06', '20').
    Checks dependencies before running. Returns stdout/stderr."""
    result = ctx.deps.run_step(step_id)
    if result["success"]:
        return f"Step {step_id} completed successfully.\n\nOutput (last 2000 chars):\n{result['stdout_tail']}"
    error = result.get("error", result.get("stderr_tail", "Unknown error"))
    return f"Step {step_id} FAILED.\n\nError:\n{error}"


@atlas_agent.tool
async def run_full_pipeline(ctx: RunContext[PipelineRunner], skip_steps: str = "") -> str:
    """Run the full Nextflow pipeline. Optionally skip steps with existing
    outputs by providing comma-separated IDs (e.g., '01,02,06')."""
    result = ctx.deps.run_nextflow(skip=skip_steps)
    if result["success"]:
        return f"Nextflow pipeline completed.\n\n{result['stdout_tail']}"
    error = result.get("error", result.get("stderr_tail", "Unknown error"))
    return f"Nextflow pipeline FAILED.\n\n{error}"


@atlas_agent.tool
async def batch_run_steps(ctx: RunContext[PipelineRunner], step_ids: str, force: bool = False) -> str:
    """Run multiple pipeline steps in dependency order. Provide comma-separated
    IDs (e.g., '35,36,37'). Automatically resolves and runs dependencies first.
    Skips steps with existing outputs unless force=True."""
    ids = [s.strip() for s in step_ids.split(",")]
    result = ctx.deps.batch_run(ids, force=force)
    lines = [f"Batch run order: {' -> '.join(result['order'])}\n"]
    for sid, res in result["results"].items():
        if sid.startswith("_"):
            continue
        if res.get("skipped"):
            lines.append(f"  [{sid}] SKIPPED (outputs exist)")
        elif res["success"]:
            lines.append(f"  [{sid}] SUCCESS")
        else:
            lines.append(f"  [{sid}] FAILED: {res.get('error', '')[:200]}")
    if result["all_success"]:
        lines.append("\nAll steps completed successfully.")
    else:
        stopped = result["results"].get("_stopped_at", "?")
        lines.append(f"\nStopped at step {stopped}.")
    return "\n".join(lines)


@atlas_agent.tool
async def show_pipeline_dag(ctx: RunContext[PipelineRunner]) -> str:
    """Show the full pipeline DAG with completion status for all registered steps."""
    return ctx.deps.show_pipeline_dag()


@atlas_agent.tool
async def preview_nextflow_process(ctx: RunContext[PipelineRunner], step_id: str) -> str:
    """Preview the Nextflow DSL2 process block that would be generated for a
    pipeline step, WITHOUT writing it to main.nf. Useful for inspection."""
    return ctx.deps.generate_nextflow_process(step_id)


@atlas_agent.tool
async def add_steps_to_nextflow(ctx: RunContext[PipelineRunner], step_ids: str) -> str:
    """Add pipeline steps to main.nf as Nextflow processes and wire them into
    the workflow. Provide comma-separated IDs (e.g., '35,36,37,38,39').
    Automatically generates process blocks and dependency wiring.
    Existing processes are not duplicated."""
    ids = [s.strip() for s in step_ids.split(",")]
    result = ctx.deps.add_to_nextflow(ids)
    if result["success"]:
        added = result.get("added", [])
        if not added:
            return result.get("message", "Nothing to add.")
        return (f"Successfully added {len(added)} processes to main.nf:\n"
                + "\n".join(f"  - {ctx.deps._step_to_process_name(s)} (step {s})"
                           for s in added)
                + f"\n\nRun with: nextflow run main.nf -resume")
    return f"Failed: {result.get('error', 'Unknown error')}"


@atlas_agent.tool
async def preview_nextflow_wiring(ctx: RunContext[PipelineRunner], step_ids: str) -> str:
    """Preview the Nextflow workflow wiring that would connect the specified steps.
    Shows dependency resolution without modifying main.nf."""
    ids = [s.strip() for s in step_ids.split(",")]
    return ctx.deps.generate_workflow_wiring(ids)


@atlas_agent.tool
async def inspect_result(ctx: RunContext[PipelineRunner], path: str) -> str:
    """Load and inspect a result file (JSON or CSV). Use relative paths from
    the atlas root, e.g., 'results/causal_bqtl_728.csv'."""
    result = ctx.deps.load_result(path)
    if result is None:
        return f"File not found: {path}"
    return json.dumps(result, indent=2, default=str)[:4000]


@atlas_agent.tool
async def list_results(ctx: RunContext[PipelineRunner]) -> str:
    """List all result files in the results directory with sizes."""
    files = ctx.deps.list_results()
    if not files:
        return "No results directory found."
    lines = []
    for f in files:
        size = f["bytes"]
        size_str = f"{size/1_000_000:.1f}MB" if size > 1_000_000 else \
                   f"{size/1000:.1f}KB" if size > 1000 else f"{size}B"
        lines.append(f"  {f['name']:50s} {size_str:>8s}")
    return f"Results ({len(lines)} files):\n" + "\n".join(lines)


@atlas_agent.tool
async def list_figures(ctx: RunContext[PipelineRunner]) -> str:
    """List all generated figure PDFs."""
    figs = ctx.deps.list_figures()
    if not figs:
        return "No figures found."
    return f"Figures ({len(figs)} PDFs):\n" + "\n".join(f"  {f}" for f in figs)


@atlas_agent.tool
async def run_verification(ctx: RunContext[PipelineRunner]) -> str:
    """Run the pipeline verification script that checks result files
    and paper statistics for consistency."""
    return ctx.deps.run_verification()


@atlas_agent.tool
async def read_pipeline_config(ctx: RunContext[PipelineRunner]) -> str:
    """Read the current pipeline_config.py to inspect path settings."""
    return ctx.deps.read_config()[:3000]


@atlas_agent.tool
async def check_data_files(ctx: RunContext[PipelineRunner]) -> str:
    """Check which required data files exist in data/raw and data/processed."""
    raw = ctx.deps.base / "data" / "raw"
    proc = ctx.deps.base / "data" / "processed"
    lines = ["Data directory status:\n"]
    for label, d in [("data/raw", raw), ("data/processed", proc)]:
        if not d.exists():
            lines.append(f"  {label}: MISSING")
            continue
        files = sorted(d.iterdir())
        lines.append(f"  {label}: {len(files)} items")
        for f in files[:20]:
            if f.is_file():
                size = f.stat().st_size
                size_str = f"{size/1_000_000:.1f}MB" if size > 1_000_000 else \
                           f"{size/1000:.1f}KB"
                lines.append(f"    {f.name:50s} {size_str}")
            elif f.is_dir():
                n = len(list(f.iterdir()))
                lines.append(f"    {f.name + '/':50s} {n} files")
    return "\n".join(lines)


# ── Paper companion tools ─────────────────────────────────────────────

@atlas_agent.tool
async def list_paper_sections(ctx: RunContext[PipelineRunner]) -> str:
    """List all sections of the paper with their topics, linked pipeline steps, and figures."""
    sections = ctx.deps.paper.list_sections()
    lines = [f"Paper: {len(sections)} sections\n"]
    for i, sec in enumerate(sections):
        steps = ', '.join(sec.get('steps', [])) or 'n/a'
        figs = ', '.join(sec.get('figures', [])) if sec.get('figures') else ''
        line = f"  [{i:2d}] {sec['title']}"
        line += f"  (steps: {steps})"
        if figs:
            line += f"  [{figs}]"
        lines.append(line)
    return "\n".join(lines)


@atlas_agent.tool
async def read_paper_section(ctx: RunContext[PipelineRunner], section_index: int) -> str:
    """Read a specific paper section by index (from list_paper_sections).
    Returns the section text, linked pipeline steps, figures, and key statistics."""
    result = ctx.deps.paper.read_section(section_index)
    if "error" in result:
        return result["error"]
    lines = [
        f"## {result['title']}",
        f"\nSummary: {result['summary']}",
    ]
    if result["steps"]:
        lines.append(f"Pipeline steps: {', '.join(result['steps'])}")
    if result["figures"]:
        lines.append(f"Figures: {', '.join(result['figures'])}")
    if result["key_stats"]:
        lines.append(f"Key statistics: {', '.join(result['key_stats'])}")
    lines.append(f"\n--- LaTeX preview (first ~80 lines) ---\n{result['text_preview']}")
    return "\n".join(lines)


@atlas_agent.tool
async def verify_paper_statistics(ctx: RunContext[PipelineRunner]) -> str:
    """Cross-check all key statistics in the paper against actual result files.
    Reports matches and mismatches."""
    checks = ctx.deps.paper.verify_statistics()
    if not checks:
        return "Could not verify: no result files found."
    all_ok = all(c["match"] for c in checks)
    lines = [f"Paper verification: {'ALL PASS' if all_ok else 'MISMATCHES FOUND'}\n"]
    for c in checks:
        icon = "pass" if c["match"] else "FAIL"
        in_paper = "in paper" if c.get("in_paper") else "NOT in paper"
        lines.append(f"  [{icon}] {c['claim']}: actual={c['actual']} ({in_paper})")
    return "\n".join(lines)


@atlas_agent.tool
async def explain_figure(ctx: RunContext[PipelineRunner], figure_name: str) -> str:
    """Get detailed information about a paper figure: what it shows, which script
    generates it, which result files it depends on, and whether it exists."""
    info = ctx.deps.paper.figure_info(figure_name)
    if "error" in info:
        return info["error"]
    lines = [
        f"Figure: {figure_name}",
        f"Description: {info['desc']}",
        f"Generated by: scripts/{info['script']} (step {info['step']})",
        f"Depends on: {', '.join(info.get('results', []))}",
        f"Exists: {info['exists']}",
    ]
    if info.get("size_kb"):
        lines.append(f"Size: {info['size_kb']:.0f} KB")
    return "\n".join(lines)


@atlas_agent.tool
async def check_all_figures(ctx: RunContext[PipelineRunner]) -> str:
    """Check all paper figures: do they exist? Are they up to date with results?"""
    results = ctx.deps.paper.check_figures()
    lines = [f"Paper figures ({len(results)} total):\n"]
    for r in results:
        status = "ok" if r["exists"] else "MISSING"
        if r.get("stale"):
            status = f"STALE (result {r['newer_result']} is newer)"
        size = f"{r['size_kb']:.0f}KB" if r.get("size_kb") else ""
        lines.append(f"  [{status:8s}] {r['figure']:50s} {size}")
    return "\n".join(lines)


@atlas_agent.tool
async def regenerate_figure(ctx: RunContext[PipelineRunner], figure_name: str) -> str:
    """Regenerate a specific paper figure by running its generating script.
    Use explain_figure first to see which script will run."""
    result = ctx.deps.paper.regenerate_figure(figure_name, python=ctx.deps.python)
    if result["success"]:
        return f"Figure {figure_name} regenerated successfully.\n{result['stdout_tail']}"
    error = result.get("error", result.get("stderr_tail", "Unknown error"))
    return f"Figure regeneration FAILED: {error}"


# ── Interactive chat with Rich ────────────────────────────────────────

async def chat_loop(runner: PipelineRunner, model: str):
    """Interactive chat with the atlas agent using Rich console."""
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    console = Console()
    console.print(Panel.fit(
        "[bold cyan]Causal bQTL Atlas Agent[/bold cyan]\n"
        f"Model: {model}\n"
        "Type 'quit' to exit, 'status' for pipeline overview",
        border_style="cyan",
    ))

    while True:
        try:
            user_input = console.input("\n[bold green]you>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break
        if user_input.lower() == "status":
            status = runner.validate_all()
            complete = sum(1 for s in status.values() if s["outputs_complete"])
            console.print(f"\n[bold]Pipeline: {complete}/{len(status)} steps complete[/bold]")
            for sid, s in status.items():
                icon = "[green]done[/green]" if s["outputs_complete"] else "[yellow]pending[/yellow]"
                console.print(f"  {icon} [{sid}] {s['name']}")
            continue

        console.print("[dim]Thinking...[/dim]")
        try:
            result = await atlas_agent.run(
                user_input,
                deps=runner,
                model=model,
            )
            console.print()
            console.print(Markdown(result.output))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


# ── Non-interactive CLI commands ──────────────────────────────────────

def cmd_validate(runner: PipelineRunner):
    status = runner.validate_all()
    complete = sum(1 for s in status.values() if s["outputs_complete"])
    print(f"Pipeline status: {complete}/{len(status)} steps complete\n")
    for sid, s in status.items():
        icon = "done" if s["outputs_complete"] else "    "
        print(f"  [{icon}] [{sid}] {s['name']}")
        if s["missing_inputs"]:
            print(f"         Missing inputs: {', '.join(s['missing_inputs'])}")


def cmd_explain(runner: PipelineRunner):
    print("\n" + "=" * 60)
    print("  Causal bQTL Atlas — Results Summary")
    print("=" * 60)

    ww = runner.load_result("results/causal_bqtl_728.csv")
    if ww and "rows" in ww:
        print(f"\n  Well-watered causal bQTL: {ww['rows']} targets")

    comp = runner.load_result("results/ww_vs_ds_comparison.json")
    if comp and "error" not in comp:
        print(f"  Drought causal bQTL: {comp.get('n_ds_causal', '?')} targets")
        pct = comp.get('pct_ww_also_ds', 0)
        if isinstance(pct, (int, float)):
            print(f"  Overlap: {comp.get('n_overlap', '?')} positions ({pct:.1f}% of WW)")

    pred = runner.load_result("results/causality_prediction_results.json")
    if pred and "phase1_3to1" in pred:
        rf = pred["phase1_3to1"]["RF"]
        print(f"\n  Causality prediction (RF): AUC = {rf['auc_mean']:.3f} "
              f"+/- {rf['auc_std']:.3f}")

    gs = runner.load_result("results/condition_specific_gene_sets.json")
    if gs and "error" not in gs:
        print(f"\n  Gene set decomposition:")
        print(f"    WW-only: {len(gs.get('ww_only_genes', []))}")
        print(f"    DS-only: {len(gs.get('ds_only_genes', []))}")
        print(f"    Shared (same variant): {len(gs.get('shared_same_variant', []))}")
        print(f"    Shared (diff variant): {len(gs.get('shared_diff_variant', []))}")
    print()


def cmd_run(runner: PipelineRunner, step_ids: Optional[list[str]] = None):
    if step_ids:
        for sid in step_ids:
            print(f"\nRunning step {sid}...")
            result = runner.run_step(sid)
            if not result["success"]:
                print(f"Step {sid} failed: {result.get('error', '')}")
                return
            print(f"Step {sid} complete.")
    else:
        while True:
            next_ids = runner.get_next_steps()
            if not next_ids:
                print("\nAll steps complete (or blocked on missing inputs).")
                break
            for sid in next_ids:
                step = STEP_MAP[sid]
                print(f"\nRunning {sid}: {step.name}...")
                result = runner.run_step(sid)
                if not result["success"]:
                    print(f"Step {sid} failed: {result.get('error', '')}")
                    return
                print(f"Step {sid} complete.")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Causal bQTL Atlas Agent — orchestrate, validate, and explain "
                    "the causal bQTL analysis pipeline with Pydantic AI."
    )
    parser.add_argument("--base-dir", type=str,
                        default=str(Path(__file__).resolve().parent),
                        help="Atlas base directory")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python interpreter for running scripts")

    # Modes
    parser.add_argument("--validate", action="store_true",
                        help="Validate pipeline state and exit")
    parser.add_argument("--run", nargs="*", default=None,
                        help="Run pipeline steps (all if no args, or specific IDs)")
    parser.add_argument("--explain", action="store_true",
                        help="Print results summary and exit")

    # LLM settings
    parser.add_argument("--model", type=str, default="openai:gpt-4o",
                        help="Model string (e.g., 'openai:gpt-4o', "
                             "'anthropic:claude-sonnet-4-20250514', 'test')")

    args = parser.parse_args()
    runner = PipelineRunner(Path(args.base_dir), python=args.python)

    if args.validate:
        cmd_validate(runner)
        return
    if args.run is not None:
        cmd_run(runner, args.run if args.run else None)
        return
    if args.explain:
        cmd_explain(runner)
        return

    # Interactive chat
    asyncio.run(chat_loop(runner, args.model))


if __name__ == "__main__":
    main()
