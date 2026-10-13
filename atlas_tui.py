#!/usr/bin/env python3
"""
atlas_tui.py — Interactive TUI for the Causal bQTL Atlas pipeline.

A terminal UI built with Textual that provides:
- Pipeline status overview with live updates
- Result file browser and inspector
- Figure listing
- Integrated chat with the Pydantic AI agent
- Key findings dashboard

Usage:
  python atlas_tui.py
  python atlas_tui.py --model openai:gpt-4o
  python atlas_tui.py --base-dir /path/to/atlas
"""

import sys
import json
import asyncio
import argparse
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Header, Footer, Static, Tree, RichLog, Input,
    DataTable, TabbedContent, TabPane, Markdown, Label,
)
from textual.reactive import reactive
from rich.text import Text

from atlas_agent import PipelineRunner, PIPELINE_STEPS, STEP_MAP, atlas_agent


# ── Pipeline Status Widget ────────────────────────────────────────────

class PipelineStatus(Static):
    """Shows pipeline step completion status."""

    def __init__(self, runner: PipelineRunner, **kwargs):
        super().__init__(**kwargs)
        self.runner = runner

    def on_mount(self) -> None:
        self.refresh_status()

    def refresh_status(self) -> None:
        status = self.runner.validate_all()
        complete = sum(1 for s in status.values() if s["outputs_complete"])
        total = len(status)
        lines = [f"[bold]Pipeline: {complete}/{total} steps complete[/bold]\n"]
        for sid, s in status.items():
            if s["outputs_complete"]:
                lines.append(f"  [green]●[/green] [{sid}] {s['name']}")
            else:
                lines.append(f"  [yellow]○[/yellow] [{sid}] {s['name']}")
                if s["missing_inputs"]:
                    lines.append(f"    [dim]Missing: {', '.join(s['missing_inputs'])}[/dim]")
        self.update("\n".join(lines))


# ── Findings Dashboard ────────────────────────────────────────────────

class FindingsDashboard(Static):
    """Shows key results summary."""

    def __init__(self, runner: PipelineRunner, **kwargs):
        super().__init__(**kwargs)
        self.runner = runner

    def on_mount(self) -> None:
        self.refresh_findings()

    def refresh_findings(self) -> None:
        lines = ["[bold cyan]Key Findings[/bold cyan]\n"]

        ww = self.runner.load_result("results/causal_bqtl_728.csv")
        if ww and "rows" in ww:
            lines.append(f"  Well-watered causal bQTL: [bold]{ww['rows']}[/bold]")

        comp = self.runner.load_result("results/ww_vs_ds_comparison.json")
        if comp and "error" not in comp:
            lines.append(f"  Drought causal bQTL: [bold]{comp.get('n_ds_causal', '?')}[/bold]")
            pct = comp.get('pct_ww_also_ds', 0)
            if isinstance(pct, (int, float)):
                lines.append(f"  Overlap: {comp.get('n_overlap', '?')} ({pct:.1f}%)")

        pred = self.runner.load_result("results/causality_prediction_results.json")
        if pred and "phase1_3to1" in pred:
            rf = pred["phase1_3to1"]["RF"]
            lines.append(f"\n  Causality RF AUC: [bold]{rf['auc_mean']:.3f}[/bold] "
                        f"± {rf['auc_std']:.3f}")

        gs = self.runner.load_result("results/condition_specific_gene_sets.json")
        if gs and "error" not in gs:
            lines.append(f"\n  [bold]Gene sets:[/bold]")
            lines.append(f"    WW-only: {len(gs.get('ww_only_genes', []))}")
            lines.append(f"    DS-only: {len(gs.get('ds_only_genes', []))}")
            lines.append(f"    Shared-same: {len(gs.get('shared_same_variant', []))}")
            lines.append(f"    Shared-diff: {len(gs.get('shared_diff_variant', []))}")

        if len(lines) == 1:
            lines.append("  [dim]No results found. Run the pipeline first.[/dim]")

        self.update("\n".join(lines))


# ── Result Inspector ──────────────────────────────────────────────────

class ResultTree(Tree):
    """File tree for results directory."""

    def __init__(self, runner: PipelineRunner, **kwargs):
        super().__init__("Results", **kwargs)
        self.runner = runner

    def on_mount(self) -> None:
        self.refresh_tree()

    def refresh_tree(self) -> None:
        self.clear()
        self.root.expand()

        # Results
        results_node = self.root.add("results/", expand=True)
        for f in self.runner.list_results():
            size = f["bytes"]
            size_str = f"{size/1_000_000:.1f}MB" if size > 1_000_000 else \
                       f"{size/1000:.1f}KB" if size > 1000 else f"{size}B"
            results_node.add_leaf(f"{f['name']}  ({size_str})",
                                 data=f"results/{f['name']}")

        # Figures
        figs = self.runner.list_figures()
        if figs:
            fig_node = self.root.add(f"figures/ ({len(figs)} PDFs)", expand=False)
            for name in figs:
                fig_node.add_leaf(name, data=f"figures/paper_figures/{name}")


class ResultViewer(RichLog):
    """Displays contents of selected result file."""
    pass


# ── Chat Widget ───────────────────────────────────────────────────────

class ChatLog(RichLog):
    """Chat message log."""
    pass


class ChatInput(Input):
    """Chat input field."""
    pass


# ── Main TUI App ─────────────────────────────────────────────────────

class AtlasTUI(App):
    """Causal bQTL Atlas — Interactive Terminal UI."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #main-tabs {
        height: 1fr;
    }
    #status-panel {
        height: 1fr;
        overflow-y: auto;
    }
    #findings-panel {
        height: 1fr;
        overflow-y: auto;
    }
    #overview-left {
        width: 1fr;
        height: 1fr;
    }
    #overview-right {
        width: 1fr;
        height: 1fr;
    }
    #result-tree {
        width: 2fr;
        min-width: 30;
        height: 1fr;
    }
    #result-viewer {
        width: 3fr;
        height: 1fr;
        border: solid $accent;
    }
    #chat-log {
        height: 1fr;
        border: solid $accent;
    }
    #chat-input {
        dock: bottom;
        height: 3;
    }
    PipelineStatus {
        padding: 1 2;
    }
    FindingsDashboard {
        padding: 1 2;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("ctrl+l", "clear_chat", "Clear chat"),
    ]

    TITLE = "Causal bQTL Atlas"
    SUB_TITLE = "Interactive Pipeline Explorer"

    def __init__(self, runner: PipelineRunner, model: str = "openai:gpt-4o"):
        super().__init__()
        self.runner = runner
        self.model = model

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(id="main-tabs"):
            with TabPane("Overview", id="tab-overview"):
                with Horizontal():
                    with Vertical(id="overview-left"):
                        yield PipelineStatus(self.runner, id="status-panel")
                    with Vertical(id="overview-right"):
                        yield FindingsDashboard(self.runner, id="findings-panel")

            with TabPane("Results", id="tab-results"):
                with Horizontal():
                    yield ResultTree(self.runner, id="result-tree")
                    yield ResultViewer(id="result-viewer", highlight=True,
                                       markup=True, wrap=True)

            with TabPane("Chat", id="tab-chat"):
                yield ChatLog(id="chat-log", highlight=True, markup=True,
                             wrap=True, auto_scroll=True)
                yield ChatInput(placeholder="Ask about the pipeline, results, "
                               "or biology... (Enter to send)",
                               id="chat-input")

            with TabPane("Steps", id="tab-steps"):
                yield Markdown(self._steps_markdown(), id="steps-md")

        yield Footer()

    def _steps_markdown(self) -> str:
        lines = ["# Pipeline Steps\n"]
        status = self.runner.validate_all()
        for step in PIPELINE_STEPS:
            s = status.get(step.id, {})
            icon = "+" if s.get("outputs_complete") else "-"
            lines.append(f"## [{step.id}] {step.name}")
            lines.append(f"**Status:** {'Complete' if s.get('outputs_complete') else 'Pending'}")
            lines.append(f"**Script:** `{step.script}`")
            lines.append(f"**Description:** {step.description}")
            if step.depends_on:
                lines.append(f"**Depends on:** {', '.join(step.depends_on)}")
            if step.outputs:
                lines.append(f"**Outputs:** {', '.join(step.outputs)}")
            lines.append("")
        return "\n".join(lines)

    # ── Event handlers ──

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """When a result file is clicked, show its contents."""
        if event.node.data:
            path = event.node.data
            viewer = self.query_one("#result-viewer", RichLog)
            viewer.clear()
            viewer.write(f"[bold]{path}[/bold]\n")

            result = self.runner.load_result(path)
            if result is None:
                viewer.write("[red]File not found[/red]")
            elif "error" in result:
                viewer.write(f"[red]Error: {result['error']}[/red]")
            else:
                formatted = json.dumps(result, indent=2, default=str)
                # Truncate for display
                if len(formatted) > 5000:
                    formatted = formatted[:5000] + "\n\n... (truncated)"
                viewer.write(formatted)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle chat input."""
        if event.input.id != "chat-input":
            return
        msg = event.value.strip()
        if not msg:
            return

        chat_log = self.query_one("#chat-log", RichLog)
        chat_input = self.query_one("#chat-input", Input)
        chat_input.value = ""

        chat_log.write(f"[bold green]you>[/bold green] {msg}")

        if msg.lower() in ("quit", "exit"):
            self.exit()
            return
        if msg.lower() == "status":
            status = self.runner.validate_all()
            complete = sum(1 for s in status.values() if s["outputs_complete"])
            chat_log.write(f"\n[bold]Pipeline: {complete}/{len(status)} complete[/bold]")
            return

        chat_log.write("[dim]Thinking...[/dim]")
        try:
            result = await atlas_agent.run(
                msg,
                deps=self.runner,
                model=self.model,
            )
            chat_log.write(f"\n[bold cyan]agent>[/bold cyan] {result.output}\n")
        except Exception as e:
            chat_log.write(f"\n[red]Error: {e}[/red]\n")

    def action_refresh(self) -> None:
        """Refresh pipeline status."""
        try:
            self.query_one("#status-panel", PipelineStatus).refresh_status()
        except Exception:
            pass
        try:
            self.query_one("#findings-panel", FindingsDashboard).refresh_findings()
        except Exception:
            pass
        try:
            self.query_one("#result-tree", ResultTree).refresh_tree()
        except Exception:
            pass

    def action_clear_chat(self) -> None:
        try:
            self.query_one("#chat-log", RichLog).clear()
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Causal bQTL Atlas TUI — interactive pipeline explorer."
    )
    parser.add_argument("--base-dir", type=str,
                        default=str(Path(__file__).resolve().parent))
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--model", type=str, default="openai:gpt-4o",
                        help="Model for chat (e.g., openai:gpt-4o)")
    args = parser.parse_args()

    runner = PipelineRunner(Path(args.base_dir), python=args.python)
    app = AtlasTUI(runner, model=args.model)
    app.run()


if __name__ == "__main__":
    main()
