"""
Vision Desktop Agent — CLI entry point.

    python src/main.py [OPTIONS]
    python src/main.py --help
    python src/main.py run --iterations 3
    python src/main.py ground --query "Notepad icon" --screenshot path/to/img.png
    python src/main.py benchmark
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils.logger import configure_logging, get_logger

app = typer.Typer(
    name="vision-agent",
    help="Vision-based desktop automation with dynamic icon grounding.",
    add_completion=False,
)
console = Console()
logger = get_logger(__name__)


# ─── Shared options ───────────────────────────────────────────────────────────

def _setup(log_level: str, log_file: Optional[str]) -> None:
    configure_logging(
        level=log_level.upper(),
        log_file=log_file,
        json_logs=False,
    )


# ─── Commands ─────────────────────────────────────────────────────────────────

@app.command()
def run(
    iterations: int = typer.Option(3, "--iterations", "-n", help="Workflow cycles (0=infinite)"),
    posts: int = typer.Option(10, "--posts", "-p", help="Posts per iteration"),
    output_dir: str = typer.Option(
        "C:/Users/Public/Desktop/tjm-project",
        "--output", "-o",
        help="Directory to save Notepad files",
    ),
    confidence: float = typer.Option(0.25, "--confidence", "-c", help="Min detection confidence"),
    retries: int = typer.Option(3, "--retries", "-r", help="Detection retry attempts"),
    no_ocr: bool = typer.Option(False, "--no-ocr", help="Disable OCR scoring bonus"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
    log_file: Optional[str] = typer.Option(None, "--log-file"),
) -> None:
    """
    Run the full Notepad automation workflow.

    Locates the Notepad icon via dynamic visual grounding, opens it,
    writes the first N posts from JSONPlaceholder, saves each as a .txt
    file, then repeats for the specified number of iterations.
    """
    _setup(log_level, log_file)

    console.print(
        Panel.fit(
            "[bold cyan]Vision Desktop Agent[/bold cyan]\n"
            f"Iterations: {iterations or 'infinite'}  |  Posts/iter: {posts}  |  "
            f"Confidence: {confidence}  |  Output: {output_dir}",
            title="Starting",
        )
    )

    from src.automation.desktop_controller import DesktopController
    from src.automation.notepad_workflow import NotepadWorkflow
    from src.api.posts_client import PostsClient
    from src.grounding import VisualGrounder
    from src.utils.screenshot import ScreenshotCapture

    grounder = VisualGrounder(
        confidence_threshold=confidence,
        use_ocr=not no_ocr,
    )
    workflow = NotepadWorkflow(
        output_dir=Path(output_dir),
        controller=DesktopController(),
        grounder=grounder,
        capture=ScreenshotCapture(),
        api_client=PostsClient(),
        detection_retries=retries,
        post_limit=posts,
        iterations=iterations,
    )

    t0 = time.perf_counter()
    workflow.run()
    elapsed = time.perf_counter() - t0

    console.print(
        f"\n[bold green]Done![/bold green] Completed in {elapsed:.1f}s"
    )


@app.command()
def ground(
    query: str = typer.Option("Notepad", "--query", "-q", help="Target element name or query"),
    screenshot: Optional[str] = typer.Option(
        None, "--screenshot", "-s",
        help="Path to screenshot PNG (omit to capture live)"
    ),
    confidence: float = typer.Option(0.20, "--confidence", "-c"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Show top-k results"),
    save_annotated: Optional[str] = typer.Option(
        None, "--save", help="Save annotated result image to this path"
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
    log_file: Optional[str] = typer.Option(None, "--log-file"),
) -> None:
    """
    Run grounding on a screenshot and show the top-k detections.

    Useful for debugging and tuning confidence thresholds without
    running the full automation workflow.
    """
    _setup(log_level, log_file)

    import cv2
    import numpy as np
    from src.grounding import VisualGrounder
    from src.grounding.detector import RegionProposalDetector
    from src.grounding.ranking import CandidateRanker
    from src.grounding.embeddings import CLIPEmbeddingExtractor
    from src.utils.screenshot import ScreenshotCapture

    # Load or capture screenshot
    if screenshot:
        img = cv2.imread(screenshot)
        if img is None:
            console.print(f"[red]Could not load screenshot: {screenshot}[/red]")
            raise typer.Exit(1)
        console.print(f"Loaded screenshot: {screenshot} ({img.shape[1]}×{img.shape[0]})")
    else:
        console.print("Capturing live screenshot…")
        img = ScreenshotCapture().capture()
        console.print(f"Captured: {img.shape[1]}×{img.shape[0]}")

    # Run pipeline with verbose top-k output
    extractor = CLIPEmbeddingExtractor()
    detector = RegionProposalDetector(max_proposals=200)
    ranker = CandidateRanker(extractor=extractor, confidence_threshold=0.01)

    console.print(f"[cyan]Generating region proposals…[/cyan]")
    proposals = detector.propose(img)
    console.print(f"  {len(proposals)} proposals generated")

    console.print(f"[cyan]Ranking against query: '{query}'…[/cyan]")
    full_query = f"a {query} application icon on a Windows desktop, small square icon"
    ranked = ranker.rank(proposals, full_query, query.lower())

    # Display results table
    table = Table(title=f"Top-{top_k} Detections for '{query}'")
    table.add_column("Rank", style="bold")
    table.add_column("X", justify="right")
    table.add_column("Y", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("CLIP", justify="right")
    table.add_column("OCR", justify="right")
    table.add_column("Detected Text")

    for i, r in enumerate(ranked[:top_k], 1):
        style = "green" if r.confidence >= confidence else "yellow"
        table.add_row(
            str(i),
            str(r.x),
            str(r.y),
            f"[{style}]{r.confidence:.3f}[/{style}]",
            f"{r.clip_score:.3f}",
            f"{r.ocr_score:.3f}",
            r.detected_text[:30] if r.detected_text else "—",
        )

    console.print(table)

    if not ranked:
        console.print("[yellow]No detections found.[/yellow]")
        return

    # Save annotated image if requested
    if save_annotated and ranked:
        from src.utils.screenshot import ScreenshotCapture as SC
        best = ranked[0]
        annotated = SC.annotate(
            img,
            x=best.box.x, y=best.box.y,
            w=best.box.w, h=best.box.h,
            confidence=best.confidence,
            label=query,
        )
        cv2.imwrite(save_annotated, annotated)
        console.print(f"[green]Annotated image saved: {save_annotated}[/green]")


@app.command()
def benchmark(
    screenshot: Optional[str] = typer.Option(None, "--screenshot", "-s"),
    runs: int = typer.Option(5, "--runs", "-n", help="Number of benchmark runs"),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """
    Benchmark grounding pipeline performance (proposals + ranking).
    """
    _setup(log_level, None)
    import psutil
    import cv2
    from src.grounding.detector import RegionProposalDetector
    from src.grounding.embeddings import CLIPEmbeddingExtractor
    from src.grounding.ranking import CandidateRanker
    from src.utils.screenshot import ScreenshotCapture

    img = cv2.imread(screenshot) if screenshot else ScreenshotCapture().capture()
    console.print(f"Benchmarking on {img.shape[1]}×{img.shape[0]} image, {runs} runs…\n")

    extractor = CLIPEmbeddingExtractor()  # Warm up model
    detector = RegionProposalDetector(max_proposals=200)
    ranker = CandidateRanker(extractor=extractor, confidence_threshold=0.01)

    proposal_times, ranking_times = [], []
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / 1024 / 1024  # MB

    for i in range(runs):
        t0 = time.perf_counter()
        proposals = detector.propose(img)
        t1 = time.perf_counter()
        ranker.rank(proposals, "Notepad application icon on desktop", "notepad")
        t2 = time.perf_counter()
        proposal_times.append(t1 - t0)
        ranking_times.append(t2 - t1)
        console.print(f"  Run {i+1}: proposals={t1-t0:.3f}s  ranking={t2-t1:.3f}s")

    mem_after = proc.memory_info().rss / 1024 / 1024
    cpu = psutil.cpu_percent(interval=0.1)

    table = Table(title="Benchmark Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    avg_prop = sum(proposal_times) / len(proposal_times)
    avg_rank = sum(ranking_times) / len(ranking_times)
    table.add_row("Avg proposal time", f"{avg_prop*1000:.0f} ms")
    table.add_row("Avg ranking time", f"{avg_rank*1000:.0f} ms")
    table.add_row("Avg total pipeline", f"{(avg_prop+avg_rank)*1000:.0f} ms")
    table.add_row("Memory delta", f"{mem_after - mem_before:.1f} MB")
    table.add_row("CPU (snapshot)", f"{cpu:.1f}%")
    console.print(table)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
