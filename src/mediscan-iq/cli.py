from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import settings
from .logging import get_logger
from .preprocess.anonymizer import anonymize
from .preprocess.segmenter import split_sentences
from .services.analyze import Analyzer
from .nlp.summarizer import Summarizer
from .nlp.risk_tagger import RiskTagger

app = typer.Typer(add_completion=False)
logger = get_logger("mediscan.cli")

REPORT_TYPES = settings.accepted_report_types


@app.command("ingest")
def ingest_cli(
    file: Path = typer.Argument(..., exists=True, readable=True, help="Path to raw report .txt"),
    report_type: str = typer.Option(..., "--type", "-t", help=f"Report type: {', '.join(REPORT_TYPES)}"),
    show_sentences: bool = typer.Option(True, "--sentences/--no-sentences", help="Split into sentences"),
    dump_only: bool = typer.Option(False, "--dump", help="Print anonymized text only"),
):
    """Ingest a clinical report file: validate, anonymize, optionally sentence-split."""
    if report_type not in REPORT_TYPES:
        typer.secho(f"[error] Unsupported report type: {report_type}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    text = file.read_text(encoding="utf-8").strip()
    if len(text) > settings.max_chars:
        typer.secho(
            f"[error] File too large ({len(text)} chars). Max: {settings.max_chars}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(3)

    anon, counts = anonymize(text)

    if dump_only:
        typer.echo(anon)
        raise typer.Exit(0)

    typer.secho("== Ingestion Summary ==", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"Report type     : {report_type}")
    typer.echo(f"Chars (raw)     : {len(text)}")
    typer.echo(f"PHI detections  : {', '.join(f'{k}:{v}' for k, v in counts.items()) or 'none'}")

    if show_sentences:
        sents = split_sentences(anon)
        typer.echo(f"Sentences       : {len(sents)}")
        typer.secho("\n-- First 5 sentences (anonymized) --", fg=typer.colors.BLUE)
        for i, s in enumerate(sents[:5], 1):
            typer.echo(f"[{i}] {s}")

    out = Path("anonymized_" + file.name)
    out.write_text(anon, encoding="utf-8")
    typer.secho(f"\nSaved anonymized text -> {out}", fg=typer.colors.GREEN, bold=True)


@app.command("summarize")
def summarize_cli(
    file: Path = typer.Argument(..., exists=True, readable=True),
    report_type: str = typer.Option("radiology", "--type", "-t"),
):
    """Summarize a clinical report (anonymize -> summarize)."""
    if report_type not in REPORT_TYPES:
        typer.secho(f"[error] Unsupported report type: {report_type}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    text = file.read_text(encoding="utf-8")
    anon, _ = anonymize(text)
    summ = Summarizer()
    out, meta = summ.summarize(anon, report_type=report_type)
    typer.secho("== Summary ==", fg=typer.colors.CYAN, bold=True)
    typer.echo(out)
    typer.secho(f"\n[meta] {meta}", fg=typer.colors.BLUE)


@app.command("risk")
def risk_cli(
    file: Path = typer.Argument(..., exists=True, readable=True),
):
    """Risk-tag a clinical report (anonymize -> risk tagging)."""
    text = file.read_text(encoding="utf-8")
    anon, _ = anonymize(text)
    tagger = RiskTagger()
    label, probs, meta = tagger.tag(anon)
    typer.secho("== Risk ==", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"Risk level: {label}")
    typer.echo("Probabilities:")
    for k, v in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
        typer.echo(f"  {k:12s}: {v:.3f}")
    typer.secho(f"\n[meta] {meta}", fg=typer.colors.BLUE)


@app.command("analyze")
def analyze_cli(
    file: Path = typer.Argument(..., exists=True, readable=True),
    report_type: str = typer.Option("radiology", "--type", "-t"),
):
    """Full pipeline: anonymize -> sentence split -> summarize -> risk."""
    if report_type not in REPORT_TYPES:
        typer.secho(f"[error] Unsupported report type: {report_type}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    text = file.read_text(encoding="utf-8")
    a = Analyzer()
    out = a.run(text, report_type=report_type)
    typer.secho("== Summary ==", fg=typer.colors.CYAN, bold=True)
    typer.echo(out.summary)
    typer.secho("\n== Risk Tagger ==", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"Risk level: {out.risk_level}")
    for k, v in sorted(out.risk_probs.items(), key=lambda kv: kv[1], reverse=True):
        typer.echo(f"  {k:12s}: {v:.3f}")
    typer.secho("\n== Meta ==", fg=typer.colors.CYAN, bold=True)
    for k, v in out.meta.items():
        typer.echo(f"{k}: {v}")


if __name__ == "__main__":
    app()
