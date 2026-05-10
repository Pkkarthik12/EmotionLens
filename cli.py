"""
EmotionLens Command-Line Interface
===================================

    emotionlens predict "I can't believe how amazing this is!"
    emotionlens predict --audio speech.wav
    emotionlens predict "Today was hard" --audio sad.wav --explain
    emotionlens batch --file sentences.txt --output results.jsonl
    emotionlens serve --port 8000
    emotionlens demo
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Optional

try:
    import click
except ImportError:
    raise ImportError("CLI requires click: pip install click")


@click.group()
@click.version_option(version="0.1.0", prog_name="emotionlens")
def cli():
    """EmotionLens — multimodal emotion intelligence."""


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("text", default=None, required=False)
@click.option("--audio", "-a", type=click.Path(exists=True), default=None,
              help="Path to an audio file.")
@click.option("--image", "-i", type=click.Path(exists=True), default=None,
              help="Path to an image file.")
@click.option("--explain/--no-explain", default=True,
              help="Print natural-language explanation.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Output raw JSON.")
@click.option("--fusion", default="confidence_gating",
              type=click.Choice(["weighted_average", "confidence_gating", "attention"]),
              show_default=True, help="Fusion strategy.")
def predict(text, audio, image, explain, as_json, fusion):
    """Predict emotion from TEXT and/or AUDIO / IMAGE inputs."""
    from emotionlens.pipeline import EmotionPipeline

    if not any([text, audio, image]):
        click.echo("Error: provide at least one of TEXT, --audio, or --image.", err=True)
        sys.exit(1)

    pipe = EmotionPipeline(fusion_strategy=fusion, explain=explain)

    click.echo("Analysing…", err=True)
    result = pipe.predict(text=text, audio_path=audio, image_path=image)

    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        _pretty_print(result)


def _pretty_print(result):
    from emotionlens.emotions import EmotionResult
    COLORS = {
        "joy": "bright_yellow",
        "sadness": "blue",
        "anger": "red",
        "fear": "magenta",
        "surprise": "cyan",
        "disgust": "green",
        "contempt": "white",
        "neutral": "white",
    }
    label = result.label.value
    color = COLORS.get(label, "white")

    click.echo("")
    click.echo(click.style(f"  ★  {label.upper()}  ★", fg=color, bold=True))
    bar = "█" * int(result.confidence * 20) + "░" * (20 - int(result.confidence * 20))
    click.echo(f"  Confidence : {bar}  {result.confidence:.1%}")
    v, a, d = result.vad.valence, result.vad.arousal, result.vad.dominance
    click.echo(f"  VAD        : V={v:+.2f}  A={a:+.2f}  D={d:+.2f}")

    if result.fusion_weights:
        fw = "  ".join(f"{k}={v:.0%}" for k, v in result.fusion_weights.items())
        click.echo(f"  Fusion     : {fw}")

    click.echo("")
    if result.explanation:
        wrapped = textwrap.fill(result.explanation, width=72, subsequent_indent="     ")
        click.echo(f"  Why: {wrapped}")

    click.echo("")
    click.echo("  Score distribution:")
    sorted_scores = sorted(result.all_scores.items(), key=lambda x: x[1], reverse=True)
    for emo, sc in sorted_scores:
        bar = "▓" * int(sc * 30)
        click.echo(f"    {emo:<12} {bar:<32} {sc:.1%}")
    click.echo("")


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--file", "-f", "input_file", required=True, type=click.Path(exists=True),
              help="Input file with one text per line.")
@click.option("--output", "-o", default=None,
              help="Output JSONL file (default: print to stdout).")
@click.option("--fusion", default="confidence_gating",
              type=click.Choice(["weighted_average", "confidence_gating", "attention"]))
def batch(input_file, output, fusion):
    """Predict emotions for every line in a text file."""
    from emotionlens.pipeline import EmotionPipeline

    texts = Path(input_file).read_text().splitlines()
    texts = [t.strip() for t in texts if t.strip()]
    click.echo(f"Processing {len(texts)} texts…", err=True)

    pipe = EmotionPipeline(fusion_strategy=fusion, explain=False)
    out_fh = open(output, "w") if output else sys.stdout

    try:
        for i, result in enumerate(pipe.stream_predict(texts)):
            line = json.dumps({"index": i, **result.to_dict()})
            out_fh.write(line + "\n")
            if (i + 1) % 10 == 0:
                click.echo(f"  {i+1}/{len(texts)} done", err=True)
    finally:
        if output:
            out_fh.close()

    click.echo(f"Done. {len(texts)} predictions written.", err=True)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option("--reload", is_flag=True, default=False, help="Hot-reload on code changes.")
def serve(host, port, reload):
    """Start the EmotionLens REST API server."""
    try:
        import uvicorn
    except ImportError:
        click.echo("Error: uvicorn is required. pip install uvicorn", err=True)
        sys.exit(1)

    click.echo(f"Starting EmotionLens API on http://{host}:{port}")
    uvicorn.run("emotionlens.api:app", host=host, port=port, reload=reload)


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

@cli.command()
def demo():
    """Run a quick interactive demo with example sentences."""
    from emotionlens.pipeline import EmotionPipeline

    EXAMPLES = [
        "I just got the promotion I've been working toward for three years!",
        "I can't stop crying. Everything feels hopeless.",
        "How DARE they cancel the event without any warning!",
        "There's something behind you... don't move.",
        "Wait, you're already married?! I had no idea.",
        "Meh. Another Monday.",
    ]

    pipe = EmotionPipeline(explain=True)

    click.echo(click.style("\n  EmotionLens Interactive Demo\n", bold=True))
    for sentence in EXAMPLES:
        click.echo(click.style(f'  "{sentence}"', fg="bright_white"))
        result = pipe.predict(text=sentence)
        _pretty_print(result)
        input("  Press Enter for the next example …\n")


if __name__ == "__main__":
    cli()
