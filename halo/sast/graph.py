"""Knowledge-graph build + query via the graphify CLI."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path  # noqa: TC003


logger = logging.getLogger(__name__)

_GRAPHIFY_TIMEOUT_SECONDS = 900


def graph_path_for(run_dir: Path) -> Path:
    return run_dir / "graphify-out" / "graph.json"


def build_graph(source_path: Path, run_dir: Path) -> Path | None:
    """Extract the code knowledge graph for a local source tree."""
    if shutil.which("graphify") is None:
        logger.warning("graphify CLI not on PATH; kg tools unavailable")
        return None
    try:
        proc = subprocess.run(  # noqa: S603
            [
                shutil.which("graphify") or "graphify",
                "extract",
                str(source_path),
                "--out",
                str(run_dir),
            ],
            capture_output=True,
            text=True,
            timeout=_GRAPHIFY_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("graphify extract failed: %s", exc)
        return None
    candidate = graph_path_for(run_dir)
    if candidate.is_file():
        return candidate
    alt = source_path / "graphify-out" / "graph.json"
    if alt.is_file():
        return alt
    logger.warning(
        "graphify extract produced no graph.json (rc=%s): %s", proc.returncode, proc.stderr[:400]
    )
    return None


def query_graph_text(graph_json: Path, kind: str, *parts: str) -> str:
    """Run `graphify query|path|explain --graph ...` and return stdout."""
    if shutil.which("graphify") is None or not graph_json.is_file():
        return f"graph unavailable: graphify missing or {graph_json} not found"
    cmd = [shutil.which("graphify") or "graphify", kind, *parts, "--graph", str(graph_json)]
    try:
        proc = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"graphify {kind} failed: {exc}"
    output = proc.stdout.strip()
    return output or proc.stderr.strip() or f"no output from graphify {kind}"
