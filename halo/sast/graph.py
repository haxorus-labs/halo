"""Knowledge-graph build + query via the graphify CLI."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path  # noqa: TC003


logger = logging.getLogger(__name__)

_GRAPHIFY_TIMEOUT_SECONDS = 900


def graph_path_for(run_dir: Path) -> Path:
    return run_dir / "graphify-out" / "graph.json"


def _copy_into_run_dir(src_graph: Path, run_dir: Path) -> Path | None:
    dest = graph_path_for(run_dir)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_graph, dest)
    except OSError as exc:
        logger.warning("could not copy graph.json into run dir: %s", exc)
        return None
    return dest if dest.is_file() else None


def build_graph(source_path: Path, run_dir: Path) -> Path | None:
    """Extract the code knowledge graph for a local source tree.

    Uses ``graphify update`` (AST-only, no LLM needed) rather than
    ``graphify extract``: ``extract`` aborts the whole build when a repo
    contains docs/papers/images but no semantic-extraction LLM key is set.
    The graph is copied into the run dir so artifacts stay with the run.
    """
    graphify = shutil.which("graphify")
    if graphify is None:
        logger.warning("graphify CLI not on PATH; kg tools unavailable")
        return None
    try:
        proc = subprocess.run(  # noqa: S603
            [graphify, "update", str(source_path)],
            capture_output=True,
            text=True,
            timeout=_GRAPHIFY_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("graphify update failed: %s", exc)
        return None
    if proc.returncode != 0:
        logger.warning("graphify update failed (rc=%s): %s", proc.returncode, proc.stderr[:400])
        return None
    # `update` writes next to the source; relocate into the run dir.
    src_graph = source_path / "graphify-out" / "graph.json"
    if src_graph.is_file():
        dest = _copy_into_run_dir(src_graph, run_dir)
        if dest is not None:
            try:
                if str(source_path.resolve()).startswith(tempfile.gettempdir()):
                    shutil.rmtree(source_path / "graphify-out", ignore_errors=True)
            except OSError:
                pass
            return dest
    candidate = graph_path_for(run_dir)
    if candidate.is_file():
        return candidate
    logger.warning("graphify update produced no graph.json (rc=%s)", proc.returncode)
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
