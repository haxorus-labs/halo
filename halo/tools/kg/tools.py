"""Knowledge-graph agent tools backed by the graphify CLI."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

from agents import RunContextWrapper, function_tool

from halo.sast.graph import build_graph, graph_path_for, query_graph_text
from halo.sast.paths import resolve_source_root


_state_dir: Path | None = None
_default_source_roots: list[str] = []


def set_default_source_roots(roots: list[str]) -> None:
    global _default_source_roots  # noqa: PLW0603
    _default_source_roots = list(roots)


def hydrate_kg_from_disk(state_dir: Path) -> None:
    global _state_dir  # noqa: PLW0603
    _state_dir = state_dir


def _run_dir() -> Path:
    if _state_dir is None:
        raise RuntimeError("KG store not initialized; whitebox run required")
    return _state_dir.parent


def _graph_json() -> Path:
    return graph_path_for(_run_dir())


async def _kg_build_impl(source_path: str) -> dict[str, Any]:
    root = resolve_source_root(source_path, _default_source_roots)
    if root is None:
        return {"error": f"source path not found: {source_path or _default_source_roots}"}
    graph_json = await asyncio.to_thread(build_graph, root, _run_dir())
    if graph_json is None:
        return {"error": "graphify extraction failed or produced no graph.json"}
    return {"graph_json": str(graph_json)}


@function_tool(timeout=1200)
async def kg_build(ctx: RunContextWrapper, source_path: str = "") -> str:
    """Build the code knowledge graph (AST, call edges, communities).

    Parses the source tree with tree-sitter via graphify and writes
    graph.json under the run directory. Run once before kg_query/kg_path/kg_explain.

    Args:
        source_path: Absolute path to the local source tree. Defaults to the
            scanned local source / cloned repository root when omitted.
            Sandbox paths such as ``/workspace/<repo>`` are mapped to the
            matching host checkout automatically.
    """
    return json.dumps(await _kg_build_impl(source_path), ensure_ascii=False, default=str)


@function_tool(timeout=120)
async def kg_query(ctx: RunContextWrapper, query: str) -> str:
    """Semantic search over the code knowledge graph.

    Args:
        query: Free-text question (e.g. "routes that call the database").
    """
    return await asyncio.to_thread(query_graph_text, _graph_json(), "query", query)


@function_tool(timeout=120)
async def kg_path(ctx: RunContextWrapper, a: str, b: str) -> str:
    """Shortest path between two nodes in the code knowledge graph.

    Args:
        a: Source node label (e.g. a route handler name).
        b: Target node label (e.g. a sink function name).
    """
    return await asyncio.to_thread(query_graph_text, _graph_json(), "path", a, b)


@function_tool(timeout=120)
async def kg_explain(ctx: RunContextWrapper, node: str) -> str:
    """Plain-language explanation of a node and its neighbors.

    Args:
        node: Node label to explain.
    """
    return await asyncio.to_thread(query_graph_text, _graph_json(), "explain", node)
