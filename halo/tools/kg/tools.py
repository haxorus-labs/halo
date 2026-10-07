"""Knowledge-graph agent tools backed by the graphify CLI."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

from agents import RunContextWrapper, function_tool

from halo.sast.authmap import (
    auth_for_route,
    normalize_route,
    record_auth_probes,
    unguarded_routes,
    verdict_for_status,
)
from halo.sast.fixtures import login_body_for, match_fixture
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


def _read_graph() -> dict[str, Any]:
    path = _graph_json()
    if not path.is_file():
        return {"nodes": [], "edges": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"nodes": [], "edges": []}
    return data if isinstance(data, dict) else {"nodes": [], "edges": []}


@function_tool(timeout=120)
async def kg_auth_personas(ctx: RunContextWrapper, repo_url: str = "") -> str:
    """List seeded login personas for the target (anon + known users/roles).

    Returns each persona's login route, request body template, and role so
    probed routes can be recorded with ``kg_auth_record``. Matches a fixture
    by repo URL, else by the scanned source root.

    Args:
        repo_url: Repository URL to match a fixture (optional when the scan
            target already identifies the source).
    """
    return json.dumps(await _kg_auth_personas_impl(repo_url), ensure_ascii=False, default=str)


async def _kg_auth_personas_impl(repo_url: str) -> dict[str, Any]:
    source = repo_url.strip() or (resolve_source_root("", _default_source_roots) or "")
    fixture = match_fixture(repo_url=repo_url, source_path=str(source))
    if fixture is None:
        return {"personas": [], "hint": "no fixture matched; probe manually"}
    return {
        "fixture": fixture.id,
        "login": {
            "route": fixture.login_route,
            "method": fixture.login_method,
            "token_path": fixture.token_path,
        },
        "personas": [
            {
                "id": p.id,
                "role": p.role,
                "body": login_body_for(fixture, p),
            }
            for p in fixture.personas
        ],
        "denied_statuses": list(fixture.denied_statuses),
        "notes": fixture.notes,
    }


@function_tool(timeout=120)
async def kg_auth_record(
    ctx: RunContextWrapper, route: str, persona: str, status: int, evidence: str = ""
) -> str:
    """Record one measured authorization probe in the knowledge graph.

    Log in (or not) as ``persona``, request ``route``, and store the observed
    HTTP ``status`` as an allows/denies edge. Non-allow/deny statuses are
    reported as inconclusive without writing an edge.

    Args:
        route: Probed route path or URL.
        persona: Persona id from ``kg_auth_personas`` (e.g. anon, customer, admin).
        status: Observed HTTP status code.
        evidence: Optional note (e.g. redirect target, body marker).
    """
    return json.dumps(await _kg_auth_record_impl(route, persona, status, evidence))


async def _kg_auth_record_impl(
    route: str, persona: str, status: int, evidence: str = ""
) -> dict[str, Any]:
    verdict = verdict_for_status(status)
    if verdict == "inconclusive":
        return {"route": normalize_route(route), "persona": persona, "verdict": verdict}
    record_auth_probes(
        _graph_json(),
        [
            {
                "route": route,
                "persona": persona,
                "verdict": verdict,
                "evidence": evidence,
            }
        ],
    )
    return {
        "route": normalize_route(route),
        "persona": persona,
        "status": status,
        "verdict": verdict,
    }


@function_tool(timeout=120)
async def kg_authmap(ctx: RunContextWrapper, route: str) -> str:
    """Who can reach a route, per recorded probes (allows/denies by persona).

    Args:
        route: Route path or URL to look up.
    """
    return json.dumps(await _kg_authmap_impl(route), ensure_ascii=False)


@function_tool(timeout=120)
async def kg_unguarded(ctx: RunContextWrapper) -> str:
    """Routes with a recorded anon allows edge (publicly reachable surface)."""
    return json.dumps(await _kg_unguarded_impl(), ensure_ascii=False)


async def _kg_authmap_impl(route: str) -> dict[str, Any]:
    result = auth_for_route(_read_graph(), route)
    return {"route": normalize_route(route), **result}


async def _kg_unguarded_impl() -> dict[str, Any]:
    return {"unguarded": unguarded_routes(_read_graph())}
