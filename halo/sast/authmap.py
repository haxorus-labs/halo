"""Dynamic authorization map: observed allow/deny edges in the knowledge graph.

Probes are measured at runtime (log in as a persona, request a route, record
the status), never guessed from source. ``actor:*`` and ``route:*`` nodes plus
``allows``/``denies`` edges are appended to ``graph.json`` with INFERRED
confidence, powering ``kg_authmap`` / ``kg_unguarded`` queries for IDOR and
broken-access-control reasoning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path  # noqa: TC003
from typing import Any, cast
from urllib.parse import urlsplit


logger = logging.getLogger(__name__)

ALLOWS = "allows"
DENIES = "denies"
INCONCLUSIVE = "inconclusive"


def verdict_for_status(status: int, denied_statuses: tuple[int, ...] = (401, 403)) -> str:
    """Map an observed HTTP status to allows / denies / inconclusive."""
    if 200 <= status < 300:
        return ALLOWS
    if status in denied_statuses:
        return DENIES
    return INCONCLUSIVE


def normalize_route(route: str) -> str:
    """Reduce a route to its path (``https://host/api/x?y=1`` -> ``/api/x``)."""
    text = (route or "").strip()
    if not text:
        return "/"
    if "://" in text:
        return urlsplit(text).path or "/"
    path = text.split("?", 1)[0].split("#", 1)[0]
    return path if path.startswith("/") else f"/{path}"


def auth_nodes_for_probe(
    route: str, persona: str, verdict: str, evidence: str = ""
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build actor/route nodes plus the allows/denies edge for one probe."""
    path = normalize_route(route)
    actor_id = f"actor:{persona}"
    route_id = f"route:{path}"
    nodes = [
        {
            "id": actor_id,
            "label": persona,
            "source_file": "",
            "source_location": "",
            "type": "actor",
            "role": persona,
        },
        {
            "id": route_id,
            "label": path,
            "source_file": "",
            "source_location": "",
            "type": "route",
            "evidence": evidence,
        },
    ]
    edges = [
        {
            "source": actor_id,
            "target": route_id,
            "relation": verdict,
            "confidence": "INFERRED",
        }
    ]
    return nodes, edges


def record_auth_probes(graph_json: Path, probes: list[dict[str, Any]]) -> Path:
    """Append auth nodes/edges for measured probes; idempotent per edge."""
    try:
        data: dict[str, Any] = json.loads(graph_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {"nodes": [], "edges": []}

    nodes_raw = data.get("nodes")
    edges_raw = data.get("edges")
    if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
        return graph_json
    nodes = cast("list[dict[str, Any]]", nodes_raw)
    edges = cast("list[dict[str, Any]]", edges_raw)

    node_ids = {str(n.get("id")) for n in nodes}
    edge_keys = {
        (str(e.get("source")), str(e.get("target")), str(e.get("relation"))) for e in edges
    }
    for probe in probes:
        verdict = str(probe.get("verdict") or "")
        if verdict not in (ALLOWS, DENIES):
            continue
        new_nodes, new_edges = auth_nodes_for_probe(
            str(probe.get("route") or ""),
            str(probe.get("persona") or ""),
            verdict,
            str(probe.get("evidence") or ""),
        )
        for node in new_nodes:
            if node["id"] not in node_ids:
                nodes.append(node)
                node_ids.add(str(node["id"]))
        for edge in new_edges:
            key = (edge["source"], edge["target"], edge["relation"])
            if key not in edge_keys:
                edges.append(edge)
                edge_keys.add(key)

    try:
        graph_json.parent.mkdir(parents=True, exist_ok=True)
        graph_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("graph.json unwritable at %s; skipping auth enrichment", graph_json)
    return graph_json


def auth_for_route(graph: dict[str, Any], route: str) -> dict[str, list[str]]:
    """Personas with allows/denies edges to a route (empty lists when unprobed)."""
    path = normalize_route(route)
    target = f"route:{path}"
    result: dict[str, list[str]] = {ALLOWS: [], DENIES: []}
    edges = graph.get("edges")
    if not isinstance(edges, list):
        return result
    for edge in edges:
        if not isinstance(edge, dict) or str(edge.get("target")) != target:
            continue
        relation = str(edge.get("relation"))
        if relation not in result:
            continue
        source = str(edge.get("source"))
        if source.startswith("actor:"):
            result[relation].append(source[len("actor:") :])
    result[ALLOWS].sort()
    result[DENIES].sort()
    return result


def unguarded_routes(graph: dict[str, Any]) -> list[str]:
    """Routes with a recorded anon ``allows`` edge (publicly reachable)."""
    edges = graph.get("edges")
    if not isinstance(edges, list):
        return []
    routes = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        if str(edge.get("source")) == "actor:anon" and str(edge.get("relation")) == ALLOWS:
            target = str(edge.get("target"))
            if target.startswith("route:"):
                routes.add(target[len("route:") :])
    return sorted(routes)
