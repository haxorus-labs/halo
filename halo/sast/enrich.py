"""Add SAST findings as nodes/edges into a graphify graph.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path  # noqa: TC003
from typing import Any, cast

from halo.sast.findings import Finding  # noqa: TC001
from halo.sast.graph import writable_edge_list


logger = logging.getLogger(__name__)


def enrich_graph_json(graph_json: Path, findings: list[Finding]) -> Path:
    try:
        data: dict[str, Any] = json.loads(graph_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("graph.json unreadable at %s; skipping enrichment", graph_json)
        return graph_json

    nodes_raw = data.get("nodes")
    if not isinstance(nodes_raw, list):
        return graph_json
    nodes = cast("list[dict[str, Any]]", nodes_raw)
    edges = writable_edge_list(data)

    node_ids = {str(n.get("id")) for n in nodes}
    for finding in findings:
        node_id = f"finding:{finding.id}"
        if node_id not in node_ids:
            nodes.append(
                {
                    "id": node_id,
                    "label": f"{finding.tool}:{finding.rule_id}",
                    "source_file": finding.file,
                    "source_location": f"L{finding.line}",
                    "type": "finding",
                    "state": finding.state,
                    "severity": finding.severity,
                    "taxonomy": finding.taxonomy,
                }
            )
            node_ids.add(node_id)
        symbol_id = finding.sink_symbol
        if symbol_id and symbol_id in node_ids:
            edges.append(
                {
                    "source": node_id,
                    "target": symbol_id,
                    "relation": "targets",
                    "confidence": "EXTRACTED",
                }
            )

    graph_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return graph_json
