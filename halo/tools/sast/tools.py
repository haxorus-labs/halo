"""SAST agent tools: scan stated source, verify, and track finding states."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

from agents import RunContextWrapper, function_tool

from halo.config import load_settings
from halo.sast.enrich import enrich_graph_json
from halo.sast.findings import STATE_REFUTED, STATE_UNVERIFIED, VALID_STATES, FindingStore
from halo.sast.gates import adversarial_verify, sanitizer_gate
from halo.sast.graph import graph_path_for
from halo.sast.paths import resolve_source_root
from halo.sast.runners import scan_source
from halo.sast.taxonomy import recipe_for_finding


logger = logging.getLogger(__name__)

_store: FindingStore | None = None
_default_source_roots: list[str] = []


def set_default_source_roots(roots: list[str]) -> None:
    global _default_source_roots  # noqa: PLW0603
    _default_source_roots = list(roots)


def hydrate_sast_from_disk(state_dir: Path) -> None:
    global _store  # noqa: PLW0603
    _store = FindingStore(path=state_dir / "findings.jsonl")


def _require_store() -> FindingStore:
    if _store is None:
        raise RuntimeError("SAST store not initialized; whitebox run required")
    return _store


def _run_dir() -> Path:
    store = _require_store()
    return store.path.parent.parent


async def _sast_scan_impl(source_path: str) -> dict[str, Any]:
    store = _require_store()
    root = resolve_source_root(source_path, _default_source_roots)
    if root is None:
        return {"error": f"source path not found: {source_path or _default_source_roots}"}

    candidates = await asyncio.to_thread(scan_source, root)
    known = store.current()
    new_findings = [f for f in candidates if f.id not in known]

    for finding in new_findings:
        store.record(finding)

    graph_json = graph_path_for(_run_dir())
    if graph_json.is_file() and new_findings:
        await asyncio.to_thread(enrich_graph_json, graph_json, new_findings)

    return {
        "new_findings": [f.to_dict() for f in new_findings],
        "new_count": len(new_findings),
        "known_count": len(known),
        "states": {s: sum(1 for f in known.values() if f.state == s) for s in VALID_STATES},
    }


@function_tool(timeout=1200)
async def sast_scan(ctx: RunContextWrapper, source_path: str = "") -> str:
    """Run semgrep, gitleaks and trivy over a local source tree.

    Normalizes raw output into fingerprinted findings, dedupes against the
    run's knowledge store (findings.jsonl), enriches the knowledge graph
    with finding nodes, and returns only new findings.

    Args:
        source_path: Absolute path to the local source tree to analyze.
            Defaults to the scanned local source / cloned repository root
            when omitted. Sandbox paths such as ``/workspace/<repo>`` are
            mapped to the matching host checkout automatically.
    """
    return json.dumps(await _sast_scan_impl(source_path), ensure_ascii=False, default=str)


async def _finding_verify_impl(finding_id: str, source_slice: str) -> dict[str, Any]:
    store = _require_store()
    finding = store.get(finding_id)
    if finding is None:
        return {"error": f"unknown finding_id: {finding_id}"}

    recipe = recipe_for_finding(finding)
    gate = sanitizer_gate(source_slice, recipe.guards if recipe is not None else ())
    if gate == "defended":
        store.set_state(finding_id, STATE_REFUTED, reason="adequate guard/sanitizer present")
        return {
            "finding_id": finding_id,
            "state": STATE_REFUTED,
            "reason": "sanitizer gate: defended",
        }

    verdict = await adversarial_verify(finding, source_slice, load_settings(), recipe)
    mapping = {"CONFIRMED": "VERIFIED", "REFUTED": STATE_REFUTED, "UNVERIFIED": STATE_UNVERIFIED}
    state = mapping.get(verdict["verdict"], STATE_UNVERIFIED)
    store.set_state(finding_id, state, reason=verdict["reason"])
    return {"finding_id": finding_id, "state": state, "reason": verdict["reason"]}


@function_tool(timeout=120)
async def finding_verify(ctx: RunContextWrapper, finding_id: str, source_slice: str) -> str:
    """Adversarially verify a finding with an LLM pass over its code slice.

    Runs the sanitizer gate first; if the sink looks adequately guarded the
    finding is REFUTED without spending LLM tokens. Otherwise one LLM call
    adjudicates CONFIRMED / REFUTED / UNVERIFIED and updates the store.

    Args:
        finding_id: Fingerprint id returned by sast_scan.
        source_slice: The relevant code around the sink (a few dozen lines).
    """
    return json.dumps(
        await _finding_verify_impl(finding_id, source_slice), ensure_ascii=False, default=str
    )


@function_tool(timeout=30)
async def finding_state(
    ctx: RunContextWrapper, finding_id: str, state: str, reason: str = ""
) -> str:
    """Manually set a finding's state.

    Args:
        finding_id: Fingerprint id returned by sast_scan.
        state: One of PROPOSED, REFUTED, UNVERIFIED, VERIFIED.
        reason: Optional justification recorded with the state change.
    """
    store = _require_store()
    try:
        finding = store.set_state(finding_id, state, reason=reason)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    if finding is None:
        return json.dumps({"error": f"unknown finding_id: {finding_id}"})
    return json.dumps(finding.to_dict(), ensure_ascii=False, default=str)
