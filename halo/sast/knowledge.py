"""Per-machine knowledge persistence across scans.

``~/.halo/knowledge/<target-key>/`` keeps the last ``graph.json`` and
``findings.jsonl`` per target so terminal finding states (REFUTED/VERIFIED)
survive across runs. Unchanged fingerprints keep their verdict; anything
new or changed re-enters as PROPOSED. Run dirs stay the source of truth
for a single run; this store is a cache. ``HALO_KNOWLEDGE_DIR`` overrides
the location (used by tests).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from halo.sast.findings import STATE_REFUTED, STATE_VERIFIED, Finding


logger = logging.getLogger(__name__)

CARRIED_STATES = frozenset({STATE_REFUTED, STATE_VERIFIED})


def knowledge_dir() -> Path:
    override = os.environ.get("HALO_KNOWLEDGE_DIR", "").strip()
    return Path(override).expanduser() if override else Path.home() / ".halo" / "knowledge"


def target_key(source_roots: list[str], repo_url: str = "") -> str:
    """Stable per-target identity: repo URL when known, else resolved paths."""
    if repo_url.strip():
        payload = f"repo:{repo_url.strip().lower()}"
    else:
        resolved = sorted({str(Path(r).expanduser().resolve()) for r in source_roots if r.strip()})
        payload = "paths:" + "\x00".join(resolved)
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def snapshot_dir(key: str) -> Path:
    return knowledge_dir() / key


def load_snapshot_states(key: str) -> dict[str, Finding]:
    """Terminal finding states from the previous snapshot (empty when none)."""
    path = snapshot_dir(key) / "findings.jsonl"
    if not path.is_file():
        return {}
    states: dict[str, Finding] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record: Any = json.loads(line)
            finding = Finding.from_dict(record)
        except (json.JSONDecodeError, TypeError):
            continue
        if finding.state in CARRIED_STATES:
            states[finding.id] = finding
    return states


def save_snapshot(key: str, graph_json: Path | None, findings_path: Path) -> Path | None:
    """Persist the current findings + graph as this target's latest snapshot."""
    if not findings_path.is_file():
        return None
    dest = snapshot_dir(key)
    try:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(findings_path, dest / "findings.jsonl")
        if graph_json is not None and graph_json.is_file():
            shutil.copy2(graph_json, dest / "graph.json")
    except OSError as exc:
        logger.warning("knowledge snapshot failed for %s: %s", key, exc)
        return None
    return dest


def prune_knowledge(max_age_days: int = 90) -> list[str]:
    """Delete snapshots older than ``max_age_days``; return removed keys."""
    base = knowledge_dir()
    if not base.is_dir():
        return []
    cutoff = time.time() - max_age_days * 86400
    removed: list[str] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            continue
        if mtime < cutoff:
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child.name)
    return removed
