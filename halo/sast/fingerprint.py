"""Deterministic fingerprints for SAST findings."""

from __future__ import annotations

import hashlib


def compute_fingerprint(
    file_path: str,
    sink_symbol: str,
    ast_slice: str,
    taxonomy_id: str,
) -> str:
    """Stable finding ID: path + sink + AST slice + taxonomy."""
    payload = f"{file_path}\x00{sink_symbol}\x00{ast_slice}\x00{taxonomy_id}"
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
