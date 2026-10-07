"""Static analysis pipeline for Halo whitebox scans."""

from halo.sast.enrich import enrich_graph_json
from halo.sast.findings import (
    STATE_PROPOSED,
    STATE_REFUTED,
    STATE_UNVERIFIED,
    STATE_VERIFIED,
    Finding,
    FindingStore,
)
from halo.sast.fingerprint import compute_fingerprint
from halo.sast.gates import adversarial_verify, sanitizer_gate
from halo.sast.graph import build_graph, graph_path_for, query_graph_text
from halo.sast.runners import parse_gitleaks, parse_semgrep, parse_trivy, scan_source


__all__ = [
    "STATE_PROPOSED",
    "STATE_REFUTED",
    "STATE_UNVERIFIED",
    "STATE_VERIFIED",
    "Finding",
    "FindingStore",
    "adversarial_verify",
    "build_graph",
    "compute_fingerprint",
    "enrich_graph_json",
    "graph_path_for",
    "parse_gitleaks",
    "parse_semgrep",
    "parse_trivy",
    "query_graph_text",
    "sanitizer_gate",
    "scan_source",
]
