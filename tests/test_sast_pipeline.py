import json
import shutil

import pytest

from halo.sast.enrich import enrich_graph_json
from halo.sast.findings import STATE_REFUTED, Finding, FindingStore
from halo.sast.fingerprint import compute_fingerprint
from halo.sast.gates import sanitizer_gate
from halo.sast.graph import build_graph, graph_path_for, query_graph_text
from halo.sast.paths import resolve_source_root
from halo.sast.runners import parse_gitleaks, parse_semgrep, parse_trivy


def test_fingerprint_is_deterministic():
    a = compute_fingerprint("app/main.py", "sink.x", "slice1", "semgrep")
    b = compute_fingerprint("app/main.py", "sink.x", "slice1", "semgrep")
    c = compute_fingerprint("app/main.py", "sink.y", "slice1", "semgrep")
    assert a == b
    assert a != c
    assert len(a) == 16


def test_parse_semgrep_normalizes_results():
    payload = json.dumps(
        {
            "results": [
                {
                    "check_id": "python.flask.security.xss",
                    "path": "app/views.py",
                    "start": {"line": 42},
                    "extra": {"message": "XSS sink", "severity": "ERROR"},
                }
            ]
        }
    )
    findings = parse_semgrep(payload)
    assert len(findings) == 1
    assert findings[0].tool == "semgrep"
    assert findings[0].rule_id == "python.flask.security.xss"
    assert findings[0].file == "app/views.py"
    assert findings[0].line == 42
    assert findings[0].severity == "ERROR"


def test_parse_semgrep_handles_garbage():
    assert parse_semgrep("not json") == []


def test_parse_gitleaks_normalizes_results():
    payload = json.dumps(
        [
            {
                "RuleID": "aws-access-token",
                "File": "config.env",
                "StartLine": 3,
                "Description": "AWS key",
            }
        ]
    )
    findings = parse_gitleaks(payload)
    assert len(findings) == 1
    assert findings[0].taxonomy == "secret"
    assert findings[0].line == 3


def test_parse_trivy_normalizes_results():
    payload = json.dumps(
        {
            "Results": [
                {
                    "Target": "requirements.txt",
                    "Vulnerabilities": [
                        {"VulnerabilityID": "CVE-2024-0001", "Title": "bad lib", "Severity": "HIGH"}
                    ],
                }
            ]
        }
    )
    findings = parse_trivy(payload)
    assert len(findings) == 1
    assert findings[0].taxonomy == "dependency_cve"
    assert findings[0].severity == "HIGH"


def test_finding_store_dedupes_and_updates(tmp_path):
    store = FindingStore(path=tmp_path / "findings.jsonl")
    f = Finding(
        id="abc123",
        tool="semgrep",
        rule_id="r",
        file="a.py",
        line=1,
        sink_symbol="a.py",
        message="m",
        taxonomy="semgrep",
        severity="HIGH",
    )
    store.record(f)
    store.set_state("abc123", STATE_REFUTED, reason="guarded")
    current = store.current()
    assert current["abc123"].state == STATE_REFUTED
    assert current["abc123"].reason == "guarded"
    assert store.is_settled("abc123")


def test_finding_store_rejects_bad_state(tmp_path):
    store = FindingStore(path=tmp_path / "findings.jsonl")
    try:
        store.set_state("x", "MAYBE")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_sanitizer_gate():
    assert sanitizer_gate('htmlspecialchars($_GET["x"])') == "defended"
    assert sanitizer_gate("eval($_GET['x'])") == "undefended"
    assert sanitizer_gate("   ") == "inconclusive"


def test_enrich_graph_json(tmp_path):
    graph = tmp_path / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "nodes": [{"id": "app/views.py", "label": "views.py"}],
                "edges": [],
            }
        )
    )
    f = Finding(
        id="f1",
        tool="semgrep",
        rule_id="r",
        file="app/views.py",
        line=7,
        sink_symbol="app/views.py",
        message="m",
        taxonomy="semgrep",
        severity="HIGH",
    )
    enrich_graph_json(graph, [f])
    data = json.loads(graph.read_text())
    ids = {n["id"] for n in data["nodes"]}
    assert "finding:f1" in ids
    assert any(e["relation"] == "targets" for e in data["edges"])


def test_build_graph_and_explain(tmp_path):

    if shutil.which("graphify") is None:
        pytest.skip("graphify CLI not installed")
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("def main():\n    login()\n\ndef login():\n    pass\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    graph_json = build_graph(src, run_dir)
    assert graph_json is not None
    assert graph_path_for(run_dir).is_file()
    out = query_graph_text(graph_json, "explain", "main")
    assert "main" in out


def test_resolve_source_root_prefers_existing_host_path(tmp_path):
    src = tmp_path / "repo"
    src.mkdir()
    assert resolve_source_root(str(src), []) == src
    assert resolve_source_root("", [str(src)]) == src


def test_resolve_source_root_maps_workspace_path(tmp_path):
    clone = tmp_path / "halo_repos" / "run1" / "juice-shop"
    clone.mkdir(parents=True)
    # sandbox path the agent sees; host checkout is the clone
    assert resolve_source_root("/workspace/juice-shop", [str(clone)]) == clone


def test_resolve_source_root_maps_workspace_subdir(tmp_path):
    clone = tmp_path / "halo_repos" / "run1" / "juice-shop"
    (clone / "routes").mkdir(parents=True)
    assert resolve_source_root("/workspace/juice-shop/routes", [str(clone)]) == clone / "routes"
    # missing subtree falls back to the repo root
    assert resolve_source_root("/workspace/juice-shop/nope", [str(clone)]) == clone


def test_resolve_source_root_unknown_without_roots():
    assert resolve_source_root("/workspace/juice-shop", []) is None
    assert resolve_source_root("", []) is None
    assert resolve_source_root("/does/not/exist", []) is None
