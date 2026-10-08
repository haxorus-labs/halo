import asyncio
import json
import shutil

import pytest

from halo.sast.authmap import (
    auth_for_route,
    normalize_route,
    record_auth_probes,
    unguarded_routes,
    verdict_for_status,
)
from halo.sast.enrich import enrich_graph_json
from halo.sast.findings import STATE_REFUTED, Finding, FindingStore
from halo.sast.fingerprint import compute_fingerprint
from halo.sast.fixtures import load_fixtures, login_body_for, match_fixture
from halo.sast.gates import sanitizer_gate
from halo.sast.graph import build_graph, graph_path_for, query_graph_text
from halo.sast.paths import resolve_source_root
from halo.sast.runners import parse_gitleaks, parse_semgrep, parse_trivy
from halo.sast.taxonomy import load_recipes, match_recipe, recipe_for_finding
from halo.tools.kg import tools as kg_tools


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


def test_taxonomy_pack_loads_with_unique_ids_and_valid_regexes():
    recipes = load_recipes()
    assert len(recipes) >= 20
    assert len({r.id for r in recipes}) == len(recipes)
    # sorted by id for deterministic first-match-wins
    assert [r.id for r in recipes] == sorted(r.id for r in recipes)


def test_taxonomy_match_prefers_family_over_fallback():
    recipe = match_recipe(
        "semgrep",
        "javascript.sequelize.security.audit.sequelize-injection-express.express-sequelize-injection",
    )
    assert recipe is not None
    assert recipe.id == "sqli-sequelize"
    assert match_recipe("semgrep", "something.totally.unknown.rule") is None
    api_key = match_recipe("gitleaks", "generic-api-key")
    assert api_key is not None
    assert api_key.id == "secret-api-key"


def test_taxonomy_guard_hit_and_sanitizer_gate():
    recipe = match_recipe("semgrep", "python.flask.security.xss")
    _ = recipe  # unrelated family: generic gate still applies
    assert sanitizer_gate("eval(req.query.q)") == "undefended"
    sql = match_recipe(
        "semgrep", "javascript.sequelize.security.audit.foo.express-sequelize-injection"
    )
    assert sql is not None
    assert sql.guard_hit("db.query(q, { replacements: { id } })") is not None
    assert sql.guard_hit("db.query('SELECT ' + req.query.q)") is None
    assert sanitizer_gate("db.query(q, { replacements: { id } })", sql.guards) == "defended"


def test_parse_semgrep_uses_recipe_taxonomy():
    seq_check = (
        "javascript.sequelize.security.audit."
        "sequelize-injection-express.express-sequelize-injection"
    )
    payload = json.dumps(
        {
            "results": [
                {
                    "check_id": seq_check,
                    "path": "routes/search.ts",
                    "start": {"line": 10},
                    "extra": {"message": "m", "severity": "ERROR"},
                },
                {
                    "check_id": "totally.unknown.rule",
                    "path": "a.py",
                    "start": {"line": 1},
                    "extra": {"message": "m", "severity": "WARNING"},
                },
            ]
        }
    )
    findings = parse_semgrep(payload)
    assert findings[0].taxonomy == "sqli-sequelize"
    assert findings[0].id != findings[1].id
    assert findings[1].taxonomy == "semgrep"


def test_recipe_for_finding_prefers_stored_id():
    leak = {"RuleID": "generic-api-key", "File": "a.env", "StartLine": 1, "Description": "d"}
    findings = parse_gitleaks(json.dumps([leak]))
    assert len(findings) == 1
    assert findings[0].taxonomy == "secret-api-key"
    recipe = recipe_for_finding(findings[0])
    assert recipe is not None
    assert recipe.id == "secret-api-key"


def test_fixture_pack_loads_and_matches_juice_shop(tmp_path):
    fixtures = load_fixtures()
    assert {f.id for f in fixtures} == {"juice-shop"}
    by_url = match_fixture(repo_url="https://github.com/juice-shop/juice-shop")
    assert by_url is not None
    assert by_url.id == "juice-shop"
    by_path = match_fixture(source_path=str(tmp_path / "halo_repos" / "run1" / "juice-shop"))
    assert by_path is not None
    assert by_path.id == "juice-shop"
    assert match_fixture(repo_url="https://github.com/other/app") is None
    admin = next(p for p in by_url.personas if p.id == "admin")
    assert admin.role == "admin"
    body = login_body_for(by_url, admin)
    assert body == {"email": "admin@juice-sh.op", "password": "admin123"}
    anon = next(p for p in by_url.personas if p.id == "anon")
    assert login_body_for(by_url, anon) == {}


def test_authmap_record_query_unguarded(tmp_path):
    assert verdict_for_status(200) == "allows"
    assert verdict_for_status(401) == "denies"
    assert verdict_for_status(403) == "denies"
    assert verdict_for_status(500) == "inconclusive"
    assert verdict_for_status(302) == "inconclusive"
    assert normalize_route("https://host/api/x?y=1") == "/api/x"
    assert normalize_route("api/x") == "/api/x"

    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps({"nodes": [], "edges": []}))
    record_auth_probes(
        graph,
        [
            {"route": "/rest/admin", "persona": "anon", "verdict": "denies"},
            {"route": "/rest/admin", "persona": "admin", "verdict": "allows"},
            {"route": "/", "persona": "anon", "verdict": "allows"},
            {"route": "/rest/admin", "persona": "admin", "verdict": "allows"},
            {"route": "/x", "persona": "anon", "verdict": "inconclusive"},
        ],
    )
    data = json.loads(graph.read_text())
    assert auth_for_route(data, "/rest/admin") == {
        "allows": ["admin"],
        "denies": ["anon"],
    }
    assert auth_for_route(data, "/never-probed") == {"allows": [], "denies": []}
    assert unguarded_routes(data) == ["/"]


def test_kg_auth_tools_roundtrip(tmp_path):
    state = tmp_path / ".state"
    state.mkdir()
    kg_tools.hydrate_kg_from_disk(state)
    try:
        personas = asyncio.run(
            kg_tools._kg_auth_personas_impl("https://github.com/juice-shop/juice-shop")
        )
        assert personas["fixture"] == "juice-shop"
        assert {p["id"] for p in personas["personas"]} == {"anon", "customer", "admin"}

        rec = asyncio.run(kg_tools._kg_auth_record_impl("https://t/rest/admin/users", "anon", 401))
        assert rec["verdict"] == "denies"
        rec = asyncio.run(kg_tools._kg_auth_record_impl("/rest/admin/users", "admin", 200))
        assert rec["verdict"] == "allows"
        rec = asyncio.run(kg_tools._kg_auth_record_impl("/flaky", "anon", 500))
        assert rec["verdict"] == "inconclusive"

        looked = asyncio.run(kg_tools._kg_authmap_impl("/rest/admin/users"))
        assert looked == {
            "route": "/rest/admin/users",
            "allows": ["admin"],
            "denies": ["anon"],
        }
        unguarded = asyncio.run(kg_tools._kg_unguarded_impl())
        assert unguarded == {"unguarded": []}

        graph = json.loads((state.parent / "graphify-out" / "graph.json").read_text())
        assert any(n.get("type") == "actor" for n in graph["nodes"])
        assert any(n.get("type") == "route" for n in graph["nodes"])
    finally:
        kg_tools.hydrate_kg_from_disk.__globals__["_state_dir"] = None


def test_enrich_and_authmap_work_on_links_style_graph(tmp_path):
    graph = tmp_path / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "nodes": [{"id": "routes_login", "label": "login"}],
                "links": [{"source": "routes_login", "target": "db_query", "relation": "calls"}],
            }
        )
    )
    finding = Finding(
        id="f9",
        tool="semgrep",
        rule_id="r",
        file="routes/login.ts",
        line=3,
        sink_symbol="routes_login",
        message="m",
        taxonomy="t",
        severity="HIGH",
    )
    enrich_graph_json(graph, [finding])
    record_auth_probes(graph, [{"route": "/login", "persona": "anon", "verdict": "allows"}])
    data = json.loads(graph.read_text())
    assert "edges" not in data
    assert any(n["id"] == "finding:f9" for n in data["nodes"])
    assert any(n["id"] == "actor:anon" for n in data["nodes"])
    relations = {e["relation"] for e in data["links"]}
    assert {"targets", "allows"} <= relations
    assert auth_for_route(data, "/login") == {"allows": ["anon"], "denies": []}
