"""Tests for the authorization matrix engine and the ``auth_matrix`` tool."""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
import tempfile
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from agents import ModelSettings

import halo.tools.notes.tools as notes_tools
import halo.tools.todo.tools as todo_tools
from halo.agents import factory
from halo.core import runner
from halo.core.agents import AgentCoordinator
from halo.runtime import session_manager
from halo.tools.verify.auth_matrix import _auth_matrix_impl, auth_matrix
from halo.verify.matrix import (
    ALLOWED,
    DENIED,
    EMPTY,
    INCONCLUSIVE,
    REDIRECT,
    MatrixCell,
    MatrixPersona,
    build_requests,
    compute_candidates,
    ensure_anon_persona,
    evidence_verdict,
    matrix_to_dict,
    operation_from_dict,
    persona_from_dict,
    substitute_token,
    verdict_for_cell,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


ALICE_CRED = "alice-token"
BOB_CRED = "bob-token"

#: ``(method, path) -> {token: (status, body)}``. ``None`` is the anonymous
#: caller (no ``Authorization`` header at all).
ROUTES: dict[tuple[str, str], dict[str | None, tuple[int, str]]] = {
    # The IDOR: the peer's token reads the owner's object.
    ("GET", "/api/orders/1"): {
        ALICE_CRED: (200, '{"id":1,"owner":"alice"}'),
        BOB_CRED: (200, '{"id":1,"owner":"alice"}'),
        None: (401, '{"error":"login required"}'),
    },
    # Silent enforcement: the peer gets a 200 with no object data.
    ("GET", "/api/orders/2"): {
        ALICE_CRED: (200, '{"id":2,"owner":"alice"}'),
        BOB_CRED: (200, ""),
        None: (401, '{"error":"login required"}'),
    },
    # Admin-only data that every caller can read.
    ("GET", "/api/admin/stats"): {
        ALICE_CRED: (200, '{"users":10}'),
        BOB_CRED: (200, '{"users":10}'),
        None: (403, '{"error":"forbidden"}'),
    },
    # Unauthenticated reachability of a protected route.
    ("GET", "/api/reports/7"): {
        ALICE_CRED: (200, '{"report":"q3"}'),
        BOB_CRED: (403, '{"error":"forbidden"}'),
        None: (200, '{"report":"q3"}'),
    },
    # Same path, different methods: the write is open where the read is not.
    ("GET", "/api/notes"): {
        ALICE_CRED: (403, '{"error":"forbidden"}'),
        BOB_CRED: (403, '{"error":"forbidden"}'),
        None: (403, '{"error":"forbidden"}'),
    },
    ("POST", "/api/notes"): {
        ALICE_CRED: (200, '{"note":"created"}'),
        BOB_CRED: (200, '{"note":"created"}'),
        None: (403, '{"error":"forbidden"}'),
    },
    # A redirect is not a served object.
    ("GET", "/api/legacy"): {
        ALICE_CRED: (302, ""),
        BOB_CRED: (302, ""),
        None: (302, ""),
    },
    # An error nobody can judge.
    ("GET", "/api/boom"): {
        ALICE_CRED: (500, "boom"),
        BOB_CRED: (500, "boom"),
        None: (500, "boom"),
    },
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:
        return

    def _respond(self, method: str) -> None:
        path = self.path.split("?", 1)[0]
        route = ROUTES.get((method, path))
        if route is None:
            self.send_response(404)
            self.end_headers()
            return
        header = self.headers.get("Authorization")
        token: str | None = header if header in route else None
        status, body = route.get(token, (403, '{"error":"forbidden"}'))
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def do_GET(self) -> None:
        self._respond("GET")

    def do_POST(self) -> None:
        self._respond("POST")


@pytest.fixture
def base_url() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _ctx(**entries: Any) -> Any:
    return SimpleNamespace(context=dict(entries))


def _personas() -> list[dict[str, Any]]:
    return [
        {"id": "alice", "role": "user", "token": ALICE_CRED},
        {"id": "bob", "role": "user", "token": BOB_CRED},
    ]


def _operations() -> list[dict[str, Any]]:
    return [
        {"id": "order1", "method": "GET", "path": "/api/orders/1", "owner": "alice"},
        {"id": "order2", "method": "GET", "path": "/api/orders/2", "owner": "alice"},
        {
            "id": "admin",
            "method": "GET",
            "path": "/api/admin/stats",
            "allowed_roles": ["admin"],
        },
        {"id": "reports", "method": "GET", "path": "/api/reports/7", "owner": "alice"},
        {"id": "notes_read", "method": "GET", "path": "/api/notes"},
        {"id": "notes_write", "method": "POST", "path": "/api/notes"},
        {"id": "legacy", "method": "GET", "path": "/api/legacy"},
        {"id": "boom", "method": "GET", "path": "/api/boom"},
    ]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def test_substitute_token_replaces_every_placeholder() -> None:
    assert substitute_token("Bearer {token}", "abc") == "Bearer abc"
    assert substitute_token("{token}/{token}", "abc") == "abc/abc"
    assert substitute_token("", "abc") == ""
    assert substitute_token("plain", "") == "plain"


def test_verdict_classifies_ambiguous_answers() -> None:
    assert verdict_for_cell(MatrixCell("op", "p", 200, body_sample="{}")) == EMPTY
    assert verdict_for_cell(MatrixCell("op", "p", 200, body_sample="[]")) == EMPTY
    assert verdict_for_cell(MatrixCell("op", "p", 200, body_sample="null")) == EMPTY
    assert verdict_for_cell(MatrixCell("op", "p", 200, body_sample="secret")) == ALLOWED
    assert verdict_for_cell(MatrixCell("op", "p", 403)) == DENIED
    assert verdict_for_cell(MatrixCell("op", "p", 302)) == REDIRECT
    assert verdict_for_cell(MatrixCell("op", "p", 500)) == INCONCLUSIVE
    assert verdict_for_cell(MatrixCell("op", "p", None, error="boom")) == INCONCLUSIVE


def test_evidence_verdict_filters_ambiguous_answers() -> None:
    assert evidence_verdict(ALLOWED) == ALLOWED
    assert evidence_verdict(DENIED) == DENIED
    assert evidence_verdict(EMPTY) is None
    assert evidence_verdict(REDIRECT) is None
    assert evidence_verdict(INCONCLUSIVE) is None


def test_ensure_anon_persona_adds_anonymous_principal() -> None:
    personas = ensure_anon_persona([MatrixPersona(id="alice")])
    assert [p.id for p in personas] == ["alice", "anon"]
    assert personas[-1].is_anon
    # An explicitly supplied anon persona is kept as-is.
    again = ensure_anon_persona(personas)
    assert [p.id for p in again] == ["alice", "anon"]


def test_build_requests_substitutes_tokens_and_adds_authorization() -> None:
    operations = [operation_from_dict({"id": "op", "path": "/x/{token}"})]
    personas = [persona_from_dict({"id": "alice", "token": "t"})]
    plan = build_requests(operations, personas, "http://h/")
    assert len(plan) == 1
    _op, _persona, request = plan[0]
    assert request.url == "http://h/x/t"
    assert dict(request.headers)["Authorization"] == "t"


def test_compute_candidates_flags_differential_access() -> None:
    operations = [
        operation_from_dict({"id": "order1", "path": "/api/orders/1", "owner": "alice"}),
        operation_from_dict(
            {
                "id": "admin",
                "path": "/api/admin/stats",
                "allowed_roles": ["admin"],
            }
        ),
    ]
    personas = [
        persona_from_dict({"id": "alice", "role": "user"}),
        persona_from_dict({"id": "bob", "role": "user"}),
    ]
    cells = [
        MatrixCell("order1", "alice", 200, body_sample='{"id":1}'),
        MatrixCell("order1", "bob", 200, body_sample='{"id":1}'),
        MatrixCell("admin", "alice", 200, body_sample='{"users":10}'),
        MatrixCell("admin", "bob", 200, body_sample='{"users":10}'),
    ]
    rules = {c.rule for c in compute_candidates(operations, personas, cells)}
    assert "horizontal_idor" in rules
    assert "vertical_escalation" in rules


def test_matrix_to_dict_keeps_inconclusive_cells() -> None:
    cells = [MatrixCell("op", "p", None, error="timeout")]
    payload = matrix_to_dict([], [], cells, [])
    assert payload["inconclusive_cells"]
    assert payload["counts"]["inconclusive"] == 1


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def test_auth_matrix_reports_idor_and_keeps_empty_body_peer_unflagged(
    base_url: str,
) -> None:
    out = _run_matrix(base_url)
    rules = {c["rule"] for c in out["candidates"]}
    assert "horizontal_idor" in rules
    assert "vertical_escalation" in rules
    assert "anonymous_access" in rules
    assert "method_asymmetry" in rules

    # The peer read of a 200-with-empty-body route is not an exposure.
    empty_peer = [
        c
        for c in out["candidates"]
        if c["rule"] == "horizontal_idor" and c["operation_id"] == "order2"
    ]
    assert empty_peer == []

    idor = next(c for c in out["candidates"] if c["rule"] == "horizontal_idor")
    assert idor["operation_id"] == "order1"
    assert idor["persona_id"] == "bob"
    assert idor["evidence"]
    assert idor["counterevidence"]
    assert idor["advisory"] is True


def test_auth_matrix_keeps_redirects_and_errors_out_of_evidence(base_url: str) -> None:
    out = _run_matrix(base_url)
    flagged_operations = {c["operation_id"] for c in out["candidates"]}
    assert "legacy" not in flagged_operations
    assert "boom" not in flagged_operations
    verdicts = {row["verdict"] for row in out["cells"] if row["operation_id"] == "legacy"}
    assert verdicts == {REDIRECT}
    assert out["inconclusive_cells"]
    assert all(row["operation_id"] == "boom" for row in out["inconclusive_cells"])


def test_auth_matrix_adds_the_anonymous_persona(base_url: str) -> None:
    out = _run_matrix(base_url)
    assert "anon" in {persona["id"] for persona in out["personas"]}


def test_auth_matrix_reports_errors_without_raising(base_url: str) -> None:
    assert "error" in _impl("", _operations(), _personas())
    assert "error" in _impl(base_url, [], _personas())
    bad = _impl(base_url, [{"id": "no-path"}], _personas())
    assert "error" in bad


def test_auth_matrix_output_is_json_serializable(base_url: str) -> None:
    out = _run_matrix(base_url)
    assert json.loads(json.dumps(out)) == out


def test_auth_matrix_tool_is_registerable_for_every_agent() -> None:
    """The runner registers it for all scans; the tool must survive that."""
    saved = list(factory._EXTRA_TOOLS)
    factory._EXTRA_TOOLS.clear()
    try:
        factory.register_agent_tools(auth_matrix)
        agent = factory.build_halo_agent(is_root=True)
        assert "auth_matrix" in [tool.name for tool in agent.tools]
    finally:
        factory._EXTRA_TOOLS[:] = saved


@pytest.mark.asyncio
async def test_runner_registers_auth_matrix_for_non_whitebox_scans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """The tool must reach the agents of a plain (non-white-box) scan."""
    monkeypatch.setattr(runner, "run_dir_for", lambda _scan_id: tmp_path)
    monkeypatch.setattr(runner, "runtime_state_dir", lambda _run_dir: tmp_path)
    monkeypatch.setattr(runner, "setup_scan_logging", lambda _run_dir: lambda: None)
    monkeypatch.setattr(runner, "set_scan_id", lambda _scan_id: None)
    monkeypatch.setattr(runner, "load_settings", _settings)
    monkeypatch.setattr(runner, "configure_sdk_model_defaults", lambda _s: None)
    monkeypatch.setattr(runner, "uses_chat_completions_tool_schema", lambda _m, _s: False)
    monkeypatch.setattr(todo_tools, "hydrate_todos_from_disk", lambda _d: None)
    monkeypatch.setattr(notes_tools, "hydrate_notes_from_disk", lambda _d: None)

    async def _create_or_reuse(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"client": object(), "session": object(), "caido_client": None}

    async def _cleanup(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(session_manager, "create_or_reuse", _create_or_reuse)
    monkeypatch.setattr(session_manager, "cleanup", _cleanup)
    monkeypatch.setattr(runner, "build_root_task", lambda _c: "task")
    monkeypatch.setattr(runner, "build_scope_context", lambda _c: "")
    monkeypatch.setattr(runner, "make_model_settings", lambda *_a, **_k: ModelSettings())
    monkeypatch.setattr(runner, "make_child_factory", lambda **_k: lambda **_kk: object())
    monkeypatch.setattr(runner, "open_agent_session", lambda _root_id, _db: object())

    seen: dict[str, bool] = {}

    def _build_halo_agent(**_kwargs: Any) -> object:
        names = [tool.name for tool in factory.registered_agent_tools()]
        seen.setdefault("auth_matrix", "auth_matrix" in names)
        return object()

    monkeypatch.setattr(runner, "build_halo_agent", _build_halo_agent)

    coordinator = AgentCoordinator()

    async def _root_ends(**_kwargs: Any) -> Any:
        return SimpleNamespace(final_output="done")

    monkeypatch.setattr(runner, "run_agent_loop", _root_ends)

    saved = list(factory._EXTRA_TOOLS)
    factory._EXTRA_TOOLS.clear()
    try:
        await runner.run_halo_scan(
            scan_config={"targets": [], "scan_mode": "deep"},
            scan_id="scan-auth-matrix",
            image="img",
            coordinator=coordinator,
        )
    finally:
        factory._EXTRA_TOOLS[:] = saved

    assert seen.get("auth_matrix") is True


def _settings() -> Any:
    return types.SimpleNamespace(
        llm=types.SimpleNamespace(
            model="openai/gpt-4o",
            reasoning_effort="high",
            force_required_tool_choice=False,
            timeout=300,
            prompt_cache=True,
            extra_headers=None,
        ),
        runtime=types.SimpleNamespace(max_context_images=3),
    )


def test_auth_matrix_sandbox_transport_uses_the_runner_script(base_url: str) -> None:
    session = _FakeSession()
    out = asyncio.run(
        _auth_matrix_impl(
            _ctx(sandbox_session=session), base_url, _operations(), _personas(), "sandbox", 5.0
        )
    )
    assert "error" not in out
    assert session.exec_calls
    assert {c["rule"] for c in out["candidates"]} >= {"horizontal_idor"}
    # The peer read of the 200-with-empty-body route stays unflagged.
    assert not [
        c
        for c in out["candidates"]
        if c["rule"] == "horizontal_idor" and c["operation_id"] == "order2"
    ]


def test_auth_matrix_sandbox_transport_requires_a_session(base_url: str) -> None:
    out = asyncio.run(
        _auth_matrix_impl(_ctx(), base_url, _operations(), _personas(), "sandbox", 5.0)
    )
    assert "error" in out


class _FakeSession:
    """Minimal stand-in for the sandbox session used by the runner script.

    The script itself is executed by the local interpreter against the local
    stub server, so the sandbox code path is covered end to end without
    touching anything outside the test process.
    """

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.exec_calls: list[tuple[str, ...]] = []

    async def write(self, path: Any, data: Any) -> None:
        self.files[_posix(path)] = data.read()

    async def exec(self, *command: str, timeout: float | None = None) -> Any:
        self.exec_calls.append(command)
        _ = timeout
        script = self.files["/workspace/.halo-auth-matrix-runner.py"]
        plan = self.files["/workspace/.halo-auth-matrix-plan.json"]
        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "runner.py"
            plan_path = Path(tmp) / "plan.json"
            out_path = Path(tmp) / "out.json"
            script_path.write_bytes(script)
            plan_path.write_bytes(plan)
            subprocess.run(  # noqa: S603 - fixed argv, the runner script is our own
                [sys.executable, str(script_path), str(plan_path), str(out_path)],
                check=True,
                capture_output=True,
            )
            self.files["/workspace/.halo-auth-matrix-out.json"] = out_path.read_bytes()
        return SimpleNamespace(ok=lambda: True, exit_code=0, stdout=b"", stderr=b"")

    async def read(self, path: Any) -> Any:
        return io.BytesIO(self.files[_posix(path)])


def _posix(path: Any) -> str:
    return path.as_posix() if hasattr(path, "as_posix") else str(path)


def _impl(
    url: str,
    operations: list[dict[str, Any]],
    personas: list[dict[str, Any]],
) -> dict[str, Any]:
    return asyncio.run(_auth_matrix_impl(_ctx(), url, operations, personas, "host", 5.0))


def _run_matrix(base_url: str) -> dict[str, Any]:
    return _impl(base_url, _operations(), _personas())
