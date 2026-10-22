"""Tests for the authorization matrix engine and the ``auth_matrix`` tool."""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import pytest
from agents import ModelSettings

import halo.tools.notes.tools as notes_tools
import halo.tools.todo.tools as todo_tools
from halo.agents import factory
from halo.core import runner
from halo.core.agents import AgentCoordinator
from halo.runtime import session_manager
from halo.tools.verify.auth_matrix import _auth_matrix_impl, auth_matrix
from halo.verify.http_sender import (
    BUDGET_EXHAUSTED_ERROR,
    DEFAULT_TIMEOUT_S,
    MAX_WALL_CLOCK_S,
    HttpRequest,
    HttpResponse,
    HttpxSender,
    clamp_timeout,
    send_sequential,
)
from halo.verify.matrix import (
    ALLOWED,
    CONTENT_DIFFERENT,
    CONTENT_IDENTICAL,
    CONTENT_SAME_STRUCTURE,
    CONTENT_UNKNOWN,
    DENIED,
    EMPTY,
    INCONCLUSIVE,
    REDIRECT,
    MatrixCell,
    MatrixPersona,
    WithheldCandidate,
    body_digest,
    body_shape,
    build_requests,
    cells_from_responses,
    compare_content,
    compute_candidates,
    compute_differentials,
    ensure_anon_persona,
    evidence_verdict,
    matrix_to_dict,
    operation_from_dict,
    persona_from_dict,
    run_matrix_in_session,
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
    # A 200 that is a different object for the peer: same status, other data.
    ("GET", "/api/orders/3"): {
        ALICE_CRED: (200, '{"id":3,"owner":"alice","item":"book"}'),
        BOB_CRED: (200, '{"error":"not found","hint":"try /orders/99"}'),
        None: (401, '{"error":"login required"}'),
    },
    # A 200 that is the owner's object rendered per viewer: same shape, one
    # field differs, so the structural match is the honest signal.
    ("GET", "/api/orders/4"): {
        ALICE_CRED: (200, '{"id":4,"owner":"alice","viewer":"alice"}'),
        BOB_CRED: (200, '{"id":4,"owner":"alice","viewer":"bob"}'),
        None: (401, '{"error":"login required"}'),
    },
    # A protected route that answers an anonymous caller with a login page:
    # status says "allowed", the body says "nothing was served".
    ("GET", "/api/dashboard"): {
        ALICE_CRED: (200, '{"user":"alice","widgets":3}'),
        BOB_CRED: (403, '{"error":"forbidden"}'),
        None: (200, "<html>sign in</html>"),
    },
}


class _StubServer(ThreadingHTTPServer):
    """Threaded so one stalled handler cannot wedge teardown."""

    daemon_threads = True
    allow_reuse_address = True
    block_on_close = False


class _Handler(BaseHTTPRequestHandler):
    # Bound the socket read so a client that connects and never writes cannot
    # leave a handler blocked forever.
    timeout = 5

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
    server = _StubServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="auth-matrix-stub")
    thread.start()
    try:
        host, port = server.server_address[:2]
        assert host == "127.0.0.1"
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive(), "stub server thread did not stop"


def _assert_loopback(url: str) -> None:
    """Tests may only ever talk to their own stub server."""
    host = urlparse(url).hostname
    assert host in {"127.0.0.1", "::1", "localhost"}, f"refusing non-loopback target {url!r}"


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
        {"id": "order3", "method": "GET", "path": "/api/orders/3", "owner": "alice"},
        {"id": "order4", "method": "GET", "path": "/api/orders/4", "owner": "alice"},
        {
            "id": "dashboard",
            "method": "GET",
            "path": "/api/dashboard",
            "allowed_roles": ["user"],
        },
    ]


def _cell(operation_id: str, persona_id: str, status: int | None, body: str = "") -> MatrixCell:
    """A cell carrying the same fingerprints ``cells_from_responses`` computes."""
    return MatrixCell(
        operation_id=operation_id,
        persona_id=persona_id,
        status=status,
        body_len=len(body),
        body_sample=body[:300],
        content_digest=body_digest(body),
        content_shape=body_shape(body),
    )


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
# Content and structure differentials
# ---------------------------------------------------------------------------


def test_body_digest_ignores_insignificant_whitespace() -> None:
    compact = '{"id":1,"owner":"alice"}'
    pretty = '{\n  "id": 1,\n  "owner": "alice"\n}'
    assert body_digest(compact) == body_digest(pretty)
    assert body_digest(compact) != body_digest('{"id":2,"owner":"alice"}')
    assert body_digest("") == ""


def test_body_shape_compares_structure_not_values() -> None:
    assert body_shape('{"id":1,"owner":"alice"}') == body_shape('{"id":2,"owner":"bob"}')
    assert body_shape('{"id":1}') != body_shape('{"id":1,"extra":true}')
    # Text has no structure worth comparing; only its digest is meaningful.
    assert body_shape("<html>sign in</html>") == ""
    assert body_shape("not json at all") == ""


def test_compare_content_classifies_every_pairing() -> None:
    owner = _cell("op", "alice", 200, '{"id":1,"owner":"alice"}')
    identical = _cell("op", "bob", 200, '{ "id": 1, "owner": "alice" }')
    same_shape = _cell("op", "bob", 200, '{"id":1,"owner":"bob"}')
    different = _cell("op", "bob", 200, '{"error":"not found"}')
    assert compare_content(identical, owner) == CONTENT_IDENTICAL
    assert compare_content(same_shape, owner) == CONTENT_SAME_STRUCTURE
    assert compare_content(different, owner) == CONTENT_DIFFERENT
    assert compare_content(owner, None) == CONTENT_UNKNOWN
    assert compare_content(_cell("op", "bob", 401), owner) == CONTENT_UNKNOWN


def test_cells_from_responses_fingerprints_the_body() -> None:
    plan = [
        ("op", "alice", HttpRequest("GET", "http://127.0.0.1:9/o")),
        ("op", "bob", HttpRequest("GET", "http://127.0.0.1:9/o")),
    ]
    cells = cells_from_responses(
        plan,
        [HttpResponse(status=200, body='{"id":1}'), HttpResponse(status=200, body='{ "id": 1 }')],
    )
    assert cells[0].content_digest == cells[1].content_digest
    assert compare_content(cells[1], cells[0]) == CONTENT_IDENTICAL
    assert cells[0].content_shape == "id:number"


def _same_role_matrix(peer_body: str, peer_status: int = 200) -> tuple[list[Any], list[Any]]:
    operations = [operation_from_dict({"id": "order", "path": "/o/1", "owner": "alice"})]
    personas = [
        persona_from_dict({"id": "alice", "role": "user"}),
        persona_from_dict({"id": "bob", "role": "user"}),
    ]
    cells = [
        _cell("order", "alice", 200, '{"id":1,"owner":"alice"}'),
        _cell("order", "bob", peer_status, peer_body),
    ]
    return compute_differentials(operations, personas, cells)


def test_horizontal_idor_reports_identical_content() -> None:
    candidates, withheld = _same_role_matrix('{"id":1,"owner":"alice"}')
    assert withheld == []
    idor = next(c for c in candidates if c.rule == "horizontal_idor")
    assert idor.details["content_match"] == CONTENT_IDENTICAL
    assert "identical" in " ".join(idor.evidence)


def test_horizontal_idor_reports_a_structural_match() -> None:
    candidates, withheld = _same_role_matrix('{"id":1,"owner":"bob"}')
    assert withheld == []
    idor = next(c for c in candidates if c.rule == "horizontal_idor")
    assert idor.details["content_match"] == CONTENT_SAME_STRUCTURE
    assert "same structure" in " ".join(idor.evidence)


def test_horizontal_idor_withholds_a_different_object() -> None:
    candidates, withheld = _same_role_matrix('{"error":"not found","hint":"try /o/99"}')
    assert [c for c in candidates if c.rule == "horizontal_idor"] == []
    assert [w.rule for w in withheld] == ["horizontal_idor"]
    assert withheld[0].persona_id == "bob"
    assert "differs" in withheld[0].reason


def test_horizontal_idor_ignores_a_denied_same_role_peer() -> None:
    """Same-role control: a peer the target refuses is not an IDOR."""
    candidates, withheld = _same_role_matrix('{"error":"forbidden"}', peer_status=403)
    assert candidates == []
    assert withheld == []


def _different_role_matrix(lower_body: str, lower_status: int = 200) -> tuple[list[Any], list[Any]]:
    operations = [
        operation_from_dict({"id": "admin", "path": "/admin", "allowed_roles": ["admin"]})
    ]
    personas = [
        persona_from_dict({"id": "root", "role": "admin"}),
        persona_from_dict({"id": "carol", "role": "user"}),
    ]
    cells = [
        _cell("admin", "root", 200, '{"users":10,"audit":[]}'),
        _cell("admin", "carol", lower_status, lower_body),
    ]
    return compute_differentials(operations, personas, cells)


def test_vertical_escalation_compares_against_the_authorized_role() -> None:
    """Different-role case: the lower role saw the authorized role's document."""
    candidates, withheld = _different_role_matrix('{"users":10,"audit":[]}')
    assert withheld == []
    escalation = next(c for c in candidates if c.rule == "vertical_escalation")
    assert escalation.persona_id == "carol"
    assert escalation.details["baseline_persona"] == "root"
    assert escalation.details["content_match"] == CONTENT_IDENTICAL
    assert "authorized caller root" in " ".join(escalation.evidence)


def test_vertical_escalation_withholds_when_the_lower_role_gets_another_document() -> None:
    candidates, withheld = _different_role_matrix('{"error":"forbidden"}')
    assert [c for c in candidates if c.rule == "vertical_escalation"] == []
    assert [w.rule for w in withheld] == ["vertical_escalation"]
    assert "differs" in withheld[0].reason


def test_vertical_escalation_flags_a_missing_authorized_baseline() -> None:
    operations = [
        operation_from_dict({"id": "admin", "path": "/admin", "allowed_roles": ["admin"]})
    ]
    personas = [persona_from_dict({"id": "carol", "role": "user"})]
    cells = [_cell("admin", "carol", 200, '{"users":10}')]
    candidates, withheld = compute_differentials(operations, personas, cells)
    assert withheld == []
    escalation = next(c for c in candidates if c.rule == "vertical_escalation")
    assert escalation.details["content_match"] == CONTENT_UNKNOWN
    assert any("status alone" in line for line in escalation.counterevidence)


def _anonymous_matrix(anon_body: str, anon_status: int = 200) -> tuple[list[Any], list[Any]]:
    operations = [operation_from_dict({"id": "report", "path": "/r/7", "owner": "alice"})]
    personas = [
        persona_from_dict({"id": "alice", "role": "user"}),
        persona_from_dict({"id": "anon"}),
    ]
    cells = [
        _cell("report", "alice", 200, '{"report":"q3"}'),
        _cell("report", "anon", anon_status, anon_body),
    ]
    return compute_differentials(operations, personas, cells)


def test_anonymous_access_reports_a_matching_document() -> None:
    """Unauthenticated case: the anon body is the authenticated caller's view."""
    candidates, withheld = _anonymous_matrix('{"report":"q3"}')
    assert withheld == []
    anon = next(c for c in candidates if c.rule == "anonymous_access")
    assert anon.persona_id == "anon"
    assert anon.details["content_match"] == CONTENT_IDENTICAL
    assert anon.details["baseline_persona"] == "alice"


def test_anonymous_access_withholds_a_login_page() -> None:
    """A 200 login page is not exposure, and the diff must say so."""
    candidates, withheld = _anonymous_matrix("<html>sign in</html>")
    assert [c for c in candidates if c.rule == "anonymous_access"] == []
    assert [w.rule for w in withheld] == ["anonymous_access"]
    assert "login" in withheld[0].reason


def test_anonymous_access_ignores_a_denied_anonymous_probe() -> None:
    """Unauthenticated control: a refused anon probe is enforcement working."""
    candidates, withheld = _anonymous_matrix('{"error":"login required"}', anon_status=401)
    assert candidates == []
    assert withheld == []


def test_matrix_to_dict_exposes_withheld_and_cell_fingerprints() -> None:
    operations = [operation_from_dict({"id": "op", "path": "/o", "owner": "alice"})]
    personas = [persona_from_dict({"id": "alice"})]
    cells = [_cell("op", "alice", 200, '{"id":1}')]
    withheld = [
        WithheldCandidate(
            rule="horizontal_idor",
            operation_id="op",
            persona_id="bob",
            reason="the peer's body differs",
        )
    ]
    payload = matrix_to_dict(operations, personas, cells, [], withheld)
    assert payload["withheld"] == [
        {
            "rule": "horizontal_idor",
            "operation_id": "op",
            "persona_id": "bob",
            "reason": "the peer's body differs",
        }
    ]
    assert payload["counts"]["withheld"] == 1
    assert payload["cells"][0]["content_digest"] == body_digest('{"id":1}')
    assert payload["cells"][0]["content_structure"] == ["id:number"]
    # The tool response must stay JSON-serializable.
    assert json.loads(json.dumps(payload))["counts"]["withheld"] == 1


# ---------------------------------------------------------------------------
# Timeouts and transport bounds
# ---------------------------------------------------------------------------


def test_clamp_timeout_bounds_caller_values() -> None:
    assert clamp_timeout(None) == DEFAULT_TIMEOUT_S
    assert clamp_timeout(0) == DEFAULT_TIMEOUT_S
    assert clamp_timeout(-5) == DEFAULT_TIMEOUT_S
    assert clamp_timeout("bogus") == DEFAULT_TIMEOUT_S
    assert clamp_timeout(3.5) == 3.5
    assert clamp_timeout(1e9) == MAX_WALL_CLOCK_S


class _StalledSender:
    """A transport whose send never returns on its own."""

    timeout = 0.2

    def __init__(self) -> None:
        self.calls = 0

    async def send(self, _request: HttpRequest) -> HttpResponse:
        self.calls += 1
        await asyncio.sleep(30)
        return HttpResponse(status=200, body="late")

    async def aclose(self) -> None:
        return None


def test_send_sequential_bounds_a_stalled_sender() -> None:
    sender = _StalledSender()
    started = time.monotonic()
    responses = asyncio.run(send_sequential(sender, [HttpRequest("GET", "http://127.0.0.1:9/o")]))
    assert time.monotonic() - started < 10
    assert responses[0].status is None
    assert responses[0].error and "timed out" in responses[0].error


def test_send_sequential_marks_probes_left_by_an_exhausted_budget() -> None:
    sender = _StalledSender()
    responses = asyncio.run(
        send_sequential(sender, [HttpRequest("GET", "http://127.0.0.1:9/o")], budget_s=0)
    )
    assert responses[0].error == BUDGET_EXHAUSTED_ERROR
    assert sender.calls == 0


def test_httpx_sender_refuses_non_http_urls() -> None:
    async def _run() -> HttpResponse:
        sender = HttpxSender(timeout=1)
        try:
            return await sender.send(HttpRequest("GET", "file:///etc/passwd"))
        finally:
            await sender.aclose()

    response = asyncio.run(_run())
    assert response.status is None
    assert response.error and "non-HTTP" in response.error


class _RecordingSession:
    """Sandbox stand-in that records the plan and the exec timeout."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.timeouts: list[float | None] = []

    async def write(self, path: Any, data: Any) -> None:
        self.files[_posix(path)] = data.read()

    async def exec(self, *_command: str, timeout: float | None = None) -> Any:
        self.timeouts.append(timeout)
        plan = json.loads(self.files["/workspace/.halo-auth-matrix-plan.json"])
        out = [{"status": 200, "body": '{"ok":true}', "location": None} for _ in plan]
        self.files["/workspace/.halo-auth-matrix-out.json"] = json.dumps(out).encode()
        return SimpleNamespace(ok=lambda: True, exit_code=0, stdout=b"", stderr=b"")

    async def read(self, path: Any) -> Any:
        return io.BytesIO(self.files[_posix(path)])


def test_run_matrix_in_session_passes_a_per_request_timeout() -> None:
    session = _RecordingSession()
    plan = [("op", "p", HttpRequest("GET", "http://127.0.0.1:9/x"))]
    responses = asyncio.run(run_matrix_in_session(session, plan, timeout=2.5))
    assert len(responses) == 1
    payload = json.loads(session.files["/workspace/.halo-auth-matrix-plan.json"])
    assert payload[0]["timeout"] == 2.5
    # One timeout per probe, plus startup slack, never unbounded.
    assert session.timeouts == [33]


def test_run_matrix_in_session_caps_the_runner_process() -> None:
    session = _RecordingSession()
    plan = [("op", f"p{index}", HttpRequest("GET", "http://127.0.0.1:9/x")) for index in range(400)]
    asyncio.run(run_matrix_in_session(session, plan, timeout=15.0))
    assert session.timeouts == [600]


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
    # The peer received the owner's own document, not merely a 200.
    assert idor["details"]["content_match"] == CONTENT_IDENTICAL

    # A per-viewer render is a structural match, and the engine says so.
    per_viewer = [
        c
        for c in out["candidates"]
        if c["rule"] == "horizontal_idor" and c["operation_id"] == "order4"
    ]
    assert [c["details"]["content_match"] for c in per_viewer] == [CONTENT_SAME_STRUCTURE]


def test_auth_matrix_withholds_status_only_matches(base_url: str) -> None:
    out = _run_matrix(base_url)
    withheld = {(w["rule"], w["operation_id"]): w for w in out["withheld"]}
    # A 200 for the peer that is a different object is not an IDOR.
    assert ("horizontal_idor", "order3") in withheld
    # A 200 for the anonymous caller that is a login page is not exposure.
    assert ("anonymous_access", "dashboard") in withheld
    assert all(w["reason"] for w in out["withheld"])
    assert out["counts"]["withheld"] == len(out["withheld"])
    candidate_keys = {(c["rule"], c["operation_id"]) for c in out["candidates"]}
    assert ("horizontal_idor", "order3") not in candidate_keys
    assert ("anonymous_access", "dashboard") not in candidate_keys


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


def test_auth_matrix_rejects_a_non_http_base_url() -> None:
    out = asyncio.run(
        _auth_matrix_impl(_ctx(), "file:///etc", _operations(), _personas(), "host", 5.0)
    )
    assert "error" in out
    assert "http" in str(out["error"])


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
        self.exec_timeouts: list[float] = []

    async def write(self, path: Any, data: Any) -> None:
        self.files[_posix(path)] = data.read()

    async def exec(self, *command: str, timeout: float | None = None) -> Any:
        self.exec_calls.append(command)
        # A sandbox exec without a timeout can wedge the whole scan.
        assert timeout is not None and timeout > 0, "sandbox exec must carry a timeout"
        self.exec_timeouts.append(timeout)
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
                timeout=timeout,
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
    if url:
        _assert_loopback(url)
    return asyncio.run(_auth_matrix_impl(_ctx(), url, operations, personas, "host", 5.0))


def _run_matrix(base_url: str) -> dict[str, Any]:
    return _impl(base_url, _operations(), _personas())
