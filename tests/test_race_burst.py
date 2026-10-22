"""Tests for the race / TOCTOU detector.

The stub server is a textbook check-then-act (TOCTOU) endpoint: it reads the
counter under a lock, sleeps in the window between the check and the act, and
only then decrements. Sequential requests therefore win once, while
concurrent requests all read the same pre-check state and win several times.
That is the difference the detector measures.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from halo.tools.verify import race as race_tool
from halo.verify import race as race_engine
from halo.verify.http_sender import redact_headers, redact_text, select_mode
from halo.verify.race import (
    DURABILITY_FAILED,
    DURABILITY_PASSED,
    DURABILITY_SKIPPED,
    MODE_CONCURRENT,
    VERDICT_ERROR,
    VERDICT_NOT_A_RACE,
    VERDICT_RACE_CONFIRMED,
    VERDICT_RACE_SUSPECTED,
    PhaseResult,
    RaceResult,
    RequestSpec,
    decide_verdict,
    default_threshold,
    phase_from_records,
    race_result_to_dict,
    substitute_slots,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


#: A single test may never wait longer than this, even if a stub server or the
#: runner wedges. pytest-timeout is the outer guard (see pyproject.toml); this
#: keeps an individual call bounded so the failure names the call site.
TEST_TIMEOUT_SECONDS = 30.0
#: The stub server thread must stop within this budget when a test ends.
SERVER_SHUTDOWN_TIMEOUT = 5.0


# --------------------------------------------------------------------------
# Pure engine
# --------------------------------------------------------------------------


def test_substitute_slots_fills_known_and_keeps_unknown() -> None:
    text = "user-{{name}} wants {{count}} of {{unknown}}"
    assert substitute_slots(text, {"name": "ada", "count": "3"}) == (
        "user-ada wants 3 of {{unknown}}"
    )
    assert substitute_slots("", {"name": "ada"}) == ""
    assert substitute_slots("plain", {}) == "plain"


def test_default_threshold_scales_with_the_burst() -> None:
    assert default_threshold(4) == 2
    assert default_threshold(8) == 2
    assert default_threshold(40) == 10
    assert default_threshold(1) == 2


def test_phase_from_records_counts_wins_and_redacts_secrets() -> None:
    records = [
        {"index": 0, "status": 200, "elapsed_ms": 5, "error": None, "body": "token=abc"},
        {"index": 1, "status": 409, "elapsed_ms": 6, "error": None, "body": "denied"},
        {"index": 2, "status": None, "elapsed_ms": 7, "error": "boom", "body": ""},
    ]
    phase = phase_from_records("baseline", records, secrets=["abc"])
    assert phase.total == 3
    assert phase.success_count == 1
    assert phase.runs[0].body_excerpt == "token=[REDACTED]"
    assert phase.runs[1].success is False
    assert phase.runs[2].status is None
    assert phase.runs[2].success is False


def test_phase_from_records_honours_explicit_success_statuses() -> None:
    records = [{"index": 0, "status": 201, "elapsed_ms": 1, "error": None, "body": ""}]
    phase = phase_from_records("burst", records, success_statuses=(201,))
    assert phase.success_count == 1
    phase = phase_from_records("burst", records, success_statuses=(200,))
    assert phase.success_count == 0


def _phase(name: str, statuses: list[int | None]) -> PhaseResult:
    records = [
        {"index": index, "status": status, "elapsed_ms": 1, "error": None, "body": ""}
        for index, status in enumerate(statuses)
    ]
    return phase_from_records(name, records)


def test_decide_verdict_covers_the_table() -> None:
    # Every baseline request won: nothing was ever guarded, so there is no
    # race to find no matter what the burst did.
    assert (
        decide_verdict(_phase("baseline", [200, 200, 200]), _phase("burst", [200, 200, 200]), 2)
        == VERDICT_NOT_A_RACE
    )
    # The burst won more than the baseline, but nothing confirmed the effect.
    assert (
        decide_verdict(_phase("baseline", [200, 409, 409]), _phase("burst", [200, 200, 200]), 2)
        == VERDICT_RACE_SUSPECTED
    )
    # Same signal with a passing durability probe.
    assert (
        decide_verdict(
            _phase("baseline", [200, 409, 409]),
            _phase("burst", [200, 200, 200]),
            2,
            DURABILITY_PASSED,
        )
        == VERDICT_RACE_CONFIRMED
    )
    # A failing durability probe never confirms.
    assert (
        decide_verdict(
            _phase("baseline", [200, 409, 409]),
            _phase("burst", [200, 200, 200]),
            2,
            DURABILITY_FAILED,
        )
        == VERDICT_RACE_SUSPECTED
    )
    # The burst did not beat the baseline by enough.
    assert (
        decide_verdict(_phase("baseline", [200, 409, 409]), _phase("burst", [200, 409, 409]), 2)
        == VERDICT_NOT_A_RACE
    )
    # A phase that produced no measurements is an error, never a verdict.
    assert decide_verdict(_phase("baseline", []), _phase("burst", [200]), 2) == VERDICT_ERROR


def test_race_result_to_dict_is_json_ready() -> None:
    baseline = _phase("baseline", [200, 409, 409])
    burst = _phase("burst", [200, 200, 200])
    result = RaceResult(
        verdict=VERDICT_RACE_SUSPECTED,
        threshold=2,
        baseline=baseline,
        burst=burst,
        delta=1,
        durability=DURABILITY_SKIPPED,
    )
    payload = race_result_to_dict(result)
    assert json.loads(json.dumps(payload))["verdict"] == VERDICT_RACE_SUSPECTED
    assert payload["baseline"]["success_count"] == 1
    assert payload["burst"]["success_count"] == 3
    assert payload["delta"] == 1


# --------------------------------------------------------------------------
# Transport helpers
# --------------------------------------------------------------------------


def test_select_mode_keeps_loopback_local_and_the_rest_in_the_sandbox() -> None:
    assert select_mode("http://127.0.0.1:8080/x") == "local"
    assert select_mode("http://localhost:8080/x") == "local"
    assert select_mode("https://example.test/x") == "sandbox"


def test_redaction_hides_credential_headers() -> None:
    headers = {"Authorization": "Bearer s3cret", "Cookie": "sid=abc", "Accept": "*/*"}
    redacted = redact_headers(headers)
    assert redacted["Authorization"] == "[REDACTED]"
    assert redacted["Cookie"] == "[REDACTED]"
    assert redacted["Accept"] == "*/*"
    assert redact_text("value s3cret here", ["s3cret"]) == "value [REDACTED] here"


# --------------------------------------------------------------------------
# Live race against a local TOCTOU stub
# --------------------------------------------------------------------------


class _RaceState:
    """Shared state of the stub: a counter guarded by a check-then-act window."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.balance = 1
        self.won = 0

    def reset(self) -> None:
        with self.lock:
            self.balance = 1
            self.won = 0

    def claim(self) -> bool:
        # The check and the act are not atomic: the sleep sits between them,
        # which is exactly the window a concurrent request slips through.
        with self.lock:
            available = self.balance > 0
        # Generous enough that an overloaded CI runner still fits the whole
        # wave into the window, short enough to keep the test fast.
        time.sleep(0.25)
        with self.lock:
            if available:
                self.balance -= 1
                self.won += 1
                return True
            return False


class _StubServer(ThreadingHTTPServer):
    """A loopback stub whose threads can never keep a test alive."""

    daemon_threads = True
    block_on_close = False
    #: The default backlog (5) is smaller than a typical burst, which can make
    #: Linux drop SYNs for the last connections of a wave.
    request_queue_size = 64


class _QuietHandler(BaseHTTPRequestHandler):
    state: Any

    def _respond(self, status: int, body: str) -> None:
        payload = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args: Any) -> None:
        return


class _RaceHandler(_QuietHandler):
    state: _RaceState

    def do_GET(self) -> None:
        if self.path.startswith("/reset"):
            self.state.reset()
            self._respond(200, "reset")
            return
        if self.path.startswith("/state"):
            with self.state.lock:
                body = f"balance={self.state.balance} won={self.state.won}"
            self._respond(200, body)
            return
        if self.state.claim():
            self._respond(200, "granted")
            return
        self._respond(409, "denied")


class _BarrierState:
    """A stub that answers only when every burst request is in flight at once."""

    def __init__(self, parties: int) -> None:
        self.barrier = threading.Barrier(parties, timeout=SERVER_SHUTDOWN_TIMEOUT)

    def wait(self) -> bool:
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            return False
        return True


class _BarrierHandler(_QuietHandler):
    state: _BarrierState

    def do_GET(self) -> None:
        if self.state.wait():
            self._respond(200, "arrived")
        else:
            self._respond(503, "not simultaneous")


@contextlib.contextmanager
def _serve(state: Any, handler_cls: type[_QuietHandler] = _RaceHandler) -> Iterator[str]:
    """Serve ``state`` on a loopback port and always tear the server down.

    The handler threads are daemons, the listener is closed on the way out,
    and the join makes a leaked server thread a test failure instead of a
    hang that only shows up as a cancelled CI job.
    """
    handler = type("_BoundHandler", (handler_cls,), {"state": state})
    server = _StubServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=SERVER_SHUTDOWN_TIMEOUT)
        assert not thread.is_alive(), "the stub server thread did not stop"


def _ctx(**entries: Any) -> Any:
    return SimpleNamespace(context=dict(entries))


def _raw(url: str, path: str) -> str:
    return f"GET {url}{path} HTTP/1.1\r\nHost: {url.split('//', 1)[1]}\r\n\r\n"


def _run_race(ctx: Any, **kwargs: Any) -> dict[str, Any]:
    """Call the tool with a bounded wait so a bug fails instead of hanging."""

    async def _call() -> dict[str, Any]:
        return await asyncio.wait_for(
            race_tool._race_burst_impl(ctx, **kwargs), timeout=TEST_TIMEOUT_SECONDS
        )

    return asyncio.run(_call())


def test_race_burst_detects_a_local_toctou_race() -> None:
    state = _RaceState()
    with _serve(state) as base:
        outcome = _run_race(
            _ctx(),
            raw_request=_raw(base, "/claim"),
            burst_size=6,
            baseline_runs=3,
            reset_request=_raw(base, "/reset"),
            probe_request=_raw(base, "/state"),
            probe_body_contains=["won=6"],
        )

    assert "error" not in outcome, outcome
    assert outcome["verdict"] == VERDICT_RACE_CONFIRMED
    assert outcome["durability"] == DURABILITY_PASSED
    assert outcome["baseline"]["success_count"] == 1
    assert outcome["burst"]["success_count"] == 6
    assert outcome["delta"] == 5


def test_race_burst_reports_a_missing_race_as_not_a_race() -> None:
    """A guarded endpoint that always refuses the second claim is not a race."""
    state = _RaceState()

    def _claim() -> bool:
        with state.lock:
            available = state.balance > 0
            if available:
                state.balance -= 1
                state.won += 1
            return available

    state.claim = _claim  # type: ignore[method-assign]
    with _serve(state) as base:
        outcome = _run_race(
            _ctx(),
            raw_request=_raw(base, "/claim"),
            burst_size=4,
            baseline_runs=2,
            reset_request=_raw(base, "/reset"),
        )

    assert "error" not in outcome, outcome
    assert outcome["verdict"] == VERDICT_NOT_A_RACE


def test_race_burst_never_leaks_credential_headers() -> None:
    state = _RaceState()
    credential = "Bearer super-secret-token"
    with _serve(state) as base:
        outcome = _run_race(
            _ctx(),
            raw_request=(
                f"GET {base}/claim HTTP/1.1\r\n"
                f"Host: {base.split('//', 1)[1]}\r\n"
                f"Authorization: {credential}\r\n"
                "Cookie: sid=abc\r\n\r\n"
            ),
            burst_size=3,
            baseline_runs=2,
        )

    assert "error" not in outcome, outcome
    assert credential not in json.dumps(outcome)
    assert outcome["request"]["headers"]["Authorization"] == "[REDACTED]"
    assert outcome["request"]["headers"]["Cookie"] == "[REDACTED]"


def test_race_burst_refuses_targets_outside_the_scan_scope() -> None:
    outcome = _run_race(
        _ctx(scan_targets=["https://app.example.test"]),
        url="https://elsewhere.example.test/claim",
    )
    assert "error" in outcome
    assert "outside the scope" in outcome["error"]


def test_race_burst_reports_missing_input_as_an_error() -> None:
    outcome = _run_race(_ctx())
    assert "error" in outcome
    assert "request_id" in outcome["error"]


def test_race_burst_rejects_the_reserved_index_slot() -> None:
    outcome = _run_race(_ctx(), url="http://127.0.0.1:1/x", slots={"index": "3"})
    assert "error" in outcome
    assert "reserved" in outcome["error"]


def test_race_burst_survives_an_unreachable_target() -> None:
    """A failing target is reported as an error, never as an exception."""
    outcome = _run_race(
        _ctx(),
        url="http://127.0.0.1:1/claim",
        burst_size=2,
        baseline_runs=1,
    )
    assert "error" not in outcome
    assert outcome["verdict"] == VERDICT_ERROR
    assert all(run["error"] for run in outcome["burst"]["runs"])


def test_burst_releases_every_worker_together() -> None:
    """The burst is one wave: a strict barrier admits it only if all arrive.

    The stub refuses to answer until every request of the burst is in flight
    at the same time, so a regression to a sequential ramp (or a worker that
    starts late) cannot pass this test.
    """
    state = _BarrierState(parties=6)
    with _serve(state, _BarrierHandler) as base:
        phase = asyncio.run(
            asyncio.wait_for(
                race_engine._run_phase(
                    None,
                    RequestSpec(method="GET", url=f"{base}/barrier"),
                    name="burst",
                    mode=MODE_CONCURRENT,
                    count=6,
                    timeout=TEST_TIMEOUT_SECONDS,
                ),
                timeout=TEST_TIMEOUT_SECONDS,
            )
        )

    assert phase.error is None, phase.error
    assert phase.total == 6
    assert phase.success_count == 6
