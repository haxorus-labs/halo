"""Tests for deterministic replay, sandbox reset, knowledge store, controls."""

from __future__ import annotations

import asyncio
import json
import os
import time
from types import SimpleNamespace
from typing import Any

from halo.sast.findings import Finding, FindingStore
from halo.sast.knowledge import (
    load_snapshot_states,
    prune_knowledge,
    save_snapshot,
    target_key,
)
from halo.tools.sast import tools as sast_tools
from halo.tools.verify import tools as verify_tools
from halo.verify.replay import Oracle, evaluate_oracle, result_to_dict, run_replays


def _ctx(**entries: Any) -> Any:
    return SimpleNamespace(context=dict(entries))


def test_evaluate_oracle_matches_and_misses() -> None:
    response = {"status_code": 200, "body": "welcome admin panel"}
    oracle = Oracle(status=200, body_contains=("admin",), body_absent=("error",))
    ok, matched, missed = evaluate_oracle(response, oracle)
    assert ok and not missed and len(matched) == 3

    oracle = Oracle(status=302, body_contains=("admin", "nope"), body_absent=("panel",))
    ok, _, missed = evaluate_oracle(response, oracle)
    assert not ok and len(missed) == 3

    ok, matched, missed = evaluate_oracle(None, Oracle(status=200))
    assert not ok and missed == ("no response",)


def test_run_replays_reproduced_and_not() -> None:
    async def send(_index: int) -> dict[str, Any]:
        return {
            "elapsed_ms": 5,
            "error": None,
            "response_raw": b"HTTP/1.1 200 OK\r\n\r\nhello admin",
        }

    result = asyncio.run(run_replays(send, Oracle(status=200, body_contains=("admin",)), 3))
    assert result.verdict == "REPRODUCED"
    assert result.runs_passed == 3
    assert all(r.ok for r in result.runs)

    result = asyncio.run(run_replays(send, Oracle(status=302), 2))
    assert result.verdict == "NOT_REPRODUCED"
    assert result.runs_passed == 0

    async def boom(_index: int) -> dict[str, Any]:
        raise RuntimeError("no route")

    result = asyncio.run(run_replays(boom, Oracle(status=200), 2))
    assert result.verdict == "NOT_REPRODUCED"
    assert result.runs[0].error == "no route"

    result = asyncio.run(run_replays(send, Oracle(), 99))
    assert result.runs_total == 10
    result = asyncio.run(run_replays(send, Oracle(), 0))
    assert result.runs_total == 1

    as_dict = result_to_dict(result)
    assert as_dict["verdict"] in ("REPRODUCED", "NOT_REPRODUCED")
    assert len(as_dict["runs"]) == as_dict["runs_total"]


class _ReplayStub:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw
        self.calls = 0
        self.sessions = self.Sessions()

    class Sessions:
        async def create(self) -> Any:
            return SimpleNamespace(id="sess-1")

    @property
    def replay(self) -> Any:
        outer = self

        class _Replay:
            sessions = outer.sessions

            async def send(self, _session_id: str, _options: Any) -> Any:
                outer.calls += 1
                entry = SimpleNamespace(response=SimpleNamespace(raw=outer._raw))
                return SimpleNamespace(status="DONE", error=None, entry=entry)

        return _Replay()


def test_replay_poc_raw_request_end_to_end() -> None:
    raw = "GET http://target.local/rest/admin/users HTTP/1.1\r\nHost: target.local\r\n\r\n"
    body = b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n[{"id": 1}]'
    out = asyncio.run(
        verify_tools._replay_poc_impl(
            _ctx(caido_client=_ReplayStub(body)),
            "",
            raw,
            {},
            200,
            ['"id"'],
            ["forbidden"],
            2,
        )
    )
    assert out["verdict"] == "REPRODUCED"
    assert out["runs_total"] == 2
    assert out["runs_passed"] == 2


def test_replay_poc_no_client_and_bad_input() -> None:
    out = asyncio.run(verify_tools._replay_poc_impl(_ctx(), "", "", {}, None, [], [], 1))
    assert "error" in out
    out = asyncio.run(
        verify_tools._replay_poc_impl(_ctx(caido_client=object()), "", "", {}, None, [], [], 1)
    )
    assert "error" in out


class _ExecResult:
    def __init__(self, ok: bool, code: int, out: str, err: bytes) -> None:
        self._ok = ok
        self.exit_code = code
        self.stdout = out
        self.stderr = err

    def ok(self) -> bool:
        return self._ok


class _SessionStub:
    def __init__(self, results: list[_ExecResult]) -> None:
        self._results = list(results)
        self.commands: list[str] = []

    async def exec(self, *args: Any, **_kwargs: Any) -> _ExecResult:
        self.commands.append(" ".join(str(a) for a in args[1:]))
        return self._results.pop(0)


def test_reset_sandbox_runs_commands_and_stops_on_failure() -> None:
    ok_res = _ExecResult(ok=True, code=0, out="done\n", err=b"")
    fail_res = _ExecResult(ok=False, code=1, out="", err=b"boom")
    session = _SessionStub([ok_res, fail_res, ok_res])
    out = asyncio.run(
        verify_tools._reset_sandbox_impl(_ctx(sandbox_session=session), ["a", "b", "c"])
    )
    assert out["reset_performed"] is False
    assert [s["command"] for s in out["steps"]] == ["a", "b"]
    assert session.commands == ["-c a", "-c b"]

    session = _SessionStub([_ExecResult(ok=True, code=0, out="x" * 5000, err=b"")])
    out = asyncio.run(verify_tools._reset_sandbox_impl(_ctx(sandbox_session=session), ["a"]))
    assert out["reset_performed"] is True
    assert len(out["steps"][0]["stdout"]) <= 2000


def test_reset_sandbox_no_session_or_commands() -> None:
    out = asyncio.run(verify_tools._reset_sandbox_impl(_ctx(), ["a"]))
    assert out == {"reset_performed": False, "reason": "no sandbox session in run context"}
    verify_tools.set_default_source_roots([])
    out = asyncio.run(
        verify_tools._reset_sandbox_impl(_ctx(sandbox_session=_SessionStub([])), None)
    )
    assert out["reset_performed"] is False


def test_knowledge_roundtrip_carry_and_prune(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("HALO_KNOWLEDGE_DIR", str(tmp_path / "knowledge"))
    key = target_key([], "https://github.com/org/repo")
    assert key == target_key([], "HTTPS://GITHUB.COM/ORG/REPO")
    assert key != target_key([str(tmp_path / "other")])
    assert load_snapshot_states(key) == {}

    run_state = tmp_path / "run" / ".state"
    run_state.mkdir(parents=True)
    store = FindingStore(path=run_state / "findings.jsonl")
    verified = Finding(
        id="abc",
        tool="semgrep",
        rule_id="r",
        file="a.py",
        line=1,
        sink_symbol="s",
        message="m",
        taxonomy="t",
        severity="HIGH",
        state="VERIFIED",
        reason="poc",
    )
    store.record(verified)
    assert save_snapshot(key, None, store.path) is not None
    states = load_snapshot_states(key)
    assert states["abc"].state == "VERIFIED"

    old = tmp_path / "knowledge" / key
    ancient = time.time() - 100 * 86400
    os.utime(old, (ancient, ancient))
    assert prune_knowledge(max_age_days=90) == [key]
    assert load_snapshot_states(key) == {}


def test_finding_verify_negative_control_refutes(tmp_path: Any) -> None:
    state = tmp_path / ".state"
    state.mkdir()
    sast_tools.hydrate_sast_from_disk(state)
    try:
        store = sast_tools._require_store()
        store.record(
            Finding(
                id="ctrl1",
                tool="semgrep",
                rule_id="r",
                file="a.py",
                line=1,
                sink_symbol="s",
                message="m",
                taxonomy="t",
                severity="HIGH",
            )
        )
        out = asyncio.run(sast_tools._finding_verify_impl("ctrl1", "sink(x)", 200))
        assert out["state"] == "REFUTED"
        assert "negative control" in out["reason"]
        current = store.get("ctrl1")
        assert current is not None
        assert current.state == "REFUTED"
        out = asyncio.run(sast_tools._finding_verify_impl("missing", "x", 200))
        assert "error" in out
    finally:
        sast_tools.hydrate_sast_from_disk.__globals__["_store"] = None


def test_scan_result_json_shape() -> None:
    out = asyncio.run(verify_tools._replay_poc_impl(_ctx(), "nope", "", {}, None, [], [], 1))
    assert "error" in out
    parsed = json.loads(json.dumps(out))
    assert parsed["error"]
