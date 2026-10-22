"""Deterministic race / TOCTOU detection: sequential baseline vs concurrent burst.

A race is proven by contrast. The same request is fired twice under two
conditions and the outcomes are compared:

* **baseline** - the requests go out one at a time, so the target's own
  serialization decides the outcome. This is the control: it shows how many
  of them the target is willing to honour when nothing overlaps.
* **burst** - the requests go out together, so the target has to protect a
  read-modify-write window it does not control.

The engine is pure: :func:`decide_verdict` turns two measured phases into a
verdict with no I/O of its own, so the verdict table can be tested without a
target. The network work is delegated to :mod:`halo.verify.http_sender`, which
decides whether the traffic leaves from the host or from the sandbox session.

Verdict table
-------------
=========================  ==========================================
``NOT_A_RACE``             every baseline request succeeded, so the
                           target never guarded the operation and
                           concurrency is not what changed the
                           outcome; or the burst won fewer extra
                           requests than ``threshold`` requires.
``RACE_SUSPECTED``         the burst won at least ``threshold`` more
                           requests than the baseline, but no
                           durability probe confirmed that the
                           duplicated effect persisted.
``RACE_CONFIRMED``         same signal, and a durability probe
                           confirmed the effect survived.
``ERROR``                  a phase produced no measurements at all.
=========================  ==========================================
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from halo.verify import http_sender


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


logger = logging.getLogger(__name__)

#: Concurrent requests fired in the burst phase.
DEFAULT_BURST_SIZE = 8
#: Sequential requests fired in the baseline phase.
DEFAULT_BASELINE_SIZE = 8
#: Smallest margin that is not attributable to a single flaky response.
MIN_THRESHOLD = 2
#: Characters of a response body kept as evidence per request.
BODY_EXCERPT_CHARS = 500
#: Characters of the durability probe response read while looking for markers.
PROBE_BODY_CHARS = 20000
#: Upper bound on a single request inside a phase. A race window is short; a
#: request that has not answered by then is a failed measurement, never a wait
#: that outlives the phase.
MAX_REQUEST_TIMEOUT_SECONDS = 30.0

VERDICT_RACE_CONFIRMED = "RACE_CONFIRMED"
VERDICT_RACE_SUSPECTED = "RACE_SUSPECTED"
VERDICT_NOT_A_RACE = "NOT_A_RACE"
VERDICT_ERROR = "ERROR"

DURABILITY_PASSED = "passed"
DURABILITY_FAILED = "failed"
DURABILITY_SKIPPED = "skipped"

MODE_SEQUENTIAL = "sequential"
MODE_CONCURRENT = "concurrent"

#: Slot reserved for the per-request index; it cannot be set by a caller.
INDEX_SLOT = "index"

#: The script that fires a phase. The sandbox runs it as a file, the host
#: runs it in a worker thread, and both get this exact text, so a burst
#: behaves identically wherever it runs.
RACE_RUNNER_SCRIPT = r'''
"""Fire one request many times, sequentially or concurrently."""

import asyncio
import json
import os
import re
import ssl
import sys
import time

import httpx

DEFAULT_BODY_LIMIT = 2000
DEFAULT_REQUEST_TIMEOUT = 30.0
SLOT_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def _fill(template, index):
    if not template:
        return template
    return SLOT_RE.sub(
        lambda match: str(index) if match.group(1) == "index" else match.group(0), template
    )


def _build_client(connections, request_timeout):
    # The burst deliberately bypasses the ambient proxy configuration: the
    # proxy serializes requests, and a burst has to leave as one wave.
    kwargs = {"trust_env": False}
    ca_file = os.environ.get("SSL_CERT_FILE")
    if ca_file and os.path.exists(ca_file):
        kwargs["verify"] = ssl.create_default_context(cafile=ca_file)
    kwargs["limits"] = httpx.Limits(
        max_connections=connections, max_keepalive_connections=connections
    )
    # Every phase of a request is bounded, including the wait for a pooled
    # connection, so no single request can hold the phase open forever.
    kwargs["timeout"] = httpx.Timeout(request_timeout)
    return httpx.AsyncClient(**kwargs)


async def _one(client, spec, index, body_limit):
    started = time.monotonic()
    headers = {key: _fill(value, index) for key, value in (spec.get("headers") or {}).items()}
    try:
        response = await client.request(
            spec.get("method", "GET"),
            _fill(spec["url"], index),
            headers=headers,
            content=_fill(spec.get("body") or "", index) or None,
        )
    except Exception as exc:
        return {
            "index": index,
            "status": None,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "error": "%s: %s" % (type(exc).__name__, exc),
            "body": "",
        }
    return {
        "index": index,
        "status": response.status_code,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
        "error": None,
        "body": response.text[:body_limit],
    }


async def _concurrent(client, spec, count, body_limit):
    """Fire every request as one wave: all workers ready, then one release.

    A shared gate is what makes the measurement a burst instead of a ramp.
    The workers park on the gate, the coordinator waits until every one of
    them has parked, and then releases them all in the same event-loop turn.
    """
    gate = asyncio.Event()
    ready = asyncio.Event()
    parked = 0

    async def worker(index):
        nonlocal parked
        parked += 1
        if parked >= count:
            ready.set()
        await gate.wait()
        return await _one(client, spec, index, body_limit)

    tasks = [asyncio.create_task(worker(index)) for index in range(count)]
    try:
        try:
            # The workers only await the gate, so this is near-instant; the
            # bound exists so a wedged worker can never keep the phase waiting.
            await asyncio.wait_for(ready.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        gate.set()
        return list(await asyncio.gather(*tasks))
    finally:
        gate.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _fire(client, spec, mode, count, body_limit):
    if mode == "concurrent":
        return await _concurrent(client, spec, count, body_limit)
    records = []
    for index in range(count):
        records.append(await _one(client, spec, index, body_limit))
    return records


async def _run(payload):
    spec = payload["request"]
    mode = payload.get("mode", "sequential")
    count = int(payload.get("count", 1))
    body_limit = int(payload.get("body_limit") or DEFAULT_BODY_LIMIT)
    request_timeout = float(payload.get("request_timeout") or DEFAULT_REQUEST_TIMEOUT)
    overall_timeout = float(payload.get("overall_timeout") or request_timeout * max(count, 1))
    try:
        async with _build_client(max(count, 8), request_timeout) as client:
            records = await asyncio.wait_for(
                _fire(client, spec, mode, count, body_limit), timeout=overall_timeout
            )
    except asyncio.TimeoutError:
        return {
            "error": "phase exceeded its %.0fs budget" % overall_timeout,
            "records": [],
        }
    except Exception as exc:
        return {"error": "%s: %s" % (type(exc).__name__, exc), "records": []}
    return {"error": None, "records": records}


def main(payload):
    return asyncio.run(_run(payload))


if __name__ == "__main__":
    sys.stdout.write(json.dumps(main(json.loads(sys.stdin.read() or "{}"))))
'''

_SLOT_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


@dataclass(frozen=True)
class RequestSpec:
    """One HTTP request, ready to be fired."""

    method: str = "GET"
    url: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    body: str = ""


@dataclass(frozen=True)
class RaceRun:
    """The outcome of a single fired request."""

    index: int
    status: int | None
    success: bool
    elapsed_ms: int
    error: str | None = None
    body_excerpt: str = ""


@dataclass(frozen=True)
class PhaseResult:
    """Every measurement taken in one phase of the race."""

    name: str
    runs: tuple[RaceRun, ...] = ()
    success_count: int = 0
    total: int = 0
    error: str | None = None


@dataclass(frozen=True)
class RaceResult:
    """The verdict plus the measurements it was derived from."""

    verdict: str
    threshold: int
    baseline: PhaseResult
    burst: PhaseResult
    delta: int = 0
    durability: str = DURABILITY_SKIPPED
    notes: tuple[str, ...] = ()


def substitute_slots(text: str, slots: Mapping[str, str]) -> str:
    """Fill ``{{name}}`` placeholders in ``text`` from ``slots``.

    Placeholders without a matching slot are left untouched, so a request
    body that merely looks like a template is never corrupted.
    """
    if not text or not slots:
        return text
    return _SLOT_RE.sub(lambda match: slots.get(match.group(1), match.group(0)), text)


def default_threshold(burst_size: int = DEFAULT_BURST_SIZE) -> int:
    """Extra concurrent wins required above the sequential baseline.

    Two is the floor: a single extra win is as likely to be a flaky response
    as a race, while larger bursts demand a proportionally larger margin so
    that a noisy target cannot flip the verdict.
    """
    return max(MIN_THRESHOLD, burst_size // 4)


def _is_success(status: int | None, success_statuses: Sequence[int]) -> bool:
    """True when ``status`` counts as the request winning."""
    if status is None:
        return False
    if success_statuses:
        return status in success_statuses
    return 200 <= status < 300


def phase_from_records(
    name: str,
    records: Iterable[Mapping[str, Any]],
    *,
    success_statuses: Sequence[int] = (),
    secrets: Sequence[str] = (),
) -> PhaseResult:
    """Build a :class:`PhaseResult` from the records a phase produced."""
    runs: list[RaceRun] = []
    for position, record in enumerate(records):
        raw_status = record.get("status")
        status = int(raw_status) if isinstance(raw_status, int) else None
        error = record.get("error")
        error_text = http_sender.redact_text(str(error), secrets) if error else None
        excerpt = http_sender.redact_text(str(record.get("body") or ""), secrets)
        runs.append(
            RaceRun(
                index=int(record.get("index") or position),
                status=status,
                success=_is_success(status, success_statuses),
                elapsed_ms=int(record.get("elapsed_ms") or 0),
                error=error_text,
                body_excerpt=excerpt[:BODY_EXCERPT_CHARS],
            )
        )
    return PhaseResult(
        name=name,
        runs=tuple(runs),
        success_count=sum(1 for run in runs if run.success),
        total=len(runs),
    )


def _measured(phase: PhaseResult) -> bool:
    """True when at least one request in the phase reached the target.

    A request that never got a response (a refused connection, a timeout)
    carries no information about the target's concurrency behaviour, so a
    phase without a single completed request is an error rather than a
    measurement.
    """
    return any(run.status is not None for run in phase.runs)


def decide_verdict(
    baseline: PhaseResult,
    burst: PhaseResult,
    threshold: int,
    durability: str = DURABILITY_SKIPPED,
) -> str:
    """Turn two measured phases into a verdict.

    ``NOT_A_RACE`` when every baseline request succeeded: the target never
    guarded the operation, so the burst proves nothing about concurrency.
    Otherwise the signal is the number of extra requests the burst won over
    the baseline, and the verdict only becomes ``RACE_CONFIRMED`` when a
    durability probe passed.
    """
    if baseline.total == 0 or burst.total == 0:
        return VERDICT_ERROR
    if not _measured(baseline) or not _measured(burst):
        return VERDICT_ERROR
    if baseline.success_count == baseline.total:
        return VERDICT_NOT_A_RACE
    if burst.success_count - baseline.success_count < threshold:
        return VERDICT_NOT_A_RACE
    if durability == DURABILITY_PASSED:
        return VERDICT_RACE_CONFIRMED
    return VERDICT_RACE_SUSPECTED


def race_result_to_dict(result: RaceResult) -> dict[str, Any]:
    """Render a :class:`RaceResult` as JSON-ready evidence."""
    return {
        "verdict": result.verdict,
        "threshold": result.threshold,
        "delta": result.delta,
        "durability": result.durability,
        "baseline": _phase_to_dict(result.baseline),
        "burst": _phase_to_dict(result.burst),
        "notes": list(result.notes),
    }


def _phase_to_dict(phase: PhaseResult) -> dict[str, Any]:
    return {
        "total": phase.total,
        "success_count": phase.success_count,
        "error": phase.error,
        "runs": [
            {
                "index": run.index,
                "status": run.status,
                "success": run.success,
                "elapsed_ms": run.elapsed_ms,
                "error": run.error,
                "body_excerpt": run.body_excerpt,
            }
            for run in phase.runs
        ],
    }


async def _run_phase(
    session: Any | None,
    spec: RequestSpec,
    *,
    name: str,
    mode: str,
    count: int,
    timeout: float,
    success_statuses: Sequence[int] = (),
    secrets: Sequence[str] = (),
    body_limit: int = BODY_EXCERPT_CHARS,
) -> PhaseResult:
    """Fire ``count`` copies of ``spec`` and collect what came back."""
    payload: dict[str, Any] = {
        "request": {
            "method": spec.method,
            "url": spec.url,
            "headers": dict(spec.headers),
            "body": spec.body,
        },
        "mode": mode,
        "count": count,
        "body_limit": body_limit,
        # Both bounds travel with the payload: the runner enforces them itself,
        # so a wedged socket or a stalled pool wait ends as a failed phase
        # instead of an unbounded wait in the caller.
        "request_timeout": min(float(timeout), MAX_REQUEST_TIMEOUT_SECONDS),
        "overall_timeout": float(timeout),
    }
    try:
        if http_sender.select_mode(spec.url) == http_sender.MODE_SANDBOX:
            if session is None:
                return PhaseResult(
                    name=name, error="no sandbox session available for a remote target"
                )
            code, stdout, stderr = await http_sender.run_in_session(
                session, RACE_RUNNER_SCRIPT, payload, timeout
            )
        else:
            code, stdout, stderr = await http_sender.run_locally(
                RACE_RUNNER_SCRIPT, payload, timeout
            )
    except Exception as exc:  # noqa: BLE001 - a failed phase is a verdict input, not a crash
        logger.warning("race %s phase failed: %s", name, exc)
        return PhaseResult(name=name, error=f"{type(exc).__name__}: {exc}")

    if code != 0:
        detail = http_sender.redact_text(stderr.strip() or stdout.strip(), secrets)
        return PhaseResult(name=name, error=detail or f"runner exited with code {code}")
    try:
        parsed = json.loads(stdout or "{}")
    except json.JSONDecodeError as exc:
        return PhaseResult(name=name, error=f"runner output was not JSON: {exc}")
    if parsed.get("error"):
        return PhaseResult(name=name, error=http_sender.redact_text(str(parsed["error"]), secrets))
    return phase_from_records(
        name,
        parsed.get("records") or [],
        success_statuses=success_statuses,
        secrets=secrets,
    )


async def run_race_in_session(
    session: Any | None,
    spec: RequestSpec,
    *,
    baseline_runs: int = DEFAULT_BASELINE_SIZE,
    burst_size: int = DEFAULT_BURST_SIZE,
    threshold: int | None = None,
    success_statuses: Sequence[int] = (),
    reset_spec: RequestSpec | None = None,
    probe_spec: RequestSpec | None = None,
    probe_expect_status: int | None = None,
    probe_body_contains: Sequence[str] = (),
    timeout: float = 120.0,
) -> RaceResult:
    """Run both phases of a race against ``spec`` and decide the verdict.

    ``reset_spec`` is fired before each phase so both phases start from the
    same target state; without it the burst runs against whatever the
    baseline left behind. ``probe_spec`` is the durability probe: it is
    fired after the burst and confirms that the duplicated effect is still
    there, which is what separates a confirmed race from a suspected one.
    """
    baseline_count = max(1, baseline_runs)
    burst_count = max(1, burst_size)
    effective_threshold = (
        threshold if threshold and threshold > 0 else default_threshold(burst_count)
    )
    secrets = http_sender.sensitive_header_values(spec.headers)
    notes: list[str] = []

    async def _reset() -> None:
        if reset_spec is None:
            return
        await _run_phase(
            session,
            reset_spec,
            name="reset",
            mode=MODE_SEQUENTIAL,
            count=1,
            timeout=timeout,
            success_statuses=success_statuses,
            secrets=secrets,
        )

    await _reset()
    baseline = await _run_phase(
        session,
        spec,
        name="baseline",
        mode=MODE_SEQUENTIAL,
        count=baseline_count,
        timeout=timeout,
        success_statuses=success_statuses,
        secrets=secrets,
    )
    await _reset()
    burst = await _run_phase(
        session,
        spec,
        name="burst",
        mode=MODE_CONCURRENT,
        count=burst_count,
        timeout=timeout,
        success_statuses=success_statuses,
        secrets=secrets,
    )

    durability = DURABILITY_SKIPPED
    if probe_spec is not None:
        probe = await _run_phase(
            session,
            probe_spec,
            name="durability",
            mode=MODE_SEQUENTIAL,
            count=1,
            timeout=timeout,
            success_statuses=success_statuses,
            secrets=secrets,
            body_limit=PROBE_BODY_CHARS,
        )
        if probe.error or not probe.runs:
            durability = DURABILITY_FAILED
            notes.append(f"durability probe failed: {probe.error or 'no response'}")
        else:
            run = probe.runs[0]
            status_ok = probe_expect_status is None or run.status == probe_expect_status
            body_ok = all(marker in run.body_excerpt for marker in probe_body_contains)
            if status_ok and body_ok:
                durability = DURABILITY_PASSED
            else:
                durability = DURABILITY_FAILED
                notes.append("durability probe did not show the duplicated effect")

    verdict = decide_verdict(baseline, burst, effective_threshold, durability)
    if baseline.error:
        notes.append(f"baseline phase failed: {baseline.error}")
    if burst.error:
        notes.append(f"burst phase failed: {burst.error}")
    if (
        verdict == VERDICT_NOT_A_RACE
        and baseline.total
        and not baseline.error
        and baseline.success_count == baseline.total
    ):
        notes.append(
            "every baseline request succeeded, so the target never guarded the "
            "operation and the burst proves nothing about concurrency"
        )
    return RaceResult(
        verdict=verdict,
        threshold=effective_threshold,
        baseline=baseline,
        burst=burst,
        delta=burst.success_count - baseline.success_count,
        durability=durability,
        notes=tuple(notes),
    )


__all__ = [
    "BODY_EXCERPT_CHARS",
    "DEFAULT_BASELINE_SIZE",
    "DEFAULT_BURST_SIZE",
    "DURABILITY_FAILED",
    "DURABILITY_PASSED",
    "DURABILITY_SKIPPED",
    "INDEX_SLOT",
    "MAX_REQUEST_TIMEOUT_SECONDS",
    "MIN_THRESHOLD",
    "MODE_CONCURRENT",
    "MODE_SEQUENTIAL",
    "PROBE_BODY_CHARS",
    "VERDICT_ERROR",
    "VERDICT_NOT_A_RACE",
    "VERDICT_RACE_CONFIRMED",
    "VERDICT_RACE_SUSPECTED",
    "PhaseResult",
    "RaceResult",
    "RaceRun",
    "RequestSpec",
    "decide_verdict",
    "default_threshold",
    "phase_from_records",
    "race_result_to_dict",
    "run_race_in_session",
    "substitute_slots",
]
