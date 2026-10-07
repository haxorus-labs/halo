"""Deterministic PoC replay: send N times, assert the oracle each time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from halo.tools.proxy.caido_api import parse_raw_response


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


MAX_RUNS = 10
BODY_EXCERPT_CHARS = 2000


@dataclass(frozen=True)
class Oracle:
    """What a single replayed response must satisfy to count as reproduced."""

    status: int | None = None
    body_contains: tuple[str, ...] = ()
    body_absent: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayRun:
    index: int
    ok: bool
    status_code: int | None
    matched: tuple[str, ...]
    missed: tuple[str, ...]
    elapsed_ms: int
    error: str | None
    body_excerpt: str


@dataclass(frozen=True)
class ReplayResult:
    runs_total: int
    runs_passed: int
    verdict: str  # REPRODUCED | NOT_REPRODUCED | ERROR
    runs: tuple[ReplayRun, ...] = ()
    error: str | None = None


def evaluate_oracle(
    response: dict[str, Any] | None, oracle: Oracle
) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    """Check one parsed response against the oracle predicates."""
    matched: list[str] = []
    missed: list[str] = []
    if response is None:
        return False, (), ("no response",)
    if oracle.status is not None:
        if response.get("status_code") == oracle.status:
            matched.append(f"status=={oracle.status}")
        else:
            missed.append(f"status=={oracle.status} (got {response.get('status_code')})")
    body = str(response.get("body") or "")
    for marker in oracle.body_contains:
        if marker in body:
            matched.append(f"body contains {marker!r}")
        else:
            missed.append(f"body contains {marker!r}")
    for marker in oracle.body_absent:
        if marker not in body:
            matched.append(f"body absent {marker!r}")
        else:
            missed.append(f"body absent {marker!r}")
    return not missed, tuple(matched), tuple(missed)


async def run_replays(
    send_one: Callable[[int], Awaitable[dict[str, Any]]],
    oracle: Oracle,
    runs: int,
) -> ReplayResult:
    """Send the PoC ``runs`` times sequentially, evaluating the oracle per run."""
    total = max(1, min(runs, MAX_RUNS))
    collected: list[ReplayRun] = []
    for index in range(total):
        try:
            replay = await send_one(index)
        except Exception as exc:  # noqa: BLE001 - one failed send must not kill the harness
            collected.append(
                ReplayRun(
                    index=index,
                    ok=False,
                    status_code=None,
                    matched=(),
                    missed=("send failed",),
                    elapsed_ms=0,
                    error=str(exc),
                    body_excerpt="",
                )
            )
            continue
        response = parse_raw_response(replay.get("response_raw"))
        ok, matched, missed = evaluate_oracle(response, oracle)
        body = str((response or {}).get("body") or "")
        collected.append(
            ReplayRun(
                index=index,
                ok=ok,
                status_code=(response or {}).get("status_code"),
                matched=matched,
                missed=missed,
                elapsed_ms=int(replay.get("elapsed_ms") or 0),
                error=replay.get("error"),
                body_excerpt=body[:BODY_EXCERPT_CHARS],
            )
        )
    passed = sum(1 for run in collected if run.ok)
    verdict = "REPRODUCED" if passed == total else "NOT_REPRODUCED"
    return ReplayResult(
        runs_total=total,
        runs_passed=passed,
        verdict=verdict,
        runs=tuple(collected),
    )


def result_to_dict(result: ReplayResult) -> dict[str, Any]:
    return {
        "verdict": result.verdict,
        "runs_total": result.runs_total,
        "runs_passed": result.runs_passed,
        "error": result.error,
        "runs": [
            {
                "index": run.index,
                "ok": run.ok,
                "status_code": run.status_code,
                "matched": list(run.matched),
                "missed": list(run.missed),
                "elapsed_ms": run.elapsed_ms,
                "error": run.error,
                "body_excerpt": run.body_excerpt,
            }
            for run in result.runs
        ],
    }
