"""Race / TOCTOU tool: fire one request concurrently and compare the outcome.

The engine lives in :mod:`halo.verify.race`; this module is the thin agent
facing wrapper that materializes the request, runs both phases, and attaches
the proxy side evidence.
"""

from __future__ import annotations

import ipaddress
import json
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from agents import RunContextWrapper, function_tool

from halo.runtime.caido_handle import CaidoBootstrapHandle
from halo.tools.proxy import caido_api
from halo.tools.verify.materialize import materialize_request, parse_raw_request
from halo.verify import http_sender
from halo.verify.race import (
    DURABILITY_SKIPPED,
    INDEX_SLOT,
    VERDICT_RACE_CONFIRMED,
    VERDICT_RACE_SUSPECTED,
    RequestSpec,
    race_result_to_dict,
    run_race_in_session,
    substitute_slots,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


logger = logging.getLogger(__name__)

#: Upper bound on concurrent requests; a burst is a race window, not a flood.
MAX_BURST_SIZE = 32
MAX_BASELINE_RUNS = 16
#: Wall clock budget for one phase.
PHASE_TIMEOUT_SECONDS = 120


async def _ctx_client(ctx: RunContextWrapper) -> Any | None:
    inner = ctx.context if isinstance(ctx.context, dict) else {}
    client = inner.get("caido_client")
    if isinstance(client, CaidoBootstrapHandle):
        try:
            return await client.get()
        except Exception:  # noqa: BLE001
            logger.warning("Caido bootstrap failed; race evidence unavailable", exc_info=True)
            return None
    return client


def _ctx_session(ctx: RunContextWrapper) -> Any | None:
    inner = ctx.context if isinstance(ctx.context, dict) else {}
    return inner.get("sandbox_session")


def _ctx_scan_targets(ctx: RunContextWrapper) -> list[str]:
    inner = ctx.context if isinstance(ctx.context, dict) else {}
    targets = inner.get("scan_targets")
    return [str(target) for target in targets] if isinstance(targets, list) else []


def _host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _is_local_host(host: str) -> bool:
    """True for hosts that name this machine or the sandbox gateway."""
    if not host:
        return False
    if http_sender.is_loopback_host(host):
        return True
    if host in {"host.docker.internal", "host.containers.internal"}:
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return False


def _in_scope(host: str, scan_targets: list[str]) -> bool:
    """True when ``host`` is covered by the scan's declared targets.

    A scan that declared no host at all (a source-only scan) authorizes no
    remote host, so nothing remote is in scope.
    """
    for target in scan_targets:
        target_host = _host_of(target)
        if not target_host:
            continue
        if host == target_host or host.endswith(f".{target_host}"):
            return True
    return False


def _apply_slots(spec: RequestSpec, slots: Mapping[str, str]) -> RequestSpec:
    """Fill ``{{name}}`` placeholders in a request from ``slots``."""
    if not slots:
        return spec
    return RequestSpec(
        method=spec.method,
        url=substitute_slots(spec.url, slots),
        headers={key: substitute_slots(value, slots) for key, value in spec.headers.items()},
        body=substitute_slots(spec.body, slots),
    )


async def _caido_exchange_evidence(client: Any, spec: RequestSpec) -> dict[str, Any]:
    """Replay one exemplar through the proxy and name the exchange it created.

    The burst itself is fired outside the proxy because the proxy API
    serializes requests, which is exactly what a race needs to break. One
    exemplar is therefore replayed through the proxy afterwards so the run
    carries a proxy side exchange id as evidence.
    """
    evidence: dict[str, Any] = {}
    try:
        connection, raw = caido_api.build_raw_request(
            method=spec.method,
            url=spec.url,
            headers=dict(spec.headers),
            body=spec.body,
        )
        replay = await caido_api.replay_send_raw(client, raw=raw, connection=connection)
    except Exception as exc:  # noqa: BLE001 - evidence is best effort, never fatal
        logger.warning("exemplar replay failed: %s", exc)
        return {"error": f"{type(exc).__name__}: {exc}"}
    evidence["status"] = replay.get("status")
    evidence["session_id"] = replay.get("session_id")
    if replay.get("error"):
        evidence["error"] = replay.get("error")
    host = _host_of(spec.url)
    try:
        connection = await caido_api.list_requests_with_client(
            client,
            httpql_filter=f'req.host.eq:"{host}" AND req.method.eq:"{spec.method}"',
            first=5,
            sort_by="timestamp",
            sort_order="desc",
        )
        edges = list(getattr(connection, "edges", []) or [])
        if edges:
            evidence["exchange_id"] = str(edges[0].node.request.id)
    except Exception as exc:  # noqa: BLE001 - the exchange id is a bonus, not the proof
        logger.warning("exchange lookup failed: %s", exc)
        evidence["exchange_id_error"] = f"{type(exc).__name__}: {exc}"
    return evidence


async def _race_burst_impl(
    ctx: RunContextWrapper,
    request_id: str = "",
    raw_request: str = "",
    modifications: dict[str, Any] | None = None,
    url: str = "",
    method: str = "",
    headers: dict[str, str] | None = None,
    body: str = "",
    slots: dict[str, str] | None = None,
    burst_size: int = 8,
    baseline_runs: int = 8,
    threshold: int = 0,
    success_statuses: list[int] | None = None,
    reset_request: str = "",
    probe_request: str = "",
    probe_status: int | None = None,
    probe_body_contains: list[str] | None = None,
) -> dict[str, Any]:
    try:
        slot_values = dict(slots or {})
        if INDEX_SLOT in slot_values:
            return {"error": f"the {INDEX_SLOT!r} slot is reserved for the request index"}

        client = await _ctx_client(ctx)
        if request_id or raw_request:
            spec = await materialize_request(client, request_id, raw_request, modifications)
        elif url:
            spec = RequestSpec(
                method=method or "GET",
                url=url,
                headers={str(k): str(v) for k, v in (headers or {}).items()},
                body=body,
            )
        else:
            return {"error": "pass request_id, raw_request, or url"}

        spec = _apply_slots(spec, slot_values)

        host = _host_of(spec.url)
        if not host:
            return {"error": "the request needs an absolute URL"}
        if not _is_local_host(host) and not _in_scope(host, _ctx_scan_targets(ctx)):
            return {
                "error": (
                    f"{host} is outside the scope of this scan; race bursts only "
                    "target the declared scan targets or local hosts"
                )
            }

        burst = max(1, min(int(burst_size), MAX_BURST_SIZE))
        baseline = max(1, min(int(baseline_runs), MAX_BASELINE_RUNS))

        reset_spec = parse_raw_request(reset_request) if reset_request else None
        probe_spec = parse_raw_request(probe_request) if probe_request else None

        session = _ctx_session(ctx)
        result = await run_race_in_session(
            session,
            spec,
            baseline_runs=baseline,
            burst_size=burst,
            threshold=int(threshold or 0),
            success_statuses=tuple(int(status) for status in (success_statuses or [])),
            reset_spec=reset_spec,
            probe_spec=probe_spec,
            probe_expect_status=probe_status,
            probe_body_contains=tuple(probe_body_contains or []),
            timeout=PHASE_TIMEOUT_SECONDS,
        )

        evidence: dict[str, Any] = race_result_to_dict(result)
        evidence["request"] = {
            "method": spec.method,
            "url": spec.url,
            "headers": http_sender.redact_headers(spec.headers),
        }
        notes = list(evidence.get("notes") or [])
        if result.durability == DURABILITY_SKIPPED:
            notes.append("no durability probe was run, so the race is only suspected")
        elif result.verdict == VERDICT_RACE_CONFIRMED:
            notes.append("the durability probe confirmed the duplicated effect persisted")
        elif result.verdict == VERDICT_RACE_SUSPECTED:
            notes.append(
                "the burst won more requests than the baseline, but the effect was "
                "not confirmed to persist"
            )
        evidence["notes"] = notes
        if client is None:
            evidence["caido"] = {"error": "Caido client unavailable; no proxy evidence"}
        elif http_sender.select_mode(spec.url) != http_sender.MODE_LOCAL:
            evidence["caido"] = await _caido_exchange_evidence(client, spec)
    except Exception as exc:  # a tool must never raise into the agent loop
        logger.exception("race_burst failed")
        return {"error": f"{type(exc).__name__}: {exc}"}
    return evidence


@function_tool(timeout=600, strict_mode=False)
async def race_burst(
    ctx: RunContextWrapper,
    request_id: str = "",
    raw_request: str = "",
    modifications: dict[str, Any] | None = None,
    url: str = "",
    method: str = "",
    headers: dict[str, str] | None = None,
    body: str = "",
    slots: dict[str, str] | None = None,
    burst_size: int = 8,
    baseline_runs: int = 8,
    threshold: int = 0,
    success_statuses: list[int] | None = None,
    reset_request: str = "",
    probe_request: str = "",
    probe_status: int | None = None,
    probe_body_contains: list[str] | None = None,
) -> str:
    """Fire one request many times at once and prove a race condition.

    The same request is fired twice: once one at a time (the baseline) and
    once all at once (the burst). When the burst wins clearly more requests
    than the baseline did, the target protected a read-modify-write window
    it does not control, which is a race condition. The verdict is
    ``RACE_CONFIRMED`` when a durability probe confirms the duplicated effect
    persisted, ``RACE_SUSPECTED`` when it does not, and ``NOT_A_RACE``
    otherwise. The tool reports the measurements; it never files a finding.

    Burst targets are restricted to this scan's declared targets and to local
    hosts.

    Args:
        request_id: Captured Caido request id to fire (preferred; keeps the
            original scheme, host, headers and cookies).
        raw_request: Raw HTTP request text alternative (needs an absolute URL
            or a Host header).
        modifications: Patch dict applied to a captured request (same keys as
            ``repeat_request``).
        url: Absolute URL to fire when there is no captured request.
        method: HTTP method for the ``url`` form (default ``GET``).
        headers: Request headers for the ``url`` form.
        body: Request body for the ``url`` form.
        slots: Values for ``{{name}}`` placeholders in the URL, headers and
            body, so one template can drive several requests. ``{{index}}`` is
            reserved and expands to the request index.
        burst_size: Concurrent requests in the burst phase (1-32, default 8).
        baseline_runs: Sequential requests in the baseline phase (1-16,
            default 8). Keep it equal to ``burst_size`` so the two phases are
            comparable.
        threshold: Extra requests the burst must win over the baseline
            (default: derived from ``burst_size``).
        success_statuses: Status codes that count as "the request won"
            (default: any 2xx).
        reset_request: Raw HTTP request that restores the target's state,
            fired before each phase so both phases start alike.
        probe_request: Raw HTTP request that reads the state back after the
            burst; the durability probe.
        probe_status: Status the probe must return to count as passed.
        probe_body_contains: Substrings the probe response must contain to
            count as passed.
    """
    return json.dumps(
        await _race_burst_impl(
            ctx,
            request_id=request_id,
            raw_request=raw_request,
            modifications=modifications,
            url=url,
            method=method,
            headers=headers,
            body=body,
            slots=slots,
            burst_size=burst_size,
            baseline_runs=baseline_runs,
            threshold=threshold,
            success_statuses=success_statuses,
            reset_request=reset_request,
            probe_request=probe_request,
            probe_status=probe_status,
            probe_body_contains=probe_body_contains,
        ),
        ensure_ascii=False,
        default=str,
    )


__all__ = ["race_burst"]
