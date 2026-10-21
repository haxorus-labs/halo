"""Deterministic verification tools: PoC replay and sandbox reset."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from agents import RunContextWrapper, function_tool

from halo.runtime.caido_handle import CaidoBootstrapHandle
from halo.sast.fixtures import match_fixture
from halo.sast.paths import resolve_source_root
from halo.tools.proxy import caido_api
from halo.tools.verify.materialize import materialize_request
from halo.verify.replay import Oracle, result_to_dict, run_replays


logger = logging.getLogger(__name__)

_RESET_TIMEOUT_SECONDS = 120
_RESET_MAX_COMMANDS = 10
_RESET_OUTPUT_CHARS = 2000

_default_source_roots: list[str] = []


def set_default_source_roots(roots: list[str]) -> None:
    global _default_source_roots  # noqa: PLW0603
    _default_source_roots = list(roots)


async def _ctx_client(ctx: RunContextWrapper) -> Any | None:
    inner = ctx.context if isinstance(ctx.context, dict) else {}
    client = inner.get("caido_client")
    if isinstance(client, CaidoBootstrapHandle):
        try:
            return await client.get()
        except Exception:  # noqa: BLE001
            logger.warning("Caido bootstrap failed; proxy tools unavailable", exc_info=True)
            return None
    return client


def _ctx_session(ctx: RunContextWrapper) -> Any | None:
    inner = ctx.context if isinstance(ctx.context, dict) else {}
    return inner.get("sandbox_session")


def _text(value: Any, limit: int = _RESET_OUTPUT_CHARS) -> str:
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value or "")
    return text[-limit:]


async def _build_sender(
    client: Any,
    request_id: str,
    raw_request: str,
    modifications: dict[str, Any],
) -> Any:
    """Return an async send(i) closure dispatching the PoC via Caido replay."""
    spec = await materialize_request(client, request_id, raw_request, modifications)
    connection, raw = caido_api.build_raw_request(
        method=spec.method, url=spec.url, headers=dict(spec.headers), body=spec.body
    )

    async def send(_index: int) -> dict[str, Any]:
        return await caido_api.replay_send_raw(client, raw=raw, connection=connection)

    return send


async def _replay_poc_impl(
    ctx: RunContextWrapper,
    request_id: str,
    raw_request: str,
    modifications: dict[str, Any] | None,
    expect_status: int | None,
    expect_body_contains: list[str],
    expect_body_absent: list[str],
    runs: int,
) -> dict[str, Any]:
    client = await _ctx_client(ctx)
    if client is None:
        return {"error": "Caido client unavailable; proxy tools unavailable"}
    try:
        send = await _build_sender(client, request_id, raw_request, modifications or {})
    except ValueError as exc:
        return {"error": str(exc)}
    oracle = Oracle(
        status=expect_status,
        body_contains=tuple(expect_body_contains),
        body_absent=tuple(expect_body_absent),
    )
    result = await run_replays(send, oracle, runs)
    return result_to_dict(result)


@function_tool(timeout=600, strict_mode=False)
async def replay_poc(
    ctx: RunContextWrapper,
    request_id: str = "",
    raw_request: str = "",
    modifications: dict[str, Any] | None = None,
    expect_status: int | None = None,
    expect_body_contains: list[str] | None = None,
    expect_body_absent: list[str] | None = None,
    runs: int = 3,
) -> str:
    """Replay a PoC request N times, asserting the oracle on every run.

    Produces a reproducible evidence bundle (per-run status, matched/missed
    predicates, response excerpts) instead of a one-off observation: verdict
    is ``REPRODUCED`` only when every run satisfies the oracle. Prefer
    ``request_id`` (a captured Caido request) over ``raw_request`` so scheme,
    host, headers, and cookies stay faithful; ``modifications`` patches the
    captured request (same keys as ``repeat_request``).

    Args:
        request_id: Captured Caido request id to replay.
        raw_request: Raw HTTP request text alternative (needs absolute URL
            or Host header; scheme defaults to http).
        modifications: Patch dict applied to a captured request.
        expect_status: Required response status code (optional).
        expect_body_contains: Substrings every response body must include.
        expect_body_absent: Substrings no response body may include.
        runs: Repetitions, 1-10 (default 3).
    """
    return json.dumps(
        await _replay_poc_impl(
            ctx,
            request_id,
            raw_request,
            modifications,
            expect_status,
            list(expect_body_contains or []),
            list(expect_body_absent or []),
            runs,
        ),
        ensure_ascii=False,
        default=str,
    )


async def _reset_sandbox_impl(ctx: RunContextWrapper, commands: list[str] | None) -> dict[str, Any]:
    session = _ctx_session(ctx)
    if session is None:
        return {"reset_performed": False, "reason": "no sandbox session in run context"}
    resolved = list(commands or [])
    if not resolved:
        source = resolve_source_root("", _default_source_roots) or ""
        fixture = match_fixture(source_path=str(source))
        if fixture is not None and fixture.reset_commands:
            resolved = list(fixture.reset_commands)
    if not resolved:
        return {
            "reset_performed": False,
            "reason": (
                "no reset commands given and no fixture defines any for this target; "
                "pass explicit commands or record target restart steps first"
            ),
        }
    if len(resolved) > _RESET_MAX_COMMANDS:
        return {
            "reset_performed": False,
            "reason": f"too many commands (max {_RESET_MAX_COMMANDS})",
        }
    steps: list[dict[str, Any]] = []
    for cmd in resolved:
        try:
            result = await asyncio.wait_for(
                session.exec("sh", "-c", cmd, timeout=_RESET_TIMEOUT_SECONDS),
                timeout=_RESET_TIMEOUT_SECONDS + 5,
            )
            ok = bool(result.ok())
            steps.append(
                {
                    "command": cmd,
                    "ok": ok,
                    "exit_code": result.exit_code,
                    "stdout": _text(result.stdout),
                    "stderr": _text(result.stderr),
                }
            )
            if not ok:
                break
        except Exception as exc:  # noqa: BLE001
            steps.append({"command": cmd, "ok": False, "error": str(exc)})
            break
    return {"reset_performed": all(s["ok"] for s in steps), "steps": steps}


@function_tool(timeout=300)
async def reset_sandbox(ctx: RunContextWrapper, commands: list[str] | None = None) -> str:
    """Reset target state inside the sandbox by running reset commands.

    Executes each command via the sandbox session (stops at the first
    failure) so a replay starts from identical state. Defaults to the
    target fixture's reset commands when defined; pass explicit commands
    otherwise. Reports honestly when no reset is defined instead of
    pretending.

    Args:
        commands: Shell commands to run for the reset (optional).
    """
    return json.dumps(await _reset_sandbox_impl(ctx, commands), ensure_ascii=False, default=str)
