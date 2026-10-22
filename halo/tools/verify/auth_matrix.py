"""Authorization-matrix tool: probe an API as several personas and diff.

Wraps the pure engine in :mod:`halo.verify.matrix` as an agent tool. The tool
only measures and reports: it never files findings, so the agent stays in
control of what becomes a vulnerability report.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents import RunContextWrapper, function_tool

from halo.verify.http_sender import (
    DEFAULT_TIMEOUT_S,
    MAX_WALL_CLOCK_S,
    HttpxSender,
    clamp_timeout,
    is_http_url,
    send_sequential,
)
from halo.verify.matrix import (
    MAX_MATRIX_CELLS,
    MatrixOperation,
    MatrixPersona,
    build_requests,
    cells_from_responses,
    compute_differentials,
    ensure_anon_persona,
    matrix_to_dict,
    operation_from_dict,
    persona_from_dict,
    run_matrix_in_session,
)


logger = logging.getLogger(__name__)


def _ctx_session(ctx: RunContextWrapper) -> Any | None:
    inner = ctx.context if isinstance(ctx.context, dict) else {}
    return inner.get("sandbox_session")


def _parse_matrix(
    base_url: str,
    operations: list[dict[str, Any]] | None,
    personas: list[dict[str, Any]] | None,
) -> tuple[tuple[MatrixOperation, ...], tuple[MatrixPersona, ...]] | dict[str, Any]:
    """Parse the tool inputs, or return the structured error to report."""
    if not base_url.strip():
        return {"error": "base_url is required"}
    try:
        parsed_operations = tuple(operation_from_dict(raw) for raw in (operations or []))
        parsed_personas = tuple(persona_from_dict(raw) for raw in (personas or []))
    except (TypeError, ValueError) as exc:
        return {"error": f"invalid matrix input: {exc}"}
    if not parsed_operations:
        return {"error": "operations is required; list the endpoints to probe"}
    with_anon = ensure_anon_persona(parsed_personas)
    if len(parsed_operations) * len(with_anon) > MAX_MATRIX_CELLS:
        return {
            "error": (
                f"matrix too large: {len(parsed_operations)} operations x "
                f"{len(with_anon)} personas exceeds {MAX_MATRIX_CELLS} cells"
            )
        }
    return parsed_operations, with_anon


async def _auth_matrix_impl(
    ctx: RunContextWrapper,
    base_url: str,
    operations: list[dict[str, Any]] | None,
    personas: list[dict[str, Any]] | None,
    transport: str,
    timeout: float,
) -> dict[str, Any]:
    parsed = _parse_matrix(base_url, operations, personas)
    if isinstance(parsed, dict):
        return parsed
    parsed_operations, parsed_personas = parsed

    request_timeout = clamp_timeout(timeout)
    plan = build_requests(parsed_operations, parsed_personas, base_url)
    if plan and not all(is_http_url(request.url) for _op, _p, request in plan):
        return {"error": "base_url must be an absolute http:// or https:// URL"}

    try:
        if transport == "sandbox":
            session = _ctx_session(ctx)
            if session is None:
                return {"error": "sandbox transport requested but no sandbox session is available"}
            responses = await run_matrix_in_session(session, plan, timeout=request_timeout)
        else:
            sender = HttpxSender(timeout=request_timeout)
            try:
                responses = await send_sequential(
                    sender,
                    [request for _op, _p, request in plan],
                    budget_s=MAX_WALL_CLOCK_S,
                )
            finally:
                await sender.aclose()
    except Exception as exc:
        logger.exception("auth_matrix run failed")
        return {"error": f"auth matrix run failed: {exc}"}

    cells = cells_from_responses(plan, responses)
    candidates, withheld = compute_differentials(parsed_operations, parsed_personas, cells)
    return matrix_to_dict(parsed_operations, parsed_personas, cells, candidates, withheld)


@function_tool(timeout=900, strict_mode=False)
async def auth_matrix(
    ctx: RunContextWrapper,
    base_url: str,
    operations: list[dict[str, Any]] | None = None,
    personas: list[dict[str, Any]] | None = None,
    transport: str = "host",
    timeout: float = DEFAULT_TIMEOUT_S,
) -> str:
    """Probe an API as several personas and report differential access results.

    Sends every operation as every persona (sequentially, so the probe order
    is reproducible) and compares the answers. An implicit ``anon`` persona is
    always added, so unauthenticated reachability is measured even when you
    only supply authenticated personas.

    Each operation is a mapping:

    - ``id`` - unique name for the operation (required).
    - ``method`` - HTTP method (default ``GET``).
    - ``path`` - path appended to ``base_url`` (required). ``{token}`` is
      replaced with the persona's token.
    - ``headers`` / ``body`` - extra request parts; ``{token}`` is substituted.
    - ``owner`` - persona id that owns the object this operation touches.
      Set it to enable the horizontal-IDOR rule.
    - ``allowed_roles`` - roles that should be allowed to call it. Set it to
      enable the vertical-escalation rule.
    - ``public`` / ``shared`` - mark a genuinely public or intentionally
      shared operation so the corresponding rules do not fire.

    Each persona is a mapping:

    - ``id`` - unique name (required).
    - ``role`` - role name (default ``user``); ``user``, ``staff``, ``admin``
      and ``owner`` are ranked for the vertical rule.
    - ``token`` - credential sent as ``Authorization`` (and substituted into
      ``{token}`` placeholders).
    - ``headers`` - extra headers for this persona.

    The result contains every cell (including the ``inconclusive`` ones, which
    are kept rather than dropped) plus advisory candidates. Each candidate
    carries the evidence that produced it and the counterevidence that could
    explain it away. Differential rules compare response *content and
    structure* against a baseline observation, not just status codes; a
    status-only match whose body proves the probe got a different document is
    reported under ``withheld`` with the reason instead of becoming a
    candidate. The tool never files findings itself.

    Args:
        base_url: Target base URL (e.g. ``http://localhost:3000``).
        operations: Operations to probe (required).
        personas: Personas to probe with; ``anon`` is added automatically.
        transport: ``host`` (default) sends from the host, ``sandbox`` sends
            from inside the sandbox container.
        timeout: Per-request timeout in seconds.
    """
    return json.dumps(
        await _auth_matrix_impl(
            ctx,
            base_url,
            operations,
            personas,
            transport,
            timeout,
        ),
        ensure_ascii=False,
        default=str,
    )
