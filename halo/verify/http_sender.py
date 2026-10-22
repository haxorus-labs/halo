"""Shared HTTP transport for authorization-matrix probes.

Two transports share one request/response shape:

* :class:`HttpxSender` - host-side, ``httpx`` client, sequential sends.
* ``run_matrix_in_session`` (in :mod:`halo.verify.matrix`) - the same plan
  replayed inside the sandbox for targets that are only reachable there.

Every network wait is bounded. A probe can stall in four places - connecting,
sending, waiting for the response head, and reading the body - so the timeout
is set per phase rather than as one blanket number, and ``send_sequential``
enforces a wall-clock budget on the whole plan. A matrix that cannot finish
inside that budget yields ``inconclusive`` cells with the reason attached; it
never blocks the scan and it never silently drops a probe.

Sends are sequential by design: the probe order is part of the evidence, and
parallel sends would race on rate limits, session state and log ordering.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)

#: Bodies are truncated here; fingerprints are taken from the full slice.
MAX_BODY_CHARS = 4096

#: Default per-request read timeout. A probe is a single small request, so a
#: server that has not finished answering in this long is not going to.
DEFAULT_TIMEOUT_S = 15.0

#: Connecting is much cheaper than waiting for an answer, so it gets a
#: tighter bound: an unreachable host should not consume the whole budget.
CONNECT_TIMEOUT_S = 5.0

#: Retries are capped, and only transport failures are retried. An HTTP
#: status is an observation, not a failure, and re-probing it would corrupt
#: the evidence the matrix is built from.
DEFAULT_MAX_RETRIES = 1
RETRY_BACKOFF_S = 0.2

#: Whole-plan wall clock. This is the backstop that keeps a hung target from
#: parking an agent: the tool's own timeout is longer than this on purpose.
MAX_WALL_CLOCK_S = 600.0

BUDGET_EXHAUSTED_ERROR = "matrix wall-clock budget exhausted before the probe ran"

_HTTP_SCHEMES = frozenset({"http", "https"})


@dataclass(frozen=True)
class HttpRequest:
    """One planned request (no credentials resolved yet)."""

    method: str
    url: str
    headers: tuple[tuple[str, str], ...] = ()
    body: str = ""


@dataclass(frozen=True)
class HttpResponse:
    """The observed response for a request, normalized across transports."""

    status: int | None = None
    headers: tuple[tuple[str, str], ...] = ()
    body: str = ""
    error: str | None = None
    location: str | None = None


class HttpSender:
    """Protocol-ish base for a transport that answers one request at a time."""

    #: Per-request read timeout the sender applies, in seconds.
    timeout: float = DEFAULT_TIMEOUT_S

    async def send(self, request: HttpRequest) -> HttpResponse:  # pragma: no cover - interface
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release transport resources (no-op for stateless transports)."""


def is_http_url(url: str) -> bool:
    """True when ``url`` is an absolute ``http``/``https`` URL.

    Anything else (``file:``, ``gopher:``, a bare path) is refused before a
    probe is planned, so the transport can only talk to an HTTP endpoint.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme.lower() in _HTTP_SCHEMES and bool(parts.hostname)


class HttpxSender(HttpSender):
    """Host-side sender built on ``httpx`` with redirects left unfollowed.

    Redirects are reported rather than followed: a login redirect is evidence
    of enforced access control, while following it would hide that signal.
    """

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT_S,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.timeout = clamp_timeout(timeout)
        self._max_retries = max(0, min(int(max_retries), DEFAULT_MAX_RETRIES))
        self._client: Any | None = None

    async def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = httpx.AsyncClient(
                follow_redirects=False,
                timeout=httpx.Timeout(
                    connect=min(CONNECT_TIMEOUT_S, self.timeout),
                    read=self.timeout,
                    write=self.timeout,
                    pool=CONNECT_TIMEOUT_S,
                ),
                # One connection, no keep-alive pool: probes are sequential,
                # and a pooled socket to a previous gateway is a liability.
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
            )
        return self._client

    async def send(self, request: HttpRequest) -> HttpResponse:
        if not is_http_url(request.url):
            return HttpResponse(error=f"refused non-HTTP URL: {request.url!r}")
        client = await self._ensure_client()
        attempts = self._max_retries + 1
        last_error = "request failed"
        for attempt in range(attempts):
            try:
                response = await client.request(
                    request.method,
                    request.url,
                    headers=dict(request.headers),
                    content=request.body or None,
                )
            except (httpx.TransportError, httpx.InvalidURL) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.debug("probe to %s failed (attempt %d): %s", request.url, attempt, exc)
                if attempt + 1 < attempts:
                    await asyncio.sleep(RETRY_BACKOFF_S)
                    continue
                return HttpResponse(error=last_error)
            except Exception as exc:  # noqa: BLE001 - one failed probe must not kill the matrix
                logger.debug("request to %s failed: %s", request.url, exc)
                return HttpResponse(error=str(exc))
            body = response.text[:MAX_BODY_CHARS]
            return HttpResponse(
                status=response.status_code,
                headers=tuple(response.headers.items()),
                body=body,
                location=response.headers.get("location"),
            )
        return HttpResponse(error=last_error)

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()


def clamp_timeout(timeout: float | None) -> float:
    """Keep a caller-supplied timeout inside the range a probe can use."""
    try:
        value = float(timeout) if timeout is not None else DEFAULT_TIMEOUT_S
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_S
    if not value > 0:
        return DEFAULT_TIMEOUT_S
    return min(value, MAX_WALL_CLOCK_S)


async def send_sequential(
    sender: HttpSender,
    requests: Sequence[HttpRequest],
    *,
    budget_s: float = MAX_WALL_CLOCK_S,
) -> list[HttpResponse]:
    """Send every request in order and return the responses in the same order.

    Each send is bounded by ``sender.timeout`` and the whole sequence by
    ``budget_s``. Probes that cannot run inside the budget come back as
    ``inconclusive`` responses carrying the reason, so the cell stays in the
    matrix instead of vanishing.
    """
    budget = max(0.0, float(budget_s))
    deadline = time.monotonic() + budget
    per_request = clamp_timeout(getattr(sender, "timeout", DEFAULT_TIMEOUT_S))
    responses: list[HttpResponse] = []
    for request in requests:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            responses.append(HttpResponse(error=BUDGET_EXHAUSTED_ERROR))
            continue
        try:
            responses.append(
                await asyncio.wait_for(sender.send(request), timeout=min(remaining, per_request))
            )
        except TimeoutError:
            responses.append(
                HttpResponse(error=f"request timed out after {min(remaining, per_request):.1f}s")
            )
    return responses
