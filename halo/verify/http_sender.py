"""Shared HTTP transport for authorization-matrix probes.

Two transports share one request/response shape:

* :class:`HttpxSender` — host-side, ``httpx`` client, sequential sends.
* ``run_matrix_in_session`` (in :mod:`halo.verify.matrix`) — the same plan
  replayed inside the sandbox for targets that are only reachable there.

Sends are sequential by design: the probe order is part of the evidence, and
parallel sends would race on rate limits, session state and log ordering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)

MAX_BODY_CHARS = 4096
DEFAULT_TIMEOUT_S = 15.0


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

    async def send(self, request: HttpRequest) -> HttpResponse:  # pragma: no cover - interface
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release transport resources (no-op for stateless transports)."""


class HttpxSender(HttpSender):
    """Host-side sender built on ``httpx`` with redirects left unfollowed.

    Redirects are reported rather than followed: a login redirect is evidence
    of enforced access control, while following it would hide that signal.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_S) -> None:
        self._timeout = timeout
        self._client: Any | None = None

    async def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = httpx.AsyncClient(follow_redirects=False, timeout=self._timeout)
        return self._client

    async def send(self, request: HttpRequest) -> HttpResponse:
        client = await self._ensure_client()
        try:
            response = await client.request(
                request.method,
                request.url,
                headers=dict(request.headers),
                content=request.body or None,
            )
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

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()


async def send_sequential(
    sender: HttpSender,
    requests: Sequence[HttpRequest],
) -> list[HttpResponse]:
    """Send every request in order and return the responses in the same order."""
    return [await sender.send(request) for request in requests]
