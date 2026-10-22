"""Outbound HTTP plumbing for the deterministic verification tools.

Two callers share this module:

* a race burst needs many genuinely concurrent requests against one target,
  which a serialized proxy API cannot provide, so the traffic is sent
  directly. The host and the sandbox are different network vantage points: a
  loopback target is only reachable from the host, every other target from the
  sandbox session where the rest of the scan runs.
* an authorization matrix sends its probes one at a time and compares the
  responses. Probe order is part of the evidence, so parallel sends would race
  on rate limits and session state.

Every network wait is bounded. A request can stall while connecting, sending,
waiting for the response head or reading the body, so timeouts are set per
phase rather than as one blanket number, and whole sequences get a wall-clock
budget. Credential header values (``Authorization``, ``Cookie``) never appear
in evidence, not even when a target echoes them back in a response body.

The race runner script is the same text in both places: the sandbox executes
it as a file, the host executes it on a worker thread. The request logic
therefore exists exactly once.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


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

#: Header names whose values are credentials and must never reach evidence.
SENSITIVE_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie"})

REDACTED = "[REDACTED]"

MODE_LOCAL = "local"
MODE_SANDBOX = "sandbox"

# The runner is written inside the sandbox container, so the path is a Linux
# one regardless of where the command is assembled. The file is written and
# consumed by the same command, so it never outlives the burst.
_RUNNER_PATH = "/tmp/halo_race_runner.py"  # noqa: S108  # nosec B108 - sandbox scratch file
_PAYLOAD_PATH = "/tmp/halo_race_payload.json"  # noqa: S108  # nosec B108 - sandbox scratch file


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


def sensitive_header_values(headers: Mapping[str, str]) -> list[str]:
    """Collect the credential values carried by ``headers``."""
    return [
        value
        for name, value in headers.items()
        if name.strip().lower() in SENSITIVE_HEADERS and value
    ]


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Copy ``headers`` with every credential value replaced by a marker."""
    return {
        name: REDACTED if name.strip().lower() in SENSITIVE_HEADERS else value
        for name, value in headers.items()
    }


def redact_text(text: str, secrets: Sequence[str] = ()) -> str:
    """Replace every known credential value inside ``text``."""
    if not text:
        return text
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    return text


def is_loopback_host(host: str) -> bool:
    """True when ``host`` names the host machine itself."""
    name = host.strip().lower().rstrip(".")
    if name in {"localhost", "127.0.0.1", "::1"}:
        return True
    if name.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def select_mode(url: str) -> str:
    """Pick the execution vantage point for ``url``.

    A loopback target is hit from the host, because the sandbox container's
    loopback is the container itself rather than the target. Every other
    target is reached from the sandbox session.
    """
    host = urlsplit(url).hostname or ""
    if is_loopback_host(host):
        return MODE_LOCAL
    return MODE_SANDBOX


def build_command(
    script: str,
    payload: Mapping[str, Any],
    *,
    runner_path: str = _RUNNER_PATH,
    payload_path: str = _PAYLOAD_PATH,
) -> str:
    """Build a shell command that runs ``script`` with ``payload`` on stdin.

    Both parts travel base64-encoded inside quoted heredocs, so no quoting
    rule of the shell can be tripped by a header value or a JSON body.
    """
    script_b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    payload_b64 = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8")).decode(
        "ascii"
    )
    return (
        f"base64 -d > {runner_path} <<'HALO_RUNNER_EOF'\n"
        f"{script_b64}\n"
        f"HALO_RUNNER_EOF\n"
        f"base64 -d > {payload_path} <<'HALO_PAYLOAD_EOF'\n"
        f"{payload_b64}\n"
        f"HALO_PAYLOAD_EOF\n"
        f"python3 {runner_path} < {payload_path}"
    )


def _execute_locally(script: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run the runner script in this process and return its result.

    The script drives its own event loop, so it runs on a dedicated thread and
    returns its result rather than printing it, so the caller never has to
    scrape stdout.
    """
    namespace: dict[str, Any] = {"__name__": "halo_race_runner", "__file__": "<halo_race_runner>"}
    exec(compile(script, "<halo_race_runner>", "exec"), namespace)  # noqa: S102  # nosec B102
    entry = namespace.get("main")
    if not callable(entry):
        raise TypeError("runner script does not define a callable main()")
    result = entry(json.loads(json.dumps(payload, ensure_ascii=False)))
    if not isinstance(result, dict):
        raise TypeError("runner script returned a non-mapping result")
    return result


async def run_in_session(
    session: Any,
    script: str,
    payload: Mapping[str, Any],
    timeout: float,
) -> tuple[int, str, str]:
    """Run ``script`` in the sandbox session; return (exit code, stdout, stderr)."""
    result = await session.exec("sh", "-c", build_command(script, payload), timeout=timeout)
    stdout = result.stdout
    stderr = result.stderr
    out_text = (
        stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else str(stdout or "")
    )
    err_text = (
        stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else str(stderr or "")
    )
    return int(result.exit_code or 0), out_text, err_text


async def run_locally(
    script: str,
    payload: Mapping[str, Any],
    timeout: float,
) -> tuple[int, str, str]:
    """Run ``script`` with ``payload`` on the host; return (exit code, stdout, stderr).

    The work happens on a daemon thread rather than an executor thread. That
    distinction matters: ``asyncio.run`` joins the loop's default executor at
    shutdown with no timeout, so a phase that overran its budget would still
    pin the process forever. A daemon thread is never joined, so a slow or
    wedged target can only produce a timed-out phase.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()

    def _settle(result: dict[str, Any] | None, error: BaseException | None) -> None:
        if future.done():  # the caller already timed out and moved on
            return
        if error is not None:
            future.set_exception(error)
        elif result is not None:
            future.set_result(result)

    def _deliver(result: dict[str, Any] | None, error: BaseException | None) -> None:
        """Hand the outcome back to the loop, tolerating a loop that moved on."""
        if loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(_settle, result, error)
        except RuntimeError:  # the loop closed between the check and the call
            return

    def _target() -> None:
        try:
            outcome = _execute_locally(script, payload)
        except BaseException as exc:  # noqa: BLE001 - forwarded to the awaiting coroutine
            _deliver(None, exc)
        else:
            _deliver(outcome, None)

    threading.Thread(target=_target, name="halo-race-runner", daemon=True).start()
    try:
        result = await asyncio.wait_for(future, timeout=timeout)
    except TimeoutError:
        raise TimeoutError(f"race phase exceeded its {timeout:.0f}s budget") from None
    return 0, json.dumps(result, ensure_ascii=False), ""


__all__ = [
    "BUDGET_EXHAUSTED_ERROR",
    "CONNECT_TIMEOUT_S",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT_S",
    "MAX_BODY_CHARS",
    "MAX_WALL_CLOCK_S",
    "MODE_LOCAL",
    "MODE_SANDBOX",
    "REDACTED",
    "RETRY_BACKOFF_S",
    "SENSITIVE_HEADERS",
    "HttpRequest",
    "HttpResponse",
    "HttpSender",
    "HttpxSender",
    "build_command",
    "clamp_timeout",
    "is_http_url",
    "is_loopback_host",
    "redact_headers",
    "redact_text",
    "run_in_session",
    "run_locally",
    "select_mode",
    "send_sequential",
    "sensitive_header_values",
]
