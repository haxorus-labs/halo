"""Outbound HTTP plumbing for the deterministic verification tools.

A race burst needs many genuinely concurrent requests against one target,
which a serialized proxy API cannot provide, so the traffic is sent directly.
Two details matter here:

* **Where the traffic comes from.** The host and the sandbox are different
  network vantage points. A loopback target is only reachable from the host
  (the sandbox container's own loopback is the container, not the target);
  every other target is reached from the sandbox session, which is where the
  rest of the scan runs.
* **What the evidence shows.** Credential header values (``Authorization``,
  ``Cookie``) never appear in the evidence bundle, not even when a target
  echoes them back in a response body.

The runner script is the same text in both places: the sandbox executes it
as a file, the host executes it in a worker thread. The request logic
therefore exists exactly once.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


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

    The script is executed in a worker thread because it drives its own
    event loop, and it returns its result rather than printing it, so the
    caller never has to scrape stdout.
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
    """Run ``script`` with ``payload`` on the host; return (exit code, stdout, stderr)."""
    result = await asyncio.wait_for(
        asyncio.to_thread(_execute_locally, script, payload),
        timeout=timeout,
    )
    return 0, json.dumps(result, ensure_ascii=False), ""


__all__ = [
    "MODE_LOCAL",
    "MODE_SANDBOX",
    "REDACTED",
    "SENSITIVE_HEADERS",
    "build_command",
    "is_loopback_host",
    "redact_headers",
    "redact_text",
    "run_in_session",
    "run_locally",
    "select_mode",
    "sensitive_header_values",
]
