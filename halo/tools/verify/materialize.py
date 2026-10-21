"""Turn a captured or raw request into one that can be fired.

Both verification tools start from the same place: a captured Caido request
or raw HTTP request text. Materializing that once keeps the two tools
sending byte-identical requests, which is what makes their measurements
comparable.
"""

from __future__ import annotations

from typing import Any

from halo.tools.proxy import caido_api
from halo.verify.race import RequestSpec


def parse_raw_request(raw_request: str) -> RequestSpec:
    """Build a request spec from raw HTTP request text.

    The request line needs an absolute URL or a ``Host`` header, since the
    spec carries the URL the request is fired at.
    """
    components = caido_api.parse_raw_request(raw_request)
    first = str(components["url_path"])
    if first.startswith(("http://", "https://")):
        url = first
    else:
        host = str(components["headers"].get("Host") or "")
        if not host:
            raise ValueError("raw_request needs an absolute URL or a Host header")
        url = f"http://{host}{first if first.startswith('/') else '/' + first}"
    return RequestSpec(
        method=str(components["method"]),
        url=url,
        headers={str(k): str(v) for k, v in components["headers"].items()},
        body=str(components["body"]),
    )


async def materialize_request(
    client: Any,
    request_id: str = "",
    raw_request: str = "",
    modifications: dict[str, Any] | None = None,
) -> RequestSpec:
    """Resolve a captured or raw request into a :class:`RequestSpec`.

    ``request_id`` names a captured Caido request, which keeps the scheme,
    host, headers and cookies of the original exchange. ``raw_request`` is
    raw HTTP request text. ``modifications`` patch a captured request the
    same way ``repeat_request`` patches one.
    """
    mods = dict(modifications or {})
    if request_id:
        fetched = await caido_api.get_request_with_client(client, request_id, part="request")
        if fetched is None or fetched.request.raw is None:
            raise ValueError(f"request {request_id} not found or has no raw bytes")
        original = fetched.request
        components = caido_api.parse_raw_request(original.raw.decode("utf-8", errors="replace"))
        full_url = caido_api.full_url_from_components(original, components, mods)
        modified = caido_api.apply_modifications(components, mods, full_url)
        return RequestSpec(
            method=str(modified["method"]),
            url=str(modified["url"]),
            headers={str(k): str(v) for k, v in modified["headers"].items()},
            body=str(modified["body"]),
        )
    if raw_request:
        return parse_raw_request(raw_request)
    raise ValueError("pass request_id or raw_request")


__all__ = ["materialize_request", "parse_raw_request"]
