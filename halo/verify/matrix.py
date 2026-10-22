"""Authorization matrix engine: persona x operation differential analysis.

The engine is pure: it plans requests, classifies responses and derives
advisory candidates, but never touches the network itself. Transports live in
:mod:`halo.verify.http_sender` (host-side ``httpx``) and in
:func:`run_matrix_in_session` (the same plan replayed inside the sandbox).

Differential rules enforced here:

* **anonymous access** - an operation expected to be protected answers an
  unauthenticated probe with real content.
* **horizontal IDOR** - a peer of the resource owner reads the owner's object.
* **vertical escalation** - a lower-privileged role reaches a restricted
  operation.
* **method asymmetry** - a state-changing method is allowed where the read
  method on the same path is denied.

Every rule is differential across *content*, not just status: the three access
rules require a baseline observation (the owner for horizontal, an authorized
role for vertical, an authenticated caller for anonymous) and compare the
probe's body against it. Identical content is the strongest evidence, an equal
JSON structure with different values is weaker but still evidence, and a body
that differs outright means the probe was served *something else* - a login
page rather than the object - so the candidate is withheld and recorded in
``withheld`` with the reason. When no baseline was observed the comparison is
``unknown`` and the rule still fires, with the missing baseline named in the
counterevidence.

Candidates are advisory. Every candidate carries the evidence that produced
it *and* the counterevidence that could explain it away, so a reviewer can
judge instead of trusting the tool.

A cell is only *evidence* when the answer is unambiguous. A 2xx with a masked
body (empty, ``{}``, ``[]``, ``null``) is silent enforcement rather than
exposure, and a redirect is the request being pushed elsewhere - neither is
used to support a candidate in either direction. Cells that cannot be judged
at all (transport errors, 5xx, unusual statuses) are kept as
``inconclusive`` rather than dropped, so absence of a finding is never
confused with absence of a probe.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from halo.verify.http_sender import (
    DEFAULT_TIMEOUT_S,
    HttpRequest,
    HttpResponse,
    clamp_timeout,
)


if TYPE_CHECKING:
    from agents.sandbox.session import BaseSandboxSession


IMPLICIT_ANON_ID = "anon"
BODY_SAMPLE_CHARS = 300
MAX_MATRIX_CELLS = 200

#: Ceiling on the sandbox runner process, whatever the plan size.
MAX_SANDBOX_EXEC_S = 600.0

#: Bound on the structural walk, so an adversarial body cannot turn the
#: fingerprint into an expensive traversal.
MAX_STRUCTURE_NODES = 200
MAX_STRUCTURE_DEPTH = 12

#: The answer carried content: the caller reached the object.
ALLOWED = "allowed"
#: The answer refused the caller (401 / 403 / 404).
DENIED = "denied"
#: A 2xx whose body carries no object data: silent enforcement, not exposure.
EMPTY = "empty"
#: A 3xx: the request was redirected instead of served.
REDIRECT = "redirect"
#: Anything else (transport error, 5xx, unusual status): unjudgeable.
INCONCLUSIVE = "inconclusive"

#: Verdicts that may act as evidence. ``EMPTY`` and ``REDIRECT`` never do.
EVIDENCE_VERDICTS = frozenset({ALLOWED, DENIED})

DENIED_STATUSES = frozenset({401, 403, 404})
READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: Known roles and their privilege rank. A role missing from this table
#: cannot be compared to another role, so vertical checks skip it.
ROLE_RANK: dict[str, int] = {
    "anon": 0,
    "user": 1,
    "customer": 1,
    "staff": 2,
    "admin": 3,
    "owner": 4,
}

#: Payloads that carry no object data.
EMPTY_PAYLOADS = frozenset({"", "{}", "[]", "null"})

#: How a probe's body relates to the baseline it is compared against.
CONTENT_IDENTICAL = "identical"
CONTENT_SAME_STRUCTURE = "same_structure"
CONTENT_DIFFERENT = "different"
CONTENT_UNKNOWN = "unknown"

#: Content comparisons that support an access differential. ``DIFFERENT``
#: means the probe was served a different document, so it cannot show that
#: the probe reached the baseline's object.
SUPPORTING_CONTENT = frozenset({CONTENT_IDENTICAL, CONTENT_SAME_STRUCTURE})


@dataclass(frozen=True)
class MatrixPersona:
    """One principal the matrix probes with."""

    id: str
    role: str = "user"
    token: str = ""
    headers: tuple[tuple[str, str], ...] = ()

    @property
    def is_anon(self) -> bool:
        return self.id == IMPLICIT_ANON_ID


@dataclass(frozen=True)
class MatrixOperation:
    """One operation (method + path) probed by every persona."""

    id: str
    method: str = "GET"
    path: str = ""
    headers: tuple[tuple[str, str], ...] = ()
    body: str = ""
    owner: str = ""
    allowed_roles: tuple[str, ...] = ()
    public: bool = False
    shared: bool = False


@dataclass(frozen=True)
class MatrixCell:
    """The observation for one (operation, persona) pair."""

    operation_id: str
    persona_id: str
    status: int | None
    body_len: int = 0
    body_sample: str = ""
    error: str | None = None
    #: Digest of the whole observed body, truncated for the response. Two
    #: probes that got the same document share it.
    content_digest: str = ""
    #: Canonical JSON shape of the body (``path:type`` lines), or ``""`` when
    #: the body is not JSON and therefore has no comparable structure.
    content_shape: str = ""


@dataclass(frozen=True)
class WithheldCandidate:
    """A rule that matched on status but was rejected by the content diff."""

    rule: str
    operation_id: str
    persona_id: str
    reason: str


@dataclass(frozen=True)
class MatrixCandidate:
    """An advisory differential finding with evidence and counterevidence."""

    rule: str
    operation_id: str
    persona_id: str
    summary: str
    evidence: tuple[str, ...] = ()
    counterevidence: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


def substitute_token(template: str, token: str) -> str:
    """Replace every ``{token}`` placeholder in ``template`` with ``token``."""
    if not template:
        return ""
    return template.replace("{token}", token)


def role_rank(role: str) -> int | None:
    """Privilege rank of ``role``, or ``None`` when the role is not ranked."""
    return ROLE_RANK.get((role or "").strip().lower())


def _is_empty_payload(body: str) -> bool:
    return body.strip().lower() in EMPTY_PAYLOADS


def evidence_verdict(verdict: str) -> str | None:
    """The verdict when it may act as evidence, else ``None``.

    ``EMPTY`` and ``REDIRECT`` are deliberately not evidence: a masked body
    and a redirect both mean "no object was served here", and treating them
    as either an allow or a deny would manufacture findings.
    """
    return verdict if verdict in EVIDENCE_VERDICTS else None


_WHITESPACE = re.compile(r"\s+")


def _canonical_text(text: str) -> str:
    """A value-level canonical form of ``text``.

    JSON is re-serialized with sorted keys and no insignificant whitespace, so
    key order and pretty-printing cannot make the same document look different.
    Anything else is whitespace-folded text.
    """
    if text[:1] in "{[":
        try:
            return json.dumps(
                json.loads(text),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except (ValueError, RecursionError):
            pass
    return _WHITESPACE.sub(" ", text).strip()


def body_digest(body: str) -> str:
    """A stable fingerprint of ``body`` with insignificant whitespace folded.

    Whitespace outside a string literal is meaningless to a JSON client, so
    two probes that received the same document must not differ only because
    one was pretty-printed. A blank body has no digest.
    """
    canonical = _canonical_text((body or "").strip())
    if not canonical:
        return ""
    return hashlib.sha256(canonical.encode("utf-8", "replace")).hexdigest()[:16]


def _shape_of(value: Any, prefix: str, depth: int, out: set[str]) -> None:
    """Collect ``path:type`` lines for a decoded JSON value."""
    if len(out) >= MAX_STRUCTURE_NODES or depth > MAX_STRUCTURE_DEPTH:
        return
    if isinstance(value, dict):
        if not value:
            out.add(f"{prefix}:object")
        for key in sorted(value, key=str):
            _shape_of(value[key], f"{prefix}.{key}" if prefix else str(key), depth + 1, out)
        return
    if isinstance(value, list):
        if not value:
            out.add(f"{prefix}:array")
        for item in value:
            _shape_of(item, f"{prefix}[]", depth + 1, out)
        return
    if isinstance(value, bool):
        kind = "bool"
    elif isinstance(value, (int, float)):
        kind = "number"
    elif value is None:
        kind = "null"
    else:
        kind = "string"
    out.add(f"{prefix}:{kind}" if prefix else kind)


def body_shape(body: str) -> str:
    """A canonical description of a JSON body's structure, or ``""``.

    The shape ignores values on purpose: a per-viewer field (a name, a
    timestamp) must not hide that two callers received the same document
    layout. Non-JSON bodies return ``""`` because text has no structure worth
    comparing - only its digest is meaningful.
    """
    text = (body or "").strip()
    if not text or text[0] not in "{[":
        return ""
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return ""
    out: set[str] = set()
    _shape_of(value, "", 0, out)
    return "\n".join(sorted(out))


def compare_content(observed: MatrixCell | None, baseline: MatrixCell | None) -> str:
    """How ``observed``'s body relates to ``baseline``'s body."""
    if observed is None or baseline is None:
        return CONTENT_UNKNOWN
    if not observed.content_digest or not baseline.content_digest:
        return CONTENT_UNKNOWN
    if observed.content_digest == baseline.content_digest:
        return CONTENT_IDENTICAL
    if observed.content_shape and observed.content_shape == baseline.content_shape:
        return CONTENT_SAME_STRUCTURE
    return CONTENT_DIFFERENT


def _content_sentence(subject: str, match: str, baseline: str) -> str:
    """Render one content-comparison line for a candidate's evidence."""
    if match == CONTENT_IDENTICAL:
        return f"{subject} body is identical to {baseline}"
    if match == CONTENT_SAME_STRUCTURE:
        return f"{subject} body has the same structure as {baseline} but different values"
    if match == CONTENT_DIFFERENT:
        return f"{subject} body differs from {baseline}"
    return f"{subject} body could not be compared with {baseline} (no baseline body)"


def verdict_for_cell(cell: MatrixCell) -> str:
    """Classify one cell: allowed / denied / empty / redirect / inconclusive."""
    if cell.error or cell.status is None:
        return INCONCLUSIVE
    status = cell.status
    if 200 <= status < 300:
        return EMPTY if _is_empty_payload(cell.body_sample) else ALLOWED
    if 300 <= status < 400:
        return REDIRECT
    if status in DENIED_STATUSES:
        return DENIED
    return INCONCLUSIVE


def persona_from_dict(raw: Mapping[str, Any]) -> MatrixPersona:
    """Build a persona from a tool-supplied mapping."""
    persona_id = str(raw.get("id") or "").strip()
    if not persona_id:
        raise ValueError("persona needs an id")
    return MatrixPersona(
        id=persona_id,
        role=str(raw.get("role") or "user"),
        token=str(raw.get("token") or ""),
        headers=_headers_from(raw.get("headers")),
    )


def operation_from_dict(raw: Mapping[str, Any]) -> MatrixOperation:
    """Build an operation from a tool-supplied mapping."""
    operation_id = str(raw.get("id") or "").strip()
    if not operation_id:
        raise ValueError("operation needs an id")
    path = str(raw.get("path") or "").strip()
    if not path:
        raise ValueError(f"operation {operation_id} needs a path")
    roles = raw.get("allowed_roles") or ()
    if isinstance(roles, str):
        roles = (roles,)
    return MatrixOperation(
        id=operation_id,
        method=str(raw.get("method") or "GET").strip().upper(),
        path=path,
        headers=_headers_from(raw.get("headers")),
        body=str(raw.get("body") or ""),
        owner=str(raw.get("owner") or "").strip(),
        allowed_roles=tuple(str(role) for role in roles),
        public=bool(raw.get("public")),
        shared=bool(raw.get("shared")),
    )


def _headers_from(raw: Any) -> tuple[tuple[str, str], ...]:
    if not raw:
        return ()
    if isinstance(raw, Mapping):
        return tuple((str(k), str(v)) for k, v in raw.items())
    return tuple((str(k), str(v)) for k, v in raw)


def ensure_anon_persona(personas: Sequence[MatrixPersona]) -> tuple[MatrixPersona, ...]:
    """Add the implicit anonymous persona when the caller did not supply it."""
    if any(persona.id == IMPLICIT_ANON_ID for persona in personas):
        return tuple(personas)
    return (*personas, MatrixPersona(id=IMPLICIT_ANON_ID, role="anon"))


def build_requests(
    operations: Sequence[MatrixOperation],
    personas: Sequence[MatrixPersona],
    base_url: str,
) -> list[tuple[str, str, HttpRequest]]:
    """Plan one request per (operation, persona) pair, in matrix order.

    Operation-major order keeps every persona's probe of the same operation
    adjacent, which is what a reviewer reads side by side.
    """
    base = base_url.rstrip("/")
    planned: list[tuple[str, str, HttpRequest]] = []
    for operation in operations:
        for persona in personas:
            url = operation.path
            if not url.startswith("http"):
                url = f"{base}{url}"
            headers: dict[str, str] = {}
            for name, value in operation.headers:
                headers[name] = substitute_token(value, persona.token)
            for name, value in persona.headers:
                headers[name] = substitute_token(value, persona.token)
            if persona.token and not any(
                name.strip().lower() == "authorization" for name in headers
            ):
                headers["Authorization"] = persona.token
            headers = {name: value for name, value in headers.items() if value != ""}
            planned.append(
                (
                    operation.id,
                    persona.id,
                    HttpRequest(
                        method=operation.method,
                        url=substitute_token(url, persona.token),
                        headers=tuple(headers.items()),
                        body=substitute_token(operation.body, persona.token),
                    ),
                )
            )
    return planned


def cells_from_responses(
    plan: Sequence[tuple[str, str, HttpRequest]],
    responses: Sequence[HttpResponse],
) -> list[MatrixCell]:
    """Zip planned requests with their responses into cells."""
    cells: list[MatrixCell] = []
    for index, (operation_id, persona_id, _request) in enumerate(plan):
        response = responses[index] if index < len(responses) else None
        if response is None:
            cells.append(
                MatrixCell(
                    operation_id=operation_id,
                    persona_id=persona_id,
                    status=None,
                    error="no response",
                )
            )
            continue
        body = response.body or ""
        cells.append(
            MatrixCell(
                operation_id=operation_id,
                persona_id=persona_id,
                status=response.status,
                body_len=len(body),
                body_sample=body[:BODY_SAMPLE_CHARS],
                error=response.error,
                content_digest=body_digest(body),
                content_shape=body_shape(body),
            )
        )
    return cells


def _describe(cell: MatrixCell | None) -> str:
    if cell is None:
        return "no observation"
    if cell.error:
        return f"error ({cell.error})"
    return f"HTTP {cell.status} ({cell.body_len} bytes)"


def compute_differentials(
    operations: Sequence[MatrixOperation],
    personas: Sequence[MatrixPersona],
    cells: Sequence[MatrixCell],
) -> tuple[list[MatrixCandidate], list[WithheldCandidate]]:
    """Apply the differential rules.

    Returns the advisory candidates plus the status-only matches that the
    content comparison rejected, so a withheld differential stays visible
    instead of looking like a probe that never fired.
    """
    cell_index = {(cell.operation_id, cell.persona_id): cell for cell in cells}
    # Only unambiguous verdicts may act as evidence; EMPTY and REDIRECT are
    # filtered out here so no rule can accidentally treat them as access.
    verdicts = {
        key: evidence_verdict(verdict_for_cell(cell)) or "" for key, cell in cell_index.items()
    }

    candidates: list[MatrixCandidate] = []
    withheld: list[WithheldCandidate] = []
    for rule in (
        _anonymous_access_candidates,
        _horizontal_idor_candidates,
        _vertical_escalation_candidates,
        _method_asymmetry_candidates,
    ):
        found, rejected = rule(operations, personas, cell_index, verdicts)
        candidates.extend(found)
        withheld.extend(rejected)
    return candidates, withheld


def compute_candidates(
    operations: Sequence[MatrixOperation],
    personas: Sequence[MatrixPersona],
    cells: Sequence[MatrixCell],
) -> list[MatrixCandidate]:
    """Apply the differential rules and return advisory candidates."""
    candidates, _withheld = compute_differentials(operations, personas, cells)
    return candidates


def _authorized_baseline(
    operation: MatrixOperation,
    personas: Sequence[MatrixPersona],
    cell_index: Mapping[tuple[str, str], MatrixCell],
    verdicts: Mapping[tuple[str, str], str],
    *,
    exclude: str,
    predicate: Any,
) -> tuple[MatrixPersona | None, MatrixCell | None]:
    """The first allowed caller matching ``predicate``, and its cell."""
    for persona in personas:
        if persona.id == exclude or not predicate(persona):
            continue
        if verdicts.get((operation.id, persona.id)) == ALLOWED:
            return persona, cell_index.get((operation.id, persona.id))
    return None, None


def _anonymous_access_candidates(
    operations: Sequence[MatrixOperation],
    personas: Sequence[MatrixPersona],
    cell_index: Mapping[tuple[str, str], MatrixCell],
    verdicts: Mapping[tuple[str, str], str],
) -> tuple[list[MatrixCandidate], list[WithheldCandidate]]:
    candidates: list[MatrixCandidate] = []
    withheld: list[WithheldCandidate] = []
    anon_ids = [persona.id for persona in personas if persona.is_anon]
    for operation in operations:
        if operation.public:
            continue
        protected = bool(operation.owner) or bool(operation.allowed_roles)
        peer_denied = False
        evidence: list[str] = []
        for anon_id in anon_ids:
            if verdicts.get((operation.id, anon_id)) != ALLOWED:
                continue
            cell = cell_index[(operation.id, anon_id)]
            evidence.append(f"anonymous probe answered with content: {_describe(cell)}")
            for persona in personas:
                if persona.is_anon:
                    continue
                if verdicts.get((operation.id, persona.id)) == DENIED:
                    protected = True
                    peer_denied = True
                    evidence.append(
                        f"authenticated peer {persona.id} was denied the same operation"
                    )
        if not evidence or not protected:
            continue

        anon_cell = cell_index.get((operation.id, IMPLICIT_ANON_ID))
        baseline_persona, baseline_cell = _authorized_baseline(
            operation,
            personas,
            cell_index,
            verdicts,
            exclude=IMPLICIT_ANON_ID,
            predicate=lambda persona: not persona.is_anon,
        )
        match = compare_content(anon_cell, baseline_cell)
        if match == CONTENT_DIFFERENT:
            withheld.append(
                WithheldCandidate(
                    rule="anonymous_access",
                    operation_id=operation.id,
                    persona_id=IMPLICIT_ANON_ID,
                    reason=(
                        "the anonymous body differs from the authenticated caller's "
                        "document, so it is not evidence of exposure (a 200 login or "
                        "error page looks like this)"
                    ),
                )
            )
            continue

        counterevidence = [
            "the operation may be intentionally public; mark it public to suppress this rule",
        ]
        if baseline_persona is not None:
            evidence.append(_content_sentence("anonymous", match, f"{baseline_persona.id}'s view"))
        else:
            counterevidence.append(
                "no authenticated caller was observed to be allowed, so this rests on "
                "status alone; confirm a protected caller can read the same document"
            )
        if not peer_denied:
            counterevidence.append(
                "no authenticated peer was observed to be denied, so the endpoint may "
                "simply be unauthenticated by design"
            )
        candidates.append(
            MatrixCandidate(
                rule="anonymous_access",
                operation_id=operation.id,
                persona_id=IMPLICIT_ANON_ID,
                summary=(
                    f"{operation.method} {operation.path} served content to an unauthenticated "
                    "caller although the operation is expected to be protected"
                ),
                evidence=tuple(evidence),
                counterevidence=tuple(counterevidence),
                details={
                    "method": operation.method,
                    "path": operation.path,
                    "content_match": match,
                    "baseline_persona": baseline_persona.id if baseline_persona else None,
                },
            )
        )
    return candidates, withheld


def _horizontal_idor_candidates(
    operations: Sequence[MatrixOperation],
    personas: Sequence[MatrixPersona],
    cell_index: Mapping[tuple[str, str], MatrixCell],
    verdicts: Mapping[tuple[str, str], str],
) -> tuple[list[MatrixCandidate], list[WithheldCandidate]]:
    candidates: list[MatrixCandidate] = []
    withheld: list[WithheldCandidate] = []
    for operation in operations:
        if not operation.owner or operation.shared:
            continue
        owner = next((p for p in personas if p.id == operation.owner), None)
        if owner is None:
            continue
        for persona in personas:
            if persona.is_anon or persona.id == owner.id:
                continue
            # Same role, different principal: the owner's object read by a peer.
            if persona.role != owner.role:
                continue
            if verdicts.get((operation.id, persona.id)) != ALLOWED:
                continue
            peer_cell = cell_index[(operation.id, persona.id)]
            owner_cell = cell_index.get((operation.id, owner.id))
            owner_allowed = verdicts.get((operation.id, owner.id)) == ALLOWED
            match = compare_content(peer_cell, owner_cell)
            if owner_allowed and match == CONTENT_DIFFERENT:
                withheld.append(
                    WithheldCandidate(
                        rule="horizontal_idor",
                        operation_id=operation.id,
                        persona_id=persona.id,
                        reason=(
                            f"the peer's body differs from the owner {owner.id}'s body, so "
                            "the peer was served a different object; confirm the object id "
                            "before treating this as IDOR"
                        ),
                    )
                )
                continue

            evidence = [
                f"peer {persona.id} (role {persona.role}) read a same-role object: "
                f"{_describe(peer_cell)}"
            ]
            if owner_cell is not None:
                evidence.append(f"owner {owner.id} reads the same object: {_describe(owner_cell)}")
                evidence.append(
                    _content_sentence(f"peer {persona.id}", match, f"owner {owner.id}'s view")
                )
            else:
                evidence.append(
                    "no owner observation was recorded, so the content comparison is unknown"
                )
            candidates.append(
                MatrixCandidate(
                    rule="horizontal_idor",
                    operation_id=operation.id,
                    persona_id=persona.id,
                    summary=(
                        f"{persona.id} (same role as owner {owner.id}) was served "
                        f"{operation.method} {operation.path}, which belongs to {owner.id}"
                    ),
                    evidence=tuple(evidence),
                    counterevidence=(
                        "the resource may be intentionally shared between peers; "
                        "mark the operation shared to suppress this rule",
                    ),
                    details={
                        "method": operation.method,
                        "path": operation.path,
                        "owner": owner.id,
                        "owner_verdict": verdicts.get((operation.id, owner.id)),
                        "content_match": match,
                    },
                )
            )
    return candidates, withheld


def _vertical_escalation_candidates(
    operations: Sequence[MatrixOperation],
    personas: Sequence[MatrixPersona],
    cell_index: Mapping[tuple[str, str], MatrixCell],
    verdicts: Mapping[tuple[str, str], str],
) -> tuple[list[MatrixCandidate], list[WithheldCandidate]]:
    candidates: list[MatrixCandidate] = []
    withheld: list[WithheldCandidate] = []
    for operation in operations:
        if not operation.allowed_roles:
            continue
        ranks = [rank for rank in (role_rank(role) for role in operation.allowed_roles) if rank]
        if not ranks:
            continue
        required = min(ranks)
        for persona in personas:
            if persona.is_anon:
                continue  # covered by the anonymous-access rule
            rank = role_rank(persona.role)
            if rank is None or rank >= required:
                continue
            if verdicts.get((operation.id, persona.id)) != ALLOWED:
                continue
            cell = cell_index[(operation.id, persona.id)]
            baseline_persona, baseline_cell = _authorized_baseline(
                operation,
                personas,
                cell_index,
                verdicts,
                exclude=persona.id,
                predicate=lambda other, required=required: (role_rank(other.role) or 0) >= required,
            )
            match = compare_content(cell, baseline_cell)
            if baseline_persona is not None and match == CONTENT_DIFFERENT:
                withheld.append(
                    WithheldCandidate(
                        rule="vertical_escalation",
                        operation_id=operation.id,
                        persona_id=persona.id,
                        reason=(
                            f"the body differs from the authorized caller "
                            f"{baseline_persona.id}'s body, so the lower role was not "
                            "served the restricted document"
                        ),
                    )
                )
                continue

            evidence = [_describe(cell)]
            counterevidence: tuple[str, ...]
            if baseline_persona is not None:
                evidence.append(
                    _content_sentence(
                        f"{persona.id}",
                        match,
                        f"authorized caller {baseline_persona.id}'s view",
                    )
                )
                counterevidence = (
                    "the role names may not map to the target's real privilege tiers; "
                    "confirm the intended audience before treating this as escalation",
                )
            else:
                counterevidence = (
                    "no authorized caller was observed to be allowed, so this rests on "
                    "status alone; confirm which role is meant to reach this operation",
                    "the role names may not map to the target's real privilege tiers; "
                    "confirm the intended audience before treating this as escalation",
                )
            candidates.append(
                MatrixCandidate(
                    rule="vertical_escalation",
                    operation_id=operation.id,
                    persona_id=persona.id,
                    summary=(
                        f"{persona.id} (role {persona.role}) reached {operation.method} "
                        f"{operation.path}, restricted to roles "
                        f"{', '.join(operation.allowed_roles)}"
                    ),
                    evidence=tuple(evidence),
                    counterevidence=counterevidence,
                    details={
                        "method": operation.method,
                        "path": operation.path,
                        "allowed_roles": list(operation.allowed_roles),
                        "content_match": match,
                        "baseline_persona": baseline_persona.id if baseline_persona else None,
                    },
                )
            )
    return candidates, withheld


def _method_asymmetry_candidates(
    operations: Sequence[MatrixOperation],
    personas: Sequence[MatrixPersona],
    cell_index: Mapping[tuple[str, str], MatrixCell],
    verdicts: Mapping[tuple[str, str], str],
) -> tuple[list[MatrixCandidate], list[WithheldCandidate]]:
    candidates: list[MatrixCandidate] = []
    withheld: list[WithheldCandidate] = []
    grouped: dict[str, list[MatrixOperation]] = {}
    for operation in operations:
        grouped.setdefault(operation.path, []).append(operation)

    for path, group in grouped.items():
        reads = [op for op in group if op.method in READ_METHODS]
        writes = [op for op in group if op.method in WRITE_METHODS]
        if not reads or not writes:
            continue
        for persona in personas:
            denied_reads = [op for op in reads if verdicts.get((op.id, persona.id)) == DENIED]
            allowed_writes = [op for op in writes if verdicts.get((op.id, persona.id)) == ALLOWED]
            if not denied_reads or not allowed_writes:
                continue
            read_cell = cell_index.get((denied_reads[0].id, persona.id))
            write_cell = cell_index[(allowed_writes[0].id, persona.id)]
            evidence = [
                f"{persona.id} was denied the read on {path}: {_describe(read_cell)}",
                f"{persona.id} was allowed the write on {path}: {_describe(write_cell)}",
            ]
            if write_cell.content_digest:
                evidence.append(
                    f"the write response carried content (digest {write_cell.content_digest})"
                )
            candidates.append(
                MatrixCandidate(
                    rule="method_asymmetry",
                    operation_id=allowed_writes[0].id,
                    persona_id=persona.id,
                    summary=(
                        f"{persona.id} was denied {denied_reads[0].method} {path} but allowed "
                        f"{allowed_writes[0].method} {path}"
                    ),
                    evidence=tuple(evidence),
                    counterevidence=(
                        "the write path may intentionally have different authorization "
                        "than the read path; confirm the endpoint's intended policy",
                    ),
                    details={
                        "path": path,
                        "read_operation": denied_reads[0].id,
                        "write_operation": allowed_writes[0].id,
                        "read_method": denied_reads[0].method,
                        "write_method": allowed_writes[0].method,
                        "write_content_digest": write_cell.content_digest,
                    },
                )
            )
    return candidates, withheld


def matrix_to_dict(
    operations: Sequence[MatrixOperation],
    personas: Sequence[MatrixPersona],
    cells: Sequence[MatrixCell],
    candidates: Sequence[MatrixCandidate],
    withheld: Sequence[WithheldCandidate] = (),
) -> dict[str, Any]:
    """Serialize the whole matrix (cells included) for the tool response."""
    verdicts = [verdict_for_cell(cell) for cell in cells]
    cell_rows = [
        {
            "operation_id": cell.operation_id,
            "persona_id": cell.persona_id,
            "status": cell.status,
            "verdict": verdict,
            "body_len": cell.body_len,
            "body_sample": cell.body_sample,
            "content_digest": cell.content_digest,
            "content_structure": sorted(cell.content_shape.splitlines()),
            "error": cell.error,
        }
        for cell, verdict in zip(cells, verdicts, strict=True)
    ]
    return {
        "operations": [
            {
                "id": operation.id,
                "method": operation.method,
                "path": operation.path,
                "owner": operation.owner,
                "allowed_roles": list(operation.allowed_roles),
                "public": operation.public,
                "shared": operation.shared,
            }
            for operation in operations
        ],
        "personas": [
            {"id": persona.id, "role": persona.role, "anonymous": persona.is_anon}
            for persona in personas
        ],
        "cells": cell_rows,
        "inconclusive_cells": [row for row in cell_rows if row["verdict"] == INCONCLUSIVE],
        "candidates": [
            {
                "rule": candidate.rule,
                "operation_id": candidate.operation_id,
                "persona_id": candidate.persona_id,
                "summary": candidate.summary,
                "evidence": list(candidate.evidence),
                "counterevidence": list(candidate.counterevidence),
                "details": candidate.details,
                "advisory": True,
            }
            for candidate in candidates
        ],
        # Status-only matches the content diff rejected. They stay visible so
        # "no candidate" is never mistaken for "no such probe ran".
        "withheld": [
            {
                "rule": item.rule,
                "operation_id": item.operation_id,
                "persona_id": item.persona_id,
                "reason": item.reason,
            }
            for item in withheld
        ],
        "counts": {
            "operations": len(operations),
            "personas": len(personas),
            "cells": len(cells),
            "allowed": sum(1 for v in verdicts if v == ALLOWED),
            "denied": sum(1 for v in verdicts if v == DENIED),
            "empty": sum(1 for v in verdicts if v == EMPTY),
            "redirect": sum(1 for v in verdicts if v == REDIRECT),
            "inconclusive": sum(1 for v in verdicts if v == INCONCLUSIVE),
            "candidates": len(candidates),
            "withheld": len(withheld),
        },
    }


# ---------------------------------------------------------------------------
# In-sandbox transport
# ---------------------------------------------------------------------------

#: Standalone runner executed inside the sandbox. It reads a JSON plan of
#: requests from a file and writes a JSON list of results, so the host only
#: has to parse one document. Stdlib only: the runner must not depend on a
#: third-party HTTP client being present in the sandbox image.
MATRIX_RUNNER_SCRIPT = '''#!/usr/bin/env python3
"""Run an authorization-matrix plan and emit one JSON result per request."""

import json
import sys
import urllib.error
import urllib.request


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _send(item):
    body = item.get("body") or ""
    data = body.encode("utf-8") if body else None
    request = urllib.request.Request(item["url"], data=data, method=item.get("method", "GET"))
    for name, value in (item.get("headers") or {}).items():
        request.add_header(name, value)
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=item.get("timeout", 20)) as response:
            payload = response.read(4096).decode("utf-8", "replace")
            return {
                "status": response.status,
                "body": payload,
                "location": response.headers.get("Location"),
            }
    except urllib.error.HTTPError as exc:
        payload = exc.read(4096).decode("utf-8", "replace")
        return {
            "status": exc.code,
            "body": payload,
            "location": exc.headers.get("Location"),
        }
    except Exception as exc:
        return {"status": None, "body": "", "error": str(exc)}


def main():
    plan_path, out_path = sys.argv[1], sys.argv[2]
    with open(plan_path, encoding="utf-8") as handle:
        plan = json.load(handle)
    results = [_send(item) for item in plan]
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


#: Paths inside the sandbox container. Kept as POSIX strings because the
#: container is always Linux; the host may not be.
_RUNNER_SCRIPT_PATH = "/workspace/.halo-auth-matrix-runner.py"
_RUNNER_PLAN_PATH = "/workspace/.halo-auth-matrix-plan.json"
_RUNNER_OUT_PATH = "/workspace/.halo-auth-matrix-out.json"


async def run_matrix_in_session(
    session: BaseSandboxSession,
    plan: Sequence[tuple[str, str, HttpRequest]],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> list[HttpResponse]:
    """Replay ``plan`` inside the sandbox and return the observed responses.

    Targets that only answer from inside the container network need this
    transport; the probe order is preserved. Every request carries its own
    timeout and the runner process as a whole is bounded, so a stalled target
    cannot hold the sandbox open.
    """
    per_request_timeout = clamp_timeout(timeout)
    items = [
        {
            "method": request.method,
            "url": request.url,
            "headers": dict(request.headers),
            "body": request.body,
            "timeout": per_request_timeout,
        }
        for _operation_id, _persona_id, request in plan
    ]
    await session.write(
        Path(_RUNNER_PLAN_PATH),
        io.BytesIO(json.dumps(items, ensure_ascii=False).encode("utf-8")),
    )
    await session.write(
        Path(_RUNNER_SCRIPT_PATH),
        io.BytesIO(MATRIX_RUNNER_SCRIPT.encode("utf-8")),
    )
    result = await session.exec(
        "python3",
        _RUNNER_SCRIPT_PATH,
        _RUNNER_PLAN_PATH,
        _RUNNER_OUT_PATH,
        timeout=_exec_timeout(len(items), per_request_timeout),
    )
    if not result.ok():
        stderr = result.stderr.decode("utf-8", errors="replace")[:200]
        raise RuntimeError(f"auth matrix runner failed (exit {result.exit_code}): {stderr}")
    raw = await session.read(Path(_RUNNER_OUT_PATH))
    payload = json.loads(raw.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise TypeError("auth matrix runner returned a non-list payload")
    return [
        HttpResponse(
            status=item.get("status"),
            body=str(item.get("body") or ""),
            error=item.get("error"),
            location=item.get("location"),
        )
        for item in payload
        if isinstance(item, dict)
    ]


def _exec_timeout(count: int, per_request: float) -> int:
    """Bound the runner process: one timeout per probe plus startup slack."""
    budget = count * (per_request + 1.0) + 30.0
    return int(min(budget, MAX_SANDBOX_EXEC_S))
