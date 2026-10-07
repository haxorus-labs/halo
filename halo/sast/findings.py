"""SAST findings: schema, states, and a JSONL-backed store."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path  # noqa: TC003
from typing import Any


STATE_PROPOSED = "PROPOSED"
STATE_REFUTED = "REFUTED"
STATE_UNVERIFIED = "UNVERIFIED"
STATE_VERIFIED = "VERIFIED"
VALID_STATES = frozenset({STATE_PROPOSED, STATE_REFUTED, STATE_UNVERIFIED, STATE_VERIFIED})

# Terminal-ish states we do not re-probe without new evidence
_SETTLED_RETEST_EXEMPT = frozenset({STATE_REFUTED, STATE_VERIFIED})


@dataclass
class Finding:
    id: str
    tool: str
    rule_id: str
    file: str
    line: int
    sink_symbol: str
    message: str
    taxonomy: str
    severity: str
    state: str = STATE_PROPOSED
    reason: str = ""
    ast_slice_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class FindingStore:
    """Append-only JSONL store; current state is the last record per id."""

    path: Path
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def record(self, finding: Finding) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(finding.to_dict(), ensure_ascii=False) + "\n")

    def _all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def current(self) -> dict[str, Finding]:
        latest: dict[str, Finding] = {}
        for record in self._all():
            try:
                finding = Finding.from_dict(record)
            except TypeError:
                continue
            latest[finding.id] = finding
        return latest

    def get(self, finding_id: str) -> Finding | None:
        return self.current().get(finding_id)

    def is_settled(self, finding_id: str) -> bool:
        finding = self.get(finding_id)
        return finding is not None and finding.state in _SETTLED_RETEST_EXEMPT

    def set_state(self, finding_id: str, state: str, reason: str = "") -> Finding | None:
        if state not in VALID_STATES:
            raise ValueError(f"invalid state {state!r}; expected one of {sorted(VALID_STATES)}")
        finding = self.get(finding_id)
        if finding is None:
            return None
        finding.state = state
        finding.reason = reason
        self.record(finding)
        return finding
