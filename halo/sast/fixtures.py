"""Seeded-tenant fixtures: known personas per target class.

Fixtures live in ``halo/sast/fixtures/*.yaml``. Fixture ids are stable;
they are referenced from auth-map edges, so never rename one.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PACK_DIR = Path(__file__).parent / "fixtures"

_REQUIRED_KEYS = frozenset({"id", "version", "title", "match", "login", "personas"})
_OPTIONAL_KEYS = frozenset({"auth_signals", "evidence"})


@dataclass(frozen=True)
class Persona:
    id: str
    role: str
    email: str = ""
    password: str = ""


@dataclass(frozen=True)
class Fixture:
    id: str
    version: int
    title: str
    repo_urls: tuple[str, ...]
    path_markers: tuple[str, ...]
    login_route: str
    login_method: str
    login_body: dict[str, str]
    token_path: str
    success_markers: tuple[str, ...]
    personas: tuple[Persona, ...]
    denied_statuses: tuple[int, ...]
    notes: str = ""
    evidence: tuple[str, ...] = ()


def _load_file(path: Path) -> Fixture:
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"unreadable fixture {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"fixture {path.name} must be a mapping")
    unknown = set(data) - _REQUIRED_KEYS - _OPTIONAL_KEYS
    if unknown:
        raise ValueError(f"fixture {path.name} has unknown keys: {sorted(unknown)}")
    missing = _REQUIRED_KEYS - set(data)
    if missing:
        raise ValueError(f"fixture {path.name} missing keys: {sorted(missing)}")
    match = data["match"]
    if not isinstance(match, dict):
        raise TypeError(f"fixture {path.name}: match must be a mapping")
    login = data["login"]
    if not isinstance(login, dict):
        raise TypeError(f"fixture {path.name}: login must be a mapping")
    if "route" not in login:
        raise ValueError(f"fixture {path.name}: login must have a route")
    personas = []
    for entry in data["personas"] or []:
        if not isinstance(entry, dict) or "id" not in entry or "role" not in entry:
            raise ValueError(f"fixture {path.name}: personas need id + role")
        personas.append(
            Persona(
                id=str(entry["id"]),
                role=str(entry["role"]),
                email=str(entry.get("email") or ""),
                password=str(entry.get("password") or ""),
            )
        )
    signals = data.get("auth_signals") or {}
    if not isinstance(signals, dict):
        raise TypeError(f"fixture {path.name}: auth_signals must be a mapping")
    body = login.get("body") or {}
    if not isinstance(body, dict):
        raise TypeError(f"fixture {path.name}: login.body must be a mapping")
    return Fixture(
        id=str(data["id"]),
        version=int(data["version"]),
        title=str(data["title"]),
        repo_urls=tuple(str(u) for u in match.get("repo_urls") or []),
        path_markers=tuple(str(m) for m in match.get("path_markers") or []),
        login_route=str(login["route"]),
        login_method=str(login.get("method") or "POST").upper(),
        login_body={str(k): str(v) for k, v in body.items()},
        token_path=str(login.get("token_path") or ""),
        success_markers=tuple(str(s) for s in login.get("success_markers") or []),
        personas=tuple(personas),
        denied_statuses=tuple(int(s) for s in signals.get("denied_statuses") or (401, 403)),
        notes=str(signals.get("notes") or ""),
        evidence=tuple(str(e) for e in data.get("evidence") or []),
    )


@lru_cache(maxsize=8)
def load_fixtures(pack_dir: str | None = None) -> tuple[Fixture, ...]:
    """Load and validate the fixture pack, sorted by id for determinism."""
    root = Path(pack_dir) if pack_dir else PACK_DIR
    files = sorted(root.glob("*.yaml"))
    if not files:
        raise ValueError(f"no fixtures in {root}")
    fixtures = [_load_file(f) for f in files]
    ids = [f.id for f in fixtures]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate fixture ids: {dupes}")
    return tuple(sorted(fixtures, key=lambda f: f.id))


def match_fixture(
    repo_url: str = "",
    source_path: str = "",
    fixtures: tuple[Fixture, ...] | None = None,
) -> Fixture | None:
    """First fixture (by id order) matching repo URL or source path markers."""
    pack = fixtures if fixtures is not None else load_fixtures()
    lowered_url = repo_url.lower()
    lowered_path = source_path.lower()
    for fixture in pack:
        if lowered_url and any(
            fnmatch.fnmatchcase(lowered_url, pattern.lower()) for pattern in fixture.repo_urls
        ):
            return fixture
        if lowered_path and any(marker.lower() in lowered_path for marker in fixture.path_markers):
            return fixture
    return None


def login_body_for(fixture: Fixture, persona: Persona) -> dict[str, str]:
    """Render the login body template for a persona (anon yields {})."""
    if not persona.email:
        return {}
    return {
        key: value.format(email=persona.email, password=persona.password)
        for key, value in fixture.login_body.items()
    }
