"""Curated SAST taxonomy: rule families with guards, severity, and mutants.

Recipes live in ``halo/sast/taxonomy/*.yaml``. Recipe ids are stable forever:
they are hashed into finding fingerprints, so renaming an id orphans stored
terminal states. Supersede with a new id rather than renaming.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from halo.sast.findings import Finding  # noqa: TC001


PACK_DIR = Path(__file__).parent / "taxonomy"

_REQUIRED_KEYS = frozenset(
    {"id", "version", "title", "match", "severity", "cwe", "guards", "notes", "mutants"}
)
_OPTIONAL_KEYS = frozenset({"evidence"})


@dataclass(frozen=True)
class Recipe:
    id: str
    version: int
    title: str
    tool: str
    check_ids: tuple[str, ...]
    severity: str
    cwe: str
    guards: tuple[str, ...]
    notes: str
    mutants: tuple[str, ...]
    evidence: tuple[str, ...] = ()
    _compiled: tuple[re.Pattern[str], ...] = field(default=(), repr=False, compare=False)

    def guard_hit(self, source_slice: str) -> str | None:
        """Return the first guard pattern matching the slice, else None."""
        for pattern, compiled in zip(self.guards, self._compiled, strict=True):
            if compiled.search(source_slice):
                return pattern
        return None


def _load_file(path: Path) -> Recipe:
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"unreadable recipe {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"recipe {path.name} must be a mapping")
    unknown = set(data) - _REQUIRED_KEYS - _OPTIONAL_KEYS
    if unknown:
        raise ValueError(f"recipe {path.name} has unknown keys: {sorted(unknown)}")
    missing = _REQUIRED_KEYS - set(data)
    if missing:
        raise ValueError(f"recipe {path.name} missing keys: {sorted(missing)}")
    match = data["match"]
    if not isinstance(match, dict) or "tool" not in match or "check_ids" not in match:
        raise ValueError(f"recipe {path.name}: match must have tool + check_ids")
    guards = [str(g) for g in data["guards"] or []]
    compiled = []
    for pattern in guards:
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(f"recipe {path.name}: bad guard regex {pattern!r}: {exc}") from exc
    return Recipe(
        id=str(data["id"]),
        version=int(data["version"]),
        title=str(data["title"]),
        tool=str(match["tool"]),
        check_ids=tuple(str(c) for c in match["check_ids"]),
        severity=str(data["severity"]).upper(),
        cwe=str(data["cwe"]),
        guards=tuple(guards),
        notes=str(data["notes"] or ""),
        mutants=tuple(str(m) for m in data["mutants"] or []),
        evidence=tuple(str(e) for e in data.get("evidence") or []),
        _compiled=tuple(compiled),
    )


@lru_cache(maxsize=8)
def load_recipes(pack_dir: str | None = None) -> tuple[Recipe, ...]:
    """Load and validate the recipe pack, sorted by id for determinism."""
    root = Path(pack_dir) if pack_dir else PACK_DIR
    files = sorted(root.glob("*.yaml"))
    if not files:
        raise ValueError(f"no recipes in {root}")
    recipes = [_load_file(f) for f in files]
    ids = [r.id for r in recipes]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate recipe ids: {dupes}")
    return tuple(sorted(recipes, key=lambda r: r.id))


def match_recipe(
    tool: str, rule_id: str, recipes: tuple[Recipe, ...] | None = None
) -> Recipe | None:
    """First recipe (by id order) whose tool matches and any check_id glob matches."""
    pack = recipes if recipes is not None else load_recipes()
    for recipe in pack:
        if recipe.tool != tool:
            continue
        if any(fnmatch.fnmatchcase(rule_id, pattern) for pattern in recipe.check_ids):
            return recipe
    return None


def recipe_for_finding(
    finding: Finding, recipes: tuple[Recipe, ...] | None = None
) -> Recipe | None:
    """Recipe by stored taxonomy id, else by tool + rule match (backward compatible)."""
    pack = recipes if recipes is not None else load_recipes()
    for recipe in pack:
        if recipe.id == finding.taxonomy:
            return recipe
    return match_recipe(finding.tool, finding.rule_id, pack)
