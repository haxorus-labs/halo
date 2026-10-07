"""Resolve a source path given to SAST/KG tools to a host directory.

Agent tools execute on the host, but agents usually think in sandbox paths
(``/workspace/<repo>``). This maps sandbox-style paths onto the known host
source roots so ``sast_scan("/workspace/juice-shop")`` just works.
"""

from __future__ import annotations

from pathlib import Path


def resolve_source_root(source_path: str, default_roots: list[str]) -> Path | None:
    """Return a host directory for ``source_path`` or ``None`` when unknown.

    Rules, in order:
    1. An existing host directory is used as-is.
    2. An empty path falls back to the first existing default root.
    3. A ``/workspace/...`` sandbox path is mapped onto default roots by
       matching the repo directory name (e.g. ``/workspace/juice-shop`` →
       ``/tmp/halo_repos/<run>/juice-shop``). Deeper sandbox paths resolve
       to the matching subtree when it exists on the host, else the root.
    4. Otherwise the first existing default root is used as a last resort.
    """
    roots = [Path(r).expanduser() for r in default_roots if r]
    existing_roots = [r for r in roots if r.is_dir()]

    if source_path:
        candidate = Path(source_path).expanduser()
        if candidate.is_dir():
            return candidate
        mapped = _map_sandbox_path(candidate, existing_roots)
        if mapped is not None:
            return mapped

    if existing_roots:
        return existing_roots[0]
    return None


def _map_sandbox_path(candidate: Path, roots: list[Path]) -> Path | None:
    try:
        relative = candidate.resolve().relative_to("/workspace")
    except (OSError, RuntimeError, ValueError):
        return None
    parts = relative.parts
    if not parts:
        return roots[0] if roots else None
    for root in roots:
        resolved = root.resolve()
        if resolved.name not in parts:
            continue
        idx = parts.index(resolved.name)
        rest = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else None
        if rest is None:
            return root
        sub = resolved / rest
        return sub if sub.is_dir() else root
    return None
