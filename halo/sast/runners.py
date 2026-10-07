"""Run SAST CLI tools (semgrep, gitleaks, trivy) and parse their output."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path  # noqa: TC003
from typing import Any, cast

from halo.sast.findings import Finding
from halo.sast.fingerprint import compute_fingerprint
from halo.sast.taxonomy import match_recipe


logger = logging.getLogger(__name__)

_TOOL_TIMEOUT_SECONDS = 600


def _slice_hash(message: str, file: str, line: int) -> str:
    payload = f"{file}:{line}:{message}"
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def parse_semgrep(payload: str) -> list[Finding]:
    try:
        data: dict[str, Any] = json.loads(payload)
    except json.JSONDecodeError:
        return []
    findings: list[Finding] = []
    results: list[dict[str, Any]] = data.get("results") or []
    for result in results:
        extra: dict[str, Any] = result.get("extra") or {}
        path = str(result.get("path") or "")
        start: dict[str, Any] = result.get("start") or {}
        line = int(start.get("line") or 0)
        check_id = str(result.get("check_id") or "unknown")
        message = str(extra.get("message") or "")
        severity = str(extra.get("severity") or "WARNING").upper()
        sink = path  # semgrep JSON has no resolved symbol; path is the anchor
        recipe = match_recipe("semgrep", check_id)
        taxonomy = recipe.id if recipe else "semgrep"
        fingerprint = compute_fingerprint(
            path, check_id, _slice_hash(message, path, line), taxonomy
        )
        findings.append(
            Finding(
                id=fingerprint,
                tool="semgrep",
                rule_id=check_id,
                file=path,
                line=line,
                sink_symbol=sink,
                message=message,
                taxonomy=taxonomy,
                severity=severity,
                ast_slice_hash=_slice_hash(message, path, line),
            )
        )
    return findings


def parse_gitleaks(payload: str) -> list[Finding]:
    try:
        data: Any = json.loads(payload)
    except json.JSONDecodeError:
        return []
    items: list[dict[str, Any]] = (
        cast("list[dict[str, Any]]", data) if isinstance(data, list) else []
    )
    findings: list[Finding] = []
    for item in items:
        path = str(item.get("File") or "")
        line = int(item.get("StartLine") or 0)
        rule_id = str(item.get("RuleID") or "unknown")
        message = str(item.get("Description") or "")
        recipe = match_recipe("gitleaks", rule_id)
        taxonomy = recipe.id if recipe else "secret"
        fingerprint = compute_fingerprint(path, rule_id, _slice_hash(message, path, line), taxonomy)
        findings.append(
            Finding(
                id=fingerprint,
                tool="gitleaks",
                rule_id=rule_id,
                file=path,
                line=line,
                sink_symbol=path,
                message=message,
                taxonomy=taxonomy,
                severity="HIGH",
                ast_slice_hash=_slice_hash(message, path, line),
            )
        )
    return findings


def parse_trivy(payload: str) -> list[Finding]:
    try:
        data: dict[str, Any] = json.loads(payload)
    except json.JSONDecodeError:
        return []
    findings: list[Finding] = []
    results: list[dict[str, Any]] = data.get("Results") or []
    for result in results:
        target = str(result.get("Target") or "")
        vulns: list[dict[str, Any]] = result.get("Vulnerabilities") or []
        for vuln in vulns:
            vuln_id = str(vuln.get("VulnerabilityID") or "unknown")
            title = str(vuln.get("Title") or "")
            severity = str(vuln.get("Severity") or "UNKNOWN").upper()
            fingerprint = compute_fingerprint(
                target, vuln_id, _slice_hash(title, target, 0), "trivy"
            )
            findings.append(
                Finding(
                    id=fingerprint,
                    tool="trivy",
                    rule_id=vuln_id,
                    file=target,
                    line=0,
                    sink_symbol=target,
                    message=title,
                    taxonomy="dependency_cve",
                    severity=severity,
                    ast_slice_hash=_slice_hash(title, target, 0),
                )
            )
    return findings


def _run(cmd: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_TOOL_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("SAST tool %s failed: %s", cmd[0], exc)
        return None
    if proc.returncode not in (0, 1) and not proc.stdout.strip():
        logger.warning("SAST tool %s exited %d: %s", cmd[0], proc.returncode, proc.stderr[:200])
        return None
    return proc.stdout


def scan_source(source_path: Path) -> list[Finding]:
    """Run available SAST tools over the source tree and normalize findings."""
    findings: list[Finding] = []
    if shutil.which("semgrep"):
        out = _run(
            ["semgrep", "--config", "p/default", "--json", "--quiet", "--error", "."],
            source_path,
        )
        if out:
            findings.extend(parse_semgrep(out))
    if shutil.which("gitleaks"):
        out = _run(
            ["gitleaks", "detect", "--no-git", "--report-format", "json", "--report-path", "-"],
            source_path,
        )
        if out:
            findings.extend(parse_gitleaks(out))
    if shutil.which("trivy"):
        out = _run(["trivy", "fs", "--format", "json", "--quiet", "."], source_path)
        if out:
            findings.extend(parse_trivy(out))
    return findings
