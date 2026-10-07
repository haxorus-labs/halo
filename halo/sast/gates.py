"""Sanitizer check and LLM adversarial verifier for candidate findings."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agents.models.interface import ModelTracing

from halo.config import load_settings
from halo.config.models import HaloProvider, configure_sdk_api_route
from halo.core.inputs import make_model_settings
from halo.sast.findings import Finding  # noqa: TC001


logger = logging.getLogger(__name__)

_SATISFACTORY = re.compile(
    r"(sanitiz|escape|htmlspecialchars|shlex\.quote|parameteriz|prepare\(|whitelist|\bvalidate\(|is_valid|strip_tags)",
    re.IGNORECASE,
)

ADVERSARIAL_SYSTEM_PROMPT = """You are an adversarial security reviewer.
Given a candidate static-analysis
finding and the surrounding code, try hard to refute it. A finding should only be
upheld when the bad behavior is reachable with attacker-controlled input and no
adequate guard is present. Respond with ONLY a JSON object:
{"verdict": "CONFIRMED"|"REFUTED"|"UNVERIFIED", "reason": "..."}."""


def sanitizer_gate(source_slice: str) -> str:
    """Cheap syntactic check. Returns 'defended', 'undefended', or 'inconclusive'."""
    if not source_slice.strip():
        return "inconclusive"
    if _SATISFACTORY.search(source_slice):
        return "defended"
    return "undefended"


async def adversarial_verify(
    finding: Finding,
    source_slice: str,
    settings: Any,
) -> dict[str, str]:
    """One LLM call per finding; degrade gracefully when no model is configured."""
    cfg = settings or load_settings()
    model_name = (cfg.llm.model or "").strip()
    if not model_name:
        return {"verdict": "UNVERIFIED", "reason": "No LLM model configured"}

    configure_sdk_api_route(model_name, cfg)
    model = HaloProvider().get_model(model_name)

    user_msg = (
        f"Finding: {json.dumps(finding.to_dict(), ensure_ascii=False)}\n\n"
        f"Code slice:\n```\n{source_slice[:4000]}\n```\n\n"
        "Respond with ONLY the JSON object described in the system prompt."
    )
    try:
        response = await model.get_response(
            system_instructions=ADVERSARIAL_SYSTEM_PROMPT,
            input=user_msg,
            model_settings=make_model_settings(
                cfg.llm.reasoning_effort,
                model_name=model_name,
                force_required_tool_choice=False,
                request_timeout=cfg.llm.timeout,
                extra_headers=cfg.llm.extra_headers,
                has_tools=False,
            ),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
    except Exception:
        logger.exception("Adversarial verification failed for %s", finding.id)
        return {"verdict": "UNVERIFIED", "reason": "LLM call failed"}

    text = ""
    for item in getattr(response, "output", []) or []:
        for chunk in getattr(item, "content", []) or []:
            value = getattr(chunk, "text", None)
            if value:
                text += value
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {"verdict": "UNVERIFIED", "reason": "Unparseable LLM response"}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"verdict": "UNVERIFIED", "reason": "Unparseable LLM response"}
    verdict = str(parsed.get("verdict") or "UNVERIFIED").upper()
    if verdict not in ("CONFIRMED", "REFUTED", "UNVERIFIED"):
        verdict = "UNVERIFIED"
    return {"verdict": verdict, "reason": str(parsed.get("reason") or "")[:500]}
