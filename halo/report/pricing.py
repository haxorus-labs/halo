"""LiteLLM model-name resolution for local cost estimates.

A cost estimate must never cost more than it saves, so this module resolves a
model name against LiteLLM's *local* cost map only. In particular it must never
hand a name to LiteLLM's provider machinery for a provider that authenticates
interactively: ``completion_cost`` resolves the provider, and for
``github_copilot/*`` that resolution starts a device-code OAuth login which
polls until a human approves it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, cast


#: Providers whose ``get_api_key()`` opens an interactive OAuth device flow.
#: Estimation runs on the event loop during an unattended scan, so a name for
#: one of these is not resolvable: the provider-reported cost (when there is
#: one) is the only safe source.
INTERACTIVE_AUTH_PROVIDERS = frozenset({"github_copilot", "chatgpt"})


def _provider_of(model: str) -> str:
    """The leading provider segment of ``model``, or ``""``."""
    return model.split("/", 1)[0].strip().lower() if "/" in model else ""


def needs_interactive_auth(model: str) -> bool:
    """True when pricing ``model`` would start an interactive login."""
    return _provider_of(model) in INTERACTIVE_AUTH_PROVIDERS


@lru_cache(maxsize=512)
def resolve_litellm_model(model: str) -> str | None:
    """Return a provider-qualified model name that LiteLLM can price.

    ``None`` means "no local price is known"; callers then fall back to the
    provider-reported cost instead of guessing. A name the caller already
    qualified with a known provider is returned unchanged (unless that
    provider authenticates interactively), and bare names are widened to a
    provider only when the price is unambiguous within one provider family.
    """
    try:
        import litellm

        normalized = model.strip()
        for prefix in ("litellm/", "any-llm/", "openai/"):
            if normalized.startswith(prefix):
                normalized = normalized.removeprefix(prefix)
                break
        if not normalized:
            return None

        model_cost = cast(
            "dict[str, dict[str, Any]]",
            getattr(litellm, "model_cost"),  # noqa: B009
        )
        bare_entry = model_cost.get(normalized)
        if "/" not in normalized and isinstance(bare_entry, dict):
            provider = bare_entry.get("litellm_provider")
            if isinstance(provider, str) and provider and not needs_interactive_auth(provider):
                return f"{provider}/{normalized}"

        # A suffix match may only stay inside the provider the caller named.
        # Matching across providers is how ``openrouter/anthropic/claude-...``
        # used to resolve to ``github_copilot/claude-...``: the first key by
        # sort order, at another vendor's price.
        providers = {
            str(getattr(entry, "value", entry)).lower()
            for entry in getattr(litellm, "provider_list", ())
        }
        family = _provider_of(normalized)
        family_known = bool(family) and (
            family in providers or any(str(key).startswith(f"{family}/") for key in model_cost)
        )
        if family_known:
            # The caller named the provider itself, so the name cannot drift
            # into another vendor's entry. Hand it over unchanged: LiteLLM may
            # still know a price the local map does not carry, and a name it
            # cannot price only raises inside ``completion_cost``, which every
            # caller already handles. Providers that authenticate are the one
            # exception, because pricing them is what used to open a login.
            return None if needs_interactive_auth(normalized) else normalized
        family = ""

        names = [normalized]
        if "/" in normalized:
            names.append(normalized.rsplit("/", 1)[-1])
        for name in names:
            matches = [
                key
                for key in sorted(str(entry) for entry in model_cost)
                if key.endswith(f"/{name}")
                and (not family or key.startswith(f"{family}/"))
                and not needs_interactive_auth(key)
            ]
            if not matches:
                continue
            prices = {
                (
                    model_cost[key].get("input_cost_per_token"),
                    model_cost[key].get("output_cost_per_token"),
                )
                for key in matches
                if isinstance(model_cost.get(key), dict)
            }
            if len(matches) == 1 or len(prices) == 1:
                return matches[0]
        return None  # noqa: TRY300
    except Exception:  # noqa: BLE001
        return None
