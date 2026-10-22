"""LiteLLM model-name resolution for local cost estimates."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, cast


def cost_map_provider(model: str) -> str | None:
    """Return the provider segment that keeps ``model`` on its cost-map entry.

    ``litellm.completion_cost`` infers a provider with ``get_llm_provider`` when
    the caller supplies none, and that inference can pick a provider which
    authenticates while resolving its config: ``claude-sonnet-4.6`` is priced
    only under ``github_copilot/``, and resolving that provider starts a GitHub
    device login and polls it with ``time.sleep``. A cost estimate is a local
    read of the cost map, so the provider is handed to LiteLLM explicitly
    instead of being inferred.
    """
    if "/" in model:
        return model.split("/", 1)[0]
    try:
        import litellm

        entry = cast(
            "dict[str, dict[str, Any]]",
            getattr(litellm, "model_cost"),  # noqa: B009
        ).get(model)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(entry, dict):
        provider = entry.get("litellm_provider")
        if isinstance(provider, str) and provider:
            return provider
    return None


@lru_cache(maxsize=512)
def resolve_litellm_model(model: str) -> str | None:
    """Return a provider-qualified model name that LiteLLM can price."""
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
            if isinstance(provider, str) and provider:
                return f"{provider}/{normalized}"
        if "/" in normalized and isinstance(bare_entry, dict):
            return normalized

        names = [normalized]
        if "/" in normalized:
            names.append(normalized.rsplit("/", 1)[-1])
        for name in names:
            matches = sorted(key for key in model_cost if key.endswith(f"/{name}"))
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
