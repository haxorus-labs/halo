from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import litellm
from agents.usage import Usage

from halo.report import state as report_state
from halo.report.pricing import needs_interactive_auth, resolve_litellm_model
from halo.report.usage import LLMUsageLedger


if TYPE_CHECKING:
    import pytest


def test_resolves_common_bare_model_names() -> None:
    resolve_litellm_model.cache_clear()
    assert resolve_litellm_model("deepseek-v4-flash") == "deepseek/deepseek-v4-flash"
    assert resolve_litellm_model("openai/deepseek-v4-flash") == "deepseek/deepseek-v4-flash"
    # LiteLLM prices grok-4.5 identically under xai/, perplexity/xai/ and
    # openrouter/x-ai/, so the resolver may return any of them; the contract
    # is a provider-qualified name LiteLLM can price.
    grok = resolve_litellm_model("grok-4.5")
    assert grok is not None
    assert grok.endswith("/grok-4.5")
    assert grok in litellm.model_cost
    # MiniMax-M3 is sold by several LiteLLM providers at different prices, so
    # the resolver must not guess from its bare name. A provider-qualified
    # model remains deterministic.
    assert resolve_litellm_model("minimax/MiniMax-M3") == "minimax/MiniMax-M3"


def test_resolver_returns_none_for_unresolvable_model() -> None:
    resolve_litellm_model.cache_clear()
    assert resolve_litellm_model("provider/not-a-real-model") is None


def test_ledger_uses_estimate_when_routed_provider_reports_no_cost() -> None:
    usage = Usage()
    usage.requests = 1
    usage.input_tokens = 1000
    usage.output_tokens = 200
    usage.total_tokens = 1200
    ledger = LLMUsageLedger()

    with patch("litellm.completion_cost", return_value=0.42):
        ledger.record(agent_id="a", usage=usage, model="openai/deepseek-v4-flash")

    assert ledger.total_cost == 0.42


def test_ledger_prefers_observed_cost_over_estimate() -> None:
    usage = Usage()
    usage.requests = 1
    usage.input_tokens = 1000
    usage.output_tokens = 200
    usage.total_tokens = 1200
    ledger = LLMUsageLedger()

    with patch("litellm.completion_cost", return_value=0.42):
        ledger.record(agent_id="a", usage=usage, model="openai/deepseek-v4-flash")
    ledger.record_observed_cost(0.17)

    assert ledger.total_cost == 0.17


def test_hydrated_estimate_continues_accumulating_new_estimates() -> None:
    usage = Usage()
    usage.requests = 1
    usage.input_tokens = 1000
    usage.output_tokens = 200
    usage.total_tokens = 1200
    ledger = LLMUsageLedger()
    ledger.hydrate({"cost": 0.42})

    with patch("litellm.completion_cost", return_value=0.17):
        ledger.record(agent_id="a", usage=usage, model="openai/deepseek-v4-flash")

    assert ledger.total_cost == 0.59


def test_zero_cost_disables_both_observed_and_estimated_costs() -> None:
    usage = Usage()
    usage.requests = 1
    usage.input_tokens = 1000
    usage.output_tokens = 200
    usage.total_tokens = 1200
    ledger = LLMUsageLedger()
    ledger.zero_cost = True

    with patch("litellm.completion_cost", return_value=0.42) as estimate:
        ledger.record(agent_id="a", usage=usage, model="deepseek-v4-flash")
        ledger.record_observed_cost(1.0)

    estimate.assert_not_called()
    assert ledger.total_cost == 0.0


def test_resolver_uses_provider_when_bare_entry_has_one() -> None:
    original = litellm.model_cost
    litellm.model_cost = {
        "example": {
            "litellm_provider": "example-provider",
            "input_cost_per_token": 1.0,
            "output_cost_per_token": 2.0,
        }
    }
    try:
        resolve_litellm_model.cache_clear()
        assert resolve_litellm_model("example") == "example-provider/example"
    finally:
        litellm.model_cost = original
        resolve_litellm_model.cache_clear()


def test_resolver_does_not_guess_between_differently_priced_providers() -> None:
    original = litellm.model_cost
    litellm.model_cost = {
        "provider-a/example": {
            "input_cost_per_token": 1.0,
            "output_cost_per_token": 2.0,
        },
        "provider-b/example": {
            "input_cost_per_token": 3.0,
            "output_cost_per_token": 4.0,
        },
    }
    try:
        resolve_litellm_model.cache_clear()
        assert resolve_litellm_model("example") is None
    finally:
        litellm.model_cost = original
        resolve_litellm_model.cache_clear()


def test_resolver_never_crosses_provider_families() -> None:
    """A suffix match must not borrow another vendor's entry (or its price).

    Regression: ``openrouter/anthropic/claude-sonnet-4.6`` used to resolve to
    ``github_copilot/claude-sonnet-4.6``, the first key by sort order, which
    sent ``completion_cost`` into an interactive OAuth login.
    """
    resolve_litellm_model.cache_clear()
    for model in (
        "openrouter/anthropic/claude-sonnet-4.6",
        "openrouter/qwen/qwen3-coder",
        "openrouter/z-ai/glm-5.3",
    ):
        resolved = resolve_litellm_model(model)
        assert resolved is None or resolved.split("/", 1)[0] == "openrouter", resolved


def test_resolver_keeps_a_known_provider_entry_and_never_borrows_a_sibling() -> None:
    """A provider-qualified name LiteLLM prices is returned unchanged."""
    original = litellm.model_cost
    litellm.model_cost = {
        "openrouter/anthropic/claude-sonnet-4.6": {
            "input_cost_per_token": 3e-06,
            "output_cost_per_token": 1.5e-05,
        },
        "github_copilot/claude-sonnet-4.6": {
            "input_cost_per_token": 3e-06,
            "output_cost_per_token": 1.5e-05,
        },
    }
    try:
        resolve_litellm_model.cache_clear()
        assert (
            resolve_litellm_model("openrouter/anthropic/claude-sonnet-4.6")
            == "openrouter/anthropic/claude-sonnet-4.6"
        )
        # Without an openrouter entry, the same-suffix copilot entry must not
        # be borrowed, even though LiteLLM prices it identically.
        del litellm.model_cost["openrouter/anthropic/claude-sonnet-4.6"]
        resolve_litellm_model.cache_clear()
        assert resolve_litellm_model("openrouter/anthropic/claude-sonnet-4.6") is None
    finally:
        litellm.model_cost = original
        resolve_litellm_model.cache_clear()


def test_resolver_refuses_providers_that_need_interactive_auth() -> None:
    """Pricing must never be able to open a device-code login."""
    resolve_litellm_model.cache_clear()
    assert needs_interactive_auth("github_copilot/claude-sonnet-4.6")
    assert needs_interactive_auth("chatgpt/gpt-5")
    assert not needs_interactive_auth("openrouter/anthropic/claude-sonnet-4.6")
    assert resolve_litellm_model("github_copilot/claude-sonnet-4.6") is None
    assert resolve_litellm_model("github_copilot/gpt-4o") is None


def test_cost_estimate_never_reaches_an_interactive_login(monkeypatch: pytest.MonkeyPatch) -> None:
    """The estimate path must stay inert for a provider that authenticates."""
    priced: list[str] = []

    def _record(**kwargs: object) -> float:
        priced.append(str(kwargs.get("model")))
        raise ValueError("no local price")

    monkeypatch.setattr("litellm.completion_cost", _record)
    resolve_litellm_model.cache_clear()
    payload = {"model": "github_copilot/claude-sonnet-4.6", "usage": {"prompt_tokens": 10}}
    assert report_state._estimate_response_cost({"model": payload["model"]}, payload) is None
    # A bare-name fallback may still be priced locally, but the provider that
    # opens a device-code login must never be handed to ``completion_cost``.
    assert not [model for model in priced if model.startswith("github_copilot/")]
