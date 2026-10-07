"""Deterministic verification primitives (replay harness)."""

from halo.verify.replay import (
    MAX_RUNS,
    Oracle,
    ReplayResult,
    evaluate_oracle,
    result_to_dict,
    run_replays,
)


__all__ = [
    "MAX_RUNS",
    "Oracle",
    "ReplayResult",
    "evaluate_oracle",
    "result_to_dict",
    "run_replays",
]
