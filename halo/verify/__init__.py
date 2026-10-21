"""Deterministic verification primitives (replay harness, auth matrix)."""

from halo.verify.http_sender import (
    HttpRequest,
    HttpResponse,
    HttpxSender,
    send_sequential,
)
from halo.verify.matrix import (
    IMPLICIT_ANON_ID,
    MatrixCandidate,
    MatrixCell,
    MatrixOperation,
    MatrixPersona,
    build_requests,
    cells_from_responses,
    compute_candidates,
    ensure_anon_persona,
    evidence_verdict,
    matrix_to_dict,
    operation_from_dict,
    persona_from_dict,
    run_matrix_in_session,
    substitute_token,
    verdict_for_cell,
)
from halo.verify.replay import (
    MAX_RUNS,
    Oracle,
    ReplayResult,
    evaluate_oracle,
    result_to_dict,
    run_replays,
)


__all__ = [
    "IMPLICIT_ANON_ID",
    "MAX_RUNS",
    "HttpRequest",
    "HttpResponse",
    "HttpxSender",
    "MatrixCandidate",
    "MatrixCell",
    "MatrixOperation",
    "MatrixPersona",
    "Oracle",
    "ReplayResult",
    "build_requests",
    "cells_from_responses",
    "compute_candidates",
    "ensure_anon_persona",
    "evaluate_oracle",
    "evidence_verdict",
    "matrix_to_dict",
    "operation_from_dict",
    "persona_from_dict",
    "result_to_dict",
    "run_matrix_in_session",
    "run_replays",
    "send_sequential",
    "substitute_token",
    "verdict_for_cell",
]
