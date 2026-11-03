"""Paired joint projection on a shared sample mask and DCT columns."""

import numpy as np

from .score import _real_array, aggregate_networks, residual_operator

TOLERANCE_MULTIPLIERS = (0.1, 1.0, 10.0, 100.0)
IDENTITY_TOLERANCE = 1e-10


def normalize_mask(mask, nraw):
    """Validate a retained suffix mask and return its Boolean encoding.

    Numeric 0/1 and Boolean masks are equivalent. At least 18 samples must be
    retained, with no internal gaps or trailing excluded samples.
    """
    if isinstance(nraw, (bool, np.bool_)) or not isinstance(nraw, (int, np.integer)) or nraw < 18:
        raise ValueError("nraw must be an integer of at least 18")
    raw = np.asarray(mask)
    if (
        raw.shape != (nraw,)
        or not np.issubdtype(raw.dtype, np.number)
        and raw.dtype != np.dtype(bool)
    ):
        raise ValueError("One numeric or boolean indicator per source row required")
    if not np.isfinite(raw).all() or not np.all((raw == 0) | (raw == 1)):
        raise ValueError("Retained indicator must contain exact finite 0/1 values")
    selected = raw.astype(bool)
    indices = np.flatnonzero(selected)
    if len(indices) < 18 or not np.array_equal(indices, np.arange(indices[0], nraw)):
        raise ValueError("A nonempty retained suffix without internal holes is required")
    return selected


def paired_fields(parcels, motion24, A5, cosines, membership):
    """Jointly residualize 400 parcels, then average into seven networks.

    Inputs must already share the same retained rows: parcels (N, 400),
    motion24 (N, 24), A5 (N, 5), and saved cosines (N, K). A (N, 0) cosine
    matrix omits the shared DCT adjustment. Membership is an integer vector
    containing every label 0 through 6; outputs follow that label order.

    BASE_DCT removes intercept, linear trend, motion24 and cosines. A5_DCT
    adds all five anatomical nuisance columns in the same joint projection.
    Returns raw network fields and rank/energy diagnostics; scoring performs
    the subsequent channel centering and unit-RMS scaling.
    """
    x, m, a, c = [_real_array(v) for v in (parcels, motion24, A5, cosines)]
    if x.ndim != 2 or x.shape[1] != 400 or x.shape[0] < 18:
        raise ValueError("Finite N-by-400 input with N>=18 required")
    n = len(x)
    if m.shape != (n, 24) or a.shape != (n, 5) or c.ndim != 2 or c.shape[0] != n:
        raise ValueError("Common temporal support and exact M24/A5 widths required")
    if not all(np.isfinite(v).all() for v in (x, m, a, c)):
        raise ValueError("Nonfinite input")
    membership = np.asarray(membership)
    if membership.shape != (400,) or not np.issubdtype(membership.dtype, np.integer):
        raise ValueError("Integer 400-parcel network mapping required")
    if set(membership.tolist()) != set(range(7)):
        raise ValueError("Exactly seven nonempty networks required")
    common = np.column_stack([np.linspace(-1.0, 1.0, n, dtype=float), m, c])
    designs = {"BASE_DCT": common, "A5_DCT": np.column_stack([common, a])}
    operators = {}
    details = {}
    for name, design in designs.items():
        candidates = [
            residual_operator(
                design, operator_id=name, include_intercept=True, tolerance_multiplier=t
            )
            for t in TOLERANCE_MULTIPLIERS
        ]
        ranks = [op.rank for op in candidates]
        if len(set(ranks)) != 1:
            raise ValueError("Rank ambiguity across frozen tolerances: " + name)
        op = candidates[1]
        if n - op.rank <= 0:
            raise ValueError("No residual degrees of freedom: " + name)
        operators[name] = op
        details[name] = {
            "nuisance_rank": op.rank,
            "residual_df": n - op.rank,
            "ranks_by_tolerance_multiplier": ranks,
        }
    p, q = operators["BASE_DCT"].matrix(), operators["A5_DCT"].matrix()
    increment = operators["A5_DCT"].rank - operators["BASE_DCT"].rank
    if not 0 <= increment <= 5:
        raise ValueError("Invalid nested rank increment")

    def norm(values):
        return float(np.linalg.norm(values, ord="fro"))

    scale = max(1.0, norm(p), norm(q))
    errors = {
        "P_symmetry": norm(p - p.T) / scale,
        "Q_symmetry": norm(q - q.T) / scale,
        "P_idempotence": norm(p @ p - p) / scale,
        "Q_idempotence": norm(q @ q - q) / scale,
        "PQ_equals_Q": norm(p @ q - q) / scale,
        "QP_equals_Q": norm(q @ p - q) / scale,
    }
    if max(errors.values()) > IDENTITY_TOLERANCE:
        raise ArithmeticError("Joint projection identity failed: " + repr(errors))
    parcel_fields = {name: op.apply(x) for name, op in operators.items()}
    fields = {name: aggregate_networks(v, membership) for name, v in parcel_fields.items()}
    source_c7 = aggregate_networks(x, membership)
    aggregation_errors = {
        name: norm(fields[name] - op.apply(source_c7)) / max(1.0, norm(source_c7))
        for name, op in operators.items()
    }
    energies = {name: float(np.sum(v * v)) for name, v in parcel_fields.items()}
    lost = parcel_fields["BASE_DCT"] - parcel_fields["A5_DCT"]
    energy_decomposition_error = abs(
        energies["BASE_DCT"] - energies["A5_DCT"] - float(np.sum(lost * lost))
    ) / max(1.0, energies["BASE_DCT"])
    if (
        max(aggregation_errors.values()) > IDENTITY_TOLERANCE
        or energy_decomposition_error > IDENTITY_TOLERANCE
    ):
        raise ArithmeticError("Full-field aggregation/energy identity failed")
    diagnostics = {
        "samples": n,
        "cosine_columns": c.shape[1],
        "nodes": details,
        "effective_A5_rank": increment,
        "projection_errors": errors,
        "aggregation_relative_errors": aggregation_errors,
        "parcel_residual_energies": energies,
        "raw_energy_decomposition_relative_error": energy_decomposition_error,
    }
    return fields, diagnostics
