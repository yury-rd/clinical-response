"""Five controlled observation and proxy worlds from the completed panel."""

import hashlib

import numpy as np

from . import score as h1
from .score import _real_array, aggregate_networks, residual_operator

NETWORKS = ["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis"]
COUNTS = [52, 91, 46, 26, 47, 77, 61]
WORLDS = {
    "null": [0.0, 0.25, 0.5, 0.75, 1.0],
    "observation_amplitude": [0.0, 0.5, 1.0, 2.0, 4.0],
    "proxy_quality": [0.0, 0.25, 0.5, 0.75, 1.0],
    "latent_sharing": [0.0, 0.2, 0.4, 0.6, 0.8],
    "proxy_contamination": [0.0, 0.5, 1.0, 2.0, 4.0],
}


def membership():
    """Return the fixed 400-parcel mapping used by the synthetic generators.

    Labels index NETWORKS. Parcel order is preserved within two hemispheres;
    this helper is not a substitute for a local dataset's atlas assignment.
    """
    return np.repeat(
        [6, 5, 2, 4, 3, 0, 1, 6, 5, 2, 4, 3, 0, 1],
        [31, 37, 23, 22, 13, 22, 52, 30, 40, 23, 25, 13, 30, 39],
    ).astype(int)


def array_hash(x):
    """SHA-256 of C-contiguous float64 bytes for deterministic diagnostics."""
    return hashlib.sha256(np.ascontiguousarray(_real_array(x)).tobytes()).hexdigest()


def ar_from_innovations(e, phi):
    """AR(1) paths with initial state e[0] and innovation scale sqrt(1-phi²)."""
    e = _real_array(e)
    if (
        e.ndim != 2
        or len(e) < 2
        or e.shape[1] < 1
        or not np.isfinite(e).all()
        or not np.isscalar(phi)
        or not np.isfinite(phi)
        or not abs(phi) < 1
    ):
        raise ValueError("Finite innovations and stationary phi required")
    y = np.empty_like(e)
    y[0] = e[0]
    sigma = np.sqrt(1.0 - phi * phi)
    for t in range(1, len(e)):
        y[t] = phi * y[t - 1] + sigma * e[t]
    return y


def ar(n, k, phi, seed):
    """Generate n-by-k AR(1) paths from seeded standard-normal innovations."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer of at least 2")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return ar_from_innovations(
        np.random.default_rng(int(seed)).standard_normal(size=(n, k), dtype=np.float64), phi
    )


def bank(n, seed):
    """Build the latent, artifact, motion and within-network simulation bank.

    Use the same bank across levels to pair the generator randomness. The
    saved mechanism panels use n in {197, 788}; projection/scoring need adequate
    residual degrees of freedom beyond the nuisance design rank.
    """
    idx = membership()
    z = ar(n, 7, 0.5, seed)
    u = ar(n, 1, 0.5, seed + 1)
    v = ar(n, 1, 0.5, seed + 2)
    h = ar(n, 1, 0.5, seed + 3)
    extra = ar(n, 4, 0.5, seed + 4)
    motion = ar(n, 6, 0.3, seed + 5)
    derivative = np.vstack((np.zeros((1, 6)), np.diff(motion, axis=0)))
    m24 = np.column_stack((motion, derivative, motion**2, derivative**2))
    within = 0.3 * ar(n, 400, 0.5, seed + 6)
    for j in range(7):
        within[:, idx == j] -= within[:, idx == j].mean(axis=1, keepdims=True)
    coefficients = 0.01 * np.random.default_rng(int(seed + 7)).standard_normal(
        size=(24, 400), dtype=np.float64
    )
    trend = np.linspace(-1.0, 1.0, n, dtype=float)[:, None]
    design = np.column_stack((trend, m24))
    base = residual_operator(design, operator_id="SYNTHETIC_BASE", include_intercept=True)
    return dict(
        n=n,
        seed=int(seed),
        membership=idx,
        z=z,
        u=u,
        v=v,
        h=h,
        extra=extra,
        m24=m24,
        within=within,
        motion_field=m24 @ coefficients,
        design=design,
        base=base,
    )


def world(b, name, level):
    """Construct BASE/A5 network fields and known signal/artifact diagnostics.

    ``name`` is a key in WORLDS. The null world's level is ignored;
    proxy_quality requires [0, 1], and latent_sharing requires [0, 1).
    The amplitude and contamination worlds accept any finite scalar level.
    """
    if not np.isscalar(level) or not np.isfinite(level):
        raise ValueError("level must be a finite scalar")
    z, u, v, h = b["z"], b["u"], b["v"], b["h"]
    if name == "null":
        signal = z
        artifact = np.zeros_like(z)
        proxy = v
    elif name == "observation_amplitude":
        signal = z
        artifact = np.broadcast_to(level * u, z.shape)
        proxy = u + 0.5 * v
    elif name == "proxy_quality":
        if not 0 <= level <= 1:
            raise ValueError("q outside support")
        signal = z
        artifact = np.broadcast_to(u, z.shape)
        proxy = np.sqrt(level) * u + np.sqrt(1 - level) * v
    elif name == "latent_sharing":
        if not 0 <= level < 1:
            raise ValueError("r outside support")
        signal = np.sqrt(1 - level) * z + np.sqrt(level) * h
        artifact = np.broadcast_to(u, z.shape)
        proxy = u + 0.5 * v
    elif name == "proxy_contamination":
        signal = np.sqrt(0.75) * z + 0.5 * h
        artifact = np.broadcast_to(u, z.shape)
        proxy = u + level * h + 0.5 * v
    else:
        raise ValueError("Unknown world")
    proxies = np.column_stack((proxy, b["extra"]))
    clean_observed = signal + artifact
    parcels = clean_observed[:, b["membership"]] + b["within"] + b["motion_field"]
    a5 = residual_operator(
        np.column_stack((b["design"], proxies)), operator_id="SYNTHETIC_A5", include_intercept=True
    )
    fields = {}
    energies = {}
    errors = []
    for label, op in [("BASE", b["base"]), ("A5", a5)]:
        fields[label] = aggregate_networks(op.apply(parcels), b["membership"])
        s = op.apply(signal)
        a = op.apply(artifact)
        errors.append(float(np.max(abs(fields[label] - s - a))))
        energies[label] = {
            "signal": float(np.sum(s * s)),
            "artifact": float(np.sum(a * a)),
            "cross": float(2 * np.sum(s * a)),
            "total": float(np.sum(fields[label] ** 2)),
        }
    e0, e1 = energies["BASE"], energies["A5"]
    diagnostics = {
        "rank_BASE": b["base"].rank,
        "rank_A5": a5.rank,
        "full400_projection_aggregation_error": max(errors),
        "energy": energies,
        "signal_retention_A5_over_BASE": e1["signal"] / e0["signal"],
        "artifact_retention_A5_over_BASE": e1["artifact"] / e0["artifact"]
        if e0["artifact"] > 0
        else None,
        "acquired_parcels_sha256": array_hash(parcels),
        "A5_proxies_sha256": array_hash(proxies),
        "signal_sha256": array_hash(signal),
        "BASE_C7_sha256": array_hash(fields["BASE"]),
    }
    return dict(
        fields=fields,
        signal=signal,
        artifact=artifact,
        proxies=proxies,
        parcels=parcels,
        a5=a5,
        diagnostics=diagnostics,
        alternative_signal=clean_observed if name == "observation_amplitude" else None,
    )


def score(x, seed, cache):
    """Cache fullscore results by field bytes and the first reference seed."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    # Equal bytes with an invalid reshaped field must not bypass validation.
    x = h1._field(x)
    key = (array_hash(x), int(seed))
    if key not in cache:
        cache[key] = h1.fullscore(x, seed_start=int(seed), repetitions=199, batch_size=16)
    return cache[key]


def summarize_replicates(values):
    """Summarize 16 replicate-by-level curves and paired adjacent differences.

    Simulation SE is the sample standard deviation divided by sqrt(16).
    It describes generator replication, not patient-level uncertainty.
    """
    a = _real_array(values)
    if a.ndim != 2 or a.shape[0] != 16 or a.shape[1] < 1 or not np.isfinite(a).all():
        raise ValueError("16 complete replicate curves required")
    diff = np.diff(a, axis=1)
    return {
        "mean": a.mean(axis=0).tolist(),
        "simulation_SE": (a.std(axis=0, ddof=1) / np.sqrt(16)).tolist(),
        "adjacent_mean_difference": diff.mean(axis=0).tolist(),
        "adjacent_simulation_SE": (diff.std(axis=0, ddof=1) / np.sqrt(16)).tolist(),
        "replicates": 16,
        "scope": "Mean response and MC standard error across independent generator replicates, paired across levels; no p/CI or patient sampling inference.",
    }
