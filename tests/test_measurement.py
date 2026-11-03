"""Numerical contracts for projection, delay energy, and synthetic controls.

All fixtures are synthetic float64 arrays. Tolerances accommodate roundoff in
SVD, least-squares, and FFT implementations; they are not statistical bounds.
"""

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from clinical_response import lag_control, mechanisms, paired, score


@pytest.fixture(autouse=True)
def single_blas_thread():
    """Avoid oversubscribing BLAS for the small surrogate eigensystems."""
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def field():
    return np.random.default_rng(2103).normal(size=(97, 7))


@pytest.fixture
def paired_input():
    rng = np.random.default_rng(822)
    n = 97
    time = np.arange(n, dtype=np.float64)
    cosines = np.cos(np.pi * (time[:, None] + 0.5) * np.arange(1, 4) / n)
    motion = rng.normal(size=(n, 24))
    anatomical = rng.normal(size=(n, 5)) + 0.3 * motion[:, :5]
    parcels = rng.normal(size=(n, 400))
    # Unequal network sizes detect accidental sum or unweighted-total pooling.
    membership = mechanisms.membership()
    return parcels, motion, anatomical, cosines, membership


def test_joint_projection_matches_direct_least_squares():
    rng = np.random.default_rng(109)
    design = rng.normal(size=(79, 6)) + np.arange(6)
    values = rng.normal(size=(79, 11))
    op = score.residual_operator(design, operator_id="test")
    augmented = np.column_stack((np.ones(79), design))
    expected = values - augmented @ np.linalg.lstsq(augmented, values, rcond=None)[0]
    np.testing.assert_allclose(op.apply(values), expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(augmented.T @ op.apply(values), 0, rtol=0, atol=1e-11)
    assert op.rank == 7


def test_projector_is_invariant_to_span_coordinates_and_duplicate_columns():
    rng = np.random.default_rng(110)
    design = rng.normal(size=(61, 4))
    changed = np.column_stack((design * [1e-5, -3, 50, 1e5], design[:, 2], np.ones(61)))
    a = score.residual_operator(design, operator_id="a")
    b = score.residual_operator(changed, operator_id="b")
    assert a.rank == b.rank == 5
    np.testing.assert_allclose(a.matrix(), b.matrix(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.matrix() @ a.matrix(), a.matrix(), rtol=0, atol=1e-12)


def test_common_dct_projection_and_nested_energy_match_direct_fits(paired_input):
    parcels, motion, anatomical, cosines, membership = paired_input
    fields, diagnostics = paired.paired_fields(*paired_input)
    n = len(parcels)
    common = np.column_stack((np.ones(n), np.linspace(-1, 1, n), motion, cosines))
    designs = {"BASE_DCT": common, "A5_DCT": np.column_stack((common, anatomical))}
    residuals = {}
    for name, design in designs.items():
        residual = parcels - design @ np.linalg.lstsq(design, parcels, rcond=None)[0]
        residuals[name] = residual
        expected = np.column_stack(
            [residual[:, membership == network].mean(axis=1) for network in range(7)]
        )
        np.testing.assert_allclose(fields[name], expected, rtol=0, atol=2e-12)
        np.testing.assert_allclose(cosines.T @ fields[name], 0, rtol=0, atol=1e-11)
        assert diagnostics["nodes"][name]["nuisance_rank"] == design.shape[1]

    base, a5 = residuals["BASE_DCT"], residuals["A5_DCT"]
    assert diagnostics["cosine_columns"] == 3
    assert diagnostics["effective_A5_rank"] == 5
    np.testing.assert_allclose(
        np.sum(base * base), np.sum(a5 * a5) + np.sum((base - a5) ** 2), rtol=1e-12
    )
    assert diagnostics["raw_energy_decomposition_relative_error"] < 1e-12


def test_redundant_anatomical_columns_do_not_invent_a_response(paired_input):
    parcels, motion, _, cosines, membership = paired_input
    fields, diagnostics = paired.paired_fields(parcels, motion, motion[:, :5], cosines, membership)
    assert diagnostics["effective_A5_rank"] == 0
    np.testing.assert_allclose(fields["BASE_DCT"], fields["A5_DCT"], rtol=0, atol=1e-12)


def test_known_world_and_empty_dct_measurement_agree():
    bank = mechanisms.bank(97, 202619000000)
    world = mechanisms.world(bank, "observation_amplitude", 2.0)
    fields, diagnostics = paired.paired_fields(
        world["parcels"], bank["m24"], world["proxies"], np.empty((97, 0)), bank["membership"]
    )
    np.testing.assert_allclose(fields["BASE_DCT"], world["fields"]["BASE"], rtol=0, atol=1e-11)
    np.testing.assert_allclose(fields["A5_DCT"], world["fields"]["A5"], rtol=0, atol=1e-11)
    assert diagnostics["cosine_columns"] == 0
    assert world["diagnostics"]["full400_projection_aggregation_error"] < 1e-11


def test_mask_encodings_agree_and_preserve_suffix():
    indicator = np.r_[np.zeros(5), np.ones(20)]
    selected = paired.normalize_mask(indicator, 25)
    np.testing.assert_array_equal(selected, paired.normalize_mask(indicator.astype(bool), 25))
    np.testing.assert_array_equal(np.flatnonzero(selected), np.arange(5, 25))


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros(25),
        np.r_[np.zeros(8), np.ones(17)],
        np.r_[np.ones(20), np.zeros(5)],
        np.r_[0, np.ones(10), 0, np.ones(13)],
        np.full(25, 0.5),
        np.full(25, np.nan),
        np.ones((25, 1)),
        np.full(25, "1"),
    ],
)
def test_invalid_masks_are_rejected(mask):
    with pytest.raises(ValueError):
        paired.normalize_mask(mask, 25)


def test_delay_embedding_uses_channel_major_nonwrapping_windows():
    values = np.column_stack((np.arange(8), 100 + np.arange(8)))
    delay = score.joint_block_hankel(values, 3)
    assert delay.shape == (6, 6)
    np.testing.assert_array_equal(delay[:, 0], [0, 1, 2, 100, 101, 102])
    np.testing.assert_array_equal(delay[:, -1], [5, 6, 7, 105, 106, 107])


def test_rho2_matches_direct_singular_value_energy(field):
    centered = field - field.mean(axis=0)
    normalized = centered / np.sqrt(np.mean(centered**2, axis=0))
    # Construct directly by coordinates rather than calling the embedding helper.
    delay = np.array(
        [
            normalized[lag : lag + len(field) - 9, channel]
            for channel in range(7)
            for lag in range(10)
        ]
    )
    singular = np.linalg.svd(delay, compute_uv=False)
    expected = np.sum(singular[:2] ** 2) / np.sum(singular**2)
    assert score.rho2(field) == pytest.approx(expected, abs=1e-13)
    assert score.rho2(field * np.arange(1, 8) + np.arange(7)) == pytest.approx(expected, abs=1e-13)


@pytest.mark.parametrize("samples", [97, 98])
def test_independent_phase_preserves_each_marginal_spectrum(samples):
    rng = np.random.default_rng(311)
    # Identical channels make common-phase randomization detectable.
    values = np.repeat(rng.normal(size=(samples, 1)), 7, axis=1)
    surrogate = score.independent_phase_surrogate(values, seed=512)
    expected = np.fft.rfft(values, axis=0)
    actual = np.fft.rfft(surrogate, axis=0)
    np.testing.assert_allclose(np.abs(actual), np.abs(expected), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=1e-12)
    if samples % 2 == 0:
        np.testing.assert_allclose(actual[-1], expected[-1], rtol=0, atol=1e-12)
    assert not np.allclose(surrogate[:, 0], surrogate[:, 1])
    np.testing.assert_array_equal(surrogate, score.independent_phase_surrogate(values, seed=512))


def test_fullscore_has_199_references_and_is_batch_invariant(field):
    a = score.fullscore(field, seed_start=712, batch_size=16)
    b = score.fullscore(field, seed_start=712, batch_size=31)
    assert len(a["surrogate_rho2"]) == a["surrogates"] == 199
    np.testing.assert_allclose(a["surrogate_rho2"], b["surrogate_rho2"], rtol=0, atol=1e-13)
    for index in (0, 99, 198):
        direct = score.rho2(score.independent_phase_surrogate(field, seed=712 + index))
        assert a["surrogate_rho2"][index] == pytest.approx(direct, abs=1e-13)
    assert a["excess"] == pytest.approx(
        a["observed_rho2"] - np.mean(a["surrogate_rho2"]), abs=1e-15
    )
    assert a["montecarlo_standard_error"] == pytest.approx(
        np.std(a["surrogate_rho2"], ddof=1) / np.sqrt(199), abs=1e-15
    )
    assert a["maximum_relative_marginal_spectrum_error"] < 1e-12


def test_lag_control_uses_independent_prehistory_without_wraparound():
    innovations = np.random.default_rng(50).normal(size=(98, 7))
    original = innovations.copy()
    uncoupled = lag_control.lag_world(innovations, 0)
    np.testing.assert_array_equal(uncoupled, innovations[1:])
    coupled = lag_control.lag_world(innovations, 1)
    np.testing.assert_array_equal(coupled[:, 1], innovations[:-1, 0])
    np.testing.assert_array_equal(
        coupled[:, [0, 2, 3, 4, 5, 6]], innovations[1:, [0, 2, 3, 4, 5, 6]]
    )
    changed = innovations.copy()
    changed[-1, 0] += 100
    np.testing.assert_array_equal(lag_control.lag_world(changed, 1)[:-1], coupled[:-1])
    np.testing.assert_array_equal(innovations, original)


@pytest.mark.parametrize(
    "values", [np.ones((97, 7)), np.zeros((17, 7)), np.zeros((97, 6)), np.full((97, 7), np.nan)]
)
def test_invalid_scoring_fields_are_rejected(values):
    with pytest.raises(ValueError):
        score.rho2(values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"repetitions": 198},
        {"repetitions": 199.0},
        {"batch_size": 0},
        {"batch_size": 1.5},
        {"batch_size": True},
        {"seed_start": -1},
        {"seed_start": 1.5},
    ],
)
def test_invalid_score_parameters_are_rejected(field, kwargs):
    with pytest.raises(ValueError):
        score.fullscore(field, **({"seed_start": 0} | kwargs))


@pytest.mark.parametrize(
    "which", ["width", "support", "nonfinite", "membership", "missing_network"]
)
def test_invalid_paired_inputs_are_rejected(paired_input, which):
    values = list(paired_input)
    if which == "width":
        values[1] = values[1][:, :-1]
    elif which == "support":
        values[3] = values[3][:-1]
    elif which == "nonfinite":
        values[0][0, 0] = np.inf
    elif which == "membership":
        values[4] = values[4].astype(float)
    else:
        values[4][values[4] == 6] = 5
    with pytest.raises(ValueError):
        paired.paired_fields(*values)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan])
def test_lag_control_rejects_invalid_coupling(alpha):
    with pytest.raises(ValueError):
        lag_control.lag_world(np.zeros((20, 7)), alpha)


@pytest.mark.parametrize("values", [np.empty((0, 7)), np.empty((20, 0))])
def test_empty_channel_normalization_is_rejected(values):
    with pytest.raises(ValueError):
        score.normalize_channels(values, center=True, scale="unit-rms")


@pytest.mark.parametrize("values", [2.0, np.full((20, 2), np.nan), np.zeros((20, 2, 3))])
def test_residual_operator_rejects_invalid_apply_inputs(values):
    op = score.residual_operator(np.ones((20, 1)), operator_id="test")
    with pytest.raises(ValueError):
        op.apply(values)


@pytest.mark.parametrize("multiplier", [0, -1, np.nan, np.inf])
def test_rank_tolerance_must_be_positive_finite(multiplier):
    with pytest.raises(ValueError):
        score.residual_operator(
            np.ones((20, 1)), operator_id="test", tolerance_multiplier=multiplier
        )


def test_centered_design_without_intercept_preserves_constant_vectors():
    design = np.arange(20, dtype=float)[:, None] + 100
    op = score.residual_operator(design, operator_id="test", include_intercept=False)
    assert op.rank == 1
    np.testing.assert_allclose(op.apply(np.ones(20)), 1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(op.apply(design - design.mean()), 0, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("name", "level"),
    [
        ("null", 0),
        ("observation_amplitude", 2),
        ("proxy_quality", 0.5),
        ("latent_sharing", 0.4),
        ("proxy_contamination", 4),
    ],
)
def test_control_world_signal_artifact_decomposition(name, level):
    bank = mechanisms.bank(97, 251)
    result = mechanisms.world(bank, name, level)
    for label, operator in [("BASE", bank["base"]), ("A5", result["a5"])]:
        # Parcel-local noise and added motion should disappear upon aggregation
        # and projection, leaving the stipulated signal plus artifact.
        expected = operator.apply(result["signal"] + result["artifact"])
        np.testing.assert_allclose(result["fields"][label], expected, rtol=0, atol=1e-11)
        energy = result["diagnostics"]["energy"][label]
        assert energy["total"] == pytest.approx(
            energy["signal"] + energy["artifact"] + energy["cross"], rel=1e-12
        )


def test_null_world_has_no_artifact_and_ignores_nominal_level():
    bank = mechanisms.bank(97, 251)
    low = mechanisms.world(bank, "null", 0)
    high = mechanisms.world(bank, "null", 1)
    np.testing.assert_array_equal(low["artifact"], 0)
    for name in ("BASE", "A5"):
        np.testing.assert_array_equal(low["fields"][name], high["fields"][name])


def test_score_cache_cannot_bypass_field_shape_validation(field):
    cache = {}
    result = mechanisms.score(field, 1000, cache)
    assert mechanisms.score(field.copy(), 1000, cache) is result
    with pytest.raises(ValueError):
        mechanisms.score(field.reshape(7, 97), 1000, cache)


@pytest.mark.parametrize(
    "function", [score.rho2, score.normalize_channels, lag_control.finite_matrix]
)
def test_complex_observations_are_not_silently_cast_to_real(field, function):
    kwargs = {"center": True, "scale": "unit-rms"} if function is score.normalize_channels else {}
    with pytest.raises(ValueError, match="real"):
        function(field.astype(complex) + 1j, **kwargs)
