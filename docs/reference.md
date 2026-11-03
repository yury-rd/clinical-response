# Command and Python API reference

See the [README](../README.md) for installation, the [local-data
guide](local-data.md) for input preparation, and [methods](methods.md) for the
meaning of the measurement.

## Command line

`clinical-response` and `python -m clinical_response` expose the same command.

| Option | Default | Contract |
|---|---|---|
| `--mode {demo,summary}` | `demo` | Run computations or read two external aggregate JSON files from an authorized private data checkout. |
| `--data-dir PATH` | None | Summary mode only: directory containing `aggregate_results.json` and `lag_aggregate.json`. Overrides `CLINICAL_RESPONSE_DATA_DIR` and the current working directory's `data/` fallback. |
| `--input PATH` | None | NPZ containing a local paired measurement input; without it, demo mode runs synthetic examples. |
| `--output PATH` | None | Write JSON to this path; without it, write JSON to standard output. |
| `--seed INTEGER` | `202620000000` | Nonnegative first phase-reference seed. |
| `--batch-size INTEGER` | `16` | Positive number of references processed together; all 199 references are still evaluated. |
| `--version` | — | Show package version and exit. |
| `--help` | — | Show usage and exit. |

```sh
clinical-response --mode demo --output results/demo.json
clinical-response --mode summary --data-dir data --output results/aggregates.json
clinical-response --input local_parcels.npz --output results/local.json
```

Summary mode requires access to the external tables and rejects `--input`;
seed and batch settings do not change saved aggregates. `--data-dir` is
accepted only with `--mode summary`. There are no data files bundled in the
installed package. The input and output paths must differ. Output directories are
created as needed. With `--output`, standard output contains a short JSON
receipt with `output` and `mode`; the result itself is in the file. Without
`--output`, standard output contains the result. Errors in arguments, input
data, or file access produce a diagnostic on standard error and exit status 2.

The synthetic demo returns `scope`, `phase_references` (199), `worlds` (two
records with world name, level, BASE/A5 scores, their difference, and simulation
diagnostics), and `lag_control` (three `alpha`/`J` records). Local measurements
return `scope`, full `BASE` and `A5` score records, `delta_J`, and `projection`
diagnostics. Both computational command modes append `runtime` with package
and NumPy versions, phase seed, batch size, units, and phase-pairing convention.
Summary mode returns only the saved tables keyed by filename.

Local NPZ arrays must share retained rows and use these exact keys:

| Key | Shape | Values |
|---|---|---|
| `parcels` | `(N, 400)` | Finite prepared parcel series, `N >= 18`. |
| `motion24` | `(N, 24)` | Finite, already constructed motion regressors. |
| `a5` | `(N, 5)` | Finite, already selected anatomical nuisance columns. |
| `cosines` | `(N, K)` | Finite saved cosine columns on the same rows; `K = 0` omits this common block. |
| `membership` | `(400,)` | Integer labels 0 through 6, all present. |

The projector must leave positive residual degrees of freedom, and every
resulting network must support RMS normalization. Thus `N >= 18` is a shape
minimum, not a guarantee that a chosen nuisance design can be scored.

## Python modules

Import functions from `clinical_response.score`, `clinical_response.paired`,
`clinical_response.mechanisms`, or `clinical_response.lag_control`. Numerical
arrays are converted to NumPy `float64`, except network labels and masks.
Observed-value arrays must be real; complex arrays are rejected.
Names beginning with `_` are implementation details.

### `cli` convenience API

| Function | Contract |
|---|---|
| `load_aggregates(data_dir=None)` | Reads both external JSON tables and returns them keyed by filename. An explicit path takes precedence over `CLINICAL_RESPONSE_DATA_DIR`; otherwise uses `Path.cwd() / "data"`. Missing data raises an actionable error. No packaged-data fallback. |
| `measure_input(path, *, seed_start=202620000000, batch_size=16)` | Reads the required NPZ arrays with pickle loading disabled; returns the local paired result described above. |
| `run_demo(*, seed_start=202620000000, batch_size=16)` | Returns the synthetic demo result. Generator seed is fixed at `202619000000`; `seed_start` changes phase references only. |
| `main(argv=None)` | Parses an optional list of command arguments, writes JSON or an output receipt, and returns 0 on success. `None` reads process arguments. Argument and data errors raise `SystemExit(2)` through argparse. |

Only `main` adds runtime metadata and limits numerical-library threads to one.
The other functions return dictionaries and leave thread configuration to the
caller. `clinical_response.__version__` is the package version string.

### `score`

Constants: `WINDOW = 10`, `CHANNELS = 7`.

| Function | Input and result |
|---|---|
| `normalize_channels(values, *, center, scale)` | Nonempty finite matrix. `center` is an explicit Boolean; `scale` is `"none"` or `"unit-rms"`. Returns a copy, centered first if requested, then scaled. RMS at or below float64 machine epsilon is rejected for `unit-rms`. |
| `scalar_hankel(values, window)` | Finite vector with at least three samples; integer `2 <= window <= N-1`. Returns noncircular windows as a `(window, N-window+1)` matrix. |
| `joint_block_hankel(values, window)` | Finite `(N, C)` matrix, `N >= 3`, `C >= 2`, same window constraint. Returns `(C*window, N-window+1)`, stacking channel blocks in input order. Does not normalize the input. |
| `residual_operator(design, *, operator_id, include_intercept=True, tolerance_multiplier=1.0)` | Finite vector or matrix with at least two rows; explicit Boolean `include_intercept`; finite positive `tolerance_multiplier`. Centers design columns, drops numerically constant columns, normalizes their Euclidean norms, optionally adds an intercept, and computes an SVD basis. Returns `OrthogonalResidualOperator`. |
| `aggregate_networks(values, membership)` | Finite `(N, P)` matrix, `N >= 16`, `P >= 2`; one label per column and at least two nonempty groups. Returns unweighted group means in sorted label order. `paired_fields` imposes the stricter 400-parcel/seven-network contract. |
| `rho2(values)` | Finite `(N, 7)` field, `N >= 18`, no constant channels after centering. Returns the leading-two singular-energy fraction of the centered, unit-RMS, length-10 joint delay matrix. |
| `independent_phase_surrogate(values, *, seed)` | Same field contract; nonnegative integer seed. Returns one `(N, 7)` reference preserving marginal Fourier magnitudes, DC, and the even-length Nyquist coefficient. It does not RMS-normalize the returned field. |
| `fullscore(values, *, seed_start, repetitions=199, batch_size=16)` | Same field contract; nonnegative integer first seed; exactly 199 references required; positive integer batch size. Returns the score and diagnostics below. |

`OrthogonalResidualOperator` is a frozen dataclass with `operator_id` (string),
`basis` (`N × rank` orthonormal array), `rank` (integer), and `samples`
(integer). `apply(values)` takes a finite vector or matrix and returns
`values - basis @ (basis.T @ values)` on matching sample rows. `matrix()`
returns the dense `N × N` residual projector.

The SVD cutoff in `residual_operator` is
`tolerance_multiplier * eps * max(normalized_design.shape) * largest_singular_value`.
Design centering happens even when `include_intercept=False`; that option
controls only the additional intercept direction. The main paired workflow
always uses `include_intercept=True`.

#### `fullscore` result

`J` is stored under `excess`. Scores and differences use fractional units;
multiply by 100 to express a difference in percentage points.

| Key | Type / meaning |
|---|---|
| `observed_rho2` | Float; measured leading-two energy fraction. |
| `surrogate_rho2` | List of 199 floats, in seed order `seed_start + 0` through `seed_start + 198`. |
| `surrogate_mean` | Mean of the 199 reference scores. |
| `excess` | `observed_rho2 - surrogate_mean`. |
| `surrogate_std` | Sample standard deviation, denominator 198. |
| `montecarlo_standard_error` | `surrogate_std / sqrt(199)`. |
| `first100_minus_last99` | Difference between means of the first 100 and final 99 reference scores. |
| `surrogate_rho2_min`, `surrogate_rho2_max` | Extremes across reference scores. |
| `observed_hankel_trace` | Squared Frobenius norm of the observed normalized delay matrix. |
| `surrogate_hankel_trace_min`, `surrogate_hankel_trace_max` | Extremes of that delay energy across references. |
| `observed_channel_hankel_traces` | Seven floats; each channel's contribution to observed delay energy. |
| `surrogate_channel_trace_ratio_min`, `surrogate_channel_trace_ratio_max` | Seven floats each; per-channel extremes of reference/observed delay energy. |
| `maximum_relative_marginal_spectrum_error` | Largest absolute Fourier-magnitude discrepancy across references, divided by the largest original Fourier magnitude. |
| `seed_start`, `surrogates`, `samples`, `channels`, `window` | Integers recording the first seed, 199, `N`, 7, and 10. |

Example using a synthetic field:

```python
import numpy as np
from clinical_response.score import fullscore

field = np.random.default_rng(42).normal(size=(97, 7))
result = fullscore(field, seed_start=1000)
print(result["excess"])
assert result["surrogates"] == 199
```

### `paired`

`normalize_mask(mask, nraw)` requires an integer `nraw >= 18` and accepts a
length-`nraw` Boolean or numeric 0/1 indicator, returning a Boolean array.
Selected rows must form a contiguous
suffix ending at the last raw row, with at least 18 selected rows. Nonfinite
values, internal holes, and other values are rejected.

`paired_fields(parcels, motion24, A5, cosines, membership)` takes the arrays
specified in the NPZ table above. Its Python parameter is named `A5`; the NPZ
key is lower-case `a5`. It returns `(fields, diagnostics)`:

- `fields["BASE_DCT"]` and `fields["A5_DCT"]` are unnormalized `(N, 7)`
  residual network means. These names are retained even with zero cosine
  columns.
- `diagnostics` contains the fields below.

| Key | Meaning |
|---|---|
| `samples`, `cosine_columns` | `N` and `K`. |
| `nodes` | One record per branch: `nuisance_rank`, `residual_df`, and `ranks_by_tolerance_multiplier`. |
| `effective_A5_rank` | A5 rank minus BASE rank, constrained to 0 through 5. |
| `projection_errors` | Relative Frobenius residuals for `P_symmetry`, `Q_symmetry`, `P_idempotence`, `Q_idempotence`, `PQ_equals_Q`, and `QP_equals_Q`. |
| `aggregation_relative_errors` | Per-branch difference between project-then-average and average-then-project, relative to `max(1, norm(original_networks))`. |
| `parcel_residual_energies` | Sum of squared residual parcel values per branch, before network averaging or RMS scaling. |
| `raw_energy_decomposition_relative_error` | Residual of the nested raw-energy identity, relative to `max(1, BASE_energy)`. |

`TOLERANCE_MULTIPLIERS = (0.1, 1.0, 10.0, 100.0)` must produce the same rank.
The actual operator uses multiplier 1.0. `IDENTITY_TOLERANCE = 1e-10` governs
projection, aggregation, and raw-energy checks. Invalid inputs, ambiguous rank,
or exhausted residual degrees of freedom raise `ValueError`; failed numerical
identities raise `ArithmeticError`.

### `mechanisms`

These lower-level functions expose the controlled simulation components. They
do not fit a clinical model.

| Function | Contract |
|---|---|
| `membership()` | Returns the fixed length-400 integer network assignment for the simulator. |
| `array_hash(x)` | SHA-256 hex digest of contiguous float64 array bytes; shape is not included in the digest. |
| `ar_from_innovations(e, phi)` | Finite innovation matrix with at least two rows and one column; finite scalar `abs(phi) < 1`. Returns `y[0] = e[0]`, then `y[t] = phi*y[t-1] + sqrt(1-phi**2)*e[t]`. |
| `ar(n, k, phi, seed)` | Integer `n >= 2`, integer `k >= 1`, nonnegative integer seed. Draws a seeded `(n, k)` standard-normal innovation matrix and applies `ar_from_innovations`. |
| `bank(n, seed)` | Builds a shared simulation dictionary: latent/nuisance streams, parcel membership, motion design, within-network noise, and BASE projector. |
| `world(b, name, level)` | Builds one of the five worlds from a bank; finite scalar `level` with world-specific constraints below. Returns fields, component arrays, proxies, parcels, the A5 operator, and diagnostics. Network aggregation requires at least 16 frames; scoring requires at least 18. |
| `score(x, seed, cache)` | Caches `fullscore` in a caller-supplied dictionary using `(array_hash(x), int(seed))`; uses 199 references and batch size 16. |
| `summarize_replicates(values)` | Finite matrix with exactly 16 rows (replicates) and at least one column (levels). Returns per-level means and simulation standard errors, plus paired adjacent-level differences and their standard errors. |

`NETWORKS` gives names in label order. `COUNTS` is
`[52, 91, 46, 26, 47, 77, 61]`. The built-in panel levels in `WORLDS` are:

| Name | Levels |
|---|---|
| `null` | 0, 0.25, 0.5, 0.75, 1 |
| `observation_amplitude` | 0, 0.5, 1, 2, 4 |
| `proxy_quality` | 0, 0.25, 0.5, 0.75, 1 |
| `latent_sharing` | 0, 0.2, 0.4, 0.6, 0.8 |
| `proxy_contamination` | 0, 0.5, 1, 2, 4 |

The bank's independent AR streams are `z` (seven latent channels), `u`, `v`,
and `h` (one channel each), and `extra` (four additional nuisance channels),
all with `phi=0.5`. Six motion streams use `phi=0.3`. Their values, first
differences (initial derivative zero), squared values, and squared differences
form `m24`. Seeds are the supplied seed through seed + 7. Within-network
parcel noise is centered across parcels at each frame, and a small motion
field is added to all observed parcels.

In the following table, scalar streams broadcast across the seven networks;
`a`, `q`, and `r` denote `level`. Each world's five proxy columns are its
listed proxy followed by the four `extra` streams.

| World | Signal | Artifact | First proxy |
|---|---|---|---|
| `null` | `z` | 0 | `v` |
| `observation_amplitude` | `z` | `a*u` | `u + 0.5*v` |
| `proxy_quality` | `z` | `u` | `sqrt(q)*u + sqrt(1-q)*v`, `0 <= q <= 1` |
| `latent_sharing` | `sqrt(1-r)*z + sqrt(r)*h` | `u` | `u + 0.5*v`, `0 <= r < 1` |
| `proxy_contamination` | `sqrt(0.75)*z + 0.5*h` | `u` | `u + a*h + 0.5*v` |

The null world's level is ignored. Amplitude and contamination levels may be
any finite scalar in the API; the saved panel uses the nonnegative grid above.

`world` returns `fields` (`BASE` and `A5`, each `N × 7`), `signal`,
`artifact`, `proxies` (`N × 5`), `parcels` (`N × 400`), `a5` (operator),
`diagnostics`, and `alternative_signal` (signal + artifact only for
`observation_amplitude`, otherwise `None`). Diagnostics record both ranks,
full-parcel projection/aggregation error, signal/artifact/cross/total energy
per branch, A5/BASE signal and artifact energy ratios, and array hashes.
Artifact retention is `None` when BASE artifact energy is zero.

`summarize_replicates` returns `mean`, `simulation_SE`,
`adjacent_mean_difference`, `adjacent_simulation_SE`, `replicates` (16), and
`scope`. Standard errors use the sample standard deviation divided by 4.
Adjacent differences are taken within each replicate before summarization.

### `lag_control`

`finite_matrix(values)` converts a nonempty finite real matrix to float64 or
raises `ValueError`.

`lag_world(innovations, alpha)` requires a finite `(N+1, 7)` matrix, `N >= 1`,
and a finite scalar `alpha` in `[0, 1]`. It returns `(N, 7)`:

```text
x = innovations[1:].copy()
x[:, 1] = alpha * innovations[:-1, 0]
          + sqrt(1-alpha**2) * innovations[1:, 1]
```

The first innovation row is prehistory, not a retained sample. Scoring this
field still requires `N >= 18`. The function does not test whether supplied
innovations are Gaussian or independent.

## Private aggregate files

The JSON files are stored in the private
[clinical-response-data](https://github.com/yury-rd/clinical-response-data)
repository on its `data` branch. The public repository's `data/` submodule
pins a revision of that companion; fetching it requires authorized GitHub
access. The public source tree, wheels, and source distributions contain no
saved tables. See the [access instructions](local-data.md#export-private-aggregate-tables)
for submodule and external-directory workflows.

Summary mode returns the two external tables keyed by filename. Their values
are saved results, not recomputed estimates. Keys ending in `_pp` use
percentage points; `_percent` uses percent. Other measurement fields use
fractional units unless their name specifies another quantity.

### `aggregate_results.json`

| Top-level key | Contents |
|---|---|
| `clinical_units` | `C1` and `DCT` branch summaries; changes between them; a saved `frozen_reference_t` summary. |
| `mechanism_summaries` | Ten records: five worlds at each of `N=197` and `N=788`. Each has `N`, `world`, five `levels`, and `metrics`. |
| `population_functionals` | Five world names, each mapping to five level records with branch/oracle `rho2`, `reference`, `J`, and `delta_J`. Observation amplitude also includes `alternative_oracle`. |
| `scope` | Scope statement for the saved outputs. |

Each clinical branch records `slope_pp_per_CDRSB` and
`hc3_se_pp_per_CDRSB` for BASE, A5, and their paired contrast; `mean_J_percent`
for BASE and A5; `mean_delta_J_pp`; `decomposition_pp_per_CDRSB`; and phase
Monte Carlo diagnostics. `frozen_reference_t` contains its saved mean, sample
SD, draw count, exceedance count, and p-value. These records do not expose the
participant-level fit or its complete model specification.

Mechanism metrics include `delta_J`, `delta_observed`, `delta_reference`,
`paired_reference_MCSE`, `signal_retention`, and `artifact_retention`; BASE,
A5, and oracle each have excess, observed-rho2, and reference-mean summaries.
Ordinary metric records follow the six-key `summarize_replicates` schema.
Undefined artifact-retention entries are JSON `null`, with
`defined_replicates` and `status_by_level` describing availability; they must
not be interpreted as zeros. Population functionals are not finite-sample
expectations of `J`.

### `lag_aggregate.json`

| Top-level key | Contents |
|---|---|
| `cells` | Six presentation records, with `variant`, `N`, `alpha`, `J_pp`, and `simulation_SE_pp`. |
| `paired_contrasts` | Three variant records across `N=[97,197,788]`, reporting paired alpha-1 minus alpha-0 changes and standard errors in percentage points. |
| `all_measurement_cells` | All 27 combinations of three lengths, alpha values 0/0.5/1, and variants `noncircular_L10`, `circular_L10`, `noncircular_L1`. |
| `precision` | Saved precision target, observed simulation SE, Boolean `met`, and consequence if unmet. |
| `boundary_diagnostics` | `maximum_recorded_check_values` and nine `by_cell` records with boundary-energy and common-phase-change diagnostics. |
| `scope` | Scope statement for the saved outputs. |

Each complete measurement cell has `N`, `alpha`, `variant`, `original`,
`common_phase_change`, and `mean_conditional_reference_MCSE`. The two nested
measurement dictionaries contain `observed_rho2`, `reference_mean`, and `J`;
each metric records `mean`, `simulation_SD`, `simulation_SE`, and `replicates`
(64). These nested measurements are fractional; the presentation fields
ending in `_pp` have already been multiplied by 100.

The runnable score API implements `noncircular_L10`. The other variants,
clinical fits, and population functionals are available as private saved
aggregates only. See [methods](methods.md#runnable-computations-and-saved-results) for
reproduction scope.
