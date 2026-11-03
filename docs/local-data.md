# How to prepare and score local data

This guide creates an NPZ example and measures its paired nuisance-regression response. It also explains how authorized users can load the saved aggregate tables from the private data companion. Install the package using the [README](../README.md) first. The synthetic example contains no participant data and runs without private repository access.

## Create and score a working example

From the repository root with your virtual environment active:

```sh
python examples/make_input.py --output local-data/synthetic.npz
clinical-response --input local-data/synthetic.npz --output results/local.json
python -c 'import json; d=json.load(open("results/local.json")); print("Delta J (pp):", 100*d["delta_J"])'
```

The JSON contains the complete BASE and A5 scores, each with 199 reference scores, plus `delta_J` and joint-projection diagnostics. Check that `effective_A5_rank` is 5 for this example and the projection errors are below `1e-10`:

```sh
python - <<'PY'
import json
d = json.load(open("results/local.json"))
assert d["projection"]["effective_A5_rank"] == 5
assert max(d["projection"]["projection_errors"].values()) < 1e-10
assert len(d["BASE"]["surrogate_rho2"]) == 199
assert len(d["A5"]["surrogate_rho2"]) == 199
print("Paired measurement verified")
PY
```

## Prepare your own input

Save these five arrays together with `numpy.savez_compressed`. Arrays must be real, finite, and aligned to exactly the same N retained frames. Membership is an integer vector; the time-series arrays are converted to float64.

| Key | Shape | Preparation responsibility |
|---|---|---|
| `parcels` | N × 400 | Prepared parcel series in the order of your membership vector |
| `motion24` | N × 24 | The fixed motion design selected by your protocol |
| `a5` | N × 5 | Five anatomical nuisance columns selected before this measurement |
| `cosines` | N × K | Saved cosine columns, restricted to the same frames |
| `membership` | 400 | Labels 0 through 6, with every label represented |

N must be at least 18, but both nuisance designs must also leave positive residual degrees of freedom. Full-rank motion and anatomical designs therefore require more rows than this bare minimum. Network residuals must have nonzero centered RMS for scoring.

Keep the acquired frame spacing in the retained series. The API cannot detect missing internal time points from an already shortened matrix. For a protocol retaining a suffix of an acquisition, `paired.normalize_mask(mask, nraw)` validates a Boolean or exact 0/1 indicator with at least 18 retained rows and no internal holes; apply that same mask to every time-series array before saving.

Do not regenerate cosine columns after discarding frames. Both nuisance branches receive the same saved columns and are projected jointly with an intercept, a trend, and motion. An N × 0 cosine array omits the common DCT regressors, as in the synthetic example. See the [methods](methods.md) for the distinction between this example and the saved common-DCT sensitivity analysis.

The package does not load raw MRI images, extract parcels, select anatomical components, align clinical records, or fit clinical associations. Supply those preprocessing and analysis steps from your study protocol. The seven-network label ordering used by the synthetic generator is documented in the [reference](reference.md); do not reuse its mapping with a different parcel ordering.

## Control random references

```sh
clinical-response --input local-data/synthetic.npz --seed 202620000000 --batch-size 8 --output results/local-batch8.json
```

`--seed` sets the first of 199 consecutive phase-reference seeds. Both branches use that sequence. `--batch-size` changes the number evaluated together, not the definition or number of references. Default runs limit numerical library thread pools to one thread. Output records the package and NumPy versions, phase seed, and batch size. Changing the seed changes Monte Carlo estimates; changing the batch size should agree to floating-point tolerance.

## Export private aggregate tables

You need GitHub authentication and read permission for
[clinical-response-data](https://github.com/yury-rd/clinical-response-data).
From the public repository root, initialize its `data/` submodule:

```sh
git submodule update --init data
clinical-response --mode summary --output results/aggregates.json
```

To keep data in a separate directory instead, clone the companion's `data`
branch and pass its location explicitly:

```sh
git clone --branch data https://github.com/yury-rd/clinical-response-data.git ../clinical-response-data
clinical-response --mode summary --data-dir ../clinical-response-data --output results/aggregates.json
```

For repeated use from different working directories, set
`CLINICAL_RESPONSE_DATA_DIR` to the absolute path of that data checkout.
An explicit `--data-dir` takes precedence over the environment variable.
Without either setting, the command looks for `data/` under the current
working directory. It never falls back to packaged tables.

The directory must contain `aggregate_results.json` and `lag_aggregate.json`.
Summary mode exports them keyed by filename and performs no participant
fitting or simulation. Check the export locally:

```sh
python - <<'PY'
import json
d = json.load(open("results/aggregates.json"))
assert set(d) == {"aggregate_results.json", "lag_aggregate.json"}
print("Both aggregate tables exported")
PY
```

Read the [aggregate schema](reference.md) before combining fields: `_pp`
fields are percentage points while live score outputs are fractions. The
export contains private results. Keep it in authorized storage and do not
publish it as a public repository file, release asset, or CI artifact.

## Troubleshooting

| Error or symptom | Action |
|---|---|
| `clinical-response: command not found` | Activate the environment used to install the package; try `python -m clinical_response --help`. |
| Missing arrays or wrong widths | Check exact NPZ keys and dimensions in the table above. Save membership with integer dtype. |
| Nonfinite input | Resolve missing/invalid values under your preprocessing protocol; do not discard frames independently by array. |
| Rank ambiguity | Inspect duplicate or nearly dependent nuisance columns. The projector checks rank across four fixed tolerance multipliers. |
| No residual degrees of freedom | Check the number of independent nuisance columns and retained rows. |
| Constant channel | Inspect network residuals; centered unit-RMS normalization cannot score a constant channel. |
| `--input` with `--mode summary` | Use separate commands. Summary reads saved tables and accepts no local input. |
| Missing aggregate tables | Initialize the private `data/` submodule with authorized credentials, or pass `--data-dir` pointing to both JSON files. Wheels and source distributions contain no saved tables. |
| Private repository not found or access denied | Check the GitHub identity used by Git and obtain read access to `clinical-response-data`; public code access does not grant data access. |
| `--data-dir` outside summary mode | Use `--data-dir` only with `--mode summary`; use `--input` for a prepared local NPZ. |
| Output path error | Choose a writable directory and a path different from the input file. Existing output files are overwritten. |

## What you built

You created a prepared synthetic input and measured the paired response with complete phase-reference diagnostics. For a study analysis, replace the synthetic NPZ with authorized, consistently prepared data, then use the [reference](reference.md) and [methods](methods.md) to interpret the outputs.
