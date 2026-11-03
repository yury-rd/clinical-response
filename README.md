# Clinical response to nuisance regression

Code accompanying **Dementia Severity Is Associated with Resting-State fMRI Network Responses to Nuisance Regression**, by Ildar Gabitov, Aleksandr Volkov, Tamoghna Chattopadhyay, Paul M. Thompson, and Yury Maximov.

This paper is a submission to the **[2027 IEEE International Symposium on Biomedical Imaging (ISBI 2027)](https://biomedicalimaging.org/2027/)**.

This package measures how a seven-network resting-state fMRI representation changes when five anatomical nuisance columns are added to a joint nuisance-regression design. The public repository contains the measurement code and controlled synthetic examples. Saved clinical and simulation aggregates are held in the private [clinical-response-data](https://github.com/yury-rd/clinical-response-data) companion repository.

The paired response is **ΔJ = J(A5) − J(BASE)**. Each J is the leading-two energy fraction of a length-10 joint delay matrix minus its mean over 199 independent-channel phase references. Measurements are fractions; multiply by 100 to express differences in percentage points.

## Quick start

Requires Python 3.11 or newer. From a terminal:

```sh
git clone https://github.com/yury-rd/clinical-response.git
cd clinical-response
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m pip install --no-deps -e .
clinical-response --output results/demo.json
```

On Windows, activate with `.venv\Scripts\activate`. The example writes two nuisance-regression mechanism settings and three delayed-coupling settings to `results/demo.json`. Its generators and phase seeds are fixed. NumPy and `threadpoolctl` are the only runtime dependencies.

`python -m clinical_response` and `python run.py` accept the same flags. The installed command works from any directory. The synthetic demo needs no private data access.

## Access saved results

The `data/` Git submodule points to the private companion. Access requires GitHub authentication and permission to read that repository. From the public repository root, authorized users can fetch the pinned data revision and export the saved tables:

```sh
git submodule update --init data
clinical-response --mode summary --output results/aggregates.json
```

Alternatively, keep the private data checkout elsewhere:

```sh
git clone --branch data https://github.com/yury-rd/clinical-response-data.git ../clinical-response-data
clinical-response --mode summary --data-dir ../clinical-response-data --output results/aggregates.json
```

The data directory must contain `aggregate_results.json` and `lag_aggregate.json`. Resolution order is `--data-dir`, then the `CLINICAL_RESPONSE_DATA_DIR` environment variable, then `data/` beneath the current working directory. Summary mode reports a missing-data error if the files are unavailable. Data files are excluded from public source distributions and wheels; there is no bundled fallback. Exported summaries contain private results and should remain in authorized storage.

## Measure prepared local data

```sh
clinical-response --input local-data/subject.npz --output results/subject.json
```

The NPZ must contain `parcels` (N × 400), `motion24` (N × 24), `a5` (N × 5), `cosines` (N × K), and `membership` (400 integer labels covering 0–6). All series must use the same retained frames. See [How to prepare and score local data](docs/local-data.md) for a runnable synthetic NPZ example, preprocessing responsibilities, and error handling.

## What can be reproduced

| Component | Scope |
|---|---|
| Paired measurement | Joint projection, seven-network aggregation, length-10 delay energy, 199 phase references, and numerical diagnostics |
| Five mechanism generators | Null, observation amplitude, proxy quality, latent sharing, and proxy contamination |
| Delayed-coupling generator | Noncircular Gaussian coupling with independent prehistory |
| Private saved aggregates | Clinical estimates, the 800-condition mechanism panel, population functionals, and lag-control summaries; authorized companion access required |

The default demo runs a small selection of synthetic settings. With authorized data available, `--mode summary` reads the saved tables; it does not recompute them. Participant-level clinical fitting, bootstrap analysis, anatomical component selection, raw-image preprocessing, and full-panel generation scripts are outside this package. The private circular-delay, length-1, and population-functional sensitivity results do not have corresponding regeneration routines here.

ADNI participant records and participant-level derivatives are not distributed. Obtain authorized data through [ADNI](https://adni.loni.usc.edu/) before attempting a clinical analysis. Local examples use common phase seeds for the two branches; historical clinical estimates used their saved branch-specific seeds. Thus the local workflow demonstrates the measurement and does not replay the clinical estimates.

## Documentation

- [Reference](docs/reference.md): commands, Python functions, input/output schemas, and saved tables.
- [Methods and interpretation](docs/methods.md): mathematical conventions, projection checks, phase references, and limitations.
- [How to prepare and score local data](docs/local-data.md): preparation, scoring, and exporting results.
- [Citation](CITATION.md): the associated paper and a BibTeX entry.

## Development and verification

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check .
ruff format --check .
python -m build
```

Tests compare projection with a direct least-squares calculation, delay energy with an SVD calculation, and phase references with Fourier-magnitude invariants. They also exercise the CLI's external-data loading and missing-data handling using synthetic fixtures. The public test suite requires no private data. Numerical calculations use float64; the projection identity threshold is `1e-10`.

A [GitHub Actions template](.github/ci-template.yml) covers Python 3.11 and 3.14 and checks an installed wheel outside the checkout. To enable it, copy the template to `.github/workflows/ci.yml` and push with credentials permitted to update workflows. The template is inactive until installed there.

Keep input data in the ignored `local-data/` directory and generated outputs in `results/`. Commands read external tables without modifying them. Do not add private results or initialized submodule contents to the public repository or its release assets. Public code and synthetic examples use the [MIT license](LICENSE); data access and reuse must follow the terms in the private companion and any applicable ADNI agreement.
