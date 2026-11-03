# Measurement and interpretation

This package accompanies *Dementia Severity Is Associated with Resting-State
fMRI Network Responses to Nuisance Regression* by Ildar Gabitov, Aleksandr
Volkov, Tamoghna Chattopadhyay, Paul M. Thompson, and Yury Maximov. It computes
a paired response to nuisance regression from prepared parcel time series and
provides controlled simulations. Saved aggregate results are stored separately
in the private [clinical-response-data](https://github.com/yury-rd/clinical-response-data)
companion repository.

The paper is a submission to the **2027 IEEE International Symposium on
Biomedical Imaging (ISBI 2027)**.

The measurement compares two processing branches on the same observations.
It does not estimate dementia severity from a new scan. Participant-level
clinical fitting, anatomical component selection, and raw-image preprocessing
are outside this package.

## From parcels to a paired measurement

```text
Prepared N × 400 parcels, same retained frames in every input
                     |
       +-------------+----------------+
       |                              |
BASE nuisance design            A5 nuisance design
intercept + trend + M24          BASE + five A5 columns
+ saved cosine columns          (same cosine columns)
       |                              |
joint orthogonal projection     joint orthogonal projection
       |                              |
fixed seven-network means       fixed seven-network means
       |                              |
center and scale each network to unit RMS
       |                              |
length-10 joint delay matrix; leading-two energy fraction
       |                              |
subtract mean of 199 independent-channel phase references
       |                              |
     J_BASE                          J_A5
       +------------ delta_J = J_A5 - J_BASE
```

### Shared support and joint projection

All arrays supplied to `paired_fields` must already have the same retained
frames in the same order. Equal row counts alone cannot establish alignment;
preparation of the input is the caller's responsibility. The helper
`normalize_mask` accepts only a retained suffix with at least 18 frames, so it
does not treat a time series with internal gaps as a regularly sampled series.
It validates a mask; it does not apply one automatically.

The common design contains a trend spanning −1 to 1, 24 supplied motion
columns, and the supplied cosine columns. The A5 design adds five supplied
anatomical nuisance columns. An intercept enters both designs. Cosine columns
must come from the intended preprocessing design on the original time grid;
the package does not recreate them after frame removal.

For an orthonormal basis `U` of a design's column span, the residual is

```text
P X = X − U (Uᵀ X).
```

One joint projection removes all nuisance columns together. Applying several
separate regressions in sequence generally gives a different result unless
their projectors commute. Centering and column normalization before the SVD
make the rank calculation less sensitive to nuisance-column units. Numerical
rank must agree at tolerance multipliers 0.1, 1, 10, and 100. The code rejects
ambiguous rank and designs that leave no residual degrees of freedom.

The A5 nuisance span contains the BASE span. If `P` and `Q` denote their
residual projectors, the implementation checks symmetry, idempotence, and
`PQ = QP = Q` to relative tolerance `1e-10`. It also checks

```text
‖P X‖²_F = ‖Q X‖²_F + ‖(P − Q) X‖²_F.
```

These are floating-point numerical checks, not formal proofs. They establish
properties of the supplied designs; they do not establish that a nuisance
column represents only unwanted signal.

### Fixed network aggregation

Each parcel has one integer network label from 0 through 6. Every network must
be represented. For each frame, the code takes the unweighted mean of the
residual parcels in each network, in increasing label order. The same mapping
is used for both branches. Because the mapping is fixed and linear, averaging
then projecting agrees with projecting then averaging; the implementation
checks this identity numerically.

The simulator includes one fixed 400-parcel mapping with network counts
`[52, 91, 46, 26, 47, 77, 61]`, corresponding to `Cont`, `Default`, `DorsAttn`,
`Limbic`, `SalVentAttn`, `SomMot`, and `Vis`. Local data must supply a mapping
that matches its own parcel order. A matching shape does not identify an atlas
or establish the correct ordering.

### Delay energy and the phase reference

For each network time series, the code subtracts its temporal mean and divides
by its root mean square (RMS). Constant channels cannot be scaled and are
rejected. This gives each network the same full-record RMS before delay
construction.

With seven channels and delay length `L = 10`, the matrix `H` has 70 rows and
`N − 9` columns. Its rows are arranged by channel, then by lag:

```text
H[10 c + ℓ, t] = normalized_field[t + ℓ, c]
c = 0,…,6; ℓ = 0,…,9; t = 0,…,N−10.
```

Windows do not wrap across the record boundary, and rows of `H` are not
recentered. The leading-two singular-energy fraction is

```text
rho2 = (σ₁(H)² + σ₂(H)²) / ‖H‖²_F.
```

The implementation obtains the numerator from the two largest eigenvalues
of `H Hᵀ`. `rho2` is dimensionless. Normalization makes it invariant to a
nonzero constant rescaling of an individual channel, up to floating-point
effects, provided the rescaled channel remains above the RMS rejection
threshold.

Each phase reference retains the magnitude of every channel's real Fourier
coefficients and adds independent uniform phases for each channel and
frequency. DC is unchanged; the Nyquist coefficient is unchanged for even
record lengths. This preserves each channel's marginal spectrum to numerical
precision while changing relationships between channels. It does not preserve
cross-spectra or the exact empirical distribution of time-domain amplitudes.

Every reference is centered, RMS-scaled, and measured by the same delay
procedure. With exactly 199 references,

```text
J = observed_rho2 − mean(reference_rho2)
delta_J = J_A5 − J_BASE.
```

The API and demo report fractional units. Multiply `J` by 100 for percent and
`delta_J` by 100 for percentage points. `J` may be negative. The reported
Monte Carlo standard error is the sample standard deviation of the 199
reference scores divided by `sqrt(199)`. It describes finite-reference
uncertainty conditional on one observed field; it is not a participant-level
standard error, confidence interval, or p-value.

The local paired command uses the same phase seeds in both branches. This
allows a paired comparison of reference scores. The saved human analyses used
their saved branch-specific seeds; running the local command does not replay
their clinical bootstrap or regression calculations.

### Finite-record boundaries matter

Full-record RMS is preserved by phase shifts, but a noncircular delay matrix
weights interior samples more often than boundary samples. Its energy can
therefore change under a phase shift. The score includes total and per-channel
delay-energy diagnostics for this reason. Equal marginal spectra do not imply
equal noncircular delay matrices.

The private saved lag results include circular length-10 and noncircular
length-1 controls. Those controls are recorded aggregates; the runnable score API uses
the specified noncircular length-10 measurement.

## What the controlled simulations test

`mechanisms.bank` generates independent Gaussian innovation streams with fixed
seeds. AR(1) filtering gives the latent and nuisance streams temporal
correlation. A shared bank across levels holds the random realization fixed
while changing a mechanism parameter. The five worlds vary:

| World | What changes with the level |
|---|---|
| `null` | Nothing; repeated levels give an internal negative control. |
| `observation_amplitude` | The amplitude of a common observation artifact. |
| `proxy_quality` | The proxy's mixture of the true artifact and independent noise. |
| `latent_sharing` | The fraction of a shared latent signal across networks. |
| `proxy_contamination` | The amount of latent signal present in a nuisance proxy. |

The simulation knows which components it called signal and artifact, so it can
report their residual energies separately. The observation alone does not
provide those labels. In particular, a changed `delta_J` does not by itself
identify improved denoising, preservation of neuronal signal, or a unique
biological mechanism. Removing nuisance directions always reduces raw parcel
residual energy, but need not reduce the normalized, phase-referenced score.

`lag_control.lag_world` constructs a seven-channel Gaussian field in which the
second channel mixes its own innovation with the previous innovation of the
first channel. The extra initial innovation row supplies independent
prehistory, avoiding wraparound from the last observed sample to the first.
Changing the coupling strength probes sensitivity to lagged cross-channel
dependence without introducing a hidden-state model.

## Runnable computations and saved results

The demo runs two mechanism settings at `N = 197` and three lag-coupling
settings. The Python API allows other mechanism levels and lengths. The
private companion's JSON files retain larger completed analyses:

- Mechanism summaries cover two lengths, five worlds, five levels, and 16
  replicates: 800 conditions.
- Lag summaries retain all 27 combinations of three lengths, three coupling
  strengths, and three measurement variants, each summarized over 64 banks.
- Clinical summaries include the saved `C1` and common-`DCT` results, with
  slopes in percentage points per CDR-SB unit.
- Population functionals are saved theoretical summaries. They are not the
  expected value of the finite-record estimator `J`.

Authorized users can load those files with `--mode summary --data-dir PATH`,
through the `CLINICAL_RESPONSE_DATA_DIR` environment variable, or from an
initialized `data/` submodule beneath the working directory. The command
reads the aggregates; it does not refit or regenerate them. The public code,
synthetic demo, and tests run without the private tables, which are excluded
from source distributions and wheels.

The package has no participant records, clinical-model implementation, full
panel execution plan, circular/L1 scoring implementation, or population
functional solver. Reproducing the clinical inference requires authorized
participant data and the study's participant-level analysis protocol beyond
this release.

See the [reference](reference.md) for precise API contracts and saved-data
schemas, the [local-data guide](local-data.md) for input preparation, and the
[README](../README.md) for installation and a first run.
