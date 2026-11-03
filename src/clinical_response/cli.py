"""Run synthetic examples, score a prepared local NPZ, or export saved aggregates."""

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import __version__, lag_control, mechanisms, paired, score

DEFAULT_SEED = 202620000000
GENERATOR_SEED = 202619000000
INPUT_KEYS = ("parcels", "motion24", "a5", "cosines", "membership")
AGGREGATE_FILES = ("aggregate_results.json", "lag_aggregate.json")


def _aggregate_directory(data_dir=None):
    """Resolve an external private-data directory, never package resources."""
    return Path(data_dir or os.environ.get("CLINICAL_RESPONSE_DATA_DIR") or Path.cwd() / "data")


def load_aggregates(data_dir=None):
    """Read saved tables from an authorized checkout of the private data repo."""
    data = _aggregate_directory(data_dir)
    missing = [name for name in AGGREGATE_FILES if not (data / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Private aggregate data not found in {data}: {', '.join(missing)}. "
            "With access to yury-rd/clinical-response-data, run "
            "'git submodule update --init data' from the code checkout, "
            "or provide --data-dir PATH / CLINICAL_RESPONSE_DATA_DIR. "
            "The synthetic demo works without private data."
        )
    return {name: json.loads((data / name).read_text(encoding="utf-8")) for name in AGGREGATE_FILES}


def measure_input(path, *, seed_start=DEFAULT_SEED, batch_size=16):
    """Score a prepared NPZ with shared phase seeds in both nuisance branches."""
    loaded = np.load(path, allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError("Input must be an NPZ archive with named arrays")
    with loaded as arrays:
        missing = sorted(set(INPUT_KEYS) - set(arrays.files))
        if missing:
            raise ValueError("Input NPZ is missing arrays: " + ", ".join(missing))
        fields, diagnostics = paired.paired_fields(*(arrays[key] for key in INPUT_KEYS))
    scores = {
        name: score.fullscore(fields[name + "_DCT"], seed_start=seed_start, batch_size=batch_size)
        for name in ("BASE", "A5")
    }
    return {
        "scope": "Local paired measurement; no clinical model is fitted",
        **scores,
        "delta_J": scores["A5"]["excess"] - scores["BASE"]["excess"],
        "projection": diagnostics,
    }


def run_demo(*, seed_start=DEFAULT_SEED, batch_size=16):
    """Run two mechanism settings and three delayed-coupling settings."""
    bank = mechanisms.bank(197, GENERATOR_SEED)
    worlds = []
    for name, level in (("observation_amplitude", 2.0), ("proxy_contamination", 4.0)):
        world = mechanisms.world(bank, name, level)
        scores = {
            key: score.fullscore(value, seed_start=seed_start, batch_size=batch_size)
            for key, value in world["fields"].items()
        }
        worlds.append(
            {
                "world": name,
                "level": level,
                "BASE_J": scores["BASE"]["excess"],
                "A5_J": scores["A5"]["excess"],
                "delta_J": scores["A5"]["excess"] - scores["BASE"]["excess"],
                "diagnostics": world["diagnostics"],
            }
        )
    innovations = np.random.default_rng(GENERATOR_SEED).standard_normal((198, 7))
    lag = [
        {
            "alpha": alpha,
            "J": score.fullscore(
                lag_control.lag_world(innovations, alpha),
                seed_start=seed_start,
                batch_size=batch_size,
            )["excess"],
        }
        for alpha in (0.0, 0.5, 1.0)
    ]
    return {
        "scope": "Known-world software examples; see saved aggregates for full panels",
        "phase_references": 199,
        "worlds": worlds,
        "lag_control": lag,
    }


def _integer_at_least(minimum):
    def parse(value):
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be an integer") from exc
        if number < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}")
        return number

    return parse


def main(argv=None):
    """Parse CLI arguments and write strict JSON; return zero on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--mode",
        choices=("demo", "summary"),
        default="demo",
        help="run examples (default) or export saved aggregate results",
    )
    parser.add_argument(
        "--input", type=Path, help="prepared local NPZ; replaces the synthetic demo"
    )
    parser.add_argument("--output", type=Path, help="write JSON here instead of standard output")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="private aggregate directory for summary mode (env CLINICAL_RESPONSE_DATA_DIR or ./data)",
    )
    parser.add_argument(
        "--seed",
        type=_integer_at_least(0),
        default=DEFAULT_SEED,
        help=f"first phase-reference seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--batch-size",
        type=_integer_at_least(1),
        default=16,
        help="phase references processed together (default: 16)",
    )
    args = parser.parse_args(argv)
    if args.mode == "summary" and args.input is not None:
        parser.error("--input cannot be combined with --mode summary")
    if args.data_dir is not None and args.mode != "summary":
        parser.error("--data-dir is only used with --mode summary")
    if args.input and args.output and args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input")
    if (
        args.mode == "summary"
        and args.output
        and args.output.resolve()
        in {(_aggregate_directory(args.data_dir) / name).resolve() for name in AGGREGATE_FILES}
    ):
        parser.error("--output must not overwrite private source tables")
    try:
        with threadpool_limits(limits=1):
            if args.mode == "summary":
                result = load_aggregates(args.data_dir)
            elif args.input is not None:
                result = measure_input(args.input, seed_start=args.seed, batch_size=args.batch_size)
            else:
                result = run_demo(seed_start=args.seed, batch_size=args.batch_size)
            if args.mode != "summary":
                result["runtime"] = {
                    "package_version": __version__,
                    "numpy_version": np.__version__,
                    "phase_seed_start": args.seed,
                    "batch_size": args.batch_size,
                    "units": "fraction; multiply by 100 for percentage points",
                    "phase_pairing": "shared seeds across BASE and A5",
                }
            payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
            print(json.dumps({"output": str(args.output), "mode": args.mode}))
        else:
            print(payload, end="")
    except (OSError, ValueError, ArithmeticError, EOFError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    return 0
