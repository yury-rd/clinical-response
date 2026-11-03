"""Exercise public entry points and external-data loading with generated fixtures."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from clinical_response import mechanisms
from clinical_response.cli import load_aggregates, main, measure_input, run_demo

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def local_input(tmp_path):
    bank = mechanisms.bank(197, 202619000000)
    world = mechanisms.world(bank, "observation_amplitude", 2.0)
    path = tmp_path / "input.npz"
    np.savez(
        path,
        parcels=world["parcels"],
        motion24=bank["m24"],
        a5=world["proxies"],
        cosines=np.empty((197, 0)),
        membership=bank["membership"],
    )
    return path


def test_local_input_matches_demo_and_preserves_data(local_input, tmp_path, capsys):
    before = local_input.read_bytes()
    output = tmp_path / "new-directory" / "measurement.json"
    assert main(["--input", str(local_input), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    demo = run_demo()
    assert result["delta_J"] == pytest.approx(demo["worlds"][0]["delta_J"], abs=1e-12)
    assert result["delta_J"] == result["A5"]["excess"] - result["BASE"]["excess"]
    assert result["projection"]["effective_A5_rank"] == 5
    assert result["runtime"]["phase_seed_start"] == 202620000000
    assert len(result["BASE"]["surrogate_rho2"]) == 199
    assert local_input.read_bytes() == before
    assert json.loads(capsys.readouterr().out)["output"] == str(output)


def test_custom_phase_seed_propagates_to_both_branches(local_input):
    result = measure_input(local_input, seed_start=701, batch_size=8)
    assert result["BASE"]["seed_start"] == result["A5"]["seed_start"] == 701


@pytest.fixture
def external_tables(tmp_path):
    data = tmp_path / "private-checkout"
    data.mkdir()
    for name in ("aggregate_results.json", "lag_aggregate.json"):
        (data / name).write_text(json.dumps({"scope": "generated test fixture", "file": name}))
    return data


def test_aggregate_loader_reads_explicit_external_files(external_tables):
    result = load_aggregates(external_tables)
    assert set(result) == {"aggregate_results.json", "lag_aggregate.json"}
    for name, value in result.items():
        assert value == json.loads((external_tables / name).read_text())


def test_aggregate_loader_environment_and_explicit_precedence(
    external_tables, tmp_path, monkeypatch
):
    monkeypatch.setenv("CLINICAL_RESPONSE_DATA_DIR", str(external_tables))
    assert load_aggregates() == load_aggregates(external_tables)
    with pytest.raises(FileNotFoundError, match="Private aggregate data"):
        load_aggregates(tmp_path / "missing")


def test_missing_private_data_gives_actionable_cli_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CLINICAL_RESPONSE_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(["--mode", "summary"])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "git submodule update --init data" in message
    assert "synthetic demo works" in message


def test_module_summary_from_another_directory(tmp_path, external_tables):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "clinical_response",
            "--mode",
            "summary",
            "--data-dir",
            str(external_tables),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == load_aggregates(external_tables)


def test_summary_cannot_overwrite_private_source_table(external_tables, capsys):
    source = external_tables / "aggregate_results.json"
    before = source.read_bytes()
    with pytest.raises(SystemExit):
        main(["--mode", "summary", "--data-dir", str(external_tables), "--output", str(source)])
    assert source.read_bytes() == before
    assert "must not overwrite" in capsys.readouterr().err


def test_example_script_produces_valid_npz(tmp_path):
    output = tmp_path / "synthetic.npz"
    subprocess.run(
        [sys.executable, str(ROOT / "examples/make_input.py"), "--output", str(output)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    result = measure_input(output)
    assert result["projection"]["samples"] == 197


@pytest.mark.parametrize(
    "arguments",
    [
        ["--seed", "-1"],
        ["--seed", "0.5"],
        ["--batch-size", "0"],
        ["--mode", "unknown"],
        ["--mode", "summary", "--input", "unused.npz"],
        ["--data-dir", "unused"],
    ],
)
def test_invalid_arguments_fail_cleanly(arguments, capsys):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_cannot_overwrite_input(local_input, capsys):
    before = local_input.read_bytes()
    with pytest.raises(SystemExit):
        main(["--input", str(local_input), "--output", str(local_input)])
    assert local_input.read_bytes() == before
    assert "must differ" in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["missing", "missing_keys", "npy", "empty", "object", "bad_zip"])
def test_invalid_files_fail_with_actionable_errors(kind, tmp_path, capsys):
    path = tmp_path / "bad.npz"
    if kind == "missing_keys":
        np.savez(path, parcels=np.ones((18, 400)))
    elif kind == "npy":
        with path.open("wb") as output:
            np.save(output, np.ones((18, 400)))
    elif kind == "empty":
        path.touch()
    elif kind == "bad_zip":
        path.write_bytes(b"PK\x03\x04broken")
    elif kind == "object":
        np.savez(
            path,
            **{
                key: np.array([object()], dtype=object)
                for key in ("parcels", "motion24", "a5", "cosines", "membership")
            },
        )
    with pytest.raises(SystemExit) as error:
        main(["--input", str(path)])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "error:" in message
    assert "Traceback" not in message
