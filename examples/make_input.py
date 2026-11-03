"""Create a synthetic prepared NPZ; this does not contain participant data."""

import argparse
from pathlib import Path

import numpy as np

from clinical_response import mechanisms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("local-data/synthetic.npz"))
    args = parser.parse_args()
    bank = mechanisms.bank(197, 202619000000)
    world = mechanisms.world(bank, "observation_amplitude", 2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Use a file handle so NumPy honors the exact supplied filename.
    with args.output.open("wb") as output:
        np.savez_compressed(
            output,
            parcels=world["parcels"],
            motion24=bank["m24"],
            a5=world["proxies"],
            cosines=np.empty((197, 0)),
            membership=bank["membership"],
        )
    print(f"Wrote synthetic input to {args.output}")


if __name__ == "__main__":
    main()
