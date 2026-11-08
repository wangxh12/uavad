"""Prepare selected ALFA flights with chronological within-flight splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.alfa_within_flight import prepare_within_flight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/alfa_20hz_l_18f")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/alfa_within_flight_v1")
    parser.add_argument("--flights-file", type=Path,
                        help="One exact CSV filename per line, in concatenation order; defaults to all labeled flights")
    parser.add_argument("--train-fraction", type=float, default=0.6,
                        help="Fraction of each NORMAL PREFIX assigned to training")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of each NORMAL PREFIX assigned to validation")
    parser.add_argument("--normalization", choices=("zscore", "minmax"), default="zscore")
    parser.add_argument("--window-size", type=int, default=96,
                        help="Minimum segment size and expected window for the saved coverage audit")
    args = parser.parse_args()
    names = None
    if args.flights_file is not None:
        names = [line.strip() for line in args.flights_file.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    manifest = prepare_within_flight(
        args.raw_dir, args.output_dir, selected_names=names,
        train_fraction=args.train_fraction, val_fraction=args.val_fraction,
        normalization=args.normalization, window=args.window_size)
    print(json.dumps({"output": str(args.output_dir.resolve()), "protocol": manifest["protocol"],
                      "selected_flights": len(manifest["flights"]), "excluded": manifest["excluded"],
                      "normalization": manifest["normalization"], "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
