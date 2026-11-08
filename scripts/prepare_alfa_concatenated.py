"""Export ALFA splits as continuous, cross-flight windowed arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.alfa_concatenated import prepare_concatenated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        default=ROOT / "data/processed/alfa_within_flight_v1")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/processed/alfa_concatenated_v1")
    parser.add_argument("--window-size", type=int, default=96)
    args = parser.parse_args()
    manifest = prepare_concatenated(args.source_dir, args.output_dir, args.window_size)
    print(json.dumps({"output": str(args.output_dir.resolve()),
                      "selected_flights": len(manifest["selected_flights"]),
                      "excluded": manifest["excluded"], "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
