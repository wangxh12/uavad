"""Build ALFA 60/20/20 forecast splits and two synthetic test datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "datasets/processed/alfa_20hz_linear_18features"
DEFAULT_OUTPUT = ROOT / "datasets/ALFA/forecast"
DATASETS = (
    ("alfa_20hz_roll_bias8_v1", "field.measured.roll", 8.0),
    ("alfa_20hz_pitch_bias1p58_v1", "field.measured.pitch", 1.58),
)
SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_normal_sequences(source_dir: Path) -> tuple[list[str], list[dict]]:
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = [
        item for item in source_manifest["sequences"]
        if int(item["label_counts"].get("0", 0)) == int(item["rows"])
        and int(item["rows"]) > 0
    ]
    if not entries:
        raise ValueError(f"No fully normal sequences found in {source_dir}")
    columns, selected = source_manifest["columns"], []
    for entry in entries:
        path = source_dir / entry["output_file"]
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != columns:
                raise ValueError(f"Unexpected columns in {path}")
            rows = [row for row in reader if row]
        if len(rows) != int(entry["rows"]) or any(row[-1] != "0" for row in rows):
            raise ValueError(f"Row count or labels do not match manifest in {path}")
        selected.append({"name": entry["output_file"], "rows": rows,
                         "sha256": sha256_file(path)})
    return columns, selected


def split_sequences(sequences: list[dict]) -> tuple[dict, dict]:
    rows_by_split = {split: [] for split in SPLITS}
    segments = {split: [] for split in SPLITS}
    for sequence in sequences:
        rows = sequence["rows"]
        train_end, val_end = round(0.6 * len(rows)), round(0.8 * len(rows))
        for split, source_start, source_end in (
            ("train", 0, train_end), ("val", train_end, val_end),
            ("test", val_end, len(rows)),
        ):
            concat_start = len(rows_by_split[split])
            rows_by_split[split].extend(rows[source_start:source_end])
            segments[split].append({
                "file": sequence["name"],
                "source_rows_0based_half_open": [source_start, source_end],
                "concatenated_rows_0based_half_open": [concat_start, len(rows_by_split[split])],
            })
    return rows_by_split, segments


def build_dataset(output_dir: Path, name: str, columns: list[str], rows_by_split: dict,
                  segments: dict, sequences: list[dict], source_dir: Path,
                  target: str, bias: float) -> None:
    target_index, label_index = columns.index(target), columns.index("label")
    split_rows = {split: [row.copy() for row in rows_by_split[split]] for split in SPLITS}
    test_rows = split_rows["test"]
    if len(test_rows) <= 1200:
        raise ValueError("The concatenated test split must contain more than 1200 rows")
    anomaly_start = len(test_rows) - 1200
    for row in test_rows[anomaly_start:]:
        row[target_index] = format(float(row[target_index]) + bias, ".12g")
        row[label_index] = "1"

    output_dir.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        with (output_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(split_rows[split])

    test_segments = []
    for segment in segments["test"]:
        start, end = segment["concatenated_rows_0based_half_open"]
        test_segments.append({**segment, "anomaly_rows": max(0, end - max(start, anomaly_start))})
    manifest = {
        "dataset": name,
        "source_directory": str(source_dir.relative_to(ROOT)),
        "source_sampling_hz": 20,
        "columns": columns,
        "concatenation_order": "source manifest order (chronological)",
        "split_rule": "per-file chronological 60/20/20; boundaries use Python round",
        "normal_flights": [
            {"file": seq["name"], "rows": len(seq["rows"]), "sha256": seq["sha256"]}
            for seq in sequences
        ],
        "splits": {
            split: {
                "rows": len(split_rows[split]),
                "label_counts": {
                    "0": len(split_rows[split]) if split != "test" else anomaly_start,
                    "1": 1200 if split == "test" else 0,
                },
                "segments": test_segments if split == "test" else segments[split],
            }
            for split in SPLITS
        },
        "injection": {
            "split": "test", "target": target, "type": "constant additive bias",
            "bias": bias, "start_row_0based": anomaly_start,
            "start_row_1based": anomaly_start + 1,
            "end_row_1based_inclusive": len(test_rows), "points": 1200,
            "label": 1, "all_other_features_unchanged": True,
        },
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source_dir, output_root = args.source_dir.resolve(), args.output_root.resolve()
    if not (source_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing source manifest in {source_dir}")
    destinations = [output_root / item[0] for item in DATASETS]
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))

    columns, sequences = read_normal_sequences(source_dir)
    if columns[0] != "time_s" or columns[-1] != "label" or len(columns) != 20:
        raise ValueError("Expected time_s, 18 features, and label in that order")
    rows_by_split, segments = split_sequences(sequences)
    if [len(rows_by_split[s]) for s in SPLITS] != [6679, 2227, 2225]:
        raise ValueError("Unexpected split totals; inspect source inventory and split rule")

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".alfa_forecast_build_", dir=output_root) as temp:
        temp_root = Path(temp)
        for name, target, bias in DATASETS:
            build_dataset(temp_root / name, name, columns, rows_by_split, segments,
                          sequences, source_dir, target, bias)
        for (name, _, _), destination in zip(DATASETS, destinations):
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
            os.replace(temp_root / name, destination)

    print(json.dumps({
        "output_root": str(output_root), "source_files": len(sequences),
        "source_points": sum(len(seq["rows"]) for seq in sequences),
        "datasets": [{"path": str(path), "train": 6679, "val": 2227,
                      "test": 2225, "injected_points": 1200} for path in destinations],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
