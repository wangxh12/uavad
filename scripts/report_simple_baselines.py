"""Align all held-out ALFA flights across the fixed CrossAD and simple baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts.compare_crossad_flights import read_scores, summarize
from scripts.crossad_alfa import ROOT, fault_type


RUNS = {
    "CrossAD": ROOT / "results/crossad_alfa/main",
    "Relation": ROOT / "results/crossad_alfa/relation_v1",
    "PCA-8": ROOT / "results/simple_alfa/pca8",
    "MLP-AE": ROOT / "results/simple_alfa/mlp_ae",
    "LSTM-AE": ROOT / "results/simple_alfa/lstm_ae",
}


def report(runs: dict[str, Path], manifest: Path) -> list[dict]:
    inventory = {Path(entry["name"]).stem: entry for entry in
                 json.loads(manifest.read_text(encoding="utf-8"))["flights"]
                 if entry["split"] == "test"}
    reference = None
    rows = []
    for name, directory in runs.items():
        flights = read_scores(directory)
        if flights.keys() != inventory.keys():
            raise ValueError(f"Incomplete or mismatched test flights in {directory}")
        if reference is None:
            reference = flights
        threshold = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))["threshold"]
        for flight in sorted(flights):
            data, entry = flights[flight], inventory[flight]
            if not (np.array_equal(data["labels"], reference[flight]["labels"])
                    and np.array_equal(data["time_s"], reference[flight]["time_s"])):
                raise ValueError(f"Labels or timestamps differ: {name}, {flight}")
            if len(data["scores"]) != entry["points"] or not np.isfinite(data["scores"]).all():
                raise ValueError(f"Incomplete or nonfinite scores: {name}, {flight}")
            values = summarize(data)
            anomaly = np.flatnonzero(data["labels"])
            onset = int(anomaly[0]) if len(anomaly) else len(data["labels"])
            alerts = np.flatnonzero(data["predictions"][onset:])
            delay = (float(data["time_s"][onset + alerts[0]] - data["time_s"][onset])
                     if len(alerts) else None)
            rows.append({"model": name, "flight": flight, "group": entry["group"],
                         "fault_type": fault_type(flight), "normal_points": onset,
                         "anomaly_points": int(len(anomaly)), "threshold": threshold,
                         **values, "first_alarm_delay_s": delay})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = report(RUNS, ROOT / "data/processed/alfa_crossad/manifest.json")
    output = ROOT / "results/simple_alfa"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "all_flights_comparison.csv", rows)
    selected = [row for row in rows if row["group"] == "2018-09-11-15-06-34"]
    write_csv(output / "three_rudder_flights.csv", selected)
    print(f"Verified {len(RUNS)} models x 16 complete held-out flights")
    for row in selected:
        print(f"{row['model']:8s} {row['flight'].replace('carbonZ_', ''):55s} "
              f"AUROC={row['auroc']:.3f} F1={row['f1']:.3f} "
              f"false_alarm_points={row['false_alarm_points']}")


if __name__ == "__main__":
    main()
