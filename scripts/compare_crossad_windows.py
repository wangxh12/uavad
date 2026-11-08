"""Compare saved point scores from two ALFA window-length experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def compare(baseline: Path, candidate: Path) -> list[dict]:
    old = {path.name: path for path in (baseline / "scores").glob("*.npz")}
    new = {path.name: path for path in (candidate / "scores").glob("*.npz")}
    if not old or old.keys() != new.keys():
        raise ValueError("Both runs must have the same nonempty set of scored flights")

    rows = []
    for name in sorted(old):
        with np.load(old[name]) as left, np.load(new[name]) as right:
            labels = left["labels"]
            if not np.array_equal(labels, right["labels"]) or not np.array_equal(
                left["time_s"], right["time_s"]
            ):
                raise ValueError(f"Mismatched point labels or timestamps in {name}")
            if any(not np.isfinite(data["scores"]).all() or len(data["scores"]) != len(labels)
                   for data in (left, right)):
                raise ValueError(f"Missing or nonfinite point scores in {name}")
            onset = int(np.flatnonzero(labels)[0]) if labels.any() else len(labels)
            row = {"flight": name.removesuffix(".npz") + ".csv",
                   "anomaly_points": int(labels.sum())}
            for prefix, data in (("baseline", left), ("candidate", right)):
                row[f"{prefix}_auroc"] = (float(roc_auc_score(labels, data["scores"]))
                                           if 0 < labels.sum() < len(labels) else None)
                row[f"{prefix}_auprc"] = (float(average_precision_score(labels, data["scores"]))
                                           if labels.any() else None)
                row[f"{prefix}_f1"] = float(f1_score(labels, data["predictions"], zero_division=0))
                row[f"{prefix}_false_alarm_points"] = int(data["predictions"][:onset].sum())
            for metric in ("auroc", "auprc", "f1", "false_alarm_points"):
                first, second = row[f"baseline_{metric}"], row[f"candidate_{metric}"]
                row[f"delta_{metric}"] = second - first if first is not None else None
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = compare(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    faults = [row for row in rows if row["anomaly_points"]]
    print(f"Compared {len(rows)} flights ({len(faults)} fault flights); "
          f"AUROC improved on {sum(row['delta_auroc'] > 0 for row in faults)} flights")


if __name__ == "__main__":
    main()
