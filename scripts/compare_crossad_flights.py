"""Compare every held-out ALFA flight against the unchanged CrossAD baseline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from scripts.crossad_alfa import fault_type


ROOT = Path(__file__).resolve().parents[1]


def read_scores(directory: Path) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for path in sorted((directory / "scores").glob("*.npz")):
        with np.load(path) as data:
            result[path.stem] = {key: data[key].copy() for key in
                                 ("time_s", "labels", "scores", "predictions")}
    if not result:
        raise ValueError(f"No per-flight scores in {directory}")
    return result


def summarize(data: dict[str, np.ndarray]) -> dict:
    labels, scores, predictions = (data[key] for key in ("labels", "scores", "predictions"))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0)
    onset = int(np.flatnonzero(labels)[0]) if labels.any() else len(labels)
    return {"auroc": float(roc_auc_score(labels, scores)) if labels.any() else None,
            "auprc": float(average_precision_score(labels, scores)) if labels.any() else None,
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "false_alarm_points": int(predictions[:onset].sum())}


def compare(baseline: Path, candidate: Path) -> list[dict]:
    old, new = read_scores(baseline), read_scores(candidate)
    if old.keys() != new.keys():
        raise ValueError("Test flight inventories differ")
    rows = []
    for name in sorted(old):
        if not (np.array_equal(old[name]["labels"], new[name]["labels"])
                and np.array_equal(old[name]["time_s"], new[name]["time_s"])):
            raise ValueError(f"Flight labels or timestamps differ: {name}")
        previous, current = summarize(old[name]), summarize(new[name])
        row = {"flight": name, "fault_type": fault_type(name),
               "anomaly_points": int(new[name]["labels"].sum())}
        for metric in previous:
            row[f"baseline_{metric}"] = previous[metric]
            row[f"relation_{metric}"] = current[metric]
        row["delta_auroc"] = (current["auroc"] - previous["auroc"]
                               if current["auroc"] is not None else None)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/crossad_alfa/main")
    parser.add_argument("--candidate", type=Path, default=ROOT / "results/crossad_alfa/relation_v1")
    parser.add_argument("--output", type=Path, help="Comparison CSV path (defaults to candidate directory)")
    args = parser.parse_args()
    rows = compare(args.baseline, args.candidate)
    output = args.output or args.candidate / "comparison.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    faults = sorted((row for row in rows if row["anomaly_points"]),
                    key=lambda row: row["relation_f1"], reverse=True)
    print(f"Compared {len(rows)} flights ({len(faults)} with faults); candidate F1 leaders:")
    for row in faults[:5]:
        print(f"{row['flight']}: F1 {row['relation_f1']:.3f}, "
              f"AUROC {row['relation_auroc']:.3f}, delta AUROC {row['delta_auroc']:+.3f}")


if __name__ == "__main__":
    main()
