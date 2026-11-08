"""Compare per-flight ALFA results for two runs on identical test segments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "engine": "#277da1", "elevator": "#f8961e", "aileron": "#b44d60",
    "rudder": "#43aa8b", "aileron+rudder": "#8e6fbd",
}


def read_flights(run_dir: Path) -> dict[str, dict]:
    with (run_dir / "flights.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({row["flight"] for row in rows}) != len(rows):
        raise ValueError(f"Empty or duplicate per-flight report: {run_dir}")
    return {row["flight"]: row for row in rows}


def compare(baseline: Path, candidate: Path, output: Path,
            baseline_label: str = "Original CrossAD",
            candidate_label: str = "CrossAD relation") -> None:
    left, right = read_flights(baseline), read_flights(candidate)
    if left.keys() != right.keys():
        raise ValueError("Flight inventories differ")
    for name in left:
        if (left[name]["anomaly_points"] != right[name]["anomaly_points"] or
                left[name]["fault_type"] != right[name]["fault_type"]):
            raise ValueError(f"Flight labels or type differ: {name}")
        stem = Path(name).stem + ".npz"
        with np.load(baseline / "scores" / stem, allow_pickle=False) as old, np.load(
                candidate / "scores" / stem, allow_pickle=False) as new:
            if (not np.array_equal(old["labels"], new["labels"]) or
                    not np.array_equal(old["time_s"], new["time_s"])):
                raise ValueError(f"Point labels or timestamps differ: {name}")
    faults = sorted((name for name in left if int(left[name]["anomaly_points"])),
                    key=lambda name: left[name]["fault_type"])
    normals = [name for name in left if not int(left[name]["anomaly_points"])]
    categories = [key for key in COLORS if any(left[name]["fault_type"] == key for name in faults)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, metric, title in ((axes[0, 0], "auroc", "Fault-flight AUROC"),
                               (axes[0, 1], "f1", "Fault-flight F1 at validation threshold")):
        for category in categories:
            selected = [name for name in faults if left[name]["fault_type"] == category]
            ax.scatter([float(left[name][metric]) for name in selected],
                       [float(right[name][metric]) for name in selected],
                       color=COLORS[category], edgecolors="white", linewidths=0.5,
                       s=54, alpha=0.85, label=f"{category} ({len(selected)})")
        ax.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1)
        ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xlabel=baseline_label,
               ylabel=candidate_label, title=title)
        ax.grid(alpha=0.15)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")

    differences = np.array([float(right[name]["auroc"]) - float(left[name]["auroc"])
                            for name in faults])
    axes[1, 0].hist(differences, bins=16, color="#457b9d", edgecolor="white")
    axes[1, 0].axvline(0, color="#333333", linestyle="--", linewidth=1.1)
    axes[1, 0].set(xlabel=f"{candidate_label} minus {baseline_label} AUROC", ylabel="Fault flights",
                   title=f"Per-flight change: {int((differences > 0).sum())}/{len(faults)} improved")
    axes[1, 0].grid(axis="y", alpha=0.15)

    labels = ["AUROC >= .80", "F1 >= .50", "Fault detected", "Normal flight\nwith alarm"]
    def counts(rows: dict[str, dict]) -> list[int]:
        return [sum(float(rows[name]["auroc"]) >= .8 for name in faults),
                sum(float(rows[name]["f1"]) >= .5 for name in faults),
                sum(rows[name]["detected"].lower() == "true" for name in faults),
                sum(int(rows[name]["false_alarm_points"]) > 0 for name in normals)]
    places = np.arange(len(labels))
    axes[1, 1].bar(places - .19, counts(left), width=.36, color="#277da1", label=baseline_label)
    axes[1, 1].bar(places + .19, counts(right), width=.36, color="#b44d60", label=candidate_label)
    axes[1, 1].set(xticks=places, xticklabels=labels, ylabel="Flight count",
                   title=f"All {len(faults)} fault and {len(normals)} normal flights")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(axis="y", alpha=0.15)
    fig.suptitle("ALFA within-flight chronological split | 96-point offline windows", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}; {int((differences > 0).sum())}/{len(faults)} fault-flight AUROCs improved")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-label", default="Original CrossAD")
    parser.add_argument("--candidate-label", default="CrossAD relation")
    args = parser.parse_args()
    compare(args.baseline, args.candidate, args.output,
            args.baseline_label, args.candidate_label)


if __name__ == "__main__":
    main()
