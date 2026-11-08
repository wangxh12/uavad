"""Diagnostic score plots for the fixed-threshold ALFA CrossAD run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"engine": "#277da1", "elevator": "#f8961e", "aileron": "#b44d60", "rudder": "#43aa8b"}


def kind(name: str) -> str:
    return next((key for key in COLORS if key in name), "normal")


def load_scores(run_dir: Path) -> list[dict]:
    flights = []
    for path in sorted((run_dir / "scores").glob("*.npz")):
        with np.load(path) as data:
            flights.append({"name": path.stem.replace("carbonZ_", ""), "kind": kind(path.stem),
                            "time": data["time_s"].copy(), "label": data["labels"].copy(),
                            "score": data["scores"].copy(), "pred": data["predictions"].copy()})
    if not flights:
        raise ValueError(f"No per-flight scores under {run_dir}")
    return flights


def summary_figure(flights: list[dict], threshold: float, output: Path) -> None:
    normal = np.concatenate([f["score"][f["label"] == 0] for f in flights])
    fault = np.concatenate([f["score"][f["label"] == 1] for f in flights])
    fig, (ax, ranked) = plt.subplots(2, 1, figsize=(10.5, 8.5),
                                     gridspec_kw={"height_ratios": [1.1, 1.25]}, constrained_layout=True)
    floor = 1e-8
    bins = np.linspace(np.log10(max(min(normal.min(), fault.min()), floor)),
                       np.log10(max(normal.max(), fault.max(), floor * 10)), 85)
    for values, label, color in ((normal, f"Normal points (n={len(normal):,})", "#277da1"),
                                 (fault, f"Fault points (n={len(fault):,})", "#b44d60")):
        ax.hist(np.log10(np.maximum(values, floor)), bins=bins, density=True, histtype="step", linewidth=1.8,
                color=color, label=label)
    ax.axvline(np.log10(max(threshold, floor)), color="#222222", linestyle="--", linewidth=1.5,
               label=f"Validation threshold = {threshold:.2f}")
    ax.set(xlabel="log10(anomaly score)", ylabel="Density",
           title="Pooled test score distributions")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.18)

    for i, category in enumerate(COLORS):
        chosen = [f for f in flights if f["kind"] == category and f["label"].any()]
        aucs = [roc_auc_score(f["label"], f["score"]) for f in chosen]
        ranked.scatter(np.full(len(aucs), i) + np.linspace(-0.12, 0.12, len(aucs)), aucs,
                       s=55, alpha=0.9, color=COLORS[category], edgecolors="white", linewidths=0.5,
                       label=f"{category} ({len(aucs)} flights)")
        ranked.plot([i - 0.19, i + 0.19], [np.median(aucs)] * 2, color=COLORS[category], linewidth=2)
    ranked.axhline(0.5, color="#555555", linestyle="--", linewidth=1, label="Chance (0.5)")
    ranked.set(xlim=(-0.5, 3.5), ylim=(0, 1.03), ylabel="Within-flight AUROC",
               title="Within-flight ranking by fault type")
    ranked.set_xticks(range(4), ["Engine", "Elevator", "Aileron", "Rudder"])
    ranked.grid(axis="y", alpha=0.18)
    ranked.text(0.01, -0.22, "Each dot is one fault flight; short bars are category medians. AUC does not use the threshold.",
                transform=ranked.transAxes, fontsize=9, color="#555555")
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def timeline_figure(flights: list[dict], threshold: float, output: Path) -> None:
    faults = sorted((f for f in flights if f["label"].any()),
                    key=lambda f: roc_auc_score(f["label"], f["score"]))
    normal = sorted((f for f in flights if not f["label"].any()),
                    key=lambda f: int(f["pred"].sum()), reverse=True)
    cases = [(faults[-1], "Highest fault-flight AUROC"),
             (faults[0], "Lowest fault-flight AUROC"),
             (normal[0], "Most alarms on a normal flight"),
             (normal[-1], "Fewest alarms on a normal flight")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True, constrained_layout=True)
    for ax, (f, title) in zip(axes.flat, cases):
        t, score, label = f["time"], f["score"], f["label"]
        if label.any():
            onset = t[np.flatnonzero(label)[0]]
            ax.axvspan(onset, t[-1], color="#b44d60", alpha=0.12, label="Labeled fault")
            ax.axvline(onset, color="#b44d60", linewidth=1.2)
        ax.plot(t, score, color="#277da1", linewidth=0.85, label="Anomaly score")
        ax.axhline(threshold, color="#222222", linestyle="--", linewidth=1.25,
                   label=f"Threshold {threshold:.2f}")
        ax.set(yscale="log", xlim=(t[0], t[-1]), xlabel="Time in flight (s)",
               ylabel="Anomaly score (log scale)", title=title)
        ax.grid(alpha=0.18, which="major")
        ax.text(0.02, 0.04, f["name"], transform=ax.transAxes, fontsize=8, color="#555555",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2})
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def flight_panel(ax, flight: dict, threshold: float) -> None:
    time, scores, labels, predicted = (flight[key] for key in ("time", "score", "label", "pred"))
    visible_scores = np.maximum(scores, 1e-8)
    bottom = max(min(float(visible_scores.min()), threshold) * 0.75, 1e-8)
    top = max(float(visible_scores.max()), threshold) * 1.35
    onset = np.flatnonzero(labels)
    if len(onset):
        ax.axvspan(time[onset[0]], time[-1], color="#b44d60", alpha=0.13)
    ax.plot(time, visible_scores, color="#277da1", linewidth=0.9, label="Score")
    ax.plot(time, np.where(predicted == 1, visible_scores, np.nan), color="#f8961e",
            linewidth=1.8, label="Alarm")
    ax.axhline(threshold, color="#222222", linestyle="--", linewidth=1.2, label="Threshold")
    ax.set(xlim=(time[0], time[-1]), ylim=(bottom, top), yscale="log",
           xlabel="Time in flight (s)", ylabel="Anomaly score (log)")
    ax.grid(alpha=0.18)
    if len(onset):
        auc = roc_auc_score(labels, scores)
        recall = predicted[labels == 1].mean()
        heading = f"{flight['kind'].title()} | AUC {auc:.2f} | fault recall {recall:.1%}"
    else:
        heading = f"Normal | false alarms {predicted.sum()}/{len(predicted)} points"
    ax.set_title(heading, fontsize=11)
    ax.text(0.02, 0.03, flight["name"], transform=ax.transAxes, fontsize=8,
            color="#555555", bbox={"facecolor": "white", "alpha": 0.86,
                                   "edgecolor": "none", "pad": 2})


def all_flight_figures(flights: list[dict], threshold: float, output: Path) -> None:
    individual = output / "per-flight"
    individual.mkdir(exist_ok=True)
    for flight in flights:
        fig, ax = plt.subplots(figsize=(11, 3.5), constrained_layout=True)
        flight_panel(ax, flight, threshold)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 1.15))
        fig.savefig(individual / f"{flight['name']}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    groups = (
        ("engine-flights", "Engine faults", [f for f in flights if f["kind"] == "engine"]),
        ("aileron-flights", "Aileron faults", [f for f in flights if f["kind"] == "aileron"]),
        ("rudder-elevator-flights", "Rudder and elevator faults",
         [f for f in flights if f["kind"] in ("rudder", "elevator")]),
        ("normal-flights", "Normal flights", [f for f in flights if f["kind"] == "normal"]),
    )
    for name, title, selected in groups:
        if not selected:
            raise ValueError(f"No flights for {name}")
        columns = 1 if name == "normal-flights" else 2
        rows = (len(selected) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(14, rows * 3.3),
                                 squeeze=False)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.86,
                            hspace=0.70, wspace=0.22)
        fig.suptitle(title, fontsize=15, y=0.99)
        for ax, flight in zip(axes.flat, selected):
            flight_panel(ax, flight, threshold)
        for ax in list(axes.flat)[len(selected):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 0.955))
        fig.savefig(output / f"{name}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/crossad_alfa/main")
    args = parser.parse_args()
    metrics = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    flights = load_scores(args.run_dir)
    output = args.run_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    summary_figure(flights, metrics["threshold"], output / "separation.png")
    timeline_figure(flights, metrics["threshold"], output / "flight-timelines.png")
    all_flight_figures(flights, metrics["threshold"], output)
    print(f"Wrote four grouped sheets and {len(flights)} individual plots to {output}")


if __name__ == "__main__":
    main()
