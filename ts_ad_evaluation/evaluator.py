import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .spot import SPOT
from . import affiliation, auc_vus, f1, smooth


class Evaluator:
    def __init__(self, init_score, gt, anomaly_score, save_path, args):
        """
        input:
            gt: np.ndarray[int],
            anomaly_score: np.ndarray[float],
            save_path: str
        """
        self.init_score = init_score
        self.gt = gt
        self.anomaly_score = anomaly_score
        self.save_path = save_path
        self.args = args
        self.thres = None
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.smoothed_res = np.asarray(self.anomaly_score).reshape(-1)  # init

    def evaluate(self, metrics, thres_method="percentile", smooth_method=None, verbose=True):
        """Evaluate test residuals and save the metrics and score plots.

        The threshold is calculated from the original residuals. When smoothing
        is enabled, the smoothed test residuals are used to calculate metrics.
        """
        self.thres = self.find_thres(thres_method, verbose=verbose)

        # smooth residual
        self.smoothed_res = np.asarray(self.anomaly_score).reshape(-1)
        if smooth_method:
            self.smooth_residual(smooth_method, verbose=verbose)

        thresholds = np.atleast_1d(self.thres).tolist()
        metric_options = {}
        for name in ("f1_raw", "f1_pa", "affiliation"):
            if name in metrics:
                metric_options[name] = thresholds

        # eval
        results = {}
        labels = np.asarray(self.gt).reshape(-1)
        f1.evaluate(results, metrics, labels=labels, score=self.smoothed_res, **metric_options)
        affiliation.evaluate(results, metrics, labels=labels, score=self.smoothed_res, **metric_options)
        auc_vus.evaluate(results, metrics, labels=labels, score=self.smoothed_res, **metric_options)

        self.plot_scores()
        output = {
            "threshold_method": thres_method,
            "threshold": thresholds,
            "smooth_method": smooth_method,
            "metrics": {
                name: json.loads(frame.to_json(orient="records"))
                for name, frame in results.items()
            },
        }
        with open(os.path.join(self.save_path, "metrics.json"), "w", encoding="utf-8") as file:
            json.dump(output, file, indent=2, ensure_ascii=False)

        # Print metrics and absolute paths for quick access from the terminal.
        if verbose:
            # print("\nEvaluation results")
            for name, frame in results.items():
                print(f"\n{name}")
                print(frame.to_string(index=False, float_format="{:.6f}".format))
            result_dir = Path(self.save_path).absolute()
            print(f"\nResults directory: {result_dir}")
            print(f"Metrics file: {result_dir / 'metrics.json'}")
        return output

    def _shade_anomalies(self, ax):
        """Shade labeled anomaly intervals, including each endpoint sample."""
        mask = np.asarray(self.gt).reshape(-1).astype(bool)
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        for i, (start, end) in enumerate(edges.reshape(-1, 2)):
            ax.axvspan(start - 0.5, end - 0.5, color="#e88b7d", alpha=0.18,
                       linewidth=0, zorder=0, label="Labeled anomaly" if i == 0 else None)

    def plot_scores(self):
        raw = np.asarray(self.anomaly_score).reshape(-1)
        fig, ax = plt.subplots(figsize=(12, 4))
        self._shade_anomalies(ax)
        ax.plot(raw, color="#7c8895", linewidth=1, alpha=0.7, label="Raw residual")
        ax.plot(self.smoothed_res, color="#2878b5", linewidth=1.2, label="Evaluation score")
        for i, threshold in enumerate(np.atleast_1d(self.thres)):
            ax.axhline(threshold, color="#d95f5f", linestyle="--", linewidth=1.1,
                       label="Threshold" if i == 0 else None)
        ax.set(xlabel="Sample index", ylabel="Residual", title="Anomaly scores and threshold")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_path, "anomaly_scores.png"), dpi=200)
        plt.close(fig)

    def find_thres(self, method, verbose=True):
        """Calculate a threshold from validation residuals.

        Args:
            method: One of percentile, sigma, spot, mad or iqr.
            verbose: Print the threshold and method when True.

        Returns:
            The threshold as a scalar.
        """
        scores = np.asarray(self.init_score, dtype=np.float64)
        value = self.args.thres_args

        if method == "percentile":
            percentile = value if value is not None else 100 - self.args.anomaly_ratio
            threshold = np.percentile(scores, percentile)
        elif method == "sigma":
            sigma = value if value is not None else self.args.threshold_sigma
            threshold = scores.mean() + sigma * scores.std()
        elif method == "spot":
            spot = SPOT(value if value is not None else 0.001)
            spot.fit(scores.reshape(-1), np.asarray(self.anomaly_score).reshape(-1))
            spot.initialize(verbose=False)
            threshold = np.mean(spot.run()["thresholds"])
        elif method == "mad":
            median = np.median(scores)
            mad = np.median(np.abs(scores - median))
            threshold = median + self.args.threshold_mad_k * 1.4826 * mad
        elif method == "iqr":
            q1, q3 = np.percentile(scores, [25, 75])
            threshold = q3 + self.args.threshold_iqr_k * (q3 - q1)

        if verbose:
            print(f"threshold: {threshold}, method: {method}")
        return float(threshold)

    def smooth_residual(self, method="bi_aewma", verbose=True):
        """Smooth test residuals using args.smooth_args and save a comparison plot."""
        self.smoothed_res = getattr(smooth, method)(
            np.asarray(self.anomaly_score).reshape(-1), args=self.args.smooth_args
        )

        if verbose:
            print(f"smooth_method: {method}, smooth_args: {self.args.smooth_args}")

        # plot diff
        fig, ax = plt.subplots(figsize=(12, 4))
        self._shade_anomalies(ax)
        ax.plot(np.asarray(self.anomaly_score).reshape(-1), color="#80848a", linewidth=1, alpha=0.7, label="Raw residual")
        ax.plot(
            self.smoothed_res,
            color="#2878b5",
            linewidth=1.4,
            label="Smoothed residual",
        )
        ax.set(xlabel="Sample index", ylabel="Residual", title="Residual smoothing")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_path, "smooth_residual.png"), dpi=200)
        plt.close(fig)

    def predict(
        self, method="percentile", percentile=99.5, sigma=3, q=1e-3, level=0.98
    ):

        if method == "percentile":
            threshold = np.percentile(self.init_score, percentile)

        elif method == "sigma":
            threshold = self.init_score.mean() + sigma * self.init_score.std()

        elif method == "spot":
            thresholds = np.empty_like(self.score)
            pred = np.zeros(len(self.score), dtype=int)

            for part in self._flights():
                spot = SPOT(q)
                spot.fit(self.init_score, self.score[part])
                spot.initialize(level=level, verbose=False)

                initial_threshold = spot.extreme_quantile
                result = spot.run(with_alarm=True)

                thresholds[part] = np.r_[initial_threshold, result["thresholds"][:-1]]

                alarm_indices = np.asarray(result["alarms"], dtype=int)
                pred[part.start + alarm_indices] = 1

            return pred, thresholds

        else:
            raise ValueError(f"Unknown threshold method: {method}")

        thresholds = np.full(len(self.score), threshold)
        pred = (self.score > thresholds).astype(int)

        return pred, thresholds
