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
        self.smoothed_res = np.asarray(self.anomaly_score).reshape(-1)
        if smooth_method:
            self.smooth_residual(smooth_method, rho=self.args.smooth_rho, verbose=verbose)

        thresholds = np.atleast_1d(self.thres).tolist()
        metric_options = {}
        for name in ("f1_raw", "f1_pa", "affiliation"):
            if name in metrics:
                metric_options[name] = thresholds

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
        return output

    def plot_scores(self):
        raw = np.asarray(self.anomaly_score).reshape(-1)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(raw, color="#aeb7c1", linewidth=0.8, alpha=0.7, label="Raw residual")
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
            method: One of percentile, 3sigma, spot, mad or iqr.
            verbose: Print the threshold and method when True.

        For percentile, 3sigma and spot, args.thres_args specifies the
        percentile, standard deviation multiplier and SPOT q respectively.
        If omitted, they default to 100 - anomaly_ratio, threshold_sigma
        and 0.001. MAD and IQR use threshold_mad_k and threshold_iqr_k.
        SPOT also uses test residuals and returns the mean dynamic threshold.

        Returns:
            The threshold as a scalar.
        """
        scores = np.asarray(self.init_score, dtype=np.float64)
        value = self.args.thres_args

        if method == "percentile":
            percentile = value if value is not None else 100 - self.args.anomaly_ratio
            threshold = np.percentile(scores, percentile)
        elif method == "3sigma":
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

    def smooth_residual(self, method="bi_aewma", verbose=True, **kwargs):
        """Smooth test residuals and save a comparison with the original values."""
        self.smoothed_res = getattr(smooth, method)(
            np.asarray(self.anomaly_score).reshape(-1), **kwargs
        )

        if verbose:
            print(f"smooth_method: {method}, smooth_args: {kwargs}")

        # plot diff
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(np.asarray(self.anomaly_score).reshape(-1), color="#aeb7c1", linewidth=0.8, alpha=0.7, label="Raw residual")
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

        elif method == "3sigma":
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

    def evaluate11(
        self, method="percentile", percentile=99, sigma=3, q=1e-3, level=0.98
    ):

        pred, thresholds = self.predict(method, percentile, sigma, q, level)

        precision, recall, f1, _ = precision_recall_fscore_support(
            self.gt, pred, average="binary", zero_division=0
        )

        auroc = (
            roc_auc_score(self.gt, self.score)
            if np.unique(self.gt).size == 2
            else np.nan
        )

        auprc = (
            average_precision_score(self.gt, self.score) if self.gt.any() else np.nan
        )

        n_events = 0
        detected = 0
        delays = []

        for part in self._flights():
            for begin, end in self._events(self.gt[part]):
                n_events += 1

                hits = np.flatnonzero(pred[part][begin:end])

                if hits.size:
                    detected += 1
                    delays.append(int(hits[0]))

        result = {
            "method": method,
            "precision": precision,
            "recall": recall,
            "f1_raw": f1,
            "auroc": auroc,
            "auprc": auprc,
            "event_recall": (detected / n_events if n_events else np.nan),
            "mean_delay_steps": (np.mean(delays) if delays else np.nan),
        }

        records = pd.DataFrame(
            {
                "flight": np.repeat(
                    np.arange(len(self.flight_lengths)), self.flight_lengths
                ),
                "time_step": np.concatenate(
                    [np.arange(n) for n in self.flight_lengths]
                ),
                "label": self.gt,
                "score": self.score,
                "threshold": thresholds,
                "pred": pred,
            }
        )

        records.to_csv(self.save_path / f"{method}_predictions.csv", index=False)

        pd.DataFrame([result]).to_csv(
            self.save_path / f"{method}_metrics.csv", index=False
        )

        print(pd.Series(result))
        return result

    def evaluate_ranges(self, metrics=("r_auc", "vus"), sliding_window=100):
        from . import auc_vus

        rows = []

        for flight_id, part in enumerate(self._flights()):
            labels = self.gt[part]

            if np.unique(labels).size < 2:
                continue

            storage = {}
            auc_vus.evaluate(
                storage,
                metrics,
                labels=labels,
                score=self.score[part],
                slidingWindow=sliding_window,
            )

            row = {"flight": flight_id}

            for values in storage.values():
                frame = pd.DataFrame.from_dict(values)
                row.update(frame.iloc[0].to_dict())

            rows.append(row)

        table = pd.DataFrame(rows)
        table.to_csv(self.save_path / "range_metrics.csv", index=False)
        return table
