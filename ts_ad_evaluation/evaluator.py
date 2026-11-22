from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .spot import SPOT


class Evaluator:
    def __init__(self, gt, anomaly_score, save_path, init_score, flight_lengths=None):
        self.gt = np.asarray(gt, dtype=int).ravel()
        self.score = np.asarray(anomaly_score, dtype=float).ravel()
        self.anomaly_score = self.score
        self.init_score = np.asarray(init_score, dtype=float).ravel()

        self.flight_lengths = list(
            flight_lengths if flight_lengths is not None else [len(self.gt)]
        )

        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        assert len(self.gt) == len(self.score) == sum(self.flight_lengths)

    def _flights(self):
        start = 0
        for length in self.flight_lengths:
            yield slice(start, start + length)
            start += length

    @staticmethod
    def _events(labels):
        edges = np.flatnonzero(np.diff(np.r_[0, labels, 0]))
        return edges.reshape(-1, 2)

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

    def evaluate(
        self, method="percentile", percentile=99.5, sigma=3, q=1e-3, level=0.98
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
