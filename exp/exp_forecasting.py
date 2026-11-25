import json
import os
import time
from contextlib import nullcontext

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.residual_filter import bi_aewma
from utils.threshold import calculate_threshold


class Exp_Forecasting(Exp_Basic):
    target_excluded_from_x = True

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or name.endswith("bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.args.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
        )

    def _select_criterion(self):
        return nn.MSELoss()

    def _autocast(self):
        if self.args.use_amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _forward(self, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        # No historical target observations or calendar time features.
        dec_inp = batch_x.new_zeros(
            (batch_x.size(0), self.args.pred_len, self.args.dec_in)
        )
        with self._autocast():
            outputs = self.model(batch_x, None, dec_inp, None)

        outputs = outputs[:, -self.args.pred_len :, :].float()
        assert outputs.shape == batch_y.shape, "Prediction and target shapes differ"
        return outputs, batch_y

    @torch.no_grad()
    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss, n_samples = 0.0, 0

        for batch_x, batch_y, _ in vali_loader:
            outputs, target = self._forward(batch_x, batch_y)
            loss = criterion(outputs, target)
            total_loss += loss.item() * batch_x.size(0)
            n_samples += batch_x.size(0)

        self.model.train()
        return total_loss / n_samples

    def train(self, setting):
        _, train_loader = self._get_data("train")
        _, vali_loader = self._get_data("val")
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        grad_scaler = torch.amp.GradScaler(
            "cuda", enabled=self.args.use_amp and self.device.type == "cuda"
        )

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_start = time.time()
            total_loss, n_samples = 0.0, 0

            for batch_x, batch_y, _ in train_loader:
                optimizer.zero_grad(set_to_none=True)
                outputs, target = self._forward(batch_x, batch_y)
                loss = criterion(outputs, target)

                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                total_loss += loss.item() * batch_x.size(0)
                n_samples += batch_x.size(0)

            train_loss = total_loss / n_samples
            val_loss = self.vali(vali_loader, criterion)
            print(
                f"Epoch {epoch + 1}/{self.args.train_epochs} | "
                f"Train {train_loss:.6f} | Val {val_loss:.6f} | "
                f"{time.time() - epoch_start:.1f}s"
            )

            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        self._load_checkpoint(setting)
        return self.model

    @torch.no_grad()
    def _collect_predictions(self, loader):
        self.model.eval()
        preds, trues, labels = [], [], []

        for batch_x, batch_y, batch_label in loader:
            outputs, target = self._forward(batch_x, batch_y)
            preds.append(outputs.cpu().numpy())
            trues.append(target.cpu().numpy())
            labels.append(batch_label.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        labels = np.concatenate(labels, axis=0).reshape(-1)

        scores = np.abs(preds - trues).mean(axis=(1, 2))
        return preds, trues, scores, labels

    def _smooth_scores(self, raw_scores, offsets):
        scores = raw_scores.copy()
        if self.args.use_biaewma:
            # Reset the filter at every flight boundary.
            for begin, end in zip(offsets[:-1], offsets[1:]):
                scores[begin:end] = bi_aewma(
                    raw_scores[begin:end].copy(), self.args.smooth_rho
                )
        return scores

    def test(self, setting, test=0):
        if test:
            print("Loading model checkpoint")
            self._load_checkpoint(setting)

        val_data, val_loader = self._get_data("val")
        test_data, test_loader = self._get_data("test")

        _, _, val_raw_scores, _ = self._collect_predictions(val_loader)
        preds, trues, test_raw_scores, labels = self._collect_predictions(test_loader)

        # Validation and test loaders are sequential and retain the last batch.
        val_offsets = np.cumsum([0, *val_data.window_sizes])
        test_offsets = np.cumsum([0, *test_data.window_sizes])
        val_scores = self._smooth_scores(val_raw_scores, val_offsets)
        test_scores = self._smooth_scores(test_raw_scores, test_offsets)

        threshold = float(
            calculate_threshold(
                val_scores=val_scores,
                method=self.args.threshold_method,
                anomaly_ratio=self.args.anomaly_ratio,
                sigma=self.args.threshold_sigma,
            )
        )
        detections = (test_scores > threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, detections, average="binary", zero_division=0
        )
        metrics = {
            "mae": float(test_raw_scores.mean()),
            "threshold": threshold,
            "accuracy": float(accuracy_score(labels, detections)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        result_dir = os.path.join("results", setting)
        os.makedirs(result_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(result_dir, "forecast_scores.npz"),
            pred=preds,
            true=trues,
            val_raw_scores=val_raw_scores,
            test_raw_scores=test_raw_scores,
            val_scores=val_scores,
            test_scores=test_scores,
            labels=labels,
            predictions=detections,
            threshold=threshold,
            val_offsets=val_offsets,
            test_offsets=test_offsets,
            val_file_names=np.asarray(val_data.file_names, dtype=str),
            test_file_names=np.asarray(test_data.file_names, dtype=str),
        )
        with open(os.path.join(result_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        # Plot each flight on its original row-index axis.
        for file_name, begin, end in zip(
            test_data.file_names, test_offsets[:-1], test_offsets[1:]
        ):
            flight_dir = os.path.join(result_dir, os.path.splitext(file_name)[0])
            os.makedirs(flight_dir, exist_ok=True)
            self.plot_forecast_results(
                preds=preds[begin:end, 0, 0],
                trues=trues[begin:end, 0, 0],
                scores=test_scores[begin:end],
                threshold=threshold,
                labels=labels[begin:end],
                save_dir=flight_dir,
                start_index=self.args.seq_len,
            )

        print("\nForecasting Anomaly Detection Results:")
        for key, value in metrics.items():
            print(f"{key}: {value:.6f}")
        return metrics

    def plot_forecast_results(
        self, preds, trues, scores, threshold, labels, save_dir, start_index=0
    ):
        """Plot one flight's target and time-aligned anomaly scores."""
        indices = np.arange(start_index, start_index + len(scores))
        anomaly_mask = labels.astype(bool)

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(indices, trues, label="Observed target", linewidth=1.2)
        ax.plot(indices, preds, label="Prediction", linewidth=1.1)
        ax.fill_between(
            indices,
            0,
            1,
            where=anomaly_mask,
            transform=ax.get_xaxis_transform(),
            alpha=0.12,
            step="mid",
            label="Labeled anomaly",
        )
        ax.set(
            xlabel="Flight row index",
            ylabel="Target (Dataset scale)",
            title=f"Forecast vs. observation: {self.args.target}",
        )
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "forecast_vs_gt.png"), dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(indices, scores, label="Anomaly score", linewidth=1)
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1.4,
            label=f"Threshold = {threshold:.4g}",
        )
        ax.fill_between(
            indices,
            0,
            1,
            where=anomaly_mask,
            transform=ax.get_xaxis_transform(),
            alpha=0.12,
            step="mid",
            label="Labeled anomaly",
        )
        ax.set(
            xlabel="Flight row index",
            ylabel="Anomaly score",
            title="Anomaly scores vs. validation threshold",
        )
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "anomaly_scores.png"), dpi=200)
        plt.close(fig)
