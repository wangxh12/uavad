"""Reconstruction-based UAV anomaly detection, following TSLib's Exp pattern.

Dataset contract (data_provider(args, flag)):
    train/val: (x, labels) or (x, ...), x: [B, seq_len, enc_in]
    test:      (x, point_labels), labels: [B, seq_len] or [B, seq_len, 1]

The model must reconstruct *all* input channels and return [B, seq_len, enc_in].
Scores are per-time-point mean squared errors averaged over all channels.
No point adjustment; threshold fitted on training scores only.

For overlapping windows, the flattened scores/labels represent window positions,
not unique timestamps. Aggregate by original timestamps in your data pipeline
before reporting true point-level metrics.
"""

import json
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.trainer_pt_utils import get_parameter_names

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from ts_ad_evaluation import Evaluator


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        module = self.model_dict[self.args.model]
        model_class = module.Model if hasattr(module, "Model") else module
        model = model_class(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        # return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
       
        optim_groups = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        model_optim = torch.optim.AdamW(
            optim_groups,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=1e-8,
        )
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def _autocast(self):
        if self.args.use_amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _forward(self, x):
        # Models using TSLib forward(x_enc, x_mark_enc, x_dec, x_mark_dec).
        outputs = self.model(x, None, None, None)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        return outputs

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, (tuple, list)):
            return batch[0], (batch[1] if len(batch) > 1 else None)
        return batch, None

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in vali_loader:
                batch_x, _ = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                with self._autocast():
                    outputs = self._forward(batch_x)
                    loss = criterion(outputs, batch_x)
                losses.append(float(loss.item()))
        if not losses:
            raise ValueError("Validation loader contains no samples")
        return float(np.mean(losses))

    def train(self, setting):
        _, train_loader = self._get_data("train")
        vali_data, vali_loader = self._get_data("val")
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        stopper = EarlyStopping(patience=self.args.patience, verbose=True)
        amp_enabled = self.args.use_amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        
        print('Model parameters: ', sum(param.numel() for param in self.model.parameters() if param.requires_grad))

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()
            losses = []
            for batch in train_loader:
                batch_x, _ = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    outputs = self._forward(batch_x)
                    loss = criterion(outputs, batch_x)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.item()))

            train_loss = float(np.mean(losses))
            val_loss = self.vali(vali_data, vali_loader, criterion)
            print(f"Epoch {epoch + 1}/{self.args.train_epochs} | "
                  f"Train: {train_loss:.6f}, Val: {val_loss:.6f} | "
                  f"{time.time() - epoch_time:.1f}s")
            stopper(val_loss, self.model, path)
            if stopper.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        self._load_checkpoint(setting)
        return self.model

    def _load_checkpoint(self, setting):
        ckpt = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
        self.model.load_state_dict(
            torch.load(ckpt, map_location=self.device, weights_only=True)
        )

    def _collect_scores(self, data_loader, collect_labels=False):
        self.model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for batch in data_loader:
                batch_x, batch_label = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                with self._autocast():
                    reconstructed = self._forward(batch_x)
                # [B, L, C] -> [B, L] (all channels participate).
                score = (reconstructed.float() - batch_x.float()).square().mean(dim=-1)
                scores.append(score.cpu().numpy().reshape(-1))

                if collect_labels:
                    if batch_label is None:
                        raise ValueError("Point-level labels required for evaluation")
                    label = torch.as_tensor(batch_label)
                    if label.ndim == 3 and label.shape[-1] == 1:
                        label = label[..., 0]
                    if label.shape != score.shape:
                        raise ValueError(
                            f"Expected point labels {tuple(score.shape)}, "
                            f"got {tuple(label.shape)}"
                        )
                    labels.append(label.cpu().numpy().reshape(-1))

        if not scores:
            raise ValueError("Data loader contains no samples")
        all_scores = np.concatenate(scores)
        all_labels = np.concatenate(labels) if collect_labels else None
        return all_scores, all_labels

    def test(self, setting, test=0):
        if test:
            print('loading model')
            self._load_checkpoint(setting)
            
        _, train_loader = self._get_data("train")
        _, test_loader = self._get_data("test")

        with torch.no_grad():
            train_scores, _ = self._collect_scores(train_loader)
            test_scores, labels = self._collect_scores(test_loader, collect_labels=True)
        
        # TSLib anomaly_ratio is a percentage: e.g. 1.0 means the top 1%.
        threshold = float(np.percentile(train_scores, 100 - self.args.anomaly_ratio))
        predictions = (test_scores > threshold).astype(np.int64)
        labels = (labels != 0).astype(np.int64)
        
        
        result_dir = os.path.join("results", setting)
        os.makedirs(result_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(result_dir, "anomaly_scores.npz"),
            train_scores=train_scores,
            test_scores=test_scores,
            labels=labels,
            predictions=predictions,
            threshold=threshold,
        )


        # TimeRadar's Evaluator handles metric computation and CSV outputs.
        # Example optional CLI: --eval_metrics f1_raw auc
        selected = getattr(self.args, "eval_metrics", ["f1_raw", "auc"])
        if isinstance(selected, str):
            selected = selected.replace(",", " ").split()
        metric_names = list(dict.fromkeys(selected))
        allowed = {"f1_raw", "f1_pa", "affiliation", "auc", "r_auc", "vus", "best_f1"}
        if not metric_names or any(m not in allowed for m in metric_names):
            raise ValueError(f"Invalid eval_metrics: {metric_names}. Allowed: {sorted(allowed)}")

        # ROC-AUC / AP require both normal and abnormal ground-truth classes.
        if np.unique(labels).size < 2:
            skipped = {"auc", "r_auc", "vus"} & set(metric_names)
            if skipped:
                print(f"Skipping {sorted(skipped)}: test labels contain only one class")
                metric_names = [m for m in metric_names if m not in skipped]

        evaluator = Evaluator(gt=labels, anomaly_score=test_scores, save_path=result_dir)
        results = evaluator.evaluate(
            metrics=metric_names,
            f1_raw=[threshold],
            f1_pa=[threshold],
            affiliation=[threshold],
            slidingWindow=self.args.seq_len,
        )

        # Evaluator returns pandas DataFrames, not JSON-ready scalar dictionaries.
        metrics = {"threshold": threshold}
        for name, df in results.items():
            if len(df) > 0:
                metrics[name] = {key: float(value) for key, value in df.iloc[0].items()}

        # Maintain the same top-level keys as the original test() for callers.
        raw = metrics.get("f1_raw")
        if raw is not None:
            metrics.update({
                "accuracy": raw["ACC_raw"],
                "precision": raw["P_raw"],
                "recall": raw["R_raw"],
                "f1": raw["F1_raw"],
            })

        with open(os.path.join(result_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print("Reconstruction AD:", metrics)
        return metrics
        
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )
        metrics = {
            "threshold": threshold,
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        result_dir = os.path.join("results", setting)
        os.makedirs(result_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(result_dir, "anomaly_scores.npz"),
            train_scores=train_scores,
            test_scores=test_scores,
            labels=labels,
            predictions=predictions,
            threshold=threshold,
        )
        with open(os.path.join(result_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print("Reconstruction AD:", metrics)
        return metrics
