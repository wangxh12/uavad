import json
import os
import time
from contextlib import nullcontext

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.residual_filter import bi_aewma
from utils.threshold import calculate_threshold


class Exp_Forecasting(Exp_Basic):
    target_excluded_from_x = True
    
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        # return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or name.endswith('bias'):
                no_decay.append(param)
            else:
                decay.append(param)
        groups = [
            {'params': decay, 'weight_decay': getattr(self.args, 'weight_decay', 0.01)},
            {'params': no_decay, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(
            groups,
            lr=self.args.learning_rate,
            betas=(getattr(self.args, 'adam_beta1', 0.9),
                   getattr(self.args, 'adam_beta2', 0.999)),
        )

    def _select_criterion(self):
        return nn.MSELoss()

    def _autocast(self):
        if self.args.use_amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()
    
    def _forward_batch(self, batch):
        batch_x, batch_y, x_mark, y_mark = batch

        # Dataset: [B, L, 18] -> Exp encoder: [B, L, 17].
        # Target is the last channel; do NOT pass it to the model.
        
        x = batch_x[:, :, :-1] if self.target_excluded_from_x else batch_x
        x = x.float().to(self.device)
        y = batch_y[:, -self.args.pred_len:, -1:].float().to(self.device)
        x_mark = x_mark.float().to(self.device)
        y_mark = y_mark[:, -self.args.pred_len:, :].float().to(self.device)

        # Never feed historical/future target values to the decoder.
        dec_inp = torch.zeros(
            x.size(0), self.args.pred_len, self.args.dec_in,
            dtype=x.dtype, device=x.device,
        )
        with self._autocast():
            outputs = self.model(x, x_mark, dec_inp, y_mark)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
        if outputs.shape != y.shape:
            raise ValueError(
                f"Model must output [B, pred_len, 1]; got {tuple(outputs.shape)} "
                f"instead of {tuple(y.shape)}."
            )
        return outputs, y
    
    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss, n_samples = 0.0, 0
        with torch.no_grad():
            for batch in vali_loader:
                outputs, target = self._forward_batch(batch)
                loss = criterion(outputs.float(), target)
                total_loss += loss.item() * target.size(0)
                n_samples += target.size(0)
        return total_loss / n_samples
    
    def train(self, setting):
        _, train_loader = self._get_data('train')
        val_data, val_loader = self._get_data('val')
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        grad_scaler = torch.amp.GradScaler(
            'cuda', enabled=self.args.use_amp and self.device.type == 'cuda'
        )

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_start = time.time()
            total_loss, n_samples = 0.0, 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                outputs, target = self._forward_batch(batch)
                loss = criterion(outputs.float(), target)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                total_loss += loss.item() * target.size(0)
                n_samples += target.size(0)

            train_loss = total_loss / n_samples
            val_loss = self.vali(val_data, val_loader, criterion)
            print(f'Epoch {epoch + 1}/{self.args.train_epochs} | '
                  f'Train {train_loss:.6f} | Val {val_loss:.6f} | '
                  f'{time.time() - epoch_start:.1f}s')

            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print('Early stopping')
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        self._load_checkpoint(setting)
        return self.model

        
    def _collect_predictions(self, loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in loader:
                outputs, target = self._forward_batch(batch)
                preds.append(outputs.float().cpu().numpy())
                trues.append(target.cpu().numpy())
        predictions = np.concatenate(preds, axis=0).reshape(-1)
        targets = np.concatenate(trues, axis=0).reshape(-1)
        # scores = (predictions - targets) ** 2
        scores = np.abs(predictions - targets)
        return predictions, targets, scores


    def test(self, setting, test=0):
        if test:
            print('Loading model checkpoint')
            self._load_checkpoint(setting)

        # _, train_loader = self._get_data('train')
        _, val_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')
        # _, _, train_scores = self._collect_predictions(train_loader)
        val_preds, val_trues, val_scores = self._collect_predictions(val_loader)
        preds, trues, test_scores = self._collect_predictions(test_loader)
        
        
        # Bi-AEWMA in the paper operates on ABSOLUTE residuals, not squared MSE.
        use_biaewma = bool(getattr(self.args, 'use_biaewma', False))
        rho = getattr(self.args, 'smooth_rho', None)
        if rho is None:
            rho = 1.0 / (self.args.seq_len + 1)

        print(f"Using Bi-AEWMA: {use_biaewma}, rho: {rho}")
        if use_biaewma:
            # Always smooth validation and test INDEPENDENTLY, in time order.
            val_scores = bi_aewma(val_scores, rho)
            test_scores = bi_aewma(test_scores, rho)
        

        # No cross-split overlap: first prediction corresponds to test row seq_len.
        labels = pd.read_csv(os.path.join(self.args.root_path, 'label.csv')).iloc[:, 0].to_numpy()
        
        labels = (labels[self.args.seq_len:] != 0).astype(np.int64)
        if len(labels) != len(test_scores):
            raise ValueError('Test scores and shifted labels have inconsistent lengths')

        # anomaly_ratio is a percentage, e.g. 1.0 means the top 1% of train errors.
        # threshold = float(np.percentile(val_scores, 100 - self.args.anomaly_ratio))
        threshold = calculate_threshold(
            val_scores=val_scores,
            method=self.args.threshold_method,
            anomaly_ratio=self.args.anomaly_ratio,
            sigma=self.args.threshold_sigma,
        )
        predictions = (test_scores > threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        metrics = {
            'mae': float(np.mean(np.abs(preds - trues))),
            # 'mse': float(test_scores.mean()),
            'threshold': threshold,
            'accuracy': float(accuracy_score(labels, predictions)),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }

        result_dir = os.path.join('results', setting)
        os.makedirs(result_dir, exist_ok=True)
        np.savez_compressed(
        os.path.join(result_dir, 'forecast_scores.npz'),
            pred=preds,
            true=trues,
            # train_scores=train_scores,
            val_scores=val_scores,
            test_scores=test_scores,
            labels=labels,
            predictions=predictions,
            threshold=threshold,
        )
        # 保存评价指标
        with open(os.path.join(result_dir, 'metrics.json'),
                'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        
        self.plot_forecast_results(
            preds=preds,
            trues=trues,
            scores=test_scores,
            threshold=threshold,
            labels=labels,
            save_dir=result_dir,
            start_index=self.args.seq_len,
        )

        print('\nForecasting Anomaly Detection Results:')
        for key, value in metrics.items():
            print(f'{key}: {value:.6f}')

        return metrics

            
    
    def plot_forecast_results(self, preds, trues, scores, threshold, labels,
                            save_dir, start_index=0):
        """Save prediction and anomaly-score plots, aligned to test-data row indices.

        Args:
            preds, trues: 1-D pointwise predicted/observed target values.
            scores: 1-D pointwise anomaly scores.
            threshold: Scalar anomaly-score threshold.
            labels: 1-D binary pointwise anomaly labels (0=normal, 1=anomaly).
            save_dir: Output directory.
            start_index: Original test row index of the first prediction.
        """
        preds = np.asarray(preds).reshape(-1)
        trues = np.asarray(trues).reshape(-1)
        scores = np.asarray(scores).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(bool)
        if not (len(preds) == len(trues) == len(scores) == len(labels)):
            raise ValueError("preds, trues, scores and labels must have the same length")

        os.makedirs(save_dir, exist_ok=True)
        indices = np.arange(start_index, start_index + len(preds))

        # 1. Target value versus prediction.
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(indices, trues, label="Ground truth", linewidth=1.2)
        ax.plot(indices, preds, label="Prediction", linewidth=1.1)
        ax.fill_between(indices, 0, 1, where=labels, transform=ax.get_xaxis_transform(),
                        alpha=0.12, step="mid", label="Labeled anomaly")
        ax.set(xlabel="Test row index", ylabel="Target (scaled)",
            title="Forecast vs. Ground Truth")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "forecast_vs_gt.png"), dpi=200)
        plt.close(fig)

        # 2. Pointwise anomaly score versus validation-calibrated threshold.
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(indices, scores, label="Absolute prediction error", linewidth=1)
        ax.axhline(float(threshold), linestyle="--", linewidth=1.4,
                label=f"Threshold = {float(threshold):.4g}")
        ax.fill_between(indices, 0, 1, where=labels, transform=ax.get_xaxis_transform(),
                        alpha=0.12, step="mid", label="Labeled anomaly")
        ax.set(xlabel="Test row index", ylabel="Anomaly score",
            title="Anomaly Scores vs. Threshold")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "anomaly_scores.png"), dpi=200)
        plt.close(fig)
