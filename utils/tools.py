"""Utilities for UAV anomaly detection experiments (TSLib style).

Used by:
    exp.exp_forecasting.Exp_Forecasting
    exp.exp_anomaly_detection.Exp_Anomaly_Detection

Only checkpoint-based early stopping and epoch-based learning-rate schedules
are required. Anomaly thresholding and metrics belong in the Exp classes.
"""

import math
import os

import torch


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Each improvement saves ``<path>/checkpoint.pth`` as a PyTorch state_dict,
    matching the _load_checkpoint methods in the two Exp classes.

    Args:
        patience: Number of consecutive non-improving epochs to tolerate.
        verbose: Print checkpoint saving messages.
        delta: Minimum decrease in validation loss to count as improvement.
    """

    def __init__(self, patience=7, verbose=False, delta=0.0):
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if delta < 0:
            raise ValueError("delta must be >= 0")

        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model, path):
        val_loss = float(val_loss)
        if not math.isfinite(val_loss):
            raise ValueError(f"Validation loss must be finite, got {val_loss}")

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, "checkpoint.pth")
        temporary_path = checkpoint_path + ".tmp"

        # A temporary file prevents leaving a partially written checkpoint.
        torch.save(model.state_dict(), temporary_path)
        os.replace(temporary_path, checkpoint_path)

        if self.verbose:
            print(
                f"Validation loss decreased "
                f"({self.val_loss_min:.6f} --> {val_loss:.6f}). "
                # f"Saving model to {checkpoint_path}"
            )
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust the optimizer LR at the END of epoch ``epoch`` (1-based).

    Supported --lradj values:
        type1:    TSLib-style: half the initial LR at epoch 2, 3, ...
        type2:    TSLib-style milestone schedule, scaled by the initial LR
        type3:    initial LR for epochs < 3, then multiply by 0.9 each epoch
        cosine:   cosine decay from the initial LR to zero
        constant: keep the current LR unchanged

    Example: with type1, lr=1e-4, the values after epochs 1, 2, 3
    are 1e-4, 5e-5, 2.5e-5 respectively.
    """
    if epoch < 1:
        raise ValueError("epoch must be >= 1")

    initial_lr = float(args.learning_rate)
    if initial_lr <= 0:
        raise ValueError("learning_rate must be positive")

    schedule = args.lradj
    if schedule == "constant":
        return
    elif schedule == "type1":
        lr = initial_lr * (0.5 ** (epoch - 1))
    elif schedule == "type2":
        # Original TSLib milestones for initial_lr=1e-4, expressed as
        # multipliers so other user-provided learning rates also work.
        factors = {
            2: 0.5,
            4: 0.1,
            6: 0.05,
            8: 0.01,
            10: 0.005,
            15: 0.001,
            20: 0.0005,
        }
        if epoch not in factors:
            return
        lr = initial_lr * factors[epoch]
    elif schedule == "type3":
        lr = initial_lr if epoch < 3 else initial_lr * (0.9 ** (epoch - 3))
    elif schedule == "cosine":
        if args.train_epochs <= 0:
            raise ValueError("train_epochs must be positive for cosine schedule")
        lr = 0.5 * initial_lr * (1 + math.cos(math.pi * epoch / args.train_epochs))
    else:
        raise ValueError(
            f"Unknown lradj={schedule!r}; "
            "expected type1, type2, type3, cosine, or constant"
        )

    for group in optimizer.param_groups:
        group["lr"] = lr
    print(f"Updating learning rate to {lr:.8g}")
