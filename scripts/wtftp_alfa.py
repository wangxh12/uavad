"""Train/evaluate the WTFTP Haar backbone as an ALFA reconstruction detector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from data.alfa import load_prepared
from models.wtftp_reconstruction import WTFTPReconstruction, WTFTPReconstructionConfig
from scripts.crossad_alfa import evaluate, loader, restore_rng, save_checkpoint, set_seed, validation_loss


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = "d6293e23ad83b29861b781f84737aa7a59cd20c1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def settings_for(args: argparse.Namespace) -> dict:
    manifest = json.loads((args.processed_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != "within_flight_chronological_v1":
        raise ValueError("Expected the within-flight ALFA split")
    if manifest["windowing"]["expected_window"] != args.window_size:
        raise ValueError("Window size differs from the prepared dataset")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    return {
        "architecture": "wtftp_haar_reconstruction", "model": WTFTPReconstructionConfig(
            seq_len=args.window_size).to_dict(),
        "training_objective": "MSE(low) + MSE(high); original WTFTP defaults, no temporal loss",
        "score": "per-point and per-feature squared reconstruction error",
        "paper_task_adaptation": "same-window 18-channel reconstruction instead of trajectory forecast",
        "official_reference_commit": REFERENCE_COMMIT, "git_revision": revision,
        "model_sha256": sha256(ROOT / "models/wtftp_reconstruction.py"),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "manifest_sha256": sha256(args.processed_dir / "manifest.json"),
        "normalization": manifest["normalization"], "data": str(args.processed_dir.resolve()),
        "seed": args.seed, "batch_size": args.batch_size, "max_batches": args.max_batches,
        "learning_rate": args.learning_rate, "patience": args.patience,
    }


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    settings = settings_for(args)
    flights = load_prepared(args.processed_dir)
    device = torch.device(args.device)
    if args.command == "evaluate":
        checkpoint = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
        if checkpoint["settings"] != settings:
            raise ValueError("Checkpoint settings differ from the requested data or source code")
        model = WTFTPReconstruction(WTFTPReconstructionConfig(**settings["model"])).to(device)
        model.load_state_dict(checkpoint["model"])
        print(json.dumps(evaluate(model, flights, device, args.batch_size, args.run_dir), indent=2), flush=True)
        return

    args.run_dir.mkdir(parents=True, exist_ok=True)
    latest = args.run_dir / "latest.pt"
    model = WTFTPReconstruction(WTFTPReconstructionConfig(**settings["model"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_loss, bad_epochs, first_epoch = float("inf"), 0, 0
    if args.resume:
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint["settings"] != settings:
            raise ValueError("Resume settings differ from the saved run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_loss, bad_epochs, first_epoch = (checkpoint["best_loss"], checkpoint["bad_epochs"],
                                               checkpoint["epoch"] + 1)
        restore_rng(checkpoint)
    else:
        if any(args.run_dir.iterdir()):
            raise FileExistsError(f"Use a new run directory: {args.run_dir}")
        (args.run_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        with (args.run_dir / "training.csv").open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(["epoch", "train_loss", "val_loss", "best_val_loss", "lr"])

    train_batches = loader(flights, "train", args.batch_size, args.window_size)
    val_batches = loader(flights, "val", args.batch_size, args.window_size)
    for epoch in range(first_epoch, args.epochs):
        lr = args.learning_rate * 0.5 ** (epoch // 10)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        total, points = 0.0, 0
        for batch_index, (values, _, _) in enumerate(train_batches):
            values = values.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(values)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(values)
            points += len(values)
            if args.max_batches is not None and batch_index + 1 >= args.max_batches:
                break
        val_loss = validation_loss(model, val_batches, device)
        if not np.isfinite(val_loss):
            raise RuntimeError("Nonfinite validation loss")
        if val_loss < best_loss:
            best_loss, bad_epochs = val_loss, 0
            save_checkpoint(args.run_dir / "best.pt", epoch, model, optimizer,
                            best_loss, bad_epochs, settings)
        else:
            bad_epochs += 1
        save_checkpoint(latest, epoch, model, optimizer, best_loss, bad_epochs, settings)
        with (args.run_dir / "training.csv").open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow([epoch + 1, total / points, val_loss, best_loss, lr])
        print(f"epoch={epoch + 1} train_loss={total / points:.6f} val_loss={val_loss:.6f} "
              f"best={best_loss:.6f} bad_epochs={bad_epochs}", flush=True)
        if bad_epochs >= args.patience:
            print("early stopping", flush=True)
            break
    if args.evaluate_after:
        checkpoint = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        print(json.dumps(evaluate(model, flights, device, args.batch_size, args.run_dir), indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed/alfa_within_flight_v1")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/wtftp_alfa/within_flight_w96")
    parser.add_argument("--window-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-after", action="store_true")
    args = parser.parse_args()
    if (args.window_size < 2 or args.window_size % 2 or args.batch_size < 1 or
            args.epochs < 1 or args.patience < 1 or args.learning_rate <= 0 or
            args.max_batches is not None and args.max_batches < 1):
        parser.error("Invalid window, batch, epoch, patience, learning rate, or batch limit")
    if args.command == "evaluate" and (args.resume or args.evaluate_after):
        parser.error("--resume and --evaluate-after apply only to training")
    run(args)


if __name__ == "__main__":
    main()
