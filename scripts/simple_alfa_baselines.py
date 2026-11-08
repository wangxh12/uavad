"""Train and evaluate fixed PCA, MLP-AE and LSTM-AE ALFA baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from data.alfa import FEATURES, load_prepared
from models.simple_ae import LSTMAE, MLPAE, PCABaseline
from scripts.crossad_alfa import (ROOT, evaluate, loader, restore_rng, save_checkpoint,
                                  set_seed, validation_loss)


CONFIGS = {
    "pca8": {"features": len(FEATURES), "components": 8},
    "mlp_ae": {"length": 192, "features": len(FEATURES), "hidden": 256, "latent": 64},
    "lstm_ae": {"features": len(FEATURES), "hidden": 64, "latent": 32},
}
MODELS = {"pca8": PCABaseline, "mlp_ae": MLPAE, "lstm_ae": LSTMAE}


def build(settings: dict) -> torch.nn.Module:
    return MODELS[settings["architecture"]](**settings["model"])


def fit_pca(model: PCABaseline, flights: list) -> dict:
    normal = np.concatenate([flight.values[:flight.normal_end]
                             for flight in flights if flight.split == "train"])
    pca = PCA(n_components=model.basis.shape[0], svd_solver="full").fit(normal)
    with torch.no_grad():
        model.mean.copy_(torch.as_tensor(pca.mean_, device=model.mean.device))
        model.basis.copy_(torch.as_tensor(pca.components_, device=model.basis.device))
    return {"fit_normal_points": len(normal),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist()}


def experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    flights = load_prepared(args.processed_dir)
    device = torch.device(args.device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "evaluate":
        checkpoint = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
        model = build(checkpoint["settings"]).to(device)
        model.load_state_dict(checkpoint["model"])
        result = evaluate(model, flights, device, args.batch_size, args.run_dir)
        print(json.dumps(result, indent=2), flush=True)
        return

    if args.model is None:
        raise ValueError("--model is required for training")
    model = MODELS[args.model](**CONFIGS[args.model]).to(device)
    settings = {"architecture": args.model, "model": CONFIGS[args.model], "seed": args.seed,
                "data": str(args.processed_dir.resolve()), "batch_size": args.batch_size,
                "max_batches": args.max_batches, "lr": 1e-3 if args.model != "pca8" else None,
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                        text=True).strip(),
                "model_sha256": hashlib.sha256((ROOT / "models/simple_ae.py").read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256(
                    (args.processed_dir / "manifest.json").read_bytes()).hexdigest()}
    manifest = json.loads((args.processed_dir / "manifest.json").read_text(encoding="utf-8"))
    settings["normalization"] = manifest.get("normalization", "zscore")

    latest = args.run_dir / "latest.pt"
    if not args.resume and (latest.exists() or (args.run_dir / "best.pt").exists()):
        raise FileExistsError(f"Run already exists: {args.run_dir}")
    if not args.resume:
        (args.run_dir / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    if args.model == "pca8":
        if args.resume:
            raise ValueError("PCA is fitted in one pass and has no resume state")
        diagnostics = fit_pca(model, flights)
        (args.run_dir / "pca_fit.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        torch.save({"epoch": 0, "model": model.state_dict(), "settings": settings},
                   args.run_dir / "best.pt")
        print(json.dumps(diagnostics, indent=2), flush=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_loss, bad_epochs, first_epoch = float("inf"), 0, 0
        if args.resume:
            state = torch.load(latest, map_location=device, weights_only=False)
            if state["settings"] != settings:
                raise ValueError("Resume settings differ from the saved run")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            best_loss, bad_epochs, first_epoch = (state["best_loss"], state["bad_epochs"],
                                                  state["epoch"] + 1)
            restore_rng(state)
        train_batches = loader(flights, "train", args.batch_size)
        val_batches = loader(flights, "val", args.batch_size)
        for epoch in range(first_epoch, args.epochs):
            model.train()
            total, count = 0.0, 0
            for batch_number, (values, _, _) in enumerate(train_batches):
                values = values.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = model(values)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                total += loss.item() * len(values)
                count += len(values)
                if args.max_batches is not None and batch_number + 1 >= args.max_batches:
                    break
            val_loss = validation_loss(model, val_batches, device)
            if val_loss < best_loss:
                best_loss, bad_epochs = val_loss, 0
                save_checkpoint(args.run_dir / "best.pt", epoch, model, optimizer,
                                best_loss, bad_epochs, settings)
            else:
                bad_epochs += 1
            save_checkpoint(latest, epoch, model, optimizer, best_loss, bad_epochs, settings)
            print(f"epoch={epoch + 1} train_loss={total / count:.6f} val_loss={val_loss:.6f} "
                  f"best={best_loss:.6f} bad_epochs={bad_epochs}", flush=True)
            if bad_epochs >= 3:
                print("early stopping", flush=True)
                break

    if args.evaluate_after:
        checkpoint = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        result = evaluate(model, flights, device, args.batch_size, args.run_dir)
        print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--model", choices=tuple(CONFIGS))
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed/alfa_crossad")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-after", action="store_true")
    args = parser.parse_args()
    if args.command == "evaluate" and (args.model is not None or args.resume):
        parser.error("evaluate reads the architecture from best.pt")
    experiment(args)


if __name__ == "__main__":
    main()
