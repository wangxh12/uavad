"""Prepare, train, and evaluate the ALFA CrossAD reconstruction baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from data.alfa import FEATURES, FlightWindows, load_prepared, prepare
from models.crossad import CrossAD, CrossADConfig
from models.crossad_alternatives import CrossADDual, CrossADCrossScale, DualConfig, CrossScaleConfig
from models.crossad_intervar import CrossADInterVar, InterVarConfig
from models.crossad_relation import CrossADRelation, RelationConfig


ROOT = Path(__file__).resolve().parents[1]


def model_from_settings(settings: dict) -> CrossAD:
    config = dict(settings["model"])
    legacy_intervar = config.pop("cross_channel", False)
    architecture = settings.get("architecture", "crossad_intervar" if legacy_intervar else "crossad")
    if architecture == "crossad_intervar":
        return CrossADInterVar(InterVarConfig(**config))
    if architecture == "crossad_relation":
        return CrossADRelation(RelationConfig(**config))
    if architecture == "crossad_dual":
        return CrossADDual(DualConfig(**config))
    if architecture == "crossad_crossscale":
        return CrossADCrossScale(CrossScaleConfig(**config))
    if architecture == "crossad":
        return CrossAD(CrossADConfig(**config))
    raise ValueError(f"Unknown checkpoint architecture: {architecture}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(path: Path, epoch: int, model: CrossAD, optimizer: torch.optim.Optimizer,
                    best_loss: float, bad_epochs: int, settings: dict) -> None:
    state = {
        "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "best_loss": best_loss, "bad_epochs": bad_epochs, "settings": settings,
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_rng(state: dict) -> None:
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])


def loader(flights: list, split: str, batch_size: int, window: int = 192) -> DataLoader:
    windows = FlightWindows(flights, split, window=window)
    return DataLoader(windows, batch_size=batch_size, shuffle=split == "train", num_workers=0)


@torch.no_grad()
def validation_loss(model: CrossAD, batches: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for values, _, _ in batches:
        values = values.to(device)
        loss = model(values)
        total += loss.item() * len(values)
        count += len(values)
    return total / count


@torch.no_grad()
def score_flights(model: CrossAD, flights: list, split: str, device: torch.device,
                  batch_size: int) -> list[tuple[object, np.ndarray]]:
    window = getattr(getattr(model, "config", None), "seq_len", 192)
    batches = loader(flights, split, batch_size, window=window)
    model.eval()
    accum = [np.zeros((f.normal_end if split == "val" else len(f.labels), len(FEATURES)),
                      dtype=np.float64) for f in batches.dataset.flights]
    counts = [np.zeros(len(scores), dtype=np.int32) for scores in accum]
    for values, indexes, starts in batches:
        scores = model.score(values.to(device)).cpu().numpy()
        for scores_i, index, start in zip(scores, indexes.tolist(), starts.tolist()):
            accum[index][start:start + len(scores_i)] += scores_i
            counts[index][start:start + len(scores_i)] += 1
    for count in counts:
        if not np.all(count > 0):
            raise RuntimeError("Window scoring left some points uncovered")
    return [(f, (scores / count[:, None]).astype(np.float32))
            for f, scores, count in zip(batches.dataset.flights, accum, counts)]


def fault_type(name: str) -> str:
    types = [kind for kind in ("engine", "elevator", "aileron", "rudder") if kind in name]
    return "+".join(types) if types else "none"


def metrics_for(rows: list[dict]) -> dict:
    labels = np.concatenate([row["labels"] for row in rows])
    scores = np.concatenate([row["scores"] for row in rows])
    predictions = np.concatenate([row["predictions"] for row in rows])
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0)
    return {
        "points": int(len(labels)), "anomaly_points": int(labels.sum()),
        "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "auprc": float(average_precision_score(labels, scores)) if labels.sum() else None,
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }


def evaluate(model: CrossAD, flights: list, device: torch.device, batch_size: int,
             run_dir: Path) -> dict:
    validation = score_flights(model, flights, "val", device, batch_size)
    threshold = float(np.quantile(np.concatenate([s.mean(1) for _, s in validation]), 0.99))
    test = score_flights(model, flights, "test", device, batch_size)
    score_dir = run_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    rows, report = [], []
    for flight, features in test:
        scores = features.mean(1)
        predictions = (scores > threshold).astype(np.int8)
        labels = flight.labels
        onset = flight.normal_end
        detected = np.flatnonzero(predictions[onset:]) if onset < len(labels) else np.array([], dtype=int)
        delay = float(flight.time[onset + detected[0]] - flight.time[onset]) if len(detected) else None
        rows.append({"labels": labels, "scores": scores, "predictions": predictions,
                     "type": fault_type(flight.name)})
        flight_metrics = metrics_for([rows[-1]])
        report.append({"flight": flight.name, "group": flight.group, "fault_type": fault_type(flight.name),
                       "anomaly_points": int(labels.sum()), "detected": bool(len(detected)),
                       "first_alarm_delay_s": delay, "false_alarm_points": int(predictions[:onset].sum()),
                       "normal_flight_false_alarm": bool(predictions.any()) if onset == len(labels) else None,
                       **{key: flight_metrics[key] for key in
                          ("auroc", "auprc", "precision", "recall", "f1")}})
        np.savez_compressed(score_dir / f"{Path(flight.name).stem}.npz", time_s=flight.time,
                            labels=labels, feature_scores=features, scores=scores,
                            predictions=predictions)
    with (run_dir / "flights.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    normal = [r for r in report if r["anomaly_points"] == 0]
    faults = [r for r in report if r["anomaly_points"] > 0]
    result = {
        "protocol": (f"offline {getattr(getattr(model, 'config', None), 'seq_len', 192)}-point windows; "
                     "no point adjustment; validation-only 99th percentile threshold"),
        "threshold": threshold, "overall": metrics_for(rows),
        "normal_flights": {"count": len(normal),
                           "with_false_alarm": sum(r["normal_flight_false_alarm"] for r in normal)},
        "fault_events": {"count": len(faults), "detected": sum(r["detected"] for r in faults),
                         "delays_s": [r["first_alarm_delay_s"] for r in faults]},
        "by_fault_type": {kind: metrics_for([r for r in rows if r["type"] == kind])
                          for kind in sorted({r["type"] for r in rows})},
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    flights = load_prepared(args.processed_dir)
    device = torch.device(args.device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "evaluate":
        checkpoint = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
        model = model_from_settings(checkpoint["settings"]).to(device)
        model.load_state_dict(checkpoint["model"])
        print(json.dumps(evaluate(model, flights, device, args.batch_size, args.run_dir), indent=2), flush=True)
        return

    constructors = {"crossad": lambda: CrossAD(CrossADConfig(seq_len=args.window_size)),
                    "crossad_intervar": lambda: CrossADInterVar(InterVarConfig(seq_len=args.window_size)),
                    "crossad_relation": lambda: CrossADRelation(RelationConfig(seq_len=args.window_size)),
                    "crossad_dual": lambda: CrossADDual(DualConfig(seq_len=args.window_size)),
                    "crossad_crossscale": lambda: CrossADCrossScale(CrossScaleConfig(seq_len=args.window_size))}
    model = constructors[args.model]().to(device)
    if args.init_checkpoint is not None:
        source = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        baseline_config = CrossADConfig(seq_len=args.window_size).to_dict()
        if (source["settings"].get("architecture", "crossad") != "crossad" or
                source["settings"]["model"] != baseline_config):
            raise ValueError("Initialization requires a checkpoint of the same original CrossAD backbone")
        missing, unexpected = model.load_state_dict(source["model"], strict=False)
        if unexpected or not missing or not all(
                name.startswith(("other_channels.", "conditional_head.", "gate",
                                 "cross_scale.")) for name in missing):
            raise ValueError(f"Unexpected warm-start state: missing={missing}, unexpected={unexpected}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    settings = {"seed": args.seed, "model": model.config.to_dict(), "batch_size": args.batch_size,
                "max_batches": args.max_batches, "data": str(args.processed_dir),
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                        text=True).strip()}
    manifest = json.loads((args.processed_dir / "manifest.json").read_text(encoding="utf-8"))
    if "normalization" in manifest:
        settings["normalization"] = manifest["normalization"]
    if args.model != "crossad":
        settings["architecture"] = args.model
    if args.model == "crossad_relation":
        settings["model_sha256"] = hashlib.sha256(
            (ROOT / "models/crossad_relation.py").read_bytes()).hexdigest()
        settings["manifest_sha256"] = hashlib.sha256(
            (args.processed_dir / "manifest.json").read_bytes()).hexdigest()
    if args.model in ("crossad_dual", "crossad_crossscale"):
        settings["model_sha256"] = hashlib.sha256(
            (ROOT / "models/crossad_alternatives.py").read_bytes()).hexdigest()
        settings["manifest_sha256"] = hashlib.sha256(
            (args.processed_dir / "manifest.json").read_bytes()).hexdigest()
    if args.init_checkpoint is not None:
        settings["init_checkpoint"] = str(args.init_checkpoint.resolve())
        settings["init_checkpoint_sha256"] = hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest()
    latest = args.run_dir / "latest.pt"
    best_loss, bad_epochs, first_epoch = float("inf"), 0, 0
    if args.resume:
        state = torch.load(latest, map_location=device, weights_only=False)
        for field in ("init_checkpoint", "init_checkpoint_sha256"):
            if field in state["settings"]:
                settings[field] = state["settings"][field]
        if settings != state["settings"]:
            raise ValueError("Resume settings differ from the saved run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best_loss, bad_epochs, first_epoch = state["best_loss"], state["bad_epochs"], state["epoch"] + 1
        restore_rng(state)
    else:
        if latest.exists():
            raise FileExistsError(f"Existing run; specify --resume or another --run-dir: {latest}")
        (args.run_dir / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    train_batches = loader(flights, "train", args.batch_size, window=model.config.seq_len)
    val_batches = loader(flights, "val", args.batch_size, window=model.config.seq_len)
    for epoch in range(first_epoch, args.epochs):
        model.train()
        losses = []
        for batch_number, (values, _, _) in enumerate(train_batches):
            values = values.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(values)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if args.max_batches is not None and batch_number + 1 >= args.max_batches:
                break
        val_loss = validation_loss(model, val_batches, device)
        if val_loss < best_loss:
            best_loss, bad_epochs = val_loss, 0
            save_checkpoint(args.run_dir / "best.pt", epoch, model, optimizer, best_loss, bad_epochs, settings)
        else:
            bad_epochs += 1
        if epoch + 1 in {2, 4, 6, 8, 10, 15, 20}:
            rates = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
            for group in optimizer.param_groups:
                group["lr"] = rates[epoch + 1]
        save_checkpoint(latest, epoch, model, optimizer, best_loss, bad_epochs, settings)
        print(f"epoch={epoch + 1} train_loss={np.mean(losses):.6f} val_loss={val_loss:.6f} "
              f"best={best_loss:.6f} bad_epochs={bad_epochs}", flush=True)
        if bad_epochs >= 3:
            print("early stopping", flush=True)
            break
    if args.evaluate_after:
        state = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        print(json.dumps(evaluate(model, flights, device, args.batch_size, args.run_dir), indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/alfa_20hz_l_18f")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed/alfa_crossad")
    parser.add_argument("--normalization", choices=("zscore", "minmax"),
                        help="Preprocessing method for prepare; train/evaluate read the manifest")
    parser.add_argument("--model", choices=("crossad", "crossad_intervar", "crossad_relation",
                                            "crossad_dual", "crossad_crossscale"), default="crossad",
                        help="Model architecture for train; evaluate reads the checkpoint")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Warm-start a new multivariate model from an original CrossAD checkpoint")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/crossad_alfa/main")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--window-size", type=int, default=192,
                        help="Points per window for training; evaluate reads the checkpoint")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-after", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.model != "crossad" or args.init_checkpoint is not None:
            parser.error("--model and --init-checkpoint apply only to train")
        manifest = prepare(args.raw_dir, args.processed_dir, args.normalization or "zscore")
        print(json.dumps({"flights": len(manifest["flights"]), "excluded": manifest["excluded"],
                          "groups": {k: len(v) for k, v in manifest["groups"].items()}}, indent=2))
    else:
        if args.window_size < 1:
            parser.error("--window-size must be positive")
        if args.normalization is not None:
            parser.error("--normalization applies only to prepare; train/evaluate read the processed manifest")
        if args.command == "evaluate" and args.model != "crossad":
            parser.error("evaluate loads the model architecture from the checkpoint")
        if args.init_checkpoint is not None and (args.command != "train" or args.resume or
                                                 args.model not in ("crossad_dual", "crossad_crossscale")):
            parser.error("--init-checkpoint requires a fresh crossad_dual or crossad_crossscale training run")
        experiment(args)


if __name__ == "__main__":
    main()
