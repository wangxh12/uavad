"""Evaluate a trained relation model with only its cross-variable gates zeroed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.alfa import load_prepared
from models.crossad_relation import CrossADRelation
from scripts.crossad_alfa import ROOT, evaluate, model_from_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/crossad_alfa/relation_v1")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/crossad_alfa/relation_v1_gate_zero")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Ablation output exists: {args.output_dir}")
    device = torch.device(args.device)
    checkpoint = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
    model = model_from_settings(checkpoint["settings"]).to(device)
    if not isinstance(model, CrossADRelation):
        raise ValueError("Expected a CrossAD relation checkpoint")
    model.load_state_dict(checkpoint["model"])
    gates = torch.tanh(model.mixer.gate).detach().cpu().numpy()
    with torch.no_grad():
        model.mixer.gate.zero_()
    args.output_dir.mkdir(parents=True)
    record = {"source": str(args.run_dir), "epoch": checkpoint["epoch"],
              "ablation": "Set only the trained mixer residual gates to zero at evaluation",
              "original_gate_mean_abs": float(abs(gates).mean()),
              "original_gate_max_abs": float(abs(gates).max()),
              "settings": checkpoint["settings"]}
    (args.output_dir / "ablation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    flights = load_prepared(Path(checkpoint["settings"]["data"]))
    result = evaluate(model, flights, device, checkpoint["settings"]["batch_size"], args.output_dir)
    print(json.dumps({"gate_mean_abs": record["original_gate_mean_abs"],
                      "overall": result["overall"], "fault_events": result["fault_events"]}, indent=2))


if __name__ == "__main__":
    main()
