#!/usr/bin/env python3
"""Compare two independently solved OLSPCD checkpoints.

This compares tensor contents rather than whole-file hashes because torch.save
container metadata is not the reproducibility target. By default the command
fails unless the fused weights are bitwise identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


WEIGHT_KEYS = (
    "linear.linear_fusion.weight",
    "linear_fusion.weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_a", type=Path)
    parser.add_argument("checkpoint_b", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument(
        "--allow-close",
        action="store_true",
        help="Return success for allclose weights when exact equality fails.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        checkpoint = torch.load(
            str(path), map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    return checkpoint


def find_weight(checkpoint: Mapping[str, Any]) -> tuple[str, torch.Tensor]:
    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("model_state is not a mapping")
    for key in WEIGHT_KEYS:
        value = state.get(key)
        if torch.is_tensor(value):
            return key, value.detach().cpu().contiguous()
    raise KeyError(f"None of the OLSPCD weight keys were found: {WEIGHT_KEYS}")


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    checkpoint_a = load_checkpoint(args.checkpoint_a)
    checkpoint_b = load_checkpoint(args.checkpoint_b)
    key_a, weight_a = find_weight(checkpoint_a)
    key_b, weight_b = find_weight(checkpoint_b)

    if weight_a.shape != weight_b.shape:
        raise SystemExit(
            f"[FAIL] shape mismatch: {tuple(weight_a.shape)} vs "
            f"{tuple(weight_b.shape)}"
        )
    if weight_a.dtype != weight_b.dtype:
        raise SystemExit(
            f"[FAIL] dtype mismatch: {weight_a.dtype} vs {weight_b.dtype}"
        )

    difference = (weight_a - weight_b).abs().double()
    exact = bool(torch.equal(weight_a, weight_b))
    close = bool(
        torch.allclose(
            weight_a,
            weight_b,
            rtol=args.rtol,
            atol=args.atol,
        )
    )

    metadata_a = checkpoint_a.get("fit_metadata")
    metadata_b = checkpoint_b.get("fit_metadata")
    if not isinstance(metadata_a, Mapping) or not isinstance(metadata_b, Mapping):
        raise SystemExit("[FAIL] one or both checkpoints lack fit_metadata")

    sha_a = tensor_sha256(weight_a)
    sha_b = tensor_sha256(weight_b)
    embedded_sha_a = metadata_a.get("saved_weight_sha256")
    embedded_sha_b = metadata_b.get("saved_weight_sha256")

    report = {
        "checkpoint_a": str(args.checkpoint_a.resolve()),
        "checkpoint_b": str(args.checkpoint_b.resolve()),
        "weight_key_a": key_a,
        "weight_key_b": key_b,
        "shape": tuple(weight_a.shape),
        "dtype": str(weight_a.dtype),
        "exactly_equal": exact,
        "allclose": close,
        "max_absolute_difference": float(difference.max().item()),
        "mean_absolute_difference": float(difference.mean().item()),
        "sha256_a": sha_a,
        "sha256_b": sha_b,
        "embedded_sha256_a_valid": embedded_sha_a == sha_a,
        "embedded_sha256_b_valid": embedded_sha_b == sha_b,
        "fit_metadata_equal": dict(metadata_a) == dict(metadata_b),
        "fit_metadata_a": dict(metadata_a),
        "fit_metadata_b": dict(metadata_b),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    hashes_valid = (
        report["embedded_sha256_a_valid"]
        and report["embedded_sha256_b_valid"]
    )
    metadata_equal = report["fit_metadata_equal"]
    if exact and hashes_valid and metadata_equal:
        print("[PASS] OLSPCD weights are bitwise reproducible.")
        return
    if args.allow_close and close and hashes_valid and metadata_equal:
        print("[PASS] OLSPCD weights are reproducible within tolerance.")
        return
    raise SystemExit("[FAIL] OLSPCD weights are not reproducible.")


if __name__ == "__main__":
    main()
