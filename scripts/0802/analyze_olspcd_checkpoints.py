#!/usr/bin/env python3
"""Recursively inspect newly trained OLSPCD checkpoints.

This script is read-only: it never modifies a checkpoint.

Example:
    python scripts/0802/analyze_olspcd_checkpoints.py \
        /linyuanping/dzs/codes/TSTTA/checkpoints/0802_olspcd_fixed/OLSPCD
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_ROOT = Path(
    "/linyuanping/dzs/codes/TSTTA/"
    "checkpoints/0802_olspcd_fixed/OLSPCD"
)


@dataclass
class CheckpointReport:
    path: Path
    status: str
    pred_len: int | None
    seq_len: int | None
    enc_in: int | None
    c_out: int | None
    train_ids: Any
    val_ids: Any
    test_ids: Any
    weight_name: str
    actual_shape: tuple[int, ...]
    expected_shape: tuple[int, int] | None
    finite: bool
    max_abs: float
    mean: float
    std: float
    zero_fraction: float
    initial_bound: float
    max_to_initial_ratio: float
    sha256: str
    file_size_mib: float


def load_checkpoint(path: Path) -> Any:
    """Load a trusted local checkpoint on CPU."""
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions without weights_only.
        return torch.load(path, map_location="cpu")


def parse_saved_config(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {}

    saved_config = checkpoint.get("cfg")
    if isinstance(saved_config, dict):
        return saved_config

    if not isinstance(saved_config, str):
        return {}

    try:
        import yaml

        parsed = yaml.safe_load(saved_config)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def nested_get(
    mapping: dict[str, Any],
    *keys: str,
) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def find_model_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint 顶层对象不是字典")

    for key in ("model_state", "model_state_dict", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state

    if checkpoint and all(
        isinstance(name, str) and torch.is_tensor(value)
        for name, value in checkpoint.items()
    ):
        return checkpoint

    raise KeyError("没有找到 model_state/state_dict")


def find_fused_weight(
    state: dict[str, torch.Tensor],
) -> tuple[str, torch.Tensor]:
    preferred_suffixes = (
        "linear.linear_fusion.weight",
        "linear_fusion.weight",
    )

    for suffix in preferred_suffixes:
        matches = [
            (name, value)
            for name, value in state.items()
            if name.endswith(suffix) and torch.is_tensor(value)
        ]
        if len(matches) == 1:
            return matches[0]

    candidates = [
        (name, value)
        for name, value in state.items()
        if name.endswith("weight")
        and torch.is_tensor(value)
        and value.ndim == 2
    ]
    if len(candidates) == 1:
        return candidates[0]

    names = [name for name, _ in candidates]
    raise KeyError(
        "无法唯一确定 OLSPCD 融合权重，候选为："
        f"{names}"
    )


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def inspect_checkpoint(path: Path) -> CheckpointReport:
    checkpoint = load_checkpoint(path)
    config = parse_saved_config(checkpoint)
    state = find_model_state(checkpoint)
    weight_name, weight = find_fused_weight(state)

    pred_len = as_int(nested_get(config, "MODEL", "pred_len"))
    if pred_len is None:
        pred_len = as_int(nested_get(config, "DATA", "PRED_LEN"))

    seq_len = as_int(nested_get(config, "MODEL", "seq_len"))
    if seq_len is None:
        seq_len = as_int(nested_get(config, "DATA", "SEQ_LEN"))

    enc_in = as_int(nested_get(config, "MODEL", "enc_in"))
    c_out = as_int(nested_get(config, "MODEL", "c_out"))
    instance_norm = bool(nested_get(config, "MODEL", "instance_norm"))

    expected_shape = None
    if None not in (pred_len, seq_len, enc_in, c_out):
        per_channel_features = seq_len + (1 if instance_norm else 0)
        expected_shape = (
            c_out * pred_len,
            enc_in * per_channel_features,
        )

    actual_shape = tuple(weight.shape)
    finite = bool(torch.isfinite(weight).all().item())

    weight_float = weight.detach().float().cpu()
    max_abs = float(weight_float.abs().max().item())
    mean = float(weight_float.mean().item())
    std = float(weight_float.std(unbiased=False).item())
    zero_fraction = float((weight_float == 0).float().mean().item())

    input_features = weight.shape[1] if weight.ndim == 2 else 0
    initial_bound = (
        1.0 / math.sqrt(input_features)
        if input_features > 0
        else float("nan")
    )
    max_to_initial_ratio = (
        max_abs / initial_bound
        if initial_bound > 0
        else float("nan")
    )

    if not finite:
        status = "FAIL_NONFINITE"
    elif expected_shape is None:
        status = "WARN_NO_CONFIG"
    elif actual_shape == expected_shape:
        status = "NEW_OK"
    else:
        old_shape = None
        if None not in (pred_len, seq_len, enc_in):
            old_shape = (
                enc_in * pred_len,
                enc_in * (seq_len + 1),
            )
        status = "OLD_LAYOUT" if actual_shape == old_shape else "SHAPE_MISMATCH"

    return CheckpointReport(
        path=path,
        status=status,
        pred_len=pred_len,
        seq_len=seq_len,
        enc_in=enc_in,
        c_out=c_out,
        train_ids=nested_get(config, "DATA", "TRAIN_VEHICLE_IDS"),
        val_ids=nested_get(config, "DATA", "VAL_VEHICLE_IDS"),
        test_ids=nested_get(config, "DATA", "TEST_VEHICLE_IDS"),
        weight_name=weight_name,
        actual_shape=actual_shape,
        expected_shape=expected_shape,
        finite=finite,
        max_abs=max_abs,
        mean=mean,
        std=std,
        zero_fraction=zero_fraction,
        initial_bound=initial_bound,
        max_to_initial_ratio=max_to_initial_ratio,
        sha256=tensor_sha256(weight),
        file_size_mib=path.stat().st_size / (1024 ** 2),
    )


def format_shape(shape: tuple[int, ...] | None) -> str:
    return "?" if shape is None else "x".join(map(str, shape))


def print_report(report: CheckpointReport, root: Path) -> None:
    try:
        relative_path = report.path.relative_to(root)
    except ValueError:
        relative_path = report.path

    print("\n" + "=" * 78)
    print(f"checkpoint : {relative_path}")
    print(f"status     : {report.status}")
    print(
        "task       : "
        f"P{report.pred_len}, seq={report.seq_len}, "
        f"enc_in={report.enc_in}, c_out={report.c_out}"
    )
    print(
        "vehicles   : "
        f"train={report.train_ids}, val={report.val_ids}, "
        f"test={report.test_ids}"
    )
    print(f"weight key : {report.weight_name}")
    print(
        "shape      : "
        f"actual={format_shape(report.actual_shape)}, "
        f"expected={format_shape(report.expected_shape)}"
    )
    print(
        "statistics : "
        f"finite={report.finite}, max_abs={report.max_abs:.6e}, "
        f"mean={report.mean:.6e}, std={report.std:.6e}, "
        f"zero={report.zero_fraction:.2%}"
    )
    print(
        "init check : "
        f"PyTorch initial bound≈{report.initial_bound:.6e}, "
        f"max_abs/bound={report.max_to_initial_ratio:.2f}"
    )
    print(f"sha256     : {report.sha256}")
    print(f"file size  : {report.file_size_mib:.2f} MiB")


def print_summary(reports: list[CheckpointReport]) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = (
        f"{'status':<16} {'P':>4} {'train':<10} {'test':<10} "
        f"{'actual':<14} {'expected':<14} {'max_abs':>12} {'hash':<12}"
    )
    print(header)
    print("-" * len(header))

    for report in sorted(
        reports,
        key=lambda item: (
            item.pred_len if item.pred_len is not None else -1,
            str(item.train_ids),
            str(item.test_ids),
            str(item.path),
        ),
    ):
        print(
            f"{report.status:<16} "
            f"{str(report.pred_len):>4} "
            f"{str(report.train_ids):<10.10} "
            f"{str(report.test_ids):<10.10} "
            f"{format_shape(report.actual_shape):<14} "
            f"{format_shape(report.expected_shape):<14} "
            f"{report.max_abs:>12.4e} "
            f"{report.sha256[:12]:<12}"
        )

    status_counts: dict[str, int] = {}
    for report in reports:
        status_counts[report.status] = status_counts.get(report.status, 0) + 1

    print("\n状态统计：")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    duplicate_hashes: dict[str, list[Path]] = {}
    for report in reports:
        duplicate_hashes.setdefault(report.sha256, []).append(report.path)
    duplicates = {
        digest: paths
        for digest, paths in duplicate_hashes.items()
        if len(paths) > 1
    }
    if duplicates:
        print("\n[WARN] 发现完全相同的权重：")
        for digest, paths in duplicates.items():
            print(f"  {digest[:12]}:")
            for path in paths:
                print(f"    - {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="分析 OLSPCD checkpoint 是否为新结构并检查权重。"
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"checkpoint 根目录，默认：{DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--pattern",
        default="checkpoint_best.pth",
        help="递归搜索的文件名，默认 checkpoint_best.pth",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"目录不存在：{root}")

    checkpoint_paths = sorted(root.rglob(args.pattern))
    if not checkpoint_paths:
        parser.error(
            f"在 {root} 下没有找到 {args.pattern}"
        )

    print(f"扫描目录：{root}")
    print(f"checkpoint 数量：{len(checkpoint_paths)}")

    reports: list[CheckpointReport] = []
    failures = 0
    for path in checkpoint_paths:
        try:
            report = inspect_checkpoint(path)
        except Exception as error:
            failures += 1
            print("\n" + "=" * 78)
            print(f"checkpoint : {path}")
            print(f"status     : READ_ERROR")
            print(f"error      : {type(error).__name__}: {error}")
            continue

        reports.append(report)
        print_report(report, root)

    if reports:
        print_summary(reports)

    bad_statuses = {
        "FAIL_NONFINITE",
        "OLD_LAYOUT",
        "SHAPE_MISMATCH",
    }
    invalid_reports = sum(
        report.status in bad_statuses
        for report in reports
    )

    print("\n判定说明：")
    print("  NEW_OK         新的两目标 OLSPCD 权重形状正确")
    print("  OLD_LAYOUT     检测到旧的全通道/追加 stdev 结构")
    print("  SHAPE_MISMATCH 权重形状与 checkpoint 配置不一致")
    print("  FAIL_NONFINITE 权重包含 NaN 或 Inf")
    print("  WARN_NO_CONFIG checkpoint 中缺少可解析配置")

    return 1 if failures or invalid_reports else 0


if __name__ == "__main__":
    sys.exit(main())
