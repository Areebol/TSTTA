"""One-step BP/ZO memory benchmark on a real ETTh2 -> ETTh1 batch."""

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import get_cfg_defaults
from datasets.build import update_cfg_from_dataset
from datasets.loader import get_domain_shift_dataloader
from models.build import build_model, load_best_model
from models.forecast import forecast
from tta.zo_utils import OnlineMemoryProfiler, zero_order_step
from utils.misc import prepare_inputs, set_seeds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("bp", "zo"), required=True)
    parser.add_argument("--model", default="PatchTST")
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--perturbation-scale", type=float, default=1e-3)
    parser.add_argument("--sp-avg", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_cfg_defaults()
    cfg.SEED = args.seed
    cfg.DATA.NAME = "ETTh2"
    cfg.DATA.DOMAIN_SHIFT_TARGET = "ETTh1"
    cfg.DATA.PRED_LEN = args.pred_len
    cfg.MODEL.NAME = args.model
    cfg.MODEL.pred_len = args.pred_len
    cfg.TRAIN.CHECKPOINT_DIR = args.checkpoint_dir
    cfg.DATA_LOADER.NUM_WORKERS = 0
    cfg.TEST.BATCH_SIZE = args.batch_size
    cfg.TTA.DOMAIN_SHIFT = True
    update_cfg_from_dataset(cfg, cfg.DATA.NAME)
    set_seeds(args.seed)

    model = load_best_model(cfg, build_model(cfg))
    model.eval()
    inputs = prepare_inputs(next(iter(get_domain_shift_dataloader(cfg))))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=1e-5)

    def loss_closure():
        pred, ground_truth = forecast(cfg, inputs, model)
        return F.mse_loss(pred, ground_truth)

    # Warm up kernels and allocate the immutable batch before measuring an update.
    with torch.no_grad():
        loss_closure()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    profiler = OnlineMemoryProfiler()
    allocated_before = torch.cuda.memory_allocated()
    profiler.begin_step()
    with profiler.track_saved_tensors():
        if args.method == "zo":
            loss = zero_order_step(
                optimizer,
                loss_closure,
                perturbation_scale=args.perturbation_scale,
                sp_avg=args.sp_avg,
            )
        else:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_closure()
            loss.backward()
            optimizer.step()
    profiler.end_step()

    report = {
        "method": args.method,
        "model": args.model,
        "source": "ETTh2",
        "target": "ETTh1",
        "pred_len": args.pred_len,
        "batch_size": args.batch_size,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "allocated_before_mb": allocated_before / (1024.0 ** 2),
        "loss": float(loss),
        **profiler.summary(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
