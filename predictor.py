import os
from typing import Dict, Optional
import pickle

import wandb
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.loader import get_val_dataloader, get_test_dataloader, get_domain_shift_dataloader
from utils.misc import prepare_inputs
from utils.misc import mkdir
from config import get_norm_method
from tta.utils import save_summary_to_csv, save_tta_results
from device_manager import global_device

class Predictor:
    def __init__(self, cfg, model, norm_module: Optional[torch.nn.Module] = None):
        self.cfg = cfg

        self.model = model
        self.norm_method = get_norm_method(cfg)
        self.norm_module = norm_module

        cfg.TRAIN.SHUFFLE, cfg.TRAIN.DROP_LAST = False, False
        if cfg.TTA.DOMAIN_SHIFT:
            self.test_loader = get_domain_shift_dataloader(cfg)
            self.val_loader = get_domain_shift_dataloader(cfg)
        else:
            self.val_loader = get_val_dataloader(cfg)
            self.test_loader = get_test_dataloader(cfg)
        
        self.test_inference_memory = {}
        self.test_errors, self.val_errors = self._get_test_errors(), self._get_val_errors()

    @torch.no_grad()
    def predict(self):
        self.model.eval()
        self.norm_module.requires_grad_(False).eval() if self.norm_module is not None else None
        log_dict = {}
        
        self.errors_all = {
            "test_mse": self.test_errors['mse'], 
            "test_mae": self.test_errors['mae'], 
            "val_mse": self.val_errors['mse'], 
            "val_mae": self.val_errors['mae'], 
        }

        results = self.get_results()  # {test_mse: , test_mae:, val_mse: val_mae: }
        self.save_results(results)
        
        if 'mse_per_var' in self.test_errors:
            print(f"Test MSE per channels: {self.test_errors['mse_per_var'].mean(axis=0)}")
        
        self.save_to_npy(**self.errors_all)

        # log to W&B
        log_dict.update({f"Test/{metric}": value for metric, value in results.items()})
        if self.cfg.WANDB.ENABLE:
            wandb.log(log_dict)
        dataset_name = self.cfg.DATA.NAME if not self.cfg.TTA.DOMAIN_SHIFT else f"{self.cfg.DATA.NAME}_2_{self.cfg.DATA.DOMAIN_SHIFT_TARGET}"
        save_tta_results(tta_method="None",
                         seed=self.cfg.SEED,
                         model_name=self.cfg.MODEL.NAME,
                         dataset_name=dataset_name,
                         pred_len=self.cfg.DATA.PRED_LEN,
                         mse_after_tta=self.test_errors['mse'].mean().astype(float),
                         mae_after_tta=self.test_errors['mae'].mean().astype(float),)
        if self.test_inference_memory:
            memory_record = {
                "method": "base",
                "model": self.cfg.MODEL.NAME,
                "dataset_name": dataset_name,
                "pred_len": self.cfg.DATA.PRED_LEN,
                "seed": self.cfg.SEED,
                "batch_size": self.cfg.TEST.BATCH_SIZE,
                **self.test_inference_memory,
            }
            save_summary_to_csv(
                memory_record,
                csv_path=os.path.join(
                    self.cfg.RESULT_DIR, "base_inference_memory_results.csv"
                ),
                key_fields=[
                    "method", "model", "dataset_name", "pred_len", "seed", "batch_size"
                ],
            )
            print(
                "Base inference memory: "
                f"{memory_record['peak_allocated_mb']:.2f} MiB peak allocated, "
                f"{memory_record['peak_incremental_allocated_mb']:.2f} MiB "
                "peak incremental allocated"
            )

    @torch.no_grad()
    def _get_errors_from_dataloader(self, dataloader, tta=False, split='test'):
        self.model.eval()
        self.norm_module.requires_grad_(False).eval() if self.norm_module is not None else None
        mse_all = []
        mae_all = []
        mse_per_var_all = []
        profile_cuda = split == "test" and torch.cuda.is_available()
        num_batches = 0
        if profile_cuda:
            torch.cuda.synchronize()
            allocated_before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        else:
            allocated_before = 0

        for inputs in tqdm(dataloader, desc='Calculating Errors'):
            num_batches += 1
            enc_window_raw, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
            if self.norm_method == 'SAN':
                enc_window, statistics_pred = self.norm_module.normalize(enc_window_raw)
            else:  # Normalization from Non-stationary Transformer
                means = enc_window_raw.mean(1, keepdim=True).detach()
                enc_window = enc_window_raw - means
                stdev = torch.sqrt(torch.var(enc_window, dim=1, keepdim=True, unbiased=False) + 1e-5)
                enc_window /= stdev
            
            target_start = self.cfg.DATA.TARGET_START_IDX
            target_end = target_start + self.cfg.MODEL.c_out
            
            # ground_truth = dec_window[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:].float()
            ground_truth = dec_window[:, -self.cfg.DATA.PRED_LEN:, target_start:target_end].float()
            dec_zeros = torch.zeros_like(dec_window[:, -self.cfg.DATA.PRED_LEN:, :]).float()
            dec_window = torch.cat([dec_window[:, :self.cfg.DATA.LABEL_LEN:, :], dec_zeros], dim=1).float().to(global_device)
            
            model_cfg = self.cfg.MODEL
            pred = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)
            if model_cfg.output_attention:
                pred = pred[0]
            
            # pred = pred[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:]
            # pred = pred[:, -self.cfg.DATA.PRED_LEN:, target_start:target_end]
            pred = pred[:, -self.cfg.DATA.PRED_LEN:, :]
            if pred.shape[-1] == self.cfg.MODEL.c_out:
                # New PCD models directly return the two target channels.
                pass
            elif pred.shape[-1] >= target_end:
                # Legacy PCD models return all input channels.
                pred = pred[:, :, target_start:target_end]
            else:
                raise ValueError(
                    f"Model returned {pred.shape[-1]} channels, but c_out="
                    f"{self.cfg.MODEL.c_out} and target slice "
                    f"[{target_start}:{target_end}] was requested."
                )
            
            if self.norm_method == 'SAN':
                pred = self.norm_module.de_normalize(pred, statistics_pred)
            else:  # De-Normalization from Non-stationary Transformer
                # Align stats to predicted channels to avoid size mismatch
                B, T_pred, C_pred = pred.shape
                start = int(getattr(self.cfg.DATA, 'TARGET_START_IDX', 0))
                stdev_sel = stdev[:, 0, start:start + C_pred]
                means_sel = means[:, 0, start:start + C_pred]
                stdev_exp = stdev_sel.unsqueeze(1).repeat(1, T_pred, 1)
                means_exp = means_sel.unsqueeze(1).repeat(1, T_pred, 1)
                # Optional assertion for quicker debugging
                assert stdev_exp.shape[-1] == C_pred and means_exp.shape[-1] == C_pred
                pred = pred * stdev_exp + means_exp
            
            mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1))
            mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1))
            mse_per_var = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=-2)
            
            mse_all.append(mse)
            mae_all.append(mae)
            mse_per_var_all.append(mse_per_var)
                
        if profile_cuda:
            torch.cuda.synchronize()
            peak_allocated = torch.cuda.max_memory_allocated()
            self.test_inference_memory = {
                "num_batches": num_batches,
                "allocated_before_mb": allocated_before / (1024.0 ** 2),
                "peak_allocated_mb": peak_allocated / (1024.0 ** 2),
                "peak_reserved_mb": (
                    torch.cuda.max_memory_reserved() / (1024.0 ** 2)
                ),
                "peak_incremental_allocated_mb": (
                    max(0, peak_allocated - allocated_before) / (1024.0 ** 2)
                ),
            }

        mse_all = torch.flatten(torch.concat(mse_all, dim=0)).cpu().numpy()
        mae_all = torch.flatten(torch.concat(mae_all, dim=0)).cpu().numpy()
        mse_per_var_all = torch.concat(mse_per_var_all, dim=0).cpu().numpy()

        return {'mse': mse_all, 'mae': mae_all, 'mse_per_var': mse_per_var_all}

    def _get_val_errors(self):
        return self._get_errors_from_dataloader(self.val_loader, tta=False, split='val')

    def _get_test_errors(self):
        return self._get_errors_from_dataloader(self.test_loader, tta=self.cfg.TTA.ENABLE, split='test')

    def get_results(self) -> Dict[str, float]:
        test_mse = self.test_errors['mse'].mean().astype(float)
        test_mae = self.test_errors['mae'].mean().astype(float)
        val_mse = self.val_errors['mse'].mean().astype(float)
        val_mae = self.val_errors['mae'].mean().astype(float)
        
        return {"test_mse": test_mse, "test_mae": test_mae, "val_mse": val_mse, "val_mae": val_mae}

    def save_results(self, results):
        results_string = ", ".join([f"{metric}: {value:.04f}" for metric, value in results.items()])
        print(f"Results without TSF-TTA for pred_len: {self.cfg.DATA.PRED_LEN}:")
        print(results_string)

        with open(os.path.join(mkdir(self.cfg.RESULT_DIR) / "test.txt"), "w") as f:
            f.write(results_string)

    def save_to_npy(self, **kwargs):
        for key, value in kwargs.items():
            np.save(os.path.join(self.cfg.RESULT_DIR, f"{key}.npy"), value)
