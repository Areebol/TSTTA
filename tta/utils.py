import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def save_summary_to_csv(
    record: Dict,
    csv_path: Optional[str] = None,
    key_fields: Optional[List[str]] = None
):
    """
    通用保存/更新 summary csv 的函数。
    - record: 要保存的一条记录（dict）
    - csv_path: csv 文件路径，默认 results/COSA_tta_summary.csv
    - key_fields: 用于判定唯一性的列列表，默认 ["model","dataset_name","pred_len","seed"]
    """
    if csv_path is None:
        csv_path = os.path.join("results", "COSA_tta_summary.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    df_new = pd.DataFrame([record])

    if key_fields is None:
        key_fields = ["model", "dataset_name", "pred_len", "seed"]

    try:
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            for col in key_fields:
                if col not in df_existing.columns:
                    df_existing[col] = np.nan

            # 用 merge 去重并更新
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
            df_out = df_out.drop_duplicates(subset=key_fields, keep='last')
            df_out.to_csv(csv_path, index=False)
        else:
            df_new.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Error saving to {csv_path}: {e}")

def save_tta_results(
    tta_method: str,
    model_name: str,
    dataset_name: str,
    pred_len: int,
    seed: int,
    mse_after_tta: float,
    mae_after_tta: float,
    save_dir: str = "./results"
):
    record = {
        "model": model_name,
        "dataset_name": dataset_name,
        "pred_len": pred_len,
        "mse_after_tta": mse_after_tta,
        "mae_after_tta": mae_after_tta,
        "seed": seed
    }
    csv_path = os.path.join(save_dir, f"{tta_method}_results.csv")
    save_summary_to_csv(record, csv_path=csv_path)
    
class TTADataManager:
    def __init__(self, cfg, enabled=True):
        self.cfg = cfg
        self.enabled = enabled
        self.reset()

    def reset(self):
        self.storage = {
            "preds_base": [],
            "preds_tta": [],
            "gts": [],
            "gating_weights": [],
            "mse_steps": []
        }

    def collect(self, base_pred=None, tta_pred=None, gt=None, gating=None, mse=None):
        if not self.enabled:
            return

        if base_pred is not None: self.storage["preds_base"].append(base_pred.detach().cpu().numpy())
        if tta_pred is not None: self.storage["preds_tta"].append(tta_pred.detach().cpu().numpy())
        if gt is not None: self.storage["gts"].append(gt.detach().cpu().numpy())
        if gating is not None: self.storage["gating_weights"].append(gating.detach().cpu().numpy())
        if mse is not None: self.storage["mse_steps"].append(mse)

    def get_full_data(self):
        if not self.enabled or len(self.storage["gts"]) == 0:
            return None
        
        return {
            "preds_base": np.concatenate(self.storage["preds_base"], axis=0),
            "preds_tta": np.concatenate(self.storage["preds_tta"], axis=0),
            "gts": np.concatenate(self.storage["gts"], axis=0),
            "gating": np.concatenate(self.storage["gating_weights"], axis=0),
            "mse": np.concatenate(self.storage["mse_steps"], axis=0) if self.storage["mse_steps"] else None
        }

    def save_to_disk(self, save_dir):
        if not self.enabled: return
        
        data = self.get_full_data()
        np.savez(os.path.join(save_dir, "tta_raw_data.npz"), **data)
        print(f"Raw TTA data saved to {save_dir}")