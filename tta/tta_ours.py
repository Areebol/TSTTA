import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from typing import List, Optional
from torch.utils.data import DataLoader, Subset

from config import get_norm_method
from models.forecast import forecast
from utils.misc import prepare_inputs
from models.optimizer import get_optimizer
from datasets.loader import get_test_dataloader, get_tta_train_dataloader

from tta.loss import *
from tta.tta_dual_utils.GCM import *
from tta.tta_dual_utils.model_manager import TTAModelManager
from tta.utils import save_tta_results
from tta.visualizer import TTAVisualizer
import os
import matplotlib.pyplot as plt

class TTADataManager:
    def __init__(self, cfg, enabled=True):
        self.cfg = cfg
        self.enabled = enabled
        self.reset()

    def reset(self):
        self.storage = {
            "inputs": [],
            "preds_base": [],
            "preds_tta": [],
            "gts": [],
            "mse_steps": []
        }

    def collect(self, inputs=None, base_pred=None, tta_pred=None, gt=None, gating=None, mse=None):
        if not self.enabled:
            return
        if inputs is not None: self.storage['inputs'].append(inputs.detach().cpu().numpy())
        if base_pred is not None: self.storage["preds_base"].append(base_pred.detach().cpu().numpy())
        if tta_pred is not None: self.storage["preds_tta"].append(tta_pred.detach().cpu().numpy())
        if gt is not None: self.storage["gts"].append(gt.detach().cpu().numpy())
        if gating is not None: self.storage["gating_weights"].append(gating.detach().cpu().numpy())
        if mse is not None: self.storage["mse_steps"].append(mse)

    def get_full_data(self):
        if not self.enabled or len(self.storage["gts"]) == 0:
            return None
        
        return {
            "inputs": np.concatenate(self.storage["inputs"], axis=0),
            "preds_base": np.concatenate(self.storage["preds_base"], axis=0),
            "preds_tta": np.concatenate(self.storage["preds_tta"], axis=0),
            "gts": np.concatenate(self.storage["gts"], axis=0),
            "mse": np.concatenate(self.storage["mse_steps"], axis=0) if self.storage["mse_steps"] else None
        }

    def save_to_disk(self, save_dir):
        if not self.enabled: return
        
        data = self.get_full_data()
        np.savez(os.path.join(save_dir, "tta_raw_data.npz"), **data)
        print(f"Raw TTA data saved to {save_dir}")

class TTAVisualizer:
    def __init__(self, save_dir, cfg):
        self.save_dir = save_dir
        self.cfg = cfg
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-muted')

    def plot_all(self, data_dict):
        os.makedirs(self.save_dir, exist_ok=True)
        if data_dict is None:
            print("Visualizer: No data provided to plot.")
            return

        print(f"Visualizer: Generating plots in {self.save_dir}...")

        if data_dict['mse'] is not None:
            self._plot_error_analysis(data_dict['mse'])
            
        self.plot_best_samples_per_channel(data_dict)
        self.plot_worst_samples_per_channel(data_dict)
        self.plot_sample_predictions(data_dict, sample_idx=1560)
        self.plot_input_and_predictions(data_dict, sample_idx=1560, channel_idx=0, prefix="full_sequence")

    def _plot_predictions(self, base, tta, gt, num_samples=3, start_idx=300,):
        # base/tta/gt 形状: [Total_Samples, Pred_Len, Channels]
        n_samples, pred_len, n_vars = gt.shape
        samples_to_plot = min(num_samples + start_idx, n_samples)
        
        channels_to_plot = [0, n_vars // 2, n_vars - 1] if n_vars > 2 else range(n_vars)

        for s_idx in range(start_idx, samples_to_plot):
            for c_idx in channels_to_plot:
                plt.figure(figsize=(10, 5))
                
                # 绘制三条线
                plt.plot(gt[s_idx, :, c_idx], label='Ground Truth', color='#2c3e50', linewidth=2, linestyle='--')
                plt.plot(base[s_idx, :, c_idx], label='Base Model', color='#e74c3c', alpha=0.7)
                plt.plot(tta[s_idx, :, c_idx], label='After TTA', color='#3498db', linewidth=1.5)

                plt.title(f"Forecast Comparison | Sample {s_idx} | Channel {c_idx}")
                plt.xlabel("Time Step (within Pred_Len)")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                save_path = os.path.join(self.save_dir, f"pred_s{s_idx}_c{c_idx}.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()

    def _plot_error_analysis(self, mse_steps):
        plt.figure(figsize=(10, 5))
        
        window_size = min(20, len(mse_steps))
        if window_size > 0:
            smoothed_mse = np.convolve(mse_steps, np.ones(window_size)/window_size, mode='valid')
            plt.plot(smoothed_mse, color='#8e44ad', label=f'MSE (Moving Avg {window_size})')
        
        plt.plot(mse_steps, color='#8e44ad', alpha=0.2, label='Batch MSE')
        
        plt.title("Prediction Error (MSE) over Test Stream")
        plt.xlabel("Test Step")
        plt.ylabel("MSE")
        plt.legend()
        plt.yscale('log') # 误差通常建议用对数坐标查看
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        plt.savefig(os.path.join(self.save_dir, "error_trend.png"), bbox_inches='tight')
        plt.close()
        
    def plot_best_samples_per_channel(self, data_dict):
        base = data_dict['preds_base']
        tta = data_dict['preds_tta']
        gt = data_dict['gts']
        
        mse_base = np.mean((base - gt)**2, axis=1) # [Samples, Channels]
        mse_tta = np.mean((tta - gt)**2, axis=1)   # [Samples, Channels]
        
        improvement = mse_base - mse_tta
        
        best_improvement_indices = np.argmax(improvement, axis=0)
        
        num_channels = gt.shape[2]
        print(f"Visualizer: Plotting MOST IMPROVED samples for {num_channels} channels...")

        for c_idx in range(num_channels):
            s_idx = best_improvement_indices[c_idx]
            self._draw_adapter_impact_plot(
                base, tta, gt, 
                sample_idx=s_idx, 
                channel=c_idx,
                title_suffix=f"(Most Improved for Var {c_idx})",
                prefix="best_improvement"
            )

    def plot_worst_samples_per_channel(self, data_dict):
        base = data_dict['preds_base']
        tta = data_dict['preds_tta']
        gt = data_dict['gts']
        
        mse_base = np.mean((base - gt)**2, axis=1)
        mse_tta = np.mean((tta - gt)**2, axis=1)
        improvement = mse_base - mse_tta
        
        worst_improvement_indices = np.argmin(improvement, axis=0)
        
        num_channels = gt.shape[2]
        print(f"Visualizer: Plotting MOST DEGRADED samples for {num_channels} channels...")

        for c_idx in range(num_channels):
            s_idx = worst_improvement_indices[c_idx]
            self._draw_adapter_impact_plot(
                base, tta, gt, 
                sample_idx=s_idx, 
                channel=c_idx,
                title_suffix=f"(Most Degraded for Var {c_idx})",
                prefix="worst_degradation"
            )

    def plot_sample_predictions(self, data_dict, sample_idx):
        base = data_dict['preds_base']
        tta = data_dict['preds_tta']
        gt = data_dict['gts']
        
        for c_idx in range(gt.shape[2]):
            self._draw_adapter_impact_plot(
                base, tta, gt, 
                sample_idx=sample_idx, 
                channel=c_idx,
                title_suffix=f"",
                prefix="sample_prediction")

    def _draw_adapter_impact_plot(self, base, tta, gt, sample_idx, channel, title_suffix="", prefix="best"):
        delta = tta - base
        y_gt = gt[sample_idx, :, channel]
        y_base = base[sample_idx, :, channel]
        y_tta = tta[sample_idx, :, channel]
        y_delta = delta[sample_idx, :, channel]

        mse_base = np.mean((y_base - y_gt)**2)
        mse_tta = np.mean((y_tta - y_gt)**2)
        improvement = (mse_base - mse_tta) / (mse_base + 1e-9) * 100

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True, 
                                       gridspec_kw={'height_ratios': [1, 0.7]})

        ax1.plot(y_gt, label='Ground Truth', color='#333333', alpha=0.4, linewidth=1.2, linestyle='-')
        ax1.plot(y_base, label='Base Pred', color="#26CE99", linestyle='--', linewidth=1.8)
        ax1.plot(y_tta, label='TTA Pred', color="#EC591F", linewidth=2.2, linestyle='-')
        
        ax1.set_title(f"Prediction Comparison (Sample {sample_idx}, Var {channel})\n{title_suffix}", fontsize=13, pad=15)
        ax1.legend(loc='upper left', frameon=True)
        ax1.grid(True, alpha=0.2)

        info_color = 'green' if improvement > 0 else 'red'
        info_text = (
            f"Base MSE: {mse_base:.4f}\n"
            f"TTA MSE:  {mse_tta:.4f}\n"
            f"Change:   {improvement:+.2f}%"
        )
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=info_color)
        ax1.text(0.98, 0.95, info_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right', bbox=props, color=info_color, weight='bold')

        ax2.plot(y_delta, label='Adapter Adjustment (Delta)', color='#27ae60', linewidth=2)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
        
        ax2.fill_between(range(len(y_delta)), y_delta, 0, color='#27ae60', alpha=0.1)
        
        ax2.set_title("Adapter Contribution (TTA_Pred - Base_Pred)", fontsize=11)
        ax2.set_ylabel("Adjustment Value")
        ax2.set_xlabel("Time Step")
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.2)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{prefix}_impact_var{channel}_s{sample_idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_input_and_predictions(self, data_dict, sample_idx=0, channel_idx=0, prefix="full_sequence"):
        """
        核心可视化方法：
        将 [Input Window] 和 [Prediction Window] 拼接展示，对比 Base Pred 和 TTA Pred。
        """
        # 1. 从 data_dict 提取数据并转为 numpy (如果还是 tensor 的话)
        def to_numpy(x): return x.cpu().numpy() if hasattr(x, 'cpu') else x

        try:
            # 必须包含的数据
            inputs = to_numpy(data_dict['inputs'])      # [N, Seq_Len, C]
            gts = to_numpy(data_dict['gts'])            # [N, Pred_Len, C]
            base = to_numpy(data_dict['preds_base'])    # [N, Pred_Len, C]
            tta = to_numpy(data_dict['preds_tta'])      # [N, Pred_Len, C]
        except KeyError as e:
            print(f"Visualizer Error: Missing key {e} in data_dict.")
            return

        # 2. 提取具体样本和通道数据
        y_inp = inputs[sample_idx, :, channel_idx]
        y_gt = gts[sample_idx, :, channel_idx]
        y_base = base[sample_idx, :, channel_idx]
        y_tta = tta[sample_idx, :, channel_idx]
        y_delta = y_tta - y_base
        
        seq_len = len(y_inp)
        pred_len = len(y_gt)
        
        # 3. 设置时间轴：Input 在前，Prediction 在后
        x_inp = np.arange(seq_len)
        x_pred = np.arange(seq_len, seq_len + pred_len)

        # 4. 计算 MSE 和改善程度
        mse_base = np.mean((y_base - y_gt)**2)
        mse_tta = np.mean((y_tta - y_gt)**2)
        improvement = (mse_base - mse_tta) / (mse_base + 1e-9) * 100

        # 5. 绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, 
                                       gridspec_kw={'height_ratios': [1, 0.4]})

        # --- Subplot 1: Sequence Comparison ---
        # 历史窗口 (Input)
        ax1.plot(x_inp, y_inp, label='Input (Lookback)', color='#7f8c8d', linewidth=2, alpha=0.8)
        
        # 预测窗口 (GT, Base, TTA)
        ax1.plot(x_pred, y_gt, label='Ground Truth', color='#2c3e50', linestyle='--', linewidth=1.5)
        ax1.plot(x_pred, y_base, label='Base Model Pred', color="#26CE99", linestyle='-', alpha=0.6, marker='.', markersize=4)
        ax1.plot(x_pred, y_tta, label='TTA Adjusted Pred', color="#EC591F", linewidth=2.5)

        # 垂直分割线
        ax1.axvline(x=seq_len - 1, color='#e74c3c', linestyle='-', linewidth=1.2, alpha=0.7)
        ax1.text(seq_len - 1, ax1.get_ylim()[1], ' Forecast Start ', color='#e74c3c', 
                 ha='right', va='top', fontweight='bold', fontsize=10)

        ax1.set_title(f"TTA Adaption Impact | Sample {sample_idx} | Channel {channel_idx}", fontsize=14)
        ax1.legend(loc='upper left', frameon=True, fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.2)

        # 标注 MSE 变化
        info_color = 'green' if improvement > 0 else 'red'
        info_text = f"Base MSE: {mse_base:.4f}\nTTA MSE:  {mse_tta:.4f}\nChange:   {improvement:+.2f}%"
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=info_color)
        ax1.text(0.98, 0.95, info_text, transform=ax1.transAxes, fontsize=10,
                 va='top', ha='right', bbox=props, color=info_color, weight='bold', family='monospace')

        # --- Subplot 2: Adapter Contribution (Delta) ---
        ax2.plot(x_pred, y_delta, label='TTA Adjustment (Delta)', color='#3498db', linewidth=2)
        ax2.fill_between(x_pred, y_delta, 0, color='#3498db', alpha=0.2)
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
        ax2.axvline(x=seq_len - 1, color='#e74c3c', linestyle='-', alpha=0.2)
        
        ax2.set_ylabel("Delta")
        ax2.set_xlabel("Time Steps")
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.2)

        # 6. 保存
        plt.tight_layout()
        save_name = f"{prefix}_s{sample_idx}_c{channel_idx}.png"
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        # print(f"Visualizer: Saved full sequence plot to {save_path}")

def build_calibration_module(cfg) -> Optional[CalibrationContainer]:
    def get_model_dims(cfg):
        is_patchtst = (cfg.MODEL.NAME == 'PatchTST')
        if cfg.TTA.DUAL.CALI_NAME == 'coba-GCM':
            n_var = cfg.DATA.N_VAR
        else:
            n_var = 1 if is_patchtst else cfg.DATA.N_VAR
        if cfg.DATA.NAME == 'eVED':
            n_var = 2
        return cfg.DATA.SEQ_LEN, cfg.DATA.PRED_LEN, n_var
    
    if not cfg.TTA.DUAL.CALI_MODULE:
        return None
    
    seq_len, pred_len, n_var = get_model_dims(cfg)
    params = {
        'hidden_dim': cfg.TTA.DUAL.HIDDEN_DIM,
        'gating_init': cfg.TTA.DUAL.GATING_INIT,
        'var_wise': cfg.TTA.DUAL.GCM_VAR_WISE,
    }
    model_type = getattr(cfg.TTA.DUAL, 'CALI_NAME', 'tafas_GCM')
    
    constructors = {
        'tafas-GCM': tafas_GCM,
        'petsa-GCM': petsa_GCM,
        'coba-GCM': CoBA_GCM,
        'identity': IdentityAdapter,
    }
    if model_type == 'coba-GCM':
        coba_params = {
            'n_bases': cfg.TTA.DUAL.GCM_N_BASES,
        }
        params.update(coba_params)
    elif model_type == 'identity':
        return CalibrationContainer(None, None)

    ModelClass = constructors.get(model_type)
    if not ModelClass:
        raise ValueError(f"Unknown adapter type: {model_type}")

    in_model = None
    out_model = None
    
    if cfg.TTA.DUAL.CALI_INPUT_ENABLE:
        in_model = ModelClass(seq_len, n_var, **params)
    if cfg.TTA.DUAL.CALI_OUTPUT_ENABLE:
        out_model = ModelClass(pred_len, n_var, **params)
    return CalibrationContainer(in_model, out_model)

def build_loss_fn(cfg) -> nn.Module:
    loss_name = getattr(cfg.TTA.DUAL, 'LOSS_NAME', 'MSE')
    if loss_name == 'MSE':
        return StandardMSELoss()
    elif loss_name == 'PETSA': 
        alpha = getattr(cfg.TTA.DUAL, 'PETSA_LOSS_ALPHA', 0.1)
        return PETSALoss(alpha=alpha)
    elif loss_name == "COBA":
        return CoBA_Loss(lambda_ortho=0.001)
    else:
        raise ValueError(f"Unknown Loss type: {loss_name}")

def get_optimizer(optim_params, cfg):
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )

class Adapter(nn.Module):
    def __init__(self, cfg, model: nn.Module, norm_module=None):
        super(Adapter, self).__init__()
        self.cfg = cfg
        
        self.model = model
        self.norm_method = get_norm_method(cfg)
        self.norm_module = norm_module
        self.cali = build_calibration_module(cfg).cuda()
        self.loss_fn = build_loss_fn(cfg)

        self.manager = TTAModelManager(model, norm_module, self.cali)
        trainable_params = self.manager.configure_adaptation(cfg.TTA.MODULE_NAMES_TO_ADAPT)
        self.manager.snapshot()
        self.optimizer = get_optimizer(trainable_params, cfg.TTA)
        self.optimizer_state = deepcopy(self.optimizer.state_dict())
        
        self.test_loader = get_test_dataloader(cfg)
        self.test_data = self.test_loader.dataset.test
        batch_size = len(self.test_loader.dataset)
        self.test_loader = get_test_dataloader(cfg, batch_size=batch_size)

        self.tta_train_loader = get_tta_train_dataloader(cfg)
        self.tta_train_data = self.tta_train_loader.dataset.train
        
        self.cur_step = cfg.DATA.SEQ_LEN - 2
        self.pred_step_end_dict = {}
        self.inputs_dict = {}
        self.n_adapt = 0

        cali_name = getattr(self.cfg.TTA.DUAL, 'CALI_NAME', 'unknown')
        loss_name = getattr(self.cfg.TTA.DUAL, 'LOSS_NAME', 'MSE')
        input_enable = getattr(self.cfg.TTA.DUAL, 'CALI_INPUT_ENABLE', False)
        output_enable = getattr(self.cfg.TTA.DUAL, 'CALI_OUTPUT_ENABLE', False)

        parts = [
            f'dual-cali-{cali_name}',
            f'loss-{loss_name}'
        ]

        if input_enable:
            parts.append("in")
        if output_enable:
            parts.append("out")

        self.save_name = "-".join(parts)
        self.mse_all = []
        self.mae_all = []

        ds = self.test_loader.dataset
        self.is_eved_like = (
            hasattr(ds, "get_num_test_csvs")
            and hasattr(ds, "get_test_csv_window_range")
            and hasattr(ds, "get_test_windows_for_csv")
        )
        self._pretrain_adapter()
        print("Adapter pre-training completed.")
        optim_params = self.cali.out_cali.get_optim_params()
        self.optimizer = torch.optim.Adam(
            optim_params,
            lr=1e-4,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        ) 
        
        
        self.data_manager = TTADataManager(
            cfg, 
            enabled=getattr(cfg.TTA, 'SAVE_ANALYSIS_DATA', True)
        )
        self.visualizer = TTAVisualizer(save_dir=f"./visualize/{cfg.MODEL.NAME}-{cfg.DATA.NAME}-{cfg.DATA.PRED_LEN}/{self.save_name}", cfg=cfg)
    def _pretrain_adapter(self):
        self._switch_model_to_train()
        for epoch in range(5):
            for inputs in self.tta_train_loader:
                enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
                inputs = (enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all)
                if self.cali.input_calibration is not None:
                    inputs = self.cali.input_calibration(inputs)
                pred, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
                if self.cali.output_calibration is not None:
                    pred = self.cali.output_calibration(pred)
                if isinstance(self.loss_fn, CoBA_Loss):
                    loss = self.loss_fn(pred, ground_truth, bases=self.cali.out_cali.bases)
                else:
                    loss = self.loss_fn(pred, ground_truth) 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._switch_model_to_eval()

    def _reset(self):
        self.manager.reset()
        self.optimizer.load_state_dict(deepcopy(self.optimizer_state))

    def _switch_model_to_train(self):
        self.manager.train()
    
    def _switch_model_to_eval(self):
        self.manager.eval()   
    
    def _calculate_period_and_batch_size(self, enc_window_first):
        fft_result = torch.fft.rfft(enc_window_first - enc_window_first.mean(dim=0), dim=0)
        amplitude = torch.abs(fft_result)
        power = torch.mean(amplitude ** 2, dim=0)
        try:
            period = enc_window_first.shape[0] // torch.argmax(amplitude[:, power.argmax()]).item()
        except:
            period = 24
        period *= self.cfg.TTA.DUAL.PERIOD_N
        batch_size = period + 1
        return period, batch_size

    def _adapt_with_full_ground_truth_if_available(self):
        while self.cur_step >= self.pred_step_end_dict[min(self.pred_step_end_dict.keys())]:
            batch_idx_available = min(self.pred_step_end_dict.keys())
            inputs_history = self.inputs_dict.pop(batch_idx_available)
            for _ in range(self.cfg.TTA.DUAL.STEPS):
                self.n_adapt += 1
                
                self._switch_model_to_train()

                if self.cali.input_calibration is not None:
                    inputs_history = self.cali.input_calibration(inputs_history)
                pred, ground_truth = forecast(self.cfg, inputs_history, self.model, self.norm_module)
                
                if self.cali.output_calibration is not None:
                    pred = self.cali.output_calibration(pred)
                    
                if isinstance(self.loss_fn, CoBA_Loss):
                    loss = self.loss_fn(pred, ground_truth, bases=self.cali.out_cali.bases)
                else:
                    loss = self.loss_fn(pred, ground_truth) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self._switch_model_to_eval()
            
            self.pred_step_end_dict.pop(batch_idx_available)

    def _adapt_with_partial_ground_truth(self, inputs, period, batch_size, batch_idx):
        for _ in range(self.cfg.TTA.DUAL.STEPS):
            self.n_adapt += 1
            
            if self.cali.input_calibration is not None:
                inputs = self.cali.input_calibration(inputs)
            pred, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
        
            if self.cali.output_calibration is not None:
                pred = self.cali.output_calibration(pred)
                
            pred_partial, ground_truth_partial = pred[0][:period], ground_truth[0][:period]
            if isinstance(self.loss_fn, CoBA_Loss):
                loss_partial = self.loss_fn(pred_partial, ground_truth_partial, bases=self.cali.out_cali.bases)
            else:
                loss_partial = self.loss_fn(pred_partial, ground_truth_partial) 
            self.optimizer.zero_grad()
            loss_partial.backward()
            self.optimizer.step()
        return pred, ground_truth

    @torch.no_grad()
    def _adjust_prediction(self, pred, inputs, batch_size, period):
        if self.cali.input_calibration is not None:
            inputs = self.cali.input_calibration(inputs)
        pred_after_adapt, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
        if self.cali.output_calibration is not None:
            pred_after_adapt = self.cali.output_calibration(pred_after_adapt)
        
        for i in range(batch_size-1):
            pred[i, period-i:] = pred_after_adapt[i, period-i:]
        return pred, ground_truth
    
    def _report(self):
        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        assert len(self.mse_all) == len(self.test_loader.dataset)
        
        save_tta_results(
            tta_method=self.save_name,
            seed=self.cfg.SEED,
            model_name=self.cfg.MODEL.NAME,
            dataset_name=self.cfg.DATA.NAME,
            pred_len=self.cfg.DATA.PRED_LEN,
            mse_after_tta=self.mse_all.mean(),
            mae_after_tta=self.mae_all.mean(),
        )
        full_data = self.data_manager.get_full_data()
        if full_data:
            self.visualizer.plot_all(full_data)
        self.model.eval()
    
    def adapt(self):
        self.data_manager.reset()
        if getattr(self, "is_eved_like", False):
            self._adapt_eved()
        else:
            self._adapt_regular()
    
    @torch.enable_grad()
    def _adapt_regular(self):
        is_last = False
        test_len = len(self.test_loader.dataset)
        
        self._switch_model_to_eval()
        inputs = next(iter(self.test_loader))
        enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
        batch_start = 0
        batch_end = 0
        batch_idx = 0
        self.cur_step = self.cfg.DATA.SEQ_LEN - 2
        total_len = len(enc_window_all) 
        while batch_end < len(enc_window_all):
            enc_window_first = enc_window_all[batch_start]
            
            if self.cfg.TTA.DUAL.PAAS:
                period, batch_size = self._calculate_period_and_batch_size(enc_window_first)
            else:
                batch_size = self.cfg.TTA.DUAL.BATCH_SIZE
                period = batch_size - 1
            batch_end = batch_start + batch_size

            if batch_end > len(enc_window_all):
                batch_end = len(enc_window_all)
                batch_size = batch_end - batch_start
                is_last = True

            self.cur_step += batch_size

            inputs = enc_window_all[batch_start:batch_end], enc_window_stamp_all[batch_start:batch_end], dec_window_all[batch_start:batch_end], dec_window_stamp_all[batch_start:batch_end]
            
            self.pred_step_end_dict[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN
            self.inputs_dict[batch_idx] = inputs
            
            self._adapt_with_full_ground_truth_if_available()
            pred, ground_truth = self._adapt_with_partial_ground_truth(inputs, period, batch_size, batch_idx)
            base_pred = pred.clone().detach()
            if self.cfg.TTA.DUAL.ADJUST_PRED:
                pred, ground_truth = self._adjust_prediction(pred, inputs, batch_size, period)
            if self.cali.output_calibration is not None:
                pred = self.cali.output_calibration(pred)
            tta_pred = pred.clone().detach()
            mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
            mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
            self.mse_all.append(mse)
            self.mae_all.append(mae)
            
            batch_start = batch_end
            batch_idx += 1
            self.data_manager.collect(
                inputs=inputs[0],
                base_pred=base_pred,
                tta_pred=tta_pred,
                gt=ground_truth,      
                mse=mse         
            )
                
        assert self.cur_step == len(self.test_data) - self.cfg.DATA.PRED_LEN - 1
        self._report()
            
    def _adapt_eved(self):
        ds = self.test_loader.dataset
        num_csv = ds.get_num_test_csvs()

        self.mse_all = []
        self.mae_all = []
        self.mse_per_var_all = []
        self.mae_per_var_all = []
        self.n_adapt = 0

        for csv_idx in range(num_csv):
            # obtain the indices for each csv
            indices = ds.get_test_windows_for_csv(csv_idx)
            if not indices:
                continue
            sub_dataset = Subset(ds, indices)
            sub_loader = DataLoader(sub_dataset, batch_size=len(sub_dataset), shuffle=False)

            self.pred_step_end_dict = {}
            self.inputs_dict = {}
            self._switch_model_to_eval()

            for idx, inputs in enumerate(sub_loader):
                enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
                batch_start = 0
                batch_end = 0
                batch_idx = 0
                self.cur_step = self.cfg.DATA.SEQ_LEN - 2
                is_last = False

                while batch_end < len(enc_window_all):
                    enc_window_first = enc_window_all[batch_start]
                    if self.cfg.TTA.DUAL.PAAS:
                        period, batch_size = self._calculate_period_and_batch_size(enc_window_first)
                    else:
                        batch_size = self.cfg.TTA.DUAL.BATCH_SIZE
                        period = batch_size - 1
                    batch_end = batch_start + batch_size

                    if batch_end > len(enc_window_all):
                        batch_end = len(enc_window_all)
                        batch_size = batch_end - batch_start
                        is_last = True

                    self.cur_step += batch_size

                    inputs_batch = (
                        enc_window_all[batch_start:batch_end],
                        enc_window_stamp_all[batch_start:batch_end],
                        dec_window_all[batch_start:batch_end],
                        dec_window_stamp_all[batch_start:batch_end],
                    )

                    self.pred_step_end_dict[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN
                    self.inputs_dict[batch_idx] = inputs_batch

                    self._adapt_with_full_ground_truth_if_available()
                    pred, ground_truth = self._adapt_with_partial_ground_truth(inputs_batch, period, batch_size, batch_idx)

                    if self.cfg.TTA.DUAL.ADJUST_PRED:
                        pred, ground_truth = self._adjust_prediction(pred, inputs_batch, batch_size, period)

                    mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                    mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                    self.mse_all.append(mse)
                    self.mae_all.append(mae)

                    mse_per_var = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=-2).detach().cpu().numpy()
                    mae_per_var = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=-2).detach().cpu().numpy()
                    self.mse_per_var_all.append(mse_per_var)
                    self.mae_per_var_all.append(mae_per_var)

                    batch_start = batch_end
                    batch_idx += 1

        self._report()

def build_adapter(cfg, model, norm_module=None):
    adapter = Adapter(cfg, model, norm_module)
    return adapter