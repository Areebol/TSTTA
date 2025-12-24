# adapted from https://github.com/DequanWang/tent/blob/master/tent.py
# Integrated with UP-PETSA Mechanism (Ours)

from typing import List
from copy import deepcopy
import os
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.optimizer import get_optimizer
from models.forecast import forecast
from datasets.loader import get_test_dataloader, get_train_dataloader
from utils.misc import prepare_inputs, mkdir
from config import get_norm_method
from tta.utils import save_tta_results, TTADataManager

# =========================================================================
# 1. UP-PETSA 核心组件 (诊断器, Loss, 治疗模块)
# =========================================================================

class CorrCoefLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = preds.reshape(-1)
        y = target.reshape(-1)
        var_x = torch.var(x)
        var_y = torch.var(y)
        if (var_x < self.eps) or (var_y < self.eps):
            return torch.zeros((), dtype=preds.dtype, device=preds.device)
        
        data = torch.stack([x, y], dim=0)
        corrmat = torch.corrcoef(data)
        corr_xy = corrmat[0, 1]
        corr_xy = torch.nan_to_num(corr_xy, nan=0.0, posinf=0.0, neginf=0.0)
        corr_xy = torch.clamp(corr_xy, -1.0, 1.0)
        return -corr_xy

class UncertaintyEstimator(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, hidden_dim=128):
        super().__init__()
        self.seq_proj = nn.Linear(seq_len, hidden_dim)
        self.pred_proj = nn.Linear(pred_len, hidden_dim)
        self.estimator = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), 
            nn.Softplus()
        )

    def forward(self, x_enc, x_pred):
        feat_seq = self.seq_proj(x_enc.permute(0, 2, 1)).mean(dim=1) 
        feat_pred = self.pred_proj(x_pred.permute(0, 2, 1)).mean(dim=1)
        combined = torch.cat([feat_seq, feat_pred], dim=-1)
        estimated_mse = self.estimator(combined)
        return estimated_mse

class UncertaintyAwareGCM(nn.Module):
    """
    带有 m_t 调制的 Low-Rank GCM (用于输出校准)
    """
    def __init__(self, window_len, n_var=1, hidden_dim=64, gating_init=0.01, var_wise=True, low_rank=16):
        super(UncertaintyAwareGCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        
        self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))
        self.low_rank = low_rank

        self.lora_A = nn.Parameter(torch.Tensor(window_len, self.low_rank))
        self.lora_B = nn.Parameter(torch.Tensor(self.low_rank, window_len, n_var))

        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # [Fix] 使用 normal 初始化避免梯度断裂
        nn.init.normal_(self.lora_B, mean=0.0, std=1e-3)
        nn.init.zeros_(self.bias)

    def forward(self, x, uncertainty_weight=1.0):
        weight = torch.einsum('ik,kjl->ijl', self.lora_A, self.lora_B)
        gate_val = torch.tanh(self.gating) 
        
        if self.var_wise:
            x_gated = gate_val * x
            delta = (torch.einsum('biv,iov->bov', x_gated, weight) + self.bias)
        else:
            x_gated = gate_val * x
            delta = (torch.einsum('biv,io->bov', x_gated, weight) + self.bias)

        if isinstance(uncertainty_weight, torch.Tensor):
            if uncertainty_weight.dim() == 1:
                uncertainty_weight = uncertainty_weight.view(-1, 1, 1)
            elif uncertainty_weight.dim() == 2:
                uncertainty_weight = uncertainty_weight.view(-1, 1, 1)
        
        x_final = x + uncertainty_weight * delta
        return x_final

class StandardGCM(nn.Module):
    """
    原始 TAFAS GCM (用于输入校准，保持轻量)
    """
    def __init__(self, window_len, n_var=1, hidden_dim=64, gating_init=0.01, var_wise=True):
        super(StandardGCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        if var_wise:
            self.weight = nn.Parameter(torch.Tensor(window_len, window_len, n_var))
        else:
            self.weight = nn.Parameter(torch.Tensor(window_len, window_len))
        self.weight.data.zero_()
        self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))

    def forward(self, x):
        if self.var_wise:
            x = x + torch.tanh(self.gating) * (torch.einsum('biv,iov->bov', x, self.weight) + self.bias)
        else:
            x = x + torch.tanh(self.gating) * (torch.einsum('biv,io->bov', x, self.weight) + self.bias)
        return x

class Calibration(nn.Module):
    def __init__(self, cfg):
        super(Calibration, self).__init__()
        self.cfg = cfg
        self.seq_len = cfg.DATA.SEQ_LEN
        self.pred_len = cfg.DATA.PRED_LEN
        self.n_var = cfg.DATA.N_VAR
        self.hidden_dim = cfg.TTA.TAFAS.HIDDEN_DIM
        self.gating_init = cfg.TTA.TAFAS.GATING_INIT
        self.var_wise = cfg.TTA.TAFAS.GCM_VAR_WISE
        
        # 输入校准使用标准 GCM (保持 TAFAS 特性)
        if cfg.MODEL.NAME == 'PatchTST':
            self.in_cali = StandardGCM(self.seq_len, 1, self.hidden_dim, self.gating_init, self.var_wise)
        else:
            self.in_cali = StandardGCM(self.seq_len, self.n_var, self.hidden_dim, self.gating_init, self.var_wise)
            
        # [修改] 输出校准使用我们的 UncertaintyAwareGCM
        # 注意: 需要从 config 读取 rank，如果没有默认 16
        rank = getattr(cfg.TTA.PETSA, 'RANK', 16)
        self.out_cali = UncertaintyAwareGCM(self.pred_len, self.n_var, self.hidden_dim, self.gating_init, self.var_wise, low_rank=rank)
        
    def input_calibration(self, inputs):
        enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
        enc_window = self.in_cali(enc_window)
        return enc_window, enc_window_stamp, dec_window, dec_window_stamp

    def output_calibration(self, outputs, m_t=1.0):
        # 接收 m_t 参数
        return self.out_cali(outputs, m_t)

# =========================================================================
# 2. Adapter (TAFAS Runner + UP-PETSA Logic)
# =========================================================================

class Adapter(nn.Module):
    def __init__(self, cfg, model: nn.Module, norm_module=None):
        super(Adapter, self).__init__()
        self.cfg = cfg
        self.model = model
        self.norm_module = norm_module
        self.device = next(self.model.parameters()).device
        self.test_loader = get_test_dataloader(cfg)
        self.test_data = self.test_loader.dataset.test

        # 初始化校准模块
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            self.cali = Calibration(cfg).to(self.device)
        
        # [Ours] 初始化诊断器
        self.uncertainty_estimator = UncertaintyEstimator(
            seq_len=cfg.DATA.SEQ_LEN,
            pred_len=cfg.DATA.PRED_LEN,
            n_vars=cfg.DATA.N_VAR
        ).to(self.device)

        # [Ours] 辅助 Loss
        self.person_cor = CorrCoefLoss()
        self.loss_alpha = getattr(cfg.TTA.PETSA, 'LOSS_ALPHA', 0.1)
        self.scale_factor = 5.0 # Sigmoid scaling
        self.threshold = None   # Auto-threshold

        # 冻结所有模型参数
        self._freeze_all_model_params()
        
        # 定义需要优化的参数 (这里主要是 Calibration 模块)
        # TAFAS 逻辑通常只优化 cali，或者部分 model 参数。
        # 为了兼容 UP-PETSA，我们重点优化 cali
        self.named_params_to_adapt = self._get_named_params_to_adapt()
        self.optimizer = get_optimizer(self.named_params_to_adapt.values(), cfg.TTA)
        
        # 状态备份 (用于 Reset)
        self.model_state, self.optimizer_state = self._copy_model_and_optimizer()
        self.cali_state = self._copy_cali() if self.cfg.TTA.TAFAS.CALI_MODULE else None
        
        # 调整 batch size
        if hasattr(self.test_loader.dataset, "get_test_num_windows"):
            batch_size = self.test_loader.dataset.get_test_num_windows()
        else:
            batch_size = len(self.test_loader.dataset)
        self.test_loader = get_test_dataloader(cfg, batch_size=batch_size)
        
        self.cur_step = cfg.DATA.SEQ_LEN - 2
        self.pred_step_end_dict = {}
        self.inputs_dict = {}
        self.n_adapt = 0

        self.mse_all = []
        self.mae_all = []

        ds = self.test_loader.dataset
        self.is_eved_like = (
            hasattr(ds, "get_num_test_csvs")
            and hasattr(ds, "get_test_csv_window_range")
            and hasattr(ds, "get_test_windows_for_csv")
        )
    
    def reset(self):
        self._load_model_and_optimizer()
    
    def _copy_model_and_optimizer(self):
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_state, optimizer_state

    def _copy_cali(self):
        return deepcopy(self.cali.state_dict())

    def _load_cali(self):
        if self.cfg.TTA.TAFAS.CALI_MODULE and self.cali_state is not None:
            self.cali.load_state_dict(deepcopy(self.cali_state), strict=True)

    def _load_model_and_optimizer(self):
        self.model.load_state_dict(deepcopy(self.model_state), strict=True)
        self.optimizer.load_state_dict(deepcopy(self.optimizer_state))
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            self._load_cali()
    
    def _get_all_models(self):
        models = [self.model]
        if self.norm_module is not None:
            models.append(self.norm_module)
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            models.append(self.cali)
        # [Note] Estimator 不参与 TTA 优化，所以不放这里
        return models

    def _freeze_all_model_params(self):
        for model in self._get_all_models():
            for param in model.parameters():
                param.requires_grad_(False)
    
    def _get_named_params_to_adapt(self):
        named_params_to_adapt = {}
        # 如果启用校准模块，优先优化校准模块
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            for name, param in self.cali.named_parameters():
                param.requires_grad_(True)
                named_params_to_adapt[f"cali.{name}"] = param
        else:
            # 原版 TAFAS 逻辑：如果没有校准模块，可能优化 Norm 层或其他
            # 这里简化处理：如果没有校准模块，UP-PETSA 实际上无法工作
            pass
        return named_params_to_adapt
    
    def switch_model_to_train(self):
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            self.cali.train()
    
    def switch_model_to_eval(self):
        self.model.eval()
        self.uncertainty_estimator.eval()
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            self.cali.eval()

    # [Ours] 核心：预训练诊断器并获取自动阈值
    def train_uncertainty_estimator(self, epochs=10, lr=1e-4):
        ckpt_name = f"UE_{self.cfg.MODEL.NAME}_{self.cfg.DATA.NAME}_sl{self.cfg.DATA.SEQ_LEN}_pl{self.cfg.DATA.PRED_LEN}.pth"
        save_dir = self.cfg.TRAIN.CHECKPOINT_DIR
        mkdir(save_dir)
        ue_ckpt_path = os.path.join(save_dir, ckpt_name)
        
        # 内部函数：计算训练集平均 MSE
        def calc_threshold():
            print("[Ours-TAFAS] Calculating auto-threshold from training set...")
            train_loader = get_train_dataloader(self.cfg)
            self.model.eval()
            mse_accum = 0.0
            count = 0
            for i, inputs in enumerate(train_loader):
                enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
                with torch.no_grad():
                    pred_base, ground_truth = forecast(self.cfg, (enc_window, enc_window_stamp, dec_window, dec_window_stamp), self.model, self.norm_module)
                    mse_accum += F.mse_loss(pred_base, ground_truth).item()
                    count += 1
                if count >= 200: break # Speed up
            return mse_accum / count

        if os.path.exists(ue_ckpt_path):
            print(f"[Ours-TAFAS] Loading pretrained UE from: {ue_ckpt_path}")
            self.uncertainty_estimator.load_state_dict(torch.load(ue_ckpt_path))
            return calc_threshold()

        print(f"\n[Ours-TAFAS] Training Uncertainty Estimator...")
        train_loader = get_train_dataloader(self.cfg)
        ue_optimizer = torch.optim.Adam(self.uncertainty_estimator.parameters(), lr=lr)
        criterion = nn.MSELoss() 
        self.model.eval() 
        self.uncertainty_estimator.train()
        
        for epoch in range(epochs):
            loss_list = []
            for i, inputs in enumerate(train_loader):
                enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
                with torch.no_grad():
                    pred_base, ground_truth = forecast(self.cfg, (enc_window, enc_window_stamp, dec_window, dec_window_stamp), self.model, self.norm_module)
                    real_mse = F.mse_loss(pred_base, ground_truth, reduction='none').mean(dim=(1, 2), keepdim=True)
                
                estimated_mse = self.uncertainty_estimator(enc_window, pred_base)
                loss = criterion(estimated_mse, real_mse)
                ue_optimizer.zero_grad()
                loss.backward()
                ue_optimizer.step()
                loss_list.append(loss.item())
            
            if (epoch+1)%2==0: print(f"  UE Epoch {epoch+1} Loss: {np.mean(loss_list):.6f}")

        torch.save(self.uncertainty_estimator.state_dict(), ue_ckpt_path)
        self.uncertainty_estimator.eval()
        return calc_threshold()

    def _get_mt(self, uncertainty_score):
        return torch.sigmoid(self.scale_factor * (uncertainty_score - self.threshold))

    @torch.enable_grad()
    def adapt_tafas(self):
        # 1. 自动阈值
        self.threshold = self.train_uncertainty_estimator()
        print(f"[Ours-TAFAS] Start TTA. Auto-Threshold: {self.threshold:.4f}")

        is_last = False
        
        self.switch_model_to_eval()
        for idx, inputs in enumerate(self.test_loader):
            enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
            batch_start = 0
            batch_end = 0
            batch_idx = 0
            self.cur_step = self.cfg.DATA.SEQ_LEN - 2
            
            while batch_end < len(enc_window_all):
                # TAFAS 逻辑：按周期性确定 batch_size (PAAS)
                enc_window_first = enc_window_all[batch_start]
                if self.cfg.TTA.TAFAS.PAAS:
                    period, batch_size = self._calculate_period_and_batch_size(enc_window_first)
                else:
                    batch_size = self.cfg.TTA.TAFAS.BATCH_SIZE
                    period = batch_size - 1
                
                batch_end = batch_start + batch_size
                if batch_end > len(enc_window_all):
                    batch_end = len(enc_window_all)
                    batch_size = batch_end - batch_start
                    is_last = True

                self.cur_step += batch_size
                
                # 当前小批次数据
                inputs_batch = (
                    enc_window_all[batch_start:batch_end], 
                    enc_window_stamp_all[batch_start:batch_end], 
                    dec_window_all[batch_start:batch_end], 
                    dec_window_stamp_all[batch_start:batch_end]
                )
                
                self.pred_step_end_dict[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN
                self.inputs_dict[batch_idx] = inputs_batch
                
                # --- [Ours] Step 1: 诊断 ---
                with torch.no_grad():
                    # 这里我们需要对当前输入做一次基础预测来估算不确定性
                    # 为了不影响 TAFAS 的状态，我们不做 input calibration 或者假设 input calibration 是轻量的
                    pred_base_diag, _ = forecast(self.cfg, inputs_batch, self.model, self.norm_module)
                    
                    # 使用 batch 的第一条数据或者整个 batch 的平均特征来估算？
                    # TAFAS 的 batch 是时间上连续的片段。我们对整个 batch 估算。
                    unc_score = self.uncertainty_estimator(inputs_batch[0], pred_base_diag)
                    batch_unc = unc_score.mean().item()
                
                # --- [Ours] Step 2: 决策 ---
                should_adapt = batch_unc > self.threshold
                m_t_tensor = self._get_mt(unc_score).detach() # [B, 1, 1]

                # --- [Ours] Step 3: 治疗 (调用 TAFAS 的 PAAS 机制) ---
                # 处理之前的全量 GT (Delayed Adaptation)
                # 对于旧数据，我们假设我们想尽可能学，或者复用当时的 mt。简化起见，用 mt=1 或者重新估算。
                # 这里为了简单，如果有全量 GT，我们强制 mt=1.0 (因为全量 GT 很难得)
                self._adapt_with_full_ground_truth_if_available(m_t_override=1.0)
                
                if should_adapt:
                    # 使用 PAAS (部分 GT) 进行适应，传入 m_t
                    pred, ground_truth = self._adapt_with_partial_ground_truth(inputs_batch, period, batch_size, batch_idx, m_t=m_t_tensor)
                else:
                    # 不适应，直接预测
                    pred, ground_truth = self._inference_only(inputs_batch, m_t_tensor)

                if self.cfg.TTA.TAFAS.ADJUST_PRED:
                    pred, ground_truth = self._adjust_prediction(pred, inputs_batch, batch_size, period, m_t=m_t_tensor)
                
                mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                self.mse_all.append(mse)
                self.mae_all.append(mae)
                
                batch_start = batch_end
                batch_idx += 1

            assert self.cur_step == len(self.test_data) - self.cfg.DATA.PRED_LEN - 1

        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        
        print('After UP-PETSA integrated TAFAS')
        print(f'Test MSE: {self.mse_all.mean():.4f}, Test MAE: {self.mae_all.mean():.4f}')
        print(f'Auto Threshold: {self.threshold:.4f}, Adapt Count: {self.n_adapt}')
        
        save_tta_results(
            tta_method='Ours-PIR-TAFAS',
            seed=self.cfg.SEED,
            model_name=self.cfg.MODEL.NAME,
            dataset_name=self.cfg.DATA.NAME,
            pred_len=self.cfg.DATA.PRED_LEN,
            mse_after_tta=self.mse_all.mean(),
            mae_after_tta=self.mae_all.mean(),
        )
        self.model.eval()

    def _calculate_period_and_batch_size(self, enc_window_first):
        fft_result = torch.fft.rfft(enc_window_first - enc_window_first.mean(dim=0), dim=0)
        amplitude = torch.abs(fft_result)
        power = torch.mean(amplitude ** 2, dim=0)
        try:
            period = enc_window_first.shape[0] // torch.argmax(amplitude[:, power.argmax()]).item()
        except:
            period = 24
        period *= self.cfg.TTA.TAFAS.PERIOD_N
        batch_size = period + 1
        return period, batch_size

    def _inference_only(self, inputs, m_t):
        if self.cfg.TTA.TAFAS.CALI_MODULE and self.cfg.MODEL.NAME != 'PatchTST':
            inputs = self.cali.input_calibration(inputs)
        
        with torch.no_grad():
            pred, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
            if self.cfg.TTA.TAFAS.CALI_MODULE:
                pred = self.cali.output_calibration(pred, m_t)
        return pred, ground_truth

    def _adapt_with_full_ground_truth_if_available(self, m_t_override=1.0):
        # 这里的逻辑是对已经流逝的、现在有了完整标签的数据进行补充训练
        while self.cur_step >= self.pred_step_end_dict[min(self.pred_step_end_dict.keys())]:
            batch_idx_available = min(self.pred_step_end_dict.keys())
            inputs_history = self.inputs_dict.pop(batch_idx_available)
            for _ in range(self.cfg.TTA.TAFAS.STEPS):
                self.n_adapt += 1
                self.switch_model_to_train()

                if self.cfg.TTA.TAFAS.CALI_MODULE and self.cfg.MODEL.NAME != 'PatchTST':
                    inputs_cali = self.cali.input_calibration(inputs_history)[0] # input calibration
                    # Hacky way to repack because calibration returns tuples
                    # Re-prepare inputs is complex inside here, simplifying:
                    # We assume calibration modifies enc_window. 
                    # For strict TAFAS compliance we should use input_calibration properly.
                    # But simpler here: forecast handles tuple.
                    pass 

                # 重新预测
                if self.cfg.TTA.TAFAS.CALI_MODULE and self.cfg.MODEL.NAME != 'PatchTST':
                    inputs_for_fc = self.cali.input_calibration(inputs_history)
                else:
                    inputs_for_fc = inputs_history

                pred, ground_truth = forecast(self.cfg, inputs_for_fc, self.model, self.norm_module)
                
                if self.cfg.TTA.TAFAS.CALI_MODULE:
                    pred = self.cali.output_calibration(pred, m_t=m_t_override)
                
                # 全量 GT 训练使用 MSE 还是 复合 Loss? 建议使用复合 Loss
                # 但为了稳定，全量 GT 通常使用 MSE 即可，或者保持一致。这里用 MSE 简单点。
                loss = F.mse_loss(pred, ground_truth)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.switch_model_to_eval()
            
            self.pred_step_end_dict.pop(batch_idx_available)

    def _adapt_with_partial_ground_truth(self, inputs, period, batch_size, batch_idx, m_t):
        for _ in range(self.cfg.TTA.TAFAS.STEPS):
            self.n_adapt += 1
            
            if self.cfg.TTA.TAFAS.CALI_MODULE and self.cfg.MODEL.NAME != 'PatchTST':
                inputs_fc = self.cali.input_calibration(inputs)
            else:
                inputs_fc = inputs
            
            pred, ground_truth = forecast(self.cfg, inputs_fc, self.model, self.norm_module)
            
            if self.cfg.TTA.TAFAS.CALI_MODULE:
                pred = self.cali.output_calibration(pred, m_t)
            
            # 取出部分 GT (PAAS)
            pred_partial, gt_partial = pred[0][:period], ground_truth[0][:period]
            # 为了适配 batch 维度 (TAFAS 这里 inputs 实际上 batch_size 对应的是时间窗切片，所以 dim 0 是 steps)
            # TAFAS 的 inputs 进来通常是 [Batch, Seq, Var]，但在 TAFAS 循环里，batch_size 实际上指的是 Time Steps 的切分
            # 这里的维度处理比较 trick。如果 enc_window_all 已经是切好的，pred 是 [TimeSteps, PredLen, Var]
            # pred_partial: [Period, PredLen, Var] ? 不，TAFAS 是针对单样本的长序列切分。
            # 假设 inputs_batch 维度是 [BatchSize_Time, Seq, Var]
            
            # --- [Ours] 使用复合 Loss ---
            # Huber
            loss_reg = F.huber_loss(pred_partial, gt_partial, delta=0.5)
            # Freq
            loss_freq = (torch.fft.rfft(pred_partial, dim=1) - torch.fft.rfft(gt_partial, dim=1)).abs().mean()
            # Corr
            loss_corr = self.person_cor(pred_partial, gt_partial)
            # Mean
            loss_mean = F.l1_loss(pred_partial.mean(dim=1), gt_partial.mean(dim=1))
            
            loss = loss_reg + self.loss_alpha * loss_freq + loss_corr + loss_mean
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return pred, ground_truth

    @torch.no_grad()
    def _adjust_prediction(self, pred, inputs, batch_size, period, m_t):
        # 调整预测，用更新后的模型再推一次重叠部分
        if self.cfg.TTA.TAFAS.CALI_MODULE and self.cfg.MODEL.NAME != 'PatchTST':
            inputs = self.cali.input_calibration(inputs)
        
        pred_after_adapt, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
        
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            pred_after_adapt = self.cali.output_calibration(pred_after_adapt, m_t)
        
        # 将重叠部分的预测更新为适应后的结果
        for i in range(batch_size-1):
            if i < pred.shape[0] and i < pred_after_adapt.shape[0]:
                pred[i, period-i:] = pred_after_adapt[i, period-i:]
        
        return pred, ground_truth
    
    # 兼容 EVED (如果需要，可按类似 adapt_tafas 的逻辑修改，这里暂略，直接继承原逻辑或抛出)
    def adapt(self):
        if getattr(self, "is_eved_like", False):
            # 简单起见，如果需要 EVED 支持，需要把 adapt_tafas 的逻辑复制到 adapt_tafas_eved 里
            print("EVED dataset not fully supported in this UP-PETSA integration yet.")
            raise NotImplementedError
        else:
            self.adapt_tafas()

# 将原来的 build_adapter 替换为 build_tta_runner
def build_tta_runner(cfg, model, norm_module=None):
    # 如果 main.py 没有传 norm_module，但配置文件启用了，我们就自己构建一个
    # 这保持了和你之前 ours.py 逻辑的一致性
    if norm_module is None and cfg.NORM_MODULE.ENABLE:
        from models.build import build_norm_module
        norm_module = build_norm_module(cfg)
        # 注意：model 已经在 device 上了，norm_module 也要转过去
        if next(model.parameters()).is_cuda:
            norm_module = norm_module.cuda()
            
    return Adapter(cfg, model, norm_module)