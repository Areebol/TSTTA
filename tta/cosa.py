from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pandas as pd
import os
from collections import deque, defaultdict
from models.optimizer import get_optimizer
from models.forecast import forecast
from datasets.loader import get_test_dataloader
from utils.misc import prepare_inputs
from config import get_norm_method
import time
from tta.utils import save_tta_results

class SimpleOutputAdapter(nn.Module):
    def __init__(self, pred_len: int, buffer_context_size: int = 5, n_vars: int = 1, 
                 var_wise_gating: bool = False, num_layers: int = 1, hidden_dim: int = 64, only_context: bool=False):
        super().__init__()
        self.buffer_context_size = buffer_context_size
        self.n_vars = n_vars
        self.var_wise = var_wise_gating
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.only_context = only_context
        if self.only_context:
            self.pre_len = 0
        else:
            self.pred_len = pred_len
            
        input_dim = self.pred_len + self.buffer_context_size
        output_dim = self.pred_len
        
        if self.var_wise:
            if self.num_layers == 1:
                self.fc_layers = nn.ModuleList([
                    nn.Linear(input_dim, output_dim) for _ in range(n_vars)
                ])
            else:
                self.fc_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim, output_dim)
                    ) for _ in range(n_vars)
                ])
            self.gate = nn.Parameter(torch.zeros(n_vars)) 
        else:
            if self.num_layers == 1:
                self.fc = nn.Linear(input_dim, output_dim)
            else:
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, output_dim)
                )
            self.gate = nn.Parameter(torch.zeros(1))
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        if self.var_wise:
            for fc_module in self.fc_layers:
                if self.num_layers == 1:
                    nn.init.xavier_uniform_(fc_module.weight, gain=0.1)
                    nn.init.zeros_(fc_module.bias)
                else:
                    for layer in fc_module:
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_uniform_(layer.weight, gain=0.1)
                            nn.init.zeros_(layer.bias)
        else:
            if self.num_layers == 1:
                nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
                nn.init.zeros_(self.fc.bias)
            else:
                for layer in self.fc:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.1)
                        nn.init.zeros_(layer.bias)
        
    def forward(self, y: torch.Tensor, context_data: torch.Tensor = None):
        if context_data is None:
            return y
        
        batch_size, pred_len, n_vars = y.shape
        
        if self.var_wise:
            corrections = []
            for var_idx in range(n_vars):
                y_var = y[:, :, var_idx]
                if self.only_context:
                    combined_input = context_data
                else:
                    combined_input = torch.cat([y_var, context_data], dim=-1)
                
                
                correction_var = self.fc_layers[var_idx](combined_input)
                corrections.append(correction_var.unsqueeze(-1))

            correction = torch.cat(corrections, dim=-1)
            
            gating_factor = torch.tanh(self.gate).unsqueeze(0).unsqueeze(0)
            
        else:
            y_flattened = y.transpose(1, 2).contiguous().view(batch_size * n_vars, pred_len)
            context_repeated = context_data.unsqueeze(1).repeat(1, n_vars, 1).view(batch_size * n_vars, -1)
            if self.only_context:
                combined_input = context_repeated
            else:
                combined_input = torch.cat([y_flattened, context_repeated], dim=-1)
            correction = self.fc(combined_input)
            correction = correction.view(batch_size, n_vars, pred_len).transpose(1, 2)
            gating_factor = torch.tanh(self.gate)
        
        return y + gating_factor * correction
    

class SimpleAdapter(nn.Module):
    def __init__(self, cfg, model: nn.Module, norm_module=None):
        super(SimpleAdapter, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.model = model
        self.norm_method = get_norm_method(cfg)
        self.norm_module = norm_module
        self.test_loader = get_test_dataloader(cfg)
        self.test_data = self.test_loader.dataset.test

        self.buffer_context_size = getattr(cfg.TTA.SIMPLE, 'BUFFER_CONTEXT_SIZE', 5)
        self.adapt_steps = getattr(cfg.TTA.SIMPLE, 'STEPS', 20)
        
        self.paas_enabled = getattr(cfg.TTA.SIMPLE, 'PAAS', False)
        self.period_n = getattr(cfg.TTA.SIMPLE, 'PERIOD_N', 1)
        
        self.fast_adaptation = getattr(cfg.TTA.SIMPLE, 'FAST_ADAPTATION', False)
        self.adaptive_lr = getattr(cfg.TTA.SIMPLE, 'ADAPTIVE_LR', False) 
        self.max_lr = getattr(cfg.TTA.SIMPLE, 'MAX_LR', 0.005)
        self.min_lr = getattr(cfg.TTA.SIMPLE, 'MIN_LR', 0.0001)
        self.convergence_threshold = getattr(cfg.TTA.SIMPLE, 'CONVERGENCE_THRESHOLD', 1e-4)
        self.var_wise_gating = getattr(cfg.TTA.SIMPLE, 'VAR_WISE_GATING', False)
        
        self.save_csv = getattr(cfg.TTA.SIMPLE, 'SAVE_CSV', False)
        self.csv_predictions = [] 

        self.save_paas_csv = getattr(cfg.TTA.SIMPLE, 'SAVE_PAAS_CSV', False)
        self.paas_batch_info = []

        self.adapter_layers = getattr(cfg.TTA.SIMPLE, 'ADAPTER_LAYERS', 1)
        self.hidden_dim = getattr(cfg.TTA.SIMPLE, 'HIDDEN_DIM', 64)

        # Per-Batch LR Reset
        self.per_batch_lr_reset = getattr(cfg.TTA.SIMPLE, 'PER_BATCH_LR_RESET', True)

        self.loss_history = deque(maxlen=5)
        self.current_lr = getattr(cfg.TTA.SOLVER, 'BASE_LR', 0.001) 

        self.sample_history = deque(maxlen=200) 
        self.current_time_idx = 0
        
        self.time_stats = defaultdict(float)
        self.time_counts = defaultdict(int)
        
        self.output_adapter = SimpleOutputAdapter(
            pred_len=cfg.DATA.PRED_LEN,
            buffer_context_size=self.buffer_context_size,
            n_vars=cfg.MODEL.c_out,
            var_wise_gating=self.var_wise_gating,
            num_layers=self.adapter_layers,
            hidden_dim=self.hidden_dim,
            only_context=cfg.TTA.SIMPLE.ONLY_CONTEXT,
        ).cuda()
        
        self.adapters_enabled = False
        self.step_count = 0
        
        self._freeze_all_model_params()
        self._unfreeze_adapter_params()
        
        self.optimizer = get_optimizer(self.output_adapter.parameters(), cfg.TTA)
        
        self.model_state, self.optimizer_state = self._copy_model_and_optimizer()

        batch_size = len(self.test_loader.dataset)
        self.test_loader = get_test_dataloader(cfg, batch_size=batch_size)
        self.cur_step = cfg.DATA.SEQ_LEN - 2
        self.n_adapt = 0

        self.mse_all = []
        self.mae_all = []

        ds = self.test_loader.dataset
        self.is_eved_like = (
            hasattr(ds, "get_num_test_csvs")
            and hasattr(ds, "get_test_csv_window_range")
            and hasattr(ds, "get_test_windows_for_csv")
        )
        # 用于“历史/部分可观测”更新的状态
        self.pred_step_end_dict = {}
        self.inputs_dict = {}


    def count_parameters(self):
        trainable_params = []
        total_sum = 0
        
        for name, param in self.named_parameters():
            param_info = {
                "name": name,
                "requires_grad": param.requires_grad,
                "size": list(param.size()),
                "numel": int(param.numel())
            }
            trainable_params.append(param_info)
            
            if param.requires_grad:
                total_sum += int(param.numel())
        
        param_json = {
            "model": "SimpleAdapter",
            "parameters": {
                "trainable_params": trainable_params,
                "total_params": total_sum
            }
        }    
    
    def forward(self, enc_window, enc_window_stamp, dec_window, dec_window_stamp):
        raise NotImplementedError
    
    def reset(self):
        self._load_model_and_optimizer()
        self.adapters_enabled = False
        self.step_count = 0
        
        self.loss_history.clear()
        self.current_lr = getattr(self.cfg.TTA.SOLVER, 'BASE_LR', 0.001)
    
    def _copy_model_and_optimizer(self):
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_state, optimizer_state

    def _load_model_and_optimizer(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
    
    def _get_all_models(self):
        models = [self.model, self.output_adapter]
        if self.norm_module is not None:
            models.append(self.norm_module)
        return models

    def _freeze_all_model_params(self):
        for param in self.model.parameters():
            param.requires_grad_(False)
        if self.norm_module is not None:
            for param in self.norm_module.parameters():
                param.requires_grad_(False)
    
    def _unfreeze_adapter_params(self):
        for param in self.output_adapter.parameters():
            param.requires_grad_(True)
    
    def switch_model_to_train(self):
        self.model.eval() 
        if self.norm_module is not None:
            self.norm_module.eval()
        self.output_adapter.train()
    
    def switch_model_to_eval(self):
        self.model.eval()
        if self.norm_module is not None:
            self.norm_module.eval()
        self.output_adapter.eval()
    

    def update_memory_buffer(self, targets: torch.Tensor, prediction: torch.Tensor):

        batch_size = targets.shape[0]
        
        # Store only batch-level statistics (omit individual sample info for speed improvement)
        seq_mse = F.mse_loss(prediction, targets).item()
        
        # Store only batch-wide average values
        batch_info = {
            'time_idx': self.current_time_idx,
            'target_mean': targets.mean().item(),
        }
        self.sample_history.append(batch_info)
        
        self.current_time_idx += batch_size
        self.step_count += 1
        
        if not self.adapters_enabled:
            self.adapters_enabled = True
        
        return seq_mse
    
    def _calculate_period_and_batch_size(self, enc_window_first):

        fft_result = torch.fft.rfft(enc_window_first - enc_window_first.mean(dim=0), dim=0)
        amplitude = torch.abs(fft_result)
        power = torch.mean(amplitude ** 2, dim=0)
        
        dominant_freq = None
        try:
            dominant_freq_idx = torch.argmax(amplitude[:, power.argmax()]).item()
            dominant_freq = float(dominant_freq_idx / enc_window_first.shape[0])
            period = enc_window_first.shape[0] // dominant_freq_idx
        except:
            period = 24 
            dominant_freq = 1.0 / 24 
            
        period *= self.period_n
        batch_size = period + 1
        return period, batch_size, dominant_freq

    def _get_individual_context_for_batch(self, batch_size, current_batch_idx):

        if len(self.sample_history) == 0:
            return torch.zeros(batch_size, self.buffer_context_size, device='cuda')
        
        history_size = min(self.buffer_context_size, len(self.sample_history))
        context_values = [self.sample_history[-(i+1)]['target_mean'] 
                         for i in range(history_size)]
        
        if len(context_values) < self.buffer_context_size:
            last_val = context_values[-1] if context_values else 0.0
            context_values.extend([last_val] * (self.buffer_context_size - len(context_values)))
        
        context_tensor = torch.tensor(context_values, dtype=torch.float32, device='cuda')
        return context_tensor.unsqueeze(0).expand(batch_size, -1) 
    
    def _adaptive_learning_rate(self, current_loss: float, step: int, batch_idx: int = 0) -> float:
        # Per-Batch Learning Rate Reset
        if step == 0 and self.per_batch_lr_reset:
            self.current_lr = getattr(self.cfg.TTA.SOLVER, 'BASE_LR', 0.001)
            self.loss_history.append(current_loss)
            return self.current_lr

        self.loss_history.append(current_loss)
        recent_losses = list(self.loss_history)[-3:]

        if len(recent_losses) < 2:
            return self.current_lr

        loss_trend = recent_losses[-1] - recent_losses[0]
        loss_variance = torch.tensor(recent_losses).var().item()

        if loss_trend > 0 and loss_variance < 1e-6:
            self.current_lr = min(self.current_lr * 1.2, self.max_lr)
        elif loss_trend < -0.01:
            self.current_lr = min(self.current_lr * 1.05, self.max_lr)
        elif abs(loss_trend) < 1e-6:
            self.current_lr = max(self.current_lr * 0.8, self.min_lr)

        if step >= 1:
            cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(step * 3.14159 / self.adapt_steps)))
            self.current_lr = self.min_lr + (self.current_lr - self.min_lr) * cosine_factor

        return self.current_lr
    
    def _save_batch_predictions_to_csv(self, original_pred: torch.Tensor, tta_pred: torch.Tensor, ground_truth: torch.Tensor, batch_idx: int):
        batch_size, pred_len, n_vars = tta_pred.shape
        
        original_pred_np = original_pred.detach().cpu().numpy()
        tta_pred_np = tta_pred.detach().cpu().numpy()
        ground_truth_np = ground_truth.detach().cpu().numpy()
        
        for sample_idx in range(batch_size):
            for time_idx in range(pred_len):
                for var_idx in range(n_vars):
                    orig_val = float(original_pred_np[sample_idx, time_idx, var_idx])
                    tta_val = float(tta_pred_np[sample_idx, time_idx, var_idx])
                    gt_val = float(ground_truth_np[sample_idx, time_idx, var_idx])
                    
                    row_data = {
                        'batch_idx': batch_idx,
                        'sample_idx': sample_idx,
                        'global_sample_idx': batch_idx * batch_size + sample_idx,
                        'timestep': time_idx + 1, 
                        'variable_idx': var_idx,
                        'original_prediction': orig_val,
                        'tta_prediction': tta_val,
                        'ground_truth': gt_val,
                        'tta_improvement': tta_val - orig_val,
                        'original_absolute_error': float(abs(orig_val - gt_val)),
                        'tta_absolute_error': float(abs(tta_val - gt_val)),
                        'original_squared_error': float((orig_val - gt_val)**2),
                        'tta_squared_error': float((tta_val - gt_val)**2),
                        'error_improvement': float(abs(orig_val - gt_val) - abs(tta_val - gt_val)) 
                    }
                    self.csv_predictions.append(row_data)
    
    def _export_predictions_to_csv(self):
        """Export stored predictions to CSV file"""
        if not self.csv_predictions:
            return
        
        df = pd.DataFrame(self.csv_predictions)
        
        model_name = self.cfg.MODEL.NAME
        dataset_name = self.cfg.DATA.NAME
        pred_len = self.cfg.DATA.PRED_LEN
        
        csv_dir = os.path.join(self.cfg.RESULT_DIR, "csv_predictions", "SIMPLE")
        os.makedirs(csv_dir, exist_ok=True)
        
        csv_filename = f"{model_name}_{dataset_name}_pred{pred_len}_predictions.csv"
        csv_path = os.path.join(csv_dir, csv_filename)
        
        df.to_csv(csv_path, index=False)
    
    def _save_paas_batch_info(self, batch_idx: int, batch_start: int, calculated_batch_size: int, 
                             actual_batch_size: int, period: int = None, fft_dominant_freq: float = None):
        batch_info = {
            'batch_idx': batch_idx,
            'batch_start_idx': batch_start,
            'calculated_batch_size': calculated_batch_size,
            'actual_batch_size': actual_batch_size,
            'period': period if period is not None else 'N/A',
            'fft_dominant_frequency': fft_dominant_freq if fft_dominant_freq is not None else 'N/A',
            'paas_enabled': self.paas_enabled,
            'period_n_multiplier': self.period_n
        }
        self.paas_batch_info.append(batch_info)
    
    def _export_paas_info_to_csv(self):
        if not self.paas_batch_info:
            return
        
        df = pd.DataFrame(self.paas_batch_info)
        
        model_name = self.cfg.MODEL.NAME
        dataset_name = self.cfg.DATA.NAME
        pred_len = self.cfg.DATA.PRED_LEN
        
        csv_dir = os.path.join(self.cfg.RESULT_DIR, "csv_predictions", "SIMPLE")
        os.makedirs(csv_dir, exist_ok=True)
        
        csv_filename = f"{model_name}_{dataset_name}_pred{pred_len}_paas_batch_sizes.csv"
        csv_path = os.path.join(csv_dir, csv_filename)
        
        df.to_csv(csv_path, index=False)
    
    @torch.enable_grad()
    def _adapt_with_full_ground_truth_if_available_simple(self):
        """
        使用已经完整可观测的历史 batch（其预测窗口已全部落在当前时间之前）
        对 output_adapter 做全窗口监督更新。
        """
        if not self.pred_step_end_dict:
            return
                
        # 部分更新步数使用 self.adapt_steps 或在 fast_adaptation 下限制
        effective_steps = self.adapt_steps
        if self.fast_adaptation:
            effective_steps = min(self.adapt_steps, 5)

        while self.cur_step >= self.pred_step_end_dict[min(self.pred_step_end_dict.keys())]:
            batch_idx_available = min(self.pred_step_end_dict.keys())
            inputs_history = self.inputs_dict.pop(batch_idx_available)
            
            # 使用原始模型得到历史 batch 的预测和标签
            pred_hist, gt_hist = forecast(self.cfg, inputs_history, self.model, self.norm_module)

            # 构造对应的 context_data（批大小根据 inputs_history 的 enc_window 大小来估计）
            enc_window_hist, _, _, _ = inputs_history
            batch_size_hist = enc_window_hist.size(0)
            context_hist = self._get_individual_context_for_batch(batch_size_hist, batch_idx_available)

            # 这里历史 batch 不再用当前 batch 的标签，而是用它自身的 ground_truth
            # 重复若干步（沿用 self.adapt_steps）
            for step in range(effective_steps):
                self.n_adapt += 1
                self.switch_model_to_train()

                # 通过 output_adapter 得到校正后的预测
                adapted_pred = self.output_adapter(pred_hist, context_hist)

                # 全窗口 MSE + L2 正则
                loss = F.mse_loss(adapted_pred, gt_hist)
                # if not hasattr(self, '_adapter_params'):
                #     self._adapter_params = list(self.output_adapter.parameters())
                # l2_reg = sum(p.pow(2).sum() for p in self._adapter_params if p.requires_grad)
                # loss = loss + 1e-4 * l2_reg

                # 可选自适应学习率
                if self.adaptive_lr and self.fast_adaptation:
                    current_lr = self._adaptive_learning_rate(loss.item(), step, batch_idx_available)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = current_lr

                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.output_adapter.parameters(), max_norm=0.1)
                self.optimizer.step()

                self.switch_model_to_eval()

            # 该 batch 已用完，移除记录的结束时间
            self.pred_step_end_dict.pop(batch_idx_available)

    @torch.enable_grad()
    def _adapt_with_partial_ground_truth_simple(self, inputs, period, batch_size, batch_idx, context_data):
        """
        使用当前 batch 中“已经过去的时间步”（长度 period）的 ground truth
        对 output_adapter 做部分窗口监督更新。
        """
        # 部分更新步数使用 self.adapt_steps 或在 fast_adaptation 下限制
        effective_steps = self.adapt_steps
        if self.fast_adaptation:
            effective_steps = min(self.adapt_steps, 5)

        # 原始模型预测
        pred, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)

        for step in range(effective_steps):
            self.n_adapt += 1
            self.switch_model_to_train()

            # 通过 output_adapter 生成校正预测
            adapted_pred = self.output_adapter(pred, context_data)

            # 只使用前 period 个时间步的标签做更新
            # 这里与 tafas 中 pred[0][:period] 一致：只对第一个样本的前 period 步做更新
            pred_partial = adapted_pred[0][:period]
            gt_partial = ground_truth[0][:period]
            loss = F.mse_loss(pred_partial, gt_partial)

            # # L2 正则与原代码保持一致
            # if not hasattr(self, '_adapter_params'):
            #     self._adapter_params = list(self.output_adapter.parameters())
            # l2_reg = sum(p.pow(2).sum() for p in self._adapter_params if p.requires_grad)
            # loss = loss + 1e-4 * l2_reg

            # 可选自适应学习率
            if self.adaptive_lr and self.fast_adaptation:
                current_lr = self._adaptive_learning_rate(loss.item(), step, batch_idx)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr

            self.optimizer.zero_grad()
            loss.backward()

            # if self.fast_adaptation:
            #     max_norm = max(0.05, min(0.5, loss.item()))
            #     torch.nn.utils.clip_grad_norm_(self.output_adapter.parameters(), max_norm=max_norm)
            # else:
            #     torch.nn.utils.clip_grad_norm_(self.output_adapter.parameters(), max_norm=0.1)

            self.optimizer.step()
            self.switch_model_to_eval()

    @torch.enable_grad()
    def adapt_simple(self):
        batch_start = 0
        batch_end = 0
        batch_idx = 0
        
        total_start_time = time.time()
        
        self.switch_model_to_eval()
        
        for _, inputs in enumerate(self.test_loader):
            enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
            
            while batch_end < len(enc_window_all):
                calculated_batch_size = None
                period = None
                dominant_freq = None
                
                if self.paas_enabled:
                    period_start = time.time()
                    enc_window_first = enc_window_all[batch_start]
                    period, calculated_batch_size, dominant_freq = self._calculate_period_and_batch_size(enc_window_first)
                    batch_size = calculated_batch_size
                    self.time_stats['paas_period_calculation'] += time.time() - period_start
                    self.time_counts['paas_period_calculation'] += 1
                else:
                    batch_size = getattr(self.cfg.TTA.SIMPLE, 'BATCH_SIZE', 64)
                    calculated_batch_size = batch_size
                    period = batch_size - 1
                
                batch_end = batch_start + batch_size
                if batch_end > len(enc_window_all):
                    batch_end = len(enc_window_all)
                    batch_size = batch_end - batch_start
                
                if self.save_paas_csv:
                    self._save_paas_batch_info(
                        batch_idx=batch_idx,
                        batch_start=batch_start,
                        calculated_batch_size=calculated_batch_size,
                        actual_batch_size=batch_size,
                        period=period,
                        fft_dominant_freq=dominant_freq
                    )

                self.cur_step += batch_size

                batch_inputs = (
                    enc_window_all[batch_start:batch_end], 
                    enc_window_stamp_all[batch_start:batch_end], 
                    dec_window_all[batch_start:batch_end], 
                    dec_window_stamp_all[batch_start:batch_end]
                )

                # 记录当前 batch 的“预测结束时间”和输入，用于后续的 full-ground-truth 更新
                self.pred_step_end_dict[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN
                self.inputs_dict[batch_idx] = batch_inputs

                # 运行基础预测，仅用于后续评估/保存
                pred_start = time.time()
                base_pred, ground_truth = forecast(self.cfg, batch_inputs, self.model, self.norm_module)
                original_pred = base_pred.clone()
                self.time_stats['base_prediction'] += time.time() - pred_start
                self.time_counts['base_prediction'] += 1

                # 先利用已经完全可观测的历史 batch 做完整监督更新
                self._adapt_with_full_ground_truth_if_available_simple()

                # 构造当前 batch 的 context
                context_start = time.time()
                context_data = self._get_individual_context_for_batch(batch_size, batch_idx)
                self.time_stats['context_generation'] += time.time() - context_start
                self.time_counts['context_generation'] += 1

                # 使用当前 batch 的“部分可观测数据”做更新（period 内）
                if self.adapters_enabled and period is not None and period > 0:
                    adapt_start = time.time()
                    self._adapt_with_partial_ground_truth_simple(
                        batch_inputs, period, batch_size, batch_idx, context_data
                    )
                    self.time_stats['total_adaptation'] += time.time() - adapt_start
                    self.time_counts['total_adaptation'] += 1

                # 适配后再生成最终预测（基于基础预测 + output_adapter）
                final_start = time.time()
                with torch.no_grad():
                    pred_after_adapt = self.output_adapter(base_pred, context_data)
                    # 若启用 PAAS，对后半段进行替换（与原逻辑保持兼容）
                    if self.paas_enabled and period is not None:
                        for i in range(batch_size - 1):
                            base_pred[i, period - i:] = pred_after_adapt[i, period - i:]
                    pred = base_pred
                self.time_stats['final_prediction'] += time.time() - final_start
                self.time_counts['final_prediction'] += 1

                # 更新 memory buffer
                buffer_start = time.time()
                self.update_memory_buffer(ground_truth, base_pred)
                self.time_stats['buffer_update'] += time.time() - buffer_start
                self.time_counts['buffer_update'] += 1

                if self.save_csv:
                    self._save_batch_predictions_to_csv(original_pred, pred, ground_truth, batch_idx)
                
                metric_start = time.time()
                mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                self.time_stats['metric_computation'] += time.time() - metric_start
                self.time_counts['metric_computation'] += 1

                self.mse_all.append(mse)
                self.mae_all.append(mae)

                batch_start = batch_end
                batch_idx += 1
        
        self.time_stats['total_time'] = time.time() - total_start_time
        
        assert self.cur_step == len(self.test_data) - self.cfg.DATA.PRED_LEN - 1
        
        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        assert len(self.mse_all) == len(self.test_loader.dataset)
        
        if self.save_csv:
            self._export_predictions_to_csv()
        
        if self.save_paas_csv:
            self._export_paas_info_to_csv()
    
        tta_method = self.cfg.TTA.LOG_NAME
        if tta_method is None:
            tta_method = "COSA-FULL"
        self._print_combined_results()
        save_tta_results(
            tta_method=tta_method,
            seed=self.cfg.SEED,
            model_name=self.cfg.MODEL.NAME,
            dataset_name=self.cfg.DATA.NAME,
            pred_len=self.cfg.DATA.PRED_LEN,
            mse_after_tta=self.mse_all.mean(),
            mae_after_tta=self.mae_all.mean(),
        )
    @torch.enable_grad()
    def adapt_simple_eved(self):
        """
        对 EVED 数据集：按每个 CSV 独立进行 SIMPLE 适配与评估。
        使用与 adapt_simple 一致的“历史 + 部分可观测”更新方式。
        """
        from torch.utils.data import DataLoader, Subset

        ds = self.test_loader.dataset
        num_csv = ds.get_num_test_csvs()

        # 全局统计重置
        self.mse_all = []
        self.mae_all = []
        self.n_adapt = 0
        self.sample_history.clear()
        self.current_time_idx = 0
        self.time_stats = defaultdict(float)
        self.time_counts = defaultdict(int)

        total_start_time = time.time()

        for csv_idx in range(num_csv):
            indices = ds.get_test_windows_for_csv(csv_idx)
            if not indices:
                continue

            sub_dataset = Subset(ds, indices)
            sub_loader = DataLoader(sub_dataset, batch_size=len(sub_dataset), shuffle=False)

            # 每个 CSV 内部的状态（时间步、buffer、历史输入）重置
            self.cur_step = self.cfg.DATA.SEQ_LEN - 2
            batch_start = 0
            batch_end = 0
            batch_idx = 0

            self.sample_history.clear()
            self.current_time_idx = 0
            self.adapters_enabled = False
            self.step_count = 0
            self.loss_history.clear()
            self.current_lr = getattr(self.cfg.TTA.SOLVER, 'BASE_LR', 0.001)

            # 历史 batch 记录也应按 CSV 重置
            self.pred_step_end_dict = {}
            self.inputs_dict = {}

            self.switch_model_to_eval()

            for _, inputs in enumerate(sub_loader):
                enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)

                while batch_end < len(enc_window_all):
                    calculated_batch_size = None
                    period = None
                    dominant_freq = None

                    if self.paas_enabled:
                        period_start = time.time()
                        enc_window_first = enc_window_all[batch_start]
                        period, calculated_batch_size, dominant_freq = self._calculate_period_and_batch_size(enc_window_first)
                        batch_size = calculated_batch_size
                        self.time_stats['paas_period_calculation'] += time.time() - period_start
                        self.time_counts['paas_period_calculation'] += 1
                    else:
                        batch_size = getattr(self.cfg.TTA.SIMPLE, 'BATCH_SIZE', 64)
                        calculated_batch_size = batch_size
                        period = batch_size - 1

                    batch_end = batch_start + batch_size
                    if batch_end > len(enc_window_all):
                        batch_end = len(enc_window_all)
                        batch_size = batch_end - batch_start

                    if self.save_paas_csv:
                        self._save_paas_batch_info(
                            batch_idx=batch_idx,
                            batch_start=batch_start,
                            calculated_batch_size=calculated_batch_size,
                            actual_batch_size=batch_size,
                            period=period,
                            fft_dominant_freq=dominant_freq
                        )

                    self.cur_step += batch_size

                    batch_inputs = (
                        enc_window_all[batch_start:batch_end],
                        enc_window_stamp_all[batch_start:batch_end],
                        dec_window_all[batch_start:batch_end],
                        dec_window_stamp_all[batch_start:batch_end]
                    )

                    # 记录当前 batch 的预测结束时间和输入，供后续 full-ground-truth 更新使用
                    self.pred_step_end_dict[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN
                    self.inputs_dict[batch_idx] = batch_inputs

                    # 基础预测（仅用于评估 / buffer）
                    pred_start = time.time()
                    base_pred, ground_truth = forecast(self.cfg, batch_inputs, self.model, self.norm_module)
                    original_pred = base_pred.clone()
                    self.time_stats['base_prediction'] += time.time() - pred_start
                    self.time_counts['base_prediction'] += 1

                    # 利用已经完全可观测的历史 batch 做完整监督更新
                    self._adapt_with_full_ground_truth_if_available_simple()

                    # 构造当前 batch 的 context
                    context_start = time.time()
                    context_data = self._get_individual_context_for_batch(batch_size, batch_idx)
                    self.time_stats['context_generation'] += time.time() - context_start
                    self.time_counts['context_generation'] += 1

                    # 使用当前 batch 的“部分可观测数据”（前 period 步）做更新
                    if self.adapters_enabled and period is not None and period > 0:
                        adapt_start = time.time()
                        self._adapt_with_partial_ground_truth_simple(
                            batch_inputs, period, batch_size, batch_idx, context_data
                        )
                        self.time_stats['total_adaptation'] += time.time() - adapt_start
                        self.time_counts['total_adaptation'] += 1

                    # 适配后再生成最终预测（通过 output_adapter）
                    final_start = time.time()
                    with torch.no_grad():
                        pred_after_adapt = self.output_adapter(base_pred, context_data)
                        if self.paas_enabled and period is not None:
                            for i in range(batch_size - 1):
                                base_pred[i, period - i:] = pred_after_adapt[i, period - i:]
                        pred = base_pred
                    self.time_stats['final_prediction'] += time.time() - final_start
                    self.time_counts['final_prediction'] += 1

                    # 更新 memory buffer（仍用基础预测 + 真值）
                    buffer_start = time.time()
                    self.update_memory_buffer(ground_truth, base_pred)
                    self.time_stats['buffer_update'] += time.time() - buffer_start
                    self.time_counts['buffer_update'] += 1

                    if self.save_csv:
                        self._save_batch_predictions_to_csv(original_pred, pred, ground_truth, batch_idx)

                    metric_start = time.time()
                    mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                    mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                    self.time_stats['metric_computation'] += time.time() - metric_start
                    self.time_counts['metric_computation'] += 1

                    self.mse_all.append(mse)
                    self.mae_all.append(mae)

                    batch_start = batch_end
                    batch_idx += 1

            # 当前 CSV 完成后可选择重置模型/优化器，以免对下一个 CSV 产生影响
            if getattr(self.cfg.TTA, "RESET", False):
                self.reset()
            self.switch_model_to_eval()

        self.time_stats['total_time'] = time.time() - total_start_time

        if self.mse_all:
            self.mse_all = np.concatenate(self.mse_all)
            self.mae_all = np.concatenate(self.mae_all)

            if self.save_csv:
                self._export_predictions_to_csv()
            if self.save_paas_csv:
                self._export_paas_info_to_csv()

            self._print_combined_results()
        else:
            print("No valid test windows for EVED dataset in SimpleAdapter.")

    def _print_combined_results(self):
        total_params = 0
        for _, param in self.named_parameters():
            if param.requires_grad:
                total_params += int(param.numel())
        
        avg_times = {}
        for key in self.time_stats:
            if key != 'total_time':
                count = self.time_counts[key] if self.time_counts[key] > 0 else 1
                avg_times[key] = self.time_stats[key] / count
        
        adapter_total = (avg_times.get('context_generation', 0) + 
                        avg_times.get('adapter_forward', 0) + 
                        avg_times.get('final_prediction', 0))
        
        time_statistics = {
            "adapter_operations": {},
            "adaptation_training": {},
            "other_operations": {},
            "overall_stats": {}
        }
        
        time_statistics["adapter_operations"] = {
            "context_generation_ms": round(avg_times.get('context_generation', 0) * 1000, 3),
            "adapter_forward_pass_ms": round(avg_times.get('adapter_forward', 0) * 1000, 3),
            "final_prediction_ms": round(avg_times.get('final_prediction', 0) * 1000, 3),
            "total_adapter_operation_ms": round(adapter_total * 1000, 3)
        }
        
        time_statistics["adaptation_training"] = {
            "loss_computation_ms": round(avg_times.get('loss_computation', 0) * 1000, 3),
            "backward_update_ms": round(avg_times.get('backward_update', 0) * 1000, 3),
            "per_adaptation_step_ms": round(avg_times.get('per_adaptation_step', 0) * 1000, 3),
            "total_per_batch_ms": round(avg_times.get('total_adaptation', 0) * 1000, 3),
            "adaptation_steps": self.adapt_steps
        }
        
        other_ops = {
            "base_model_prediction_ms": round(avg_times.get('base_prediction', 0) * 1000, 3),
            "buffer_update_ms": round(avg_times.get('buffer_update', 0) * 1000, 3),
            "metric_computation_ms": round(avg_times.get('metric_computation', 0) * 1000, 3)
        }
        if self.paas_enabled:
            other_ops["paas_period_calculation_ms"] = round(avg_times.get('paas_period_calculation', 0) * 1000, 3)
        time_statistics["other_operations"] = other_ops
        
        adaptation_count = max(self.time_counts.get('total_adaptation', 1), 1)
        
        time_statistics["overall_stats"] = {
            "total_time_seconds": round(self.time_stats['total_time'], 2),
            "total_adaptations": int(self.n_adapt),
            "avg_time_per_adaptation_ms": round(self.time_stats.get('total_adaptation', 0) / adaptation_count * 1000, 3),
            "throughput_samples_per_sec": round(len(self.test_loader.dataset) / self.time_stats['total_time'], 1)
        }
        
        combined_results = {
            "model": "SimpleAdapter",
            "time_statistics": time_statistics,
            "final_results": {
                "adaptation_count": int(self.n_adapt),
                "test_mse": float(self.mse_all.mean()),
                "test_mae": float(self.mae_all.mean())
            },
            "parameters": {
                "total_params": total_params
            }
        }
        
        # print(json.dumps(combined_results, indent=2))

        if len(self.mse_all) > 0:
            print('After TSF-TTA of TAFAS on EVED (per-CSV)')
            print(f'Number of adaptations: {self.n_adapt}')
            print(f'Test MSE: {self.mse_all.mean():.4f}, Test MAE: {self.mae_all.mean():.4f}')
            print()
        else:
            print('No valid test windows for EVED dataset.')

    def adapt(self):
        # 根据数据集类型选择不同的 TTA 策略（对齐 tafas.Adapter 的接口）
        if getattr(self, "is_eved_like", False):
            self.adapt_simple_eved()
        else:
            self.adapt_simple()


def build_adapter(cfg, model, norm_module=None):
    adapter = SimpleAdapter(cfg, model, norm_module)
    return adapter