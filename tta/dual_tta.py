# adapted from https://github.com/DequanWang/tent/blob/master/tent.py

from typing import List
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.optimizer import get_optimizer
from models.forecast import forecast
from datasets.loader import get_test_dataloader
from utils.misc import prepare_inputs
from config import get_norm_method
from tta.utils import save_tta_results
import math

class tafas_GCM(nn.Module):
    def __init__(self, window_len, n_var=1, hidden_dim=64, gating_init=0.01, var_wise=True):
        super(tafas_GCM, self).__init__()
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

class petsa_GCM(nn.Module):
    def __init__(self, window_len, n_var=1, hidden_dim=64, gating_init=0.01, var_wise=True, low_rank=16):
        super(petsa_GCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        
        self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))
        self.low_rank = low_rank

        self.lora_A = nn.Parameter(torch.Tensor(window_len, self.low_rank))
        self.lora_B = nn.Parameter(torch.Tensor(self.low_rank, window_len, n_var))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        
        weight = torch.einsum('ik,kjl->ijl', self.lora_A, self.lora_B)
        if self.var_wise:
            x_1 = torch.tanh(self.gating * x)
            new_x =  (torch.einsum('biv,iov->bov', x_1,  weight) + self.bias)
        else:
            x_1 = torch.tanh(self.gating * x)
            new_x =  (torch.einsum('biv,io->bov', x_1,  weight) + self.bias)

        x = x + new_x

        return x

class IdentityAdapter(nn.Module):
    def forward(self, x):
        return x

class CalibrationContainer(nn.Module):
    def __init__(self, input_model: nn.Module, output_model: nn.Module):
        super(CalibrationContainer, self).__init__()
        self.in_cali = input_model
        self.out_cali = output_model
        
    def input_calibration(self, inputs):
        enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
        
        if self.in_cali is not None:
            enc_window = self.in_cali(enc_window)
            
        return enc_window, enc_window_stamp, dec_window, dec_window_stamp

    def output_calibration(self, outputs):
        if self.out_cali is not None:
            return self.out_cali(outputs)
        return outputs

def build_calibration_module(cfg) -> CalibrationContainer:
    if not cfg.TTA.DUAL.CALI_MODULE:
        return None

    seq_len = cfg.DATA.SEQ_LEN
    pred_len = cfg.DATA.PRED_LEN
    n_var = cfg.DATA.N_VAR
    
    # 获取 TAFAS 相关参数
    hidden_dim = cfg.TTA.DUAL.HIDDEN_DIM
    gating_init = cfg.TTA.DUAL.GATING_INIT
    var_wise = cfg.TTA.DUAL.GCM_VAR_WISE
    low_rank = cfg.TTA.DUAL.PETSA_LOWRANK
    
    model_type = getattr(cfg.TTA.DUAL, 'CALI_NAME', 'tafas_GCM')

    if model_type == 'tafas_GCM':
        if cfg.MODEL.NAME == 'PatchTST':
            # PatchTST 通常视作单变量处理
            in_model = tafas_GCM(seq_len, 1, hidden_dim, gating_init, var_wise)
            out_model = tafas_GCM(pred_len, 1, hidden_dim, gating_init, var_wise)
        else:
            in_model = tafas_GCM(seq_len, n_var, hidden_dim, gating_init, var_wise)
            out_model = tafas_GCM(pred_len, n_var, hidden_dim, gating_init, var_wise)
    elif model_type == 'petsa_GCM':
        if cfg.MODEL.NAME == 'PatchTST':
            in_model = petsa_GCM(seq_len, 1, hidden_dim, gating_init, var_wise, low_rank)
            out_model = petsa_GCM(pred_len, 1, hidden_dim, gating_init, var_wise, low_rank)
        else:
            in_model = petsa_GCM(seq_len, n_var, hidden_dim, gating_init, var_wise, low_rank)
            out_model = petsa_GCM(pred_len, n_var, hidden_dim, gating_init, var_wise, low_rank)
    elif model_type == 'MLP':
        pass
    else:
        raise ValueError(f"Unknown adapter type: {model_type}")

    return CalibrationContainer(in_model, out_model)


class StandardMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, ground_truth):
        return F.mse_loss(pred, ground_truth)

class CorrCoefLoss(nn.Module):

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 扁平化为 1D 向量
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

class PETSALoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.person_cor = CorrCoefLoss()
        
    def forward(self, pred, ground_truth):
        loss_feq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(ground_truth, dim=1)).abs().mean() 
        loss_tmp = torch.nn.functional.huber_loss(pred, ground_truth, delta=0.5)
        loss =  loss_tmp + loss_feq * self.alpha
        coss = self.person_cor(pred, ground_truth)
        sf_pred = torch.nn.functional.softmax(pred - pred.mean(dim=1, keepdim=True))
        sf_gt   = torch.nn.functional.softmax((ground_truth - ground_truth.mean(dim=1, keepdim=True)))
        loss_var = torch.nn.functional.kl_div(sf_pred, sf_gt).mean()
        loss_mean = F.l1_loss(pred.mean(dim=1, keepdim=True), 
                                ground_truth.mean(dim=1, keepdim=True))
        loss +=  ((coss + loss_var + loss_mean))
        return loss

def build_loss_fn(cfg) -> nn.Module:
    loss_name = getattr(cfg.TTA.DUAL, 'LOSS_NAME', 'MSE')
    if loss_name == 'MSE':
        return StandardMSELoss()
    elif loss_name == 'PETSA': 
        alpha = getattr(cfg.TTA.DUAL, 'PETSA_LOSS_ALPHA', 0.1)
        return PETSALoss(alpha=alpha)
    else:
        raise ValueError(f"Unknown Loss type: {loss_name}")

class Adapter(nn.Module):
    def __init__(self, cfg, model: nn.Module, norm_module=None):
        super(Adapter, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.model = model
        self.norm_method = get_norm_method(cfg)
        self.norm_module = norm_module
        self.test_loader = get_test_dataloader(cfg)
        self.test_data = self.test_loader.dataset.test

        self.cali = build_calibration_module(cfg).cuda()
        
        self._freeze_all_model_params()
        self.named_modules_to_adapt = self._get_named_modules_to_adapt()
        self._unfreeze_modules_to_adapt()
        self.named_params_to_adapt = self._get_named_params_to_adapt()
        
        self.optimizer = get_optimizer(self.named_params_to_adapt.values(), cfg.TTA)
        
        self.model_state, self.optimizer_state = self._copy_model_and_optimizer()
        self.cali_state = self._copy_cali() if self.cali is not None else None
        self.loss_fn = build_loss_fn(cfg)
        
        if hasattr(self.test_loader.dataset, "get_test_num_windows"):
            test_num_windows = self.test_loader.dataset.get_test_num_windows()
            batch_size = test_num_windows
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
        
    def forward(self, enc_window, enc_window_stamp, dec_window, dec_window_stamp):
        raise NotImplementedError
    
    def reset(self):
        self._load_model_and_optimizer()
    
    def _copy_model_and_optimizer(self):
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_state, optimizer_state

    def _copy_cali(self):
        return deepcopy(self.cali.state_dict())

    def _load_cali(self):
        if self.cali_state is not None:
            self.cali.load_state_dict(deepcopy(self.cali_state), strict=True)

    def _load_model_and_optimizer(self):
        self.model.load_state_dict(deepcopy(self.model_state), strict=True)
        self.optimizer.load_state_dict(deepcopy(self.optimizer_state))
        if self.cali is not None:
            self._load_cali()
    
    def _get_all_models(self):
        models = [self.model]
        if self.norm_module is not None:
            models.append(self.norm_module)
        if self.cali is not None:
            models.append(self.cali)
        return models

    def _freeze_all_model_params(self):
        for model in self._get_all_models():
            for param in model.parameters():
                param.requires_grad_(False)
    
    def _get_named_modules(self):
        named_modules = []
        for model in self._get_all_models():
            named_modules += list(model.named_modules())
        return named_modules
    
    def _get_named_modules_to_adapt(self) -> List[str]:
        named_modules = self._get_named_modules()
        if self.cfg.TTA.MODULE_NAMES_TO_ADAPT == 'all':
            return named_modules
        
        named_modules_to_adapt = []
        for module_name in self.cfg.TTA.MODULE_NAMES_TO_ADAPT.split(','):
            exact_match = '(exact)' in module_name
            module_name = module_name.replace('(exact)', '')
            if exact_match:
                named_modules_to_adapt += [(name, module) for name, module in named_modules if name == module_name]
            else:
                named_modules_to_adapt += [(name, module) for name, module in named_modules if module_name in name]

        assert len(named_modules_to_adapt) > 0
        return named_modules_to_adapt
    
    def _unfreeze_modules_to_adapt(self):
        for _, module in self.named_modules_to_adapt:
            module.requires_grad_(True)
    
    def _get_named_params_to_adapt(self):
        named_params_to_adapt = {}
        for model in self._get_all_models():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    named_params_to_adapt[name] = param
        return named_params_to_adapt
    
    def switch_model_to_train(self):
        for model in self._get_all_models():
            model.train()
    
    def switch_model_to_eval(self):
        for model in self._get_all_models():
            model.eval()
    
    @torch.enable_grad()
    def _adapt(self):
        is_last = False
        test_len = len(self.test_loader.dataset)
            
        self.switch_model_to_eval()
        for idx, inputs in enumerate(self.test_loader):
            enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
            batch_start = 0
            batch_end = 0
            batch_idx = 0
            self.cur_step = self.cfg.DATA.SEQ_LEN - 2
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
                
                if self.cfg.TTA.DUAL.ADJUST_PRED:
                    pred, ground_truth = self._adjust_prediction(pred, inputs, batch_size, period)
                
                mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                self.mse_all.append(mse)
                self.mae_all.append(mae)
                
                batch_start = batch_end
                batch_idx += 1

            assert self.cur_step == len(self.test_data) - self.cfg.DATA.PRED_LEN - 1

        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        assert len(self.mse_all) == len(self.test_loader.dataset)
        
        print('After TSF-TTA of TAFAS')
        print(f'Number of adaptations: {self.n_adapt}')
        print(f'Test MSE: {self.mse_all.mean():.4f}, Test MAE: {self.mae_all.mean():.4f}')
        print()

        save_tta_results(
            tta_method='Dual-tta',
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
        period *= self.cfg.TTA.DUAL.PERIOD_N
        batch_size = period + 1
        return period, batch_size

    def _adapt_with_full_ground_truth_if_available(self):
        while self.cur_step >= self.pred_step_end_dict[min(self.pred_step_end_dict.keys())]:
            batch_idx_available = min(self.pred_step_end_dict.keys())
            inputs_history = self.inputs_dict.pop(batch_idx_available)
            for _ in range(self.cfg.TTA.DUAL.STEPS):
                self.n_adapt += 1
                
                self.switch_model_to_train()

                if self.cali is not None:
                    inputs_history = self.cali.input_calibration(inputs_history)
                pred, ground_truth = forecast(self.cfg, inputs_history, self.model, self.norm_module)
                
                if self.cali is not None:
                    pred = self.cali.output_calibration(pred)
                    
                loss = self.loss_fn(pred, ground_truth) 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.switch_model_to_eval()
            
            self.pred_step_end_dict.pop(batch_idx_available)

    def _adapt_with_partial_ground_truth(self, inputs, period, batch_size, batch_idx):
        for _ in range(self.cfg.TTA.DUAL.STEPS):
            self.n_adapt += 1
            
            if self.cali is not None:
                inputs = self.cali.input_calibration(inputs)
            pred, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
        
            if self.cali is not None:
                pred = self.cali.output_calibration(pred)
                
            pred_partial, ground_truth_partial = pred[0][:period], ground_truth[0][:period]
            loss_partial = self.loss_fn(pred_partial, ground_truth_partial) 
            self.optimizer.zero_grad()
            loss_partial.backward()
            self.optimizer.step()
        return pred, ground_truth

    @torch.no_grad()
    def _adjust_prediction(self, pred, inputs, batch_size, period):
        if self.cali is not None:
            inputs = self.cali.input_calibration(inputs)
        pred_after_adapt, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
        if self.cali is not None:
            pred_after_adapt = self.cali.output_calibration(pred_after_adapt)
        
        for i in range(batch_size-1):
            pred[i, period-i:] = pred_after_adapt[i, period-i:]
        
        return pred, ground_truth
    
    def adapt(self):
        self._adapt()

def build_adapter(cfg, model, norm_module=None):
    adapter = Adapter(cfg, model, norm_module)
    return adapter


class GCM(nn.Module):
    def __init__(self, window_len, n_var=1, hidden_dim=64, gating_init=0.01, var_wise=True):
        super(GCM, self).__init__()
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
        self.hidden_dim = cfg.TTA.DUAL.HIDDEN_DIM
        self.gating_init = cfg.TTA.DUAL.GATING_INIT
        self.var_wise = cfg.TTA.DUAL.GCM_VAR_WISE
        if cfg.MODEL.NAME == 'PatchTST':
            self.in_cali = GCM(self.seq_len, 1, self.hidden_dim, self.gating_init, self.var_wise)
            self.out_cali = GCM(self.pred_len, 1, self.hidden_dim, self.gating_init, self.var_wise)
        else:
            self.in_cali = GCM(self.seq_len, self.n_var, self.hidden_dim, self.gating_init, self.var_wise)
            self.out_cali = GCM(self.pred_len, self.n_var, self.hidden_dim, self.gating_init, self.var_wise)
        
    def input_calibration(self, inputs):
        enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
        enc_window = self.in_cali(enc_window)
        return enc_window, enc_window_stamp, dec_window, dec_window_stamp

    def output_calibration(self, outputs):
        return self.out_cali(outputs)