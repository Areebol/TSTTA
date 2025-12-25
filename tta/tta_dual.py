import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from typing import List, Optional

from config import get_norm_method
from models.forecast import forecast
from utils.misc import prepare_inputs
from models.optimizer import get_optimizer
from datasets.loader import get_test_dataloader

from tta.loss import *
from tta.tta_dual_utils.GCM import *
from tta.utils import save_tta_results

def get_model_dims(cfg):
    is_patchtst = (cfg.MODEL.NAME == 'PatchTST')
    n_var = 1 if is_patchtst else cfg.DATA.N_VAR
    return cfg.DATA.SEQ_LEN, cfg.DATA.PRED_LEN, n_var

def build_calibration_module(cfg) -> Optional[CalibrationContainer]:
    if not cfg.TTA.DUAL.CALI_MODULE:
        return None
    
    seq_len, pred_len, n_var = get_model_dims(cfg)
    params = {
        'hidden_dim': cfg.TTA.DUAL.HIDDEN_DIM,
        'gating_init': cfg.TTA.DUAL.GATING_INIT,
        'var_wise': cfg.TTA.DUAL.GCM_VAR_WISE
    }
    model_type = getattr(cfg.TTA.DUAL, 'CALI_NAME', 'tafas_GCM')
    
    constructors = {
        'tafas_GCM': tafas_GCM,
        'petsa_GCM': petsa_GCM,
    }
    
    ModelClass = constructors.get(model_type)
    if not ModelClass:
        raise ValueError(f"Unknown adapter type: {model_type}")

    in_model = ModelClass(seq_len, n_var, **params)
    out_model = ModelClass(pred_len, n_var, **params)
    
    return CalibrationContainer(in_model, out_model)

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
    def adapt(self):
        is_last = False
        test_len = len(self.test_loader.dataset)
        
        self.switch_model_to_eval()
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
    
def build_adapter(cfg, model, norm_module=None):
    adapter = Adapter(cfg, model, norm_module)
    return adapter