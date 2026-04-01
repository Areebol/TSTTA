import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os

from copy import deepcopy
from typing import List, Optional
from torch.utils.data import DataLoader, Subset

from config import get_norm_method
from models.forecast import forecast
from utils.misc import prepare_inputs
from datasets.loader import get_test_dataloader, get_tta_train_dataloader, get_domain_shift_dataloader

from tta.loss import *
from tta.tta_dual_utils.GCM import *
from tta.pattern_bank import *
from tta.tta_dual_utils.model_manager import TTAModelManager
from tta.utils import save_tta_results
from device_manager import global_device

# 注意：假设 Freq_Add_Adapter 已经在前面的代码或对应的文件中定义并 import 了


def build_calibration_module(cfg) -> Optional[CalibrationContainer]:
    def get_model_dims(cfg):
        is_patchtst = (cfg.MODEL.NAME == 'PatchTST')
        n_var = cfg.MODEL.c_out if is_patchtst else cfg.DATA.N_VAR
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
        'CoBA_GCM': CoBA_GCM,
        'lowrank-coba-GCM': CoBA_low_rank_GCM,
        'coba-online-only': CoBA_online_only,
        'identity': IdentityAdapter,
        'CoBA-FreqDomain-GCM': CoBA_FreqDomain_GCM,
        'CoBA-low-rank-FreqAdapter': CoBA_low_rank_FreqAdapter,
        'CoBA_FreqDomain_ElementWise_GCM': CoBA_FreqDomain_ElementWise_GCM,
        'RoCoBA_FreqDomain_GCM': RoCoBA_FreqDomain_GCM,
        'EnCoBA_FreqDomain_GCM': EnCoBA_FreqDomain_GCM,
        'RoCoBA_FreqDomain_Norm': RoCoBA_FreqDomain_Norm,
        'CoBA_Freq_Adapter': CoBA_Freq_Adapter,
        'Freq_Add_Adapter': Freq_Add_Adapter, 
        'CoBA_TF_Adapter': CoBA_TF_Adapter,
        'PKA_GCM': PKA_GCM,
    }
    
    if model_type == 'CoBA_GCM':
        coba_params = {
            'n_bases': cfg.TTA.DUAL.GCM_N_BASES,
        }
        params.update(coba_params)
    elif model_type in ['lowrank-coba-GCM', 'coba-online-only', 'CoBA-FreqDomain-GCM', 'CoBA-low-rank-FreqAdapter', 'CoBA_FreqDomain_ElementWise_GCM', 'RoCoBA_FreqDomain_GCM', 'EnCoBA_FreqDomain_GCM', 'RoCoBA_FreqDomain_Norm', 'CoBA_Freq_Adapter', 'Freq_Add_Adapter', 'CoBA_TF_Adapter', 'PKA_GCM']:
        coba_params = {
            'n_bases': cfg.TTA.DUAL.GCM_N_BASES,
            'low_ranks': getattr(cfg.TTA.DUAL, 'LOWRANK_RANKS', None),
            'query_type': getattr(cfg.TTA.DUAL, 'QUERY_TYPE', 'freq-base-CI'),
            'n_static': cfg.TTA.DUAL.GCM_N_BASES,
        }
        params.update(coba_params)
    elif model_type in ['RoCoBA_FreqDomain_Norm', 'CoBA_Freq_Adapter', 'Freq_Add_Adapter', 'CoBA_TF_Adapter', 'PKA_GCM']:
        coba_params = {
            'seq_len': cfg.DATA.SEQ_LEN,
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
        in_model = tafas_GCM(seq_len, n_var, **params)
    if cfg.TTA.DUAL.CALI_OUTPUT_ENABLE:
        out_model = ModelClass(pred_len, n_var, **params)

    print(params)
    return CalibrationContainer(in_model, out_model)

def build_loss_fn(cfg) -> nn.Module:
    # [说明] 如果使用 Freq_Add_Adapter，配置中最好配回 'MSE' 
    # 因为它的正则项是直接调用的，主任务 Loss 返回标准的 MSE 即可。
    loss_name = getattr(cfg.TTA.DUAL, 'LOSS_NAME', 'MSE')
    if loss_name == 'MSE':
        return StandardMSELoss()
    elif loss_name == 'PETSA': 
        alpha = getattr(cfg.TTA.DUAL, 'PETSA_LOSS_ALPHA', 0.1)
        return PETSALoss(alpha=alpha)
    elif loss_name == "CoBA_Loss":
        return CoBA_Loss(lambda_ortho=cfg.TTA.DUAL.LAMBDA_ORTHO)
    elif loss_name == "LOWRANK-COBA":
        return LowRankCoBALoss(lambda_ortho=cfg.TTA.DUAL.LAMBDA_ORTHO)
    elif loss_name == "Freq-LowRank-CoBA":
        return FreqLowRankCoBALoss(lambda_ortho=cfg.TTA.DUAL.LAMBDA_ORTHO)
    elif loss_name == "Freq-EW-CoBALoss":
        return FreqElementWiseCoBALoss(lambda_ortho=cfg.TTA.DUAL.LAMBDA_ORTHO)
    elif loss_name == "Freq-EW-SPLoss":
        return FreqElementWiseSPLoss(lambda_ortho=cfg.TTA.DUAL.LAMBDA_ORTHO)
    elif loss_name == "DiversityCoBALoss":
        return DiversityCoBALoss(lambda_base=cfg.TTA.DUAL.LAMBDA_BASE, lambda_key=cfg.TTA.DUAL.LAMBDA_KEY, margin=cfg.TTA.DUAL.DIVERSITY_MARGIN)
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
        self.cali = build_calibration_module(cfg).to(global_device)
        self.loss_fn = build_loss_fn(cfg)

        self.manager = TTAModelManager(model, norm_module, self.cali)
        trainable_params = self.manager.configure_adaptation(cfg.TTA.MODULE_NAMES_TO_ADAPT)
        self.manager.snapshot()
        self.optimizer = get_optimizer(trainable_params, cfg.TTA)
        self.optimizer_state = deepcopy(self.optimizer.state_dict())
        
        if cfg.TTA.DOMAIN_SHIFT:
            self.test_loader = get_domain_shift_dataloader(cfg)
        else:
            self.test_loader = get_test_dataloader(cfg)
        self.test_data = self.test_loader.dataset.test
        batch_size = len(self.test_loader.dataset)
        if cfg.TTA.DOMAIN_SHIFT:
            self.test_loader = get_domain_shift_dataloader(cfg, batch_size=batch_size)
        else:
            self.test_loader = get_test_dataloader(cfg, batch_size=batch_size)
        self.tta_train_loader = get_tta_train_dataloader(cfg, batch_size=cfg.TRAIN.BATCH_SIZE)
        self.tta_train_data = self.tta_train_loader.dataset.train
        
        self.cur_step = cfg.DATA.SEQ_LEN - 2
        self.pred_step_end_dict = {}
        self.inputs_dict = {}
        self.n_adapt = 0

        cali_name = getattr(self.cfg.TTA.DUAL, 'CALI_NAME', 'unknown')

        parts = []
        # 增加对 Freq_Add_Adapter 的命名支持
        if self.cfg.TTA.DUAL.CALI_NAME in ['CoBA_Freq_Adapter', 'Freq_Add_Adapter', 'CoBA_TF_Adapter', 'PKA_GCM']:
            if self.cfg.TTA.DUAL.CALI_NAME == 'Freq_Add_Adapter':
                prefix = "freq-add"
            elif self.cfg.TTA.DUAL.CALI_NAME == 'CoBA_TF_Adapter':
                prefix = "coba-tf"
            else:
                prefix = "coba-feq"

            if not self.cfg.TTA.DUAL.COBA_ONLINE_ENABLED:
                parts.append(f"{prefix}-adapter-offline")
                parts.append(f'{self.cfg.TTA.SOLVER.BASE_LR:.5f}')
                parts.append(f'n-{self.cfg.TTA.DUAL.GCM_N_BASES:03d}')
                # parts.append(f'lambda-bud-{self.cfg.TTA.DUAL.LAMBDA_BUDGET:.3f}')
                parts.append(f'lambda-ortho-{self.cfg.TTA.DUAL.LAMBDA_ORTHO:.3f}')
            else:
                parts.append(f"{prefix}-adapter-online")
                parts.append(f'offlinelr-{self.cfg.TTA.SOLVER.BASE_LR:.5f}')
                parts.append(f'onlinelr-{self.cfg.TTA.DUAL.COBA_ONLINE_LR:.5f}')
                parts.append(f'n-{self.cfg.TTA.DUAL.GCM_N_BASES:03d}')
                parts.append(f'lambda-ortho-{self.cfg.TTA.DUAL.LAMBDA_ORTHO:.3f}')
                
        elif self.cfg.TTA.DUAL.CALI_NAME == 'RoCoBA_FreqDomain_Norm' and not self.cfg.TTA.DUAL.COBA_ONLINE_ENABLED:
            parts.append("ro-coba-feq-norm-offline")
            parts.append(f'{self.cfg.TTA.SOLVER.BASE_LR:.5f}')
        else:
            parts.append(f"{cali_name}-offline")
            parts.append(f'{self.cfg.TTA.SOLVER.BASE_LR:.5f}')

        self.save_name = "-".join(parts)
        self.mse_all = []
        self.mae_all = []
        self.mse_per_var_all = []

        ds = self.test_loader.dataset
        self.is_eved_like = (
            hasattr(ds, "get_num_test_csvs")
            and hasattr(ds, "get_test_csv_window_range")
            and hasattr(ds, "get_test_windows_for_csv")
        )

        # 判断条件增加 Freq_Add_Adapter
        if isinstance(self.cali.out_cali, (CoBA_GCM, CoBA_low_rank_GCM, Auxiliary_GCM, CoBA_low_rank_FreqAdapter, CoBA_FreqDomain_GCM, CoBA_FreqDomain_ElementWise_GCM, RoCoBA_FreqDomain_GCM, EnCoBA_FreqDomain_GCM, RoCoBA_FreqDomain_Norm, CoBA_Freq_Adapter, Freq_Add_Adapter, CoBA_TF_Adapter, PKA_GCM)):
            self._pretrain_adapter()
            self.cali.out_cali.online_mode = self.cfg.TTA.DUAL.COBA_ONLINE_ENABLED 
            
            print("Adapter pre-training completed.")
            optim_params = self.cali.out_cali.get_optim_params()
            self.optimizer = torch.optim.Adam(
                optim_params,
                lr=self.cfg.TTA.DUAL.COBA_ONLINE_LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY
            ) 
        elif isinstance(self.cali.out_cali, CoBA_online_only):
            self.cali.out_cali.online_mode = self.cfg.TTA.DUAL.COBA_ONLINE_ENABLED
            print("Adapter set to online-only mode.")
            optim_params = self.cali.out_cali.get_optim_params()
            self.optimizer = torch.optim.Adam(
                optim_params,
                lr=self.cfg.TTA.DUAL.COBA_ONLINE_LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY
            ) 
        else:
            print("No adapter pre-training needed.")

        # if isinstance(self.cali.out_cali, (CoBA_GCM, PKA_GCM, CoBA_TF_Adapter)):
        #     self._pretrain_adapter()
        #     self.cali.out_cali.online_mode = self.cfg.TTA.PKA.COBA_ONLINE_ENABLED # Enable online mode after pre-training
        
        # else:
        #     print("No adapter pre-training needed.")

    def _freeze_all(self):
        self.manager._freeze_all()

    def _pretrain_adapter(self):
        self._switch_model_to_train()
        total_steps = self.cfg.TTA.DUAL.PRETRAIN_EPOCHS * len(self.tta_train_loader)

        # 提取当前超参数，可以配置在 yaml 里，这里做防御性获取
        lam_ortho = getattr(self.cfg.TTA.DUAL, 'LAMBDA_ORTHO', 0.01)
        lam_budget = getattr(self.cfg.TTA.DUAL, 'LAMBDA_BUDGET', 1.0)
        budget_gamma = getattr(self.cfg.TTA.DUAL, 'BUDGET_GAMMA', 0.05)

        for epoch in range(self.cfg.TTA.DUAL.PRETRAIN_EPOCHS):
            for step, inputs in enumerate(self.tta_train_loader):
                enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
                inputs = (enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all)
                
                if self.cali.input_calibration is not None:
                    inputs = self.cali.input_calibration(inputs)
                    
                pred, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
                
                if self.cali.output_calibration is not None:
                    if isinstance(self.cali.out_cali, (RoCoBA_FreqDomain_Norm, CoBA_Freq_Adapter, Freq_Add_Adapter, CoBA_TF_Adapter, PKA_GCM)):
                        assert enc_window_all is not None, "enc_window_all should not be None for FreqDomain_Norm"
                        pred = self.cali.output_calibration(pred, enc_window_all)
                    else:
                        pred = self.cali.output_calibration(pred)
                        
                # [修改 4] Freq_Add_Adapter 的专属复合 Loss 计算逻辑
                if isinstance(self.cali.out_cali, Freq_Add_Adapter):
                    task_loss = self.loss_fn(pred, ground_truth) # 建议使用标准 MSELoss
                    ortho_loss = self.cali.out_cali.get_orthogonal_loss()
                    budget_loss = self.cali.out_cali.get_budget_loss(gamma=budget_gamma)
                    loss = task_loss + lam_ortho * ortho_loss + lam_budget * budget_loss
                
                # 原有的其他方法 Loss
                elif isinstance(self.loss_fn, CoBA_Loss):
                    # loss = self.loss_fn(pred, ground_truth, bases=self.cali.out_cali.bases)
                    loss = self.loss_fn(pred, ground_truth, bases=self.cali.out_cali.static_keys)
                elif isinstance(self.loss_fn, (FreqElementWiseCoBALoss, FreqElementWiseSPLoss)):
                    loss = self.loss_fn(pred, ground_truth, bases_r=self.cali.out_cali.bases_r, bases_i=self.cali.out_cali.bases_i)
                elif isinstance(self.loss_fn, DiversityCoBALoss):
                    loss = self.loss_fn(pred, ground_truth, bases_r=self.cali.out_cali.bases_r, bases_i=self.cali.out_cali.bases_i, keys=self.cali.out_cali.codebook_keys)
                else:
                    loss = self.loss_fn(pred, ground_truth) 
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if hasattr(self.cali.out_cali, 'analyzer'):
                    self.cali.out_cali.analyzer.record_batch()
                    
            if hasattr(self.cali.out_cali, 'analyzer'):
                self.cali.out_cali.analyzer.end_epoch()
                
        self._switch_model_to_eval()
        # self._freeze_all()

        # 生成可视化
        if self.cfg.TTA.VISUALIZE:
            if self.cali.output_calibration is not None and isinstance(self.cali.out_cali, (CoBA_Freq_Adapter, Freq_Add_Adapter, CoBA_TF_Adapter, PKA_GCM)):
                try:
                    save_directory = f"./vis_results/{self.save_name}"
                    print(f"\n[*] Generating individual channel visualizations for {self.cali.out_cali.__class__.__name__} (N={self.cali.out_cali.n_static})...")
                    visualize_knowledge_vectors(
                        adapter=self.cali.out_cali, 
                        save_dir=save_directory,
                        max_vars=10 
                    )
                except Exception as e:
                    print(f"[!] Warning: Failed to visualize knowledge vectors: {e}")

    def _reset(self):
        self.manager.reset()
        self.optimizer.load_state_dict(deepcopy(self.optimizer_state))

    def _switch_model_to_train(self):
        self.manager.train()
    
    def _switch_model_to_eval(self):
        self.manager.eval()   
    
    def _calculate_period_and_batch_size(self, enc_window_first):
        fft_result = torch.fft.rfft(enc_window_first - enc_window_first.mean(dim=0), dim=0)
        # amplitude = torch.abs(fft_result)
        # amplitude = torch.sqrt(fft_result.real.pow(2) + fft_result.imag.pow(2))
        amplitude = stable_complex_abs(fft_result)
        power = torch.mean(amplitude ** 2, dim=0)
        try:
            period = enc_window_first.shape[0] // torch.argmax(amplitude[:, power.argmax()]).item()
        except:
            period = 24
        period *= self.cfg.TTA.DUAL.PERIOD_N
        batch_size = period + 1
        return period, batch_size
    
    def _adapt_with_full_ground_truth_if_available(self):
        lam_ortho = getattr(self.cfg.TTA.DUAL, 'LAMBDA_ORTHO', 0.05)
        lam_budget = getattr(self.cfg.TTA.DUAL, 'LAMBDA_BUDGET', 1.0)
        budget_gamma = getattr(self.cfg.TTA.DUAL, 'BUDGET_GAMMA', 0.05)
        
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
                    if isinstance(self.cali.out_cali, (RoCoBA_FreqDomain_Norm, CoBA_Freq_Adapter, Freq_Add_Adapter, CoBA_TF_Adapter, PKA_GCM)):
                        enc_history = prepare_inputs(inputs_history)[0]
                        pred = self.cali.output_calibration(pred, enc_history)
                    else:
                        pred = self.cali.output_calibration(pred)
                
                # [修改 5] 在在线学习/全局微调阶段增加复合损失
                if isinstance(self.cali.out_cali, Freq_Add_Adapter):
                    task_loss = self.loss_fn(pred, ground_truth)
                    ortho_loss = self.cali.out_cali.get_orthogonal_loss()
                    budget_loss = self.cali.out_cali.get_budget_loss(gamma=budget_gamma)
                    loss = task_loss + lam_ortho * ortho_loss + lam_budget * budget_loss
                elif isinstance(self.loss_fn, CoBA_Loss):
                    # loss = self.loss_fn(pred, ground_truth, bases=self.cali.out_cali.bases)
                    loss = self.loss_fn(pred, ground_truth, bases=self.cali.out_cali.static_keys)
                elif isinstance(self.loss_fn, LowRankCoBALoss):
                    loss = self.loss_fn(pred, ground_truth, bases_left=self.cali.out_cali.bases_left, bases_right=self.cali.out_cali.bases_right)
                elif isinstance(self.loss_fn, (FreqLowRankCoBALoss)):
                    loss = self.loss_fn(pred, ground_truth, real_left=self.cali.out_cali.bases_left_r, real_right=self.cali.out_cali.bases_right_r, imag_left=self.cali.out_cali.bases_left_i, imag_right=self.cali.out_cali.bases_right_i)
                elif isinstance(self.loss_fn, (FreqElementWiseCoBALoss, FreqElementWiseSPLoss)):
                    loss = self.loss_fn(pred, ground_truth, bases_r=self.cali.out_cali.bases_r, bases_i=self.cali.out_cali.bases_i)
                elif isinstance(self.loss_fn, DiversityCoBALoss):
                    loss = self.loss_fn(pred, ground_truth, bases_r=self.cali.out_cali.bases_r, bases_i=self.cali.out_cali.bases_i, keys=self.cali.out_cali.codebook_keys)
                else:
                    loss = self.loss_fn(pred, ground_truth) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self._switch_model_to_eval()
            
            self.pred_step_end_dict.pop(batch_idx_available)

    def _adapt_with_partial_ground_truth(self, inputs, period, batch_size, batch_idx):
        lam_ortho = getattr(self.cfg.TTA.DUAL, 'LAMBDA_ORTHO', 0.01)
        lam_budget = getattr(self.cfg.TTA.DUAL, 'LAMBDA_BUDGET', 1.0)
        budget_gamma = getattr(self.cfg.TTA.DUAL, 'BUDGET_GAMMA', 0.05)
        
        for _ in range(self.cfg.TTA.DUAL.STEPS):
            self.n_adapt += 1
            
            if self.cali.input_calibration is not None:
                inputs = self.cali.input_calibration(inputs)
            pred, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
        
            if self.cali.output_calibration is not None:
                if isinstance(self.cali.out_cali, (RoCoBA_FreqDomain_Norm, CoBA_Freq_Adapter, Freq_Add_Adapter, CoBA_TF_Adapter, PKA_GCM)):
                    enc_window = prepare_inputs(inputs)[0]
                    # 这一步前向传播会计算并记录 Freq_Add_Adapter 的 relative_energy
                    pred = self.cali.output_calibration(pred, enc_window)
                else:
                    pred = self.cali.output_calibration(pred)
            
            pred_partial, ground_truth_partial = pred[0][:period], ground_truth[0][:period]
            
            # [修改 6] Partial 阶段复用提取的 budget 和 ortho loss
            if isinstance(self.cali.out_cali, Freq_Add_Adapter):
                task_loss = self.loss_fn(pred_partial, ground_truth_partial)
                ortho_loss = self.cali.out_cali.get_orthogonal_loss()
                budget_loss = self.cali.out_cali.get_budget_loss(gamma=budget_gamma)
                loss_partial = task_loss + lam_ortho * ortho_loss + lam_budget * budget_loss
            elif isinstance(self.loss_fn, CoBA_Loss):
                # loss_partial = self.loss_fn(pred_partial, ground_truth_partial, bases=self.cali.out_cali.bases)
                loss_partial = self.loss_fn(pred_partial, ground_truth_partial, bases=self.cali.out_cali.static_keys)
            elif isinstance(self.loss_fn, LowRankCoBALoss):
                loss_partial = self.loss_fn(pred_partial, ground_truth_partial, bases_left=self.cali.out_cali.bases_left, bases_right=self.cali.out_cali.bases_right)
            elif isinstance(self.loss_fn, FreqLowRankCoBALoss):
                loss_partial = self.loss_fn(pred_partial, ground_truth_partial, real_left=self.cali.out_cali.bases_left_r, real_right=self.cali.out_cali.bases_right_r, imag_left=self.cali.out_cali.bases_left_i, imag_right=self.cali.out_cali.bases_right_i)
            elif isinstance(self.loss_fn, (FreqElementWiseCoBALoss, FreqElementWiseSPLoss)):
                loss_partial = self.loss_fn(pred_partial, ground_truth_partial, bases_r=self.cali.out_cali.bases_r, bases_i=self.cali.out_cali.bases_i)
            elif isinstance(self.loss_fn, DiversityCoBALoss):
                loss_partial = self.loss_fn(pred_partial, ground_truth_partial, bases_r=self.cali.out_cali.bases_r, bases_i=self.cali.out_cali.bases_i, keys=self.cali.out_cali.codebook_keys)
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
            if isinstance(self.cali.out_cali, (RoCoBA_FreqDomain_Norm, CoBA_Freq_Adapter, Freq_Add_Adapter, CoBA_TF_Adapter, PKA_GCM)):
                enc_window = prepare_inputs(inputs)[0]
                pred_after_adapt = self.cali.output_calibration(pred_after_adapt, enc_window)
            else:
                pred_after_adapt = self.cali.output_calibration(pred_after_adapt)
        
        for i in range(batch_size-1):
            pred[i, period-i:] = pred_after_adapt[i, period-i:]
        return pred, ground_truth
    
    def _report(self):
        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        self.mse_per_var_all = np.concatenate(self.mse_per_var_all)
        assert len(self.mse_all) == len(self.test_loader.dataset)
        
        dataset_name = self.cfg.DATA.NAME if not self.cfg.TTA.DOMAIN_SHIFT else f"{self.cfg.DATA.NAME}_2_{self.cfg.DATA.DOMAIN_SHIFT_TARGET}"
        save_tta_results(
            tta_method=self.save_name,
            seed=self.cfg.SEED,
            model_name=self.cfg.MODEL.NAME,
            dataset_name=dataset_name,
            pred_len=self.cfg.DATA.PRED_LEN,
            mse_after_tta=self.mse_all.mean(),
            mae_after_tta=self.mae_all.mean(),
            save_dir=self.cfg.RESULT_DIR
        )
        self.model.eval()

        tta_method = 'offline' if not self.cfg.TTA.DUAL.COBA_ONLINE_ENABLED else 'online'
        print(f"Final {tta_method} TTA Results for pred_len: {self.cfg.DATA.PRED_LEN}:")
        print(f"MSE mean: {self.mse_all.mean()}")
        print(f"MSE per channles: {self.mse_per_var_all.mean(axis=0)}")
    
    def adapt(self):
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
            
            if self.cfg.TTA.DUAL.ADJUST_PRED:
                pred, ground_truth = self._adjust_prediction(pred, inputs, batch_size, period)
            
            mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
            mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
            self.mse_all.append(mse)
            self.mae_all.append(mae)

            mse_per_var = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=-2).detach().cpu().numpy()
            self.mse_per_var_all.append(mse_per_var)

            batch_start = batch_end
            batch_idx += 1
                
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

def visualize_knowledge_vectors(adapter, save_dir="./vis_results", max_vars=10):
    """
    可视化 Freq_Add_Adapter (或 CoBA) 中通道的知识向量，并为每个通道单独保存一张图。
    """
    adapter.eval() 
    n_bases = getattr(adapter, 'n_bases', getattr(adapter, 'n_static', 0))
    n_var = getattr(adapter, 'n_var', 1)
    
    plot_vars = min(n_var, max_vars)
    if n_var > max_vars:
        print(f"[*] Warning: n_var ({n_var}) is large. Only plotting the first {max_vars} variables.")

    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for v in range(plot_vars):
            if hasattr(adapter, 'static_keys') and hasattr(adapter, 'static_values'):
                # For CoBA_TF_Adapter: entirely in time-domain
                keys = adapter.static_keys[v]
                keys_norm = F.normalize(keys, p=2, dim=-1)
                sim_matrix_k = torch.matmul(keys_norm, keys_norm.T).cpu().numpy()

                time_domain_bases = adapter.static_values[v]
                values_norm = F.normalize(time_domain_bases, p=2, dim=-1)
                sim_matrix_v = torch.matmul(values_norm, values_norm.T).cpu().numpy()
                time_domain_bases = time_domain_bases.cpu().numpy()
            else:
                # ==========================================
                # 1. Keys (k) 相似度矩阵
                # ==========================================
                keys = adapter.codebook_keys[v] # (n_bases, feature_dim)
                keys_norm = F.normalize(keys, p=2, dim=-1)
                sim_matrix_k = torch.matmul(keys_norm, keys_norm.T).cpu().numpy()

                # ==========================================
                # 2. Values (v) 频域相似度矩阵
                # ==========================================
                if getattr(adapter, 'var_wise', False):
                    b_r = adapter.bases_r[:, :, v] # (n_bases, freq_len)
                    b_i = adapter.bases_i[:, :, v]
                else:
                    b_r = adapter.bases_r
                    b_i = adapter.bases_i
                    
                complex_bases = torch.complex(b_r, b_i)
                mag = stable_complex_abs(complex_bases)
                mag_norm = F.normalize(mag, p=2, dim=-1)
                sim_matrix_v = torch.matmul(mag_norm, mag_norm.T).cpu().numpy()

                # ==========================================
                # 3. Values (v) 转换回时域波形
                # ==========================================
                time_domain_bases = torch.fft.irfft(complex_bases, n=adapter.window_len, dim=-1).cpu().numpy()

            # ==========================================
            # 开始为当前变量 v 单独创建画布和绘图
            # ==========================================
            
            # 计算上三角的均值（排除对角线的自身相似度 1.0）
            if n_bases > 1:
                idx = np.triu_indices(n_bases, k=1)
                mean_sim_k = np.mean(np.abs(sim_matrix_k[idx]))
                mean_sim_v = np.mean(np.abs(sim_matrix_v[idx]))
            else:
                mean_sim_k = 1.0
                mean_sim_v = 1.0
                
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))
            fig.suptitle(f'Knowledge Vectors Analysis - Var {v} (N_bases: {n_bases})\nMean Sim (Keys): {mean_sim_k:.4f} | Mean Sim (Values): {mean_sim_v:.4f}', fontsize=16)

            # 图 1: Keys
            sns.heatmap(sim_matrix_k, ax=axes[0], cmap='coolwarm', vmin=-1, vmax=1, 
                        annot=False, fmt=".2f", square=True)
            axes[0].set_title(f'Var {v} - Keys Cosine Similarity')
            axes[0].set_xlabel('Base Index')
            axes[0].set_ylabel('Base Index')

            # 图 2: Values
            sns.heatmap(sim_matrix_v, ax=axes[1], cmap='viridis', vmin=-1, vmax=1, 
                        annot=False, fmt=".2f", square=True)
            if hasattr(adapter, 'static_keys') and hasattr(adapter, 'static_values'):
                axes[1].set_title(f'Var {v} - Values (Time) Similarity')
            else:
                axes[1].set_title(f'Var {v} - Values (Freq Mag) Similarity')
            axes[1].set_xlabel('Base Index')
            axes[1].set_ylabel('Base Index')

            # 图 3: Values 时域波形
            for i in range(n_bases):
                offset = i * (np.max(time_domain_bases) - np.min(time_domain_bases) + 1e-5) * 1.5
                axes[2].plot(time_domain_bases[i] + offset, label=f'Base {i}')
            
            axes[2].set_title(f'Var {v} - Values Time-Domain Waveforms')
            axes[2].set_yticks([]) # 去掉 Y 轴刻度，因为加了 offset
            axes[2].set_xlabel('Time Step')
            if n_bases <= 15:
                axes[2].legend(loc='upper right', bbox_to_anchor=(1.25, 1))

            plt.tight_layout()
            
            # 单独保存这张图
            save_path = os.path.join(save_dir, f"knowledge_vectors_N{n_bases}_var{v}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
            # 非常重要：画完一张图必须 close，否则循环多了会内存溢出
            plt.close(fig)
            
    print(f"[*] Successfully saved {plot_vars} variable visualization images to `{save_dir}/`")