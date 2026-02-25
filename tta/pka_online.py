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
from datasets.loader import get_test_dataloader, get_tta_train_dataloader, get_domain_shift_dataloader

from tta.loss import *
from tta.tta_dual_utils.GCM import *
from tta.pattern_bank import *
from tta.tta_dual_utils.model_manager import TTAModelManager
from tta.utils import save_tta_results
from device_manager import global_device


def build_calibration_module(cfg) -> Optional[CalibrationContainer]:
    def get_model_dims(cfg):
        is_patchtst = (cfg.MODEL.NAME == 'PatchTST')
        # n_var = 1 if is_patchtst else cfg.DATA.N_VAR
        n_var = cfg.MODEL.c_out if is_patchtst else cfg.DATA.N_VAR
        return cfg.DATA.SEQ_LEN, cfg.DATA.PRED_LEN, n_var
    
    if not cfg.TTA.PKA.CALI_MODULE:
        return None
    
    seq_len, pred_len, n_var = get_model_dims(cfg)
    params = {
        'hidden_dim': cfg.TTA.PKA.HIDDEN_DIM,
        'gating_init': cfg.TTA.PKA.GATING_INIT,
        'var_wise': cfg.TTA.PKA.GCM_VAR_WISE,
    }
    model_type = getattr(cfg.TTA.PKA, 'CALI_NAME', 'PKA_GCM')
    
    constructors = {
        'tafas-GCM': tafas_GCM,
        'petsa-GCM': petsa_GCM,
        'CoBA_GCM': CoBA_GCM,
        'lowrank-coba-GCM': CoBA_low_rank_GCM,
        'identity': IdentityAdapter,
        'PKA_GCM': PKA_GCM, 
        'PKA_OnLine': PKA_OnLine,
        'PKA_LDict': PKA_LDict,
    }
    if model_type == 'CoBA_GCM':
        coba_params = {
            'n_bases': cfg.TTA.PKA.GCM_N_BASES,
        }
        params.update(coba_params)
    elif model_type == 'PKA_GCM':
        coba_params = {
            'n_static': cfg.TTA.PKA.N_PATTERNS,
            'energy_threshold': cfg.TTA.PKA.ENERGY_THRESHOLD,
            'query_type': cfg.TTA.PKA.QUERY_TYPE,
            'seq_len': seq_len,
        }
        params.update(coba_params)
    elif model_type in ['PKA_OnLine', 'PKA_LDict']:
        coba_params = {
            'n_static': cfg.TTA.PKA.N_PATTERNS,
            'energy_threshold': cfg.TTA.PKA.ENERGY_THRESHOLD,
            'query_type': cfg.TTA.PKA.QUERY_TYPE,
            'seq_len': seq_len,
            'bias_momentum': cfg.TTA.PKA.BIAS_MOMENTUM,
            'max_dynamic_capacity': cfg.TTA.PKA.MAX_DYNAMIC_CAPACITY,
            'temperature': cfg.TTA.PKA.TEMPERATURE,
        }
        params.update(coba_params)
    elif model_type in ['lowrank-coba-GCM']:
        coba_params = {
            'n_bases': cfg.TTA.PKA.GCM_N_BASES,
            'low_ranks': cfg.TTA.PKA.LOWRANK_RANKS,
            'query_type': cfg.TTA.PKA.QUERY_TYPE,
        }
        params.update(coba_params)
    elif model_type in ['RoCoBA_FreqDomain_Norm']:
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
    
    if cfg.TTA.PKA.CALI_INPUT_ENABLE:
        # in_model = ModelClass(seq_len, n_var, **params)
        in_model = tafas_GCM(seq_len, n_var, **params)
    if cfg.TTA.PKA.CALI_OUTPUT_ENABLE:
        out_model = ModelClass(pred_len, n_var, **params)
    return CalibrationContainer(in_model, out_model)

def build_loss_fn(cfg) -> nn.Module:
    loss_name = getattr(cfg.TTA.PKA, 'LOSS_NAME', 'MSE')
    if loss_name == 'MSE':
        return StandardMSELoss()
    elif loss_name == 'PETSA': 
        alpha = getattr(cfg.TTA.PKA, 'PETSA_LOSS_ALPHA', 0.1)
        return PETSALoss(alpha=alpha)
    elif loss_name == "CoBA_Loss":
        return CoBA_Loss(lambda_ortho=cfg.TTA.PKA.LAMBDA_ORTHO)
    elif loss_name == "LOWRANK-COBA":
        return LowRankCoBALoss(lambda_ortho=cfg.TTA.PKA.LAMBDA_ORTHO)
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

        # cali_name = getattr(self.cfg.TTA.PKA, 'CALI_NAME', 'unknown')
        # loss_name = getattr(self.cfg.TTA.PKA, 'LOSS_NAME', 'MSE')
        # input_enable = getattr(self.cfg.TTA.PKA, 'CALI_INPUT_ENABLE', False)
        # output_enable = getattr(self.cfg.TTA.PKA, 'CALI_OUTPUT_ENABLE', False)


        parts = []
        if self.cfg.TTA.PKA.CALI_NAME == 'PKA_GCM' and not self.cfg.TTA.PKA.COBA_ONLINE_ENABLED:
            parts.append("coba-offline")
            parts.append(f'{self.cfg.TTA.SOLVER.BASE_LR}')
            parts.append(f'patterns-{self.cfg.TTA.PKA.N_PATTERNS:03d}')
            parts.append(f'ortho-{self.cfg.TTA.PKA.LAMBDA_ORTHO}')
        elif self.cfg.TTA.PKA.CALI_NAME == 'PKA_GCM' and self.cfg.TTA.PKA.COBA_ONLINE_ENABLED:
            parts.append("coba-online")
            parts.append(f'offlinelr-{self.cfg.TTA.SOLVER.BASE_LR}')
            parts.append(f'onlinelr-{self.cfg.TTA.PKA.COBA_ONLINE_LR}')
        
        elif self.cfg.TTA.PKA.CALI_NAME == 'PKA_OnLine' and not self.cfg.TTA.PKA.COBA_ONLINE_ENABLED:
            parts.append("pka-offline")
            parts.append(f'{self.cfg.TTA.SOLVER.BASE_LR}')
            parts.append(f'patterns-{self.cfg.TTA.PKA.N_PATTERNS:03d}')
            parts.append(f'ortho-{self.cfg.TTA.PKA.LAMBDA_ORTHO}')
        elif self.cfg.TTA.PKA.CALI_NAME == 'PKA_OnLine' and self.cfg.TTA.PKA.COBA_ONLINE_ENABLED:
            parts.append("pka-online")
            parts.append(f'offlinelr-{self.cfg.TTA.SOLVER.BASE_LR}')
            parts.append(f'onlinelr-{self.cfg.TTA.PKA.COBA_ONLINE_LR}')

        elif self.cfg.TTA.PKA.CALI_NAME == 'PKA_LDict' and not self.cfg.TTA.PKA.COBA_ONLINE_ENABLED:
            parts.append("pka-ldict-offline")
            parts.append(f'{self.cfg.TTA.SOLVER.BASE_LR}')
            parts.append(f'patterns-{self.cfg.TTA.PKA.N_PATTERNS:03d}')
        elif self.cfg.TTA.PKA.CALI_NAME == 'PKA_LDict' and self.cfg.TTA.PKA.COBA_ONLINE_ENABLED:
            parts.append("pka-ldict-online")
            parts.append(f'offlinelr-{self.cfg.TTA.SOLVER.BASE_LR}')
            # parts.append(f'onlinelr-{self.cfg.TTA.PKA.COBA_ONLINE_LR}')
            parts.append(f'patterns-{self.cfg.TTA.PKA.N_PATTERNS:03d}')
        
        
        else:
            parts.append("coba-offline")
            parts.append(f'{self.cfg.TTA.SOLVER.BASE_LR}')
        # parts.append(f'lambda-ortho-{self.cfg.TTA.PKA.LAMBDA_ORTHO}')

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

        if isinstance(self.cali.out_cali, (CoBA_GCM, CoBA_low_rank_GCM, PKA_GCM, PKA_OnLine, PKA_LDict)):
            self._pretrain_adapter()
            self.cali.out_cali.online_mode = self.cfg.TTA.PKA.COBA_ONLINE_ENABLED # Enable online mode after pre-training
        
        else:
            print("No adapter pre-training needed.")

    
    def _pretrain_adapter(self):
        self._switch_model_to_train()
        for epoch in range(self.cfg.TTA.PKA.PRETRAIN_EPOCHS):
            for inputs in self.tta_train_loader:
                enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
                inputs = (enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all)
                if self.cali.input_calibration is not None:
                    inputs = self.cali.input_calibration(inputs)
                pred, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
                
                # output correction
                if self.cali.output_calibration is not None:
                    if isinstance(self.cali.out_cali, (PKA_GCM, PKA_OnLine, PKA_LDict)):
                        pred, z_t = self.cali.output_calibration(pred, inputs=enc_window_all)
                    else:
                        pred = self.cali.output_calibration(pred)
                
                # compute loss
                if isinstance(self.loss_fn, CoBA_Loss):
                    loss = self.loss_fn(pred, ground_truth, bases=self.cali.out_cali.static_keys)
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
        self._freeze_all() # fix the adapter after pre-training
        # visualize_bases_interpretation(self.cali.out_cali, self.cfg.DATA.PRED_LEN)

    def _reset(self):
        self.manager.reset()
        self.optimizer.load_state_dict(deepcopy(self.optimizer_state))

    def _switch_model_to_train(self):
        self.manager.train()
    
    def _switch_model_to_eval(self):
        self.manager.eval()
    
    def _freeze_all(self):
        self.manager._freeze_all()

    
    def _calculate_period_and_batch_size(self, enc_window_first):
        fft_result = torch.fft.rfft(enc_window_first - enc_window_first.mean(dim=0), dim=0)
        # amplitude = torch.abs(fft_result)
        amplitude = stable_complex_abs(fft_result)
        power = torch.mean(amplitude ** 2, dim=0)
        try:
            period = enc_window_first.shape[0] // torch.argmax(amplitude[:, power.argmax()]).item()
        except:
            period = 24
        period *= self.cfg.TTA.PKA.PERIOD_N
        batch_size = period + 1
        return period, batch_size

    def _adapt_with_full_ground_truth_if_available(self):
        """
        处理延迟到达的完整 Ground Truth (Delayed Feedback)
        """
        while self.cur_step >= self.pred_step_end_dict[min(self.pred_step_end_dict.keys())]:
            batch_idx_available = min(self.pred_step_end_dict.keys())
            inputs_history = self.inputs_dict.pop(batch_idx_available)

            with torch.no_grad(): # 在线更新通常不需要梯度 (除了 Static Fine-tune)
                self.n_adapt += 1

                if self.cali.input_calibration is not None:
                    inputs_history = self.cali.input_calibration(inputs_history)
                
                # 1. Get Base Prediction
                pred_base, ground_truth = forecast(self.cfg, inputs_history, self.model, self.norm_module)
                
                # 2. Get Final Prediction & Query
                if self.cali.output_calibration is not None:
                    if isinstance(self.cali.out_cali, (PKA_GCM, PKA_OnLine, PKA_LDict)):
                        enc_history = prepare_inputs(inputs_history)[0]
                        # Forward pass to get Y_final and z_t
                        pred_final, z_t = self.cali.output_calibration(pred_base, enc_history)

                        # =========================================================
                        # [关键修改] OD-TTA v3.3 在线更新流程 
                        # =========================================================
                        if self.cfg.TTA.PKA.COBA_ONLINE_ENABLED:
                            # Step 4.1: Update Bias (Always Run) 
                            # 使用 Y_base 误差更新 Bias 更稳健 
                            # self.cali.out_cali.update_bias(ground_truth, y_base_pred=pred_base)
                            
                            # Step 4.2: Update Dynamic Memory (Conditional) 
                            # 基于 Y_final 的剩余误差来决定是否新增
                            self.cali.out_cali.update_dynamic_memory(
                                z_t=z_t, 
                                y_gt=ground_truth, 
                                y_final_pred=pred_final,
                                # threshold=self.cfg.TTA.PKA.ENERGY_THRESHOLD
                            )
                    else:
                        pred_final = self.cali.output_calibration(pred_base)
                
            self.pred_step_end_dict.pop(batch_idx_available)

    def _adapt_with_partial_ground_truth(self, inputs, period, batch_size, batch_idx):
        """
        处理即时到达的部分 Ground Truth (Immediate/Partial Feedback)
        """
        for _ in range(self.cfg.TTA.PKA.STEPS):
            self.n_adapt += 1
            
            if self.cali.input_calibration is not None:
                inputs = self.cali.input_calibration(inputs)
            
            # 1. Base Prediction
            pred_base, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
        
            # 2. Output Calibration & Update
            if self.cali.output_calibration is not None:
                if isinstance(self.cali.out_cali, (PKA_GCM, PKA_OnLine, PKA_LDict)):
                    enc_window = prepare_inputs(inputs)[0]
                    # Forward
                    pred_final, z_t = self.cali.output_calibration(pred_base, enc_window)
                    
                    # 截取 partial 部分用于计算误差
                    pred_base_partial = pred_base[:, :period, :]
                    pred_final_partial = pred_final[:, :period, :]
                    ground_truth_partial = ground_truth[:, :period, :]
                    z_t_partial = z_t # Query 是一样的，它是基于 Input 的

                    # =========================================================
                    # 在线更新流程
                    # =========================================================
                    # if self.cfg.TTA.PKA.COBA_ONLINE_ENABLED:
                    #     # Update Bias Only
                    #     self.cali.out_cali.update_bias(
                    #         ground_truth_partial, 
                    #         y_base_pred=pred_base_partial
                    #     )

                else:
                    pred_final = self.cali.output_calibration(pred_base)
            
        return pred_final, ground_truth

    @torch.no_grad()
    def _adjust_prediction(self, pred, inputs, batch_size, period):
        if self.cali.input_calibration is not None:
            inputs = self.cali.input_calibration(inputs)
        pred_after_adapt, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
        if self.cali.output_calibration is not None:
            if isinstance(self.cali.out_cali, (PKA_GCM, PKA_OnLine, PKA_LDict)):
                enc_window = prepare_inputs(inputs)[0]
                pred_after_adapt, z_t = self.cali.output_calibration(pred_after_adapt, enc_window)
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
        )
        self.model.eval()

        tta_method = 'offline' if not self.cfg.TTA.PKA.COBA_ONLINE_ENABLED else 'online'
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
            
            if self.cfg.TTA.PKA.PAAS:
                period, batch_size = self._calculate_period_and_batch_size(enc_window_first)
            else:
                batch_size = self.cfg.TTA.PKA.BATCH_SIZE
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

            
            if self.cfg.TTA.PKA.ADJUST_PRED:
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
                    if self.cfg.TTA.PKA.PAAS:
                        period, batch_size = self._calculate_period_and_batch_size(enc_window_first)
                    else:
                        batch_size = self.cfg.TTA.PKA.BATCH_SIZE
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

                    if self.cfg.TTA.PKA.ADJUST_PRED:
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