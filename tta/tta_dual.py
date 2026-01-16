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
from tta.tta_dual_utils.model_manager import TTAModelManager
from tta.utils import save_tta_results
from device_manager import global_device


def build_calibration_module(cfg) -> Optional[CalibrationContainer]:
    def get_model_dims(cfg):
        is_patchtst = (cfg.MODEL.NAME == 'PatchTST')
        n_var = 1 if is_patchtst else cfg.DATA.N_VAR
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
        'lowrank-coba-GCM': CoBA_low_rank_GCM,
        'coba-online-only': CoBA_online_only,
        'identity': IdentityAdapter,
    }
    if model_type == 'coba-GCM':
        coba_params = {
            'n_bases': cfg.TTA.DUAL.GCM_N_BASES,
        }
        params.update(coba_params)
    elif model_type == 'lowrank-coba-GCM' or model_type == 'coba-online-only':
        coba_params = {
            'n_bases': cfg.TTA.DUAL.GCM_N_BASES,
            'low_ranks': cfg.TTA.DUAL.LOWRANK_RANKS,
            'query_type': cfg.TTA.DUAL.QUERY_TYPE,
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
        # in_model = ModelClass(seq_len, n_var, **params)
        in_model = tafas_GCM(seq_len, n_var, **params)
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
        return CoBA_Loss(lambda_ortho=0.01)
    elif loss_name == "LOWRANK-COBA":
        return LowRankCoBALoss(lambda_ortho=cfg.TTA.DUAL.LAMBDA_ORTHO)
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
        loss_name = getattr(self.cfg.TTA.DUAL, 'LOSS_NAME', 'MSE')
        input_enable = getattr(self.cfg.TTA.DUAL, 'CALI_INPUT_ENABLE', False)
        output_enable = getattr(self.cfg.TTA.DUAL, 'CALI_OUTPUT_ENABLE', False)

        # parts = [
        #     f'dual-cali-{cali_name}',
        #     f'loss-{loss_name}'
        # ]

        # if input_enable:
        #     parts.append("in")
        # if output_enable:
        #     parts.append("out")
        # if isinstance(self.cali.out_cali, CoBA_online_only):
        #     parts.append(f'{self.cfg.TTA.DUAL.COBA_ONLINE_LR}')
            # if self.cfg.TTA.DUAL.COBA_ONLINE_ENABLED:
            #     parts.append('out-adapter')
        # if isinstance(self.loss_fn, LowRankCoBALoss):
        #     parts.append(f'lambda-ortho-{self.cfg.TTA.DUAL.LAMBDA_ORTHO}')

        parts = []
        if self.cfg.TTA.DUAL.CALI_NAME == 'coba-online-only':
            parts.append("coba-online-only")
            parts.append(f'{self.cfg.TTA.DUAL.COBA_ONLINE_LR}')
        elif self.cfg.TTA.DUAL.COBA_ONLINE_ENABLED:
            parts.append("coba-online")
            parts.append(f'{self.cfg.TTA.DUAL.COBA_ONLINE_LR}')
        else:
            parts.append("coba-offline")
            parts.append(f'{self.cfg.TTA.SOLVER.BASE_LR}')
        parts.append(f'lambda-ortho-{self.cfg.TTA.DUAL.LAMBDA_ORTHO}')

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

        if isinstance(self.cali.out_cali, CoBA_GCM) or isinstance(self.cali.out_cali, CoBA_low_rank_GCM) or isinstance(self.cali.out_cali, Auxiliary_GCM):
            self._pretrain_adapter()
            self.cali.out_cali.online_mode = self.cfg.TTA.DUAL.COBA_ONLINE_ENABLED # Enable online mode after pre-training
        
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
    
    def _pretrain_adapter(self):
        self._switch_model_to_train()
        for epoch in range(self.cfg.TTA.DUAL.PRETRAIN_EPOCHS):
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
                elif isinstance(self.loss_fn, LowRankCoBALoss):
                    loss = self.loss_fn(pred, ground_truth, bases_left=self.cali.out_cali.bases_left, bases_right=self.cali.out_cali.bases_right)
                else:
                    loss = self.loss_fn(pred, ground_truth) 
                # print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if hasattr(self.cali.out_cali, 'analyzer'):
                    self.cali.out_cali.analyzer.record_batch()
            if hasattr(self.cali.out_cali, 'analyzer'):
                self.cali.out_cali.analyzer.end_epoch()
        self._switch_model_to_eval()
        # visualize_bases_interpretation(self.cali.out_cali, self.cfg.DATA.PRED_LEN)

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
        amplitude = torch.sqrt(fft_result.real.pow(2) + fft_result.imag.pow(2))
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
                elif isinstance(self.loss_fn, LowRankCoBALoss):
                    loss = self.loss_fn(pred, ground_truth, bases_left=self.cali.out_cali.bases_left, bases_right=self.cali.out_cali.bases_right)
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
            elif isinstance(self.loss_fn, LowRankCoBALoss):
                loss_partial = self.loss_fn(pred_partial, ground_truth_partial, bases_left=self.cali.out_cali.bases_left, bases_right=self.cali.out_cali.bases_right)
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