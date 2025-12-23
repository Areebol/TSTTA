import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from datasets.loader import get_test_dataloader
from utils.misc import prepare_inputs
from models.forecast import forecast
from tta.utils import save_tta_results
import matplotlib.pyplot as plt
from tta.gating import *
from tta.adapter import adapter_factory
from tta.utils import TTADataManager
from tta.visualizer import TTAVisualizer
from tta.loss import PETSALoss

class TTARunner(nn.Module):
    def __init__(self, cfg, model: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.model = model

        n_vars = getattr(cfg.MODEL, "c_out", None)
        if n_vars is None: 
            n_vars = getattr(cfg.MODEL, "enc_in", None)
        self.n_vars = n_vars
        self.device = next(self.model.parameters()).device
        self.s_max = getattr(cfg.TTA.OURS, 'S_MAX', 1.0)
        self.steps_per_batch = getattr(cfg.TTA.OURS, 'STEPS_PER_BATCH', 1)

        self.cur_step = self.cfg.DATA.SEQ_LEN - 2
        self.pred_step_end_dict = {}
        self.inputs_dict = {}
        self.n_adapt = 0
        self.mse_all = []
        self.mae_all = []
        
        self._setup_tta_params()
        self._setup_adapter()
        self._setup_gating()
        self._setup_require_grad()
        self._setup_optimizer()
        self._setup_tta_method()
        self.data_manager = TTADataManager(
            cfg, 
            enabled=getattr(cfg.TTA, 'SAVE_ANALYSIS_DATA', True)
        )
        self.visualizer = TTAVisualizer(save_dir=f"./visualize/{cfg.MODEL.NAME}-{cfg.DATA.NAME}-{cfg.DATA.PRED_LEN}/{self.tta_method}", cfg=cfg)
        self._setup_test_loader()

    def _calculate_period_and_batch_size(self, enc_window_first):
        fft_result = torch.fft.rfft(enc_window_first - enc_window_first.mean(dim=0), dim=0)
        amplitude = torch.abs(fft_result)
        power = torch.mean(amplitude ** 2, dim=0)
        try:
            period = enc_window_first.shape[0] // torch.argmax(amplitude[:, power.argmax()]).item()
        except:
            period = 24
        period *= self.cfg.TTA.OURS.PERIOD_N
        batch_size = period + 1
        return period, batch_size

    def _adapt_with_full_ground_truth_if_available(self):
        while self.pred_step_end_dict and self.cur_step >= self.pred_step_end_dict[min(self.pred_step_end_dict.keys())]:
            batch_idx_available = min(self.pred_step_end_dict.keys())
            inputs_history = self.inputs_dict.pop(batch_idx_available)

            pred, ground_truth = forecast(self.cfg, inputs_history, self.model, None)

            for _ in range(self.steps_per_batch):
                self.n_adapt += 1
                pred_adapted = self._forward_with_adapter(pred, inputs_history[0])
                mse_loss = F.mse_loss(pred_adapted, ground_truth)
                reg_loss = F.mse_loss(pred_adapted, pred) 
                loss = mse_loss + self.reg_coeff * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.pred_step_end_dict.pop(batch_idx_available)

    def _adapt_with_partial_ground_truth(self, pred, ground_truth, period, batch_size, batch_idx, cur_enc_window):
        for _ in range(self.steps_per_batch):
            self.n_adapt += 1
            pred_adapted = self._forward_with_adapter(pred, cur_enc_window)

            pred_partial, ground_truth_partial = pred_adapted[0][:period], ground_truth[0][:period]
            mse_partial = F.mse_loss(pred_partial, ground_truth_partial)
            reg_loss = F.mse_loss(pred_adapted, pred)
            loss = mse_partial + self.reg_coeff * reg_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def _adjust_prediction(self, pred, inputs, batch_size, period):
        pred_after_adapt, ground_truth = forecast(self.cfg, inputs, self.model, None)
        pred_after_adapt = self._forward_with_adapter(pred_after_adapt)
        
        for i in range(batch_size - 1):
            pred[i, period - i:] = pred_after_adapt[i, period - i:]
        return pred, ground_truth

    @torch.enable_grad()
    def adapt(self):
        self.data_manager.reset()
        self.mse_all = []
        self.mae_all = []
        self.n_adapt = 0
        self.pred_step_end_dict = {}
        self.inputs_dict = {}
        self.cur_step = self.cfg.DATA.SEQ_LEN - 2
        self.all_preds_base = []
        self.all_preds_tta = []
        self.all_gts = []
        self.model.eval()
        from tqdm import tqdm
        pbar = tqdm(self.test_loader, desc="[TTA Progress]", unit="batch")
        for idx, inputs in enumerate(pbar):
            enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
            batch_start = 0
            batch_end = 0
            batch_idx = 0
            self.cur_step = self.cfg.DATA.SEQ_LEN - 2

            while batch_end < len(enc_window_all):
                if self.cfg.TTA.OURS.RESET:
                    self._reset_state()
                enc_window_first = enc_window_all[batch_start]

                if self.cfg.TTA.OURS.PAAS:
                    period, batch_size = self._calculate_period_and_batch_size(enc_window_first)
                else:
                    batch_size = self.cfg.TTA.OURS.BATCH_SIZE
                    period = batch_size - 1

                batch_end = batch_start + batch_size
                if batch_end > len(enc_window_all):
                    batch_end = len(enc_window_all)
                    batch_size = batch_end - batch_start

                self.cur_step += batch_size

                cur_inputs = (
                    enc_window_all[batch_start:batch_end],
                    enc_window_stamp_all[batch_start:batch_end],
                    dec_window_all[batch_start:batch_end],
                    dec_window_stamp_all[batch_start:batch_end],
                )
                cur_enc_window = cur_inputs[0]

                self.pred_step_end_dict[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN
                self.inputs_dict[batch_idx] = cur_inputs

                # Forecast
                pred, ground_truth = forecast(self.cfg, cur_inputs, self.model, None)
                with torch.no_grad():
                    if isinstance(self.gating, CILossTrendGating):
                        se = F.mse_loss(pred, ground_truth, reduction='none')
                        per_channel_mse = se.mean(dim=(0, 1)) # 形状为 [n_vars]
                    elif isinstance(self.gating, CGLossTrendGating):
                        mse_loss = F.mse_loss(pred, ground_truth)
                self.all_preds_base.append(pred.detach().cpu().numpy())
                # Adapter Forward (No Grad for evaluation first)
                with torch.no_grad():
                    pred_adapter = self._forward_with_adapter(pred, cur_enc_window)
                
                # Adaptation
                self._adapt_with_full_ground_truth_if_available()
                self._adapt_with_partial_ground_truth(pred, ground_truth, period, batch_size, batch_idx, cur_enc_window)

                # Adjust Prediction
                if self.cfg.TTA.OURS.ADJUST_PRED:
                    pred_adapter, ground_truth = self._adjust_prediction(pred_adapter, cur_inputs, batch_size, period)
                self.all_preds_tta.append(pred_adapter.detach().cpu().numpy())
                self.all_gts.append(ground_truth.detach().cpu().numpy())
                # Metrics
                mse = F.mse_loss(pred_adapter, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred_adapter, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                self.mse_all.append(mse)
                self.mae_all.append(mae)
                if isinstance(self.gating, CILossTrendGating):
                    self.gating.update_loss(per_channel_mse)
                elif isinstance(self.gating, CGLossTrendGating):
                    self.gating.update_loss(mse_loss)
                self.data_manager.collect(
                inputs=cur_enc_window, 
                base_pred=pred,
                tta_pred=pred_adapter,
                gt=ground_truth,
                gating=self.gating_coeff,
                mse=mse
                )
                batch_start = batch_end
                batch_idx += 1

        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
       
        self.full_pred_base = np.concatenate(self.all_preds_base, axis=0)
        self.full_pred_tta = np.concatenate(self.all_preds_tta, axis=0)
        self.full_gt = np.concatenate(self.all_gts, axis=0)
        self._report()
     
    def _setup_tta_params(self):
        self.lr = self.cfg.TTA.OURS.LR
        self.adapter_name = self.cfg.TTA.OURS.ADAPTER.NAME
        self.gating_name = self.cfg.TTA.OURS.GATING.NAME
        self.reg_coeff = self.cfg.TTA.OURS.LOSS.REG_COEFF
        self.gating_win_size = self.cfg.TTA.OURS.GATING.WIN_SIZE
        
    def _setup_adapter(self):
        device = self.device
        pred_len = self.cfg.DATA.PRED_LEN
        seq_len = self.cfg.DATA.SEQ_LEN

        n_vars = getattr(self.cfg.MODEL, "c_out", None)
        if n_vars is None: n_vars = getattr(self.cfg.MODEL, "enc_in", None)
        self.base_adapter = adapter_factory(
            name=self.adapter_name,
            pred_len=pred_len,
            n_vars=n_vars,
            cfg=self.cfg,
        ).to(device)
        
    def _setup_gating(self):
        self.gating = gating_factory(
            name=self.gating_name,
            n_vars=self.n_vars,
            window_size=self.gating_win_size,
            cfg=self.cfg,
        ).to(self.device)
        
    def _setup_require_grad(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.base_adapter.setup_require_grad(require_grad=True)
        self.gating.setup_require_grad(require_grad=True)
            
    def _setup_optimizer(self):
        base_lr = getattr(self.cfg.TTA.OURS, 'LR', 1e-3)
        param_groups = [
            {
                'params': self.base_adapter.parameters(),
                'lr': base_lr,
            }
        ]

        gating_params = list(self.gating.parameters())
        if len(gating_params) > 0:
            gating_lr = base_lr * getattr(self.cfg.TTA.OURS, 'GATING_LR_SCALE', 1)
            param_groups.append({'params': gating_params, 'lr': gating_lr})
        self.optimizer = torch.optim.Adam(param_groups)

    def _setup_test_loader(self):
        self.test_loader = get_test_dataloader(self.cfg)
        batch_size = len(self.test_loader.dataset)
        self.test_loader = get_test_dataloader(self.cfg, batch_size=batch_size)

    def _setup_tta_method(self):
        self.tta_method = f"{self.lr}-{self.adapter_name}-{self.gating_name}-reg-{self.reg_coeff}-s-max-{self.s_max}"
        if self.gating_name in ['ci-loss-trend', 'cg-loss-trend']:
            self.tta_method += f"-{self.gating_win_size}"

    def _reset_state(self):
        self._setup_adapter()
        self._setup_require_grad()
        self._setup_optimizer()

    def _forward_with_adapter(self, base_pred, enc_window):
        input_full = base_pred
        adapter_out = self.base_adapter(input_full)
        
        gating_coeff = self.gating(input_full)
        self.gating_coeff = gating_coeff.squeeze(1)
        return base_pred + gating_coeff * adapter_out

    def _report(self):
        save_tta_results(
            tta_method=self.tta_method,
            seed=self.cfg.SEED,
            model_name=self.cfg.MODEL.NAME,
            dataset_name=self.cfg.DATA.NAME,
            pred_len=self.cfg.DATA.PRED_LEN,
            mse_after_tta=self.mse_all.mean(),
            mae_after_tta=self.mae_all.mean(),
        )
        full_data = self.data_manager.get_full_data()
        if full_data and self.cfg.TTA.VISUALIZE:
            self.visualizer.plot_all(full_data)

def build_tta_runner(cfg, model):
    return TTARunner(cfg, model)