from typing import Tuple, Optional

import torch
import torch.nn as nn

from config import get_norm_method
from utils.misc import prepare_inputs


def forecast(
    cfg, 
    inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
    model: nn.Module,
    norm_module: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
    norm_method = get_norm_method(cfg)
    if norm_method == 'SAN':
        enc_window, statistics = norm_module.normalize(enc_window)
    elif norm_method == 'RevIN':
        enc_window = norm_module(enc_window, 'norm')
    elif norm_method == 'DishTS':
        enc_window, _ = norm_module(enc_window, 'forward')
    else:  # Normalization from Non-stationary Transformer
        mean = enc_window.mean(1, keepdim=True).detach()          # [B,1,n_var]
        enc_centered = enc_window - mean
        stdev = torch.sqrt(torch.var(enc_centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        enc_window = enc_centered / stdev
    
    # Correct slicing: single target channel (c_out) starting at TARGET_START_IDX
    start = cfg.DATA.TARGET_START_IDX
    target_end = start + cfg.MODEL.c_out
    ground_truth = dec_window[:, -cfg.DATA.PRED_LEN:, start:target_end].float()

    # Prepare decoder input (label_len original + zeros for pred horizon)
    dec_zeros = torch.zeros_like(dec_window[:, -cfg.DATA.PRED_LEN:, :]).float()
    dec_window = torch.cat([dec_window[:, :cfg.DATA.LABEL_LEN, :], dec_zeros], dim=1).float().cuda()
    
    model_cfg = cfg.MODEL
    if model_cfg.output_attention:
        pred = model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)[0]
    else:
        pred = model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)
    
    pred = pred[:, -cfg.DATA.PRED_LEN:, start:target_end]

    # De-normalize only for Non-stationary Transformer path
    if norm_method == 'SAN':
        pred = norm_module.de_normalize(pred, statistics)
    elif norm_method == 'RevIN':
        pred = norm_module(pred, 'denorm')
    elif norm_method == 'DishTS':
        pred = norm_module(pred, 'inverse')
    elif norm_method not in ['SAN', 'RevIN', 'DishTS']:
        # old version
        # pred = pred * (stdev[:, 0, :].unsqueeze(1).repeat(1, cfg.DATA.PRED_LEN, 1))
        # pred = pred + (mean[:, 0, :].unsqueeze(1).repeat(1, cfg.DATA.PRED_LEN, 1))
        
        # mean, stdev defined only in this path
        # ground_truth was never normalized (decoder raw), so do not de-normalize it
        # breakpoint()
        # pred = pred * stdev[:, :, start:target_end] + mean[:, :, start:target_end]
        pred = pred * (stdev[:, 0, start:target_end].unsqueeze(1).repeat(1, cfg.DATA.PRED_LEN, 1))
        pred = pred + (mean[:, 0, start:target_end].unsqueeze(1).repeat(1, cfg.DATA.PRED_LEN, 1))

    return pred, ground_truth
