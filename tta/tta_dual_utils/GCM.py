from typing import List
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.misc import prepare_inputs
import math
from device_manager import global_device
from tta.tta_dual_utils.query_net import *

eved_enable = True
import logging
if eved_enable:
    logging.warning("\ntta enable in eVED mode. \nIf not using eVED, please set eved_enable = False in tta_dual_utils/GCM.py\n")

class tafas_GCM(nn.Module):
    def __init__(self, window_len, n_var=1, hidden_dim=64, gating_init=0.01, var_wise=True, **args):
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

class Fre_GCM(nn.Module):
    def __init__(self, window_len, n_var=1, hidden_dim=64, gating_init=0.01, var_wise=True, **args):
        super(Fre_GCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        self.freq_len = window_len // 2 + 1
        
        # 初始化频域校准参数 (实部和虚部)
        # scale 设为较小值，以保证初始状态接近恒等映射
        scale = 1e-5
        
        if var_wise:
            # 维度: (1, Freq, Var) 用于广播 Batch 维度
            self.freq_weight_r = nn.Parameter(scale * torch.randn(1, self.freq_len, n_var))
            self.freq_weight_i = nn.Parameter(scale * torch.randn(1, self.freq_len, n_var))
            self.freq_bias_r = nn.Parameter(torch.zeros(1, self.freq_len, n_var))
            self.freq_bias_i = nn.Parameter(torch.zeros(1, self.freq_len, n_var))
        else:
            # 维度: (1, Freq, 1) 广播 Batch 和 Var 维度
            self.freq_weight_r = nn.Parameter(scale * torch.randn(1, self.freq_len, 1))
            self.freq_weight_i = nn.Parameter(scale * torch.randn(1, self.freq_len, 1))
            self.freq_bias_r = nn.Parameter(torch.zeros(1, self.freq_len, 1))
            self.freq_bias_i = nn.Parameter(torch.zeros(1, self.freq_len, 1))

        self.gating = nn.Parameter(gating_init * torch.ones(n_var))

        
    def forward(self, x):
        # 1. Transform to Frequency Domain
        # x: (B, L, V) -> x_fft: (B, F, V)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')

        # 2. Element-wise Complex Calibration in Frequency
        # (a+bi)(c+di) = (ac-bd) + i(ad+bc)
        # R = Xr*Wr - Xi*Wi + Br
        # I = Xr*Wi + Xi*Wr + Bi
        delta_real = (
            x_fft.real * self.freq_weight_r - x_fft.imag * self.freq_weight_i + self.freq_bias_r
        )
        delta_imag = (
            x_fft.real * self.freq_weight_i + x_fft.imag * self.freq_weight_r + self.freq_bias_i
        )
        
        delta_fft = torch.complex(delta_real, delta_imag)

        # 3. Transform back to Time Domain
        delta_x = torch.fft.irfft(delta_fft, n=self.window_len, dim=1, norm='ortho')

        # 4. Gated Residual Connection
        x = x + torch.tanh(self.gating) * delta_x
        return x



class CoBA_GCM(nn.Module):
    def __init__(self, window_len, n_var=1, hidden_dim=64, 
                 gating_init=0.01, var_wise=True,
                 n_bases=8, feature_dim=32):
        super(CoBA_GCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        self.n_bases = n_bases
        self.feature_dim = feature_dim
        self.online_mode = False
        self.analyzer = CoBA_Analyzer(self)
        self.codebook_keys = nn.Parameter(torch.randn(n_bases, feature_dim))
        
        if var_wise:
            self.bases = nn.Parameter(torch.Tensor(n_bases, window_len, window_len, n_var))
        else:
            self.bases = nn.Parameter(torch.Tensor(n_bases, window_len, window_len))
        
        nn.init.xavier_uniform_(self.bases) 

        fft_len = window_len // 2 + 1
        self.query_net = nn.Sequential(
            nn.Linear(fft_len * n_var, feature_dim * 2),
            # nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

        # self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))

        if var_wise:
            self.tafas_weight = nn.Parameter(torch.Tensor(window_len, window_len, n_var))
        else:
            self.tafas_weight = nn.Parameter(torch.Tensor(window_len, window_len))
        self.tafas_weight.data.zero_()
        self.tafas_gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.tafas_bias = nn.Parameter(torch.zeros(window_len, n_var))

    def _get_query(self, x):
        batch_size = x.shape[0]
        
        x_fft = torch.fft.rfft(x, dim=1)
        x_mag = stable_complex_abs(x_fft)
        
        x_feat = x_mag.reshape(batch_size, -1)
        
        query = self.query_net(x_feat)
        
        return query

    def forward(self, x):
        """
        x shape: (Batch, Window_len, N_var)
        """
        batch_size = x.size(0)

        query = self._get_query(x)
        query_norm = F.normalize(query, p=2, dim=1)           # (B, D)
        keys_norm = F.normalize(self.codebook_keys, p=2, dim=1) # (N, D)
        similarity = torch.matmul(query_norm, keys_norm.T)
        
        coeffs = F.softmax(similarity, dim=-1) # (B, N)
        
        if self.var_wise:
            w_sample = torch.einsum('bn, nlio -> blio', coeffs, self.bases)
        else:
            w_sample = torch.einsum('bn, nli -> bli', coeffs, self.bases)

        if self.var_wise:
            feat_trans = torch.einsum('biv, boiv -> bov', x, w_sample)
        else:
            feat_trans = torch.einsum('biv, boi -> bov', x, w_sample)

        feat_trans = feat_trans + self.bias

        if self.online_mode:
            if self.var_wise:
                tafas_output = torch.tanh(self.tafas_gating) * (torch.einsum('biv,iov->bov', x, self.tafas_weight) + self.tafas_bias)
            else:
                tafas_output = torch.tanh(self.tafas_gating) * (torch.einsum('biv,io->bov', x, self.tafas_weight) + self.tafas_bias)
            # out = x + torch.tanh(self.gating) * feat_trans + tafas_output
            out = x + feat_trans + tafas_output
        else:
            # out = x + torch.tanh(self.gating) * feat_trans
            out = x + feat_trans
        
        self.coeffs = coeffs
        
        return out

    def get_optim_params(self):
        params = []
        params.append(self.tafas_weight)
        params.append(self.tafas_bias)
        params.append(self.tafas_gating)
        return params

class CoBA_low_rank_GCM(nn.Module):
    def __init__(self, window_len, n_var=1, low_ranks=64, hidden_dim=32,
                 gating_init=0.01, var_wise=True,
                 n_bases=8, feature_dim=32, query_type='freq-base-CI'):
        super(CoBA_low_rank_GCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        self.n_bases = n_bases
        self.feature_dim = feature_dim
        self.online_mode = False
        self.analyzer = CoBA_Analyzer(self)
        self.rank = low_ranks
        if var_wise:
            self.bases_left = nn.Parameter(torch.Tensor(n_bases, window_len, self.rank, n_var))
            self.bases_right = nn.Parameter(torch.Tensor(n_bases, self.rank, window_len, n_var))
            self.codebook_keys = nn.Parameter(torch.randn(n_var, n_bases, feature_dim))
        else:
            self.bases_left = nn.Parameter(torch.Tensor(n_bases, window_len, self.rank))
            self.bases_right = nn.Parameter(torch.Tensor(n_bases, self.rank, window_len))
            self.codebook_keys = nn.Parameter(torch.randn(n_bases, feature_dim))
        # Initialize bases_left with column-wise orthogonality
        with torch.no_grad():
            if var_wise:
                for n in range(n_bases):
                    for v in range(n_var):
                        nn.init.orthogonal_(self.bases_left[n, :, :, v])
            else:
                for n in range(n_bases):
                    nn.init.orthogonal_(self.bases_left[n, :, :])
        
        # nn.init.kaiming_uniform_(self.bases_left, a=math.sqrt(5))
        nn.init.zeros_(self.bases_right)
        
        # --- Query Net Selection Logic (Factory) ---
        print(f"Initializing CoBA with Query Type: {query_type}")
        if query_type == 'time':
            self.query_net = QueryNet_Time(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-CI':
            self.query_net = QueryNet_Freq_Base_ChannelIndependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-CD':
            self.query_net = QueryNet_Freq_Base_ChannelDependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-hybrid':
            self.query_net = QueryNet_Freq_Hybrid(window_len, n_var, feature_dim)
        elif query_type == 'fusion':
            self.query_net = QueryNet_Fusion_Gated(window_len, n_var, feature_dim)
        elif query_type == 'multiscale':
            self.query_net = QueryNet_MultiScale(window_len, n_var, feature_dim)
        elif query_type == 'phase':
            self.query_net = QueryNet_Phase(window_len, n_var, feature_dim)
        elif query_type == 'freq-attn':
            self.query_net = QueryNet_Freq_Attn(window_len, n_var, feature_dim)
        elif query_type == 'freq-light':
            self.query_net = QueryNet_Freq_Light(window_len, n_var, feature_dim)
        elif query_type == 'wave-ms':
            self.query_net = QueryNet_Wavelet_MS(window_len, n_var, feature_dim)
        else:
            raise ValueError(f"Unknown query_type: {query_type}")

        # self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))

        if var_wise:
            self.tafas_weight = nn.Parameter(torch.Tensor(window_len, window_len, n_var))
        else:
            self.tafas_weight = nn.Parameter(torch.Tensor(window_len, window_len))
        self.tafas_weight.data.zero_()
        self.tafas_gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.tafas_bias = nn.Parameter(torch.zeros(window_len, n_var))

    def _get_query(self, x):
        return self.query_net(x)

    def forward(self, x):
        """
        x shape: (Batch, Window_len, N_var)
        """
        batch_size = x.size(0)

        query = self._get_query(x) # (B, N_vars, D) 
        query_norm = F.normalize(query, p=2, dim=-1)           # (B, N_vars, D)
        keys_norm = F.normalize(self.codebook_keys, p=2, dim=-1) # (N_vars, N_bases, D)
        # print(keys_norm.shape)
        # similarity = torch.matmul(query_norm, keys_norm.T) # (B, N_vars, N_bases)
        similarity = torch.matmul(
            query_norm.unsqueeze(2),         # (B, N_vars, 1, D)
            keys_norm.transpose(1, 2)        # (N_vars, D, N_bases)
        ).squeeze(2)                          # (B, N_vars, N_bases)
        # print(similarity.shape)
        
        # --- 替换为 Top-K 逻辑 ---
        k = 2 
        
        # 1. 找出分数最高的 k 个值的索引和数值
        topk_vals, topk_indices = torch.topk(similarity, k=k, dim=-1)
        
        # 2. 创建一个全为 -inf 的 mask (这样 Softmax 后会变成 0)
        mask = torch.full_like(similarity, float('-inf'))
        
        # 3. 将 top-k 的位置填回原始的相似度数值
        # 在 mask 的 dim=-1 维度，按照 topk_indices 的索引，填入 topk_vals
        mask.scatter_(-1, topk_indices, topk_vals)
        
        # 4. 再做 Softmax
        coeffs = F.softmax(mask, dim=-1) # (B, N_vars, N_bases)

        if self.var_wise:
            # # U: (Batch, L_out, Rank, Var) <- 聚合后的 bases_left
            # u = torch.einsum('bn, nliv -> bliv', coeffs, self.bases_left)   
            # # V: (Batch, Rank, L_in, Var)  <- 聚合后的 bases_right
            # v = torch.einsum('bn, nriv -> briv', coeffs, self.bases_right)  
            u = torch.einsum('bvn, nlrv -> blrv', coeffs, self.bases_left)
            v = torch.einsum('bvn, nrlv -> brlv', coeffs, self.bases_right)
            
            # --- 关键优化开始 ---
            # 原始 x: (B, L_in, V)
            # Step 1: x(biv) * v(briv) -> (Rank)
            # indices: batch(b), input_len(i), var(v) AND batch(b), rank(r), input_len(i), var(v)
            # result shape: (Batch, Rank, Var)
            x_reduced = torch.einsum('biv, briv -> brv', x, v)
            
            # Step 2: intermediate(brv) * u(blrv) -> (Output_len)
            # indices: batch(b), rank(r), var(v) AND batch(b), output_len(l), rank(r), var(v)
            # result shape: (Batch, Output_len, Var)
            feat_trans = torch.einsum('brv, blrv -> blv', x_reduced, u)
            
            # # 加上 bias
            # feat_trans = feat_trans + self.bias
        else:
            u = torch.einsum('bn, nli -> bli', coeffs, self.bases_left)   # (B, L, R)
            v = torch.einsum('bn, nri -> bri', coeffs, self.bases_right)  # (B, R, L)
            
            # Step 1: Project to low rank
            # x: (B, L, V), v: (B, R, L)
            # output: (B, R, V)
            x_reduced = torch.einsum('blv, bri -> brv', x, v)
            
            # Step 2: Project back to high rank
            # x_reduced: (B, R, V), u: (B, L, R)
            # output: (B, L, V)
            feat_trans = torch.einsum('brv, bli -> blv', x_reduced, u)
            
            # feat_trans = feat_trans + self.bias

        if self.online_mode:
            if self.var_wise:
                tafas_output = torch.tanh(self.tafas_gating) * (torch.einsum('biv,iov->bov', x, self.tafas_weight) + self.tafas_bias)
            else:
                tafas_output = torch.tanh(self.tafas_gating) * (torch.einsum('biv,io->bov', x, self.tafas_weight) + self.tafas_bias)
            out = x + feat_trans + tafas_output
        else:
            out = x + feat_trans
        
        self.coeffs = coeffs
        if torch.isnan(out).any():
            print("NaN detected in CoBA_low_rank_GCM output.")
            if torch.isnan(self.bases_left).any():
                print("NaN detected in bases_left.")
                print(self.bases_left)
            if torch.isnan(self.bases_right).any():
                print("NaN detected in bases_right.")
            if torch.isnan(coeffs).any():
                print("NaN detected in coeffs.")
            exit()

        return out

    def get_optim_params(self):
        params = []
        if self.online_mode:
            params.append(self.tafas_weight)
            params.append(self.tafas_bias)
            params.append(self.tafas_gating)
            params.extend(list(self.query_net.parameters()))
            # params.append(self.bias)
        else:
            params.append(self.tafas_bias)
        return params


class CoBA_online_only(nn.Module):
    def __init__(self, window_len, n_var=1, low_ranks=64, hidden_dim=32,
                 gating_init=0.01, var_wise=True,
                 n_bases=8, feature_dim=32, query_type='freq-base-CI'):
        super(CoBA_online_only, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        self.n_bases = n_bases
        self.feature_dim = feature_dim
        self.online_mode = False
        self.analyzer = CoBA_Analyzer(self)
        self.rank = low_ranks
        if var_wise:
            self.bases_left = nn.Parameter(torch.Tensor(n_bases, window_len, self.rank, n_var))
            self.bases_right = nn.Parameter(torch.Tensor(n_bases, self.rank, window_len, n_var))
            self.codebook_keys = nn.Parameter(torch.randn(n_var, n_bases, feature_dim))
        else:
            self.bases_left = nn.Parameter(torch.Tensor(n_bases, window_len, self.rank))
            self.bases_right = nn.Parameter(torch.Tensor(n_bases, self.rank, window_len))
            self.codebook_keys = nn.Parameter(torch.randn(n_bases, feature_dim))
        
        # Initialize bases_left with column-wise orthogonality
        with torch.no_grad():
            if var_wise:
                for n in range(n_bases):
                    for v in range(n_var):
                        nn.init.orthogonal_(self.bases_left[n, :, :, v])
            else:
                for n in range(n_bases):
                    nn.init.orthogonal_(self.bases_left[n, :, :])
        
        # nn.init.kaiming_uniform_(self.bases_left, a=math.sqrt(5))
        
        nn.init.zeros_(self.bases_right)
        
        # --- Query Net Selection Logic (Factory) ---
        print(f"Initializing CoBA with Query Type: {query_type}")
        if query_type == 'time':
            self.query_net = QueryNet_Time(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-CI':
            self.query_net = QueryNet_Freq_Base_ChannelIndependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-CD':
            self.query_net = QueryNet_Freq_Base_ChannelDependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-hybrid':
            self.query_net = QueryNet_Freq_Hybrid(window_len, n_var, feature_dim)
        elif query_type == 'fusion':
            self.query_net = QueryNet_Fusion_Gated(window_len, n_var, feature_dim)
        elif query_type == 'multiscale':
            self.query_net = QueryNet_MultiScale(window_len, n_var, feature_dim)
        elif query_type == 'phase':
            self.query_net = QueryNet_Phase(window_len, n_var, feature_dim)
        elif query_type == 'freq-attn':
            self.query_net = QueryNet_Freq_Attn(window_len, n_var, feature_dim)
        elif query_type == 'freq-light':
            self.query_net = QueryNet_Freq_Light(window_len, n_var, feature_dim)
        elif query_type == 'wave-ms':
            self.query_net = QueryNet_Wavelet_MS(window_len, n_var, feature_dim)
        else:
            raise ValueError(f"Unknown query_type: {query_type}")

        self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))

        if var_wise:
            self.tafas_weight = nn.Parameter(torch.Tensor(window_len, window_len, n_var))
        else:
            self.tafas_weight = nn.Parameter(torch.Tensor(window_len, window_len))
        self.tafas_weight.data.zero_()
        self.tafas_gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.tafas_bias = nn.Parameter(torch.zeros(window_len, n_var))

    def _get_query(self, x):
        return self.query_net(x)

    def forward(self, x):
        """
        x shape: (Batch, Window_len, N_var)
        """
        batch_size = x.size(0)

        query = self._get_query(x) # (B, N_vars, D) 
        query_norm = F.normalize(query, p=2, dim=-1)           # (B, N_vars, D)
        keys_norm = F.normalize(self.codebook_keys, p=2, dim=-1) # (N_vars, N_bases, D)
        # print(keys_norm.shape)
        # similarity = torch.matmul(query_norm, keys_norm.T) # (B, N_vars, N_bases)
        similarity = torch.matmul(
            query_norm.unsqueeze(2),         # (B, N_vars, 1, D)
            keys_norm.transpose(1, 2)        # (N_vars, D, N_bases)
        ).squeeze(2)                          # (B, N_vars, N_bases)
        # print(similarity.shape)
        
        # --- 替换为 Top-K 逻辑 ---
        k = 2 
        
        # 1. 找出分数最高的 k 个值的索引和数值
        topk_vals, topk_indices = torch.topk(similarity, k=k, dim=-1)
        
        # 2. 创建一个全为 -inf 的 mask (这样 Softmax 后会变成 0)
        mask = torch.full_like(similarity, float('-inf'))
        
        # 3. 将 top-k 的位置填回原始的相似度数值
        # 在 mask 的 dim=-1 维度，按照 topk_indices 的索引，填入 topk_vals
        mask.scatter_(-1, topk_indices, topk_vals)
        
        # 4. 再做 Softmax
        coeffs = F.softmax(mask, dim=-1) # (B, N_vars, N_bases)

        if self.var_wise:
            # # U: (Batch, L_out, Rank, Var) <- 聚合后的 bases_left
            # u = torch.einsum('bn, nliv -> bliv', coeffs, self.bases_left)   
            # # V: (Batch, Rank, L_in, Var)  <- 聚合后的 bases_right
            # v = torch.einsum('bn, nriv -> briv', coeffs, self.bases_right)  
            u = torch.einsum('bvn, nlrv -> blrv', coeffs, self.bases_left)
            v = torch.einsum('bvn, nrlv -> brlv', coeffs, self.bases_right)
            
            # --- 关键优化开始 ---
            # 原始 x: (B, L_in, V)
            # Step 1: x(biv) * v(briv) -> (Rank)
            # indices: batch(b), input_len(i), var(v) AND batch(b), rank(r), input_len(i), var(v)
            # result shape: (Batch, Rank, Var)
            x_reduced = torch.einsum('biv, briv -> brv', x, v)
            
            # Step 2: intermediate(brv) * u(blrv) -> (Output_len)
            # indices: batch(b), rank(r), var(v) AND batch(b), output_len(l), rank(r), var(v)
            # result shape: (Batch, Output_len, Var)
            feat_trans = torch.einsum('brv, blrv -> blv', x_reduced, u)
            
            # 加上 bias
            feat_trans = feat_trans + self.bias
        else:
            u = torch.einsum('bn, nli -> bli', coeffs, self.bases_left)   # (B, L, R)
            v = torch.einsum('bn, nri -> bri', coeffs, self.bases_right)  # (B, R, L)
            
            # Step 1: Project to low rank
            # x: (B, L, V), v: (B, R, L)
            # output: (B, R, V)
            x_reduced = torch.einsum('blv, bri -> brv', x, v)
            
            # Step 2: Project back to high rank
            # x_reduced: (B, R, V), u: (B, L, R)
            # output: (B, L, V)
            feat_trans = torch.einsum('brv, bli -> blv', x_reduced, u)
            
            # feat_trans = feat_trans + self.bias

        if self.online_mode:
            if self.var_wise:
                tafas_output = torch.tanh(self.tafas_gating) * (torch.einsum('biv,iov->bov', x, self.tafas_weight) + self.tafas_bias)
            else:
                tafas_output = torch.tanh(self.tafas_gating) * (torch.einsum('biv,io->bov', x, self.tafas_weight) + self.tafas_bias)
            out = x + feat_trans + tafas_output
        else:
            out = x + feat_trans
        
        self.coeffs = coeffs

        return out

    def get_optim_params(self):
        params = []
        if self.online_mode:
            params.append(self.tafas_weight)
            params.append(self.tafas_bias)
            params.append(self.tafas_gating)
            params.extend(list(self.query_net.parameters()))
        else:
            params.append(self.tafas_bias)
        return params

class Auxiliary_GCM(nn.Module):
    def __init__(self, window_len, n_var=1, low_ranks=64, hidden_dim=32,
                 gating_init=0.01, var_wise=True,
                 n_bases=8, feature_dim=32):
        super(Auxiliary_GCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        self.online_mode = False
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))
        if var_wise:
            self.base = nn.Parameter(torch.Tensor(window_len, window_len, n_var))
        else:
            self.base = nn.Parameter(torch.Tensor(window_len, window_len))
        
        nn.init.xavier_uniform_(self.base) 

    def forward(self, x):
        """
        x shape: (Batch, Window_len, N_var)
        """
        batch_size = x.size(0)

        w_sample = self.base

        if self.var_wise:
            feat_trans = torch.einsum('biv,iov->bov', x, self.base)
        else:
            feat_trans = torch.einsum('biv,io->bov', x, self.base)

        feat_trans = feat_trans
        out = x + feat_trans
        
        return out

    def get_optim_params(self):
        params = []
        params.append(self.bias)
        return params
    
class CoBA_Analyzer:
    def __init__(self, model):
        """
        全能分析器，用于分析 CoBA_GCM 的 Codebook 选择行为。
        """
        self.model = model
        
        # 缓存容器
        self.current_epoch_coeffs = []  # 暂存当前 Epoch 的所有 batch 系数
        self.training_history = []      # 记录每个 Epoch 的平均分布 (用于演化图)

    def clear_cache(self):
        """清空当前 Epoch 的缓存 (通常在验证开始前调用)"""
        self.current_epoch_coeffs = []

    def record_batch(self):
        """
        【训练/验证循环中使用】
        在 model(x) 之后调用，记录当前 batch 的 coefficients。
        """
        if hasattr(self.model, 'coeffs'):
            # 转移到 CPU 并转为 numpy，减少显存占用
            self.current_epoch_coeffs.append(self.model.coeffs.detach().cpu().numpy())
        else:
            print("Warning: Model has no attribute 'coeffs'. Did you run forward()?")

    def end_epoch(self):
        """
        【训练循环末尾使用】
        结算当前 Epoch 的数据，存入 history，并清空缓存。
        """
        if not self.current_epoch_coeffs:
            return
        
        # 合并当前 Epoch 所有 Batch的数据: (Total_Samples, N_bases)
        all_data = np.concatenate(self.current_epoch_coeffs, axis=0)
        
        # 计算该 Epoch 的平均权重分布: (N_bases,)
        epoch_avg = np.mean(all_data, axis=0)
        self.training_history.append(epoch_avg)
        
        # 清空缓存，准备下一个 Epoch
        self.current_epoch_coeffs = []

    # =======================================================
    # 可视化功能 1: 静态统计 (验证集整体分析)
    # =======================================================
    def plot_stats(self, title_suffix=""):
        """
        基于当前缓存的数据 (current_epoch_coeffs) 绘制统计图。
        通常在验证集跑完后调用，但在 end_epoch() 之前调用。
        """
        if not self.current_epoch_coeffs:
            print("No data recorded to plot stats.")
            return

        data = np.concatenate(self.current_epoch_coeffs, axis=0)
        n_samples, n_bases = data.shape
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. 平均权重
        avg_weights = np.mean(data, axis=0)
        sns.barplot(x=list(range(n_bases)), y=avg_weights, ax=axes[0], palette="viridis", hue=list(range(n_bases)), legend=False)
        axes[0].set_title(f"Average Basis Weight {title_suffix}")
        
        # 2. Top-1 选择率
        top1_indices = np.argmax(data, axis=1)
        counts = np.bincount(top1_indices, minlength=n_bases)
        freq_pct = counts / n_samples
        sns.barplot(x=list(range(n_bases)), y=freq_pct, ax=axes[1], palette="magma", hue=list(range(n_bases)), legend=False)
        axes[1].set_title(f"Top-1 Selection Frequency {title_suffix}")

        # 3. 样本热力图 (采样前100个)
        vis_samples = min(n_samples, 100)
        sns.heatmap(data[:vis_samples], ax=axes[2], cmap="coolwarm", cbar_kws={'label': 'Weight'})
        axes[2].set_title("Activation Heatmap (First 100 samples)")
        axes[2].set_xlabel("Basis Index")
        axes[2].set_ylabel("Sample Index")

        plt.tight_layout()
        plt.savefig("coba_gcm_stats.png", dpi=300, bbox_inches='tight')
        # plt.show()

    # =======================================================
    # 可视化功能 2: 训练演化 (各 Epoch 的变化)
    # =======================================================
    def plot_evolution(self):
        """
        绘制训练过程中 Basis 选择分布的变化。
        需要你在每个 Epoch 结束时调用 end_epoch()。
        """
        if not self.training_history:
            print("No training history found. Did you call end_epoch()?")
            return
            
        data = np.array(self.training_history) # (Epochs, N_bases)
        epochs, n_bases = data.shape
        
        plt.figure(figsize=(10, 6))
        # 堆叠图
        plt.stackplot(range(epochs), data.T, labels=[f'Base {i}' for i in range(n_bases)], alpha=0.85)
        
        plt.title("Evolution of Basis Utilization over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Average Probability Mass")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.margins(0, 0)
        plt.tight_layout()
        plt.savefig("coba_gcm_evolution.png", dpi=300, bbox_inches='tight')
        # plt.show()

    # =======================================================
    # 可视化功能 3: 时序动态 (Waveform vs Coeffs)
    # =======================================================
    def analyze_sequence(self, sequence, window_len, stride=1):
        """
        独立功能：输入一段长序列，自动进行滑动窗口推理，并画出 波形 vs 系数热力图。
        
        Args:
            sequence: (Total_Len, N_var) 或 (Total_Len,) 的 numpy 数组
            window_len: 模型的时间窗口大小
            stride: 滑动步长 (为了绘图精细度，建议设为 1)
        """
        self.model.eval()
        
        # 1. 数据预处理
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().numpy()
        if sequence.ndim == 1:
            sequence = sequence[:, None] # (T, 1)
            
        T, n_var = sequence.shape
        if T < window_len:
            print("Sequence is shorter than window_len.")
            return

        # 2. 构造滑动窗口输入
        inputs = []
        # 我们只画能产生完整窗口的部分
        valid_steps = (T - window_len) // stride
        
        for i in range(valid_steps):
            seq = sequence[i : i+window_len]
            inputs.append(seq)
            
        input_tensor = torch.tensor(np.array(inputs), dtype=torch.float32) # (B, L, V)
        if next(self.model.parameters()).is_cuda:
            input_tensor = input_tensor.to(global_device)

        # 3. 推理 (分批处理以防显存溢出)
        batch_size = 256
        coeffs_list = []
        
        with torch.no_grad():
            for i in range(0, len(input_tensor), batch_size):
                batch_x = input_tensor[i : i+batch_size]
                _ = self.model(batch_x)
                coeffs_list.append(self.model.coeffs.cpu().numpy())
                
        coeffs = np.concatenate(coeffs_list, axis=0) # (Valid_Steps, N_bases)
        n_bases = coeffs.shape[1]

        # 4. 绘图
        # x轴对应时间点：从 window_len 开始到结束
        time_axis = np.arange(window_len, window_len + len(coeffs) * stride, step=stride)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                       gridspec_kw={'height_ratios': [1.5, 1], 'hspace': 0.1})
        
        # 上图：原始波形 (取第一个变量展示)
        # 截取对应产生 coeff 的那段时间的数据
        plot_data = sequence[window_len : window_len + len(coeffs)*stride : stride, 0]
        
        ax1.plot(time_axis, plot_data, color='#333333', lw=1.5, label='Input Series')
        ax1.set_ylabel("Value")
        ax1.set_title("Input Sequence Dynamics")
        ax1.grid(True, alpha=0.2)
        ax1.legend()

        # 下图：系数热力图
        # imshow 需要 (N_bases, Time)，所以转置
        im = ax2.imshow(coeffs.T, aspect='auto', cmap='viridis', interpolation='nearest',
                        extent=[time_axis[0], time_axis[-1], 0, n_bases], origin='lower')
        
        ax2.set_ylabel("Basis Index")
        ax2.set_xlabel("Time Step")
        ax2.set_yticks(np.arange(n_bases) + 0.5)
        ax2.set_yticklabels([f'B{i}' for i in range(n_bases)])
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.25, fraction=0.05)
        cbar.set_label('Basis Activation Probability')
        
        plt.suptitle("CoBA-GCM: Basis Adaptation over Time", y=0.95, fontsize=14)
        plt.savefig("coba_gcm_sequence_analysis.png", dpi=300, bbox_inches='tight')
        # plt.show()

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

    def output_calibration(self, outputs, inputs=None):
        if inputs is not None and self.out_cali is not None:
            outputs = self.out_cali(outputs, inputs)
            return outputs
        if self.out_cali is not None:
            return self.out_cali(outputs)
        return outputs


class CoBA_low_rank_FreqAdapter(nn.Module):
    def __init__(self, window_len, n_var=1, low_ranks=64, hidden_dim=32,
                 gating_init=0.01, var_wise=True,
                 n_bases=8, feature_dim=32, query_type='freq-base-CI'):
        super(CoBA_low_rank_FreqAdapter, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        self.n_bases = n_bases
        self.feature_dim = feature_dim
        self.online_mode = False
        self.analyzer = CoBA_Analyzer(self)
        self.rank = low_ranks
        if var_wise:
            self.bases_left = nn.Parameter(torch.Tensor(n_bases, window_len, self.rank, n_var))
            self.bases_right = nn.Parameter(torch.Tensor(n_bases, self.rank, window_len, n_var))
            self.codebook_keys = nn.Parameter(torch.randn(n_var, n_bases, feature_dim))
        else:
            self.bases_left = nn.Parameter(torch.Tensor(n_bases, window_len, self.rank))
            self.bases_right = nn.Parameter(torch.Tensor(n_bases, self.rank, window_len))
            self.codebook_keys = nn.Parameter(torch.randn(n_bases, feature_dim))
        # Initialize bases_left with column-wise orthogonality
        with torch.no_grad():
            if var_wise:
                for n in range(n_bases):
                    for v in range(n_var):
                        nn.init.orthogonal_(self.bases_left[n, :, :, v])
            else:
                for n in range(n_bases):
                    nn.init.orthogonal_(self.bases_left[n, :, :])
        
        # nn.init.kaiming_uniform_(self.bases_left, a=math.sqrt(5))
        nn.init.zeros_(self.bases_right)
        
        # --- Query Net Selection Logic (Factory) ---
        print(f"Initializing CoBA with Query Type: {query_type}")
        if query_type == 'time':
            self.query_net = QueryNet_Time(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-CI':
            self.query_net = QueryNet_Freq_Base_ChannelIndependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-CD':
            self.query_net = QueryNet_Freq_Base_ChannelDependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-hybrid':
            self.query_net = QueryNet_Freq_Hybrid(window_len, n_var, feature_dim)
        elif query_type == 'fusion':
            self.query_net = QueryNet_Fusion_Gated(window_len, n_var, feature_dim)
        elif query_type == 'multiscale':
            self.query_net = QueryNet_MultiScale(window_len, n_var, feature_dim)
        elif query_type == 'phase':
            self.query_net = QueryNet_Phase(window_len, n_var, feature_dim)
        elif query_type == 'freq-attn':
            self.query_net = QueryNet_Freq_Attn(window_len, n_var, feature_dim)
        elif query_type == 'freq-light':
            self.query_net = QueryNet_Freq_Light(window_len, n_var, feature_dim)
        elif query_type == 'wave-ms':
            self.query_net = QueryNet_Wavelet_MS(window_len, n_var, feature_dim)
        else:
            raise ValueError(f"Unknown query_type: {query_type}")

        # Gating parameter
        self.tafas_gating = nn.Parameter(gating_init * torch.ones(n_var))

        # Frequency Domain Adapter Parameters (per variable)
        self.freq_len = window_len // 2 + 1
        self.scale = 1e-5
        self.sparsity_threshold = 0.01
        
        # Parameters for real and imaginary parts
        # Dim: (1, freq_len, n_var) for broadcasting over batch
        # Element-wise multiplication
        self.freq_r = nn.Parameter(self.scale * torch.randn(1, self.freq_len, n_var))
        self.freq_i = nn.Parameter(self.scale * torch.randn(1, self.freq_len, n_var))
        
        # Bias 保持向量形式 (1, freq_len, n_var) 用于广播
        self.freq_rb = nn.Parameter(torch.zeros(1, self.freq_len, n_var))
        self.freq_ib = nn.Parameter(torch.zeros(1, self.freq_len, n_var))

    def _get_query(self, x):
        return self.query_net(x)

    def forward(self, x):
        """
        x shape: (Batch, Window_len, N_var)
        """
        batch_size = x.size(0)

        query = self._get_query(x) # (B, N_vars, D) 
        query_norm = F.normalize(query, p=2, dim=-1)           # (B, N_vars, D)
        keys_norm = F.normalize(self.codebook_keys, p=2, dim=-1) # (N_vars, N_bases, D)
        # print(keys_norm.shape)
        # similarity = torch.matmul(query_norm, keys_norm.T) # (B, N_vars, N_bases)
        similarity = torch.matmul(
            query_norm.unsqueeze(2),         # (B, N_vars, 1, D)
            keys_norm.transpose(1, 2)        # (N_vars, D, N_bases)
        ).squeeze(2)                          # (B, N_vars, N_bases)
        # print(similarity.shape)
        
        # --- 替换为 Top-K 逻辑 ---
        k = 2 
        
        # 1. 找出分数最高的 k 个值的索引和数值
        topk_vals, topk_indices = torch.topk(similarity, k=k, dim=-1)
        
        # 2. 创建一个全为 -inf 的 mask (这样 Softmax 后会变成 0)
        mask = torch.full_like(similarity, float('-inf'))
        
        # 3. 将 top-k 的位置填回原始的相似度数值
        # 在 mask 的 dim=-1 维度，按照 topk_indices 的索引，填入 topk_vals
        mask.scatter_(-1, topk_indices, topk_vals)
        
        # 4. 再做 Softmax
        coeffs = F.softmax(mask, dim=-1) # (B, N_vars, N_bases)

        if self.var_wise:
            # # U: (Batch, L_out, Rank, Var) <- 聚合后的 bases_left
            # u = torch.einsum('bn, nliv -> bliv', coeffs, self.bases_left)   
            # # V: (Batch, Rank, L_in, Var)  <- 聚合后的 bases_right
            # v = torch.einsum('bn, nriv -> briv', coeffs, self.bases_right)  
            u = torch.einsum('bvn, nlrv -> blrv', coeffs, self.bases_left)
            v = torch.einsum('bvn, nrlv -> brlv', coeffs, self.bases_right)
            
            # --- 关键优化开始 ---
            # 原始 x: (B, L_in, V)
            # Step 1: x(biv) * v(briv) -> (Rank)
            # indices: batch(b), input_len(i), var(v) AND batch(b), rank(r), input_len(i), var(v)
            # result shape: (Batch, Rank, Var)
            x_reduced = torch.einsum('biv, briv -> brv', x, v)
            
            # Step 2: intermediate(brv) * u(blrv) -> (Output_len)
            # indices: batch(b), rank(r), var(v) AND batch(b), output_len(l), rank(r), var(v)
            # result shape: (Batch, Output_len, Var)
            feat_trans = torch.einsum('brv, blrv -> blv', x_reduced, u)
            
            # # 加上 bias
            # feat_trans = feat_trans + self.bias
        else:
            u = torch.einsum('bn, nli -> bli', coeffs, self.bases_left)   # (B, L, R)
            v = torch.einsum('bn, nri -> bri', coeffs, self.bases_right)  # (B, R, L)
            
            # Step 1: Project to low rank
            # x: (B, L, V), v: (B, R, L)
            # output: (B, R, V)
            x_reduced = torch.einsum('blv, bri -> brv', x, v)
            
            # Step 2: Project back to high rank
            # x_reduced: (B, R, V), u: (B, L, R)
            # output: (B, L, V)
            feat_trans = torch.einsum('brv, bli -> blv', x_reduced, u)
            
            # feat_trans = feat_trans + self.bias

        if self.online_mode:
            # Frequency Domain Adaptation (similar to ComplexFreqAdapter)
            B, L, D = x.shape
            
            # FFT with ortho norm (Energy preserving)
            x_fft = torch.fft.rfft(x, dim=1, norm='ortho')  # (B, F, D)

            # Linear Complex Transform (element-wise)
            # Delta_real = R*r - I*i + rb
            # Delta_imag = I*r + R*i + ib
            delta_real = (
                x_fft.real * self.freq_r - x_fft.imag * self.freq_i + self.freq_rb
            )
            delta_imag = (
                x_fft.imag * self.freq_r + x_fft.real * self.freq_i + self.freq_ib
            )
            
            # Combine and softshrink (Sparsity on Residual)
            y_stack = torch.stack([delta_real, delta_imag], dim=-1)
            # y_stack = F.softshrink(y_stack, lambd=self.sparsity_threshold)
            y = torch.view_as_complex(y_stack)
            
            # iFFT with ortho norm
            output_raw = torch.fft.irfft(y, n=L, dim=1, norm='ortho')
            
            # Gating
            tafas_output = torch.tanh(self.tafas_gating) * output_raw

            out = x + feat_trans + tafas_output
        else:
            out = x + feat_trans
        
        self.coeffs = coeffs

        return out

    def get_optim_params(self):
        params = []
        if self.online_mode:
            params.append(self.freq_r)
            params.append(self.freq_i)
            params.append(self.freq_rb)
            params.append(self.freq_ib)
            params.append(self.tafas_gating)
            # params.extend(list(self.query_net.parameters()))
        else:
            pass
        return params


class CoBA_FreqDomain_GCM(nn.Module):
    """
    CoBA Frequency Domain GCM
    
    1. Main Path (Codebook): 
       Input -> FFT -> Query -> Select Freq Domain Bases (Complex Low-Rank) -> Reconstruction -> iFFT -> Residual
    
    2. Online Path (Test-time Adaptation):
       Input -> FFT -> Per-Variable Full Freq Matrix Transform -> iFFT -> Gated Residual
    """
    def __init__(self, window_len, n_var=1, low_ranks=64, hidden_dim=32,
                 gating_init=0.01, var_wise=True,
                 n_bases=8, feature_dim=32, query_type='freq-base-CI'):
        super(CoBA_FreqDomain_GCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        self.n_bases = n_bases
        self.feature_dim = feature_dim
        self.online_mode = False
        self.analyzer = CoBA_Analyzer(self)
        self.freq_len = window_len // 2 + 1
        
        # Rank needs to be compatible with frequency length if we want compression
        # Ideally low_ranks should be smaller than freq_len for bottleneck effect
        self.rank = min(low_ranks, self.freq_len) 

        # --- 1. Codebook / Bases in Frequency Domain ---
        self.codebook_keys = nn.Parameter(torch.randn(n_var, n_bases, feature_dim))
        
        # Freq Bases: Split into Real and Imaginary parts to handle Complex operations
        # Structure: Low Rank Bottleneck in Frequency Domain
        # Down: Freq_len -> Rank
        # Up:   Rank -> Freq_len
        
        if var_wise:
            # Down Projection Bases: (N_bases, Freq_in, Rank, N_var)
            self.bases_left_r = nn.Parameter(torch.Tensor(n_bases, self.freq_len, self.rank, n_var))
            self.bases_left_i = nn.Parameter(torch.Tensor(n_bases, self.freq_len, self.rank, n_var))
            
            # Up Projection Bases: (N_bases, Rank, Freq_out, N_var)
            self.bases_right_r = nn.Parameter(torch.Tensor(n_bases, self.rank, self.freq_len, n_var))
            self.bases_right_i = nn.Parameter(torch.Tensor(n_bases, self.rank, self.freq_len, n_var))
        else:
            # Shared across variables
            self.bases_left_r = nn.Parameter(torch.Tensor(n_bases, self.freq_len, self.rank))
            self.bases_left_i = nn.Parameter(torch.Tensor(n_bases, self.freq_len, self.rank))
            self.bases_right_r = nn.Parameter(torch.Tensor(n_bases, self.rank, self.freq_len))
            self.bases_right_i = nn.Parameter(torch.Tensor(n_bases, self.rank, self.freq_len))

        # Initialization
        self._init_bases()
        
        # Bias in Time Domain (Post-iFFT)
        self.bias_time = nn.Parameter(torch.zeros(window_len, n_var))

        # --- Query Net Selection Logic (Factory) ---
        print(f"Initializing FV-CoBA (Freq-View) with Query Type: {query_type}")
        if query_type == 'time':
            self.query_net = QueryNet_Time(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-CI':
            self.query_net = QueryNet_Freq_Base_ChannelIndependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-CD':
            self.query_net = QueryNet_Freq_Base_ChannelDependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-hybrid':
            self.query_net = QueryNet_Freq_Hybrid(window_len, n_var, feature_dim)
        elif query_type == 'freq-separate-CI':
            self.query_net = QueryNet_Freq_Separate_ChannelIndependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-mag-phase':
            self.query_net = QueryNet_Freq_MagPhase(window_len, n_var, feature_dim)
        else:
            # Default fallback
            print(f"Unknown query_type: {query_type}, defaulting to 'freq-base-CI'")
            self.query_net = QueryNet_Freq_Base_ChannelIndependence(window_len, n_var, feature_dim)

        # --- 2. Online Mode Parameters (Matrix Freq Adapter) ---
        self.tafas_gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.scale = 1e-5
        self.sparsity_threshold = 0.01

        # Online Matrix Parameters: (N_var, Freq_len, Freq_len) -> Modified to Element-wise (1, Freq_len, N_var)
        self.online_freq_r = nn.Parameter(self.scale * torch.randn(1, self.freq_len, n_var))
        self.online_freq_i = nn.Parameter(self.scale * torch.randn(1, self.freq_len, n_var))
        
        self.online_bias_r = nn.Parameter(torch.zeros(1, self.freq_len, n_var))
        self.online_bias_i = nn.Parameter(torch.zeros(1, self.freq_len, n_var))

    def _init_bases(self):
        # Orthogonal init for better spectral properties
        with torch.no_grad():
            if self.var_wise:
                for n in range(self.n_bases):
                    for v in range(self.n_var):
                        nn.init.orthogonal_(self.bases_left_r[n, :, :, v])
                        nn.init.orthogonal_(self.bases_left_i[n, :, :, v])
            else:
                for n in range(self.n_bases):
                    nn.init.orthogonal_(self.bases_left_r[n])
                    nn.init.orthogonal_(self.bases_left_i[n])
            # Zero init for Up projection to start with identity-like behavior or zero residual
            nn.init.zeros_(self.bases_right_r)
            nn.init.zeros_(self.bases_right_i)

    def _get_query(self, x):
        return self.query_net(x)
    
    def complex_low_rank_forward(self, x_fft, coeffs):
        """
        Perform complex low-rank transformation using aggregated bases.
        x_fft: (B, F, V) - Complex
        coeffs: (B, V, N)
        """
        # 1. Aggregate Bases based on coefficients
        # Expected aggregated shape needed for einsum: 
        # Down: (B, Freq_in, Rank, V)
        # Up:   (B, Rank, Freq_out, V)
        
        if self.var_wise:
            # coeffs: bvn, bases: nfrv -> bfrv
            w_left_r = torch.einsum('bvn, nfrv -> bfrv', coeffs, self.bases_left_r)
            w_left_i = torch.einsum('bvn, nfrv -> bfrv', coeffs, self.bases_left_i)
            
            # coeffs: bvn, bases: nrfv -> brfv
            w_right_r = torch.einsum('bvn, nrfv -> brfv', coeffs, self.bases_right_r)
            w_right_i = torch.einsum('bvn, nrfv -> brfv', coeffs, self.bases_right_i)
        else:
             # coeffs: bn (mean over V internally or broadcasted), bases: nfr -> bfr
             # Handle simplest case where coeffs might be (B, N)
             pass 
             # (Omitting non-var-wise complex logic for brevity, assuming var-wise=True per prompts)

        # 2. Complex Matrix Multiplication Stage 1 (Left Projection)
        # X (B, F, V) @ W_left (B, F, R, V) -> Z (B, R, V)
        # Note: This is an element-wise matrix mul per batch/var structure
        # Indices: b=batch, f=freq_in, r=rank, v=var
        xr, xi = x_fft.real, x_fft.imag # (B, F, V)

        # Z_real = Xr * Wr - Xi * Wi
        # Z_imag = Xr * Wi + Xi * Wr
        # Einstein sum: bfv, bfrv -> brv
        z_r = torch.einsum('bfv, bfrv -> brv', xr, w_left_r) - \
              torch.einsum('bfv, bfrv -> brv', xi, w_left_i)
        z_i = torch.einsum('bfv, bfrv -> brv', xr, w_left_i) + \
              torch.einsum('bfv, bfrv -> brv', xi, w_left_r)
        
        # 3. Complex Matrix Multiplication Stage 2 (Right Projection)
        # Z (B, R, V) @ W_right (B, R, F, V) -> Y (B, F, V)
        # Indices: b=batch, r=rank, f=freq_out, v=var
        
        y_r = torch.einsum('brv, brfv -> bfv', z_r, w_right_r) - \
              torch.einsum('brv, brfv -> bfv', z_i, w_right_i)
        y_i = torch.einsum('brv, brfv -> bfv', z_r, w_right_i) + \
              torch.einsum('brv, brfv -> bfv', z_i, w_right_r)
              
        return torch.complex(y_r, y_i)

    def forward(self, x):
        """
        x shape: (Batch, Window_len, N_var)
        """
        B, L, _ = x.shape

        # 1. Transform to Frequency Domain
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')  # (B, F, V)

        # 2. Query & Codebook Selection
        query = self._get_query(x) 
        query_norm = F.normalize(query, p=2, dim=-1)
        keys_norm = F.normalize(self.codebook_keys, p=2, dim=-1)
        
        similarity = torch.matmul(
            query_norm.unsqueeze(2),         # (B, V, 1, D)
            keys_norm.transpose(1, 2)        # (V, D, N)
        ).squeeze(2)                         # (B, V, N)
        
        # Top-K Softmax Logic
        k = 2 
        topk_vals, topk_indices = torch.topk(similarity, k=k, dim=-1)
        mask = torch.full_like(similarity, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_vals)
        coeffs = F.softmax(mask, dim=-1) # (B, V, N)
        self.coeffs = coeffs

        # 3. Main Path: Codebook-based Frequency Adaptation (Low Rank)
        # Output is complex residual in freq domain
        delta_fft_codebook = self.complex_low_rank_forward(x_fft, coeffs)
        
        # iFFT for Main Path
        delta_time_codebook = torch.fft.irfft(delta_fft_codebook, n=L, dim=1, norm='ortho')
        delta_time_codebook = delta_time_codebook + self.bias_time

        # 4. Online Path: Matrix-based Frequency Calibration
        if self.online_mode:
            # Modified to match CoBA_low_rank_FreqAdapter calculation logic (Element-wise)
            delta_real_online = (
                x_fft.real * self.online_freq_r - x_fft.imag * self.online_freq_i + self.online_bias_r
            )
            delta_imag_online = (
                x_fft.imag * self.online_freq_r + x_fft.real * self.online_freq_i + self.online_bias_i
            )
            
            # Softshrink (Sparsity)
            y_stack = torch.stack([delta_real_online, delta_imag_online], dim=-1)
            # y_stack = F.softshrink(y_stack, lambd=self.sparsity_threshold)
            y_online = torch.view_as_complex(y_stack)
            
            # iFFT Online
            delta_time_online = torch.fft.irfft(y_online, n=L, dim=1, norm='ortho')
            
            # Gating
            delta_time_online = torch.tanh(self.tafas_gating) * delta_time_online

            out = x + delta_time_codebook + delta_time_online
        else:
            out = x + delta_time_codebook

        return out

    def get_optim_params(self):
        params = []
        if self.online_mode:
            # Only update the Online Matrix Parameters and Gating
            params.append(self.online_freq_r)
            params.append(self.online_freq_i)
            params.append(self.online_bias_r)
            params.append(self.online_bias_i)
            params.append(self.tafas_gating)
            
            # Verify if query net or codebook should be frozen (Usually yes for TTA)
            # To be safe, we also include query net if we want inputs to adapt
            # params.extend(list(self.query_net.parameters())) 
        else:
            # Codebook training params not needed here as this is usually called by the TTA optimizer
            params.append(self.tafas_gating) 
        return params


class CoBA_FreqDomain_ElementWise_GCM(nn.Module):
    """
    CoBA Frequency Domain Element-Wise GCM
    
    1. Main Path (Codebook): 
       Input -> FFT -> Query -> Select Freq Domain Bases (Element-Wise) -> iFFT -> Residual
       - Replaces low-rank matrix multiplication with element-wise multiplication.
       - Bases are defined directly in frequency domain with shape matching the FFT features.
    
    2. Online Path (Test-time Adaptation):
       Input -> FFT -> Per-Variable Element-Wise Freq Transform -> iFFT -> Gated Residual
    """
    def __init__(self, window_len, n_var=1, hidden_dim=32,
                 gating_init=0.01, var_wise=True,
                 n_bases=8, feature_dim=32, query_type='freq-base-CI', **kwargs):
        super(CoBA_FreqDomain_ElementWise_GCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        self.n_bases = n_bases
        self.feature_dim = feature_dim
        self.online_mode = False
        self.analyzer = CoBA_Analyzer(self)
        self.freq_len = window_len // 2 + 1
        
        # --- 1. Codebook / Bases in Frequency Domain (Element-wise) ---
        self.codebook_keys = nn.Parameter(torch.randn(n_var, n_bases, feature_dim))
        
        if var_wise:
            # Bases: (N_bases, Freq_len, N_var)
            # Two sets for Real and Imaginary parts of the frequency filter
            self.bases_r = nn.Parameter(torch.Tensor(n_bases, self.freq_len, n_var))
            self.bases_i = nn.Parameter(torch.Tensor(n_bases, self.freq_len, n_var))
        else:
            # Bases: (N_bases, Freq_len)
            self.bases_r = nn.Parameter(torch.Tensor(n_bases, self.freq_len))
            self.bases_i = nn.Parameter(torch.Tensor(n_bases, self.freq_len))

        # Initialization
        self._init_bases()
        
        # # Bias in Time Domain (Post-iFFT)
        # self.bias_time = nn.Parameter(torch.zeros(window_len, n_var))

        # --- Query Net Selection Logic (Factory) ---
        print(f"Initializing FV-CoBA (Element-Wise) with Query Type: {query_type}")
        if query_type == 'time':
            self.query_net = QueryNet_Time(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-CI':
            self.query_net = QueryNet_Freq_Base_ChannelIndependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-CD':
            self.query_net = QueryNet_Freq_Base_ChannelDependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-separate-CI':
            self.query_net = QueryNet_Freq_Separate_ChannelIndependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-mag-phase':
            self.query_net = QueryNet_Freq_MagPhase(window_len, n_var, feature_dim)
        elif query_type == 'freq-norm-CI':
            self.query_net = QueryNet_Freq_Norm_ChannelIndependence(window_len, n_var, feature_dim)
        else:
            # Default fallback
            print(f"Unknown query_type: {query_type}, defaulting to 'freq-base-CI'")
            self.query_net = QueryNet_Freq_Base_ChannelIndependence(window_len, n_var, feature_dim)

        # --- 2. Online Mode Parameters (Element-wise Freq Adapter) ---
        self.tafas_gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.scale = 1e-5
        
        # Element-wise Parameters: (1, Freq_len, N_var)
        self.online_freq_r = nn.Parameter(self.scale * torch.randn(1, self.freq_len, n_var))
        self.online_freq_i = nn.Parameter(self.scale * torch.randn(1, self.freq_len, n_var))
        
        self.online_bias_r = nn.Parameter(torch.zeros(1, self.freq_len, n_var))
        self.online_bias_i = nn.Parameter(torch.zeros(1, self.freq_len, n_var))

    def _init_bases(self):
        # Initialize bases_r and bases_i
        # Using Xavier uniform for filter weights
        # nn.init.xavier_uniform_(self.bases_r)
        # nn.init.xavier_uniform_(self.bases_i)
        # nn.init.kaiming_normal_(self.bases_r)
        # nn.init.kaiming_normal_(self.bases_i)
    
        # 目标：对于每一个 variable (v in n_var)，
        # 使其对应的 n_bases 个向量 (length = freq_len) 相互正交
        
        with torch.no_grad():
            if self.var_wise:
                # 维度: (n_bases, freq_len, n_var)
                for v in range(self.n_var):
                    # 1. 处理实部 bases_r
                    # 构造一个 (n_bases, freq_len) 的临时矩阵进行正交化
                    # 注意：为了能正交，通常要求 freq_len >= n_bases
                    init_matrix_r = torch.empty(self.n_bases, self.freq_len)
                    nn.init.orthogonal_(init_matrix_r)
                    self.bases_r.data[:, :, v] = init_matrix_r

                    # 2. 处理虚部 bases_i
                    init_matrix_i = torch.empty(self.n_bases, self.freq_len)
                    nn.init.orthogonal_(init_matrix_i)
                    self.bases_i.data[:, :, v] = init_matrix_i
            else:
                # 维度: (n_bases, freq_len) - 只有一组
                nn.init.orthogonal_(self.bases_r)
                nn.init.orthogonal_(self.bases_i)

    def _get_query(self, x):
        return self.query_net(x)
    
    def complex_element_wise_forward(self, x_fft, coeffs):
        """
        Perform complex element-wise transformation using aggregated bases.
        x_fft: (B, F, V) - Complex
        coeffs: (B, V, N)
        """
        # 1. Aggregate Bases based on coefficients
        if self.var_wise:
            # coeffs: bvn, bases: nfv -> bfv
            w_r = torch.einsum('bvn, nfv -> bfv', coeffs, self.bases_r)
            w_i = torch.einsum('bvn, nfv -> bfv', coeffs, self.bases_i)
        else:
             # bases: nf -> bnf (broadcast) -> weighted sum -> bf
             # coeffs: bvn -> collapse V? Or broadcast bases to V?
             # Assuming we share bases across Var but select per Var (coeffs are bvn)
             w_r = torch.einsum('bvn, nf -> bfv', coeffs, self.bases_r)
             w_i = torch.einsum('bvn, nf -> bfv', coeffs, self.bases_i)

        # 2. Complex Element-Wise Multiplication
        # Z = X * W
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        xr, xi = x_fft.real, x_fft.imag # (B, F, V)

        # z_real = xr * wr - xi * wi
        # z_imag = xr * wi + xi * wr
        z_r = xr * w_r - xi * w_i
        z_i = xr * w_i + xi * w_r
              
        return torch.complex(z_r, z_i)

    def forward(self, x):
        """
        x shape: (Batch, Window_len, N_var)
        """
        B, L, _ = x.shape

        # 1. Transform to Frequency Domain
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')  # (B, F, V)

        # 2. Query & Codebook Selection
        query = self._get_query(x) 
        query_norm = F.normalize(query, p=2, dim=-1)
        keys_norm = F.normalize(self.codebook_keys, p=2, dim=-1)
        
        similarity = torch.matmul(
            query_norm.unsqueeze(2),         # (B, V, 1, D)
            keys_norm.transpose(1, 2)        # (V, D, N)
        ).squeeze(2)                         # (B, V, N)
        
        # Top-K Softmax Logic
        k = 2 
        topk_vals, topk_indices = torch.topk(similarity, k=k, dim=-1)
        mask = torch.full_like(similarity, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_vals)
        coeffs = F.softmax(mask, dim=-1) # (B, V, N)
        self.coeffs = coeffs

        # 3. Main Path: Codebook-based Frequency Adaptation (Element-Wise)
        delta_fft_codebook = self.complex_element_wise_forward(x_fft, coeffs)
        
        # iFFT for Main Path
        delta_time_codebook = torch.fft.irfft(delta_fft_codebook, n=L, dim=1, norm='ortho')
        # delta_time_codebook = delta_time_codebook + self.bias_time

        # 4. Online Path: Element-wise Frequency Calibration
        if self.online_mode:
            delta_real_online = (
                x_fft.real * self.online_freq_r - x_fft.imag * self.online_freq_i + self.online_bias_r
            )
            delta_imag_online = (
                x_fft.imag * self.online_freq_r + x_fft.real * self.online_freq_i + self.online_bias_i
            )
            
            y_online = torch.complex(delta_real_online, delta_imag_online)
            
            # iFFT Online
            delta_time_online = torch.fft.irfft(y_online, n=L, dim=1, norm='ortho')
            
            # Gating
            delta_time_online = torch.tanh(self.tafas_gating) * delta_time_online

            out = x + delta_time_codebook + delta_time_online
        else:
            out = x + delta_time_codebook

        return out

    def get_optim_params(self):
        params = []
        if self.online_mode:
            params.append(self.online_freq_r)
            params.append(self.online_freq_i)
            params.append(self.online_bias_r)
            params.append(self.online_bias_i)
            params.append(self.tafas_gating)
            # params.extend(list(self.query_net.parameters()))
            # return params, self.query_net.parameters()
        else:
            params.append(self.tafas_gating)
        return params


class RoCoBA_FreqDomain_GCM(nn.Module):
    """
    
    1. Main Path (Codebook): 
       Input -> FFT -> Query -> Select Freq Domain Bases (Element-Wise) -> iFFT -> Residual
       - Replaces low-rank matrix multiplication with element-wise multiplication.
       - Bases are defined directly in frequency domain with shape matching the FFT features.
    
    2. Online Path (Test-time Adaptation):
       Input -> FFT -> Per-Variable Element-Wise Freq Transform -> iFFT -> Gated Residual
    """
    def __init__(self, window_len, n_var=1, hidden_dim=32,
                 gating_init=0.01, var_wise=True,
                 n_bases=8, feature_dim=32, query_type='freq-base-CI', 
                 confidence_threshold=0.5, **kwargs):
        super(RoCoBA_FreqDomain_GCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        self.n_bases = n_bases
        self.feature_dim = feature_dim
        self.online_mode = False
        self.analyzer = CoBA_Analyzer(self)
        self.freq_len = window_len // 2 + 1
        
        # [Add]: Confidence Threshold (tau) for Knowledge Retrieval
        self.confidence_threshold = confidence_threshold
        
        # --- 1. Codebook / Bases in Frequency Domain (Element-wise) ---
        self.codebook_keys = nn.Parameter(torch.randn(n_var, n_bases, feature_dim))
        
        if var_wise:
            # Bases: (N_bases, Freq_len, N_var)
            # Two sets for Real and Imaginary parts of the frequency filter
            self.bases_r = nn.Parameter(torch.Tensor(n_bases, self.freq_len, n_var))
            self.bases_i = nn.Parameter(torch.Tensor(n_bases, self.freq_len, n_var))
        else:
            # Bases: (N_bases, Freq_len)
            self.bases_r = nn.Parameter(torch.Tensor(n_bases, self.freq_len))
            self.bases_i = nn.Parameter(torch.Tensor(n_bases, self.freq_len))

        # Initialization
        self.scale = 1e-5
        self._init_bases()
        
        # --- Query Net Selection Logic (Factory) ---
        print(f"Initializing FV-CoBA (Element-Wise) with Query Type: {query_type}")
        if query_type == 'time':
            self.query_net = QueryNet_Time(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-CI':
            self.query_net = QueryNet_Freq_Base_ChannelIndependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-CD':
            self.query_net = QueryNet_Freq_Base_ChannelDependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-separate-CI':
            self.query_net = QueryNet_Freq_Separate_ChannelIndependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-mag-phase':
            self.query_net = QueryNet_Freq_MagPhase(window_len, n_var, feature_dim)
        elif query_type == 'freq-norm-CI':
            self.query_net = QueryNet_Freq_Norm_ChannelIndependence(window_len, n_var, feature_dim)
        else:
            # Default fallback
            print(f"Unknown query_type: {query_type}, defaulting to 'freq-base-CI'")
            self.query_net = QueryNet_Freq_Base_ChannelIndependence(window_len, n_var, feature_dim)

        # --- 3. Online Mode Parameters ---
        self.tafas_gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.online_freq_r = nn.Parameter(self.scale * torch.randn(1, self.freq_len, n_var))
        self.online_freq_i = nn.Parameter(self.scale * torch.randn(1, self.freq_len, n_var))
        self.online_bias_r = nn.Parameter(torch.zeros(1, self.freq_len, n_var))
        self.online_bias_i = nn.Parameter(torch.zeros(1, self.freq_len, n_var))

    def _init_bases(self):
        with torch.no_grad():
            if self.var_wise:
                # 针对每个变量 (Channel) 独立进行正交化
                for v in range(self.n_var):
                    # 1. 创建联合矩阵: shape (N_bases, 2 * Freq_len)
                    # 2*Freq_len 意味着我们在 2F 维空间寻找 N 个正交方向
                    joint_bases = torch.empty(self.n_bases, 2 * self.freq_len)
                    
                    # 2. 对联合矩阵进行正交初始化
                    nn.init.orthogonal_(joint_bases)
                    
                    # 3. 拆分回实部和虚部
                    # split_chunk_size = self.freq_len
                    bases_r_chunk, bases_i_chunk = torch.split(joint_bases, self.freq_len, dim=1)
                    
                    # 4. 赋值并缩放
                    self.bases_r.data[:, :, v] = bases_r_chunk * self.scale
                    self.bases_i.data[:, :, v] = bases_i_chunk * self.scale
                    
            else:
                # 同样逻辑，应用于非 var_wise 情况
                joint_bases = torch.empty(self.n_bases, 2 * self.freq_len)
                nn.init.orthogonal_(joint_bases)
                
                bases_r_chunk, bases_i_chunk = torch.split(joint_bases, self.freq_len, dim=1)
                
                self.bases_r.data.copy_(bases_r_chunk * self.scale)
                self.bases_i.data.copy_(bases_i_chunk * self.scale)


    def _get_query(self, x):
        return self.query_net(x)
    
    def complex_element_wise_forward(self, x_fft, coeffs):
        """
        Perform complex element-wise transformation using aggregated bases.
        x_fft: (B, F, V) - Complex
        coeffs: (B, V, N)
        """
        if self.var_wise:
            w_r = torch.einsum('bvn, nfv -> bfv', coeffs, self.bases_r)
            w_i = torch.einsum('bvn, nfv -> bfv', coeffs, self.bases_i)
        else:
            w_r = torch.einsum('bvn, nf -> bfv', coeffs, self.bases_r)
            w_i = torch.einsum('bvn, nf -> bfv', coeffs, self.bases_i)

        xr, xi = x_fft.real, x_fft.imag # (B, F, V)
        z_r = xr * w_r - xi * w_i
        z_i = xr * w_i + xi * w_r
              
        return torch.complex(z_r, z_i)

    def forward(self, x):
        """
        x shape: (Batch, Window_len, N_var)
        """
        B, L, _ = x.shape

        # 1. Transform to Frequency Domain
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')  # (B, F, V)

        # 2. Query & Codebook Selection
        query = self._get_query(x) 
        query_norm = F.normalize(query, p=2, dim=-1)
        keys_norm = F.normalize(self.codebook_keys, p=2, dim=-1)
        
        # Calculate Cosine Similarity
        similarity = torch.matmul(
            query_norm.unsqueeze(2),         # (B, V, 1, D)
            keys_norm.transpose(1, 2)        # (V, D, N)
        ).squeeze(2)                         # (B, V, N)
        
        # [Add]: Logic for Confidence Gating (Equation 1)
        # Calculate s_max for each sample and variable
        s_max, _ = torch.max(similarity, dim=-1) # (B, V)
        s_max = F.relu(s_max)
        
        # gamma = 1 if s_max >= tau else s_max / tau
        # Note: s_max is cosine similarity. 
        gamma = torch.where(
            s_max >= self.confidence_threshold, 
            torch.ones_like(s_max), 
            s_max / self.confidence_threshold
        )
        # Reshape gamma for broadcasting: (B, V) -> (B, 1, V) to match time domain output
        gamma = gamma.unsqueeze(1)

        # Top-K Softmax Logic
        k = 2 
        topk_vals, topk_indices = torch.topk(similarity, k=k, dim=-1)
        mask = torch.full_like(similarity, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_vals)
        coeffs = F.softmax(mask, dim=-1) # (B, V, N)
        self.coeffs = coeffs

        # 3. Main Path: Codebook-based Frequency Adaptation (Element-Wise)
        delta_fft_codebook = self.complex_element_wise_forward(x_fft, coeffs)
        
        # iFFT for Main Path
        delta_time_codebook = torch.fft.irfft(delta_fft_codebook, n=L, dim=1, norm='ortho')
        
        # [Mod]: Apply Confidence Gating (Equation 2)
        # Y_k = Y_b + gamma * Y_c (Here out_codebook represents Y_k)
        # x is Y_b, delta_time_codebook is Y_c
        # We apply gamma scaling to the retrieved residual
        delta_time_codebook = delta_time_codebook * gamma

        # 4. Online Path: Element-wise Frequency Calibration
        if self.online_mode:
            delta_real_online = (
                x_fft.real * self.online_freq_r - x_fft.imag * self.online_freq_i + self.online_bias_r
            )
            delta_imag_online = (
                x_fft.imag * self.online_freq_r + x_fft.real * self.online_freq_i + self.online_bias_i
            )
            
            y_online = torch.complex(delta_real_online, delta_imag_online)
            
            # iFFT Online (Y_a in paper)
            delta_time_online = torch.fft.irfft(y_online, n=L, dim=1, norm='ortho')
            
            # [Mod]: Gating with tanh(lambda) (Equation 4)
            # The paper specifies: Y_f = Y_k + tanh(lambda) * Y_a
            delta_time_online = torch.tanh(self.tafas_gating) * delta_time_online

            out = x + delta_time_codebook + delta_time_online
        else:
            out = x + delta_time_codebook

        return out

    def get_optim_params(self):
        params = []
        if self.online_mode:
            params.append(self.online_freq_r)
            params.append(self.online_freq_i)
            params.append(self.online_bias_r)
            params.append(self.online_bias_i)
            params.append(self.tafas_gating)
            # params.extend(list(self.query_net.parameters()))
            # params.append(self.codebook_keys)
            # params.append(self.bases_r)
            # params.append(self.bases_i)
        else:
            params.append(self.tafas_gating)
        return params



class EnCoBA_FreqDomain_GCM(nn.Module):
    """
    EnCoBA (Binary Version) - 0 or 1 Only
    
    机制变化：
    不再使用软过度。
    如果 (Entropy > Threshold): Gate = 0 (完全忽略 Codebook)
    如果 (Entropy <= Threshold): Gate = 1 (完全信任 Codebook)
    
    这种"全有或全无"的策略在跨域 OOD 场景下能最大程度减少负迁移。
    """
    def __init__(self, window_len, n_var=1, hidden_dim=32,
                 gating_init=0.01, var_wise=True,
                 n_bases=8, feature_dim=32, query_type='freq-norm-CI', 
                 # 阈值建议：
                 # 0.6 表示只要相似度分布稍微有点乱，就直接弃权
                 entropy_threshold=0.6, 
                 **kwargs):
        super(EnCoBA_FreqDomain_GCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        self.n_bases = n_bases
        self.feature_dim = feature_dim
        self.online_mode = False
        self.freq_len = window_len // 2 + 1
        
        # 硬阈值
        self.entropy_threshold = entropy_threshold
        
        # --- 1. Codebook / Bases ---
        self.codebook_keys = nn.Parameter(torch.randn(n_var, n_bases, feature_dim))
        
        if var_wise:
            self.bases_r = nn.Parameter(torch.Tensor(n_bases, self.freq_len, n_var))
            self.bases_i = nn.Parameter(torch.Tensor(n_bases, self.freq_len, n_var))
        else:
            self.bases_r = nn.Parameter(torch.Tensor(n_bases, self.freq_len))
            self.bases_i = nn.Parameter(torch.Tensor(n_bases, self.freq_len))

        self._init_bases()

        # --- Query Net ---
        # --- Query Net Selection Logic (Factory) ---
        print(f"Initializing FV-CoBA (Element-Wise) with Query Type: {query_type}")
        if query_type == 'time':
            self.query_net = QueryNet_Time(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-CI':
            self.query_net = QueryNet_Freq_Base_ChannelIndependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-CD':
            self.query_net = QueryNet_Freq_Base_ChannelDependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-separate-CI':
            self.query_net = QueryNet_Freq_Separate_ChannelIndependence(window_len, n_var, feature_dim)
        elif query_type == 'freq-mag-phase':
            self.query_net = QueryNet_Freq_MagPhase(window_len, n_var, feature_dim)
        elif query_type == 'freq-norm-CI':
            self.query_net = QueryNet_Freq_Norm_ChannelIndependence(window_len, n_var, feature_dim)
        else:
            # Default fallback
            print(f"Unknown query_type: {query_type}, defaulting to 'freq-base-CI'")
            self.query_net = QueryNet_Freq_Base_ChannelIndependence(window_len, n_var, feature_dim)

        # --- 3. Online Mode ---
        self.scale = 1e-5
        self.tafas_gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.online_freq_r = nn.Parameter(self.scale * torch.randn(1, self.freq_len, n_var))
        self.online_freq_i = nn.Parameter(self.scale * torch.randn(1, self.freq_len, n_var))
        self.online_bias_r = nn.Parameter(torch.zeros(1, self.freq_len, n_var))
        self.online_bias_i = nn.Parameter(torch.zeros(1, self.freq_len, n_var))

    def _init_bases(self):
        with torch.no_grad():
            if self.var_wise:
                for v in range(self.n_var):
                    init_matrix_r = torch.empty(self.n_bases, self.freq_len)
                    nn.init.orthogonal_(init_matrix_r)
                    self.bases_r.data[:, :, v] = init_matrix_r
                    init_matrix_i = torch.empty(self.n_bases, self.freq_len)
                    nn.init.orthogonal_(init_matrix_i)
                    self.bases_i.data[:, :, v] = init_matrix_i
            else:
                nn.init.orthogonal_(self.bases_r)
                nn.init.orthogonal_(self.bases_i)

    # def complex_element_wise_forward(self, x_fft, coeffs):
    #     if self.var_wise:
    #         w_r = torch.einsum('bvn, nfv -> bfv', coeffs, self.bases_r)
    #         w_i = torch.einsum('bvn, nfv -> bfv', coeffs, self.bases_i)
    #     else:
    #         w_r = torch.einsum('bvn, nf -> bfv', coeffs, self.bases_r)
    #         w_i = torch.einsum('bvn, nf -> bfv', coeffs, self.bases_i)

    #     xr, xi = x_fft.real, x_fft.imag
    #     z_r = xr * w_r - xi * w_i
    #     z_i = xr * w_i + xi * w_r
    #     return torch.complex(z_r, z_i)

    def complex_element_wise_forward(self, x_fft, coeffs):
        # [关键步骤]：实时归一化 Base
        # 无论 Base 在训练中变成了多大，这里强制把它拉回单位圆
        # dim=1 是频率维度，保证每个 Base 在频域的整体能量为 1
        
        # 你的 bases 形状可能是 (N_bases, Freq_len, N_var)
        # 我们希望对每个 Base (N, V) 在频率轴 (F) 上归一化
        eps = 1e-8
        
        # 计算 L2 范数: (N, 1, V)
        norm_r = torch.norm(self.bases_r, p=2, dim=1, keepdim=True)
        norm_i = torch.norm(self.bases_i, p=2, dim=1, keepdim=True)
        
        # 归一化后的基向量
        bases_r_unit = self.bases_r / (norm_r + eps)
        bases_i_unit = self.bases_i / (norm_i + eps)

        # 使用归一化后的基进行聚合
        if self.var_wise:
            w_r = torch.einsum('bvn, nfv -> bfv', coeffs, bases_r_unit)
            w_i = torch.einsum('bvn, nfv -> bfv', coeffs, bases_i_unit)
        else:
            w_r = torch.einsum('bvn, nf -> bfv', coeffs, bases_r_unit)
            w_i = torch.einsum('bvn, nf -> bfv', coeffs, bases_i_unit)

        # 计算出标准化的残差
        xr, xi = x_fft.real, x_fft.imag
        z_r = xr * w_r - xi * w_i
        z_i = xr * w_i + xi * w_r
        return torch.complex(z_r, z_i)

    def forward(self, x):
        B, L, V = x.shape

        # 1. FFT
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')  

        # 2. Query & Similarity
        query = self.query_net(x) 
        query_norm = F.normalize(query, p=2, dim=-1)
        keys_norm = F.normalize(self.codebook_keys, p=2, dim=-1)
        
        similarity = torch.matmul(
            query_norm.unsqueeze(2), 
            keys_norm.transpose(1, 2)
        ).squeeze(2)

        # --- 核心修改：Binary Entropy Gating ---
        
        # 2.1 计算熵
        tau_entropy = 0.1 
        prob_dist = F.softmax(similarity / tau_entropy, dim=-1) 
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-9), dim=-1)
        max_entropy = torch.log(torch.tensor(self.n_bases, dtype=torch.float32, device=x.device))
        normalized_entropy = entropy / (max_entropy + 1e-9)
        
        # 2.2 二值化逻辑 (Hard Thresholding)
        # 如果 entropy < threshold，说明很确定，mask = 1.0
        # 如果 entropy >= threshold，说明不确定，mask = 0.0
        mask_hard = (normalized_entropy < self.entropy_threshold).float()
        
        # 2.3 Straight-Through Estimator (STE) 技巧
        # 如果你在 TTA 时不需要更新 QueryNet，可以直接用 mask_hard。
        # 如果需要更新，STE 允许梯度穿过这个硬开关。
        # 在前向传播中，它等于 mask_hard；在反向传播中，它看起来像是一个连续函数(如1-entropy)。
        # 这里为了简单展示效果，直接使用 hard mask
        # ent_gate = mask_hard
        ent_gate = mask_hard + (1.0 - normalized_entropy) - (1.0 - normalized_entropy).detach()
        
        self.last_ent_gate = ent_gate # 监控用，你会看到它全是 0 或 1

        # 3. Top-K & Aggregation
        k = 2 
        topk_vals, topk_indices = torch.topk(similarity, k=k, dim=-1)
        mask = torch.full_like(similarity, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_vals)
        coeffs = F.softmax(mask, dim=-1) 
        self.coeffs = coeffs

        # 4. Main Path
        delta_fft_codebook = self.complex_element_wise_forward(x_fft, coeffs)
        delta_time_codebook = torch.fft.irfft(delta_fft_codebook, n=L, dim=1, norm='ortho')

        # 应用硬开关：非黑即白
        delta_time_codebook = delta_time_codebook * ent_gate.unsqueeze(1)

        # 5. Online Path & Final Output
        if self.online_mode:
            delta_real_online = (x_fft.real * self.online_freq_r - x_fft.imag * self.online_freq_i + self.online_bias_r)
            delta_imag_online = (x_fft.imag * self.online_freq_r + x_fft.real * self.online_freq_i + self.online_bias_i)
            y_online = torch.complex(delta_real_online, delta_imag_online)
            delta_time_online = torch.fft.irfft(y_online, n=L, dim=1, norm='ortho')
            delta_time_online = torch.tanh(self.tafas_gating) * delta_time_online
            out = x + delta_time_codebook + delta_time_online
        else:
            out = x + delta_time_codebook

        return out

    def get_optim_params(self):
        params = []
        if self.online_mode:
            params.append(self.online_freq_r)
            params.append(self.online_freq_i)
            params.append(self.online_bias_r)
            params.append(self.online_bias_i)
            params.append(self.tafas_gating)
        else:
            params.append(self.tafas_gating)
        return params



class RoCoBA_FreqDomain_Norm(nn.Module):
    """
    Revised Version: Implements Statistical Alignment & Normalized Residual Learning
    
    1. Main Path (Codebook): 
       Input (Norm) -> FFT -> Query -> Select Freq Domain Bases -> iFFT -> Residual (Norm)
    
    2. Online Path (Test-time Adaptation):
       Input (Norm) -> FFT -> Per-Variable Freq Transform -> iFFT -> Gated Residual (Norm)
       
    3. Output:
       Base_Aligned + (Residual_Codebook + Residual_Online) * Input_Std
    """
    def __init__(self, window_len, n_var=1, hidden_dim=32,
                 gating_init=0.01, var_wise=True,
                 n_bases=8, feature_dim=32, query_type='freq-base-CI', 
                 confidence_threshold=0.5, seq_len=96, **kwargs):
        super(RoCoBA_FreqDomain_Norm, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        self.n_bases = n_bases
        self.feature_dim = feature_dim
        self.online_mode = False
        # self.analyzer = CoBA_Analyzer(self) # 假设你有这个类，保留
        self.freq_len = window_len // 2 + 1
        self.seq_len = seq_len
        
        # [Add]: Confidence Threshold (tau) for Knowledge Retrieval
        self.confidence_threshold = confidence_threshold
        
        # --- 1. Codebook / Bases in Frequency Domain (Element-wise) ---
        self.codebook_keys = nn.Parameter(torch.randn(n_var, n_bases, feature_dim))
        
        if var_wise:
            # Bases: (N_bases, Freq_len, N_var)
            self.bases_r = nn.Parameter(torch.Tensor(n_bases, self.freq_len, n_var))
            self.bases_i = nn.Parameter(torch.Tensor(n_bases, self.freq_len, n_var))
        else:
            # Bases: (N_bases, Freq_len)
            self.bases_r = nn.Parameter(torch.Tensor(n_bases, self.freq_len))
            self.bases_i = nn.Parameter(torch.Tensor(n_bases, self.freq_len))

        # Initialization
        self.scale = 1e-5
        self._init_bases()
        
        # --- Query Net Selection Logic (Factory) ---
        print(f"Initializing FV-CoBA (Element-Wise) with Query Type: {query_type}")
        # (保留你原来的 QueryNet 初始化逻辑，这里简化展示)
        if query_type == 'time':
            self.query_net = QueryNet_Time(window_len, n_var, feature_dim)
        elif query_type == 'freq-base-CI':
            self.query_net = QueryNet_Freq_Base_ChannelIndependence(window_len, n_var, feature_dim)
            # self.query_net = QueryNet_Freq_Base_ChannelIndependence(seq_len, n_var, feature_dim)
        # ... (保留其他 query_type 的判断) ...
        else:
            # Default fallback
            print(f"Unknown query_type: {query_type}, defaulting to 'freq-base-CI'")
            self.query_net = QueryNet_Freq_Base_ChannelIndependence(window_len, n_var, feature_dim)
            # self.query_net = QueryNet_Freq_Base_ChannelIndependence(seq_len, n_var, feature_dim)

        # --- 3. Online Mode Parameters ---
        self.tafas_gating = nn.Parameter(gating_init * torch.ones(n_var))
        
        # 注意：因为现在是在归一化空间操作，这些参数的初始化变得更加安全有效
        # self.freq_len = self.freq_len + self.seq_len // 2 # take input as window_len + 96

        if eved_enable:
            self.freq_len = self.freq_len + 24 // 2 # take input as window_len + 96
        else:
            self.freq_len = self.freq_len + 96 // 2 # take input as window_len + 96
        
        self.online_freq_r = nn.Parameter(self.scale * torch.randn(1, self.freq_len, n_var))
        self.online_freq_i = nn.Parameter(self.scale * torch.randn(1, self.freq_len, n_var))
        self.online_bias_r = nn.Parameter(torch.zeros(1, self.freq_len, n_var))
        self.online_bias_i = nn.Parameter(torch.zeros(1, self.freq_len, n_var))

    def _init_bases(self):
        with torch.no_grad():
            if self.var_wise:
                for v in range(self.n_var):
                    joint_bases = torch.empty(self.n_bases, 2 * self.freq_len)
                    nn.init.orthogonal_(joint_bases)
                    bases_r_chunk, bases_i_chunk = torch.split(joint_bases, self.freq_len, dim=1)
                    self.bases_r.data[:, :, v] = bases_r_chunk * self.scale
                    self.bases_i.data[:, :, v] = bases_i_chunk * self.scale
            else:
                joint_bases = torch.empty(self.n_bases, 2 * self.freq_len)
                nn.init.orthogonal_(joint_bases)
                bases_r_chunk, bases_i_chunk = torch.split(joint_bases, self.freq_len, dim=1)
                self.bases_r.data.copy_(bases_r_chunk * self.scale)
                self.bases_i.data.copy_(bases_i_chunk * self.scale)

    def _get_query(self, x):
        return self.query_net(x)
    
    def complex_element_wise_forward(self, x_fft, coeffs):
        if self.var_wise:
            w_r = torch.einsum('bvn, nfv -> bfv', coeffs, self.bases_r)
            w_i = torch.einsum('bvn, nfv -> bfv', coeffs, self.bases_i)
        else:
            w_r = torch.einsum('bvn, nf -> bfv', coeffs, self.bases_r)
            w_i = torch.einsum('bvn, nf -> bfv', coeffs, self.bases_i)

        xr, xi = x_fft.real, x_fft.imag 
        z_r = xr * w_r - xi * w_i
        z_i = xr * w_i + xi * w_r
              
        return torch.complex(z_r, z_i)

    def forward(self, x_base, x_enc=None):
        """
        Modified forward to handle Statistical Alignment.
        
        Args:
            x_base: Output from Pre-trained Model (e.g., iTransformer). 
                    Shape: (Batch, Window_len, N_var).
                    Issue: May have incorrect scale/shift (e.g., ~1.0 vs ~700.0).
            x_enc:  Original Input Sequence. 
                    Shape: (Batch, Seq_Len, N_var).
                    Role: Provides ground-truth statistics for alignment.
        """
        B, L, _ = x_base.shape

        # =======================================================
        # [Phase 1]: Frequency Domain Processing
        # [Phase 0]: Statistical Alignment (统计量对齐)
        # =======================================================
        if x_enc is not None:
            # 1. 计算输入的真实统计量 (Ground Truth Statistics)
            input_mean = x_enc.mean(dim=1, keepdim=True)
            input_std = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True) + 1e-5)
            
            # 2. 计算 Base Model 输出的统计量 (Biased Statistics)
            base_mean = x_base.mean(dim=1, keepdim=True)
            base_std = torch.sqrt(torch.var(x_base, dim=1, keepdim=True) + 1e-5)
            
            # 3. 构造 "内部工作变量" (Internal Normalized Variable)
            # x_internal 是标准正态分布 N(0,1)，Adapter 在这个舒适区工作
            x_internal = (x_base - base_mean) / base_std 
            x_enc_norm = (x_enc - input_mean) / input_std
            
            # 4. 构造 "对齐后的基底" (Aligned Base)
            # 将 Base Model 的输出强行拉伸到真实的量纲
            if eved_enable: # 在eVED上设置
                # input_std = input_std[:, :, -2:]  
                # input_mean = input_mean[:, :, -2:]  

                target_idx = [12, 13]  
                # 选取指定通道，而不是最后两个
                input_std = input_std[:, :, target_idx]  
                input_mean = input_mean[:, :, target_idx]  
            x_aligned = x_internal * input_std + input_mean
        else:
            # 如果没有提供 input (例如非迁移场景或为了兼容旧代码)
            # 退化为原始逻辑，但会有量纲风险
            x_internal = x_base
            x_aligned = x_base
            input_std = 1.0 

        # =======================================================
        # [Phase 1]: Frequency Domain Processing (on Normalized Data)
        # =======================================================
        
        # 1. Transform to Frequency Domain
        # x_fft = torch.fft.rfft(x_internal, dim=1, norm='ortho')  # (B, F, V)
        x_fft = torch.fft.rfft(x_base, dim=1, norm='ortho')  # (B, F, V)

        # 2. Query & Codebook Selection
        # QueryNet 看到的也是归一化的数据，保证检索稳定性
        # query = self._get_query(x_internal) 
        query = self._get_query(x_base)
        # adapter_ins = torch.cat([x_enc_norm, x_internal], dim=1)
        # query = self._get_query(adapter_ins) 
        # query = self._get_query(x_enc_norm)
        
        query_norm = F.normalize(query, p=2, dim=-1)
        keys_norm = F.normalize(self.codebook_keys, p=2, dim=-1)
        
        # Calculate Cosine Similarity
        similarity = torch.matmul(
            query_norm.unsqueeze(2),         # (B, V, 1, D)
            keys_norm.transpose(1, 2)        # (V, D, N)
        ).squeeze(2)                     # (B, V, N)
        
        # [Add]: Logic for Confidence Gating
        # s_max, _ = torch.max(similarity, dim=-1) # (B, V)
        # s_max = F.relu(s_max)
        
        # gamma = torch.where(
        #     s_max >= self.confidence_threshold, 
        #     torch.ones_like(s_max), 
        #     s_max / self.confidence_threshold
        # )
        # gamma = gamma.unsqueeze(1).pow(2) # (B, 1, V)

        # Top-K Softmax Logic
        k = 2 
        topk_vals, topk_indices = torch.topk(similarity, k=k, dim=-1)
        mask = torch.full_like(similarity, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_vals)
        coeffs = F.softmax(mask, dim=-1) # (B, V, N)
        self.coeffs = coeffs

        # 3. Main Path: Codebook-based Frequency Adaptation (Element-Wise)
        delta_fft_codebook = self.complex_element_wise_forward(x_fft, coeffs)
        
        # iFFT for Main Path
        delta_time_codebook = torch.fft.irfft(delta_fft_codebook, n=L, dim=1, norm='ortho')
        
        # Apply Confidence Gating
        # 此时 delta_time_codebook 是 "归一化的残差"
        delta_time_codebook = delta_time_codebook

        # =======================================================
        # [Phase 2]: Online Adaptation (on Normalized Data)
        # =======================================================
        # 4. Online Path: Element-wise Frequency Calibration
        if self.online_mode:
            # 这里的 x_fft 是归一化的，所以 online_freq 参数只需要在 1e-5 附近微调
            # 梯度非常稳定，收敛快
            # print(x_enc_norm.shape, x_internal.shape)

            if eved_enable:
                # x_enc_norm = x_enc_norm[:,:,-2:]
                
                target_idx = [12, 13]
                x_enc = x_enc[:, :, target_idx]
            adapter_ins = torch.cat([x_enc, x_base], dim=1)

            # adapter_ins = torch.cat([x_enc, x_base], dim=1)

            x_fft = torch.fft.rfft(adapter_ins, dim=1, norm='ortho')  # (B, F, V)
            delta_real_online = (
                x_fft.real * self.online_freq_r - x_fft.imag * self.online_freq_i + self.online_bias_r
            )
            delta_imag_online = (
                x_fft.imag * self.online_freq_r + x_fft.real * self.online_freq_i + self.online_bias_i
            )
            
            y_online = torch.complex(delta_real_online, delta_imag_online)
            
            # iFFT Online
            # delta_time_online = torch.fft.irfft(y_online, n=L, dim=1, norm='ortho')
            delta_time_online = torch.fft.irfft(y_online, n=adapter_ins.size(1), dim=1, norm='ortho')
            delta_time_online = delta_time_online[:, -L:, :]
            
            # Gating with tanh(lambda)
            delta_time_online = torch.tanh(self.tafas_gating) * delta_time_online

            # 融合归一化残差
            total_residual_norm = delta_time_codebook + delta_time_online
        else:
            total_residual_norm = delta_time_codebook

        # =======================================================
        # [Phase 3]: Final Rescaling (恢复量纲)
        # =======================================================
        out = x_base + total_residual_norm

        return out

    def get_optim_params(self):
        params = []
        if self.online_mode:
            params.append(self.online_freq_r)
            params.append(self.online_freq_i)
            params.append(self.online_bias_r)
            params.append(self.online_bias_i)
            params.append(self.tafas_gating)
        else:
            params.append(self.tafas_gating)
        return params
