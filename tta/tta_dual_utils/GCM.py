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
        x_mag = torch.sqrt(x_fft.real**2 + x_fft.imag**2)
        
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
                tafas_output = torch.tanh(self.gattafas_gatinging) * (torch.einsum('biv,io->bov', x, self.tafas_weight) + self.tafas_bias)
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
                 n_bases=8, feature_dim=32, query_type='freq-base-CD'):
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
        # self.bias = nn.Parameter(torch.zeros(window_len, n_var))

        if var_wise:
            self.tafas_weight = nn.Parameter(torch.Tensor(window_len, window_len, n_var))
        else:
            self.tafas_weight = nn.Parameter(torch.Tensor(window_len, window_len))
        self.tafas_weight.data.zero_()
        self.tafas_gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.codebook_gating = nn.Parameter(gating_init * torch.ones(n_var))
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
                tafas_output = torch.abs(torch.tanh(self.tafas_gating)) * torch.einsum('biv,iov->bov', x, self.tafas_weight) + self.tafas_bias
            else:
                tafas_output = torch.abs(torch.tanh(self.tafas_gating)) * torch.einsum('biv,io->bov', x, self.tafas_weight) + self.tafas_bias
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
            # params.extend(list(self.query_net.parameters()))
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
        k = min(2, self.n_bases)
        
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
            
            feat_trans = feat_trans + self.bias

        if self.online_mode:
            if self.var_wise:
                tafas_output = torch.tanh(self.tafas_gating) * (torch.einsum('biv,iov->bov', x, self.tafas_weight) + self.tafas_bias)
            else:
                tafas_output = torch.tanh(self.tafas_gating) * (torch.einsum('biv,io->bov', x, self.tafas_weight) + self.tafas_bias)
            out = x + feat_trans + tafas_output
        else:
            out = x + feat_trans
        
        self.coeffs = torch.zeros(batch_size, self.n_var, self.n_bases, device=x.device)
        return out

    def get_optim_params(self):
        params = []
        if self.online_mode:
            params.append(self.gating)
            params.append(self.tafas_weight)
            params.append(self.tafas_gating)
            params.append(self.tafas_bias)
        params.extend(list(self.query_net.parameters()))
        params.append(self.bases_left)
        params.append(self.bases_right)
        params.append(self.bias)
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

    def output_calibration(self, outputs):
        if self.out_cali is not None:
            return self.out_cali(outputs)
        return outputs