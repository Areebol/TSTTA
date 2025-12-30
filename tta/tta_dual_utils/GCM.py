from typing import List
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import prepare_inputs
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

class CoBA_GCM(nn.Module):
    def __init__(self, window_len, n_var=1, hidden_dim=64, 
                 gating_init=0.01, var_wise=True,
                 n_bases=8, feature_dim=32):
        """
        CoBA_GCM: Codebook-based Adaptation GCM
        
        Args:
            window_len: 时间窗口长度
            n_var: 变量数量
            hidden_dim: (原参数) 本模块暂未直接使用，保留接口兼容性
            gating_init: 门控初始化值
            var_wise: 是否对每个变量独立建模权重
            n_bases: 基向量(原子)的数量
            feature_dim: 用于检索的特征维度
        """
        super(CoBA_GCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        self.n_bases = n_bases
        self.feature_dim = feature_dim

        # ============================================================
        # 1. 静态参数 (Memory: Codebook Keys + Basis Set)
        # ============================================================
        
        # Codebook Keys: (N_bases, Feature_dim)
        # 随机初始化，用于匹配 Query
        self.codebook_keys = nn.Parameter(torch.randn(n_bases, feature_dim))
        
        # Basis Set: 替代原本的 self.weight
        # 原本 weight 形状: (window_len, window_len, n_var) 或 (window_len, window_len)
        # 现在增加一个 n_bases 维度
        if var_wise:
            self.bases = nn.Parameter(torch.Tensor(n_bases, window_len, window_len, n_var))
        else:
            self.bases = nn.Parameter(torch.Tensor(n_bases, window_len, window_len))
        
        # 初始化 Basis (类似原本 GCM 初始化为 0，这里建议用很小的值或 Xavier)
        # 为了保证训练初期稳定，初始化为较小的值
        nn.init.xavier_uniform_(self.bases) 

        # ============================================================
        # 2. 辅助组件 (Query Generator + Gating/Bias)
        # ============================================================
        
        # Query Encoder: 将频域特征映射到 feature_dim
        # FFT 后长度约为 window_len // 2 + 1
        fft_len = window_len // 2 + 1
        self.query_net = nn.Sequential(
            nn.Linear(fft_len * n_var, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

        # 保持原本 GCM 的 Bias 和 Gating 逻辑
        self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))

    def _get_query(self, x):
        """
        特征提取: Time Domain -> Frequency Domain -> Query Vector
        x shape: (Batch, Window_len, N_var)
        """
        batch_size = x.shape[0]
        
        # 1. FFT 提取频域特征 (Real FFT)
        # shape: (Batch, fft_len, N_var)
        x_fft = torch.fft.rfft(x, dim=1)
        
        # 2. 取幅度谱 (Magnitude)
        x_mag = x_fft.abs()
        
        # 3. Flatten 展平所有变量的频谱特征
        # shape: (Batch, fft_len * N_var)
        x_feat = x_mag.reshape(batch_size, -1)
        
        # 4. MLP 映射得到 Query
        # shape: (Batch, Feature_dim)
        query = self.query_net(x_feat)
        
        return query

    def forward(self, x):
        """
        x shape: (Batch, Window_len, N_var)
        """
        batch_size = x.size(0)

        # ------------------------------------------------------------
        # Step 1: 特征提取 (Query Generation)
        # ------------------------------------------------------------
        # Q: (B, D)
        query = self._get_query(x)

        # ------------------------------------------------------------
        # Step 2: 系数计算 (Coefficient Matching)
        # ------------------------------------------------------------
        # 对 Query 和 Keys 进行归一化，以便计算 Cosine Similarity
        # eps 避免除零
        query_norm = F.normalize(query, p=2, dim=1)           # (B, D)
        keys_norm = F.normalize(self.codebook_keys, p=2, dim=1) # (N, D)

        # 计算相似度 (Cosine Similarity)
        # (B, D) @ (D, N) -> (B, N)
        similarity = torch.matmul(query_norm, keys_norm.T)
        
        # Softmax 得到稀疏混合系数 w
        # 引入一个 Temperature (可选，这里设为 1)
        coeffs = F.softmax(similarity, dim=-1) # (B, N)

        # ------------------------------------------------------------
        # Step 3: 参数重构 (Reconstruction)
        # ------------------------------------------------------------
        # 利用 einsum 进行加权求和
        # w: (B, N)
        # Bases (var_wise): (N, L, L, V)
        # W_sample: (B, L, L, V)
        
        if self.var_wise:
            # 公式: sum_over_n( w[b, n] * bases[n, out, in, v] ) -> W[b, out, in, v]
            w_sample = torch.einsum('bn, nlio -> blio', coeffs, self.bases)
        else:
            # Bases: (N, L, L)
            w_sample = torch.einsum('bn, nli -> bli', coeffs, self.bases)

        # ------------------------------------------------------------
        # Step 4: 适配运算 (Adaptation / Forward Calculation)
        # ------------------------------------------------------------
        # 原本 GCM 逻辑: x = x + tanh(g) * (xW + b)
        # 现在的 W 是 sample-specific 的 (带 Batch 维度)
        
        # 计算 xW
        if self.var_wise:
            # x: (B, L, V) -> (B, In_L, V)
            # w_sample: (B, Out_L, In_L, V)
            # 目标: (B, Out_L, V)
            # 这里的 einsum: 
            # b: batch, i: input_len, v: var, o: output_len
            feat_trans = torch.einsum('biv, boiv -> bov', x, w_sample)
        else:
            # w_sample: (B, Out_L, In_L)
            feat_trans = torch.einsum('biv, boi -> bov', x, w_sample)

        # 加上 Bias (广播机制)
        feat_trans = feat_trans + self.bias

        # 加上 Residual 和 Gating
        out = x + torch.tanh(self.gating) * feat_trans
        
        self.coeffs = coeffs
        
        return out

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