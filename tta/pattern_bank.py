import torch
import torch.nn as nn
import torch.nn.functional as F
from tta.tta_dual_utils.query_net import *

class PKA_GCM(nn.Module):
    def __init__(self, window_len, n_var=1, seq_len=96, 
                 n_static=8, feature_dim=16, temperature=10.0,
                 bias_momentum=0.1, query_type='time-CI', **kwargs):
        """
        OD-TTA Memory Module
        args:
            window_len: 输入序列长度
            pred_len: 预测序列长度 (OD-TTA Output Dimension H)
            n_static: 静态原型的数量 (N)
            feature_dim: Query/Key 的维度 (d)
            bias_momentum: 在线 Bias 更新的 alpha
        """
        super(PKA_GCM, self).__init__()
        self.seq_len = seq_len
        self.window_len = window_len
        self.n_var = n_var
        self.feature_dim = feature_dim
        self.alpha = bias_momentum # Bias momentum
        self.energy_threshold = 0.2
        self.n_static = n_static
        self.max_capacity = n_static * 2 # Dynamic Memory 最大容量
        self.temperature = temperature
        
        # --- MLP Encoder ---
        print(f"Initializing Query Type: {query_type}")
        # (保留你原来的 QueryNet 初始化逻辑，这里简化展示)
        if query_type == 'time':
            self.query_net = QueryNet_Time(seq_len + window_len, n_var, feature_dim)
        elif query_type == 'time-CI':
            self.query_net = QueryNet_TimeCI(seq_len + window_len, n_var, feature_dim)
        elif query_type == 'freq-base-CI':
            self.query_net = QueryNet_Freq_Base_ChannelIndependence(seq_len + window_len, n_var, feature_dim)
        elif query_type == 'freq-base-CD':
            self.query_net = QueryNet_Freq_Base_ChannelDependence(seq_len + window_len, n_var, feature_dim)
        else:
            # Default fallback
            print(f"Unknown query_type: {query_type}, defaulting to 'time'")
            self.query_net = QueryNet_TimeCI(seq_len + window_len, n_var, feature_dim)
        
        # --- Static Prototype Memory (Offline Learned) ---
        # Key: 存储特征方向, Shape (N_static, d)
        self.static_keys = nn.Parameter(torch.randn(n_static, feature_dim))
        # Value: 存储残差修正, Shape (N_static, window_len, n_var)
        self.static_values = nn.Parameter(torch.zeros(n_static, window_len, n_var))
        
        nn.init.orthogonal_(self.static_keys)
        nn.init.zeros_(self.static_values)

        # --- Dynamic Instance Memory (Online) ---
        # 使用 register_buffer 以免被优化器更新，但在 state_dict 中保存
        # 初始为空，Shape 动态增长
        self.register_buffer('dynamic_keys', torch.empty(0, feature_dim)) 
        self.register_buffer('dynamic_values', torch.empty(0, window_len, n_var))
        
        # --- Online Bias Estimator ---
        # 专门处理 Mean Shift
        self.register_buffer('global_bias', torch.zeros(window_len, n_var))

    def _get_query(self, x, y_base):
        """
        OD-TTA MACE Input: Concat(X, Y_base)
        """
        assert x is not None
        query = torch.cat([x, y_base], dim=1) # (B, seq_len + window_len, n_var)
        
        query = self.query_net(query) 
        query = F.normalize(query, p=2, dim=-1) # (B, d)
        return query

    def forward(self, y_base, x=None):
        """
        Variable-wise Retrieval Logic
        """
        # 1. 获取 Query
        # Shape: (B, V, D)
        z_t = self._get_query(x, y_base) 

        # ================= Static Retrieval =================
        # Keys: (N, D)
        # z_t:  (B, V, D)
        # 目标: 计算每个变量 V 对每个 Key N 的相似度
        # Einsum: bvd (query), nd (keys) -> bvn (similarity)
        sim_static = torch.einsum('bvd, nd -> bvn', z_t, F.normalize(self.static_keys, p=2, dim=1))
        
        # Softmax: 沿着 Key 维度 (最后一维)
        w_static = F.softmax(self.temperature * sim_static, dim=-1) # (B, V, N)
        
        # Correction:
        # w_static: (B, V, N)
        # values:   (N, H, V) -> H is output_len
        # 逻辑: 对于每个 batch b 和变量 v，使用权重 w[b,v,:] 对 values[:,:,v] 进行加权
        # Einsum: bvn (weight), nhv (value) -> bhv (output)
        # 注意：这里的 'v' 必须对齐，确保第 v 个变量只检索第 v 个变量的 Value
        delta_static = torch.einsum('bvn, nhv -> bhv', w_static, self.static_values)

        # ================= Dynamic Retrieval =================
        delta_dynamic = torch.zeros_like(y_base)
        if self.dynamic_keys.shape[0] > 0:
            # Dynamic Keys: (K, D)
            # z_t: (B, V, D)
            # Sim: (B, V, K)
            sim_dynamic = torch.einsum('bvd, kd -> bvk', z_t, self.dynamic_keys) # keys already normalized
            
            w_dynamic = F.softmax(self.temperature * sim_dynamic, dim=-1) # (B, V, K)
            
            # Values: (K, H, V)
            # Correction: (B, H, V)
            delta_dynamic = torch.einsum('bvk, khv -> bhv', w_dynamic, self.dynamic_values)

        # 4. Fusion
        y_final = y_base + self.global_bias + delta_static + delta_dynamic
        
        return y_final, z_t

    def update_bias(self, y_gt, y_base_pred, y_final_pred=None):
        """
        更新 Global Bias [cite: 61]
        通常使用 y_gt - y_base 甚至 y_gt - y_final 来更新
        """
        # 简单策略：用当前总误差更新 Bias
        if y_final_pred is None:
            residual = y_gt - y_base_pred
        else:
            residual = y_gt - y_final_pred
            
        # EMA 更新
        current_bias_shift = residual.mean(dim=0) # Average over batch
        self.global_bias = (1 - self.alpha) * self.global_bias + self.alpha * current_bias_shift
        
    def update_dynamic_memory(self, z_t, y_final_pred, y_gt):
        """
        在线增加与冗余剔除
        
        代码逻辑:
        在遍历 Batch 时，一旦发现新模式并添加到 Memory,
        立即将其加入当前的 '检查基(K_working)' 中。
        这样 Batch 后续的相似样本再计算正交残差时，
        就会被这个新加入的 Key "解释" 掉，从而判定为 redundant 并跳过。
        """
        batch_size = z_t.shape[0]
        current_err = y_gt - y_final_pred # (B, H, V)

        # 构建初始检查基 (Working Basis)
        # 包含所有的 Static Keys 和 当前时刻已有的 Dynamic Keys
        basis_list = [F.normalize(self.static_keys, p=2, dim=1)]
        if self.dynamic_keys.shape[0] > 0:
            basis_list.append(self.dynamic_keys) # dynamic_keys 存入时已归一化
        
        # K_working: 用于当前 Batch 正交检查的临时集合
        # Shape: (N_total, d)
        K_working = torch.cat(basis_list, dim=0) 

        # 逐样本遍历
        for i in range(batch_size):
            z_i = z_t[i]       # (d)
            err_i = current_err[i] # (H, V)

            # --- Gram-Schmidt Orthogonality Check ---
            # 关键点：投影到 K_working 上，而不仅仅是原来的 keys
            # proj = \sum (z_i . k_j) * k_j
            coeffs = torch.matmul(K_working, z_i) # (N_working)
            proj = torch.matmul(coeffs, K_working) # (d)
            
            r_ortho = z_i - proj   # 正交残差
            energy = torch.norm(r_ortho, p=2) # 能量

            # --- 判定逻辑 ---
            if energy > self.energy_threshold:
                # 这是一个新模式 (Novel Pattern)
                new_key = r_ortho / energy # 归一化作为新 Key
                new_value = err_i # 对应的误差作为 Value
                
                # A. 更新永久存储 (Dynamic Memory)
                self.dynamic_keys = torch.cat([self.dynamic_keys, new_key.unsqueeze(0)], dim=0)
                self.dynamic_values = torch.cat([self.dynamic_values, new_value.unsqueeze(0)], dim=0)
                
                # 更新工作检查基 (Update Working Basis)
                # 这样下一次循环(i+1)时，如果样本和当前样本相似，
                # 它在 K_working 上的投影就会很大，r_ortho 就会很小，从而避免重复添加
                K_working = torch.cat([K_working, new_key.unsqueeze(0)], dim=0)

                # C. 容量管理 (FIFO)
                if self.dynamic_keys.shape[0] > self.max_capacity - self.n_static:
                    # 移除最早加入的记忆
                    self.dynamic_keys = self.dynamic_keys[1:]
                    self.dynamic_values = self.dynamic_values[1:]
