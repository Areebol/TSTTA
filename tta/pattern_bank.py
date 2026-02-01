import torch
import torch.nn as nn
import torch.nn.functional as F
from tta.tta_dual_utils.query_net import *

# class PKA_GCM(nn.Module):
#     def __init__(self, window_len, n_var=1, seq_len=96, 
#                  n_static=8, feature_dim=16, temperature=10.0,
#                  bias_momentum=0.1, query_type='time-CI', **kwargs):
#         """
#         OD-TTA Memory Module
#         args:
#             window_len: 输入序列长度
#             pred_len: 预测序列长度 (OD-TTA Output Dimension H)
#             n_static: 静态原型的数量 (N)
#             feature_dim: Query/Key 的维度 (d)
#             bias_momentum: 在线 Bias 更新的 alpha
#         """
#         super(PKA_GCM, self).__init__()
#         self.seq_len = seq_len
#         self.window_len = window_len
#         self.n_var = n_var
#         self.feature_dim = feature_dim
#         self.alpha = bias_momentum # Bias momentum
#         self.energy_threshold = 0.2
#         self.n_static = n_static
#         self.max_capacity = n_static * 2 # Dynamic Memory 最大容量
#         self.temperature = temperature
        
#         # --- MLP Encoder ---
#         print(f"Initializing Query Type: {query_type}")
#         # (保留你原来的 QueryNet 初始化逻辑，这里简化展示)
#         if query_type == 'time':
#             self.query_net = QueryNet_Time(seq_len + window_len, n_var, feature_dim)
#         elif query_type == 'time-CI':
#             self.query_net = QueryNet_TimeCI(seq_len + window_len, n_var, feature_dim)
#         elif query_type == 'freq-base-CI':
#             self.query_net = QueryNet_Freq_Base_ChannelIndependence(seq_len + window_len, n_var, feature_dim)
#         elif query_type == 'freq-base-CD':
#             self.query_net = QueryNet_Freq_Base_ChannelDependence(seq_len + window_len, n_var, feature_dim)
#         else:
#             # Default fallback
#             print(f"Unknown query_type: {query_type}, defaulting to 'time'")
#             self.query_net = QueryNet_TimeCI(seq_len + window_len, n_var, feature_dim)
        
#         # --- Static Prototype Memory (Offline Learned) ---
#         # Key: 存储特征方向, Shape (N_static, d)
#         self.static_keys = nn.Parameter(torch.randn(n_static, feature_dim))
#         # Value: 存储残差修正, Shape (N_static, window_len, n_var)
#         self.static_values = nn.Parameter(torch.zeros(n_static, window_len, n_var))
        
#         # nn.init.orthogonal_(self.static_keys)
#         nn.init.kaiming_normal_(self.static_keys)
#         nn.init.zeros_(self.static_values)

#         # --- Dynamic Instance Memory (Online) ---
#         # 使用 register_buffer 以免被优化器更新，但在 state_dict 中保存
#         # 初始为空，Shape 动态增长
#         self.register_buffer('dynamic_keys', torch.empty(0, feature_dim)) 
#         self.register_buffer('dynamic_values', torch.empty(0, window_len, n_var))
        
#         # --- Online Bias Estimator ---
#         # 专门处理 Mean Shift
#         self.register_buffer('global_bias', torch.zeros(window_len, n_var))

#     def _get_query(self, x, y_base):
#         """
#         OD-TTA MACE Input: Concat(X, Y_base)
#         """
#         assert x is not None
#         query = torch.cat([x, y_base], dim=1) # (B, seq_len + window_len, n_var)
        
#         query = self.query_net(query) 
#         query = F.normalize(query, p=2, dim=-1) # (B, d)
#         return query

#     def forward(self, y_base, x=None):
#         """
#         Variable-wise Retrieval Logic
#         """
#         # 1. 获取 Query
#         # Shape: (B, V, D)
#         z_t = self._get_query(x, y_base) 

#         # ================= Static Retrieval =================
#         # Keys: (N, D)
#         # z_t:  (B, V, D)
#         # 目标: 计算每个变量 V 对每个 Key N 的相似度
#         # Einsum: bvd (query), nd (keys) -> bvn (similarity)
#         sim_static = torch.einsum('bvd, nd -> bvn', z_t, F.normalize(self.static_keys, p=2, dim=1))
        
#         # Softmax: 沿着 Key 维度 (最后一维)
#         w_static = F.softmax(self.temperature * sim_static, dim=-1) # (B, V, N)
        
#         # Correction:
#         # w_static: (B, V, N)
#         # values:   (N, H, V) -> H is output_len
#         # 逻辑: 对于每个 batch b 和变量 v，使用权重 w[b,v,:] 对 values[:,:,v] 进行加权
#         # Einsum: bvn (weight), nhv (value) -> bhv (output)
#         # 注意：这里的 'v' 必须对齐，确保第 v 个变量只检索第 v 个变量的 Value
#         delta_static = torch.einsum('bvn, nhv -> bhv', w_static, self.static_values)

#         # ================= Dynamic Retrieval =================
#         delta_dynamic = torch.zeros_like(y_base)
#         if self.dynamic_keys.shape[0] > 0:
#             # Dynamic Keys: (K, D)
#             # z_t: (B, V, D)
#             # Sim: (B, V, K)
#             sim_dynamic = torch.einsum('bvd, kd -> bvk', z_t, self.dynamic_keys) # keys already normalized
            
#             w_dynamic = F.softmax(self.temperature * sim_dynamic, dim=-1) # (B, V, K)
            
#             # Values: (K, H, V)
#             # Correction: (B, H, V)
#             delta_dynamic = torch.einsum('bvk, khv -> bhv', w_dynamic, self.dynamic_values)

#         # 4. Fusion
#         y_final = y_base + self.global_bias + delta_static + delta_dynamic
        
#         return y_final, z_t

#     def update_bias(self, y_gt, y_base_pred, y_final_pred=None):
#         """
#         更新 Global Bias [cite: 61]
#         通常使用 y_gt - y_base 甚至 y_gt - y_final 来更新
#         """
#         # 简单策略：用当前总误差更新 Bias
#         if y_final_pred is None:
#             residual = y_gt - y_base_pred
#         else:
#             residual = y_gt - y_final_pred
            
#         # EMA 更新
#         current_bias_shift = residual.mean(dim=0) # Average over batch
#         self.global_bias = (1 - self.alpha) * self.global_bias + self.alpha * current_bias_shift
        
#     def update_dynamic_memory(self, z_t, y_final_pred, y_gt):
#         """
#         在线增加与冗余剔除
        
#         代码逻辑:
#         在遍历 Batch 时，一旦发现新模式并添加到 Memory,
#         立即将其加入当前的 '检查基(K_working)' 中。
#         这样 Batch 后续的相似样本再计算正交残差时，
#         就会被这个新加入的 Key "解释" 掉，从而判定为 redundant 并跳过。
#         """
#         batch_size = z_t.shape[0]
#         current_err = y_gt - y_final_pred # (B, H, V)

#         # 构建初始检查基 (Working Basis)
#         # 包含所有的 Static Keys 和 当前时刻已有的 Dynamic Keys
#         basis_list = [F.normalize(self.static_keys, p=2, dim=1)]
#         if self.dynamic_keys.shape[0] > 0:
#             basis_list.append(self.dynamic_keys) # dynamic_keys 存入时已归一化
        
#         # K_working: 用于当前 Batch 正交检查的临时集合
#         # Shape: (N_total, d)
#         K_working = torch.cat(basis_list, dim=0) 

#         # 逐样本遍历
#         for i in range(batch_size):
#             z_i = z_t[i]       # (d)
#             err_i = current_err[i] # (H, V)

#             # --- Gram-Schmidt Orthogonality Check ---
#             # 关键点：投影到 K_working 上，而不仅仅是原来的 keys
#             # proj = \sum (z_i . k_j) * k_j
#             coeffs = torch.matmul(K_working, z_i) # (N_working)
#             proj = torch.matmul(coeffs, K_working) # (d)
            
#             r_ortho = z_i - proj   # 正交残差
#             energy = torch.norm(r_ortho, p=2) # 能量

#             # --- 判定逻辑 ---
#             if energy > self.energy_threshold:
#                 # 这是一个新模式 (Novel Pattern)
#                 new_key = r_ortho / energy # 归一化作为新 Key
#                 new_value = err_i # 对应的误差作为 Value
                
#                 # A. 更新永久存储 (Dynamic Memory)
#                 self.dynamic_keys = torch.cat([self.dynamic_keys, new_key.unsqueeze(0)], dim=0)
#                 self.dynamic_values = torch.cat([self.dynamic_values, new_value.unsqueeze(0)], dim=0)
                
#                 # 更新工作检查基 (Update Working Basis)
#                 # 这样下一次循环(i+1)时，如果样本和当前样本相似，
#                 # 它在 K_working 上的投影就会很大，r_ortho 就会很小，从而避免重复添加
#                 K_working = torch.cat([K_working, new_key.unsqueeze(0)], dim=0)

#                 # C. 容量管理 (FIFO)
#                 if self.dynamic_keys.shape[0] > self.max_capacity - self.n_static:
#                     # 移除最早加入的记忆
#                     self.dynamic_keys = self.dynamic_keys[1:]
#                     self.dynamic_values = self.dynamic_values[1:]



class PKA_GCM(nn.Module):
    def __init__(self, window_len, n_var=1, seq_len=96, 
                 n_static=8, feature_dim=16, temperature=10.0,
                 bias_momentum=0.1, query_type='time-CI', **kwargs):
        """
        Channel-Specific OD-TTA Memory Module
        核心改进: 
        将 Keys 和 Values 的维度包含 n_var，实现通道间的物理隔离。
        通道 A 的模式只会存储在 A 的槽位中，绝不会干扰通道 B。
        """
        super(PKA_GCM, self).__init__()
        self.seq_len = seq_len      
        self.window_len = window_len
        self.n_var = n_var
        self.feature_dim = feature_dim
        self.alpha = bias_momentum
        self.temperature = temperature
        self.n_static = n_static
        self.max_capacity = n_static * 2

        input_len = seq_len + window_len
        # 假设已定义 QueryNet_TimeCI (输出 B, V, D)
        self.query_net = QueryNet_TimeCI(input_len, n_var, feature_dim)
        
        # =========================================================
        # [关键修改 1] 参数维度提升，实现通道隔离
        # =========================================================
        
        # Static Keys: (n_var, n_static, feature_dim)
        # 含义: 第 i 个变量拥有属于自己的 n_static 个基向量
        self.static_keys = nn.Parameter(torch.randn(n_var, n_static, feature_dim))
        
        # Static Values: (n_var, n_static, window_len)
        # 含义: 第 i 个变量的第 j 个 Key 对应的修正量 (只修正该变量自己)
        self.static_values = nn.Parameter(torch.zeros(n_var, n_static, self.window_len))
        
        # 初始化: 对每个变量的 Key 矩阵分别做正交初始化
        for v in range(n_var):
            nn.init.orthogonal_(self.static_keys[v])
            # nn.init.kaiming_uniform_(self.static_values[v])
        nn.init.zeros_(self.static_values)

        # Dynamic Memory Buffer
        # Shape: (Capacity, n_var, feature_dim) -> 随时间增长，但每个时刻都存所有变量的快照
        self.register_buffer('dynamic_keys', torch.empty(0, n_var, feature_dim))
        self.register_buffer('dynamic_values', torch.empty(0, n_var, self.window_len))
        # LFU Counts (针对每个时刻的 snapshot)
        self.register_buffer('dynamic_counts', torch.empty(0, dtype=torch.float32))

        # Bias (Global Shift)
        self.register_buffer('global_bias', torch.zeros(self.window_len, n_var))

    def _get_query(self, x, y_base):
        # x: (B, L_in, V), y_base: (B, L_out, V) -> (B, L_all, V)
        query_input = torch.cat([x, y_base], dim=1) 
        query = self.query_net(query_input) # (B, V, D)
        query = F.normalize(query, p=2, dim=-1) 
        return query

    def forward(self, y_base, x=None):
        """
        Channel-Specific Retrieval Logic
        """
        batch_size = y_base.shape[0]
        # 1. 获取 Query: (B, V, D)
        z_t = self._get_query(x, y_base) 

        # =========================================================
        # [关键修改 2] 严格的对应通道检索
        # =========================================================

        # --- Static Retrieval ---
        # Query: (B, V, D)
        # Keys:  (V, N, D)  <-- 注意这里 V 在 dim 0
        # 我们希望: Batch B 中的 变量 V，去匹配 Keys 中的 变量 V 的 N 个 Key
        
        # Einsum 解析:
        # bvd: batch, var, dim
        # vnd: var, static_idx, dim
        # -> bvn: batch, var, static_idx (每个变量得到了对自己 Memory 的相似度)
        sim_static = torch.einsum('bvd, vnd -> bvn', z_t, F.normalize(self.static_keys, p=2, dim=-1))
        
        w_static = F.softmax(self.temperature * sim_static, dim=-1) # (B, V, N)
        
        # Correction:
        # w_static: (B, V, N)
        # Values:   (V, N, H)
        # -> bvh: batch, var, horizon (输出修正量)
        delta_static = torch.einsum('bvn, vnh -> bvh', w_static, self.static_values)
        
        # Transpose output to match y_base (B, H, V)
        delta_static = delta_static.permute(0, 2, 1)

        # --- Dynamic Retrieval ---
        delta_dynamic = torch.zeros_like(y_base)
        if self.dynamic_keys.shape[0] > 0:
            # Dynamic Keys: (K, V, D) -> K is time/capacity
            # Query: (B, V, D)
            
            # Einsum:
            # bvd (query)
            # kvd (keys)
            # -> bvk (batch, var, capacity_idx)
            # 解释: 变量 V 只会和 Dynamic Memory 中该变量 V 的历史记录计算相似度
            sim_dynamic = torch.einsum('bvd, kvd -> bvk', z_t, self.dynamic_keys)
            
            # LFU Count Update (Optional, slightly complex in vectorized form)
            if self.training or True:
                # 统计哪个时间步(k)的快照对当前最有帮助
                # 简单起见，我们统计所有变量投票出的最佳 K
                # (B, V) -> best k index
                best_k = sim_dynamic.argmax(dim=-1) # (B, V)
                # 只要任何一个变量觉得第 k 个快照有用，就给它加分
                unique_k = torch.unique(best_k)
                for k_idx in unique_k:
                    if k_idx < self.dynamic_counts.shape[0]:
                         self.dynamic_counts[k_idx] += 1.0

            w_dynamic = F.softmax(self.temperature * sim_dynamic, dim=-1) # (B, V, K)
            
            # Values: (K, V, H)
            # Output: bvh
            delta_dynamic = torch.einsum('bvk, kvh -> bvh', w_dynamic, self.dynamic_values)
            delta_dynamic = delta_dynamic.permute(0, 2, 1)

        # 4. Fusion
        y_final = y_base + self.global_bias + delta_static + delta_dynamic
        
        return y_final, z_t

    def update_bias(self, y_gt, y_base_pred, y_final_pred=None):
        # Bias shape: (H, V) - 也是独立的，没问题
        if y_final_pred is None:
            residual = y_gt - y_base_pred
        else:
            residual = y_gt - y_final_pred
        current_bias_shift = residual.mean(dim=0)
        self.global_bias = (1 - self.alpha) * self.global_bias + self.alpha * current_bias_shift

    def update_dynamic_memory(self, z_t, y_gt, y_final_pred, threshold=0.2):
        """
        Channel-Specific Update Logic
        难点: 如何在保持 tensor 形状 (K, V, D) 的同时，处理不同变量稀疏的更新需求？
        策略: 只要有一个变量出现显著新模式，就存下当前所有变量的快照 (Snapshot)。
              在检索时，由于正交性，没变化的变量会自动忽略这个快照。
        """
        batch_size = z_t.shape[0]
        # Residual Error: (B, H, V) -> Permute to (B, V, H) to match value storage
        current_err = (y_gt - y_final_pred).permute(0, 2, 1)

        # Construct Working Basis: (Total, V, D)
        # Static: (V, N, D) -> Permute to (N, V, D) to stack on dim 0
        basis_list = [F.normalize(self.static_keys, p=2, dim=-1).permute(1, 0, 2)]
        if self.dynamic_keys.shape[0] > 0:
            basis_list.append(self.dynamic_keys)
        
        # K_working: (N+K, V, D)
        K_working = torch.cat(basis_list, dim=0)

        for i in range(batch_size):
            z_i = z_t[i]       # (V, D)
            err_i = current_err[i] # (V, H)

            # --- Channel-Wise Orthogonality Check ---
            # 我们需要对每个变量分别计算它是否正交
            
            # 1. Compute Coefficients: Projection of z_i onto K_working per variable
            # K_working: (Total, V, D)
            # z_i:       (V, D)
            # -> (Total, V)
            coeffs = torch.einsum('tvd, vd -> tv', K_working, z_i)
            
            # 2. Compute Projection Vector
            # coeffs: (Total, V)
            # K_working: (Total, V, D)
            # -> (V, D) (Sum over Total bases)
            proj = torch.einsum('tv, tvd -> vd', coeffs, K_working)
            
            # 3. Residual & Energy
            r_ortho = z_i - proj # (V, D)
            energy = torch.norm(r_ortho, p=2, dim=-1) # (V,) - 每个变量的能量

            # --- Decision Strategy ---
            # 策略: 如果【最大能量】超过阈值，或者【平均能量】很高，则添加
            # 这里使用 Max 策略：只要有一个通道出现未知工况，就存下该时刻
            if energy.max() > threshold:
                
                # Normalize new key (per variable)
                # 防止除以0，加上 eps
                new_key = r_ortho / (energy.unsqueeze(-1) + 1e-6) # (V, D)
                new_value = err_i # (V, H)
                # new_count = torch.tensor([1.0], device=z_i.device) # LFU init
                new_count = torch.tensor([10.0], device=z_i.device) # LFU init

                # Add to Memory
                self.dynamic_keys = torch.cat([self.dynamic_keys, new_key.unsqueeze(0)], dim=0)
                self.dynamic_values = torch.cat([self.dynamic_values, new_value.unsqueeze(0)], dim=0)
                self.dynamic_counts = torch.cat([self.dynamic_counts, new_count], dim=0)
                
                # Update Working Basis (stack at dim 0)
                K_working = torch.cat([K_working, new_key.unsqueeze(0)], dim=0)

                # Capacity Management (LFU)
                if self.dynamic_keys.shape[0] > self.max_capacity - self.n_static:
                    min_idx = torch.argmin(self.dynamic_counts)
                    keep_mask = torch.ones(self.dynamic_keys.shape[0], dtype=torch.bool, device=z_i.device)
                    keep_mask[min_idx] = False
                    
                    self.dynamic_keys = self.dynamic_keys[keep_mask]
                    self.dynamic_values = self.dynamic_values[keep_mask]
                    self.dynamic_counts = self.dynamic_counts[keep_mask] - self.dynamic_counts.min()