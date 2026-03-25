import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations
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


class PKA_OnLine(nn.Module):
    def __init__(self, window_len, n_var=1, seq_len=96, 
                 n_static=16, feature_dim=16, temperature=10.0,
                 bias_momentum=0.1, energy_threshold=0.1, max_dynamic_capacity=16, **kwargs):
        """
        OD-TTA v3.3: Bias-Augmented Orthogonal Prototype Memory
        包含 Global Bias (粗调) + Static Memory (离线精调) + Dynamic Memory (在线长尾)
        """
        super(PKA_OnLine, self).__init__()
        self.seq_len = seq_len      
        self.window_len = window_len
        self.n_var = n_var
        self.feature_dim = feature_dim
        self.alpha = bias_momentum
        self.temperature = temperature
        self.n_static = n_static
        self.max_capacity = max_dynamic_capacity # 动态库容量限制
        self.energy_threshold = energy_threshold # 新模式判定阈值

        if feature_dim <= n_static:
            self.feature_dim = n_static * 2
        
        input_len = seq_len + window_len
        # 定义 QueryNet_TimeCI (输出 B, V, D)
        self.query_net = QueryNet_TimeCI(input_len, n_var, feature_dim)

        # --- 1. Static Memory (Offline) ---
        # Static Keys: (n_var, n_static, feature_dim)
        self.static_keys = nn.Parameter(torch.randn(n_var, n_static, feature_dim))
        # Static Values: (n_var, n_static, window_len)
        self.static_values = nn.Parameter(torch.zeros(n_var, n_static, self.window_len))
        
        # 初始化 
        for v in range(n_var):
            nn.init.orthogonal_(self.static_keys[v])
        nn.init.zeros_(self.static_values)

        # --- 2. Dynamic Memory (Online) ---
        # Shape: (Capacity, n_var, feature_dim) 
        self.register_buffer('dynamic_keys', torch.empty(0, n_var, feature_dim))
        self.register_buffer('dynamic_values', torch.empty(0, n_var, self.window_len))
        # LFU Counts: 记录每个动态原型的使用频率，用于删除冗余
        self.register_buffer('dynamic_counts', torch.empty(0, dtype=torch.float32))

        # --- 3. Global Bias (Online) ---
        # Shape: (window_len, n_var) - 对应预测长度和变量数
        self.register_buffer('global_bias', torch.zeros(self.window_len, n_var))

    def _get_query(self, x, y_base):
        # 请确保 self.query_net 在外部或此处正确定义
        query_input = torch.cat([x, y_base], dim=1) 
        query = self.query_net(query_input) # (B, V, D)
        query = F.normalize(query, p=2, dim=-1) # 归一化
        return query

    def forward(self, y_base, x=None):
        """
        OD-TTA v3.3 推理流程 
        Y_final = Y_base + Bias_{t-1} + delta_static + delta_dynamic
        """
        batch_size = y_base.shape[0]
        # 1. 获取 Query: (B, V, D)
        z_t = self._get_query(x, y_base) 

        # --- 2. Static Retrieval ---
        # Einsum: bvd (query), vnd (keys) -> bvn (scores)
        sim_static = torch.einsum('bvd, vnd -> bvn', z_t, F.normalize(self.static_keys, p=2, dim=-1))
        w_static = F.softmax(self.temperature * sim_static, dim=-1) # (B, V, N)
        
        # Retrieve Values: bvn, vnh -> bvh -> permute to (B, H, V)
        delta_static = torch.einsum('bvn, vnh -> bvh', w_static, self.static_values)
        delta_static = delta_static.permute(0, 2, 1)

        # --- 3. Dynamic Retrieval ---
        delta_dynamic = torch.zeros_like(y_base)
        
        if self.dynamic_keys.shape[0] > 0:
            # Dynamic Keys: (K, V, D)
            # Query: (B, V, D)
            # -> bvk (similarity with history snapshots)
            sim_dynamic = torch.einsum('bvd, kvd -> bvk', z_t, self.dynamic_keys)
            
            # --- LFU Counter Update (During Inference) ---
            if self.training or True: # 在 TTA 过程中总是更新计数
                # 找出最相似的 Key 索引
                best_k = sim_dynamic.argmax(dim=-1) # (B, V)
                unique_k = torch.unique(best_k)
                for k_idx in unique_k:
                    if k_idx < self.dynamic_counts.shape[0]:
                         self.dynamic_counts[k_idx] += 1.0

            w_dynamic = F.softmax(self.temperature * sim_dynamic, dim=-1) # (B, V, K)
            
            # Values: (K, V, H) -> bvh -> (B, H, V)
            delta_dynamic = torch.einsum('bvk, kvh -> bvh', w_dynamic, self.dynamic_values)
            delta_dynamic = delta_dynamic.permute(0, 2, 1)

        # --- 4. Final Fusion ---
        # 注意：Global Bias 是 (H, V)，会自动广播到 (B, H, V)
        y_final = y_base + self.global_bias + delta_static + delta_dynamic
        
        return y_final, z_t


    def update_bias(self, y_gt, y_base_pred, y_final_pred=None):
        """
        支持部分长度更新的 Bias 校准
        """
        # 1. 计算残差
        if y_base_pred is not None:
            residual = y_gt - y_base_pred
        else:
            residual = y_gt - y_final_pred
            
        # residual shape: (B, current_len, V)
        # current_bias_shift: (current_len, V)
        current_bias_shift = residual.mean(dim=0)
        
        # 2. 获取当前更新的长度
        current_len = current_bias_shift.shape[0]
        full_len = self.global_bias.shape[0]

        # 3. 只更新 Bias 中对应的前 current_len 部分
        if current_len <= full_len:
            # EMA 更新：只更新观测到的部分
            self.global_bias[:current_len] = (1 - self.alpha) * self.global_bias[:current_len] + \
                                             self.alpha * current_bias_shift
        else:
            # 理论上不应发生 current_len > full_len，除非配置错误
            raise ValueError(f"Current bias shift length {current_len} exceeds global bias length {full_len}. Check configuration.")



    def update_dynamic_memory(self, z_t, y_gt, y_final_pred):
        """
        OD-TTA v3.3 动态实例记忆更新 
        基于 Gram-Schmidt 正交化 + 阈值判定 + LFU 淘汰
        """

        # 安全检查：如果长度不够，直接返回
        current_len = y_gt.shape[1]
        if current_len < self.window_len:
            # [Safety] 长度不足，无法构建完整的 Value 向量，跳过更新
            return

        batch_size = z_t.shape[0]
        # Current Error (Residual that creates new pattern): (B, H, V) -> (B, V, H)
        # New Value = Y_gt - Y_final (当前剩余未被修正的误差)
        current_err = (y_gt - y_final_pred).permute(0, 2, 1)

        # 1. 构建当前所有已知基向量 (Static + Dynamic)
        # Static: (V, N, D) -> Permute to (N, V, D)
        basis_list = [F.normalize(self.static_keys, p=2, dim=-1).permute(1, 0, 2)]
        
        if self.dynamic_keys.shape[0] > 0:
            basis_list.append(self.dynamic_keys) # (K, V, D)
        
        # K_all: (Total_Keys, V, D)
        K_all = torch.cat(basis_list, dim=0)

        # 2. 对 Batch 中每个样本检查是否需要新增
        # 为了保持 tensor 效率，这里做一个简化：只要 Batch 中有任何一个样本触发阈值，就记录其平均特征
        # 或者更精细地：逐样本处理 (此处演示 Batch 均值处理，更适合在线流式)
        
        # 计算 z_t 在 K_all 上的投影 
        # z_t: (B, V, D), K_all: (T, V, D)
        
        # Proj coeff: (B, T, V)
        coeffs = torch.einsum('bvd, tvd -> btv', z_t, K_all)
        
        # Projection vector: (B, V, D)
        proj = torch.einsum('btv, tvd -> bvd', coeffs, K_all)
        
        # Orthogonal Residual: r_ortho
        r_ortho = z_t - proj
        
        # Energy: (B, V)
        energy = torch.norm(r_ortho, p=2, dim=-1)

        # --- 判定与执行 ---
        # 策略：如果 Batch 中平均能量 > 阈值，则新增
        mean_energy = energy.mean() # Scalar
        
        if mean_energy > self.energy_threshold:
            print(f"[PKA_OnLine] Adding new dynamic pattern. Mean energy: {mean_energy.item():.4f}")
            # 生成新 Key : 归一化的正交残差
            # 取 Batch 的平均方向作为新 Pattern
            r_ortho_mean = r_ortho.mean(dim=0) # (V, D)
            new_key = F.normalize(r_ortho_mean, p=2, dim=-1)
            
            # 生成新 Value : 剩余误差
            new_value = current_err.mean(dim=0) # (V, H)
            
            # LFU 初始化
            new_count = torch.tensor([5.0], device=z_t.device) # 给一点初始热度

            # Append to Buffer
            self.dynamic_keys = torch.cat([self.dynamic_keys, new_key.unsqueeze(0)], dim=0)
            self.dynamic_values = torch.cat([self.dynamic_values, new_value.unsqueeze(0)], dim=0)
            self.dynamic_counts = torch.cat([self.dynamic_counts, new_count], dim=0)

            # --- 容量管理 (LFU) ---
            # 如果超出容量，删除 Count 最小的
            if self.dynamic_keys.shape[0] > self.max_capacity:
                # 找出最小 Count 的索引
                min_idx = torch.argmin(self.dynamic_counts)
                
                # 创建保留 Mask
                keep_mask = torch.ones(self.dynamic_keys.shape[0], dtype=torch.bool, device=z_t.device)
                keep_mask[min_idx] = False
                
                # 执行删除
                self.dynamic_keys = self.dynamic_keys[keep_mask]
                self.dynamic_values = self.dynamic_values[keep_mask]
                self.dynamic_counts = self.dynamic_counts[keep_mask]
                
                # 归一化 Counts (防止无限增长)
                self.dynamic_counts = self.dynamic_counts - self.dynamic_counts.min()


class PKA_LDict(nn.Module):
    def __init__(self, window_len, n_var=1, seq_len=96, 
                 n_static=16, feature_dim=32, temperature=10.0,
                 bias_momentum=0.1, energy_threshold=0.1, max_dynamic_capacity=16, 
                 sim_threshold=0.9, ema_alpha=0.1, **kwargs):
        """
        OD-TTA v3.4 (Fixed): Bias-Augmented Orthogonal Prototype Memory
        包含 Global Bias (粗调) + Static Memory (离线精调, 稳定基底) + Dynamic Memory (独立路由, 在线长尾)
        """
        super(PKA_LDict, self).__init__()
        self.seq_len = seq_len      
        self.window_len = window_len
        self.n_var = n_var
        self.feature_dim = feature_dim
        self.alpha = bias_momentum
        self.temperature = temperature
        self.n_static = n_static
        self.max_capacity = max_dynamic_capacity 
        self.energy_threshold = energy_threshold 
        self.sim_threshold = sim_threshold 
        self.ema_alpha = ema_alpha 

        if feature_dim <= n_static:
            self.feature_dim = n_static * 2

        input_len = seq_len + window_len
        # 定义 QueryNet_TimeCI (输出 B, V, D)
        self.query_net = QueryNet_TimeCI(input_len, n_var, self.feature_dim)
        # self.query_net = QueryNet_Freq_Base_ChannelDependence(input_len, n_var, self.feature_dim)

        # --- 1. Static Memory (Offline) ---
        self.static_keys = nn.Parameter(torch.randn(n_var, n_static, self.feature_dim))
        self.static_values = nn.Parameter(torch.zeros(n_var, n_static, self.window_len))
        
        # 正交初始化
        for v in range(n_var):
            nn.init.orthogonal_(self.static_keys[v])
        nn.init.zeros_(self.static_values)

        # --- 2. Dynamic Memory (Online) ---
        self.dynamic_keys = nn.Parameter(torch.empty(0, n_var, self.feature_dim))
        self.dynamic_values = nn.Parameter(torch.empty(0, n_var, self.window_len))
        self.register_buffer('dynamic_counts', torch.empty(0, dtype=torch.float32))

    def _get_query(self, x, y_base):
        query_input = torch.cat([x, y_base], dim=1) 
        query = self.query_net(query_input) # (B, V, D)
        query = F.normalize(query, p=2, dim=-1) # 归一化
        return query

    def forward(self, y_base, x=None):
        batch_size = y_base.shape[0]
        z_t = self._get_query(x, y_base)

        # ==========================================
        # 1. 静态检索 (使用 Softmax 保持离线一致性)
        # ==========================================
        static_keys = F.normalize(self.static_keys, p=2, dim=-1)
        sim_static = torch.einsum('bvd, vnd -> bvn', z_t, static_keys)
        w_static = F.softmax(self.temperature * sim_static, dim=-1)
        delta_static = torch.einsum('bvn, vnh -> bvh', w_static, self.static_values)
        delta_static = delta_static.permute(0, 2, 1)

        # ==========================================
        # 2. 动态检索 (独立路由机制，防止稀释静态权重)
        # ==========================================
        delta_dynamic = torch.zeros_like(y_base)
        
        if self.dynamic_keys.shape[0] > 0:
            sim_dynamic = torch.einsum('bvd, kvd -> bvk', z_t, self.dynamic_keys)
            
            # 使用带阈值的线性截断 (Thresholded Gating) 替代 Softmax
            # 如果相似度 <= threshold，权重为 0；如果相似度为 1.0，权重为 1.0
            # w_dynamic = torch.clamp((sim_dynamic - self.sim_threshold) / (1.0 - self.sim_threshold + 1e-5), min=0.0)
            w_dynamic = torch.relu((sim_dynamic - self.sim_threshold), min=0.0)
            # w_dynamic = sim_dynamic

            # 更新使用计数
            if self.training or True: 
                best_k = sim_dynamic.argmax(dim=-1) 
                max_sim_k, _ = sim_dynamic.max(dim=-1)
                valid_mask = max_sim_k > self.sim_threshold
                
                # 只统计有效激活的 Key
                unique_k = torch.unique(best_k[valid_mask])
                for k_idx in unique_k:
                    if k_idx < self.dynamic_counts.shape[0]:
                        self.dynamic_counts[k_idx] += 1.0

            delta_dynamic = torch.einsum('bvk, kvh -> bvh', w_dynamic, self.dynamic_values)
            delta_dynamic = delta_dynamic.permute(0, 2, 1)

        # ==========================================
        # 3. 最终融合
        # ==========================================
        y_final = y_base + delta_static + delta_dynamic
        
        return y_final, z_t

    def update_dynamic_memory(self, z_t, y_gt, y_base, y_final_pred):
        add_pattern_flag = False
        current_len = y_gt.shape[1]
        if current_len < self.window_len:
            return add_pattern_flag

        # 计算当前残差误差 (B, V, H)
        # current_err = (y_gt - y_final_pred).permute(0, 2, 1) 
        current_err = (y_gt - y_base).permute(0, 2, 1) 

        # ==========================================================
        # 1. 组合基向量用于计算新颖度 (Novelty Check)
        # ==========================================================
        # static_keys 形状为 (V, N, D)，需要转换为 (N, V, D) 以统一维度
        basis_list = [self.static_keys.permute(1, 0, 2)]
        if self.dynamic_keys.shape[0] > 0:
            # dynamic_keys 已经是 (K, V, D)
            basis_list.append(F.normalize(self.dynamic_keys, p=2, dim=-1))
        
        K_all = torch.cat(basis_list, dim=0) # (T, V, D), 其中 T = N + K

        # ==========================================================
        # 2. 计算最大相似度并转化为“新颖度能量” (Novelty Energy)
        # [核心修复] 彻底抛弃导致度量爆炸的投影计算，改用距离度量
        # ==========================================================
        # 计算当前查询 z_t 与所有已知原型 (静态+动态) 的余弦相似度
        # z_t: (B, V, D), K_all: (T, V, D) -> sim_all: (B, V, T)
        sim_all = torch.einsum('bvd, tvd -> bvt', z_t, K_all)
        
        # 找到与现有所有知识最接近的匹配度
        max_sim, _ = sim_all.max(dim=-1) # (B, V)
        
        # 定义能量 (新颖度) = 1.0 - 最大相似度
        # 这种计算方式确保 energy 的绝对值域被严格限制在 [0.0, 2.0] 之间，绝不会爆炸
        energy = 1.0 - max_sim # (B, V)

        # ==========================================================
        # 3. Hard Mining + 新增模式
        # ==========================================================
        max_energy_per_sample = energy.mean(dim=1) # (B,)
        best_sample_idx = torch.argmax(max_energy_per_sample)
        max_energy = max_energy_per_sample[best_sample_idx]

        # 注意：这里的 energy_threshold 语义已变为 "允许的最大不相似度距离"
        # 建议在初始化时将 self.energy_threshold 设置为 0.2 或 0.3 (意味着当 max_sim < 0.8 甚至 0.7 时才触发)
        if max_energy > self.energy_threshold:
            add_pattern_flag = True
            print(f"[PKA_OnLine] New Pattern Triggered! Max novelty energy: {max_energy.item():.4f}")
            
            # 直接保存真实的、未被扭曲的查询向量 z_t
            new_key = F.normalize(z_t[best_sample_idx], p=2, dim=-1) # (V, D)
            new_value = current_err[best_sample_idx] # (V, H)

            # 新增节点的初始使用次数计数
            new_count = torch.tensor([5.0], device=z_t.device)

            # 追加新模式
            # Re-wrap as Parameter to allow gradient updates
            new_keys_data = torch.cat([self.dynamic_keys.data, new_key.unsqueeze(0)], dim=0)
            new_values_data = torch.cat([self.dynamic_values.data, new_value.unsqueeze(0)], dim=0)
            # new_values_data = torch.cat([self.dynamic_values.data, torch.zeros_like(new_value.unsqueeze(0))], dim=0)
            
            self.dynamic_keys = nn.Parameter(new_keys_data)
            self.dynamic_values = nn.Parameter(new_values_data)

            self.dynamic_counts = torch.cat([self.dynamic_counts, new_count], dim=0)

            # 容量管理淘汰机制 (LFU 变体)
            if self.dynamic_keys.shape[0] > self.max_capacity:
                min_idx = torch.argmin(self.dynamic_counts)
                keep_mask = torch.ones(self.dynamic_keys.shape[0], dtype=torch.bool, device=z_t.device)
                keep_mask[min_idx] = False
                
                self.dynamic_keys = nn.Parameter(self.dynamic_keys.data[keep_mask])
                self.dynamic_values = nn.Parameter(self.dynamic_values.data[keep_mask])
                self.dynamic_counts = self.dynamic_counts[keep_mask]
                
                # 防止 count 的相对差异过大，每次淘汰后平移重置基线
                self.dynamic_counts = self.dynamic_counts - self.dynamic_counts.min()
        
        return add_pattern_flag