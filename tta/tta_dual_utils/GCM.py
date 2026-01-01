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

        self.analyzer = CoBA_Analyzer(self)
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
            # nn.GELU(),
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
        # print(coeffs)
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

    def get_optim_params(self):
        params = []
        params.append(self.gating)
        params.append(self.bias)
        return params

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        plt.show()

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
        plt.show()

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
            input_tensor = input_tensor.cuda()

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
        plt.show()

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