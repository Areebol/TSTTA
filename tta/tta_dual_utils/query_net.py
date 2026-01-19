import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def stable_complex_abs(z):
    x = torch.abs(z.real)
    y = torch.abs(z.imag)
    m = torch.maximum(x, y)
    r = torch.minimum(x, y) / (m + 1e-12)
    return m * torch.sqrt(1 + r * r)

# ==========================================
# 1. Baseline A: 纯时域 (Time-Only)
# ==========================================
class QueryNet_Time(nn.Module):
    """
    Baseline A: 直接使用时域数据。
    优点: 保留原始突变信息。
    缺点: 容易受高频噪声干扰，对长周期特征不敏感。
    """
    def __init__(self, window_len, n_var, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window_len * n_var, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, x):
        # x: (B, L, V) -> (B, L*V)
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)
        return self.net(x_flat)


# ==========================================
# 2. Baseline B: 基础频域 (Freq-Base, 原版)
# ==========================================
class QueryNet_Freq_Base_ChannelDependence(nn.Module):
    """
    Baseline B: 原版实现，只使用 FFT 幅度。
    优点: 对周期性强，平滑噪声。
    缺点: 丢失相位（时间位置）信息。
    """
    def __init__(self, window_len, n_var, feature_dim):
        super().__init__()
        fft_len = window_len // 2 + 1
        self.net = nn.Sequential(
            nn.Linear(fft_len * n_var, feature_dim * 2),
            # nn.GELU(), # 原版代码注释掉了激活函数，保持一致
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, x):
        # x: (B, L, V)
        batch_size = x.size(0)
        x_fft = torch.fft.rfft(x, dim=1)
        # x_mag = torch.abs(x_fft) # Magnitude
        x_mag = stable_complex_abs(x_fft)  # 更稳定的幅度计算
        x_feat = x_mag.reshape(batch_size, -1)
        query = self.net(x_feat)
        query = query.unsqueeze(1)
        return query

class QueryNet_Freq_Hybrid(nn.Module):
    """
    Hybrid QueryNet:
    结合了 'Global Context' (变量间依赖) 和 'Local Feature' (变量独立特征)。
    输出: (Batch, N_var, Feature_dim)
    """
    def __init__(self, window_len, n_var, feature_dim, global_hidden=64):
        super().__init__()
        fft_len = window_len // 2 + 1
        
        # 1. 全局分支 (Global Branch)
        # 负责“看森林”：压缩所有变量的信息到一个小的全局向量
        # 为了节省参数，我们先对 Input 做一个降维或者 Pooling
        self.global_net = nn.Sequential(
            # 输入: 所有变量的 FFT 拼接 -> 极其巨大，所以我们先用一个技巧
            # 技巧: 不直接全连接，而是先 Embedding 再聚合，或者直接处理压缩后的特征
            # 这里为了简单有效，我们对频谱在变量维度求平均 (Mean Pooling) 作为全局输入
            nn.Linear(fft_len, global_hidden), 
            nn.GELU(),
            nn.Linear(global_hidden, feature_dim) 
        )
        
        # 2. 局部分支 (Local Branch - Shared Weights)
        # 负责“看树木”：独立处理每个变量
        self.local_net = nn.Sequential(
            nn.Linear(fft_len, feature_dim),
            nn.GELU()
        )
        
        # 3. 融合层 (Fusion Layer)
        # 将 Global(D) + Local(D) -> Query(D)
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            # nn.Tanh() # 限制输出范围，让余弦相似度计算更稳定
        )

    def forward(self, x):
        # x: (Batch, Window, N_var)
        batch_size = x.size(0)
        n_var = x.size(2)
        
        # --- 预处理: FFT ---
        x_fft = torch.fft.rfft(x, dim=1)
        x_mag = stable_complex_abs(x_fft) # Shape: (B, Freq, V)
        
        # 调整为 (B, V, Freq) 以便处理
        x_feat = x_mag.permute(0, 2, 1) 
        
        # --- A. 局部分支 (Local) ---
        # 独立提取每个变量的特征
        # (B, V, F) -> (B, V, D)
        local_emb = self.local_net(x_feat)
        
        # --- B. 全局分支 (Global) ---
        # 提取全局上下文。这里使用 Mean Pooling 聚合所有变量，
        # 既保留了整体趋势（如整体周期性），又避免了参数爆炸。
        # (B, V, F) -> mean -> (B, F)
        global_input = x_feat.mean(dim=1) 
        global_emb = self.global_net(global_input) # (B, D)
        
        # --- C. 融合 (Fusion) ---
        # 将全局向量 (B, D) 扩展为 (B, V, D) 并与局部向量拼接
        global_emb_expanded = global_emb.unsqueeze(1).expand(-1, n_var, -1)
        
        # Concatenate: (B, V, 2*D)
        combined = torch.cat([local_emb, global_emb_expanded], dim=-1)
        
        # 生成最终的 Query: (B, V, D)
        queries = self.fusion_gate(combined)
        
        return queries

class QueryNet_Freq_Base_ChannelIndependence(nn.Module):
    """
    修改版: 支持 Var-wise Query 生成
    Input:  (B, L, V)
    Output: (B, V, feature_dim)
    """
    def __init__(self, window_len, n_var, feature_dim):
        super().__init__()
        self.n_var = n_var
        # RFFT 后的长度
        fft_len = window_len // 2 + 1
        
        # --- 核心修改 ---
        # 1. 输入维度不再乘以 n_var，而是针对单个变量的频谱长度
        # 2. 我们希望对每个变量独立处理，使用 Shared MLP (对 dim=-1 作用)
        self.net = nn.Sequential(
            nn.Linear(fft_len, feature_dim * 2),
            nn.GELU(), # 建议加上激活函数，增加非线性能力
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, x):
        """
        x: (Batch, Window_len, N_var)
        """
        # 1. FFT 变换
        # x_fft: (Batch, Freq_len, N_var)
        x_fft = torch.fft.rfft(x, dim=1)
        
        # 2. 计算幅度
        # x_mag: (Batch, Freq_len, N_var)
        x_mag = stable_complex_abs(x_fft)

        # 3. 维度调整 (Permute)
        # 我们希望 Linear 层独立作用于每个变量的频谱
        # 目标形状: (Batch, N_var, Freq_len)
        x_feat = x_mag.permute(0, 2, 1)

        # 4. 通过 MLP
        # nn.Linear 默认作用于最后一个维度 (Freq_len)
        # Input: (B, V, Freq_len) -> Output: (B, V, Feature_dim)
        query = self.net(x_feat)
        
        return query


# ==========================================
# 3. 方案一: 时频双域自适应门控 (Fusion-Gated) -> 论文核心创新
# ==========================================
class QueryNet_Fusion_Gated(nn.Module):
    """
    Proposed Method: 时频双支路 + 自适应门控。
    创新点: 动态决定关注频域(周期)还是时域(突变)。
    """
    def __init__(self, window_len, n_var, feature_dim):
        super().__init__()
        
        # --- 频域支路 ---
        fft_len = window_len // 2 + 1
        self.freq_net = nn.Sequential(
            nn.Linear(fft_len * n_var, feature_dim),
            nn.LeakyReLU()
        )
        
        # --- 时域支路 (使用轻量级 TCN 提取波形特征) ---
        self.time_net = nn.Sequential(
            # Input: (B, V, L)
            nn.Conv1d(n_var, feature_dim, kernel_size=3, padding=1), 
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Pooling -> (B, D, 1)
        )
        
        # --- 门控网络 ---
        # 输入是 [Freq_emb, Time_emb]，输出一个 0-1 的权重 alpha
        self.gate_generator = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid() 
        )
        
        # self.final_proj = nn.Linear(feature_dim, feature_dim)
        self.final_proj = nn.Linear(2 * feature_dim, feature_dim)

    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. Frequency Path (Log Magnitude for stability)
        # x_fft = torch.fft.rfft(x, dim=1).abs()
        x_fft = stable_complex_abs(torch.fft.rfft(x, dim=1))  # 更稳定的幅度计算
        x_fft_log = torch.log(x_fft + 1e-6) # Log scaling
        freq_emb = self.freq_net(x_fft_log.reshape(batch_size, -1))
        
        # 2. Time Path
        x_perm = x.permute(0, 2, 1) # (B, V, L)
        time_emb = self.time_net(x_perm).squeeze(-1) # (B, D)
        
        # 3. Adaptive Gating
        combined = torch.cat([freq_emb, time_emb], dim=-1)
        # alpha = self.gate_generator(combined) # (B, 1)
        
        # 4. Weighted Fusion
        # alpha 接近 1 代表更关注频域，接近 0 代表更关注时域 (可互换定义)
        # fused = alpha * freq_emb + (1 - alpha) * time_emb
        
        # return self.final_proj(fused)
        return self.final_proj(combined)


# ==========================================
# 4. 方案二: 多尺度时域卷积 (Multi-Scale)
# ==========================================
class QueryNet_MultiScale(nn.Module):
    """
    Variant 2: 多尺度 CNN。
    适用: 无明显周期，但有局部趋势特征的数据。
    """
    def __init__(self, window_len, n_var, feature_dim):
        super().__init__()
        # Inception-like blocks
        self.conv_s = nn.Conv1d(n_var, feature_dim // 2, kernel_size=3, padding=1)
        self.conv_m = nn.Conv1d(n_var, feature_dim // 2, kernel_size=7, padding=3)
        self.conv_l = nn.Conv1d(n_var, feature_dim // 2, kernel_size=11, padding=5)
        
        self.proj = nn.Linear(feature_dim // 2 * 3, feature_dim)

    def forward(self, x):
        # x: (B, L, V) -> (B, V, L)
        x = x.permute(0, 2, 1)
        
        feat_s = F.adaptive_max_pool1d(F.relu(self.conv_s(x)), 1).squeeze(-1)
        feat_m = F.adaptive_max_pool1d(F.relu(self.conv_m(x)), 1).squeeze(-1)
        feat_l = F.adaptive_max_pool1d(F.relu(self.conv_l(x)), 1).squeeze(-1)
        
        combined = torch.cat([feat_s, feat_m, feat_l], dim=-1)
        return self.proj(combined)

class QueryNet_Freq_Attn(nn.Module):
    """
    Frequency-aware Router with:
    - magnitude + phase
    - stable embedding dimension per variable
    - freq-attention over frequency axis
    - multi-band pooling
    """

    def __init__(self, window_len, n_var, feature_dim,
                 embed_dim=32, n_heads=4, n_bands=4):
        super().__init__()
        fft_len = window_len // 2 + 1
        self.fft_len = fft_len
        self.n_var = n_var
        self.n_bands = n_bands
        self.embed_dim = embed_dim

        # (mag, phase) -> embed_dim
        self.var_embed = nn.Linear(2, embed_dim)

        # attn operates on (B, F, V*embed_dim)
        self.freq_attn = nn.MultiheadAttention(
            embed_dim=n_var * embed_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        # F -> n_bands
        self.band_pool = nn.Linear(fft_len, n_bands)

        self.mlp = nn.Sequential(
            nn.Linear(n_bands * n_var * embed_dim, feature_dim * 2),
            # nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, x):
        B, L, V = x.shape
        assert V == self.n_var

        x_fft = torch.fft.rfft(x, dim=1)
        # mag = torch.abs(x_fft)
        mag = stable_complex_abs(x_fft)  # 更稳定的幅度计算
        phase = torch.angle(x_fft)

        # (B, F, V, 2)
        feat = torch.stack([mag, phase], dim=-1)

        # (B, F, V, embed_dim)
        feat = self.var_embed(feat)

        # (B, F, V*embed_dim)
        feat = feat.reshape(B, self.fft_len, V * self.embed_dim)

        # frequency attention
        feat_attn, _ = self.freq_attn(feat, feat, feat)

        # transpose for band pooling: (B, V*embed_dim, F)
        feat_attn_t = feat_attn.transpose(1, 2)

        # (B, V*embed_dim, n_bands)
        feat_band = self.band_pool(feat_attn_t)

        # flatten
        feat_flat = feat_band.reshape(B, -1)

        return self.mlp(feat_flat)

class QueryNet_Freq_Light(nn.Module):

    def __init__(self, window_len, n_var, feature_dim, n_bands=4):
        super().__init__()
        fft_len = window_len // 2 + 1
        self.n_var = n_var
        self.n_bands = n_bands

        self.band_size = fft_len // n_bands
        self.use_len = self.band_size * n_bands   # 舍弃剩余频率

        self.mlp = nn.Sequential(
            nn.Linear(n_bands * n_var, feature_dim * 2),
            # nn.GELU(),
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, x):
        B, L, V = x.shape
        x_fft = torch.fft.rfft(x, dim=1)
        mag = torch.log1p(stable_complex_abs(x_fft))

        # 只取可整除部分
        mag = mag[:, :self.use_len, :]  # (B, use_len, V)

        # 均分 n_bands 片
        bands = mag.reshape(B, self.n_bands, self.band_size, V)

        # 每个 band pooling → (B, n_bands, V)
        pooled = bands.mean(dim=2)

        # concat bands → (B, n_bands * V)
        feat = pooled.reshape(B, self.n_bands * V)

        return self.mlp(feat)



# ==========================================
# 5. 方案三: 增强频域 (Phase-Aware)
# ==========================================
class QueryNet_Phase(nn.Module):
    """
    Variant 3: 保留相位信息的频域。
    适用: 信号处理任务，需要精确定位波形位置。
    """
    def __init__(self, window_len, n_var, feature_dim):
        super().__init__()
        fft_len = window_len // 2 + 1
        # 输入维度翻倍，因为拼接了 Real 和 Imag
        self.net = nn.Sequential(
            nn.Linear(fft_len * n_var * 2, feature_dim * 2),
            # nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_fft = torch.fft.rfft(x, dim=1)
        # 拼接实部和虚部
        x_feat = torch.cat([x_fft.real, x_fft.imag], dim=-1)
        x_feat = x_feat.reshape(batch_size, -1)
        return self.net(x_feat)
  
  
try:
    from pytorch_wavelets import DWT1D
    WAVELET_AVAILABLE = True
except:
    WAVELET_AVAILABLE = False
      
class QueryNet_Wavelet_MS(nn.Module):
    """
    Multi-Scale Wavelet Gating (MSWG)

    x: (B, L, V)  # V = #variables
    输出: (B, feature_dim), 输入给 MoE Router
    """

    def __init__(self, window_len, n_var, feature_dim, wave='haar', level=3):
        super().__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.level = level

        # --- 1. Wavelet Transform ---
        if WAVELET_AVAILABLE:
            self.dwt = DWT1D(wave=wave, J=level, mode='symmetric')
        else:
            raise ImportError("Please install pytorch_wavelets: pip install pytorch_wavelets")

        # 多尺度后，有 (cA3, cD3, cD2, cD1) 共 (level + 1) 个频带
        n_bands = level + 1   # 3-level → 4 bands

        # --- 2. MLP ---
        self.mlp = nn.Sequential(
            nn.Linear(n_bands * n_var, feature_dim * 2),
            nn.GELU(),
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, x):
        """
        x: (B, L, V)
        """
        B, L, V = x.shape

        # 逐变量做 DWT
        bands = []  # list of tensors, each shape (B, V)

        # 对每个变量单独进行 DWT（保持通道独立）
        for v in range(V):
            xv = x[:, :, v].unsqueeze(1)  # (B, 1, L)
            cA, cD_list = self.dwt(xv)    # cA: lowest freq | cD_list: high→lower freq

            # cA: (B, 1, L//8), cD_list = [cD3, cD2, cD1]
            cA_pool = torch.mean(cA, dim=-1)  # (B, 1)
            bands.append(cA_pool)

            for cD in cD_list:  # from high freq to lower freq
                cD_pool = torch.mean(cD, dim=-1)  # (B, 1)
                bands.append(cD_pool)

        # bands: list length = V * (level + 1)，每个项是 (B,1)
        feat = torch.cat(bands, dim=1)  # (B, V*(level+1))

        return self.mlp(feat)
    
class QueryNet_Freq_MagPhase_ChannelIndependence(nn.Module):
    """
    修改版: 支持 模长(Magnitude) + 周期/相位(Phase) 拼接输入的 Var-wise Query 生成
    Input:  (B, L, V)
    Output: (B, V, feature_dim)
    """
    def __init__(self, window_len, n_var, feature_dim):
        super().__init__()
        self.n_var = n_var
        # RFFT 后的单侧频谱长度
        fft_len = window_len // 2 + 1
        
        # --- 核心修改 ---
        # 输入维度为 fft_len * 2 (模长 + 相位)
        # 使用 Shared MLP 独立处理每个变量 (Channel Independence)
        self.net = nn.Sequential(
            nn.Linear(fft_len * 2, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, x):
        """
        x: (Batch, Window_len, N_var)
        """
        # 1. FFT 变换 (保持原始量纲)
        # x_fft: (Batch, Freq_len, N_var)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        
        # 2. 计算模长 (Magnitude) 和 相位 (Phase/Angle)
        # x_mag: 信号的强度 (能量分布)
        # x_phase: 信号的周期/时间对齐信息
        x_mag = stable_complex_abs(x_fft)
        x_phase = torch.atan2(x_fft.imag, x_fft.real)

        # 3. 维度调整 (转置为 Batch, N_var, Freq_len)
        x_mag = x_mag.permute(0, 2, 1)
        x_phase = x_phase.permute(0, 2, 1)

        # 4. 在特征维度拼接 模长 和 相位
        # 结果形状: (Batch, N_var, Freq_len * 2)
        x_feat = torch.cat([x_mag, x_phase], dim=-1)

        # 5. 通过 Shared MLP 生成 Query
        # Input: (B, V, Freq_len * 2) -> Output: (B, V, Feature_dim)
        query = self.net(x_feat)
        
        return query