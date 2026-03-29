import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==================== 统一PCD跨通道融合头（严格对齐PatchTSTPCD） ====================
class FusedFlattenHead(nn.Module):
    def __init__(self, seq_len, embed_size, hidden_size, n_vars, pred_len, head_dropout=0.):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = pred_len
        
        # 适配FreTS维度：输入 = 变量数 × (seq_len × embed_size)
        self.input_dim = n_vars * (seq_len * embed_size)
        self.output_dim = n_vars * pred_len
        
        # 将隐层维度同步放大 n_vars 倍，否则会造成极严重的特征压缩/丢失！
        self.scaled_hidden_size = n_vars * hidden_size
        
        # 跨通道全局融合层
        self.fusion = nn.Sequential(
            nn.Linear(self.input_dim, self.scaled_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.scaled_hidden_size, self.output_dim),
            nn.Dropout(head_dropout)
        )

    def forward(self, x):
        # x: [B, n_vars, seq_len*embed_size]
        B, n_vars, _ = x.shape
        # 核心：展平所有变量特征，实现跨通道交互
        x = x.reshape(B, -1)
        # 全局融合预测
        x = self.fusion(x)
        # 重塑为输出格式 [B, n_vars, pred_len]
        x = x.reshape(B, n_vars, self.pred_len)
        return x


class Model(nn.Module):
    """
    FreTSPCD：基于FreTS的跨通道融合改进版
    严格遵循PatchTSTPCD改造规则：主干不变，仅替换预测头
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # self.embed_size = 128  # embed_size
        # self.hidden_size = 256  # hidden_size

        self.embed_size = getattr(configs, 'd_model', 32) 
        self.hidden_size = getattr(configs, 'd_ff', 64)

        self.pred_len = configs.pred_len
        self.feature_size = configs.enc_in  # channels
        self.seq_len = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        # ==================== PCD核心开关（默认开启融合头） ====================
        self.use_fused_head = getattr(configs, "use_fused_head", True)
        self.dropout = getattr(configs, "dropout", 0.1)

        # 原版FreTS频域模块（完全保留，无修改）
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        # ==================== 预测头：PCD融合头 / 原版fc 二选一 ====================
        if self.use_fused_head:
            print(f"FreTSPCD Using FusedFlattenHead: Channel Interaction Enabled at Output Layer.")
            self.fc = FusedFlattenHead(
                seq_len=self.seq_len,
                embed_size=self.embed_size,
                hidden_size=self.hidden_size,
                n_vars=self.feature_size,
                pred_len=self.pred_len,
                head_dropout=self.dropout
            )
        else:
            # 原版FreTS预测头
            self.fc = nn.Sequential(
                nn.Linear(self.seq_len * self.embed_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.pred_len)
            )

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_len, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
    # rb: the real part of bias, ib: the imaginary part of bias
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bijd,dd->bijd', x.real, r) - \
            torch.einsum('bijd,dd->bijd', x.imag, i) + \
            rb
        )

        o1_imag = F.relu(
            torch.einsum('bijd,dd->bijd', x.imag, r) + \
            torch.einsum('bijd,dd->bijd', x.real, i) + \
            ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    # ==================== 核心：forecast函数与原版一致，无任何修改 ====================
    def forecast(self, x_enc):
        # x: [Batch, Input length, Channel]
        B, T, N = x_enc.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x_enc)
        bias = x
        # [B, N, T, D]
        if self.channel_independence == '0':
            x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')