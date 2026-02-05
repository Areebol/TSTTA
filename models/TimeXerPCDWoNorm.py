import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np


# --- 1. 新增：通道融合输出头 (保持不变) ---
class FusedFlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.target_window = target_window
        self.flatten_patch = nn.Flatten(start_dim=-2)
        self.input_dim = n_vars * nf
        self.output_dim = n_vars * target_window
        self.linear_fusion = nn.Linear(self.input_dim, self.output_dim)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x): 
        x = self.flatten_patch(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear_fusion(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], self.n_vars, self.target_window)
        return x

# --- 2. 原始：独立展平输出头 (保持不变) ---
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

# --- 3. 基础组件 (EnEmbedding, Encoder, EncoderLayer 保持不变) ---
class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars

class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)
        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0])
        x_glb_attn = torch.reshape(x_glb_attn, (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)
        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        
        # --- [修改点 1] 配置目标维度和外生变量维度 ---
        # configs.target_dims 应为一个列表，例如 [12, 13]
        # 如果未指定，默认行为是取最后一维 [-1]
        if hasattr(configs, 'target_dims') and configs.target_dims:
            self.target_dims = [12, 13]
        else:
            self.target_dims = [-1]

        # 预计算：所有特征的索引列表
        all_indices = np.arange(configs.enc_in)
        
        # 将 target_dims 转为正整数索引 (处理 -1 等负数索引情况)
        self.target_ids = [i if i >= 0 else configs.enc_in + i for i in self.target_dims]
        
        # 计算外生变量的索引 (Exogenous indices = Total - Targets)
        self.exo_ids = [i for i in all_indices if i not in self.target_ids]
        
        # 转为 PyTorch LongTensor 以便后续 index_select 使用
        # 注册为 buffer，这样 device 会自动管理
        self.register_buffer('target_ids_tensor', torch.LongTensor(self.target_ids))
        self.register_buffer('exo_ids_tensor', torch.LongTensor(self.exo_ids))

        # --- [保留] 不归一化的配置 ---
        self.no_norm_indices = getattr(configs, 'no_norm_indices', [6, 7, 8, 17, 18, 19])

        # Embedding
        # 注意：EnEmbedding 用于处理 Patch (主要用于目标变量)
        # ExEmbedding 用于处理外生变量
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads),
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.use_fused_head = getattr(configs, 'use_fused_head', True)
        
        if self.use_fused_head:
            # 注意：Head 的输出维度取决于我们要预测多少个变量 (len(self.target_ids))
            # 如果是 MS 任务，这里需要调整 self.n_vars 的含义，或者让 Head 输出 len(target_dims)
            out_vars = len(self.target_ids) if self.features == 'MS' else self.n_vars
            self.head = FusedFlattenHead(out_vars, self.head_nf, configs.pred_len, head_dropout=configs.dropout)
        else:
            out_vars = len(self.target_ids) if self.features == 'MS' else self.n_vars
            self.head = FlattenHead(out_vars, self.head_nf, configs.pred_len, head_dropout=configs.dropout)

    def _get_statistics(self, x):
        dim2reduce = [1]
        means = x.mean(dim2reduce, keepdim=True).detach()
        x_centered = x - means
        stdev = torch.sqrt(torch.var(x_centered, dim=dim2reduce, keepdim=True, unbiased=False) + 1e-5)
        
        if self.no_norm_indices:
            indices = [i for i in self.no_norm_indices if i < x.shape[2]]
            if indices:
                means[:, :, indices] = 0.0
                stdev[:, :, indices] = 1.0
        return means, stdev

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        适用于 features='MS' 或 'S'。
        即：利用多变量输入，预测指定的 Target 变量。
        """
        if self.use_norm:
            means, stdev = self._get_statistics(x_enc)
            x_enc = (x_enc - means) / stdev

        # --- [修改点 2] 根据索引拆分数据 ---
        # 提取目标变量 (Batch, Seq, Num_Targets)
        x_enc_target = torch.index_select(x_enc, 2, self.target_ids_tensor)
        
        # 提取外生变量 (Batch, Seq, Num_Exo)
        x_enc_exo = torch.index_select(x_enc, 2, self.exo_ids_tensor)

        # --- [修改点 3] 分别 Embedding ---
        # 1. 对目标变量进行 Patch Embedding
        # EnEmbedding 期望输入: (Batch, N_vars, Seq_len) -> 需要 permute
        # x_enc_target.shape: [B, L, N_tar] -> [B, N_tar, L]
        en_embed, n_vars = self.en_embedding(x_enc_target.permute(0, 2, 1))
        
        # 2. 对外生变量进行 Inverted Embedding (保持原状)
        ex_embed = self.ex_embedding(x_enc_exo, x_mark_enc)

        # Encoder 前向传播
        enc_out = self.encoder(en_embed, ex_embed)
        
        # Reshape 输出
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2) # [B, N_tar, D_model, Patch_num+1]

        # Decoder Head
        dec_out = self.head(enc_out) # Output: [B, N_tar, Pred_Len]
        dec_out = dec_out.permute(0, 2, 1) # Output: [B, Pred_Len, N_tar]

        if self.use_norm:
            # --- [修改点 4] 仅对目标变量反归一化 ---
            # 获取目标对应的 stdev 和 means
            # stdev shape: [B, 1, Total_Vars] -> index_select -> [B, 1, N_tar]
            target_stdev = torch.index_select(stdev, 2, self.target_ids_tensor)
            target_means = torch.index_select(means, 2, self.target_ids_tensor)
            
            dec_out = dec_out * (target_stdev.repeat(1, self.pred_len, 1))
            dec_out = dec_out + (target_means.repeat(1, self.pred_len, 1))
            
        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        适用于 features='M'。
        通常预测所有变量。如果你的需求是 'M' 模式但也只看重那两列，
        通常这里还是输出所有列，你在 Loss 计算时只取那两列即可。
        """
        if self.use_norm:
            means, stdev = self._get_statistics(x_enc)
            x_enc = (x_enc - means) / stdev

        # M 模式下，所有变量都作为 Patch Embedding 进入 Encoder
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * (stdev.repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means.repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # 如果是 MS 模式（输入多变量，预测特定变量），走 forecast
            if self.features == 'MS' or self.features == 'S':
                return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
            else:
                # M 模式，预测所有
                return self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
        return None