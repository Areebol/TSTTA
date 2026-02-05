import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np

# --- 1. MLP 模块 (保持不变) ---
class RoadConditioning(nn.Module):
    def __init__(self, road_dim, d_model, patch_num):
        super().__init__()
        self.patch_num = patch_num
        self.mlp = nn.Sequential(
            nn.Linear(road_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x_road):
        x = x_road.permute(0, 2, 1)
        x = F.adaptive_avg_pool1d(x, self.patch_num)
        x = x.permute(0, 2, 1)
        x_emb = self.mlp(x)
        return x_emb

# --- 2. Head 模块 (保持不变) ---
class ChannelMixingHead(nn.Module):
    def __init__(self, in_vars, out_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.target_window = target_window
        
        self.channel_fusion = nn.Linear(in_vars, 1)
        self.linear_pred = nn.Linear(nf, out_vars * target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        B, V, D, P = x.shape
        x = x.reshape(B, V, -1) 
        x = x.transpose(1, 2)
        x = self.channel_fusion(x).squeeze(-1) 
        x = self.linear_pred(x)
        x = self.dropout(x)
        x = x.reshape(B, self.out_vars, self.target_window)
        return x

# --- 3. Embedding & Encoder (保持不变) ---
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


# --- 4. 修改后的 Model 类 ---
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = configs.enc_in
        
        # --- 变量拆分 ---
        # 排除指定索引，构建道路特征 ID
        assert configs.enc_in >= 20, "Encoder input dimensions must be >= 20"
        exclude = {4, 10, 11, 12, 13, 15}
        default_road_ids = [i for i in range(20) if i not in exclude]
        
        self.road_ids = configs.get('road_dims', default_road_ids)
        self.road_ids = [i if i >= 0 else self.n_vars + i for i in self.road_ids]
        
        # 剩余为动态特征
        all_indices = np.arange(configs.enc_in)
        self.dynamic_ids = [i for i in all_indices if i not in self.road_ids]
        
        self.register_buffer('dynamic_ids_tensor', torch.LongTensor(self.dynamic_ids))
        self.register_buffer('road_ids_tensor', torch.LongTensor(self.road_ids))

        # --- Embedding ---
        self.n_dynamic = len(self.dynamic_ids)
        self.n_road = len(self.road_ids)

        # 1. 动态变量 Embedding
        self.en_embedding = EnEmbedding(self.n_dynamic, configs.d_model, self.patch_len, configs.dropout)
        
        # 2. 道路变量处理
        # Path A: MLP 提取
        self.road_mlp_embedding = RoadConditioning(self.n_road, configs.d_model, self.patch_num)
        # Path B: Cross Attention 输入
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # --- [新增] 特征融合层 (Concat + Linear) ---
        # 输入维度为 2 * d_model (动态特征 + 道路特征)，映射回 d_model
        self.feature_fusion = nn.Linear(configs.d_model * 2, configs.d_model)
        self.fusion_dropout = nn.Dropout(configs.dropout)

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
        
        self.head = ChannelMixingHead(in_vars=self.n_dynamic, out_vars=self.n_vars, 
                                      nf=self.head_nf, target_window=configs.pred_len, 
                                      head_dropout=configs.dropout)

    def _get_statistics(self, x):
        dim2reduce = [1]
        means = x.mean(dim2reduce, keepdim=True).detach()
        x_centered = x - means
        stdev = torch.sqrt(torch.var(x_centered, dim=dim2reduce, keepdim=True, unbiased=False) + 1e-5)
        return means, stdev

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means, stdev = self._get_statistics(x_enc)
            x_enc = (x_enc - means) / stdev

        # 1. 拆分数据
        x_dynamic = torch.index_select(x_enc, 2, self.dynamic_ids_tensor)
        x_road = torch.index_select(x_enc, 2, self.road_ids_tensor)

        # 2. 动态特征 Embedding
        # shape: [Batch * N_dyn, Patch_Num + 1, D_model]
        enc_out_dynamic, n_vars_dyn = self.en_embedding(x_dynamic.permute(0, 2, 1))
        
        # 3. 道路特征处理 (Path A)
        road_condition = self.road_mlp_embedding(x_road)
        road_glb = road_condition[:, -1:, :]
        road_condition = torch.cat([road_condition, road_glb], dim=1) # [B, P+1, D]
        
        # 扩展 Batch 维度以匹配: [B, P, D] -> [B * N_dyn, P, D]
        road_condition_expanded = road_condition.unsqueeze(1).repeat(1, n_vars_dyn, 1, 1).reshape(-1, road_condition.shape[1], road_condition.shape[2])
        
        # --- [修改核心] Concat + Linear 融合 ---
        # 原始：enc_input = enc_out_dynamic + road_condition_expanded
        # 修改为：在特征维度(dim=-1)拼接
        combined_features = torch.cat([enc_out_dynamic, road_condition_expanded], dim=-1) # [B*N, P, 2*D]
        
        # 线性映射回 D_model 并应用 Dropout
        enc_input = self.feature_fusion(combined_features) 
        enc_input = self.fusion_dropout(enc_input) # [B*N, P, D]

        # 4. 道路特征处理 (Path B) - Cross Attention Key/Value
        ex_embed = self.ex_embedding(x_road, x_mark_enc) # [B, N_road, D]
        # 对齐 Batch 维度: [B, ...] -> [B*N_dyn, ...]
        ex_embed = ex_embed.repeat_interleave(n_vars_dyn, dim=0)

        # 5. Encoder
        enc_out = self.encoder(enc_input, cross=ex_embed)

        # 6. Head & Output
        enc_out = torch.reshape(enc_out, (-1, n_vars_dyn, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * (stdev.repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means.repeat(1, self.pred_len, 1))
            
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
        return None