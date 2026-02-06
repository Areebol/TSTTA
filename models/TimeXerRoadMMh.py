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

# --- [保持] 门控融合模块 ---
class GatedRoadInjection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # 1. 门控网络：输入 [动态特征, 道路特征]，输出 0~1 的门控值
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # 2. 道路特征变换层：将道路特征对齐到动态特征空间
        self.road_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dynamic, x_road):
        """
        x_dynamic: [Batch, Patch_Num, D_model]
        x_road:    [Batch, Patch_Num, D_model]
        """
        # A. 计算门控系数 (Gate)
        combined = torch.cat([x_dynamic, x_road], dim=-1)
        gate = self.gate_net(combined)
        
        # B. 变换道路特征
        road_feat = self.road_proj(x_road)
        road_feat = self.dropout(road_feat)
        
        # C. 加权注入 (Residual Connection)
        out = x_dynamic + gate * road_feat
        
        return out

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
        # 注意：这里假设 x 包含了所有变量，glb_token 维度需要对应
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        # 如果输入的 vars 数量与 glb_token 不一致（例如推理时切片），需要做相应处理
        # 简单起见这里假设维度匹配
        
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
        assert configs.enc_in >= 20, "Encoder input dimensions must be >= 20"
        exclude = {24, 25, 26, 27, 28}  # for oVED datasets
        default_road_ids = [i for i in range(configs.enc_in) if i not in exclude]
        
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

        # 1. 统一 Embedding：现在对所有 variables (enc_in) 进行 Embed
        # 输入维度是 n_vars (enc_in)，不再是 n_dynamic
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        
        # 2. 道路变量 Context 提取 (用于 Gate)
        self.road_mlp_embedding = RoadConditioning(self.n_road, configs.d_model, self.patch_num)
        
        # 3. Cross Attention Embedding (Keys/Values)
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # --- 使用 Gate Fusion ---
        self.road_gate_fusion = GatedRoadInjection(configs.d_model, dropout=configs.dropout)

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
        
        # --- [修改] Head 输入维度改为 n_vars ---
        # 因为 Encoder 现在输出所有变量的 Embedding
        self.head = ChannelMixingHead(in_vars=self.n_vars, out_vars=self.n_vars, 
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

        # --- 1. Embedding 所有变量 ---
        # x_enc: [Batch, Seq_Len, N_vars]
        # enc_out_all: [Batch * N_vars, Patch_Num + 1, D_model]
        enc_out_all, _ = self.en_embedding(x_enc.permute(0, 2, 1))
        
        # 为了操作方便，先 reshape 回 [Batch, N_vars, P+1, D]
        B = x_enc.shape[0]
        enc_out_all = enc_out_all.reshape(B, self.n_vars, -1, enc_out_all.shape[-1])

        # --- 2. 拆分 Dynamic 和 Road Embeddings ---
        # 这里 dim=1 是变量维度，是正确的
        enc_out_dynamic = torch.index_select(enc_out_all, 1, self.dynamic_ids_tensor)
        enc_out_road = torch.index_select(enc_out_all, 1, self.road_ids_tensor)

        # 展平 Dynamic 以适配 Gate 模块: [Batch * N_dyn, P+1, D]
        enc_out_dynamic_flat = enc_out_dynamic.reshape(-1, enc_out_dynamic.shape[2], enc_out_dynamic.shape[3])

        # --- 3. 计算全局道路 Context (Road Conditioning) ---
        # [修复点] x_enc 是 [Batch, Seq_Len, N_vars]，选变量应该用 dim=2
        # 原错误代码: x_road = torch.index_select(x_enc, 1, self.road_ids_tensor)
        x_road = torch.index_select(x_enc, 2, self.road_ids_tensor) 
        
        road_condition = self.road_mlp_embedding(x_road)
        road_glb = road_condition[:, -1:, :]
        road_condition = torch.cat([road_condition, road_glb], dim=1) # [B, P+1, D]
        
        # 扩展 Context 以匹配 Dynamic 变量数量: [B, P+1, D] -> [B * N_dyn, P+1, D]
        road_condition_expanded = road_condition.unsqueeze(1).repeat(1, self.n_dynamic, 1, 1)
        road_condition_expanded = road_condition_expanded.reshape(-1, road_condition.shape[1], road_condition.shape[2])

        # --- 4. 融合 (Gate Injection) ---
        # enc_input (Dynamic Fused): [Batch * N_dyn, P+1, D]
        enc_input_dynamic_fused = self.road_gate_fusion(enc_out_dynamic_flat, road_condition_expanded)
        
        # 还原形状 [Batch, N_dyn, P+1, D]
        enc_input_dynamic_fused = enc_input_dynamic_fused.reshape(B, self.n_dynamic, -1, enc_input_dynamic_fused.shape[-1])

        # --- 5. 拼接输入到 Encoder ---
        # 拼接: [Batch, N_dyn, ...] + [Batch, N_road, ...] -> [Batch, N_vars, ...]
        enc_input = torch.cat([enc_input_dynamic_fused, enc_out_road], dim=1)
        
        # 再次展平给 Encoder: [Batch * N_vars, P+1, D]
        enc_input_flat = enc_input.reshape(-1, enc_input.shape[2], enc_input.shape[3])

        # --- 6. Cross Attention Embedding ---
        ex_embed = self.ex_embedding(x_road, x_mark_enc) # [B, N_road, D]
        # 对齐 Batch 维度: [B, ...] -> [B * N_vars, ...]
        ex_embed = ex_embed.repeat_interleave(self.n_vars, dim=0)

        # --- 7. Encoder ---
        enc_out = self.encoder(enc_input_flat, cross=ex_embed)

        # --- 8. Head & Output ---
        enc_out = torch.reshape(enc_out, (B, self.n_vars, enc_out.shape[-2], enc_out.shape[-1]))
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