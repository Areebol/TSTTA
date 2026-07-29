import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

# ==================== 统一 PCD 跨通道融合头 ====================
class FusedLinearProjection(nn.Module):
    #  将 pred_len 抽象为 target_len，使其能够兼容预测(pred_len)和插补(seq_len)
    def __init__(
        self, d_model, input_vars, output_vars, target_len, head_dropout=0
    ):
        super().__init__()
        self.d_model = d_model
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.target_len = target_len
        
        # iTransformer 专属维度适配
        self.input_dim = input_vars * d_model
        self.output_dim = output_vars * target_len
        
        self.linear_fusion = nn.Linear(self.input_dim, self.output_dim)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x:[B, n_vars, d_model]
        x = x.reshape(x.shape[0], -1)
        x = self.linear_fusion(x)
        x = self.dropout(x)
        # 根据 target_len 还原形状
        x = x.reshape(x.shape[0], self.output_vars, self.target_len)
        return x

# ==================== iTransformerPCD 主模型 ====================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.n_vars = configs.enc_in
        self.output_vars = (
            configs.c_out
            if self.task_name in ["long_term_forecast", "short_term_forecast"]
            else self.n_vars
        )

        # Embedding（原版不变）
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout, configs.ignore_stamp)
        # Encoder（原版不变）
        self.encoder = Encoder([
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # ==================== PCD 改造：统一所有任务的 Head ====================
        self.use_fused_head = getattr(configs, "use_fused_head", True)
        
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # 预测任务的目标长度是 pred_len
            target_len = configs.pred_len
        elif self.task_name in['imputation', 'anomaly_detection']:
            # 插补和异常检测的目标长度是 seq_len
            target_len = configs.seq_len
        else:
            target_len = None

        #  让 forecast, imputation, anomaly_detection 都能使用 PCD 融合头
        if self.task_name in ['long_term_forecast', 'short_term_forecast', 'imputation', 'anomaly_detection']:
            if self.use_fused_head:
                print(f"Using FusedLinearProjection: Channel Interaction Enabled for {self.task_name}.")
                self.projection = FusedLinearProjection(
                    d_model=configs.d_model,
                    input_vars=configs.enc_in,
                    output_vars=self.output_vars,
                    target_len=target_len,
                    head_dropout=configs.dropout
                )
            else:
                self.projection = nn.Linear(configs.d_model, target_len, bias=True)

        # 分类任务（保持原版不变）
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    # ==================== 下方所有业务函数保持原样 ====================
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # DataEmbedding_inverted appends timestamp covariates as extra tokens.
        # The fused PCD head is defined over the N input-variable tokens only.
        variable_tokens = enc_out[:, :N, :]
        dec_out = self.projection(variable_tokens).permute(0, 2, 1)
        if self.output_attention: return dec_out, attns, enc_out
        else: return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        variable_tokens = enc_out[:, :N, :]
        dec_out = self.projection(variable_tokens).permute(0, 2, 1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        variable_tokens = enc_out[:, :N, :]
        dec_out = self.projection(variable_tokens).permute(0, 2, 1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in['long_term_forecast', 'short_term_forecast']:
            if self.output_attention:
                dec_out, attns, enc_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None
