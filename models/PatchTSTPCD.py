import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

class FusedFlattenHead(nn.Module):
    def __init__(
        self, input_vars, output_vars, nf, target_window, head_dropout=0
    ):
        super().__init__()
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.target_window = target_window
        
        # 1. 先把 (d_model, patch_num) 展平为 nf
        self.flatten_patch = nn.Flatten(start_dim=-2)
        
        # 2. 计算输入输出的总维度
        # 输入维度: 变量数 * 每个变量的特征数 (patch_num * d_model)
        self.input_dim = input_vars * nf
        # 输出维度: 变量数 * 预测长度
        self.output_dim = output_vars * target_window
        
        # 3. 全局融合线性层
        # 这个巨大的矩阵 W 充当了 LSTM 中 Projection 的角色
        # 它让 Channel A 的特征可以直接贡献给 Channel B 的预测
        self.linear_fusion = nn.Linear(self.input_dim, self.output_dim)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x): 
        # x input: [Batch, n_vars, d_model, patch_num]
        
        # Step 1: 展平 Patch 维度 -> [Batch, n_vars, nf]
        x = self.flatten_patch(x)
        
        # Step 2: 【关键】展平变量维度，将所有通道特征拼接到一起
        # -> [Batch, n_vars * nf]
        x = x.reshape(x.shape[0], -1)
        
        # Step 3: 全局线性映射 (Channel Mixing happening here!)
        # -> [Batch, n_vars * target_window]
        x = self.linear_fusion(x)
        x = self.dropout(x)
        
        # Step 4: 还原形状以匹配损失函数要求
        # -> [Batch, n_vars, target_window]
        x = x.reshape(x.shape[0], self.output_vars, self.target_window)
        
        return x

# 原有的 Head (保持不变，供对比或默认使用)
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  
        # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x) # 这里的 Linear 是独立作用于每个 nvars 的
        x = self.dropout(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_vars = configs.enc_in
        self.output_vars = configs.c_out
        self.target_start_idx = getattr(configs, "target_start_idx", 0)
        
        # 获取配置
        patch_len = getattr(configs, "patch_len", 16)
        stride = getattr(configs, "stride", 8)
        # 【新增配置项】是否开启输出层融合，你可以手动设置为 True
        use_fused_head = getattr(configs, "use_fused_head", True) 

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, configs.dropout)

        self.encoder = Encoder(
            [
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

        # 计算 Head 的输入特征维度 nf = d_model * patch_num
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        
        # Prediction Head 选择逻辑
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if use_fused_head:
                # 使用我们新写的融合 Head
                print(f"Using FusedFlattenHead: Channel Interaction Enabled at Output Layer.")
                self.head = FusedFlattenHead(
                    input_vars=self.input_vars,
                    output_vars=self.output_vars,
                    nf=self.head_nf,
                    target_window=configs.pred_len,
                    head_dropout=configs.dropout,
                )
            else:
                # 使用原始的独立 Head
                self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                        head_dropout=configs.dropout)
                                        
        # ... (其他 task_name 的 head 保持不变，略) ...

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1. Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # 2. Patching & Embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)

        # 3. Encoder (Channel Independent)
        # PatchTST 的 Encoder 仍然保持通道独立，这有助于提取纯净的时间特征
        enc_out, attns = self.encoder(enc_out)
        
        # 4. Reshape for Head
        # [Batch * nvars, patch_num, d_model] -> [Batch, nvars, patch_num, d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        
        # [Batch, nvars, d_model, patch_num] (注意这里 permute 调整以适配 Head 输入)
        enc_out = enc_out.permute(0, 1, 3, 2)

        # 5. Decoder (Projection)
        # 这里进入 FusedFlattenHead，发生通道交互
        dec_out = self.head(enc_out) 
        
        # output is [Batch, nvars, target_window], reshape to [Batch, target_window, nvars]
        dec_out = dec_out.permute(0, 2, 1)

        # 6. De-Normalization
        target_end = self.target_start_idx + self.output_vars
        output_stdev = stdev[:, 0, self.target_start_idx:target_end]
        output_means = means[:, 0, self.target_start_idx:target_end]
        dec_out = dec_out * output_stdev.unsqueeze(1)
        dec_out = dec_out + output_means.unsqueeze(1)
        
        return dec_out

    # ... (forward 函数和其他 task 函数保持不变) ...
    
    # 确保 forward 调用了 forecast
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        # ...
        return None