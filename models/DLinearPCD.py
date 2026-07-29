import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设你已经有了这个层，如果没有可以直接用常规的 moving average 实现
from layers.Autoformer_EncDec import series_decomp 

class FusedFlattenHead(nn.Module):
    """
    适配 DLinear 的全局通道融合输出头
    """
    def __init__(
        self, input_vars, output_vars, seq_len, pred_len, head_dropout=0.0
    ):
        super().__init__()
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 输入维度: 变量数 * 历史序列长度
        self.input_dim = input_vars * seq_len
        # 输出维度: 变量数 * 预测长度
        self.output_dim = output_vars * pred_len
        
        # 全局融合线性层：负责跨通道与跨时间的双重信息提取
        self.linear_fusion = nn.Linear(self.input_dim, self.output_dim)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x): 
        # x input shape: [Batch, n_vars, seq_len]
        
        # Step 1: 展平变量维度与时间维度 -> [Batch, n_vars * seq_len]
        x = x.reshape(x.shape[0], -1)
        
        # Step 2: 全局线性映射 (Channel & Time Mixing)
        x = self.linear_fusion(x)
        x = self.dropout(x)
        
        # Step 3: 还原形状以匹配预期输出 -> [Batch, n_vars, pred_len]
        x = x.reshape(x.shape[0], self.output_vars, self.pred_len)
        
        return x


class Model(nn.Module):
    """
    DLinearPCD: Channel Dependent DLinear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in 
        self.output_channels = configs.c_out
        self.dropout = getattr(configs, 'dropout', 0.1)
        
        # 1. 序列分解模块
        self.decompsition = series_decomp(configs.moving_avg)
        
        # 2. 替换原本的独立/共享 Linear 层，使用全新的 FusedFlattenHead
        # 季节性项的全局融合头
        self.Head_Seasonal = FusedFlattenHead(
            input_vars=self.channels,
            output_vars=self.output_channels,
            seq_len=self.seq_len, 
            pred_len=self.pred_len, 
            head_dropout=self.dropout
        )
        # 趋势项的全局融合头
        self.Head_Trend = FusedFlattenHead(
            input_vars=self.channels,
            output_vars=self.output_channels,
            seq_len=self.seq_len, 
            pred_len=self.pred_len, 
            head_dropout=self.dropout
        )

    def encoder(self, x):
        # 输入 x: [Batch, seq_len, channels]
        
        # 1. 序列分解
        seasonal_init, trend_init = self.decompsition(x)
        
        # 2. 维度转换以适配 Head [Batch, channels, seq_len]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        # 3. 通过 FusedFlattenHead 进行全局通道交互式的预测
        # 输出形状将变为: [Batch, channels, pred_len]
        seasonal_output = self.Head_Seasonal(seasonal_init)
        trend_output = self.Head_Trend(trend_init)
        
        # 4. 季节性与趋势预测结果相加
        x_out = seasonal_output + trend_output
        
        # 5. 还原回时序模型标准输出格式 [Batch, pred_len, channels]
        return x_out.permute(0, 2, 1)

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        # (可根据需要补充 imputation, anomaly_detection 等任务的路由)
        return None