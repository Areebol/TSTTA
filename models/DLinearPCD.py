import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp

# ======================== PCD 核心新增：独立分离的跨通道融合线性头 ========================
class FusedFlattenHead(nn.Module):
    """
    适配 DLinear 的全局通道融合输出头
    """
    def __init__(self, n_vars, seq_len, pred_len, head_dropout=0.0):
        super().__init__()
        self.n_vars = n_vars
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 输入维度: 变量数 * 历史序列长度
        self.input_dim = n_vars * seq_len
        # 输出维度: 变量数 * 预测长度
        self.output_dim = n_vars * pred_len
        
        # 全局融合线性层：负责跨通道与跨时间的双重信息提取
        self.linear_fusion = nn.Linear(self.input_dim, self.output_dim)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x): 
        # x input shape: [Batch, n_vars, seq_len]
        B = x.shape[0]
        
        # Step 1: 展平变量维度与时间维度 ->[Batch, n_vars * seq_len]
        x = x.reshape(B, -1)
        
        # Step 2: 全局线性映射 (Channel & Time Mixing)
        x = self.linear_fusion(x)
        x = self.dropout(x)
        
        # Step 3: 还原形状以匹配预期输出 -> [Batch, n_vars, pred_len]
        x = x.reshape(B, self.n_vars, self.pred_len)
        
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    DLinearPCD: Channel Dependent DLinear (Preserving Additive Decomposition Prior)
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = individual
        self.dropout = getattr(configs, 'dropout', 0.1)

        # 动态适配任务的预测长度（插补/异常检测输出长度与输入长度一致）
        # if self.task_name in['classification', 'anomaly_detection', 'imputation']:
        #     self.pred_len = configs.seq_len
        # else:
        #     self.pred_len = configs.pred_len

        # 1. 原版时序分解模块
        self.decomposition = series_decomp(configs.moving_avg)
        
        # ======================== PCD 开关与映射层初始化 ========================
        self.use_fused_head = getattr(configs, "use_fused_head", True)

        if self.use_fused_head:
            print("DLinearPCD Using FusedFlattenHead: Channel Interaction Enabled for Seasonal & Trend independently.")
            # PCD 模式：季节性项与趋势项分别拥有独立的全局融合头
            self.Head_Seasonal = FusedFlattenHead(
                n_vars=self.channels, 
                seq_len=self.seq_len, 
                pred_len=self.pred_len, 
                head_dropout=self.dropout
            )
            self.Head_Trend = FusedFlattenHead(
                n_vars=self.channels, 
                seq_len=self.seq_len, 
                pred_len=self.pred_len, 
                head_dropout=self.dropout
            )
        else:
            # 原版 DLinear 模式：各自通道独立
            if self.individual:
                self.Linear_Seasonal = nn.ModuleList()
                self.Linear_Trend = nn.ModuleList()
                for i in range(self.channels):
                    self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                    self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                    self.Linear_Seasonal[i].weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                    self.Linear_Trend[i].weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            else:
                self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
                self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
                self.Linear_Seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # 分类任务配置（原版完全保留）
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout_layer = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        # 1. 原版 DLinear 时序分解
        # 输入 x: [Batch, seq_len, channels]
        seasonal_init, trend_init = self.decomposition(x)
        
        # 转换维度适配映射层:[Batch, channels, seq_len]
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        # 2. 时序预测映射（核心切换分支）
        if self.use_fused_head:
            # PCD 模式：通过独立分离的全局头进行预测
            seasonal_output = self.Head_Seasonal(seasonal_init)
            trend_output = self.Head_Trend(trend_init)
        else:
            # 原版 DLinear 模式
            if self.individual:
                seasonal_output = torch.zeros([seasonal_init.size(0), self.channels, self.pred_len], dtype=seasonal_init.dtype).to(seasonal_init.device)
                trend_output = torch.zeros([trend_init.size(0), self.channels, self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
                for i in range(self.channels):
                    seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                    trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
            else:
                seasonal_output = self.Linear_Seasonal(seasonal_init)
                trend_output = self.Linear_Trend(trend_init)
        
        # 3. 季节与趋势预测结果相加 (100% 遵守 DLinear 的加法分解先验)
        x_out = seasonal_output + trend_output
        
        # 4. 还原回时序模型标准输出格式[Batch, pred_len, channels]
        return x_out.permute(0, 2, 1)

    # ======================== 所有任务处理函数（完全和原版一致） ========================
    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc):
        enc_out = self.encoder(x_enc)
        # (batch_size, seq_length * channels)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  #[B, N]
        return None