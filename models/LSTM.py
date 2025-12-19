import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # 获取配置参数
        self.pred_len = configs.pred_len   # 预测长度
        self.enc_in = configs.enc_in       # 输入特征数 (channels)
        
        # LSTM 超参数 (如果在 configs 里没定义，就给个默认值)
        self.hidden_size = getattr(configs, 'd_model', 128) 
        self.num_layers = getattr(configs, 'e_layers', 1)
        self.dropout = getattr(configs, 'dropout', 0.1)

        # 核心 LSTM 层
        # input_size=enc_in: 同时也处理所有变量 (Multivariate)
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        # 输出层 (投影层)
        # 将 LSTM 的最后一个隐状态映射到: 预测长度 * 特征数
        self.projection = nn.Linear(self.hidden_size, self.pred_len * self.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        x_enc shape: [Batch, Seq_Len, Channels]
        """
        
        # 1. LSTM 前向传播
        # output: 每个时间步的输出
        # (h_n, c_n): 最后一个时间步的隐状态和细胞状态
        _, (h_n, _) = self.lstm(x_enc)
        
        # 2. 获取最后一层的隐状态
        # h_n shape: [Num_Layers, Batch, Hidden_Size] -> 取最后一层 -> [Batch, Hidden_Size]
        last_hidden = h_n[-1, :, :]
        
        # 3. 线性投影
        # [Batch, Hidden_Size] -> [Batch, Pred_Len * Channels]
        output = self.projection(last_hidden)
        
        # 4. 重塑形状以匹配输出要求
        # [Batch, Pred_Len * Channels] -> [Batch, Pred_Len, Channels]
        output = output.view(-1, self.pred_len, self.enc_in)
        
        return output
    
    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc):
        enc_out = self.encoder(x_enc)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)
        return output