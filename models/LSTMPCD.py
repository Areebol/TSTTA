import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # 获取配置参数
        self.pred_len = configs.pred_len   # 预测长度
        self.enc_in = configs.enc_in       # 输入特征数 (channels)
        
        # LSTM 超参数
        self.hidden_size = getattr(configs, 'd_model', 128) 
        self.num_layers = getattr(configs, 'e_layers', 1)
        self.dropout = getattr(configs, 'dropout', 0.1)

        # 1. 核心 LSTM 层 (保持通道独立处理)
        # input_size=1: 每个通道的数据依然作为独立序列输入
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        # 2. 输出层 (通道信息融合)
        # 输入维度: 所有通道的隐状态拼接在一起 (Channels * Hidden_Size)
        # 输出维度: 预测所有通道的未来步长 (Pred_Len * Channels)
        self.projection = nn.Linear(
            self.enc_in * self.hidden_size, 
            self.pred_len * self.enc_in
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        x_enc shape: [Batch, Seq_Len, Channels]
        """
        B, S, C = x_enc.shape
        
        # 1. LSTM 前置处理 (Channel Independence)
        # [Batch, Seq_Len, Channels] -> [Batch, Channels, Seq_Len]
        x_enc = x_enc.permute(0, 2, 1)
        
        # [Batch, Channels, Seq_Len] -> [Batch * Channels, Seq_Len, 1]
        x_enc = x_enc.reshape(B * C, S, 1)
        
        # 2. LSTM 前向传播 (各通道独立提取特征)
        _, (h_n, _) = self.lstm(x_enc)
        
        # 3. 获取最后一层的隐状态
        # h_n shape: [Num_Layers, Batch * Channels, Hidden_Size] -> [Batch * Channels, Hidden_Size]
        last_hidden = h_n[-1, :, :]
        
        # 4. 准备通道融合 (Channel Mixing)
        # 先将 Batch 和 Channels 拆开: [Batch * Channels, Hidden_Size] -> [Batch, Channels, Hidden_Size]
        last_hidden = last_hidden.view(B, C, self.hidden_size)
        
        # 将 Channels 和 Hidden_Size 展平，以便全连接层可以同时看到所有通道的信息
        # [Batch, Channels, Hidden_Size] -> [Batch, Channels * Hidden_Size]
        last_hidden_flat = last_hidden.view(B, C * self.hidden_size)
        
        # 5. 线性投影 (进行通道融合并输出预测)
        # [Batch, Channels * Hidden_Size] -> [Batch, Pred_Len * Channels]
        output = self.projection(last_hidden_flat)
        
        # 6. 还原为最终的时序输出形状
        # [Batch, Pred_Len * Channels] -> [Batch, Pred_Len, Channels]
        output = output.view(B, self.pred_len, C)
        
        return output
    
    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def imputation(self, x_enc):
        pass

    def anomaly_detection(self, x_enc):
        pass

    def classification(self, x_enc):
        pass