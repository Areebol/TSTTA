import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # 这里的 enc_in 是通道数
        
        self.hidden_size = getattr(configs, 'd_model', 128)
        self.num_layers = getattr(configs, 'e_layers', 1)
        self.dropout = getattr(configs, 'dropout', 0.1)

        # 【核心修改 1】：input_size 变为 1
        # 因为每个通道独立进入 LSTM，不再同时看所有变量
        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        # 【核心修改 2】：输出层只预测单个通道的未来序列
        # 不再是 Pred_Len * Channels，而是映射到 Pred_Len
        self.projection = nn.Linear(self.hidden_size, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        x_enc shape: [B, L, C]
        """
        B, L, C = x_enc.shape

        # 1. 维度转换：将通道维度合并到 Batch 维度
        # [B, L, C] -> [B, C, L] -> [B * C, L, 1]
        x = x_enc.transpose(1, 2).reshape(B * C, L, 1)

        # 2. 独立进入 LSTM
        # 每个通道共享同一套 LSTM 参数，但处理过程完全独立
        _, (h_n, _) = self.lstm(x)
        
        # 3. 获取最后一层隐状态
        # h_n[-1] shape: [B * C, Hidden_Size]
        last_hidden = h_n[-1, :, :]
        
        # 4. 线性投影预测未来
        # [B * C, Hidden_Size] -> [B * C, Pred_Len]
        output = self.projection(last_hidden)
        
        # 5. 还原形状
        # [B * C, Pred_Len] -> [B, C, Pred_Len] -> [B, Pred_Len, C]
        output = output.view(B, C, self.pred_len).transpose(1, 2)
        
        return output