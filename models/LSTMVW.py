import torch
import torch.nn as nn

class Model(nn.Module):
    """
    基于 LSTM 的序列预测模型
    结构：双向 LSTM 特征提取器 + 多层 MLP 回归器
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # 1. 参数提取
        self.pred_len = configs.pred_len   # 预测未来多长
        self.enc_in = configs.enc_in       # 输入特征维度
        self.hid_size = getattr(configs, 'd_model', 128) 
        self.n_layers = 2
        self.dropout = getattr(configs, 'dropout', 0.1)

        # 2. 核心双向 LSTM 层
        # 设置 bidirectional=True，模型会同时学习正向和反向的时间依赖
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hid_size,
            num_layers=self.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.n_layers > 1 else 0
        )

        # 3. 回归器 (MLP)
        # 输入维度是 2 * hid_size (因为双向拼接了两个方向的隐状态)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.hid_size, self.hid_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hid_size, self.pred_len * self.enc_in)
        )

        # 打印简化的模型描述
        print(self._get_simple_description())

    def _get_simple_description(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"\n{'='*20} Bi-LSTM Model Initialized {'='*20}\n" \
               f"Input Dim: {self.enc_in}, Hidden Dim: {self.hid_size}\n" \
               f"Layers: {self.n_layers}, Bidirectional: True\n" \
               f"Total Trainable Params: {total_params:,}\n" \
               f"{'='*60}\n"

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        x_enc shape: [Batch, Seq_Len, Channels]
        """
        # 1. LSTM 处理序列
        # lstm_out 维度: [Batch, Seq_Len, 2 * Hidden_Size]
        lstm_out, _ = self.lstm(x_enc)
        
        # 2. 特征聚合
        # 取最后一个时间步作为整个序列的特征表示
        # last_step shape: [Batch, 2 * Hidden_Size]
        last_step = lstm_out[:, -1, :]
        
        # 3. 通过 MLP 进行预测
        # output shape: [Batch, Pred_Len * Channels]
        output = self.mlp(last_step)
        
        # 4. 重塑形状以匹配预测输出 [Batch, Pred_Len, Channels]
        return output.view(-1, self.pred_len, self.enc_in)
