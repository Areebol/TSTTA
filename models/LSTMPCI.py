import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        self.hidden_size = getattr(configs, 'd_model', 128)
        self.num_layers = getattr(configs, 'e_layers', 1)
        self.dropout = getattr(configs, 'dropout', 0.1)

        # 1. 保持 LSTM 融合：input_size = enc_in
        # 这里不同通道的信息（速度、能耗、道路）会在隐藏层深度混合
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        # 2. 【核心修改】：通道独立的投影层
        # 为每个通道创建一个独立的线性映射，Hidden_Size -> Pred_Len
        self.independent_projections = nn.ModuleList([
            nn.Linear(self.hidden_size, self.pred_len) for _ in range(self.enc_in)
        ])

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, Seq_Len, Channels]
        
        # 1. LSTM 提取融合特征
        _, (h_n, _) = self.lstm(x_enc)
        last_hidden = h_n[-1, :, :]  # [Batch, Hidden_Size]
        
        # 2. 独立投影
        # 每个通道预测自己的未来，虽然输入的是融合后的隐藏状态
        outputs = []
        for i in range(self.enc_in):
            # 第 i 个投影层只负责第 i 个变量的预测
            out_i = self.independent_projections[i](last_hidden) # [Batch, Pred_Len]
            outputs.append(out_i)
        
        # 3. 拼接结果
        # [Batch, Pred_Len, Channels]
        output = torch.stack(outputs, dim=-1)
        
        return output