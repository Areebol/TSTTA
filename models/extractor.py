import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    """
    可逆实例归一化 (Reversible Instance Normalization)
    用于消除时序数据的非平稳性 (Non-stationarity)，让聚类只关注"形状"而非"数值大小"
    """
    def __init__(self, num_features: int, eps=1e-5, affine=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class MAE_Contrastive(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.seq_len = cfg.DATA.SEQ_LEN
        self.input_dim = cfg.DATA.N_VAR
        self.patch_len = 8  # 建议 Patch 长度，可调整
        self.stride = 8
        self.d_model = 128
        
        # 计算 Patch 数量
        self.num_patches = self.seq_len // self.stride
        
        # 0. 归一化层
        self.revin = RevIN(self.input_dim)

        # 1. Patch Embedding
        self.patch_embed = nn.Linear(self.patch_len * self.input_dim, self.d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, self.d_model))
        
        # 2. Encoder (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=512, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # 3. Projector (仅用于对比学习 Loss，聚类时不使用)
        # 将特征映射到超球面上，专门用于计算 InfoNCE
        self.projector = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 64) 
        )

        # 4. Decoder (用于重建 Loss)
        self.decoder_head = nn.Linear(self.d_model, self.patch_len * self.input_dim)

    def forward_encoder(self, x, mask_ratio=0.0):
        # x: [Batch, Seq_Len, Vars]
        B = x.shape[0]
        
        # --- Step A: Instance Normalization ---
        # 关键！先去掉幅值信息，只保留波形形状
        x = self.revin(x, 'norm') 
        
        # --- Step B: Patching ---
        # reshape: [B, Num_Patches, Patch_Len * Vars]
        patches = x.view(B, self.num_patches, -1)
        
        # Embedding + Positional Encoding
        x_emb = self.patch_embed(patches) + self.pos_embed
        
        # --- Step C: Masking ---
        if mask_ratio > 0:
            num_masked = int(mask_ratio * self.num_patches)
            noise = torch.rand(B, self.num_patches, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
            ids_keep = ids_shuffle[:, :self.num_patches - num_masked]
            x_masked = torch.gather(x_emb, 1, ids_keep.unsqueeze(-1).expand(-1, -1, self.d_model))
            
            # Encoder 只处理可见部分
            h_masked = self.encoder(x_masked)
            return h_masked, patches, ids_restore, ids_keep
        else:
            # 推理/聚类模式：不 Mask
            h = self.encoder(x_emb)
            return h, patches, None, None

    def forward(self, x):
        """训练时调用：返回对比特征和重建结果"""
        # 1. 对比学习分支：生成两个视角的掩码 (Mask Ratio = 0.5 ~ 0.75)
        h1, _, _, _ = self.forward_encoder(x, mask_ratio=0.75)
        h2, _, _, _ = self.forward_encoder(x, mask_ratio=0.75)
        
        # 对 Patch 特征取平均，得到序列级特征
        z1 = self.projector(h1.mean(dim=1))
        z2 = self.projector(h2.mean(dim=1))
        
        # 2. 重建分支 (为了简单，这里复用 h1 进行重建)
        # 实际 MAE 需补全 Mask token，这里简化为只预测可见部分或通过 Decoder 预测
        pred_patches = self.decoder_head(h1) 
        
        return z1, z2, pred_patches, h1 # 返回投影特征和重建结果

    def get_embedding(self, x):
        """聚类/推理时调用"""
        self.eval()
        with torch.no_grad():
            # 不 Mask，提取全量特征
            h, _, _, _ = self.forward_encoder(x, mask_ratio=0.0)
            # Mean Pooling
            embedding = h.mean(dim=1)
            # L2 Normalize (非常重要，配合 K-Means)
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding