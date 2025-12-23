import torch
import torch.nn.functional as F

class PETSALoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha # 频域损失权重

    def forward(self, pred, target):
        # 1. Huber Loss (鲁棒性)
        l_huber = F.huber_loss(pred, target, delta=0.5)
        
        # 2. Frequency Loss (周期性)
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        l_freq = (pred_fft - target_fft).abs().mean()
        
        # 3. Structural Loss (均值与方差对齐)
        l_mean = F.l1_loss(pred.mean(dim=1), target.mean(dim=1))
        l_var = F.l1_loss(pred.var(dim=1), target.var(dim=1))
        
        return l_huber + self.alpha * l_freq + (l_mean + l_var)