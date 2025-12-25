import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, ground_truth):
        return F.mse_loss(pred, ground_truth)

class CorrCoefLoss(nn.Module):

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = preds.reshape(-1)
        y = target.reshape(-1)

        var_x = torch.var(x)
        var_y = torch.var(y)
        if (var_x < self.eps) or (var_y < self.eps):
            return torch.zeros((), dtype=preds.dtype, device=preds.device)
        
        data = torch.stack([x, y], dim=0)
        corrmat = torch.corrcoef(data)
        corr_xy = corrmat[0, 1]

        corr_xy = torch.nan_to_num(corr_xy, nan=0.0, posinf=0.0, neginf=0.0)
        corr_xy = torch.clamp(corr_xy, -1.0, 1.0)
        
        return -corr_xy

class PETSALoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.person_cor = CorrCoefLoss()
        
    def forward(self, pred, ground_truth):
        loss_feq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(ground_truth, dim=1)).abs().mean() 
        loss_tmp = torch.nn.functional.huber_loss(pred, ground_truth, delta=0.5)
        loss =  loss_tmp + loss_feq * self.alpha
        coss = self.person_cor(pred, ground_truth)
        sf_pred = torch.nn.functional.softmax(pred - pred.mean(dim=1, keepdim=True))
        sf_gt   = torch.nn.functional.softmax((ground_truth - ground_truth.mean(dim=1, keepdim=True)))
        loss_var = torch.nn.functional.kl_div(sf_pred, sf_gt).mean()
        loss_mean = F.l1_loss(pred.mean(dim=1, keepdim=True), 
                                ground_truth.mean(dim=1, keepdim=True))
        loss +=  ((coss + loss_var + loss_mean))
        return loss