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

def stable_complex_abs(z):
    x = torch.abs(z.real)
    y = torch.abs(z.imag)
    m = torch.maximum(x, y)
    r = torch.minimum(x, y) / (m + 1e-12)
    return m * torch.sqrt(1 + r * r)

class PETSALoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.person_cor = CorrCoefLoss()
    
    def forward(self, pred, ground_truth):
        # loss_feq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(ground_truth, dim=1)).to(pred.dtype).abs().mean() 
        freq_temp = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(ground_truth, dim=1))
        loss_feq = stable_complex_abs(freq_temp).mean()

        if torch.isnan(loss_feq):
            print("NaN detected in frequency loss")
            print(loss_feq)
            raise ValueError("NaN detected in frequency loss")
            
        loss_tmp = torch.nn.functional.huber_loss(pred, ground_truth, delta=0.5)
        loss =  loss_tmp + loss_feq * self.alpha
        coss = self.person_cor(pred, ground_truth)
        sf_pred = torch.nn.functional.softmax(pred - pred.mean(dim=1, keepdim=True))
        sf_gt   = torch.nn.functional.softmax((ground_truth - ground_truth.mean(dim=1, keepdim=True)))
        loss_var = torch.nn.functional.kl_div(sf_pred, sf_gt).mean()
        loss_mean = F.l1_loss(pred.mean(dim=1, keepdim=True), 
                                ground_truth.mean(dim=1, keepdim=True))
        loss +=  ((coss + loss_var + loss_mean))
        if torch.isnan(loss):
            print("NaN detected in PETSALoss")
            print(pred)
            print(loss_var)
            print(loss_mean)
            raise ValueError("NaN detected in PETSALoss")
            
        return loss
    
class OrthoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bases):
        """
        bases shape: (N_bases, Window, Window, N_var) 或 (N_bases, Window, Window)
        """
        n_bases = bases.shape[0]
        
        # 1. Flatten: 把除 N_bases 以外的所有维度展平
        # shape: (N_bases, -1)
        flat_bases = bases.view(n_bases, -1)
        
        # 2. Normalize: 为了计算 Cosine 相似度，先做 L2 归一化
        flat_bases_norm = F.normalize(flat_bases, p=2, dim=1)
        
        # 3. Gram Matrix: B * B^T
        # shape: (N_bases, N_bases)
        gram_matrix = torch.matmul(flat_bases_norm, flat_bases_norm.T)
        
        # 4. Target: 单位矩阵 Identity Matrix
        identity = torch.eye(n_bases, device=bases.device)
        
        # 5. MSE: 强迫 Gram 矩阵接近单位矩阵
        # 这意味着：对自己相似度为1，对别人相似度为0 (正交)
        return F.mse_loss(gram_matrix, identity)

class LowRankOrthoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bases_left, bases_right):
        """
        bases_left:  (N, L, R, V)
        bases_right: (N, R, L, V)
        """
        N, L, R, V = bases_left.shape
        
        # 1. 计算所有对之间的内积矩阵 (Gram Matrix)
        # 我们需要计算 G[i, j] = sum_v (Tr( (Ui,v^T Uj,v) * (Vj,v Vi,v^T) ))
        
        # 计算 U_inner: (N, N, R, R, V)
        # U_inner[i,j] = Ui^T * Uj
        U_inner = torch.einsum('ilrv, jlrv -> ijrv', bases_left, bases_left) 
        
        # 计算 V_inner: (N, N, R, R, V)
        # V_inner[i,j] = Vj * Vi^T
        V_inner = torch.einsum('irlv, jrlv -> ijrv', bases_right, bases_right)
        
        # 计算 Gram 矩阵: 两个 (R,R) 矩阵点乘并求和
        # shape: (N, N)
        gram_matrix = torch.einsum('ijrv, ijrv -> ij', U_inner, V_inner)
        
        # 2. 归一化 (让 Diagonal 变为 1)
        # 这里的 gram_matrix[i,i] 就是第 i 个基的 L2 范数的平方
        diag = torch.diag(gram_matrix).unsqueeze(0)
        norm_matrix = torch.sqrt(torch.matmul(diag.T, diag) + 1e-8)
        gram_matrix_norm = gram_matrix / norm_matrix
        
        # 3. 目标是单位矩阵
        identity = torch.eye(N, device=bases_left.device)
        return F.mse_loss(gram_matrix_norm, identity)
    
class CoBA_Loss(nn.Module):
    def __init__(self, lambda_ortho=0.1, lambda_sparse=0.01):
        super().__init__()
        self.task_loss_fn = PETSALoss(alpha=0.1)
        # self.task_loss_fn = StandardMSELoss()
        self.ortho_loss_fn = OrthoLoss()
        
        self.lambda_ortho = lambda_ortho
        self.lambda_sparse = lambda_sparse
 
    def forward(self, pred, ground_truth, bases, coeffs=None):
        """
        需要传入四个参数:
        1. pred: 模型的预测输出
        2. ground_truth: 真实标签
        3. coeffs: 模型 forward 产生的混合系数 (用于稀疏 Loss)
        4. bases: 模型的基向量参数 (用于正交 Loss)
        """
        
        # 1. 任务 Loss (MSE + ...)
        l_task = self.task_loss_fn(pred, ground_truth)
        
        # 2. 正交 Loss
        l_ortho = self.ortho_loss_fn(bases)
        
        l_total = l_task + (self.lambda_ortho * l_ortho)
        
        return l_total
    
class LowRankCoBALoss(nn.Module):
    def __init__(self, lambda_ortho=0.1, lambda_sparse=0.01):
        super().__init__()
        self.task_loss_fn = PETSALoss(alpha=0.1)
        self.ortho_loss_fn = LowRankOrthoLoss()
        # self.sparse_loss_fn = SparsityLoss()
        
        self.lambda_ortho = lambda_ortho
        self.lambda_sparse = lambda_sparse
 
    def forward(self, pred, ground_truth, bases_left, bases_right, coeffs=None):
        """
        需要传入四个参数:
        1. pred: 模型的预测输出
        2. ground_truth: 真实标签
        3. coeffs: 模型 forward 产生的混合系数 (用于稀疏 Loss)
        4. bases: 模型的基向量参数 (用于正交 Loss)
        """
        
        # 1. 任务 Loss (MSE + ...)
        l_task = self.task_loss_fn(pred, ground_truth)
        if torch.isnan(l_task):
            print("NaN detected in task loss")
            raise ValueError("NaN detected in task loss")
        # 2. 正交 Loss
        l_ortho = self.ortho_loss_fn(bases_left, bases_right)
        if torch.isnan(l_ortho):
            print("NaN detected in ortho loss")
            raise ValueError("NaN detected in ortho loss")
        
        l_total = l_task + (self.lambda_ortho * l_ortho)
        
        return l_total