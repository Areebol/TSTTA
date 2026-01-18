import torch
import torch.nn as nn
import torch.nn.functional as F
from device_manager import global_device

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
        freq_temp = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(ground_truth, dim=1))
        loss_feq = stable_complex_abs(freq_temp).mean()

        if torch.isnan(loss_feq):
            print("NaN detected in frequency loss")
            print(loss_feq)
            raise ValueError("NaN detected in frequency loss")
        if global_device == torch.device('npu'):
            loss_tmp = huber_loss(pred, ground_truth, delta=0.5)
        else:
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

# class LowRankOrthoLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, bases_left, bases_right):
#         """
#         bases_left:  (N, L, R, V)
#         bases_right: (N, R, L, V)
#         """
#         N, L, R, V = bases_left.shape
        
#         # 1. 计算所有对之间的内积矩阵 (Gram Matrix)
#         # 我们需要计算 G[i, j] = sum_v (Tr( (Ui,v^T Uj,v) * (Vj,v Vi,v^T) ))
        
#         # 计算 U_inner: (N, N, R, R, V)
#         # U_inner[i,j] = Ui^T * Uj
#         U_inner = torch.einsum('ilrv, jlrv -> ijrv', bases_left, bases_left) 
        
#         # 计算 V_inner: (N, N, R, R, V)
#         # V_inner[i,j] = Vj * Vi^T
#         V_inner = torch.einsum('irlv, jrlv -> ijrv', bases_right, bases_right)
        
#         # 计算 Gram 矩阵: 两个 (R,R) 矩阵点乘并求和
#         # shape: (N, N)
#         gram_matrix = torch.einsum('ijrv, ijrv -> ij', U_inner, V_inner)
        
#         # 2. 归一化 (让 Diagonal 变为 1)
#         # 这里的 gram_matrix[i,i] 就是第 i 个基的 L2 范数的平方
#         diag = torch.diag(gram_matrix).unsqueeze(0)
#         norm_matrix = torch.sqrt(torch.matmul(diag.T, diag) + 1e-8)
#         gram_matrix_norm = gram_matrix / norm_matrix
        
#         # 3. 目标是单位矩阵
#         identity = torch.eye(N, device=bases_left.device)
#         return F.mse_loss(gram_matrix_norm, identity)
    
class LowRankOrthoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bases_left, bases_right):
        """
        计算低秩矩阵重构的正交性 Loss。
        针对 (N, N) 的 Gram 矩阵计算 MSE Loss，使其逼近单位矩阵。
        V 维度被视为独立的视图，分别计算归一化和 Loss，最后求平均。

        参数:
        bases_left:  (N, L, R, V) -> U 矩阵
        bases_right: (N, R, L, V) -> W 矩阵
        
        N: 矩阵数量 (Batch Size)
        L: 特征维度
        R: 秩 (Rank)
        V: 视图数量 (Views)
        """
        N, L, R, V = bases_left.shape
        N_, R_, L_, V_ = bases_right.shape
        assert N == N_ and R == R_ and L == L_ and V == V_, "bases_left mismatch bases_right shape"
        
        # ----------------------------------------------------------------
        # 1. 计算 Gram 矩阵 (保留 V 维度)
        # 目标: G[i, j, v] = < (Ui*Wi)_v, (Uj*Wj)_v >_Frobenius
        # ----------------------------------------------------------------
        
        # 计算 U 部分的交叉内积
        # U_cross[i, j, r, k, v] = (Ui_v[:, r])^T * (Uj_v[:, k])
        # Einsum: ilrv, jlkv -> ijrkv (对 L 维度求和)
        U_cross = torch.einsum('ilrv, jlkv -> ijrkv', bases_left, bases_left)
        
        # 计算 W 部分的交叉内积
        # W_cross[i, j, r, k, v] = (Wi_v[r, :]) * (Wj_v[k, :])^T
        # Einsum: irlv, jklv -> ijrkv (对 L 维度求和)
        V_cross = torch.einsum('irlv, jklv -> ijrkv', bases_right, bases_right)
        
        # 组合得到 Gram 矩阵
        # 根据内积性质，我们需要对所有 Rank 组合 (r, k) 进行求和
        # output shape: (N, N, V)
        gram_matrix = torch.einsum('ijrkv, ijrkv -> ijv', U_cross, V_cross)
        
        # ----------------------------------------------------------------
        # 2. 归一化 (针对每个 View 单独进行)
        # ----------------------------------------------------------------
        
        # 提取对角线元素 (即每个矩阵自身的 L2 范数平方)
        # shape: (N, V)
        diag_elements = torch.einsum('iiv -> iv', gram_matrix)
        
        # 计算模长 sqrt(gram[i,i])
        norms = torch.sqrt(diag_elements + 1e-8)
        
        # 构建分母矩阵 denominator[i, j, v] = norms[i, v] * norms[j, v]
        # 利用广播: (N, 1, V) * (1, N, V) -> (N, N, V)
        denominator = norms.unsqueeze(1) * norms.unsqueeze(0)
        
        # 归一化得到余弦相似度矩阵 / Correlation Matrix
        gram_matrix_norm = gram_matrix / (denominator + 1e-8)
        
        # ----------------------------------------------------------------
        # 3. 计算 Loss
        # ----------------------------------------------------------------
        
        # 目标是单位矩阵 (N, N)
        # 我们将其扩展为 (N, N, 1) 以便与 (N, N, V) 进行广播
        identity = torch.eye(N, device=bases_left.device).unsqueeze(-1)
        
        # 计算 MSE Loss
        # F.mse_loss 默认是 'mean' reduction，会把 (N, N, V) 所有元素求平均
        # 这符合 "V 里面的每个单独算一个 loss，然后平均起来" 的需求
        return F.mse_loss(gram_matrix_norm, identity)
    
class CoBA_Loss(nn.Module):
    def __init__(self, lambda_ortho=0.1, lambda_sparse=0.01):
        super().__init__()
        # self.task_loss_fn = PETSALoss(alpha=0.1)
        self.task_loss_fn = StandardMSELoss()
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
        # self.task_loss_fn = PETSALoss(alpha=0.1)
        self.task_loss_fn = StandardMSELoss()
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
    
def huber_loss(input, target, delta=0.5):
    abs_diff = torch.abs(input - target)
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss.mean()


def build_loss_fn(cfg, loss_name='MSE') -> nn.Module:
    if loss_name == 'MSE':
        return StandardMSELoss()
    elif loss_name == 'PETSA': 
        alpha = getattr(cfg.TTA.DUAL, 'PETSA_LOSS_ALPHA', 0.1)
        return PETSALoss(alpha=alpha)
    elif loss_name == "COBA":
        return CoBA_Loss(lambda_ortho=0.01)
    elif loss_name == "LOWRANK-COBA":
        return LowRankCoBALoss(lambda_ortho=0.01)
    else:
        raise ValueError(f"Unknown Loss type: {loss_name}")


class FreqLowRankOrthoLoss(nn.Module):
    """
    Frequency Domain Orthogonality Loss.
    
    Since the GCM bases in the frequency domain are complex numbers split into 
    Real and Imaginary parts, this loss applies the orthogonality constraint 
    separately to both the real-part bases and the imaginary-part bases.
    
    Input:
        Real: (bases_left, bases_right)
        Imag: (bases_left, bases_right)
    """
    def __init__(self):
        super().__init__()
        # Reuse the existing spatial/time domain low-rank ortho loss
        self.ortho_loss = LowRankOrthoLoss()

    def forward(self, real_left, real_right, imag_left, imag_right):
        """
        Args:
            real_left  (Tensor): (N, L, R, V) Real part of U matrix
            real_right (Tensor): (N, R, L, V) Real part of W matrix
            imag_left  (Tensor): (N, L, R, V) Imaginary part of U matrix
            imag_right (Tensor): (N, R, L, V) Imaginary part of W matrix
            
        Returns:
            Tensor: Sum of orthogonality losses for real and imaginary parts.
        """
        loss_real = self.ortho_loss(real_left, real_right)
        loss_imag = self.ortho_loss(imag_left, imag_right)
        
        return loss_real + loss_imag


class FreqLowRankCoBALoss(nn.Module):
    """
    Frequency Domain CoBA Loss.
    Combines a reconstruction task loss (e.g., MSE) with the Frequency Domain 
    Low-Rank Orthogonality regularization.
    """
    def __init__(self, lambda_ortho=0.01, task_loss_fn=None):
        super().__init__()
        
        # Default to StandardMSELoss if no specific task loss is provided
        self.task_loss_fn = task_loss_fn if task_loss_fn else StandardMSELoss()
        self.ortho_loss_fn = FreqLowRankOrthoLoss()
        self.lambda_ortho = lambda_ortho
 
    def forward(self, pred, ground_truth, real_left, real_right, imag_left, imag_right, coeffs=None):
        """
        Args:
            pred (Tensor): Model predictions
            ground_truth (Tensor): Target values
            real_left, real_right: Real part bases decomposition
            imag_left, imag_right: Imaginary part bases decomposition
            coeffs (Tensor, optional): Coefficients (unused in this specific loss logic but kept for interface consistency)
        """
        
        # 1. Task Loss
        l_task = self.task_loss_fn(pred, ground_truth)
        if torch.isnan(l_task):
            raise ValueError("NaN detected in task loss")
            
        # 2. Frequency Ortho Loss (Real + Imag)
        l_ortho = self.ortho_loss_fn(real_left, real_right, imag_left, imag_right)
        if torch.isnan(l_ortho):
            raise ValueError("NaN detected in ortho loss")
        
        l_total = l_task + (self.lambda_ortho * l_ortho)
        
        return l_total


class ElementWiseOrthoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bases):
        """
        Calculates orthogonality loss for element-wise bases.
        Ensures that for each variable/channel, the N bases are mutually orthogonal.
        
        Args:
            bases shape: (N_bases, Freq_len, N_var) or (N_bases, Freq_len)
        """
        # Handle non-var-wise case (N, F) -> (N, F, 1)
        if bases.ndim == 2:
            bases = bases.unsqueeze(-1)
            
        N, F_len, V = bases.shape
        
        # 1. Normalize along the frequency dimension (feature dimension)
        # We treat the frequency spectrum of each basis as a vector.
        # bases_norm: (N, F, V)
        bases_norm = F.normalize(bases, p=2, dim=1)
        
        # 2. Compute Gram Matrix for each variable group independently
        # For each variable v, we want matrix B_v (N x F) to have orthogonal rows.
        # G[i, j, v] = <b_i,v, b_j,v>
        # Einsum: ifv, jfv -> ijv
        gram_matrix = torch.einsum('ifv, jfv -> ijv', bases_norm, bases_norm)
        
        # 3. Target Identity Matrix
        # Expand identity (N, N) to (N, N, 1) to broadcast against V dimension
        identity = torch.eye(N, device=bases.device).unsqueeze(-1)
        
        # 4. MSE Loss
        # Computes mean squared error between Gram matrix and Identity.
        # This enforces orthogonality (off-diagonals -> 0) and unit norm (diagonals -> 1).
        return F.mse_loss(gram_matrix, identity)

class FreqElementWiseCoBALoss(nn.Module):
    """
    Loss function for CoBA_FreqDomain_ElementWise_GCM.
    Combines task loss (e.g., MSE) with Orthogonality constraints on the 
    Element-Wise Frequency bases (bases_r and bases_i).
    """
    def __init__(self, lambda_ortho=0.01, task_loss_fn=None):
        super().__init__()
        self.task_loss_fn = task_loss_fn if task_loss_fn else StandardMSELoss()
        self.ortho_loss_fn = ElementWiseOrthoLoss()
        self.lambda_ortho = lambda_ortho
 
    def forward(self, pred, ground_truth, bases_r, bases_i, coeffs=None):
        """
        Args:
            pred: Model predictions
            ground_truth: Target values
            bases_r: Real part of bases (N, F, V)
            bases_i: Imaginary part of bases (N, F, V)
            coeffs: Optional coefficients
        """
        
        # 1. Task Loss
        l_task = self.task_loss_fn(pred, ground_truth)
        if torch.isnan(l_task):
            raise ValueError("NaN detected in task loss")
            
        # 2. Ortho Constraints on Real and Imaginary Bases
        # Constraints are applied per n_var group.
        l_ortho_r = self.ortho_loss_fn(bases_r)
        l_ortho_i = self.ortho_loss_fn(bases_i)
        
        l_total = l_task + self.lambda_ortho * (l_ortho_r + l_ortho_i)
        
        return l_total