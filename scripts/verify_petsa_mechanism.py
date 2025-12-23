import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys

# =========================================================================
# 1. Uncertainty Estimator (保持不变，已验证通过)
# =========================================================================
class UncertaintyEstimator(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, hidden_dim=128):
        super().__init__()
        self.seq_proj = nn.Linear(seq_len, hidden_dim)
        self.pred_proj = nn.Linear(pred_len, hidden_dim)
        self.estimator = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), 
            nn.Softplus()
        )

    def forward(self, x_enc, x_pred):
        feat_seq = self.seq_proj(x_enc.permute(0, 2, 1)).mean(dim=1) 
        feat_pred = self.pred_proj(x_pred.permute(0, 2, 1)).mean(dim=1)
        combined = torch.cat([feat_seq, feat_pred], dim=-1)
        estimated_mse = self.estimator(combined)
        return estimated_mse

# =========================================================================
# 2. 核心修正: UncertaintyAwareGCM
# =========================================================================
class UncertaintyAwareGCM(nn.Module):
    def __init__(self, window_len, n_var=1, hidden_dim=64, gating_init=0.01, var_wise=True, low_rank=16):
        super(UncertaintyAwareGCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        
        self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))
        self.low_rank = low_rank

        self.lora_A = nn.Parameter(torch.Tensor(window_len, self.low_rank))
        self.lora_B = nn.Parameter(torch.Tensor(self.low_rank, window_len, n_var))

        self._init_weights()
    
    def _init_weights(self):
        # 初始化 lora_A
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # --- [关键修复] ---
        # 必须使用非零初始化，否则梯度断裂且输出无变化
        nn.init.normal_(self.lora_B, mean=0.0, std=1e-3)
        # -----------------
        
        nn.init.zeros_(self.bias)
        print("    [Debug] UncertaintyAwareGCM initialized with Gaussian noise (std=1e-3).")

    def forward(self, x, uncertainty_weight=1.0):
        # weight: [Window, Window, Var]
        weight = torch.einsum('ik,kjl->ijl', self.lora_A, self.lora_B)
        
        gate_val = torch.tanh(self.gating) 
        
        if self.var_wise:
            x_gated = gate_val * x
            # einsum: x_gated[Batch, InputWin, Var] * weight[InputWin, OutWin, Var] -> [Batch, OutWin, Var]
            delta = (torch.einsum('biv,iov->bov', x_gated, weight) + self.bias)
        else:
            x_gated = gate_val * x
            delta = (torch.einsum('biv,io->bov', x_gated, weight) + self.bias)

        if isinstance(uncertainty_weight, torch.Tensor):
            if uncertainty_weight.dim() == 1:
                uncertainty_weight = uncertainty_weight.view(-1, 1, 1)
            elif uncertainty_weight.dim() == 2:
                uncertainty_weight = uncertainty_weight.view(-1, 1, 1)
        
        x_final = x + uncertainty_weight * delta
        return x_final

# =========================================================================
# 3. 验证逻辑
# =========================================================================

def run_verification():
    print("=== 开始验证 PETSA 机制有效性 (Fixed Version) ===\n")
    
    B, Seq_Len, Pred_Len, N_Var = 32, 96, 96, 7
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    x_enc = torch.randn(B, Seq_Len, N_Var).to(device)
    x_pred_base = torch.randn(B, Pred_Len, N_Var).to(device)
    
    # Test 1
    print("\n[Test 1] 验证 Uncertainty Estimator (PIR)...")
    try:
        estimator = UncertaintyEstimator(Seq_Len, Pred_Len, N_Var).to(device)
        score = estimator(x_enc, x_pred_base)
        assert score.shape == (B, 1)
        assert (score >= 0).all()
        # Check grad
        loss = score.sum()
        loss.backward()
        print(" -> PASS: 估算器正常。")
    except Exception as e:
        print(f" -> FAIL: {e}")

    # Test 2
    print("\n[Test 2] 验证 GCM 模块的 Identity 特性 (m_t=0)...")
    try:
        gcm = UncertaintyAwareGCM(Pred_Len, N_Var, hidden_dim=64).to(device)
        m_t_zero = torch.zeros(B, 1, 1).to(device)
        out_zero = gcm(x_pred_base, m_t_zero)
        diff = (out_zero - x_pred_base).abs().max().item()
        if diff < 1e-6:
            print(f" -> PASS: m_t=0 时输出不变 (Max Diff: {diff:.9f})。")
        else:
            print(f" -> FAIL: m_t=0 时输出改变了 (Max Diff: {diff:.9f})。")
    except Exception as e:
        print(f" -> FAIL: {e}")

    # Test 3
    print("\n[Test 3] 验证 GCM 模块的调制敏感性 (Modulation Sensitivity)...")
    try:
        gcm.eval()
        with torch.no_grad():
            m_t_small = torch.ones(B, 1, 1).to(device) * 0.1
            m_t_large = torch.ones(B, 1, 1).to(device) * 10.0
            
            out_small = gcm(x_pred_base, m_t_small)
            out_large = gcm(x_pred_base, m_t_large)
            
            delta_small = (out_small - x_pred_base).abs().mean().item()
            delta_large = (out_large - x_pred_base).abs().mean().item()
            
            print(f"    Delta (m_t=0.1):  {delta_small:.8f}")
            print(f"    Delta (m_t=10.0): {delta_large:.8f}")
            
            if delta_large > delta_small:
                print(" -> PASS: 机制生效，m_t 越大修正幅度越大。")
            elif delta_large == 0.0:
                print(" -> FAIL: Delta 依然是 0，请检查初始化代码！")
            else:
                print(" -> FAIL: m_t 改变未影响输出幅度。")

    except Exception as e:
        print(f" -> FAIL: {e}")

    # Test 4
    print("\n[Test 4] 模拟 TTA 优化步骤 (检查梯度)...")
    try:
        gcm.train()
        # 强制重置参数以确保测试独立性
        gcm._init_weights()
        optimizer = torch.optim.Adam(gcm.parameters(), lr=0.1) # 加大 LR 让变化明显
        
        initial_A = gcm.lora_A.clone()
        m_t = torch.ones(B, 1, 1).to(device)
        
        pred = gcm(x_pred_base, m_t)
        target = torch.randn_like(pred)
        loss = nn.MSELoss()(pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        
        # 打印梯度范数
        grad_norm_A = gcm.lora_A.grad.norm().item()
        grad_norm_B = gcm.lora_B.grad.norm().item()
        print(f"    Gradient Norm (A): {grad_norm_A:.6f}")
        print(f"    Gradient Norm (B): {grad_norm_B:.6f}")
        
        optimizer.step()
        
        diff_A = (gcm.lora_A - initial_A).abs().sum().item()
        if diff_A > 0:
            print(f" -> PASS: 参数成功更新 (A 变化量: {diff_A:.6f})。")
        else:
            print(" -> FAIL: 参数未更新，梯度断裂。")

    except Exception as e:
        print(f" -> FAIL: {e}")

if __name__ == "__main__":
    run_verification()