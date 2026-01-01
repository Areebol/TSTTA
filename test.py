import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tta.tta_dual_utils.GCM import CoBA_GCM

def visualize_bases_interpretation(model, window_len, var_idx=0):
    """
    可视化解释 CoBA_GCM 的 Bases
    Args:
        model: 训练好的 CoBA_GCM 实例
        window_len: 时间窗口长度
        var_idx: 如果是多变量模式，指定看哪个变量的切片
    """
    model.eval()
    n_bases = model.n_bases
    
    # 提取 Bases 权重
    # bases shape: (N, L, L, V) or (N, L, L)
    bases = model.bases.detach().cpu()
    if model.var_wise:
        # 取指定变量的切片 -> (N, L, L)
        bases = bases[..., var_idx]
    
    # ==========================================
    # 1. 矩阵热力图 (Matrix Heatmap)
    # ==========================================
    fig, axes = plt.subplots(2, n_bases, figsize=(3 * n_bases, 6))
    plt.suptitle(f"Analysis of {n_bases} Bases (Variable {var_idx})", fontsize=16)
    
    for i in range(n_bases):
        # 画矩阵权重
        ax = axes[0, i]
        im = ax.imshow(bases[i], cmap='RdBu_r', interpolation='nearest')
        ax.set_title(f"Base {i} Matrix")
        ax.axis('off')
        
    # ==========================================
    # 2. 典型信号响应 (Probe Signal Response)
    # ==========================================
    # 构造几种 Probe Signals: (1, L, 1) 用于模拟单变量输入
    t = torch.linspace(0, 1, window_len)
    
    probes = {
        "Step": (t > 0.5).float(),          # 阶跃
        "Trend": t,                         # 线性趋势
        "Low Freq": torch.sin(2 * np.pi * 2 * t), # 低频
        "High Freq": torch.sin(2 * np.pi * 10 * t)# 高频
    }
    
    # 为了绘图清晰，只选一个最具代表性的信号展示，或者画在这个Base下变化最剧烈的
    # 这里我们统一画 "Step" 和 "High Freq" 的混合效果，或者直接画所有 Probe 经过 Base i 的结果
    
    # 这里演示：输入一个混合信号，看不同 Base 的输出
    mixed_input = probes["Trend"] + 0.3 * probes["High Freq"]
    mixed_input_batch = mixed_input.view(1, window_len, 1).to(bases.device) # (B, L, V_fake)
    
    # 计算 X * W_i
    # Bases[i]: (L, L)
    # Input: (1, L)
    # Output = Input @ Base
    
    for i in range(n_bases):
        ax = axes[1, i]
        
        # 原始输入
        ax.plot(mixed_input.numpy(), label='Input', color='gray', alpha=0.5, linestyle='--')
        
        # 计算 Base i 的独立响应
        # (1, L) @ (L, L) -> (1, L)
        w_i = bases[i] # (L, L)
        # 注意: 你的代码里 einsum 是 'biv, boiv -> bov' (var_wise) 或 'biv, boi -> bov'
        # 简化看作: out = x @ w_i.T 或 x @ w_i
        # 根据代码: w_sample shape是 (L, L), einsum '...li...' -> l是out, i是in
        # 所以应该是: Output = W @ Input (如果是矩阵乘法通常定义) 
        # 但你的代码 einsum('biv, boi -> bov') 意味着: Input(b, i) * Weight(o, i) -> Sum over i
        # 这等价于 Output = Input @ Weight.T (如果 Weight 是 [Out, In])
        # 你的 Weight 定义是 [Out, In] 吗?
        # 代码: self.bases = (N, L, L), einsum 'bn, nli -> bli'. l=out, i=in.
        # 所以 w_sample 是 [Out, In].
        # 计算: x[b, i] * w[b, o, i].
        # 也就是 Output[o] = Sum_i ( X[i] * W[o, i] )
        
        response = torch.einsum('i, oi -> o', mixed_input, w_i)
        
        ax.plot(response.numpy(), label=f'Base {i} Out', color='red')
        ax.set_title(f"Response (Trend+Noise)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("coba_bases_interpretation.png", dpi=300)
    plt.show()

# ==========================================
# 3. 使用示例
# ==========================================
# 假设你已经初始化了模型
window_len = 64
model = CoBA_GCM(window_len=window_len, n_bases=4, var_wise=False) # 举例用4个base便于观察

# 这里手动把 bases 初始化得有特点一些，方便你看效果（实际中是训练出来的）
with torch.no_grad():
    # Base 0: 恒等映射 (Identity) -> 保留原样
    model.bases[0] = torch.eye(window_len) 
    # Base 1: 移动平均 (Smoothing) -> 模糊
    model.bases[1] = torch.ones(window_len, window_len) / window_len 
    # Base 2: 差分 (Differentiation) -> 提取边缘
    diff_matrix = torch.eye(window_len) - torch.diag(torch.ones(window_len-1), -1)
    model.bases[2] = diff_matrix
    # Base 3: 仅仅关注最后几个点 (Attention)
    model.bases[3] = torch.zeros(window_len, window_len)
    model.bases[3][:, -5:] = 1.0 # 所有输出都只受最后5个时间步影响

# 运行可视化
visualize_bases_interpretation(model, window_len)