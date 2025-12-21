import torch
import torch.nn as nn

class BaseAdapter(nn.Module):
    def __init__(self, pred_len: int, n_vars: int):
        super().__init__()
        self.pred_len = pred_len
        self.n_vars = n_vars        
    
    def forward(self, x, base_pred):
        raise NotImplementedError("BaseAdapter is an abstract class.")

    def setup_require_grad(self, require_grad: bool):
        for p in self.parameters():
            p.requires_grad_(require_grad)
    
class LinearAdapter(BaseAdapter):
    def __init__(self, pred_len: int, n_vars: int):
        super().__init__(pred_len, n_vars)
        self.layers = nn.ModuleList([
            nn.Linear(pred_len, pred_len) for _ in range(n_vars)
        ])
        for lin in self.layers:
            nn.init.xavier_uniform_(lin.weight, gain=0.1)
            nn.init.zeros_(lin.bias)
    
    def forward(self, base_pred: torch.Tensor) -> torch.Tensor:
        B, L, D = base_pred.shape
        assert L == self.pred_len and D == self.n_vars, \
            f"Adapter expects (B,{self.pred_len},{self.n_vars}), got {base_pred.shape}"
        outs = []
        for d_idx in range(D):
            y_var = base_pred[:, :, d_idx]  # (B, L)
            out_var = self.layers[d_idx](y_var)
            outs.append(out_var.unsqueeze(-1))
        return torch.cat(outs, dim=-1)
    
class LinearAdapter(BaseAdapter):
    def __init__(self, pred_len: int, n_vars: int):
        super().__init__(pred_len, n_vars)
        self.layers = nn.ModuleList([
            nn.Linear(pred_len, pred_len) for _ in range(n_vars)
        ])
        for lin in self.layers:
            nn.init.xavier_uniform_(lin.weight, gain=0.1)
            nn.init.zeros_(lin.bias)
    
    def forward(self, base_pred: torch.Tensor) -> torch.Tensor:
        B, L, D = base_pred.shape
        assert L == self.pred_len and D == self.n_vars, \
            f"Adapter expects (B,{self.pred_len},{self.n_vars}), got {base_pred.shape}"
        outs = []
        for d_idx in range(D):
            y_var = base_pred[:, :, d_idx]  # (B, L)
            out_var = self.layers[d_idx](y_var)
            outs.append(out_var.unsqueeze(-1))
        return torch.cat(outs, dim=-1)

class FreqAdapter(BaseAdapter):
    """
    Spectral Adaptation Module (FreqAdapter)
    
    在频域中分别调整幅值 (Amplitude) 和相位 (Phase)。
    - Amplitude: 对应波动的能量/强度 (Scaling)。
    - Phase: 对应波形的整体平移/滞后 (Shift)。
    
    1. 参数效率: O(L) vs Linear的 O(L^2)。
    2. 物理意义明确: 专门解决 Time Lag 和 Intensity Shift。
    3. 全局感受野: 频域操作天然覆盖整个时间窗口。
    """
    def __init__(self, pred_len: int, n_vars: int):
        super().__init__(pred_len, n_vars)
        
        self.freq_len = pred_len // 2 + 1
        
        self.amp_gain = nn.Parameter(torch.zeros(1, self.freq_len, n_vars))
        
        self.phase_shift = nn.Parameter(torch.zeros(1, self.freq_len, n_vars))
        
    def forward(self, base_pred: torch.Tensor) -> torch.Tensor:
        B, L, D = base_pred.shape
        assert L == self.pred_len and D == self.n_vars
        
        x_fft = torch.fft.rfft(base_pred, dim=1)
        
        amp = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        amp_adapted = amp * torch.exp(self.amp_gain)
        
        phase_adapted = phase + self.phase_shift
        
        x_fft_adapted = torch.polar(amp_adapted, phase_adapted)
        
        x_adapted = torch.fft.irfft(x_fft_adapted, n=L, dim=1)
        
        delta = x_adapted - base_pred
        
        return delta

class TimeFreqDualAdapter(BaseAdapter):
    """
    简单组合 Linear 和 Freq 的优点。
    输出 = Linear(x) + FreqResidual(x)
    """
    def __init__(self, pred_len: int, n_vars: int):
        super().__init__(pred_len, n_vars)
        
        self.linear_adapter = LinearAdapter(pred_len, n_vars)
        self.freq_adapter = FreqAdapter(pred_len, n_vars)
        self.lambda_freq = 10
        
    def forward(self, base_pred: torch.Tensor) -> torch.Tensor:
        pred_time = self.linear_adapter(base_pred)  # 时域变换
        delta_freq = self.freq_adapter(base_pred)   # 频域修正量
        
        return pred_time + self.lambda_freq * delta_freq

class LowRankGatedAdapter(BaseAdapter):
    """
    Low-Rank Gated Adapter (LRGA)
    
    原理：
    利用 Bottleneck 结构 (L -> r -> L) 强制模型捕捉最主要的时间模式（Trend/Principal Components），
    自动过滤掉高频噪声。
    
    结构：
    Input -> [Down Projection] -> Activation -> [Up Projection] -> Gating -> Residual Add
    
    参数量对比 (L=96, r=8):
    - Full Linear: 96*96 = 9216
    - Low Rank:    96*8 + 8*96 = 1536 (减少 83%)
    """
    def __init__(self, pred_len: int, n_vars: int, rank: int = 16):
        super().__init__(pred_len, n_vars)
        self.rank = rank
        
        self.down = nn.Linear(pred_len, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, pred_len, bias=True)
        
        self._init_weights()

    def _init_weights(self):
        import math
        nn.init.kaiming_normal_(self.down.weight, a=math.sqrt(5))
        
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, base_pred: torch.Tensor) -> torch.Tensor:
        B, L, D = base_pred.shape
        assert L == self.pred_len and D == self.n_vars
        
        x_in = base_pred.permute(0, 2, 1).reshape(B * D, L)
        
        x_low = self.down(x_in)
        x_act = self.act(x_low)
        x_out = self.up(x_act) # [B*D, L]
        
        x_out = x_out.reshape(B, D, L).permute(0, 2, 1) # [B, L, D]
        
        delta = x_out
        
        return delta

def adapter_factory(name, pred_len, n_vars, cfg):
    if name == 'linear':
        return LinearAdapter(pred_len=pred_len, n_vars=n_vars)
    elif name == 'freq':
        return FreqAdapter(pred_len=pred_len, n_vars=n_vars)
    elif name == "tf-dual":
        return TimeFreqDualAdapter(pred_len=pred_len, n_vars=n_vars)
    elif name == "low-rank":
        return LowRankGatedAdapter(pred_len=pred_len, n_vars=n_vars, rank=cfg.get('rank', 16))
    else:
        raise ValueError(f"Unknown adapter type: {name}")
