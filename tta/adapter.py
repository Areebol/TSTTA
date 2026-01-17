import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from device_manager import global_device
from tta.loss import stable_complex_abs

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
            nn.init.zeros_(lin.weight)
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
        
        if global_device == torch.device('npu'):
            amp = stable_complex_abs(x_fft)
        else:
            amp = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        amp_adapted = amp * torch.exp(self.amp_gain)
        
        phase_adapted = phase + self.phase_shift
        
        x_fft_adapted = torch.polar(amp_adapted, phase_adapted)
        
        x_adapted = torch.fft.irfft(x_fft_adapted, n=L, dim=1)
        
        delta = x_adapted - base_pred
        
        return delta
    
class ComplexFreqAdapter(BaseAdapter):
    def __init__(self, pred_len: int, n_vars: int):
        super().__init__(pred_len, n_vars)
        self.freq_len = pred_len // 2 + 1
        self.scale = 1e-5
        
        # Parameters for real and imaginary parts
        # r/i: weights, rb/ib: biases
        self.r = nn.Parameter(self.scale * torch.randn(1, self.freq_len, n_vars))
        self.i = nn.Parameter(self.scale * torch.randn(1, self.freq_len, n_vars))
        # self.r = nn.Parameter(torch.zeros(1, self.freq_len, n_vars))
        # self.i = nn.Parameter(torch.zeros(1, self.freq_len, n_vars))
        self.rb = nn.Parameter(torch.zeros(1, self.freq_len, n_vars))
        self.ib = nn.Parameter(torch.zeros(1, self.freq_len, n_vars))
        
        self.sparsity_threshold = 0.01
        
    def forward(self, base_pred: torch.Tensor) -> torch.Tensor:
        B, L, D = base_pred.shape
        assert L == self.pred_len and D == self.n_vars
        
        # FFT with ortho norm (Energy preserving)
        x_fft = torch.fft.rfft(base_pred, dim=1, norm='ortho')
        
        # Linear Complex Transform (No ReLU, directly learning residual)
        # Delta_real = R*r - I*i + rb
        # Delta_imag = I*r + R*i + ib
        delta_real = (
            x_fft.real * self.r - x_fft.imag * self.i + self.rb
        )
        delta_imag = (
            x_fft.imag * self.r + x_fft.real * self.i + self.ib
        )
        
        # Combine and softshrink (Sparsity on Residual)
        y_stack = torch.stack([delta_real, delta_imag], dim=-1)
        # y_stack = F.softshrink(y_stack, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y_stack)
        
        # iFFT with ortho norm
        # Output is the direct residual delta
        delta = torch.fft.irfft(y, n=L, dim=1, norm='ortho')
        
        return delta


class FreRIAdapter(BaseAdapter):
    def __init__(self, pred_len: int, n_vars: int):
        super().__init__(pred_len, n_vars)
        self.freq_len = pred_len // 2 + 1
        self.scale = 1e-5
        
        # Using matrix multiplication for frequency mixing, processing each variable independently.
        # Params shape: (n_vars, freq_len, freq_len)
        self.r = nn.Parameter(self.scale * torch.randn(n_vars, self.freq_len, self.freq_len))
        self.i = nn.Parameter(self.scale * torch.randn(n_vars, self.freq_len, self.freq_len))
        self.rb = nn.Parameter(torch.zeros(n_vars, self.freq_len))
        self.ib = nn.Parameter(torch.zeros(n_vars, self.freq_len))
        
        self.sparsity_threshold = 0.01

    def forward(self, base_pred: torch.Tensor) -> torch.Tensor:
        B, L, D = base_pred.shape
        assert L == self.pred_len and D == self.n_vars
        
        # FFT: (B, L, D) -> (B, L_freq, D)
        x_fft = torch.fft.rfft(base_pred, dim=1, norm='ortho')
        
        # Permute to (B, D, L_freq) for vector-matrix op per variable
        x_fft = x_fft.permute(0, 2, 1)
        
        # Matrix Multiplication Logic adapted from FreMLP
        # We mix frequencies (L_freq) for each channel (D) independently
        # x: (B, D, L_in), W: (D, L_in, L_out) -> (B, D, L_out)
        
        o_real = (
            torch.einsum('bdl,dlk->bdk', x_fft.real, self.r) - \
            torch.einsum('bdl,dlk->bdk', x_fft.imag, self.i) + \
            self.rb.unsqueeze(0)
        )
        
        o_imag = (
            torch.einsum('bdl,dlk->bdk', x_fft.imag, self.r) + \
            torch.einsum('bdl,dlk->bdk', x_fft.real, self.i) + \
            self.ib.unsqueeze(0)
        )
        
        # Softshrink sparsity
        y_stack = torch.stack([o_real, o_imag], dim=-1)
        # y_stack = F.softshrink(y_stack, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y_stack) # (B, D, L_freq)
        
        # Permute back to (B, L_freq, D) and iFFT
        y = y.permute(0, 2, 1)
        delta = torch.fft.irfft(y, n=L, dim=1, norm='ortho')
        
        return delta
    

class PolarFreqAdapter(BaseAdapter):
    def __init__(self, pred_len: int, n_vars: int):
        super().__init__(pred_len, n_vars)
        self.freq_len = pred_len // 2 + 1
        self.scale = 1e-5
        
        # --- 优化点 1: 解耦能量与周期 ---
        # 不再学习实部/虚部权重，而是学习幅值增益(Mag)和相位偏移(Phase)
        
        # mag_scale: 初始化为0，对应乘法因子为1 (无缩放)
        # 控制“能量”，修复波峰波谷幅度不对的问题
        self.mag_scale = nn.Parameter(torch.zeros(1, self.freq_len, n_vars))
        
        # phase_shift: 初始化为0，对应无相位旋转
        # 控制“周期对齐”，修复预测滞后(Lag)的问题
        self.phase_shift = nn.Parameter(torch.zeros(1, self.freq_len, n_vars))

        # 偏置项，用于修正直流分量漂移
        self.mag_bias = nn.Parameter(torch.zeros(1, self.freq_len, n_vars))
        
        self.sparsity_threshold = 0.01

    def complex_softshrink(self, complex_tensor, lambd):
        """
        --- 优化点 2: 基于能量的稀疏化 (Complex Softshrink) ---
        不同于分别截断实部和虚部，这里根据复数的模长（能量）进行截断。
        这样保持了相位的方向，只收缩能量，物理意义更明确。
        """
        mag = stable_complex_abs(complex_tensor)
        
        # 优化：避免 calculation of angle() 和 polar()，直接缩放
        # scale = (mag - lambd) / mag  (if mag > lambd else 0)
        #       = relu(mag - lambd) / (mag + eps)
        scale = F.relu(mag - lambd) / (mag + 1e-6)
        
        # 显式构造复数，避免 NPU 上 complex * real 可能的问题 (安全起见)
        return torch.complex(complex_tensor.real * scale, complex_tensor.imag * scale)

    def forward(self, base_pred: torch.Tensor) -> torch.Tensor:
        B, L, D = base_pred.shape
        
        # 1. 转到频域
        x_fft = torch.fft.rfft(base_pred, dim=1, norm='ortho')
        
        # 2. 提取输入信号的“能量”和“周期特征”
        # mag_in: (B, L_freq, D)
        mag_in = stable_complex_abs(x_fft)
        
        # 使用 atan2 替代 angle() 以规避 NPU 反向传播中的 complex abs 问题
        phase_in = torch.atan2(x_fft.imag, x_fft.real)
        
        # 3. 极坐标下的校准 (Polar Calibration)
        # 能量校准：原始能量 * (1 + 学习到的增益) + 偏置
        mag_out = mag_in * (1 + self.scale * self.mag_scale) + self.mag_bias
        
        # 周期校准：原始相位 + 学习到的偏移
        phase_out = phase_in + (self.scale * self.phase_shift)
        
        # 4. 重建校准后的复数信号
        # 使用显式 complex 构造替代 torch.polar
        delta_fft = torch.complex(
            mag_out * torch.cos(phase_out), 
            mag_out * torch.sin(phase_out)
        )
        
        # # 5. 计算残差 (Residual)
        # # 我们希望输出的是 delta，而不是校准后的全量，以便后续叠加
        # delta_fft = y_calibrated - x_fft
        
        # 6. 基于能量的稀疏处理
        delta_fft = self.complex_softshrink(delta_fft, self.sparsity_threshold)
        
        # 7. 逆变换回时域
        delta = torch.fft.irfft(delta_fft, n=L, dim=1, norm='ortho')
        
        return delta


class TimeFreqDualAdapter(BaseAdapter):
    """
    简单组合 Linear 和 Freq 的优点。
    输出 = Linear(x) + FreqResidual(x)
    """
    def __init__(self, pred_len: int, n_vars: int):
        super().__init__(pred_len, n_vars)
        
        self.linear_adapter = LinearAdapter(pred_len, n_vars)
        # self.freq_adapter = FreqAdapter(pred_len, n_vars)
        self.freq_adapter = ComplexFreqAdapter(pred_len, n_vars)
        self.lambda_freq = 10
        # self.lambda_freq = nn.Parameter(torch.ones(1, pred_len, n_vars) * 10.0)
        
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

class GCM(nn.Module):
    def __init__(self, window_len, n_var=1, gating_module=None, var_wise=True, low_rank=16):
        super(GCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        self.gating_module = gating_module  # 传入外部定义的门控
        
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))
        self.low_rank = low_rank
        self.lora_A = nn.Parameter(torch.Tensor(window_len, self.low_rank))
        self.lora_B = nn.Parameter(torch.Tensor(self.low_rank, window_len, n_var))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        weight = torch.einsum('ik,kjl->ijl', self.lora_A, self.lora_B)
        gate = self.gating_module(x) 
        x_1 = gate
        # else:
        #     x_1 = torch.tanh(x)

        if self.var_wise:
            new_x = (torch.einsum('biv,iov->bov', x_1, weight) + self.bias)
        else:
            new_x = (torch.einsum('biv,io->bov', x_1, weight) + self.bias)
        return x + new_x

class ShiftAdapter(BaseAdapter):
    def __init__(self, n_vars):
        super().__init__(0, n_vars)
        self.scale = nn.Parameter(torch.ones(1, 1, n_vars))
        self.shift = nn.Parameter(torch.zeros(1, 1, n_vars))

    def forward(self, x):
        return self.shift

    def setup_require_grad(self, require_grad):
        return super().setup_require_grad(require_grad)

class AffineAdapter(BaseAdapter):
    def __init__(self, pred_len: int, n_vars: int):
        super().__init__(pred_len, n_vars)
        # Use a linear layer for each variable to predict affine parameters [scale, shift]
        self.layers = nn.ModuleList([
            nn.Linear(pred_len, 2) for _ in range(n_vars)
        ])
        
        # Initialize alpha (scale) to 1.0 and beta (shift) to 0.0
        for layer in self.layers:
            nn.init.zeros_(layer.weight)
            layer.bias.data[0] = 1.0
            layer.bias.data[1] = 0.0

    def forward(self, base_pred, x_in=None) -> torch.Tensor:
        B, L, D = base_pred.shape
        outs = []
        for i in range(D):
            x = base_pred[:, :, i]  # (B, L)
            params = self.layers[i](x)  # (B, 2)
            
            alpha = params[:, 0:1]  # (B, 1)
            beta = params[:, 1:2]   # (B, 1)
            
            # Apply affine transformation: y = x * alpha + beta
            out = x * alpha + beta
            outs.append(out.unsqueeze(-1))
            
        return torch.cat(outs, dim=-1)
    
class NormAffineAdapter(BaseAdapter):
    def __init__(self, pred_len: int, n_vars: int):
        super().__init__(pred_len, n_vars)
        self.pred_len = pred_len
        self.n_vars = n_vars
        
        # 修改点 1: 建议将 Linear 的输入维度设为 pred_len
        # 这个 Linear 层是用来根据“当前的波形形状”预测“校准参数”的
        self.layers = nn.ModuleList([
            nn.Linear(pred_len, 2) for _ in range(n_vars)
        ])
        
        # 初始化: Scale=1.0, Shift=0.0
        # 对应的 Linear 输出应该是 [1, 0]
        for layer in self.layers:
            nn.init.zeros_(layer.weight)
            # layer.bias[0] -> scale (alpha)
            layer.bias.data[0] = 0.0
            # layer.bias[1] -> shift (beta)
            layer.bias.data[1] = 0.0

    def forward(self, base_pred, x_in) -> torch.Tensor:
        """
        base_pred: (B, L, D) 原始模型的预测输出
        x_in: (B, L_in, D) 原始输入序列 (必须提供，用于计算统计量)
        """
        B, L, D = base_pred.shape
        
        # -----------------------------------------------------------
        # 步骤 1: 计算输入的统计量 (Instance Statistics)
        # -----------------------------------------------------------
        # 加上 eps 防止除零
        mean_in = x_in.mean(dim=1, keepdim=True)  # (B, 1, D)
        std_in = x_in.std(dim=1, keepdim=True) + 1e-5 # (B, 1, D)

        # -----------------------------------------------------------
        # 步骤 2: 将预测结果归一化到标准正态空间 (Re-Normalization)
        # -----------------------------------------------------------
        # 这一步至关重要，它保证了 Linear 层的输入和 base_pred 的梯度都在一个稳定的范围内
        base_pred_norm = (base_pred - mean_in) / std_in

        outs = []
        for i in range(D):
            # 取出第 i 个变量的归一化后的预测值
            # x_norm: (B, L)
            x_norm = base_pred_norm[:, :, i] 
            
            # -------------------------------------------------------
            # 步骤 3: 预测仿射参数 (Predict Affine Params)
            # -------------------------------------------------------
            # params: (B, 2) -> [scale, shift]
            # 注意：这里输入的是归一化后的波形，这样更容易学习
            params = self.layers[i](x_norm) 
            
            alpha = 1.0 + params[:, 0:1]  # (B, 1) Scale
            beta = params[:, 1:2]   # (B, 1) Shift
            
            # -------------------------------------------------------
            # 步骤 4: 在标准空间进行仿射校准
            # -------------------------------------------------------
            # calibrated_norm: (B, L)
            calibrated_norm = x_norm * alpha + beta
            
            outs.append(calibrated_norm.unsqueeze(-1))
            
        # 合并所有通道: (B, L, D)
        y_calibrated_norm = torch.cat(outs, dim=-1)

        # -----------------------------------------------------------
        # 步骤 5: 反归一化 (De-Normalization)
        # -----------------------------------------------------------
        # 使用输入的统计量将结果还原回原始量级
        y_final = y_calibrated_norm * std_in + mean_in - base_pred
            
        return y_final

class RobustAffineAdapter(BaseAdapter):
    def __init__(self, pred_len, n_vars):
        super().__init__(pred_len, n_vars)
        # 直接定义可学习的参数，而不是网络层
        # 初始化为全 0，代表初始状态无改变 (Identity)
        self.delta_scale = nn.Parameter(torch.zeros(1, 1, n_vars)) 
        self.delta_shift = nn.Parameter(torch.zeros(1, 1, n_vars))

    def forward(self, base_pred, x_in):
        # 1. 统计量计算
        mean_in = x_in.mean(dim=1, keepdim=True)
        std_in = x_in.std(dim=1, keepdim=True) + 1e-5

        # 2. 归一化 (把问题转换到良态空间)
        y_norm = (base_pred - mean_in) / std_in

        # 3. 仿射校准 (残差形式: 1 + delta)
        y_calibrated_norm = y_norm * (1 + self.delta_scale) + self.delta_shift

        # 4. 反归一化
        y_final = y_calibrated_norm * std_in + mean_in 
        
        return y_final - base_pred



class NormLinearAdapter(BaseAdapter):
    def __init__(self, pred_len: int, n_vars: int):
        super().__init__(pred_len, n_vars)
        self.pred_len = pred_len
        self.n_vars = n_vars
        
        # 残差网络：只在归一化空间工作
        self.resid_layers = nn.ModuleList([
            nn.Linear(pred_len, pred_len) for _ in range(n_vars)
        ])
        
    #     # 显式的分布校准参数 (Shift & Scale)
    #     self.delta_mu = nn.Parameter(torch.zeros(n_vars)) 
    #     self.delta_sigma = nn.Parameter(torch.zeros(n_vars))

    #     self._init_weights()

    # def _init_weights(self):
    #     for lin in self.resid_layers:
    #         nn.init.xavier_uniform_(lin.weight, gain=0.1)
    #         nn.init.zeros_(lin.bias)

    def forward(self, base_pred, x_in=None) -> torch.Tensor:
        """
        base_pred: (B, L, D)
        """
        B, L, D = base_pred.shape
        
        # --- Step A: Instance Normalization (获取 Base 的统计量) ---
        # 针对每个样本、每个变量计算均值和标准差
        base_mean = base_pred.mean(dim=1, keepdim=True)  # (B, 1, D)
        base_std = base_pred.std(dim=1, keepdim=True) + 1e-5 # (B, 1, D)
        
        # 将输入归一化到标准空间
        z = (base_pred - base_mean) / base_std
        
        # --- Step B: 在归一化空间做残差修正 ---
        outs = []
        for d_idx in range(D):
            # z_var: (B, L)
            z_var = z[:, :, d_idx]
            # 计算残差
            z_res = self.resid_layers[d_idx](z_var)
            outs.append(z_res.unsqueeze(-1))
        
        z_resid_full = torch.cat(outs, dim=-1) # (B, L, D)
        
        # --- Step C: 反归一化回原始空间 ---
        final_pred = z_resid_full * base_std + base_mean
        
        return final_pred


def adapter_factory(name, pred_len, n_vars, cfg):
    if name == 'linear':
        return LinearAdapter(pred_len=pred_len, n_vars=n_vars)
    elif name == 'freq':
        return FreqAdapter(pred_len=pred_len, n_vars=n_vars)
    elif name == "tf-dual":
        return TimeFreqDualAdapter(pred_len=pred_len, n_vars=n_vars)
    elif name == "low-rank":
        return LowRankGatedAdapter(pred_len=pred_len, n_vars=n_vars, rank=cfg.get('rank', 16))
    elif name == "shift":
        return ShiftAdapter(n_vars=n_vars)
    elif name == "affine":
        return AffineAdapter(pred_len=pred_len, n_vars=n_vars)
    elif name == "norm-linear":
        return NormLinearAdapter(pred_len=pred_len, n_vars=n_vars)
    elif name == "norm-affine":
        return NormAffineAdapter(pred_len=pred_len, n_vars=n_vars)
    elif name == "robust-affine":
        return RobustAffineAdapter(pred_len=pred_len, n_vars=n_vars)
    elif name == "complex-freq":
        return ComplexFreqAdapter(pred_len=pred_len, n_vars=n_vars)
    elif name == "freri":
        return FreRIAdapter(pred_len=pred_len, n_vars=n_vars)
    elif name == "polar-freq":
        return PolarFreqAdapter(pred_len=pred_len, n_vars=n_vars)
    else:
        raise ValueError(f"Unknown adapter type: {name}")
