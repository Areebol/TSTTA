import os
import torch
import torch.nn as nn
import numpy as np
from collections import deque

class BaseGating(nn.Module):
    def __init__(self, n_vars: int, **kwargs):
        super().__init__()
        self.n_vars = n_vars

    def forward(self, x_ref) -> torch.Tensor:
        raise NotImplementedError

    def setup_require_grad(self, require_grad: bool):
        for param in self.parameters():
            param.requires_grad = require_grad

class TanhGating(BaseGating):
    def __init__(self, n_vars: int, init_val: float = 0.1, s_max: float = 1.0):
        super().__init__(n_vars)
        self.s_max = s_max
        self.rho = nn.Parameter(init_val * torch.ones(1, 1, n_vars))

    def forward(self, x_ref) -> torch.Tensor:
        rho_eff = self.s_max * torch.tanh(self.rho)
        return rho_eff

class AbsGating(BaseGating):
    def __init__(self, n_vars: int, init_val: float = 0.1, s_max: float = 1.0):
        super().__init__(n_vars)
        self.s_max = s_max
        self.rho = nn.Parameter(init_val * torch.ones(1, 1, n_vars))

    def forward(self, x_ref) -> torch.Tensor:
        rho_eff = self.s_max * torch.abs(self.rho)
        return rho_eff

class AbsTanhGating(BaseGating):
    def __init__(self, n_vars: int, init_val: float = 0.1, s_max: float = 1.0):
        super().__init__(n_vars)
        self.s_max = s_max
        self.rho = nn.Parameter(init_val * torch.ones(1, 1, n_vars))

    def forward(self, x_ref) -> torch.Tensor:
        rho_eff = self.s_max * torch.tanh(torch.abs(self.rho))
        return rho_eff

class MaxGating(BaseGating):
    def __init__(self, n_vars: int, init_val: float = 0.1, s_max: float = 1.0):
        super().__init__(n_vars)
        self.s_max = s_max
        self.rho = nn.Parameter(init_val * torch.ones(1, 1, n_vars))

    def forward(self, x_ref) -> torch.Tensor:
        rho_eff = self.s_max * torch.max(self.rho, torch.zeros_like(self.rho))
        return rho_eff

class CGLossTrendGating(nn.Module):
    """
    CG: Channel Generation Loss Trend Gating
    """
    def __init__(self, n_vars: int, window_size: int = 10, s_max: float = 1.0):
        super().__init__()
        self.n_vars = n_vars
        self.s_max = s_max
        self.window_size = window_size
        self.loss_history = deque(maxlen=window_size)
        self.eps = 1e-6

    def update_loss(self, loss_val):
        if torch.is_tensor(loss_val):
            loss_val = loss_val.detach().cpu().item()
        self.loss_history.append(loss_val)

    def forward(self, x_ref) -> torch.Tensor:
        device = x_ref.device
        dtype = x_ref.dtype

        if len(self.loss_history) < 2:
            return torch.zeros(1, 1, self.n_vars, device=device, dtype=dtype)

        losses = np.array(self.loss_history)
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        curr_loss = losses[-1]

        z_score = (curr_loss - mean_loss) / (std_loss + self.eps)

        z_tensor = torch.tensor(z_score, device=device, dtype=dtype)
        gate_val = self.s_max * torch.sigmoid(z_tensor)
        
        return gate_val.view(1, 1, 1).repeat(1, 1, self.n_vars)

    def setup_require_grad(self, require_grad: bool):
        pass
    
class CILossTrendGating(nn.Module):
    """
    CI: Channel Independent Loss Trend Gating
    """
    def __init__(self, n_vars: int, window_size: int = 10, s_max: float = 1.0):
        super().__init__()
        self.n_vars = n_vars
        self.s_max = s_max
        self.window_size = window_size
        self.register_buffer("loss_history", torch.zeros(window_size, n_vars))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))
        self.eps = 1e-6

    def update_loss(self, per_channel_loss):
        idx = self.step_count % self.window_size
        self.loss_history[idx] = per_channel_loss.detach()
        self.step_count += 1
        
    def forward(self, x_ref) -> torch.Tensor:
        device = x_ref.device
        dtype = x_ref.dtype
        
        self.loss_history = self.loss_history.to(device=device, dtype=dtype)

        if self.step_count < 2:
            return torch.zeros(1, 1, self.n_vars, device=device, dtype=dtype)

        valid_len = min(self.step_count.item(), self.window_size)
        current_history = self.loss_history[:valid_len] # [valid_len, n_vars]

        mean_loss = torch.mean(current_history, dim=0) # [n_vars]
        std_loss = torch.std(current_history, dim=0)   # [n_vars]
        
        last_idx = (self.step_count - 1) % self.window_size
        curr_loss = self.loss_history[last_idx]        # [n_vars]

        z_score = (curr_loss - mean_loss) / (std_loss + self.eps) # [n_vars]
        gate_val = self.s_max * torch.sigmoid(z_score) # [n_vars]
        
        return gate_val.view(1, 1, self.n_vars)

    def setup_require_grad(self, require_grad: bool):
        pass

class FixedGating(nn.Module):
    def __init__(self, n_vars: int, window_size: int = 10, s_max: float = 1.0):
        super().__init__()
        self.n_vars = n_vars
        self.s_max = s_max
        self.window_size = window_size
        self.loss_history = deque(maxlen=window_size)
        self.eps = 1e-6

    def forward(self, x_ref) -> torch.Tensor:
        device = x_ref.device
        dtype = x_ref.dtype

        const_tensor = torch.tensor(1.0, device=device, dtype=dtype)
        gate_val = self.s_max * const_tensor
        
        return gate_val.view(1, 1, 1).repeat(1, 1, self.n_vars)

    def setup_require_grad(self, require_grad: bool):
        pass
    
def gating_factory(name, n_vars, window_size, cfg):
    if name == 'tanh':
        return TanhGating(
            n_vars=n_vars, 
            init_val=cfg.TTA.OURS.GATING.INIT, 
            s_max=getattr(cfg.TTA.OURS, 'S_MAX', 1.0)
        )
    if name == 'abs':
        return AbsGating(
            n_vars=n_vars, 
            init_val=cfg.TTA.OURS.GATING.INIT, 
            s_max=getattr(cfg.TTA.OURS, 'S_MAX', 1.0)
        )
    elif name == 'abs-tanh':
        return AbsTanhGating(
            n_vars=n_vars, 
            init_val=cfg.TTA.OURS.GATING.INIT, 
            s_max=getattr(cfg.TTA.OURS, 'S_MAX', 1.0)
        )
    elif name == 'max':
        return MaxGating(
            n_vars=n_vars, 
            init_val=cfg.TTA.OURS.GATING.INIT, 
            s_max=getattr(cfg.TTA.OURS, 'S_MAX', 1.0)
        )
    elif name == 'ci-loss-trend':
        return CILossTrendGating(
            n_vars=n_vars, 
            window_size=window_size, 
            s_max=getattr(cfg.TTA.OURS, 'S_MAX', 1.0))
    elif name == 'cg-loss-trend':
        return CGLossTrendGating(
            n_vars=n_vars, 
            window_size=window_size, 
            s_max=getattr(cfg.TTA.OURS, 'S_MAX', 1.0))
    elif name == 'fixed':
        return FixedGating(
            n_vars=n_vars, 
            window_size=window_size, 
            s_max=getattr(cfg.TTA.OURS, 'S_MAX', 1.0))
    else:
        raise ValueError(f"Unknown gating type: {name}")