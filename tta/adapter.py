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

def adapter_factory(name, pred_len, n_vars, cfg):
    if name == 'linear':
        return LinearAdapter(pred_len=pred_len, n_vars=n_vars)
    else:
        raise ValueError(f"Unknown adapter type: {name}")
