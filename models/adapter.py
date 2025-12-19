import torch
import torch.nn as nn





class OutputAdapter(nn.Module):
    """
    一个轻量级输出适配层：
    - 输入: 原模型预测 y_pred (B, L, D)
    - 输出: 适配后的预测 y_adapted (B, L, D)
    实现: 对每个变量独立做一个 Linear(L -> L)，等价于在时间维上做线性变换。
    这样参数量为: D * (L * L + L)，对子集数据也比较稳定。
    """
    def __init__(self, pred_len: int, n_vars: int):
        super().__init__()
        self.pred_len = pred_len
        self.n_vars = n_vars
        # 对每个变量独立一层 Linear(L -> L)
        self.layers = nn.ModuleList([
            nn.Linear(pred_len, pred_len) for _ in range(n_vars)
        ])
        self._init_weights()

    def _init_weights(self):
        for lin in self.layers:
            # nn.init.eye_(lin.weight)  # 接近恒等映射
            nn.init.xavier_uniform_(lin.weight, gain=0.1) # 较小初始权重
            nn.init.zeros_(lin.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (B, L, D)
        """
        B, L, D = y.shape
        assert L == self.pred_len and D == self.n_vars, \
            f"Adapter expects (B,{self.pred_len},{self.n_vars}), got {y.shape}"
        outs = []
        # 对每个变量单独处理 (B, L) -> (B, L)
        for d_idx in range(D):
            y_var = y[:, :, d_idx]  # (B, L)
            out_var = self.layers[d_idx](y_var) + y_var
            outs.append(out_var.unsqueeze(-1))
        return torch.cat(outs, dim=-1)

    def direct_forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (B, L, D)
        """
        B, L, D = y.shape
        assert L == self.pred_len and D == self.n_vars, \
            f"Adapter expects (B,{self.pred_len},{self.n_vars}), got {y.shape}"
        outs = []
        # 对每个变量单独处理 (B, L) -> (B, L)
        for d_idx in range(D):
            y_var = y[:, :, d_idx]  # (B, L)
            out_var = self.layers[d_idx](y_var)
            outs.append(out_var.unsqueeze(-1))
        return torch.cat(outs, dim=-1)


class InputOutputAdapter(nn.Module):
    """
    改进的适配层：
    - 输入: 原模型输入 x (B, Seq_Len, D) 和 原模型预测 y_pred (B, Pred_Len, D)
    - 输出: 适配后的预测 y_adapted (B, Pred_Len, D)
    实现: y_adapted = y_pred + Linear(x)
    利用历史信息 x 来预测对 y_pred 的修正量。
    """
    def __init__(self, seq_len: int, pred_len: int, n_vars: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        # 对每个变量独立一层 Linear(Seq_Len -> Pred_Len)
        self.layers = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(n_vars)
        ])
        self._init_weights()

    def _init_weights(self):
        for lin in self.layers:
            # nn.init.eye_(lin.weight)  # 接近恒等映射
            nn.init.xavier_uniform_(lin.weight, gain=0.1) # 较小初始权重
            nn.init.zeros_(lin.bias)

    def forward(self, x, y_pred):
        """
        x: (B, Seq_Len, D)
        y_pred: (B, Pred_Len, D)
        """
        B, L, D = x.shape
        B_y, L_y, D_y = y_pred.shape
        
        assert L == self.seq_len and D == self.n_vars, \
            f"Adapter input x shape mismatch: expected (B, {self.seq_len}, {self.n_vars}), got {x.shape}"
        assert L_y == self.pred_len and D_y == self.n_vars, \
            f"Adapter input y_pred shape mismatch: expected (B, {self.pred_len}, {self.n_vars}), got {y_pred.shape}"

        outs = []
        for d_idx in range(D):
            x_var = x[:, :, d_idx]       # (B, Seq_Len)
            y_var = y_pred[:, :, d_idx]  # (B, Pred_Len)
            correction = self.layers[d_idx](x_var) # (B, Pred_Len)
            outs.append((y_var + correction).unsqueeze(-1))
        return torch.cat(outs, dim=-1)

class OutputContextAdapter(nn.Module):
    """
    一个轻量级输出适配层：
    - 输入: 原模型预测 y_pred (B, L, D) 和 上下文 context (D, context_size)
    - 输出: 适配后的预测 y_adapted (B, L, D)
    实现: 对每个变量独立做一个 Linear(L -> L)，等价于在时间维上做线性变换。
    这样参数量为: D * (L * L + L)，对子集数据也比较稳定。
    """
    def __init__(self, pred_len, n_vars, context_size):
        super().__init__()
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.context_size = context_size
        input_dim = pred_len + context_size
        # 对每个变量独立一层 Linear(L -> L)
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, pred_len) for _ in range(n_vars)
        ])
        self._init_weights()

    def _init_weights(self):
        for lin in self.layers:
            # nn.init.eye_(lin.weight)  # 接近恒等映射
            nn.init.xavier_uniform_(lin.weight, gain=0.1) # 较小初始权重
            nn.init.zeros_(lin.bias)

    def forward(self, y, context):
        """
        y: (B, L, D)
        """
        B, L, D = y.shape
        assert L == self.pred_len and D == self.n_vars, \
            f"Adapter expects (B,{self.pred_len},{self.n_vars}), got {y.shape}"
        assert context.shape[-1] == self.context_size, \
            f"Adapter expects context with last dimension {self.context_size}, got {context.shape}"
        
        outs = []
        # 对每个变量单独处理 (B, L) -> (B, L)
        for d_idx in range(D):
            y_var = y[:, :, d_idx]  # (B, L)
            c_var = context[:, d_idx, :]  # (B, context_size)
            c_input = torch.cat([y_var, c_var], dim=-1)
            out_var = self.layers[d_idx](c_input) + y_var
            outs.append(out_var.unsqueeze(-1))
        return torch.cat(outs, dim=-1)

    def direct_forward(self, y, context):
        """
        y: (B, L, D)
        """
        B, L, D = y.shape
        assert L == self.pred_len and D == self.n_vars, \
            f"Adapter expects (B,{self.pred_len},{self.n_vars}), got {y.shape}"
        assert context.shape[-1] == self.context_size, \
            f"Adapter expects context with last dimension {self.context_size}, got {context.shape}"
        
        outs = []
        # 对每个变量单独处理 (B, L) -> (B, L)
        for d_idx in range(D):
            y_var = y[:, :, d_idx]  # (B, L)
            c_var = context[:, d_idx, :]  # (B, context_size)
            c_input = torch.cat([y_var, c_var], dim=-1)
            out_var = self.layers[d_idx](c_input)
            outs.append(out_var.unsqueeze(-1))
        return torch.cat(outs, dim=-1)


class InputOutputContextAdapter(nn.Module):
    """
    综合适配层：
    - 输入: 原模型输入 x (B, Seq_Len, D), 原模型预测 y_pred (B, Pred_Len, D), 上下文 context (B, Context_Size, D)
    - 输出: 适配后的预测 y_adapted (B, Pred_Len, D)
    实现: y_adapted = y_pred + Linear(Concat(x, y_pred, context))
    """
    def __init__(self, seq_len: int, pred_len: int, n_vars: int, context_size: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.context_size = context_size
        
        input_dim = seq_len + pred_len + context_size
        # 对每个变量独立一层 Linear
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, pred_len) for _ in range(n_vars)
        ])
        self._init_weights()

    def _init_weights(self):
        for lin in self.layers:
            nn.init.xavier_uniform_(lin.weight, gain=0.1)
            nn.init.zeros_(lin.bias)

    def forward(self, x, y_pred, context):
        """
        x: (B, Seq_Len, D)
        y_pred: (B, Pred_Len, D)
        context: (B, Context_Size, D)
        """
        B, L, D = x.shape
        B_y, L_y, D_y = y_pred.shape
        
        assert L == self.seq_len and D == self.n_vars, \
            f"Adapter input x shape mismatch: expected (B, {self.seq_len}, {self.n_vars}), got {x.shape}"
        assert L_y == self.pred_len and D_y == self.n_vars, \
            f"Adapter input y_pred shape mismatch: expected (B, {self.pred_len}, {self.n_vars}), got {y_pred.shape}"

        outs = []
        for d_idx in range(D):
            x_var = x[:, :, d_idx]       # (B, Seq_Len)
            y_var = y_pred[:, :, d_idx]  # (B, Pred_Len)
            c_var = context[:, d_idx, :] # (B, Context_Size)
            
            inp = torch.cat([x_var, y_var, c_var], dim=-1)
            correction = self.layers[d_idx](inp) 
            outs.append((y_var + correction).unsqueeze(-1))
        return torch.cat(outs, dim=-1)

    def direct_forward(self, x, y_pred, context):
        """
        x: (B, Seq_Len, D)
        y_pred: (B, Pred_Len, D)
        context: (B, Context_Size, D)
        """
        B, L, D = x.shape
        B_y, L_y, D_y = y_pred.shape
        
        assert L == self.seq_len and D == self.n_vars, \
            f"Adapter input x shape mismatch: expected (B, {self.seq_len}, {self.n_vars}), got {x.shape}"
        assert L_y == self.pred_len and D_y == self.n_vars, \
            f"Adapter input y_pred shape mismatch: expected (B, {self.pred_len}, {self.n_vars}), got {y_pred.shape}"
        
        outs = []
        for d_idx in range(D):
            x_var = x[:, :, d_idx]       # (B, Seq_Len)
            y_var = y_pred[:, :, d_idx]  # (B, Pred_Len)
            c_var = context[:, d_idx, :] # (B, Context_Size)
            
            inp = torch.cat([x_var, y_var, c_var], dim=-1)
            correction = self.layers[d_idx](inp) 
            outs.append(correction.unsqueeze(-1))
        return torch.cat(outs, dim=-1)


def build_output_adapter_for_model(cfg, model: nn.Module) -> nn.Module:
    """
    根据 cfg 和模型输出维度构造一个 Adapter.
    """
    device = next(model.parameters()).device
    pred_len = cfg.DATA.PRED_LEN
    seq_len = getattr(cfg.DATA, "SEQ_LEN", getattr(cfg.MODEL, "seq_len", 96)) # 获取输入序列长度

    # 尝试直接从 cfg 读取通道数
    n_vars = getattr(cfg.MODEL, "c_out", None)
    if n_vars is None:
        n_vars = getattr(cfg.MODEL, "enc_in", None)

    # 使用新的 InputOutputAdapter
    adapter = InputOutputAdapter(seq_len=seq_len, pred_len=pred_len, n_vars=n_vars).to(device)
    return adapter

def build_output_context_adapter(cfg, model, context_size=5) -> nn.Module:
    """
    根据 cfg 和模型输出维度构造一个 Adapter.
    """
    device = next(model.parameters()).device
    pred_len = cfg.DATA.PRED_LEN
    seq_len = getattr(cfg.DATA, "SEQ_LEN", getattr(cfg.MODEL, "seq_len", 96)) # 获取输入序列长度

    # 尝试直接从 cfg 读取通道数
    n_vars = getattr(cfg.MODEL, "c_out", None)
    if n_vars is None:
        n_vars = getattr(cfg.MODEL, "enc_in", None)

    # 使用新的 InputOutputAdapter
    adapter = OutputContextAdapter(pred_len=pred_len, n_vars=n_vars, context_size=context_size).to(device)
    return adapter

def build_input_output_adapter(cfg, model) -> nn.Module:
    """
    根据 cfg 和模型输出维度构造一个 Adapter.
    """
    device = next(model.parameters()).device
    pred_len = cfg.DATA.PRED_LEN
    seq_len = getattr(cfg.DATA, "SEQ_LEN", getattr(cfg.MODEL, "seq_len", 96)) # 获取输入序列长度

    # 尝试直接从 cfg 读取通道数
    n_vars = getattr(cfg.MODEL, "c_out", None)
    if n_vars is None:
        n_vars = getattr(cfg.MODEL, "enc_in", None)

    # 使用新的 InputOutputAdapter
    adapter = InputOutputAdapter(seq_len=seq_len, pred_len=pred_len, n_vars=n_vars).to(device)
    return adapter

def build_input_output_context_adapter(cfg, model, context_size=5) -> nn.Module:
    """
    根据 cfg 和模型输出维度构造一个 Adapter.
    """
    device = next(model.parameters()).device
    pred_len = cfg.DATA.PRED_LEN
    seq_len = getattr(cfg.DATA, "SEQ_LEN", getattr(cfg.MODEL, "seq_len", 96)) # 获取输入序列长度

    # 尝试直接从 cfg 读取通道数
    n_vars = getattr(cfg.MODEL, "c_out", None)
    if n_vars is None:
        n_vars = getattr(cfg.MODEL, "enc_in", None)

    # 使用新的 InputOutputContextAdapter
    adapter = InputOutputContextAdapter(seq_len=seq_len, pred_len=pred_len, n_vars=n_vars, context_size=context_size).to(device)
    return adapter