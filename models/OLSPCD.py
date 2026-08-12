import torch
import torch.nn as nn
import numpy as np
import hashlib
import os
from sklearn.linear_model import Ridge
from torch.utils.data import RandomSampler

# ==================== 统一 PCD 跨通道融合头（纯线性版） ====================
class FusedLinearHead(nn.Module):
    """
    OLSPCD 核心：全局跨变量/通道特征融合预测头
    坚持 OLS 的纯线性先验：移除所有 ReLU 和隐藏层，仅保留一个跨通道映射的 Linear。
    """
    def __init__(
        self, input_dim, input_vars, output_vars, pred_len, head_dropout=0.
    ):
        super().__init__()
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.pred_len = pred_len

        # 全局融合：所有变量的特征平铺拼接
        self.input_dim = input_vars * input_dim
        self.output_dim = output_vars * pred_len

        # 纯线性全局映射（仅修改：bias=False）
        self.linear_fusion = nn.Linear(self.input_dim, self.output_dim, bias=False)

    def forward(self, x):
        # x 形状: [Batch, n_vars, input_dim]
        B, n_vars, _ = x.shape

        if n_vars != self.input_vars:
            raise ValueError(
                f"FusedLinearHead expected {self.input_vars} input channels, "
                f"but received {n_vars}."
            )

        # 1. 展平所有变量，打破通道独立，实现跨通道线性交互
        x = x.reshape(B, -1)  # [Batch, n_vars * input_dim]

        # 2. 全局纯线性映射
        x = self.linear_fusion(x)  # [Batch, n_vars * pred_len]

        # 3. 还原回独立通道的形状以便后续框架处理
        x = x.reshape(B, self.output_vars, self.pred_len)
        return x


class Model(nn.Module):
    def __init__(self, cfg):
        """
        OLS wrapper to simplify some of the OLS fitting.
        """
        super(Model, self).__init__()
        self.cfg = cfg
        self.context_length = cfg.seq_len
        self.horizon = cfg.pred_len
        self.individual = cfg.individual
        self.instance_norm = cfg.instance_norm
        self.verbose = True
        self.n_vars = cfg.enc_in  # 变量/通道数量
        self.dropout = getattr(cfg, "dropout", 0.1)

        # ==================== PCD 核心开关（默认开启融合头） ====================
        self.use_fused_head = getattr(cfg, "use_fused_head", True)
        self.outputs_targets = self.use_fused_head and not self.individual
        self.output_vars = cfg.c_out if self.outputs_targets else self.n_vars
        self.target_start_idx = getattr(cfg, "target_start_idx", 0)
        # Keep solver experiments local to OLSPCD instead of extending the
        # global YACS config. The default remains deterministic ridge; TSVD is
        # enabled explicitly through environment variables.
        self.ols_solver = os.environ.get(
            "OLSPCD_SOLVER",
            "ridge",
        ).lower()
        self.svd_rcond = float(
            os.environ.get("OLSPCD_SVD_RCOND", "0.0")
        )
        self._fit_metadata = {}

        # Disable 'fit_intercept' in Ridge regresion when instance normalization is used.
        fit_intercept = False if self.instance_norm else True

        # Keep the exact feature layout used by the original 20-channel
        # OLSPCD checkpoint.  In particular, instance_norm appends one stdev
        # value per input channel, so old weights have
        # input_vars * (seq_len + 1) input features.
        self.linear_input_dim = (
            self.context_length + 1
            if self.instance_norm
            else self.context_length
        )

        if self.individual:
            # 原版独立通道模型（基于 sklearn 的 Ridge，无修改）
            alpha = cfg.alpha
            seed = cfg.seed
            dataset_train = getattr(cfg, "dataset_train", None)
            num_vars = dataset_train.shape[1] if dataset_train is not None else self.n_vars

            self.models = []
            for _ in range(num_vars):
                self.models.append(
                    Ridge(
                        alpha=alpha,
                        fit_intercept=fit_intercept,
                        tol=0.00001,
                        copy_X=True,
                        max_iter=None,
                        solver='svd',
                        random_state=seed
                    )
                )
        else:
            # ==================== PCD 改造：预测头二选一 ====================
            if self.use_fused_head:
                print("OLSPCD Using FusedLinearHead: Linear Channel Interaction Enabled at Output Layer.")
                self.linear = FusedLinearHead(
                    input_dim=self.linear_input_dim,
                    input_vars=self.n_vars,
                    output_vars=self.output_vars,
                    pred_len=self.horizon,
                    head_dropout=self.dropout
                )
            else:
                # 原版 OLS CI 线性层（各自独立处理各个变量）
                self.linear = nn.Linear(self.linear_input_dim, self.horizon, bias=fit_intercept)

            self.alpha = cfg.alpha

    def fit_ols_solutions(self, train_loader):
        """
        Fit the OLS solutions for each series or in a global mode.
        """
        # A closed-form solution must always see the same complete design
        # matrix. Random sampling plus drop_last silently changes that matrix.
        if getattr(train_loader, "drop_last", False):
            raise ValueError("OLSPCD requires train_loader.drop_last=False.")
        if isinstance(getattr(train_loader, "sampler", None), RandomSampler):
            raise ValueError("OLSPCD requires a non-shuffled train loader.")

        enc_windows = []
        dec_windows = []

        for inputs in train_loader:
            if not isinstance(inputs, (tuple, list)) or len(inputs) < 3:
                raise TypeError(
                    "OLSPCD expects each loader batch to contain "
                    "(enc_window, enc_mark, dec_window, dec_mark)."
                )
            enc_window, _, dec_window, _ = inputs
            if not torch.is_tensor(enc_window) or not torch.is_tensor(dec_window):
                raise TypeError("OLSPCD encoder/decoder windows must be tensors.")

            # Canonical preprocessing is performed on CPU float32 so feature
            # construction is independent of CUDA/NPU kernels. The linear
            # algebra itself is promoted to float64 below.
            enc_window = enc_window.detach().to(device="cpu", dtype=torch.float32)
            dec_window = dec_window.detach().to(device="cpu", dtype=torch.float32)
            dec_window = dec_window[:, -self.horizon:, :]
            enc_windows.append(enc_window)
            dec_windows.append(dec_window)

        if not enc_windows:
            raise RuntimeError("OLSPCD received an empty training loader.")

        enc_windows = torch.cat(enc_windows, dim=0)
        dec_windows = torch.cat(dec_windows, dim=0)

        if self.instance_norm:
            # Preserve the feature construction used by the original OLSPCD
            # closed-form fit: center each raw window and append one raw-scale
            # standard-deviation feature per channel.
            means = enc_windows.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(
                torch.var(
                    enc_windows,
                    dim=1,
                    keepdim=True,
                    unbiased=False,
                )
                + 1e-5
            )
            enc_windows = enc_windows - means
            dec_windows = dec_windows - means
            enc_windows = torch.concat([enc_windows, stdev], dim=1)

        if self.outputs_targets:
            target_end = self.target_start_idx + self.output_vars
            if self.target_start_idx < 0 or target_end > dec_windows.shape[-1]:
                raise ValueError(
                    f"Invalid target slice [{self.target_start_idx}:{target_end}] "
                    f"for decoder with {dec_windows.shape[-1]} channels."
                )
            dec_windows = dec_windows[
                :, :, self.target_start_idx:target_end
            ]

        if self.verbose:
            print('Fitting')

        if self.individual:
            X = enc_windows
            y = dec_windows
            for series_idx in range(X.shape[1]):
                if self.verbose:
                    print(f'\t Fitting in individual mode, series idx {series_idx}')

                X_data = X[:, series_idx, :]
                y_data = y[:, series_idx, :]
                max_train_N = getattr(self.cfg, "max_train_N", None)

                if max_train_N is not None and X_data.shape[0] > max_train_N:
                    idxs = np.arange(X_data.shape[0])
                    idxs = np.random.choice(idxs, size=max_train_N, replace=False)
                    self.models[series_idx].fit(
                        X_data[idxs].cpu().numpy(),
                        y_data[idxs].cpu().numpy()
                    )
                else:
                    self.models[series_idx].fit(
                        X_data.cpu().numpy(),
                        y_data.cpu().numpy()
                    )
        else:
            enc_windows = enc_windows.permute(0, 2, 1)  # (batch, var, seq_len)
            dec_windows = dec_windows.permute(0, 2, 1)  # (batch, var, pred_len)

            # ==================== 核心修改区：适应 SVD 的不同 Reshape 策略 ====================
            if self.use_fused_head:
                # PCD模式: 保留 batch，把 var 和 seq_len/pred_len 融合
                # 形状变为: (batch, var * seq_len)
                enc_windows = enc_windows.reshape(enc_windows.shape[0], -1)
                # 形状变为: (batch, var * pred_len)
                dec_windows = dec_windows.reshape(dec_windows.shape[0], -1)
            else:
                # CI模式 (原版): 把 batch 和 var 融合当成不同的独立样本
                # 形状变为: (batch * var, seq_len)
                enc_windows = enc_windows.reshape(-1, enc_windows.shape[-1])
                # 形状变为: (batch * var, pred_len)
                dec_windows = dec_windows.reshape(-1, dec_windows.shape[-1])

            weight_matrix, solver_metadata = self._solve_fused_ridge(
                enc_windows,
                dec_windows,
            )

            # ==================== 权重赋值路由 ====================
            if self.use_fused_head:
                # 赋值给嵌套在 FusedLinearHead 内层的 nn.Linear
                expected_shape = self.linear.linear_fusion.weight.shape
                if weight_matrix.shape != expected_shape:
                    raise RuntimeError(
                        f"Solved OLSPCD weight has shape {tuple(weight_matrix.shape)}, "
                        f"expected {tuple(expected_shape)}."
                    )
                if not torch.isfinite(weight_matrix).all():
                    raise FloatingPointError(
                        "OLSPCD ridge solver produced NaN or Inf weights."
                    )
                with torch.no_grad():
                    self.linear.linear_fusion.weight.copy_(weight_matrix)
            else:
                # 原版赋值给直接的 nn.Linear
                with torch.no_grad():
                    self.linear.weight.copy_(weight_matrix)

            saved_weight = (
                self.linear.linear_fusion.weight
                if self.use_fused_head
                else self.linear.weight
            )
            weight_bytes = (
                saved_weight.detach().cpu().contiguous().numpy().tobytes()
            )
            self._fit_metadata = {
                **solver_metadata,
                "input_channels": int(self.n_vars),
                "output_channels": int(self.output_vars),
                "target_start_idx": int(self.target_start_idx),
                "instance_norm": bool(self.instance_norm),
                "fit_preprocessing": "legacy_olspcd_instance_norm",
                "train_batches": int(len(train_loader)),
                "train_drop_last": bool(
                    getattr(train_loader, "drop_last", False)
                ),
                "train_sampler": type(
                    getattr(train_loader, "sampler", None)
                ).__name__,
                "saved_weight_dtype": str(saved_weight.dtype),
                "saved_weight_shape": tuple(saved_weight.shape),
                "saved_weight_sha256": hashlib.sha256(weight_bytes).hexdigest(),
                "torch_version": str(torch.__version__),
                "numpy_version": str(np.__version__),
            }

    @torch.no_grad()
    def _solve_fused_ridge(self, x, y):
        """Solve deterministic ridge or truncated-SVD regression.

        x: [num_windows, input_vars * seq_len]
        y: [num_windows, output_vars * pred_len]

        The previous implementation ran float32 SVD on the active accelerator
        and explicitly inverted diag(S**2 + alpha). Both choices make the
        highly collinear eVED solution device-dependent. This implementation
        Both modes compute on CPU float64. Ridge applies the stable
        S / (S**2 + alpha) gain. TSVD removes singular directions below
        svd_rcond * largest_singular_value and uses 1 / S for retained modes.
        """
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(
                f"Ridge solver expects 2-D tensors, got x={x.shape}, y={y.shape}."
            )
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Ridge sample mismatch: x has {x.shape[0]}, y has {y.shape[0]}."
            )

        x64 = x.detach().to(device="cpu", dtype=torch.float64)
        y64 = y.detach().to(device="cpu", dtype=torch.float64)
        x_bytes = memoryview(
            x.detach().cpu().contiguous().numpy()
        ).cast("B")
        y_bytes = memoryview(
            y.detach().cpu().contiguous().numpy()
        ).cast("B")
        design_sha256 = hashlib.sha256(x_bytes).hexdigest()
        target_sha256 = hashlib.sha256(y_bytes).hexdigest()

        if self.ols_solver not in {"ridge", "tsvd"}:
            raise ValueError(
                f"Unsupported OLSPCD solver: {self.ols_solver}. "
                "Expected 'ridge' or 'tsvd'."
            )
        alpha_value = float(self.alpha)
        if not np.isfinite(alpha_value) or alpha_value <= 0:
            raise ValueError(
                f"OLSPCD alpha must be finite and positive: {self.alpha}"
            )
        if not np.isfinite(self.svd_rcond) or not 0 <= self.svd_rcond < 1:
            raise ValueError(
                "OLSPCD svd_rcond must be finite and in [0, 1): "
                f"{self.svd_rcond}"
            )

        # Keep the LAPACK reduction order fixed within one software stack.
        # The training script also fixes OMP/MKL/OpenBLAS thread counts before
        # Python starts, which is required for repeatable CPU linear algebra.
        previous_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            u, singular_values, vh = torch.linalg.svd(
                x64,
                full_matrices=False,
            )
            if singular_values.numel() == 0:
                raise RuntimeError(
                    "OLSPCD ridge solver received an empty design matrix."
                )

            alpha = torch.as_tensor(
                alpha_value,
                dtype=singular_values.dtype,
                device=singular_values.device,
            )
            projected_y = u.transpose(0, 1) @ y64

            cutoff = singular_values[0] * self.svd_rcond
            if self.ols_solver == "ridge":
                spectral_gain = singular_values / (
                    singular_values.square() + alpha
                )
                retained_mask = torch.ones_like(
                    singular_values,
                    dtype=torch.bool,
                )
            else:
                retained_mask = singular_values > cutoff
                if not retained_mask.any():
                    raise RuntimeError(
                        "OLSPCD TSVD removed every singular direction: "
                        f"rcond={self.svd_rcond}."
                    )
                spectral_gain = torch.zeros_like(singular_values)
                spectral_gain[retained_mask] = torch.reciprocal(
                    singular_values[retained_mask]
                )

            coefficients = vh.transpose(0, 1) @ (
                spectral_gain.unsqueeze(1) * projected_y
            )
            weight_matrix = coefficients.transpose(0, 1).contiguous()
        finally:
            torch.set_num_threads(previous_num_threads)

        largest_squared = singular_values[0].square()
        smallest_squared = singular_values[-1].square()
        retained_rank = int(retained_mask.sum().item())
        smallest_retained = singular_values[retained_mask][-1]
        if self.ols_solver == "ridge":
            effective_condition = (
                (largest_squared + alpha)
                / (smallest_squared + alpha)
            )
        else:
            effective_condition = singular_values[0] / smallest_retained
        if self.verbose:
            print(
                "OLSPCD spectral diagnostics: "
                f"solver={self.ols_solver}, "
                f"samples={x.shape[0]}, features={x.shape[1]}, "
                f"outputs={y.shape[1]}, "
                f"alpha={alpha.item():.6e}, "
                f"svd_rcond={self.svd_rcond:.6e}, "
                f"retained_rank={retained_rank}/{singular_values.numel()}, "
                f"effective_condition={effective_condition.item():.6e}, "
                f"max_abs_weight={weight_matrix.abs().max().item():.6e}"
            )

        target_weight = (
            self.linear.linear_fusion.weight
            if self.use_fused_head
            else self.linear.weight
        )
        saved_weight = weight_matrix.to(
            device=target_weight.device,
            dtype=target_weight.dtype,
        )
        metadata = {
            "solver": self.ols_solver,
            "linear_algebra": "torch.linalg.svd",
            "solver_device": "cpu",
            "solver_dtype": "torch.float64",
            "alpha": float(alpha.item()),
            "svd_rcond": float(self.svd_rcond),
            "singular_value_cutoff": float(cutoff.item()),
            "retained_rank": retained_rank,
            "total_rank": int(singular_values.numel()),
            "samples": int(x64.shape[0]),
            "input_features": int(x64.shape[1]),
            "output_features": int(y64.shape[1]),
            "design_matrix_sha256": design_sha256,
            "target_matrix_sha256": target_sha256,
            "largest_singular_value": float(singular_values[0].item()),
            "smallest_singular_value": float(singular_values[-1].item()),
            "smallest_retained_singular_value": float(
                smallest_retained.item()
            ),
            "effective_condition": float(effective_condition.item()),
        }
        return saved_weight, metadata

    def get_fit_metadata(self):
        return dict(self._fit_metadata)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Using the pre-fitted models and context, x, predict to horizon
        """
        # 注意：这里的 x_enc 原始形状是 [B, seq_len, n_vars]
        x_dec = x_dec[:, -self.horizon:, :]
        # Keep the original OLSPCD forward normalization so a compact head
        # produced by slicing legacy weights gives the same target prediction.
        if self.instance_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = torch.concat([x_enc, stdev], dim=1)

        if self.individual:
            X = x_enc
            preds = []
            for series_idx in range(X.shape[1]):  # 注意，如果做了norm，此时X的特征维可能是加了stdev的
                pred_i = self.models[series_idx].predict(X[:, series_idx].cpu().numpy())
                preds.append(pred_i[:, np.newaxis])
            preds = np.concatenate(preds, axis=1)
            preds = torch.from_numpy(preds).to(x_enc.device)
        else:
            # 原本 x_enc: [Batch, seq_len, n_vars]
            # Permute 后: [Batch, n_vars, seq_len(+1)]
            x_enc = x_enc.permute(0, 2, 1)

            # 进入 FusedLinearHead 或原版 nn.Linear 时，形状都完美适配
            pred = self.linear(x_enc)

            # pred 输出形状: [Batch, n_vars, pred_len]
            # 还原回标准时序预测框架的要求: [Batch, pred_len, n_vars]
            preds = pred.permute(0, 2, 1)

        if self.instance_norm:
            if self.outputs_targets:
                target_end = self.target_start_idx + self.output_vars
                means = means[:, :, self.target_start_idx:target_end]
            return preds + means  # Undo instance norm
        else:
            return preds

    def get_head_parameters(self):
        return self.linear.parameters()
