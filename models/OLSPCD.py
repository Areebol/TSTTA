import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import Ridge

from utils.misc import prepare_inputs
from device_manager import global_device

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

        # Disable 'fit_intercept' in Ridge regresion when instance normalization is used.
        fit_intercept = False if self.instance_norm else True

        # 计算输入特征维度（考虑到 instance_norm 会在特征末尾追加 stdev）
        self.linear_input_dim = self.context_length + 1 if self.instance_norm else self.context_length

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
        enc_windows = []
        dec_windows = []

        for inputs in train_loader:
            enc_window, _, dec_window, _ = prepare_inputs(inputs)
            dec_window = dec_window[:, -self.horizon:, :]
            enc_windows.append(enc_window)
            dec_windows.append(dec_window)

        enc_windows = torch.cat(enc_windows, dim=0)
        dec_windows = torch.cat(dec_windows, dim=0)

        if self.instance_norm:
            means = enc_windows.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(enc_windows, dim=1, keepdim=True, unbiased=False) + 1e-5)

            enc_windows = enc_windows - means
            dec_windows = dec_windows - means

            enc_windows = torch.concat([enc_windows, stdev], dim=1)

        if self.outputs_targets:
            target_end = self.target_start_idx + self.output_vars
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

            # svd solver
            U, S, V = torch.svd(enc_windows)
            S_diag = torch.diag(S)

            S_inv = torch.inverse(
                S_diag ** 2 + self.alpha * torch.eye(S_diag.shape[0]).to(global_device)
            )
            weight_matrix = (V @ S_inv @ (S_diag @ U.t() @ dec_windows)).t()

            # ==================== 权重赋值路由 ====================
            if self.use_fused_head:
                # 赋值给嵌套在 FusedLinearHead 内层的 nn.Linear
                self.linear.linear_fusion.weight.data = weight_matrix
            else:
                # 原版赋值给直接的 nn.Linear
                self.linear.weight.data = weight_matrix

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Using the pre-fitted models and context, x, predict to horizon
        """
        # 注意：这里的 x_enc 原始形状是 [B, seq_len, n_vars]
        x_dec = x_dec[:, -self.horizon:, :]
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