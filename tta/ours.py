import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 移除了 norm_knowledge 相关的 import
from datasets.loader import get_test_dataloader
from utils.misc import prepare_inputs
from models.forecast import forecast
from tta.utils import save_tta_results
import matplotlib.pyplot as plt

class BaseAdapter(nn.Module):
    def __init__(self, pred_len: int, n_vars: int):
        super().__init__()
        self.pred_len = pred_len
        self.n_vars = n_vars        
    def forward(self, x, base_pred):
        raise NotImplementedError("BaseAdapter is an abstract class.")
    
    def _init_weights(self):
        raise NotImplementedError("BaseAdapter is an abstract class.")
    
class SimpleLinearOutputAdapter(BaseAdapter):
    def __init__(self, pred_len: int, n_vars: int):
        super().__init__(pred_len, n_vars)
        self.layers = nn.ModuleList([
            nn.Linear(pred_len, pred_len, bias=False) for _ in range(n_vars)
        ])
        self._init_weights()
    def _init_weights(self):
        for lin in self.layers:
            nn.init.xavier_uniform_(lin.weight, gain=0.1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (B, L, D)
        """
        B, L, D = y.shape
        assert L == self.pred_len and D == self.n_vars, \
            f"Adapter expects (B,{self.pred_len},{self.n_vars}), got {y.shape}"
        outs = []
        for d_idx in range(D):
            y_var = y[:, :, d_idx]  # (B, L)
            out_var = self.layers[d_idx](y_var)
            outs.append(out_var.unsqueeze(-1))
        return torch.cat(outs, dim=-1)
def build_output_adapter_for_model(cfg, model):
    device = next(model.parameters()).device
    pred_len = cfg.DATA.PRED_LEN
    seq_len = cfg.DATA.SEQ_LEN

    n_vars = getattr(cfg.MODEL, "c_out", None)
    if n_vars is None: n_vars = getattr(cfg.MODEL, "enc_in", None)

    # 使用双流适配器
    adapter = SimpleLinearOutputAdapter(pred_len=pred_len, n_vars=n_vars)

    adapter.to(device)
    return adapter

class TTARunner(nn.Module):
    def __init__(self, cfg, model: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.model = model

        n_vars = getattr(cfg.MODEL, "c_out", None)
        if n_vars is None: 
            n_vars = getattr(cfg.MODEL, "enc_in", None)
        self.n_vars = n_vars
        self.use_context = getattr(cfg.TTA.OURS, "USE_CONTEXT", True)
        self.base_adapter = build_output_adapter_for_model(cfg, model)
        device = next(model.parameters()).device
        self.base_adapter.to(device)

        for p in self.base_adapter.parameters():
            p.requires_grad_(True)
        
        for p in self.model.parameters():
            p.requires_grad_(False)

        # 初始化 rho (残差连接的门控系数)
        # y_hat = y_base + rho * adapter(y_base)
        # self.rho = nn.Parameter(cfg.TTA.OURS.GATING_INIT * torch.ones(1, device=device))
        self.rho = nn.Parameter(
            cfg.TTA.OURS.GATING_INIT * torch.ones(1, 1, n_vars, device=device)
        )
        self.softmax = getattr(cfg.TTA.OURS, 'SOFTMAX', True) # 仅用于日志打印或rho计算(如果涉及)
        self.s_max = getattr(cfg.TTA.OURS, 'S_MAX', 1.0)
        
        self.test_loader = get_test_dataloader(cfg)
        self.test_data = self.test_loader.dataset.test
        
        if hasattr(self.test_loader.dataset, "get_test_num_windows"):
            test_num_windows = self.test_loader.dataset.get_test_num_windows()
            batch_size = test_num_windows
        else:
            batch_size = len(self.test_loader.dataset)
        self.test_loader = get_test_dataloader(cfg, batch_size=batch_size)

        self.loss_fn = nn.MSELoss()
        self.steps_per_batch = getattr(cfg.TTA.OURS, 'STEPS_PER_BATCH', 1)

        self.set_optimizer()

        self.cur_step = self.cfg.DATA.SEQ_LEN - 2
        self.pred_step_end_dict = {}
        self.inputs_dict = {}
        self.n_adapt = 0
        self.mse_all = []
        self.mae_all = []
        test_types_str = getattr(cfg.TTA.OURS, 'TEST_TYPES', 'eved')
        self.test_types = [t.strip() for t in test_types_str.split(',') if t.strip()]
        self.results = {}
        self.reg_lambda = 0.2

    def set_optimizer(self):
        """
        设置优化器，只优化 Adapter 参数和 rho。
        """
        base_lr = getattr(self.cfg.TTA.OURS, 'LR', 1e-3)
        rho_lr = base_lr * getattr(self.cfg.TTA.OURS, 'GATING_LR_SCALE', 1)

        param_groups = [
            {
                'params': [self.rho],
                'lr': rho_lr,
            },
            {
                'params': self.base_adapter.parameters(),
                'lr': base_lr,
            }
        ]
        self.optimizer = torch.optim.Adam(param_groups)

    def _forward_with_adapter(self, base_pred, enc_window):
        """
        Learnable 模式的前向传播:
        y_hat = base_pred + rho * adapter(base_pred)
        """
        if self.base_adapter is None:
            return base_pred
        if self.use_context:
            input_full = torch.cat([enc_window, base_pred], dim=1)
        else:
            input_full = base_pred
        
        adapter_out = self.base_adapter(input_full)
            
        rho_eff = self.s_max * torch.tanh(torch.abs(self.rho))
        self.log_rho_eff = rho_eff.detach().cpu()
        return base_pred + rho_eff * adapter_out

    def _calculate_period_and_batch_size(self, enc_window_first):
        fft_result = torch.fft.rfft(enc_window_first - enc_window_first.mean(dim=0), dim=0)
        amplitude = torch.abs(fft_result)
        power = torch.mean(amplitude ** 2, dim=0)
        try:
            period = enc_window_first.shape[0] // torch.argmax(amplitude[:, power.argmax()]).item()
        except:
            period = 24
        period *= self.cfg.TTA.OURS.PERIOD_N
        batch_size = period + 1
        return period, batch_size

    def _adapt_with_full_ground_truth_if_available(self):
        while self.pred_step_end_dict and self.cur_step >= self.pred_step_end_dict[min(self.pred_step_end_dict.keys())]:
            batch_idx_available = min(self.pred_step_end_dict.keys())
            inputs_history = self.inputs_dict.pop(batch_idx_available)

            pred, ground_truth = forecast(self.cfg, inputs_history, self.model, None)

            for _ in range(self.cfg.TTA.OURS.STEPS):
                self.n_adapt += 1
                pred_adapted = self._forward_with_adapter(pred, inputs_history[0])
                mse_loss = F.mse_loss(pred_adapted, ground_truth)
                reg_loss = F.mse_loss(pred_adapted, pred) 
                loss = mse_loss + self.reg_lambda * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.pred_step_end_dict.pop(batch_idx_available)

    def _adapt_with_partial_ground_truth(self, pred, ground_truth, period, batch_size, batch_idx, cur_enc_window):
        for _ in range(self.cfg.TTA.OURS.STEPS):
            self.n_adapt += 1
            pred_adapted = self._forward_with_adapter(pred, cur_enc_window)

            pred_partial, ground_truth_partial = pred_adapted[0][:period], ground_truth[0][:period]
            mse_partial = F.mse_loss(pred_partial, ground_truth_partial)
            reg_loss = F.mse_loss(pred_adapted, pred)
            loss = mse_partial + self.reg_lambda * reg_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def plot_results(self, save_name='tta_all_batches.png'):
        """
        绘制所有 Batch 拼接后的结果。
        默认策略：取出每个预测窗口的最后一个点拼接成连续曲线 (Last Step Approach)，
        这通常能最好地代表连续的时间序列走势。
        """
        n_vars = self.n_vars
        
        # 数据维度: (N_Samples, Pred_Len, N_Vars)
        # 策略A：完全展平 (N_Samples * Pred_Len)。缺点：如果有滑窗重叠，图像会呈锯齿状。
        # data_base = self.full_pred_base.reshape(-1, n_vars)
        
        # 策略B (推荐)：取每个窗口的最后一个预测点。这样能连成一条较平滑的线。
        # 假设 stride=1，这几乎就是原始序列的重建。
        # 如果你想要其他视角（比如只看每个窗口的第一步），可以改索引 [:,-1,:] 为 [:,0,:]
        data_base = self.full_pred_base[:, 0, :] 
        data_tta = self.full_pred_tta[:, 0, :]
        data_gt = self.full_gt[:, 0, :]

        # 创建图表
        # 如果变量太多，限制一下图片高度
        fig_height = min(4 * n_vars, 60) 
        fig, axes = plt.subplots(n_vars, 1, figsize=(15, fig_height), sharex=True)
        
        if n_vars == 1:
            axes = [axes]
            
        x_axis = np.arange(data_base.shape[0])

        for i in range(n_vars):
            ax = axes[i]
            # 绘制 GT
            ax.plot(x_axis, data_gt[:, i], label='Ground Truth', color='black', alpha=0.5, linewidth=1.5)
            # 绘制 Base
            ax.plot(x_axis, data_base[:, i], label='Before TTA', color='blue', linestyle='--', alpha=0.7, linewidth=1)
            # 绘制 TTA
            ax.plot(x_axis, data_tta[:, i], label='After TTA', color='red', alpha=0.9, linewidth=1)
            
            ax.set_title(f'Variable {i} (Concatenated Last-Step Predictions)', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        print(f"Comparison plot saved to {save_name}")
        # plt.show() # 如果在 Jupyter 中可以取消注释

    def analyze_simple_diagnostics(self, save_dir='analysis_results'):
        """
        针对 SimpleAdapter 的诊断可视化：
        1. 修正量分析 (Modification Analysis)
        2. 步长误差 (Error by Horizon)
        3. 散点分布 (Scatter Plot)
        """
        import matplotlib.pyplot as plt
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        if not hasattr(self, 'full_pred_base') or len(self.full_pred_base) == 0:
            print("请先运行 adapt() 获取数据。")
            return

        print("Generating diagnostics for Simple Adapter...")
        
        # --- 准备数据 ---
        # 我们取所有 Batch 拼接后的数据
        # (N_Samples, Pred_Len, N_Vars)
        # 为了画“修正量分析”，我们选一个 Rho 最大的变量，并选该变量误差最大的那个样本
        
        # 1. 找到 Rho 最大的变量 (代表 Adapter 介入最多的变量)
        rho_vals = (self.s_max * torch.abs(self.rho)).detach().cpu().numpy().flatten()
        best_var = np.argmax(rho_vals)
        print(f"Visualizing Variable {best_var} (Rho={rho_vals[best_var]:.4f})")
        
        # 2. 找到该变量在这个 Batch 中改变幅度最大的样本 (Sample Index)
        # diff = |TTA - Base|
        diffs = np.mean(np.abs(self.full_pred_tta[:, :, best_var] - self.full_pred_base[:, :, best_var]), axis=1)
        sample_idx = np.argmax(diffs)
        
        # 提取曲线
        base_curve = self.full_pred_base[sample_idx, :, best_var]
        tta_curve = self.full_pred_tta[sample_idx, :, best_var]
        gt_curve = self.full_gt[sample_idx, :, best_var]
        adjustment = tta_curve - base_curve  # 这就是 Adapter 做的“功”
        
        # --- Plot 1: Modification Analysis (修正分解) ---
        plt.figure(figsize=(10, 6))
        
        # 上图：对比 Base, TTA, GT
        plt.subplot(2, 1, 1)
        plt.plot(base_curve, 'b--', label='Base Pred', alpha=0.6)
        plt.plot(tta_curve, 'r-', label='TTA Pred')
        plt.plot(gt_curve, 'k-', label='Ground Truth', alpha=0.4)
        plt.title(f'Prediction Comparison (Sample {sample_idx}, Var {best_var})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 下图：单独画 Adapter 的修正量
        plt.subplot(2, 1, 2)
        plt.plot(adjustment, 'g-', label='Adapter Adjustment (Delta)', linewidth=2)
        plt.axhline(0, color='k', linestyle='--', alpha=0.3)
        plt.title('What did the Adapter add? (The "Delta")')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, '1_simple_decomposition.png'))
        plt.close()
        print(f"Saved modification analysis to {save_dir}/1_simple_decomposition.png")

        # --- Plot 2: Error by Horizon (预测步长误差分布) ---
        # 检查 TTA 是否牺牲了长时预测精度
        mse_base_step = np.mean((self.full_pred_base - self.full_gt)**2, axis=(0, 2))
        mse_tta_step = np.mean((self.full_pred_tta - self.full_gt)**2, axis=(0, 2))
        
        plt.figure(figsize=(8, 4))
        plt.plot(mse_base_step, label='Base MSE', marker='.', color='blue')
        plt.plot(mse_tta_step, label='TTA MSE', marker='.', color='red')
        plt.xlabel('Prediction Step (Horizon)')
        plt.ylabel('MSE')
        plt.title('Error Distribution across Prediction Horizon')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.savefig(os.path.join(save_dir, '2_error_horizon.png'))
        plt.close()
        print(f"Saved horizon error to {save_dir}/2_error_horizon.png")

        # --- Plot 3: Scatter Plot (幅度验证) ---
        # 随机采样 2000 个点
        flat_gt = self.full_gt.flatten()
        flat_base = self.full_pred_base.flatten()
        flat_tta = self.full_pred_tta.flatten()
        
        if len(flat_gt) > 2000:
            indices = np.random.choice(len(flat_gt), 2000, replace=False)
        else:
            indices = np.arange(len(flat_gt))
            
        plt.figure(figsize=(6, 6))
        min_val, max_val = flat_gt.min(), flat_gt.max()
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5) # 对角线
        
        plt.scatter(flat_gt[indices], flat_base[indices], alpha=0.3, s=15, c='blue', label='Base')
        plt.scatter(flat_gt[indices], flat_tta[indices], alpha=0.3, s=15, c='red', label='TTA')
        
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.title('Prediction vs Ground Truth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, '3_scatter.png'))
        plt.close()
        print(f"Saved scatter plot to {save_dir}/3_scatter.png")
    @torch.no_grad()
    def _adjust_prediction(self, pred, inputs, batch_size, period):
        pred_after_adapt, ground_truth = forecast(self.cfg, inputs, self.model, None)
        pred_after_adapt = self._forward_with_adapter(pred_after_adapt)
        
        for i in range(batch_size - 1):
            pred[i, period - i:] = pred_after_adapt[i, period - i:]
        return pred, ground_truth

    def _build_test_loader(self):
        self.test_loader = get_test_dataloader(self.cfg)
        self.test_data = self.test_loader.dataset.test
        if hasattr(self.test_loader.dataset, "get_test_num_windows"):
            test_num_windows = self.test_loader.dataset.get_test_num_windows()
            batch_size = test_num_windows
        else:
            batch_size = len(self.test_loader.dataset)
        self.test_loader = get_test_dataloader(self.cfg, batch_size=batch_size)

    def _reset_state(self):
        self.rho.data.fill_(self.cfg.TTA.OURS.GATING_INIT)
        
        device = next(self.model.parameters()).device
        self.base_adapter = build_output_adapter_for_model(self.cfg, self.model)
        self.base_adapter.to(device)
        for p in self.base_adapter.parameters():
            p.requires_grad_(True)
            
        self.set_optimizer()

    @torch.enable_grad()
    def _adapt(self):
        self.mse_all = []
        self.mae_all = []
        self.n_adapt = 0
        self.pred_step_end_dict = {}
        self.inputs_dict = {}
        self.cur_step = self.cfg.DATA.SEQ_LEN - 2
        self.all_preds_base = []
        self.all_preds_tta = []
        self.all_gts = []
        self.model.eval()
        log_rho_eff_list = []

        for idx, inputs in enumerate(self.test_loader):
            enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
            batch_start = 0
            batch_end = 0
            batch_idx = 0
            self.cur_step = self.cfg.DATA.SEQ_LEN - 2

            while batch_end < len(enc_window_all):
                enc_window_first = enc_window_all[batch_start]

                if self.cfg.TTA.OURS.PAAS:
                    period, batch_size = self._calculate_period_and_batch_size(enc_window_first)
                else:
                    batch_size = self.cfg.TTA.OURS.BATCH_SIZE
                    period = batch_size - 1

                batch_end = batch_start + batch_size
                if batch_end > len(enc_window_all):
                    batch_end = len(enc_window_all)
                    batch_size = batch_end - batch_start

                self.cur_step += batch_size

                cur_inputs = (
                    enc_window_all[batch_start:batch_end],
                    enc_window_stamp_all[batch_start:batch_end],
                    dec_window_all[batch_start:batch_end],
                    dec_window_stamp_all[batch_start:batch_end],
                )
                cur_enc_window = cur_inputs[0]

                self.pred_step_end_dict[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN
                self.inputs_dict[batch_idx] = cur_inputs

                # Forecast
                pred, ground_truth = forecast(self.cfg, cur_inputs, self.model, None)
                self.all_preds_base.append(pred.detach().cpu().numpy())
                # Adapter Forward (No Grad for evaluation first)
                with torch.no_grad():
                    pred_adapter = self._forward_with_adapter(pred, cur_enc_window)
                
                # Adaptation
                self._adapt_with_full_ground_truth_if_available()
                self._adapt_with_partial_ground_truth(pred, ground_truth, period, batch_size, batch_idx, cur_enc_window)

                # Adjust Prediction
                if self.cfg.TTA.OURS.ADJUST_PRED:
                    pred_adapter, ground_truth = self._adjust_prediction(pred_adapter, cur_inputs, batch_size, period)
                log_rho_eff_list.append(self.log_rho_eff)
                self.all_preds_tta.append(pred_adapter.detach().cpu().numpy())
                self.all_gts.append(ground_truth.detach().cpu().numpy())
                # Metrics
                mse = F.mse_loss(pred_adapter, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred_adapter, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                self.mse_all.append(mse)
                self.mae_all.append(mae)

                batch_start = batch_end
                batch_idx += 1

        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
       
        self.full_pred_base = np.concatenate(self.all_preds_base, axis=0)
        self.full_pred_tta = np.concatenate(self.all_preds_tta, axis=0)
        self.full_gt = np.concatenate(self.all_gts, axis=0)
        print(f'Number of adaptations: {self.n_adapt}')
        print(f'Test MSE: {self.mse_all.mean():.4f}, Test MAE: {self.mae_all.mean():.4f}')
        save_tta_results(
            tta_method='Ours-N-vars',
            seed=self.cfg.SEED,
            model_name=self.cfg.MODEL.NAME,
            dataset_name=self.cfg.DATA.NAME,
            pred_len=self.cfg.DATA.PRED_LEN,
            mse_after_tta=self.mse_all.mean(),
            mae_after_tta=self.mae_all.mean(),
        )
        log_rho_eff_arr = np.array(log_rho_eff_list)  # shape: (steps, bs, 1, n_vars)
        np.save('log_rho_eff_history.npy', log_rho_eff_arr)
        print("Plotting results for all variables...")
        self.plot_results()
        self.analyze_simple_diagnostics()
    def adapt(self):
        self._build_test_loader()
        self._adapt()
def build_tta_runner(cfg, model):
    """
    Builder function for LearnableAdapterTTA.
    Purely learnable mode, no external weights required.
    """
    return TTARunner(cfg, model)