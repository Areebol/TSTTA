import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from copy import deepcopy

from datasets.loader import get_test_dataloader, get_train_dataloader
from utils.misc import prepare_inputs, mkdir
from models.forecast import forecast
from config import get_norm_method
from tta.utils import save_tta_results, TTADataManager
from models.optimizer import get_optimizer

# -------------------------------------------------------------------------
# 1. 辅助模块: 损失函数与不确定性估算器 (PIR Part)
# -------------------------------------------------------------------------

class UncertaintyEstimator(nn.Module):
    """
    PIR Quality Estimator 的轻量化版本。
    输入：原始序列特征 (Batch, Seq_Len, D) + 初步预测 (Batch, Pred_Len, D)
    输出：估计的 MSE Error (Batch, 1)
    """
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

class CorrCoefLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = preds.reshape(-1)
        y = target.reshape(-1)
        var_x = torch.var(x)
        var_y = torch.var(y)
        if (var_x < self.eps) or (var_y < self.eps):
            return torch.zeros((), dtype=preds.dtype, device=preds.device)
        
        data = torch.stack([x, y], dim=0)
        corrmat = torch.corrcoef(data)
        corr_xy = corrmat[0, 1]
        corr_xy = torch.nan_to_num(corr_xy, nan=0.0, posinf=0.0, neginf=0.0)
        corr_xy = torch.clamp(corr_xy, -1.0, 1.0)
        return -corr_xy

# -------------------------------------------------------------------------
# 2. 核心结构: 注入不确定性的 PETSA GCM 模块 (治疗模块)
# -------------------------------------------------------------------------

class UncertaintyAwareGCM(nn.Module):
    """
    融合了不确定性调制的 PETSA GCM 模块。
    公式: Out = In + m_t * Tanh(Gate) * Adapter(In)
    """
    def __init__(self, window_len, n_var=1, hidden_dim=64, gating_init=0.01, var_wise=True, low_rank=16):
        super(UncertaintyAwareGCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        
        # PETSA 参数
        self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))
        self.low_rank = low_rank

        # LoRA 风格的低秩矩阵
        self.lora_A = nn.Parameter(torch.Tensor(window_len, self.low_rank))
        self.lora_B = nn.Parameter(torch.Tensor(self.low_rank, window_len, n_var))

        self._init_weights()
    
    def _init_weights(self):
        # lora_A 使用 Kaiming 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # --- [关键修改] ---
        # 必须使用非零初始化，否则梯度断裂且输出无变化
        nn.init.normal_(self.lora_B, mean=0.0, std=1e-3)
        
        nn.init.zeros_(self.bias)

    def forward(self, x, uncertainty_weight=1.0):
        # 计算低秩权重: [Seq, Rank] x [Rank, Seq, Var] -> [Seq, Seq, Var]
        weight = torch.einsum('ik,kjl->ijl', self.lora_A, self.lora_B)
        
        # 基础门控
        gate_val = torch.tanh(self.gating) 
        
        # 计算 Delta
        if self.var_wise:
            x_gated = gate_val * x
            delta = (torch.einsum('biv,iov->bov', x_gated, weight) + self.bias)
        else:
            x_gated = gate_val * x
            delta = (torch.einsum('biv,io->bov', x_gated, weight) + self.bias)

        # 维度对齐 uncertainty_weight
        if isinstance(uncertainty_weight, torch.Tensor):
            if uncertainty_weight.dim() == 1:
                uncertainty_weight = uncertainty_weight.view(-1, 1, 1)
            elif uncertainty_weight.dim() == 2:
                uncertainty_weight = uncertainty_weight.view(-1, 1, 1)
        
        # 注入不确定性强度 m_t
        x_final = x + uncertainty_weight * delta
        return x_final


class CalibrationModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pred_len = cfg.DATA.PRED_LEN
        self.n_var = cfg.DATA.N_VAR
        # PETSA Hyperparams
        hd = cfg.TTA.PETSA.HIDDEN_DIM
        init = cfg.TTA.PETSA.GATING_INIT
        vw = cfg.TTA.PETSA.GCM_VAR_WISE
        rank = cfg.TTA.PETSA.RANK
        self.out_cali = UncertaintyAwareGCM(self.pred_len, self.n_var, hd, init, vw, rank)

    def forward(self, pred, m_t):
        return self.out_cali(pred, m_t)

# -------------------------------------------------------------------------
# 3. 主 TTA Runner (Ours)
# -------------------------------------------------------------------------

class TTARunner(nn.Module):
    def __init__(self, cfg, model: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.device = next(self.model.parameters()).device
        
        self.test_loader = get_test_dataloader(cfg)
        
        from models.build import build_norm_module
        self.norm_module = None
        if cfg.NORM_MODULE.ENABLE:
             self.norm_module = build_norm_module(cfg).to(self.device)

        # 1. 诊断器
        self.uncertainty_estimator = UncertaintyEstimator(
            seq_len=cfg.DATA.SEQ_LEN,
            pred_len=cfg.DATA.PRED_LEN,
            n_vars=cfg.DATA.N_VAR
        ).to(self.device)
        
        # 2. 治疗模块
        self.cali = CalibrationModule(cfg).to(self.device)
        
        # 3. 优化器 (初始定义)
        # 注意：在 Adapt 过程中，我们可能需要重置它以避免历史梯度污染
        self.optimizer_cls = torch.optim.Adam
        
        # [修复] 使用 getattr 安全获取。如果配置文件里没写 TTA.LEARNING_RATE，就默认使用 1e-3
        self.lr = getattr(cfg.TTA, 'LEARNING_RATE', 1e-3)
        
        self.threshold = 0.4  
        self.scale_factor = 5.0 
        self.steps = cfg.TTA.PETSA.STEPS
        self.loss_alpha = cfg.TTA.PETSA.LOSS_ALPHA
        self.paas_period_n = cfg.TTA.PETSA.PERIOD_N
        
        self.mse_all = []
        self.mae_all = []
        self.n_adapt_triggered = 0 
        self.person_cor = CorrCoefLoss()

    def train_uncertainty_estimator(self, epochs=10, lr=1e-4):
        # ... (保持原有的训练代码不变) ...
        ckpt_name = f"UE_{self.cfg.MODEL.NAME}_{self.cfg.DATA.NAME}_sl{self.cfg.DATA.SEQ_LEN}_pl{self.cfg.DATA.PRED_LEN}.pth"
        save_dir = self.cfg.TRAIN.CHECKPOINT_DIR
        mkdir(save_dir)
        ue_ckpt_path = os.path.join(save_dir, ckpt_name)
        
        if os.path.exists(ue_ckpt_path):
            print(f"[Ours] Loading pretrained Uncertainty Estimator from: {ue_ckpt_path}")
            self.uncertainty_estimator.load_state_dict(torch.load(ue_ckpt_path))
            return

        print(f"\n[Ours] Training Uncertainty Estimator (Diagnosis Module)...")
        train_loader = get_train_dataloader(self.cfg)
        ue_optimizer = torch.optim.Adam(self.uncertainty_estimator.parameters(), lr=lr)
        criterion = nn.MSELoss() 
        
        self.model.eval() 
        self.uncertainty_estimator.train()
        
        start_time = time.time()
        for epoch in range(epochs):
            total_loss = []
            for i, inputs in enumerate(train_loader):
                enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
                with torch.no_grad():
                    pred_base, ground_truth = forecast(
                        self.cfg, (enc_window, enc_window_stamp, dec_window, dec_window_stamp), 
                        self.model, self.norm_module
                    )
                    real_mse = F.mse_loss(pred_base, ground_truth, reduction='none').mean(dim=(1, 2), keepdim=True)
                
                estimated_mse = self.uncertainty_estimator(enc_window, pred_base)
                loss = criterion(estimated_mse, real_mse)
                
                ue_optimizer.zero_grad()
                loss.backward()
                ue_optimizer.step()
                total_loss.append(loss.item())
            
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"       Epoch {epoch+1}/{epochs} | UE Training Loss: {sum(total_loss)/len(total_loss):.6f}")
        
        torch.save(self.uncertainty_estimator.state_dict(), ue_ckpt_path)
        self.uncertainty_estimator.eval()

    def _get_uncertainty_weight(self, uncertainty_score):
        return torch.sigmoid(self.scale_factor * (uncertainty_score - self.threshold))

    def _calculate_period(self, x_enc):
        fft = torch.fft.rfft(x_enc - x_enc.mean(dim=0), dim=0)
        power = fft.abs().pow(2).mean(dim=0)
        try:
            freq_idx = torch.argmax(power[1:]).item() + 1
            period = x_enc.shape[0] // freq_idx
        except:
            period = 24
        return period * self.paas_period_n

    @torch.enable_grad()
    def adapt(self):
        self.train_uncertainty_estimator(epochs=10, lr=1e-4)
        
        print("\n[Ours] Starting UP-TTA Test Phase...")
        self.model.eval() 
        self.uncertainty_estimator.eval()
        
        data_manager = TTADataManager(self.cfg)
        
        # 临时优化器 (Episodic TTA 推荐做法: 每个 batch 重置优化器状态)
        # 如果你想做 Continual TTA (参数持续累积)，请把这行移到 loop 外面
        # 这里为了保险起见，我们对参数进行持续更新，但可以在这里调整
        optimizer = self.optimizer_cls(self.cali.parameters(), lr=self.lr)

        for i, inputs in enumerate(self.test_loader):
            enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
            
            # --- Step 1: 诊断 ---
            with torch.no_grad():
                pred_base, ground_truth = forecast(
                    self.cfg, (enc_window, enc_window_stamp, dec_window, dec_window_stamp), 
                    self.model, self.norm_module
                )
                uncertainty_score = self.uncertainty_estimator(enc_window, pred_base)
                batch_uncertainty = uncertainty_score.mean().item()
            
            # [Debug] 打印前几个 batch 的不确定性，帮你确定 threshold 是否合理
            if i < 5:
                print(f"    Batch {i} Uncertainty Score: {batch_uncertainty:.4f} (Threshold: {self.threshold})")

            # --- Step 2: 决策 ---
            should_adapt = batch_uncertainty > self.threshold
           
            m_t = self._get_uncertainty_weight(uncertainty_score).detach() 
            
            # 默认 final_pred
            final_pred = pred_base

            # --- Step 3 & 4: 治疗 ---
            if should_adapt:
                self.n_adapt_triggered += 1
                
                # 计算观测长度
                if self.cfg.TTA.PETSA.PAAS:
                    period_len = self._calculate_period(enc_window[0])
                    obs_len = min(period_len, self.cfg.DATA.PRED_LEN - 1)
                    obs_len = max(obs_len, 4) 
                else:
                    obs_len = self.cfg.DATA.PRED_LEN // 2
                
                # 优化循环
                for step in range(self.steps):
                    self.cali.train()
                    
                    # 每次前向都需要传入 m_t
                    pred_adapted = self.cali(pred_base.detach(), m_t)
                    
                    # 切片计算 Loss
                    pred_obs = pred_adapted[:, :obs_len, :]
                    gt_obs = ground_truth[:, :obs_len, :]
                    
                    loss_reg = F.huber_loss(pred_obs, gt_obs, delta=0.5)
                    loss_freq = (torch.fft.rfft(pred_obs, dim=1) - torch.fft.rfft(gt_obs, dim=1)).abs().mean()
                    loss_corr = self.person_cor(pred_obs, gt_obs)
                    loss_mean = F.l1_loss(pred_obs.mean(dim=1), gt_obs.mean(dim=1))
                    
                    loss = loss_reg + self.loss_alpha * loss_freq + loss_corr + loss_mean
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # 生成最终修正结果
                self.cali.eval()
                with torch.no_grad():
                    # 注意：这里也需要传入 m_t
                    final_pred = self.cali(pred_base, m_t)
            
            # --- 记录 ---
            mse = F.mse_loss(final_pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
            mae = F.l1_loss(final_pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
            
            self.mse_all.append(mse)
            self.mae_all.append(mae)
            
            data_manager.collect(
                base_pred=pred_base,
                tta_pred=final_pred,
                gt=ground_truth,
                mse=mse
            )
            
            if (i+1) % 100 == 0:
                print(f"       Processed {i+1} test batches. Adapt Rate: {self.n_adapt_triggered/(i+1):.2%}")

        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        
        print(f"\n=== UP-TTA (Ours) Results ===")
        print(f"Dataset: {self.cfg.DATA.NAME}, Pred Len: {self.cfg.DATA.PRED_LEN}")
        print(f"Adaptation Trigger Rate: {self.n_adapt_triggered}/{len(self.test_loader)} ({self.n_adapt_triggered/len(self.test_loader)*100:.1f}%)")
        print(f"Final Test MSE: {self.mse_all.mean():.4f}")
        print(f"Final Test MAE: {self.mae_all.mean():.4f}")
        
        save_tta_results(
            tta_method=f"Ours-UP-PETSA",
            seed=self.cfg.SEED,
            model_name=self.cfg.MODEL.NAME,
            dataset_name=self.cfg.DATA.NAME,
            pred_len=self.cfg.DATA.PRED_LEN,
            mse_after_tta=self.mse_all.mean(),
            mae_after_tta=self.mae_all.mean(),
        )

def build_tta_runner(cfg, model):
    return TTARunner(cfg, model)