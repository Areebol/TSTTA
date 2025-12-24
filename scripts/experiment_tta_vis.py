import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

# ==========================================
# 1. 路径修复
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# ==========================================
# 2. 导入依赖
# ==========================================
from datasets.loader import get_test_dataloader
from utils.misc import prepare_inputs, set_devices, set_seeds
from config import get_cfg_defaults
from models.build import build_model, load_best_model, build_norm_module
import models.forecast # 导入模块以便进行 Monkey Patch

# 尝试导入 TAFAS
try:
    from tta.tafas import build_adapter
except ImportError:
    print("Error: Could not import 'build_adapter' from 'tta.tafas'.")
    sys.exit(1)

# =============================================================================
# 【核心修复】自定义 forecast 函数
# =============================================================================
def local_forecast(cfg, inputs, model, norm_module=None):
    # 1. 解包数据
    enc_window, enc_window_stamp, dec_window, dec_window_stamp = inputs
    
    # 2. 归一化 (如果启用 RevIN)
    if norm_module is not None:
        enc_window = norm_module(enc_window, mode='norm')
    
    # [调试] 打印进入模型前的形状
    # 注意：现在由于我们把 seq_len 改为了 96，这里应该打印出 shape=(B, 96, C)
    try:
        pass 
        # print(f"LOCAL_FORECAST DEBUG: enc_window.shape={enc_window.shape}")
    except Exception:
        pass

    # 3. 模型前向传播
    pred = model(enc_window, enc_window_stamp, dec_window, dec_window_stamp)
    
    # 4. 反归一化
    if norm_module is not None:
        pred = norm_module(pred, mode='denorm')
        
    # 5. 提取 Ground Truth
    ground_truth = dec_window[:, -cfg.DATA.PRED_LEN:, :]
    
    return pred, ground_truth

# 【Monkey Patch】用我们的修复版替换掉系统中的 forecast
models.forecast.forecast = local_forecast
print(">>> Successfully patched models.forecast.forecast with local_forecast.")

# ==========================================
# 实验逻辑
# ==========================================

def get_baseline_mse(cfg, model, norm_module):
    print(">>> Running Baseline Inference (Pre-TTA)...")
    loader = get_test_dataloader(cfg)
    mse_list = []
    
    model.eval()
    if norm_module:
        norm_module.eval()
        
    with torch.no_grad():
        for i, inputs in enumerate(loader):
            enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
            pass_inputs = (enc_window, enc_window_stamp, dec_window, dec_window_stamp)
            
            # 这里调用的是已经被我们替换过的 forecast
            pred, ground_truth = models.forecast.forecast(cfg, pass_inputs, model, norm_module)
            
            batch_mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1))
            mse_list.append(batch_mse.cpu().numpy())
            
    return np.concatenate(mse_list)

def visualize_results(mse_pre, mse_post, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    diff = mse_post - mse_pre
    worse_indices = diff > 0
    better_indices = diff <= 0
    
    n_worse = np.sum(worse_indices)
    n_total = len(mse_pre)
    ratio_worse = n_worse / n_total * 100 if n_total > 0 else 0
    
    print("\n" + "="*40)
    print(f"Experimental Analysis Summary (TAFAS)")
    print(f"="*40)
    print(f"Total Samples: {n_total}")
    print(f"Samples Improved: {n_total - n_worse}")
    print(f"Samples Degraded: {n_worse} ({ratio_worse:.2f}%)")
    print(f"Avg MSE Pre:  {mse_pre.mean():.4f}")
    print(f"Avg MSE Post: {mse_post.mean():.4f}")
    print(f"="*40)

    sns.set_theme(style="whitegrid")
    
    # Scatter Plot
    plt.figure(figsize=(10, 8))
    max_val = max(mse_pre.max(), mse_post.max()) if len(mse_pre) > 0 else 1.0
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='No Change')
    if np.any(better_indices):
        plt.scatter(mse_pre[better_indices], mse_post[better_indices], c='green', alpha=0.5, s=10, label='Improved')
    if np.any(worse_indices):
        plt.scatter(mse_pre[worse_indices], mse_post[worse_indices], c='red', alpha=0.5, s=10, label='Degraded')
    plt.xlabel('Original MSE')
    plt.ylabel('Adapted MSE')
    plt.title(f'TAFAS Effect\n({ratio_worse:.2f}% degraded)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scatter_pre_vs_post.png'), dpi=300)
    plt.close()
    
    # Change Dist
    plt.figure(figsize=(10, 6))
    plt.scatter(mse_pre, diff, c=diff>0, cmap='coolwarm_r', alpha=0.6, s=15)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Original MSE')
    plt.ylabel('MSE Change (Post - Pre)')
    plt.title('MSE Change vs. Original MSE')
    plt.colorbar(label='Degraded (Red) / Improved (Blue)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mse_change_distribution.png'), dpi=300)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ETTh1')
    parser.add_argument('--pred_len', type=int, default=96)
    
    # 【修改点 1】将默认 seq_len 从 336 改为 96
    # 这样 DLinear 初始化为 [96, 96]，就能匹配你的 Checkpoint 了
    parser.add_argument('--seq_len', type=int, default=96) 
    
    parser.add_argument('--model', type=str, default='DLinear')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/DLinear/ETTh1_96')
    parser.add_argument('--save-dir', type=str, default=os.path.join(project_root, 'figs_tafas_vis'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gating_init', type=float, default=0.01)
    args = parser.parse_args()

    print(f"Configuring: Model={args.model}, Dataset={args.dataset}, Seq={args.seq_len}, Pred={args.pred_len}")

    cfg = get_cfg_defaults()
    
    # 强制设置参数，防止被其他逻辑覆盖
    cfg.DATA.NAME = args.dataset
    cfg.DATA.PRED_LEN = args.pred_len
    cfg.DATA.SEQ_LEN = args.seq_len
    # 覆盖可能的别名，确保所有指向 seq_len 的变量都是 96
    if hasattr(cfg.DATA, 'IN_LEN'): cfg.DATA.IN_LEN = args.seq_len
    if hasattr(cfg.MODEL, 'IN_LEN'): cfg.MODEL.IN_LEN = args.seq_len
        
    cfg.MODEL.NAME = args.model
    # Ensure model-level seq_len / pred_len are consistent with DATA settings
    if hasattr(cfg.MODEL, 'PRED_LEN'): cfg.MODEL.PRED_LEN = args.pred_len
    
    # DLinear 可能读取的特定属性
    if hasattr(cfg.MODEL, 'seq_len'):
        cfg.MODEL.seq_len = args.seq_len
    if hasattr(cfg.MODEL, 'pred_len'):
        cfg.MODEL.pred_len = args.pred_len

    if not os.path.isabs(args.checkpoint_dir):
         cfg.TRAIN.CHECKPOINT_DIR = os.path.join(project_root, args.checkpoint_dir)
    else:
         cfg.TRAIN.CHECKPOINT_DIR = args.checkpoint_dir
    cfg.SEED = args.seed

    # 数据集变量维度
    if args.dataset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
        cfg.DATA.N_VAR = 7
        cfg.DATA.FEATURES = 'M' 
    elif args.dataset == 'weather':
        cfg.DATA.N_VAR = 21
        cfg.DATA.FEATURES = 'M'
    elif args.dataset == 'exchange_rate':
        cfg.DATA.N_VAR = 8
        cfg.DATA.FEATURES = 'M'

    # TAFAS 参数
    cfg.TTA.ENABLE = True
    cfg.TTA.METHOD = 'TAFAS'
    cfg.TTA.SOLVER.BASE_LR = args.lr
    cfg.TTA.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.TTA.TAFAS.GATING_INIT = args.gating_init
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = False

    set_devices(cfg.VISIBLE_DEVICES)
    set_seeds(cfg.SEED)

    # 构建模型
    print(f">>> Building model with seq_len={cfg.DATA.SEQ_LEN} (Expected matching Checkpoint)")
    model = build_model(cfg)
    
    # 【修改点 2】更稳健的模型加载逻辑
    try:
        model = load_best_model(cfg, model)
    except Exception as e:
        error_msg = str(e)
        if "size mismatch" in error_msg and "Linear_Seasonal.weight" in error_msg:
            print("\n" + "!"*60)
            print("【配置错误】模型形状与 Checkpoint 不匹配！")
            print(f"当前设置 seq_len: {args.seq_len}")
            print(f"错误详情: {e}")
            print("原因: Checkpoint 可能是用不同的 seq_len (例如 96) 训练的，但你设置了其他值(例如 336)。")
            print("建议: 检查 parser.add_argument('--seq_len') 的值是否与训练时的设置一致。")
            print("!"*60 + "\n")
            sys.exit(1)
        else:
            print(f"Error loading model: {e}")
            sys.exit(1)

    norm_module = None
    if cfg.NORM_MODULE.ENABLE:
        norm_module = build_norm_module(cfg)

    print("Building TAFAS Adapter...")
    adapter = build_adapter(cfg, model, norm_module)

    # 运行 Baseline
    mse_pre = get_baseline_mse(cfg, adapter.model, getattr(adapter, 'norm_module', None))

    print("\n>>> Running TAFAS Adaptation...")
    adapter.adapt() 
    
    if isinstance(adapter.mse_all, list):
        mse_post = np.concatenate(adapter.mse_all)
    else:
        mse_post = adapter.mse_all

    if len(mse_pre) != len(mse_post):
        min_len = min(len(mse_pre), len(mse_post))
        mse_pre = mse_pre[:min_len]
        mse_post = mse_post[:min_len]

    visualize_results(mse_pre, mse_post, save_dir=args.save_dir)
    print(f"\nResults saved to {args.save_dir}")