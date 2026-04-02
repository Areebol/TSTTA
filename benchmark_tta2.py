import os
import time
import csv
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from unittest.mock import patch, MagicMock

# 导入项目配置和工具
from config import get_cfg_defaults
from device_manager import global_device
from utils.misc import set_seeds
from models.build import build_model
from datasets.build import build_dataset

# 导入 TTA 方法
import tta.tafas as tafas
import tta.petsa as petsa
import tta.dynatta as dynatta
import tta.coba as coba

# ==========================================
# 1. 内存级 DataLoader (消除 I/O 瓶颈，解决 Device 冲突)
# ==========================================
class InMemoryLoader:
    """
    我们将数据留在 CPU 上。
    因为各 TTA Adapter 在初始化 (__init__) 期间可能会触发试跑，此时模型可能还在 CPU 上。
    原版的 prepare_inputs 会自动把 CPU 数据 .to(device)，完美符合原生生态。
    """
    def __init__(self, dataset, num_samples_limit=2000):
        self.dataset = dataset
        self.batches =[]
        limit = min(len(dataset), num_samples_limit)
        print(f"[*] Pre-loading {limit} samples into CPU Memory...")
        
        x_list, y_list, x_mark_list, y_mark_list = [], [], [],[]
        for i in range(limit):
            x, y, x_mark, y_mark = dataset[i]
            x_list.append(torch.from_numpy(x) if isinstance(x, np.ndarray) else x)
            y_list.append(torch.from_numpy(y) if isinstance(y, np.ndarray) else y)
            x_mark_list.append(torch.from_numpy(x_mark) if isinstance(x_mark, np.ndarray) else x_mark)
            y_mark_list.append(torch.from_numpy(y_mark) if isinstance(y_mark, np.ndarray) else y_mark)
            
        # 保持在 CPU，让框架原生的 prepare_inputs 去搬运
        self.batch_data = (
            torch.stack(x_list).float(),
            torch.stack(x_mark_list).float(),
            torch.stack(y_list).float(),
            torch.stack(y_mark_list).float()
        )
        self.total_samples = limit

    def __iter__(self):
        yield self.batch_data

    def __len__(self):
        return 1

# ==========================================
# 2. 增强型冻结工具 (冻结参数 + 清理优化器)
# ==========================================
def freeze_and_clean(adapter):
    for param in adapter.model.parameters():
        param.requires_grad = False
    
    active_params =[]
    if hasattr(adapter, 'cali') and adapter.cali is not None:
        for name, param in adapter.cali.named_parameters():
            if any(k in name for k in['online', 'gating', 'static_g', 'mlp', 'freq_r', 'freq_i']):
                param.requires_grad = True
                active_params.append(param)
            else:
                param.requires_grad = False
    else:
        for param in adapter.parameters():
            if param.requires_grad: active_params.append(param)
            
    if active_params:
        adapter.optimizer = torch.optim.Adam(active_params, lr=1e-3)

# ==========================================
# 3. 核心测试逻辑
# ==========================================
def run_benchmark():
    set_seeds(1)
    
    MODEL_NAME = "DLinear"
    DATASET_NAME = "ETTh1"
    PRED_LENS =[24, 48, 96, 192]
    NUM_SAMPLES = 2000 
    
    RESULT_FILE = f"./results/real_benchmark/{MODEL_NAME}_{DATASET_NAME}_performance.csv"
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    
    with open(RESULT_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Method', 'Pred_Len', 'Add_Params(K)', 'Peak_Memory(MB)', 'Throughput(samples/s)'])

    methods = {
        "TAFAS": tafas.build_adapter,
        "PETSA": petsa.build_adapter,
        "DynaTTA": dynatta.build_adapter,
        "COBA": coba.build_adapter
    }

    print(f"\n🚀 STARTING ACADEMIC BENCHMARK: {MODEL_NAME} on {DATASET_NAME}")

    for p_len in PRED_LENS:
        print(f"\n" + "="*30 + f" Pred_Len: {p_len} " + "="*30)
        
        cfg = get_cfg_defaults()
        cfg.defrost()
        cfg.DATA.NAME, cfg.MODEL.NAME = DATASET_NAME, MODEL_NAME
        cfg.DATA.SEQ_LEN = cfg.MODEL.seq_len = 96
        cfg.DATA.PRED_LEN = cfg.MODEL.pred_len = p_len
        cfg.DATA.N_VAR = 7
        cfg.MODEL.enc_in = cfg.MODEL.dec_in = cfg.MODEL.c_out = 7 
        cfg.TRAIN.BATCH_SIZE = 32
        cfg.TTA.ENABLE = True
        cfg.TTA.DOMAIN_SHIFT = False
        
        cfg.TTA.DUAL.CALI_NAME = 'CoBA_TF_Adapter' 
        cfg.TTA.DUAL.LOSS_NAME = 'CoBA_Loss'
        cfg.TTA.DUAL.COBA_ONLINE_ENABLED = True
        cfg.TTA.DUAL.GCM_N_BASES = 32
        cfg.TTA.DUAL.PRETRAIN_EPOCHS = 1
        cfg.TTA.DUAL.CALI_INPUT_ENABLE = False
        cfg.TTA.DUAL.CALI_OUTPUT_ENABLE = True
        # 【关键修复】：绝对不要加 cfg.freeze()，让 DynaTTA 畅通无阻
        
        # 构建真实 Dataset
        real_dataset = build_dataset(cfg, "test")
        real_train_dataset = build_dataset(cfg, "train")
        
        in_mem_test = InMemoryLoader(real_dataset, num_samples_limit=NUM_SAMPLES)
        in_mem_train = InMemoryLoader(real_train_dataset, num_samples_limit=32) 
        
        # 我们把基座模型提前丢进 NPU
        base_model = build_model(cfg).to(global_device)
        original_model_state = deepcopy(base_model.state_dict())

        # 【恢复原生 prepare_inputs】
        # 因为 InMemoryLoader 把数据存在了 CPU 上，我们需要原生函数把它搬运到 NPU。
        # 原生搬运耗时极短 (几毫秒)，不会影响最终吞吐量级别的测量。
        with patch('datasets.loader.get_test_dataloader', return_value=in_mem_test), \
             patch('datasets.loader.get_tta_train_dataloader', return_value=in_mem_train), \
             patch('tta.utils.save_tta_results', MagicMock()), \
             patch('tta.tta_dual_utils.performance.record_performance', MagicMock()):

            for method_name, build_fn in methods.items():
                print(f"  -> Benchmarking: {method_name}")
                cfg.TTA.METHOD = method_name
                
                base_model.load_state_dict(deepcopy(original_model_state))
                
                try:
                    # 实例化 Adapter
                    adapter = build_fn(cfg, base_model)
                    
                    # 确保 Adapter 及其中动态生成的层（如 Cali）全都上了 NPU
                    adapter = adapter.to(global_device)
                    
                    adapter.is_eved_like = False 
                    freeze_and_clean(adapter)
                    
                    # NPU 状态重置
                    if hasattr(torch, 'npu') and torch.npu.is_available():
                        torch.npu.synchronize()
                        torch.npu.reset_peak_memory_stats()
                        torch.npu.empty_cache()
                        
                    start_t = time.time()
                    
                    # 开始自适应
                    adapter.adapt() 
                    
                    if hasattr(torch, 'npu') and torch.npu.is_available():
                        torch.npu.synchronize()
                        peak_mem = torch.npu.max_memory_allocated() / (1024 * 1024)
                    else: 
                        peak_mem = 0.0
                        
                    end_t = time.time()
                    
                    # 计算指标
                    duration = end_t - start_t
                    throughput = in_mem_test.total_samples / duration
                    params_k = sum(p.numel() for p in adapter.parameters() if p.requires_grad) / 1e3
                    
                    print(f"     [OK] Speed: {throughput:.2f} samples/s | Mem: {peak_mem:.2f} MB | Params: {params_k:.2f} K")
                    
                    with open(RESULT_FILE, mode='a', newline='') as f:
                        csv.writer(f).writerow([MODEL_NAME, method_name, p_len, f"{params_k:.2f}", f"{peak_mem:.2f}", f"{throughput:.2f}"])
                
                except Exception as e:
                    print(f"     [ERR] {method_name} failed:")
                    import traceback; traceback.print_exc()

                if hasattr(torch, 'npu') and torch.npu.is_available():
                    torch.npu.empty_cache()
                time.sleep(0.5)

    print(f"\n✅ All Finished! Data saved to {RESULT_FILE}")

if __name__ == "__main__":
    run_benchmark()