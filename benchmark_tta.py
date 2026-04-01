import os
import time
import csv
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 导入项目配置和工具
from config import get_cfg_defaults
from device_manager import global_device
from utils.misc import set_seeds, prepare_inputs
from models.build import build_model
from models.forecast import forecast
import types
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
            x, x_mark, y, y_mark = dataset[i]
            x_list.append(torch.from_numpy(x) if isinstance(x, np.ndarray) else x)
            x_mark_list.append(torch.from_numpy(x_mark) if isinstance(x_mark, np.ndarray) else x_mark)
            y_list.append(torch.from_numpy(y) if isinstance(y, np.ndarray) else y)
            y_mark_list.append(torch.from_numpy(y_mark) if isinstance(y_mark, np.ndarray) else y_mark)
            
        self.batch_data = (
            torch.stack(x_list).float().to(global_device),
            torch.stack(x_mark_list).float().to(global_device),
            torch.stack(y_list).float().to(global_device),
            torch.stack(y_mark_list).float().to(global_device)
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
        if hasattr(adapter.cali.out_cali, 'online_mode'):
            adapter.cali.online_mode = True
            
        if hasattr(adapter.cali.out_cali, 'get_optim_params'):
            for param in adapter.cali.parameters():
                param.requires_grad = False
            for param in adapter.cali.out_cali.get_optim_params():
                param.requires_grad = True
                active_params.append(param)
        else:
            for param in adapter.cali.parameters():
                param.requires_grad = True
                active_params.append(param)
    else:
        for param in adapter.parameters():
            if param.requires_grad: active_params.append(param)
            
    if active_params:
        adapter.optimizer = torch.optim.Adam(active_params, lr=1e-3)

# ==========================================
# 3. 核心测试逻辑
# ==========================================
@torch.enable_grad()
def custom_adapt(self, enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all):
    batch_start = 0
    batch_end = 0
    batch_idx = 0
    self.cur_step = self.cfg.DATA.SEQ_LEN - 2
    total_len = len(enc_window_all)
    
    while batch_end < len(enc_window_all):
        print(f"Processing batch {batch_idx}: {batch_start} to {batch_end}")
        enc_window_first = enc_window_all[batch_start]
        
        print(f"Processing batch {batch_idx}: {batch_start} to {batch_end}")
        print(enc_window_first.shape)
        period, batch_size = self._calculate_period_and_batch_size(enc_window_first)
            
        batch_end = batch_start + batch_size

        if batch_end > len(enc_window_all):
            batch_end = len(enc_window_all)
            batch_size = batch_end - batch_start

        self.cur_step += batch_size

        batch_inputs = (
            enc_window_all[batch_start:batch_end],
            enc_window_stamp_all[batch_start:batch_end],
            dec_window_all[batch_start:batch_end],
            dec_window_stamp_all[batch_start:batch_end]
        )
        
        self.pred_step_end_dict[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN
        self.inputs_dict[batch_idx] = batch_inputs
        
        
        if hasattr(self, '_adapt_full'):
            self._adapt_full(batch_inputs)
        else:
            # Always run the full-ground-truth inference path once per batch.
            self._adapt_with_full_ground_truth_if_available(batch_inputs)
        
        if hasattr(self, '_adapt_partial'):
            pred, ground_truth = self._adapt_partial(batch_inputs, period, batch_size, batch_idx)
        else:
            pred, ground_truth = self._adapt_with_partial_ground_truth(batch_inputs, period, batch_size, batch_idx)
        
        pred, ground_truth = self._adjust_prediction(pred, batch_inputs, batch_size, period)
            
        batch_start = batch_end
        batch_idx += 1


@torch.enable_grad()
def custom_full_ground_truth_inference(self, inputs):
    """
    Pure-inference replacement of _adapt_with_full_ground_truth_if_available.
    This bypasses internal buffer-based checks and runs one forward path every call.
    """
    if hasattr(self, 'switch_model_to_train'):
        self.switch_model_to_train()
    else:
        self._switch_model_to_train()

    if self.cali.input_calibration is not None:
        inputs = self.cali.input_calibration(inputs)

    pred, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)

    if self.cali.output_calibration is not None:
        enc_window = prepare_inputs(inputs)[0]
        try:
            pred = self.cali.output_calibration(pred, enc_window)
        except TypeError:
            pred = self.cali.output_calibration(pred)

    loss = F.mse_loss(pred, ground_truth) 

    self.optimizer.zero_grad()
    loss.backward()
    # self.optimizer.step()
    if hasattr(self, 'switch_model_to_eval'):
        self.switch_model_to_eval()
    else:
        self._switch_model_to_eval()

    return pred, ground_truth

def run_benchmark():
    set_seeds(1)
    
    MODEL_NAME = "DLinear"
    DATASET_NAME = "ETTh1"
    # PRED_LENS =[96, 192, 336, 720]
    PRED_LENS =[96]
    NUM_SAMPLES = 64
    
    RESULT_FILE = f"./results/benchmark_tta/{MODEL_NAME}_{DATASET_NAME}_performance.csv"
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
        cfg.TRAIN.BATCH_SIZE = 64
        cfg.TTA.ENABLE = True
        cfg.TTA.DOMAIN_SHIFT = False
        
        cfg.TTA.DUAL.CALI_NAME = 'CoBA_TF_Adapter' 
        cfg.TTA.DUAL.LOSS_NAME = 'MSE'
        cfg.TTA.DUAL.COBA_ONLINE_ENABLED = True
        cfg.TTA.DUAL.GCM_N_BASES = 32
        cfg.TTA.DUAL.PRETRAIN_EPOCHS = 1
        cfg.TTA.DUAL.CALI_INPUT_ENABLE = False
        cfg.TTA.DUAL.CALI_OUTPUT_ENABLE = True
        
        # 构建真实 Dataset
        real_dataset = build_dataset(cfg, "test")
        # real_train_dataset = build_dataset(cfg, "train")
        
        in_mem_test = InMemoryLoader(real_dataset, num_samples_limit=NUM_SAMPLES)
        # in_mem_train = InMemoryLoader(real_train_dataset, num_samples_limit=NUM_SAMPLES) 
        
        # 我们把基座模型提前丢进 NPU
        base_model = build_model(cfg).to(global_device)
        base_params_num = sum(p.numel() for p in base_model.parameters())
        original_model_state = deepcopy(base_model.state_dict())


        for method_name, build_fn in methods.items():
            print(f"  -> Benchmarking: {method_name}")
            cfg.TTA.METHOD = method_name
            
            base_model.load_state_dict(deepcopy(original_model_state))
            
            try:
                # 实例化 Adapter
                adapter = build_fn(cfg, base_model)
                
                # 确保 Adapter 及其中动态生成的层（如 Cali）全都上了 NPU
                adapter = adapter.to(global_device)
                
                # 替换 adapt 函数为无输出、不保留指标的纯推出版
                adapter.adapt = types.MethodType(custom_adapt, adapter)
                if hasattr(adapter, '_adapt_with_full_ground_truth_if_available'):
                    adapter._adapt_with_full_ground_truth_if_available = types.MethodType(custom_full_ground_truth_inference, adapter)
                
                if hasattr(adapter, '_adapt_full'):
                    adapter._adapt_full = types.MethodType(custom_full_ground_truth_inference, adapter)
                
                adapter.is_eved_like = False 
                # freeze_and_clean(adapter)
                
                # NPU 状态重置
                if hasattr(torch, 'npu') and torch.npu.is_available():
                    torch.npu.synchronize()
                    torch.npu.reset_peak_memory_stats()
                    torch.npu.empty_cache()
                
                # 在计时前提前将数据移至 NPU 并准备好，避免影响推理耗时的统计
                prepared_inputs = prepare_inputs(in_mem_test.batch_data)

                start_t = time.time()
                
                # 直接传处理后的数据，避免内部各种 I/O 操作及数据搬迁
                adapter.adapt(*prepared_inputs)
                
                if hasattr(torch, 'npu') and torch.npu.is_available():
                    torch.npu.synchronize()
                    peak_mem = torch.npu.max_memory_allocated() / (1024 * 1024)
                else: 
                    peak_mem = 0.0
                    
                end_t = time.time()
                
                # 计算指标
                duration = end_t - start_t
                throughput = in_mem_test.total_samples / duration
                
                # 获取整个 adapter 的参数量，并计算相对于基座增加的参数
                adapter_total = sum(p.numel() for p in adapter.parameters())
                if adapter_total >= base_params_num:
                    params_k = (adapter_total - base_params_num) / 1e3
                else:
                    params_k = adapter_total / 1e3
                
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