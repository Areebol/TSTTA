import os
import csv
import torch

def synchronize_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.synchronize()

def reset_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.reset_peak_memory_stats()

def get_peak_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.npu.max_memory_allocated() / (1024 * 1024)
    return 0.0

# 【核心修改】：将 start_time/end_time 改为直接接收 duration 和 peak_memory_mb
def record_performance(cfg, adapter, duration, n_samples, peak_memory_mb):
    """
    统一的性能记录函数 (精准版)
    """
    throughput = n_samples / duration if duration > 0 else 0.0

    base_params = 0
    if hasattr(adapter, 'model') and adapter.model is not None:
        base_params = sum(p.numel() for p in adapter.model.parameters())
    
    trainable_params = 0
    if hasattr(adapter, 'optimizer') and adapter.optimizer is not None:
        for param_group in adapter.optimizer.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    trainable_params += p.numel()
    else:
        trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)

    print("\n" + "="*20 + " PERFORMANCE ANALYSIS " + "="*20)
    print(f"Method:           {cfg.TTA.METHOD}")
    print(f"Dataset:          {cfg.DATA.NAME}")
    print(f"Pred_Len:         {cfg.DATA.PRED_LEN}")
    print(f"Base Params:      {base_params / 1e6:.4f} M")
    print(f"Trainable Params: {trainable_params / 1e3:.4f} K  ({(trainable_params * 4) / (1024*1024):.4f} MB)")
    print(f"Total Samples:    {n_samples}")
    print(f"Pure Compute Time:{duration:.4f} s")
    print(f"Throughput:       {throughput:.2f} samples/s")
    print(f"Peak Memory:      {peak_memory_mb:.2f} MB")
    print("="*62 + "\n")

    if not os.path.exists(cfg.RESULT_DIR):
        os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    
    csv_path = os.path.join(cfg.RESULT_DIR, 'performance_benchmark_2.csv')
    file_exists = os.path.isfile(csv_path)
    
    headers =[
               'Model', 'Method', 'Dataset', 'Pred_Len', 
               'Base Params (M)', 'Trainable Params (K)', 
               'Peak Memory (MB)', 'Total Samples', 
               'Pure Time (s)', 'Throughput (samples/s)']
    
    row =[
        cfg.MODEL.NAME, cfg.TTA.METHOD, cfg.DATA.NAME, cfg.DATA.PRED_LEN,
        f"{base_params / 1e6:.4f}", f"{trainable_params / 1e3:.4f}",
        f"{peak_memory_mb:.2f}", n_samples, f"{duration:.4f}", f"{throughput:.2f}"
    ]
    
    try:
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(headers)
            writer.writerow(row)
    except Exception as e:
        print(f"Error saving CSV: {e}")