import os
import csv
import time
import torch

def synchronize_device():
    """NPU/GPU 设备同步"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.synchronize()

def record_performance(cfg, adapter, start_time, end_time, n_samples):
    """
    统一的性能记录函数，用于各个 TTA Adapter 内部调用
    """
    duration = end_time - start_time
    throughput = n_samples / duration if duration > 0 else 0.0

    # 1. 细粒度参数统计
    # Base Model (基座)
    base_params = 0
    if hasattr(adapter, 'model') and adapter.model is not None:
        base_params = sum(p.numel() for p in adapter.model.parameters())
    
    # Calibration Module (TTA 专用模块)
    # 优先查找 'cali' (PETSA/TAFAS), 其次查找 'query_net' (Ours), 最后查找 'gcm'
    tta_module_params = 0
    module_name = "Unknown"
    
    if hasattr(adapter, 'cali') and adapter.cali is not None:
        tta_module_params = sum(p.numel() for p in adapter.cali.parameters())
        module_name = "cali"
    elif hasattr(adapter, 'query_net') and adapter.query_net is not None:
        tta_module_params = sum(p.numel() for p in adapter.query_net.parameters())
        module_name = "query_net"
    
    # Trainable Params (实际更新的)
    trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)

    # 2. 打印控制台
    print("\n" + "="*20 + " PERFORMANCE ANALYSIS " + "="*20)
    print(f"Method:           {cfg.TTA.METHOD}")
    print(f"Dataset:          {cfg.DATA.NAME}")
    print(f"Base Params:      {base_params / 1e6:.4f} M")
    print(f"TTA Module ({module_name}): {tta_module_params / 1e3:.4f} K ")
    print(f"Trainable Params: {trainable_params / 1e3:.4f} K ")
    print(f"Throughput:       {throughput:.2f} samples/s")
    print("="*62 + "\n")

    # 3. 写入 CSV
    if not os.path.exists(cfg.RESULT_DIR):
        os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    
    csv_path = os.path.join(cfg.RESULT_DIR, 'performance_benchmark.csv')
    file_exists = os.path.isfile(csv_path)
    
    headers = [
               'Model', 
               'Method', 
               'Dataset', 
               'Pred_Len', 
               'Base Params (M)', 
               'TTA Module Params (K)', 
               'Trainable Params (K)', 
               'Total Samples', 
               'Time (s)', 
               'Throughput (samples/s)']
    
    row = [
        cfg.MODEL.NAME,
        cfg.TTA.METHOD, 
        cfg.DATA.NAME, 
        cfg.DATA.PRED_LEN,
        f"{base_params / 1e6:.4f}",
        f"{tta_module_params / 1e3:.4f}",
        f"{trainable_params / 1e3:.4f}",
        n_samples, 
        f"{duration:.4f}",
        f"{throughput:.2f}"
    ]
    
    try:
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(headers)
            writer.writerow(row)
        print(f"Saved to {csv_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}")