import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 导入项目配置和工具
from config import get_cfg_defaults
from device_manager import global_device
from utils.misc import prepare_inputs, set_seeds

# 导入所有的 TTA Adapter
import tta.tafas as tafas
import tta.petsa as petsa
import tta.dynatta as dynatta
import tta.tta_dual as tta_dual

# ==========================================
# 1. 构建极轻量级的虚拟基座模型 (Dummy Model)
# ==========================================
class DummyBaseModel(nn.Module):
    """
    一个极简的线性映射模型。
    它只用一个没有任何隐藏层的 nn.Linear 来对齐输入和输出的形状。
    基座计算量和显存占用近乎为 0。
    """
    def __init__(self, seq_len, pred_len):
        super(DummyBaseModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc 形状:[Batch, seq_len, n_vars]
        x = x_enc.transpose(1, 2)       #[Batch, n_vars, seq_len]
        x = self.linear(x)              # [Batch, n_vars, pred_len]
        return x.transpose(1, 2)        # [Batch, pred_len, n_vars]


# ==========================================
# 2. 构建万级规模的虚拟数据集 (Dummy Dataset)
# ==========================================
class DummyDataset(Dataset):
    def __init__(self, num_samples, seq_len, pred_len, n_vars):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        
        # 预先生成随机数据以节省 DataLoader 运行时的 CPU 开销
        print(f"Generating {num_samples} dummy samples in memory...")
        self.data_x = torch.randn(num_samples, seq_len, n_vars)
        self.data_y = torch.randn(num_samples, pred_len, n_vars)
        
        # 模拟时间戳协变量 (假设通常有 4 个维度)
        self.mark_x = torch.randn(num_samples, seq_len, 4)
        self.mark_y = torch.randn(num_samples, pred_len, 4)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_x[idx], self.mark_x[idx], self.data_y[idx], self.mark_y[idx]

    # 为了兼容部分 Adapter 的写法，加上 mock 属性
    @property
    def test(self):
        return self


# ==========================================
# 3. 主测试逻辑
# ==========================================
def run_benchmark():
    set_seeds(0)
    
    # --- 实验核心参数设置 ---
    NUM_SAMPLES = 10000    # 一万条测试数据
    SEQ_LEN = 192          # 历史长度
    PRED_LEN = 192         # 预测长度
    N_VARS = 20            # 变量数 (模拟 eVED)
    BASE_LR = 1e-3         # TTA 学习率

    # 获取默认配置并覆写我们要测试的参数
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.DATA.NAME = "DummyData"
    cfg.DATA.SEQ_LEN = SEQ_LEN
    cfg.DATA.PRED_LEN = PRED_LEN
    cfg.DATA.N_VAR = N_VARS
    cfg.MODEL.NAME = "DummyModel"
    cfg.MODEL.enc_in = N_VARS
    cfg.MODEL.c_out = 2   # 假设目标只预测前 2 个特征
    
    # 统一 TTA 参数
    cfg.TTA.ENABLE = True
    cfg.TTA.DOMAIN_SHIFT = False
    cfg.TTA.SOLVER.BASE_LR = BASE_LR
    
    # 保存结果的专属目录
    cfg.RESULT_DIR = "./results/pure_tta_benchmark/"
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    cfg.freeze()

    # 初始化虚拟数据集 (一次性生成 10000 条，BatchSize 设为总长度以适配原版代码的加载逻辑)
    dummy_dataset = DummyDataset(NUM_SAMPLES, SEQ_LEN, PRED_LEN, N_VARS)
    dummy_loader = DataLoader(dummy_dataset, batch_size=NUM_SAMPLES, shuffle=False)

    # 我们要评测的四种 TTA 算法
    tta_methods = {
        "TAFAS": tafas.build_adapter,
        "PETSA": petsa.build_adapter,
        "DynaTTA": dynatta.build_adapter,
        "Dual-TTA": tta_dual.build_adapter
    }

    print("\n" + "*"*50)
    print(f"STARTING PURE TTA BENCHMARK ON NPU")
    print(f"   Samples: {NUM_SAMPLES} | Length: {PRED_LEN} | Vars: {N_VARS}")
    print("*"*50 + "\n")

    for method_name, build_fn in tta_methods.items():
        print(f"\n---> Benchmarking: {method_name}")
        
        # 针对具体方法修改 cfg
        cfg.defrost()
        cfg.TTA.METHOD = method_name
        if method_name == "Dual-TTA":
            cfg.TTA.DUAL.CALI_NAME = 'RoCoBA_FreqDomain_Norm'
            cfg.TTA.DUAL.QUERY_TYPE = 'freq-base-CI'
        cfg.freeze()

        # 初始化极其轻量的 Dummy 模型
        dummy_model = DummyBaseModel(SEQ_LEN, PRED_LEN).to(global_device)

        # 构建 TTA Adapter
        adapter = build_fn(cfg, dummy_model, norm_module=None)
        
        # ！！！核心注入：强行替换掉 Adapter 内部的 loader ！！！
        adapter.test_loader = dummy_loader
        adapter.test_data = dummy_dataset
        # 强制走 Regular adaptation，避开 EVED 复杂的按文件读取逻辑
        adapter.is_eved_like = False 
        
        # 运行纯 TTA 适应
        # 注：你的 Adapter 内部自带了 record_performance 探针，它会自动计入 CSV
        try:
            adapter.adapt()
        except Exception as e:
            print(f"❌ Error while running {method_name}: {e}")
            import traceback
            traceback.print_exc()
        
        # 为防止上一个方法的显存缓存干扰下一个方法，强制清空
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.empty_cache()
            
        time.sleep(1) # 短暂休眠确保资源释放

    print("\n✅ Benchmark Complete! Check the CSV file in ./results/pure_tta_benchmark/")

if __name__ == "__main__":
    run_benchmark()