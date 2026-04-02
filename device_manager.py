import torch

def get_device():
    try:
        import torch_npu
    except ImportError:
        pass
        
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.device("npu")
    else:
        return torch.device("cpu")
    
global_device = get_device()
# global_device = "cpu"