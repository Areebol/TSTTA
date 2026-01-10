import torch
import os

root = "./checkpoints"

def convert_pth(old_path):
    new_path = old_path
    print(f"[Convert] {old_path} -> {new_path}")
    try:
        state = torch.load(old_path, map_location="cpu")
        torch.save(state, new_path)
    except Exception as e:
        print(f"[Error] Failed to convert {old_path}: {e}")

for dirpath, dirnames, filenames in os.walk(root):
    for f in filenames:
        if f.endswith(".pth") or f.endswith(".pt"):
            old_path = os.path.join(dirpath, f)
            convert_pth(old_path)
