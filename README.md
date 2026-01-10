data:
data download: https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2

NPU env:
```bash
conda env create -f npu_environment.yml -n tstta-npu
# official tutorial ：https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0004.html
# available:
# Pytorch=2.7.1 ~ Torch-NPU=2.7.1 / Python=3.10
wget https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl
pip3 install torch-2.7.1+cpu-cp310-cp310-manylinux_2_28_aarch64.whl

wget https://gitcode.com/Ascend/pytorch/releases/download/v7.2.0-pytorch2.7.1/torch_npu-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
pip3 install torch_npu-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl

# test npu envs
python3 -c "import torch;import torch_npu; a = torch.randn(3, 4).npu(); print(a + a);"
```