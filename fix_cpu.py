import torch

print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA设备数: {torch.cuda.device_count()}")