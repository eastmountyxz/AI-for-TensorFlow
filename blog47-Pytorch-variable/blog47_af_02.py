import torch

x = torch.tensor([1.0, -1.0, 0.0])
tanh_output = torch.tanh(x)
print(tanh_output)  # 输出：[0.7616, -0.7616, 0.0000]
