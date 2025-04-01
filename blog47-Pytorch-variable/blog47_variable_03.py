import torch

# 创建一个 3x3 的张量
tensor_1 = torch.tensor([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]],
                        dtype=torch.float32)

# 生成随机张量 3x3维度服从标准正态分布
tensor_2 = torch.randn(3, 3)
print(tensor_1)
print(tensor_2)
