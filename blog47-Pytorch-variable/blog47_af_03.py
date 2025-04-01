import torch
import torch.nn.functional as F

x = torch.tensor([1.0, -1.0, 0.0])
relu_output = F.relu(x)
print(relu_output)  # 输出：[1.0, 0.0, 0.0]
