import torch
import torch.nn.functional as F

x = torch.tensor([1.0, -1.0, 0.0])
sigmoid_output = torch.sigmoid(x)
print(sigmoid_output)  # 输出：[0.7311, 0.2689, 0.5000]
