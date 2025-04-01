import torch

x = torch.tensor([2.0, 1.0, 0.1])
softmax_output = torch.nn.functional.softmax(x, dim=0)
print(softmax_output)  # 输出：[0.659, 0.242, 0.099]
