import torch

leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
x = torch.tensor([1.0, -1.0, 0.0])
leaky_relu_output = leaky_relu(x)
print(leaky_relu_output)  # 输出：[1.0, -0.01, 0.0]
