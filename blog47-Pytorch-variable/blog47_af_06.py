import torch
import matplotlib.pyplot as plt

# 定义输入张量（范围：-5 到 5）
x = torch.linspace(-5, 5, 100)

# 计算激励函数值
sigmoid_y = torch.sigmoid(x)  # Sigmoid
tanh_y = torch.tanh(x)        # Tanh
relu_y = torch.relu(x)        # ReLU
leaky_relu_y = torch.nn.functional.leaky_relu(x, negative_slope=0.1)  # Leaky ReLU

# 创建子图
plt.figure(figsize=(12, 8))

# Sigmoid
plt.subplot(2, 2, 1)
plt.plot(x.numpy(), sigmoid_y.numpy(), label="Sigmoid", color="blue")
plt.title("Sigmoid Activation Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.legend()

# Tanh
plt.subplot(2, 2, 2)
plt.plot(x.numpy(), tanh_y.numpy(), label="Tanh", color="green")
plt.title("Tanh Activation Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.legend()

# ReLU
plt.subplot(2, 2, 3)
plt.plot(x.numpy(), relu_y.numpy(), label="ReLU", color="red")
plt.title("ReLU Activation Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.legend()

# Leaky ReLU
plt.subplot(2, 2, 4)
plt.plot(x.numpy(), leaky_relu_y.numpy(), label="Leaky ReLU", color="purple")
plt.title("Leaky ReLU Activation Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.legend()

# 调整布局并显示
plt.tight_layout()
plt.show()
