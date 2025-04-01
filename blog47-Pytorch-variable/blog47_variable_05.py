import torch

a = torch.tensor([2.0, 3.0])
b = torch.tensor([1.0, 4.0])

# 加减乘除
add_res = a + b       # 加法
mul_res = a * b       # 逐元素乘法
matmul_res = a @ b    # 矩阵乘法（点积）
print(add_res,"\n",mul_res,"\n",matmul_res)

# 计算均值、最大值、最小值
mean_res = a.mean()   # 均值
max_res = a.max()     # 最大值
print(mean_res,"\n",max_res)

# 调整维度
x = torch.randn(4, 4)
x_reshape = x.view(2, 8)    # 变换形状
x_transpose = x.t()         # 转置
x_unsqueeze = x.unsqueeze(0)  # 增加维度
x_squeeze = x.squeeze()       # 去除维度
print(x)
print(x_reshape)
print(x_transpose)
print(x_unsqueeze)
print(x_squeeze)

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x[0, :])  # 访问第一行
print(x[:, 1])  # 访问第二列
print(x[1, 1])  # 访问第二行第二列的元素

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 2 * x
y.sum().backward()  # 计算梯度
print(x.grad)  # 输出 dy/dx



