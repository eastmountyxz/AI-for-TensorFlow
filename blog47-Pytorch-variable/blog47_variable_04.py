import torch

#可计算梯度的变量
x = torch.tensor([[2.0,
                   3.0]],
                 requires_grad=True)  
y = x ** 2 + 3 * x
print("变量:", y)

#计算梯度 标量才能计算梯度
y.sum().backward()
print("梯度:",x.grad) #dy/dx
