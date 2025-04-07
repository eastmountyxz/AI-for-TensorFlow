import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

#生成数据
data = torch.linspace(-1, 1, 100)
print(data)

#输入 一维数据生成二维
x = torch.unsqueeze(data, dim=1) # (tensor), shape=(100,1)
print(x.shape)

#输出
y = x.pow(2) + 0.4*torch.rand(x.size())

#变量
x,y = Variable(x), Variable(y)

plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()
