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

#定义神经网络类
class Net(torch.nn.Module):
    #定义神经网络
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() #继承Module
        
        #隐藏层: 输入神经元个数、隐藏层神经元个数
        self.hidden = torch.nn.Linear(n_feature, n_hidden)

        #预测层:隐藏层神经元个数、输出个数(y)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    #搭建神经网络  
    def forward(self, x):
        x = F.relu(self.hidden(x)) #激励函数加工隐藏层信息x
        x = self.predict(x) #
        return x
        
#使用神经网络
net = Net(1, 10, 1)
print(net)

plt.ion() #something about plotting
plt.show()

#优化神经网络 优化参数学习效率小于1
optimizer = torch.optim.SGD(net.parameters(),lr=0.5)

#定义损失函数 使用均方差处理回归问题
loss_func = torch.nn.MSELoss()

#训练
for t in range(200):
    #神经网络预测
    pre = net(x)
    
    #计算真实值和预测值误差
    loss = loss_func(pre, y)
    
    #优化 1.参数梯度先降为0；2.反向传递计算节点梯度；3.优化梯度 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print(loss)

    #可视化分析
    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), pre.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.item(),
                 fontdict={'size':20, 'color': 'red'})
        plt.pause(0.1)
        
plt.ioff()
plt.show()

    
    


