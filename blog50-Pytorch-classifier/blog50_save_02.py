import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)   #reproducible

# fake data
x = torch.unsqueeze(torch.linspace(-1,1,100), dim=1) # x data (100,1)
y = x.pow(2)+0.2*torch.rand(x.size()) #noisy y data (100,1)
x,y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)

def save():
    #定义神经网络
    net = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    optimizer = torch.optim.SGD(net.parameters(),lr=0.5)
    loss_func = torch.nn.CrossEntropyLoss()

    #模型训练100次
    for t in range(100):
        pre = net(x)
        loss = loss_func(pre, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #保存整个神经网络: 模型 名称 
    torch.save(net, 'net.pkl')

    #保存神经网络中节点参数 parameters
    torch.save(net.state_dict(), 'net_params.pkl') #神经网络状态
    
def restore_net():
    #定义要提取的神经网络
    net2 = torch.load('net.pkl')

def restore_params():
    #定义相同结构的网络再提取参数
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    net3.load_state_dict('net_params.pkl')


