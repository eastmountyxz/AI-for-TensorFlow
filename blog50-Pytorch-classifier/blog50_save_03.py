import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

# fake data
x = torch.unsqueeze(torch.linspace(-1,1,100), dim=1) # x data (100,1)
y = x.pow(2)+0.2*torch.rand(x.size()) #noisy y data (100,1)

# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)

def save():
    #定义神经网络
    net = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

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

    #可视化
    plt.figure(1, figsize=(10,3))
    plt.subplot(131)
    plt.title('Net')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pre.data.numpy(), 'r-', lw=5)
    
def restore_net():
    #定义要提取的神经网络
    net2 = torch.load('net.pkl')
    pre = net2(x)
    
    #可视化
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pre.data.numpy(), 'r-', lw=5)

def restore_params():
    #定义相同结构的网络再提取参数
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    net3.load_state_dict(torch.load('net_params.pkl'))
    pre = net3(x)
    
    #可视化
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pre.data.numpy(), 'r-', lw=5)
    plt.show()

#save net
save()

#restore entire net
restore_net()

#restore only the net parameters
restore_params()

