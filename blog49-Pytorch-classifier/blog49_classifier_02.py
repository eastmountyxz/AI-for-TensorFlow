import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

#数据集生成 两个类簇
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data,1)  #class0 x data tensor shape=(100,2)
y0 = torch.zeros(100)          #class0 y data tensor shape=(100,1)
x1 = torch.normal(-2*n_data,1) #class1 x data tensor shape=(100,2)
y1 = torch.ones(100)           #class1 y data tensor shape=(100,1)
x = torch.cat((x0,x1),0).type(torch.FloatTensor) #32-bit floating
y = torch.cat((y0,y1),).type(torch.LongTensor)   #64-bit integer

x,y = Variable(x), Variable(y)

#定义神经网络类
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() #继承Module
        
        #隐藏层: 输入神经元个数、隐藏层神经元个数
        self.hidden = torch.nn.Linear(n_feature, n_hidden)

        #预测层:隐藏层神经元个数、输出个数(y)
        self.predict = torch.nn.Linear(n_hidden, n_output)
 
    def forward(self, x):
        x = F.relu(self.hidden(x)) #激励函数加工隐藏层信息x
        x = self.predict(x)
        return x

#使用神经网络
net = Net(2, 10, 2) #One-hot Class0:[1,0] Class1:[0,1]
print(net)

plt.ion() #something about plotting
plt.show()

#优化神经网络 优化参数学习效率小于1
optimizer = torch.optim.SGD(net.parameters(),lr=0.02)

#定义损失函数 使用CrossEntropyLoss处理多分类问题
loss_func = torch.nn.CrossEntropyLoss()

#训练100次
for t in range(100):
    #神经网络预测
    out = net(x)
    
    #计算真实值和预测值误差
    loss = loss_func(out, y)
    
    #优化 1.参数梯度先降为0；2.反向传递计算节点梯度；3.优化梯度 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #可视化分析
    if t % 5 == 0:
        plt.cla()
        #计算值out[2, 5, 15]会通过F.softmax转换为概率[0.1, 0.2, 0.7]
        #输出最大值所在位置(索引)
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        
        plt.scatter(x.data.numpy()[:,0],
                    x.data.numpy()[:,1],
                    c=pred_y,
                    s=100,
                    lw=0,
                    cmap='rainbow')
        accuracy = sum(pred_y == target_y) / 200
        plt.text(1.5, -4, 'Acc=%.2f' % accuracy, fontdict={'size':20, 'color': 'red'})
        plt.pause(0.5)
        
plt.ioff()
plt.show()

