import torch
from torch.autograd import Variable

#神经网络将张量放在变量中进行计算
tensor = torch.FloatTensor([[1,2], [3,4]])

#通过Variable搭建计算图纸 实现反向传播
variable = Variable(tensor, requires_grad=True)
print(tensor)
print(variable)

#计算 两者差别不大 但Tensor不能反向传播
t_out = torch.mean(tensor * tensor) #x^2
v_out = torch.mean(variable * variable)
print(t_out)
print(v_out)

#误差反向传递
v_out.backward()
# v_out = 1/4*sum(var*var)
# d(v_out)/d(var) = 1/4*2*variable = variable/2
print(variable.grad) #反向传递的梯度

#data属性
print(variable.data) 

#转换numpy格式
print(variable.data.numpy())


