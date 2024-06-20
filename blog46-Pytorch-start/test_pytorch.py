import torch
import numpy as np

print(torch.cuda.is_available())

#定义
np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)
print("numpy:\n", np_data)
print("torch:\n", torch_data)

tensor2array = torch_data.numpy()
print("tensor2array:\n", tensor2array, "\n")

#abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)    #32bit
print("numpy:", np.abs(data))       #[1 2 1 2]
print("torch:", torch.abs(tensor))  #[1 2 1 2]

print("numpy:", np.sin(data))
print("torch:", torch.sin(tensor))

print("numpy:", np.mean(data))
print("torch:", torch.mean(tensor), "\n")

#矩阵
data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data)    #32bit floating point
data = np.array(data)
print("numpy:", data.dot(data))
print("torch:", torch.mm(tensor, tensor))

A = torch.tensor([1, 1, 1])
B = torch.tensor([2, 3, 4])
print("torch:", torch.dot(A, B))
print("torch:", torch.mul(A, B))
