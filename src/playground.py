import torch

a = torch.tensor([1., 2.])
b = a.clone()
b[0] = 0
print(a)
print(b)