import torch

a = torch.tensor([1+2j, 2+3j])
b = torch.tensor(a, dtype=torch.float32)
print(b)