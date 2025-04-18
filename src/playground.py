import torch

a = torch.linspace(0, 9, 10, dtype=int)
print(torch.isin(a, torch.tensor([0,1,2])))
