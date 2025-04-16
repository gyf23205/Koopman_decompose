import torch
import os
import numpy as np
from torch.nn.utils import parameters_to_vector
import torch.nn.utils.parametrize as parametrize
import torch.nn.functional as F
from classifier import MLP
import matplotlib.pyplot as plt

# print(np.mean([99.17, 99.21, 95.87, 94.81, 95.94, 96.26, 98.88, 98.12, 96.39, 95.11]))

seed = 6
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)


# Toy 1
w = torch.nn.Linear(2, 2)
optim = torch.optim.Adam(w.parameters())
real0 = []
imag0 = []
real1 = []
imag1 = []
for i in range(1000):
    eigvals, eigvecs = torch.linalg.eig(w.weight)
    # eigvecs = eigvecs.real
    # inv = torch.linalg.inv(eigvecs)
    # print(eigvals)
    # print(eigvecs)
    target = torch.tensor([1., 0.])
    loss = torch.linalg.norm(eigvals.real - target) + torch.linalg.norm(eigvals.imag)
    # loss = torch.linalg.norm(eigvals - target)
    real0.append(eigvals[0].real.item())
    imag0.append(eigvals[0].imag.item())
    real1.append(eigvals[1].real.item())
    imag1.append(eigvals[1].imag.item())
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(w.weight.grad.abs().sum())
    print()

plt.plot(real0)
plt.plot(imag0)
plt.plot(real1)
plt.plot(imag1)
plt.legend(['real0', 'imag0', 'real1', 'imag1'])
plt.show()


# Toy 2
# import torch
# from torch.nn.utils import parameters_to_vector
# import torch.nn.utils.parametrize as parametrize
# import torch.nn.functional as F
# from classifier import MLP

# w = torch.nn.Linear(2, 2)
# optim = torch.optim.Adam(w.parameters())

# for i in range(5):
#     eigvals, eigvecs = torch.linalg.eig(w.weight)
#     eigvecs = eigvecs.real
#     # inv = torch.linalg.inv(eigvecs)
#     print(eigvals)
#     print(eigvecs)
#     loss = eigvals[0].real
#     optim.zero_grad()
#     loss.backward()
#     optim.step()
#     print(w.weight.grad.abs().sum())