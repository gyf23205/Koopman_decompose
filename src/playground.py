import torch

a = torch.tensor([[2., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 3.]])

eigvals, eigvecs = torch.linalg.eig(a)
print(eigvals)
print(eigvecs)