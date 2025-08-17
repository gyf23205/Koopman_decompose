import torch
import torch.nn as nn
import torch.nn.functional as F

class Binary(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        x = self.linear(x)
        return F.sigmoid(x)
