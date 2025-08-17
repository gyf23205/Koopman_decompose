import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        out = self.fc_layers(x)
        return out # Outputing logits, not probability


class MLPDeep(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPDeep, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        out = self.fc_layers(x)
        return out # Outputing logits, not probability 

class MaskedMLPRMT(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, alpha):
        super().__init__()
        self.alpha = alpha

        self.weight1 = nn.Parameter(torch.rand((input_size, hidden_size)))
        self.bias1 = nn.Parameter(torch.rand((hidden_size,)))
        self.mask_weight1 = nn.Parameter(torch.ones((hidden_size, input_size))*1e-2)
        self.mask_bias1 = nn.Parameter(torch.ones((hidden_size,))*1e-2)
        self.mask_weight1_bin = nn.Parameter(self.mask_weight1.clone())
        self.mask_bias1_bin = nn.Parameter(self.mask_bias1.clone())

        self.weight2 = nn.Parameter(torch.rand((hidden_size, num_classes)))
        self.bias2 = nn.Parameter(torch.rand((hidden_size,)))
        self.mask_weight2 = nn.Parameter(torch.ones((num_classes, hidden_size))*1e-2)
        self.mask_bias2 = nn.Parameter(torch.ones((num_classes,))*1e-2)
        self.mask_weight2_bin = nn.Parameter(self.mask_weight2.clone())
        self.mask_bias2_bin = nn.Parameter(self.mask_bias2.clone())


    def update_binary_mask(self):
        with torch.autograd.no_grad():
            self.mask_weight1_bin[self.mask_weight1 > self.alpha] = 1
            self.mask_weight1_bin[self.mask_weight1 <= self.alpha] = 0
            
            self.mask_bias1_bin[self.mask_bias1 > self.alpha] = 1
            self.mask_bias1_bin[self.mask_bias1 <= self.alpha] = 0
            
            self.mask_weight2_bin[self.mask_weight2 > self.alpha] = 1
            self.mask_weight2_bin[self.mask_weight2 <= self.alpha] = 0
            
            self.mask_bias2_bin[self.mask_bias2 > self.alpha] = 1
            self.mask_bias2_bin[self.mask_bias2 <= self.alpha] = 0

    def forward(self, x):
        weight1 = self.weight1 * self.mask_weight1_bin
        bias1 = self.bias1 * self.mask_bias1_bin
        x = x @ weight1.T + bias1

        x = F.relu(x)

        weight2 = self.weight2 * self.mask_weight2_bin
        bias2 = self.bias2 * self.mask_bias2_bin
        x = x @ weight2.T + bias2

        return x


    

class MLPPOP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, k):
        super(MLPPOP, self).__init__()
        self.k = k
        self.weight1 = nn.Parameter(torch.rand((input_size, hidden_size)))
        self.bias1 = nn.Parameter(torch.rand((hidden_size,)))
        self.score_weight1 = nn.Parameter(nn.init.kaiming_normal_(torch.empty((hidden_size, input_size)), nonlinearity='relu'))

        self.x_inter = None

        self.weight2 = nn.Parameter(torch.rand((hidden_size, num_classes)))
        self.bias2 = nn.Parameter(torch.rand((hidden_size,)))
        self.score_weight2 = nn.Parameter(nn.init.kaiming_normal_(torch.empty((num_classes, hidden_size)), nonlinearity='relu'))
        
    def keep_topk(self, weight, score):
        '''
        Keep largest k% entries, zero-out all the remaining entries.
        '''
        n_param = torch.prod(torch.tensor(weight.shape), dtype=int)
        n_remain = int(n_param * (self.k / 100))
        _, idx_flat = torch.topk(score.flatten(), n_remain)
        idx = torch.unravel_index(idx_flat, weight.shape)
        # print(idx)
        mask = torch.zeros_like(weight)
        mask[idx[0], idx[1]] = 1.
        weight_masked = weight * mask
        return weight_masked
        
    def forward(self, x):
        weight1_masked = self.keep_topk(self.weight1, self.score_weight1)
        self.I1 = x @ weight1_masked.T + self.bias1
        if self.training:
            self.I1.retain_grad()
        # self.I1 = I1

        x = F.relu(self.I1)
        self.x_inter = x

        weight2_masked = self.keep_topk(self.weight2, self.score_weight2)
        self.I2 = x @ weight2_masked.T + self.bias2
        if self.training:
            self.I2.retain_grad()

        out = self.I2
        return out # Outputing logits, not probability
    

class MLPNNMDR(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, k):
        super(MLPNNMDR, self).__init__()
        self.k = k
        self.weight1 = nn.Parameter(torch.rand((hidden_size, input_size)))
        self.bias1 = nn.Parameter(torch.rand((hidden_size,)))
        self.score_weight1 = nn.Parameter(nn.init.kaiming_normal_(torch.empty((hidden_size, input_size)), nonlinearity='relu'))

        self.x_inter = None

        self.weight2 = nn.Parameter(torch.rand((num_classes, hidden_size)))
        self.bias2 = nn.Parameter(torch.rand((hidden_size,)))
        self.score_weight2 = nn.Parameter(nn.init.kaiming_normal_(torch.empty((num_classes, hidden_size)), nonlinearity='relu'))

        self.weight_bin = nn.Parameter(torch.rand((num_classes, 1)))
        self.bias_bin = nn.Parameter(torch.rand((1,)))
        self.n_params = input_size * hidden_size + hidden_size + hidden_size * num_classes + num_classes
        
    def keep_topk(self, weight, score):
        '''
        Keep largest k% entries, zero-out all the remaining entries.
        '''
        n_param = torch.prod(torch.tensor(weight.shape), dtype=int)
        n_remain = int(n_param * (self.k / 100))
        _, idx_flat = torch.topk(score.flatten(), n_remain)
        idx = torch.unravel_index(idx_flat, weight.shape)
        # print(idx)
        mask = torch.zeros_like(weight)
        mask[idx[0], idx[1]] = 1.
        weight_masked = weight * mask
        return weight_masked
    
    def get_scores(self):
        return (self.score_weight1.detach(), self.score_weight2.detach()) 

    def forward_all(self, x):
        weight1_masked = self.keep_topk(self.weight1, self.score_weight1)
        self.I1 = x @ weight1_masked.T + self.bias1
        if self.training:
            self.I1.retain_grad()
        # self.I1 = I1

        x = F.relu(self.I1)
        self.x_inter = x

        weight2_masked = self.keep_topk(self.weight2, self.score_weight2)
        self.I2 = x @ weight2_masked.T + self.bias2
        if self.training:
            self.I2.retain_grad()

        logits = self.I2
        # out = F.sigmoid(self.bin(logits))
        return logits

    def forward(self, x):
        weight1_masked = self.keep_topk(self.weight1, self.score_weight1)
        self.I1 = x @ weight1_masked.T + self.bias1
        if self.training:
            self.I1.retain_grad()
        # self.I1 = I1

        x = F.relu(self.I1)
        self.x_inter = x

        weight2_masked = self.keep_topk(self.weight2, self.score_weight2)
        self.I2 = x @ weight2_masked.T + self.bias2
        if self.training:
            self.I2.retain_grad()

        logits = self.I2

        out = F.sigmoid(logits @ self.weight_bin + self.bias_bin)
        return out