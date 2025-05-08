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


    

# class MaskedMLPRMT(nn.Module):
#     def __init__(self, mlp, alpha):
#         super().__init__()
#         self.mlp = mlp
#         self.alpha = alpha
#         # Disable the gradient for the parameters of the original MLP
#         for param in self.mlp.parameters():
#             param.requires_grad = False
        
#         # Creating trainable masks
#         self.masks = nn.ParameterDict()
#         for name, param in self.mlp.named_parameters():
#             safe_name = name.replace('.', '_') 
#             mask = nn.Parameter(torch.rand(param.shape))
#             self.masks[safe_name] = mask
#         print()

#     def binarize_mask(self, masks):
#         with torch.autograd.no_grad():
#             masks_binary = self.masks.copy()
#             for name, mask in masks.items():
#                 safe_name = name.replace('.', '_')
#                 # masks_binary[safe_name] = 
#                 masks_binary[safe_name][mask > self.alpha] = 1
#                 masks_binary[safe_name][mask <= self.alpha] = 0

#             return masks_binary
        
#     def forward(self, x):
#         # Store original parameters
#         masks_binary = self.binarize_mask(self.masks)
#         original_params = {}
#         for name, param in self.mlp.named_parameters():
#             safe_name = name.replace('.', '_')
#             original_params[safe_name] = param.data
#             param.data = param.data * masks_binary[safe_name]
        
#         out = self.mlp(x)

#         # Restore original parameters
#         for name, param in self.mlp.named_parameters():
#             safe_name = name.replace('.', '_')
#             param.data = original_params[safe_name]

#         return out