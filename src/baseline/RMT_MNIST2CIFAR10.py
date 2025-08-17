import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets.MNIST import MNIST
from datasets.MNISTPerClass import MNISTPerClass
from datasets.FMNISTPerClass import FMNISTPerClass
from datasets.CIFAR10PerClass import CIFARPerClass
from classifier import MLP, MaskedMLPRMT
import matplotlib.pyplot as plt

def compute_gradient_norm(model, norm_type=2):
    with torch.no_grad():
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)
    return total_norm

def manual_load_state_dict(model_masked, state_dict):
    with torch.autograd.no_grad():
        model_masked.weight1.data = state_dict['fc_layers.0.weight']
        model_masked.weight2.data = state_dict['fc_layers.2.weight']

        model_masked.bias1.data = state_dict['fc_layers.0.bias']
        model_masked.bias2.data = state_dict['fc_layers.2.bias']

def test_downstream(model):
    with torch.autograd.no_grad():
        model.eval()
        model_acc = np.zeros((num_classes,))
        for idx in range(num_classes):
            testloader = cifar10_per_class.sub_testloaders[idx]
            total = 0
            correct = 0
            for images, labels in testloader:
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            model_acc[idx] = accuracy
        print(f'Model accuracy: {model_acc}')

if __name__=='__main__':
    # Get the pretrained classifier
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    results = torch.load('results/result.pth', map_location=torch.device(device))
    state_dict = results['param_original']
    image_size = 784  # 28x28 images flattened
    hidden_sizes = 16
    num_mode = 4
    num_classes = 10
    indicator = lambda x: (x>0).float()
    # with open('data/submodels/params.pkl', 'rb') as f:
    # classifier = MLP(image_size, hidden_sizes, num_classes).to(device)
    # classifier.load_state_dict(state_dict)
    classifier_original = MLP(image_size, hidden_sizes, num_classes).to(device)
    classifier_original.load_state_dict(state_dict)

    # Get the masked model
    alpha = 5e-3
    gamma = 1e-2
    mlp_masked = MaskedMLPRMT(image_size, hidden_sizes, num_classes, alpha)
    # mlp_masked.create_binary_mask()
    manual_load_state_dict(mlp_masked, state_dict)
    mlp_masked.train()
    # params_trainable = [p for n, p in mlp_masked.named_parameters() if 'mask' in n]
    params_trainable_bin = [p for n, p in mlp_masked.named_parameters() if 'bin' in n]
    params_trainable = [p for n, p in mlp_masked.named_parameters() if n.startswith('mask') and not n.endswith('bin')]
    

    # Set up training
    l = 1
    epoch = 2
    batch_size = 4
    idx_class_train = 5 # Target class
    # mnist = MNIST(batch_size=batch_size).train_loader
    cifar10_per_class = CIFARPerClass(batch_size=batch_size)
    trainloader = cifar10_per_class.sub_trainloaders[idx_class_train]
    optim = torch.optim.Adam(params_trainable, lr=gamma)
    cross_entropy = nn.CrossEntropyLoss()
    kl_div = nn.KLDivLoss()
    for i in range(epoch):
        print(i)
        images, labels = next(iter(trainloader))
        # plt.imshow(np.squeeze(images[0, :].numpy()), cmap='gray')
        # plt.show()
        images, labels = images.view(-1, image_size).to(device), labels.to(device)
        mlp_masked.update_binary_mask()
        logits = mlp_masked(images)
        logits_original = classifier_original(images)
        loss_ce = cross_entropy(logits, labels)
        loss_kl = kl_div(F.log_softmax(logits, dim=-1), F.softmax(logits_original, dim=-1))
        grad_ce = torch.autograd.grad(loss_ce, params_trainable_bin, retain_graph=True)
        grad_kl = torch.autograd.grad(loss_kl, params_trainable_bin)
        grad_final = []
        for j in range(len(grad_ce)):
            P = 0.5 * (1 + (torch.sign(grad_ce[j]) * (grad_ce[j] + grad_kl[j])) / (grad_kl[j].abs() + grad_ce[j].abs()))
            U = torch.rand(P.shape)
            with torch.autograd.no_grad():
                params_trainable[j].grad = (1 - l * (1 - indicator(P - U))) * grad_ce[j]
        optim.step()
        optim.zero_grad()
    
    test_downstream(mlp_masked)
    test_downstream(classifier_original)


