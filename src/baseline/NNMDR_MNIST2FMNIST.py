import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from datasets.MNIST import MNIST
# from datasets.MNISTPerClass import MNISTPerClass
from datasets.FMNISTPerClass import FMNISTPerClass
from datasets.FMNIST import FMNIST
from classifier import MLP, MLPNNMDR
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def binarize_labels(labels, target_class):
    return (labels == target_class).type(torch.float32)

def get_avg_score(modules):
    avg1 = []
    avg2 = []
    for module in modules:
        s1, s2 = module.get_scores()
        avg1.append(s1.unsqueeze(0))
        avg2.append(s2.unsqueeze(0))
    avg1, avg2 = torch.cat(avg1), torch.cat(avg2)
    return torch.mean(avg1, dim=0), torch.mean(avg2, dim=0)

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
            testloader = fmnist_per_class.sub_testloaders[idx]
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

def test_downstream_binary(model):
    with torch.autograd.no_grad():
        model.eval()
        model_acc = np.zeros((num_classes,))
        for idx in range(num_classes):
            # idx = 3
            testloader = fmnist_per_class.sub_testloaders[idx]
            total = 0
            correct = 0
            for images, labels in testloader:
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                labels_bin = binarize_labels(labels, idx) # labels should be all 1 after being binarized
                outputs = model(images).squeeze()
                predicted = (outputs > 0.5).type(torch.float32)
                total += labels.size(0)
                correct += (predicted == labels_bin).sum().item()
            accuracy = 100 * correct / total
            model_acc[idx] = np.round(accuracy, 1)
        print(f'Model accuracy: {model_acc}')

def test(model, idx):
    with torch.autograd.no_grad():
        model.eval()
        # model_acc = 0
        accuracy = 0
        testloader = fmnist.test_loader
        total = 0
        correct = 0
        for images, labels in testloader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            labels_bin = binarize_labels(labels, idx) # labels should be all 1 after being binarized
            outputs = model(images).squeeze()
            predicted = (outputs > 0.5).type(torch.float32)
            total += labels.size(0)
            correct += (predicted == labels_bin).sum().item()
        accuracy = 100 * correct / total
        # model_acc[idx] = accuracy
        print(f'Model accuracy: {accuracy}')

if __name__=='__main__':
    np.set_printoptions(linewidth=np.inf)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    results = torch.load('results/result.pth', map_location=torch.device(device))
    state_dict = results['param_original']
    image_size = 784  # 28x28 images flattened
    hidden_size = 16
    num_mode = 4
    num_classes = 10
    classifier_original = MLP(image_size, hidden_size, num_classes).to(device)
    classifier_original.load_state_dict(state_dict)
    writer = SummaryWriter(log_dir='results/baseline3')

    # Get the masked model
    k = 10
    alpha = 1
    
    
    # Set up training
    epoch = 100
    batch_size = 4
    idx_class_train = [9] # Target class
    modules = []
    mlp_init = MLPNNMDR(image_size, hidden_size, num_classes, k).to(device)
    manual_load_state_dict(mlp_init, state_dict)
    params_trainable_init = [p for n, p in mlp_init.named_parameters() if 'score' in n]
    optim_init = torch.optim.SGD(params_trainable_init, lr=1e-2, weight_decay=0.5, momentum=0.9)
        

    # Modules and optims
    fmnist = FMNIST(batch_size)
    trainloader = fmnist.train_loader
    fmnist_per_class = FMNISTPerClass(batch_size)
    train_loader_per_class = [fmnist_per_class.sub_trainloaders[i] for i in idx_class_train]
    optims = []
    for _ in idx_class_train:
        mlp_masked = MLPNNMDR(image_size, hidden_size, num_classes, k).to(device)
        manual_load_state_dict(mlp_masked, state_dict)
        mlp_masked.train()
        modules.append(mlp_masked)
        params_trainable = [p for n, p in mlp_masked.named_parameters() if 'score' in n or 'bin' in n]

        # params_trainable.append(mlp_masked.bin)
        optim = torch.optim.SGD(params_trainable, lr=1e-2, weight_decay=0.5, momentum=0.9)
        optims.append(optim)
        
    binary_cross_entropy = nn.BCELoss()
    cross_entropy = nn.CrossEntropyLoss()

    for i in range(epoch):
        # print(i)
        images, labels = next(iter(trainloader))
        images, labels = images.view(-1, image_size).to(device), labels.to(device)
        if i == 0:# No stemming
            logits = mlp_init.forward_all(images)
            loss_ce = cross_entropy(logits, labels)
            optim_init.zero_grad()
            loss_ce.backward()
            grad_in1 = mlp_init.I1.grad
            grad1 = grad_in1.unsqueeze(2) * mlp_init.weight1.repeat(batch_size, 1, 1) * images.unsqueeze(1)

            grad_in2 = mlp_init.I2.grad
            grad2 = grad_in2.unsqueeze(2) * mlp_init.weight2.repeat(batch_size, 1, 1) * mlp_init.x_inter.unsqueeze(1)
            with torch.autograd.no_grad():
                mlp_init.score_weight1.grad = torch.mean(grad1, dim=0)
                mlp_init.score_weight2.grad = torch.mean(grad2, dim=0)
            optim_init.step()
            avg1, avg2 = get_avg_score([mlp_init])
        else:
            # print("no")
            # Get average scores
            for j, module in enumerate(modules):
                # Data
                images_pure, labels_pure = next(iter(train_loader_per_class[j]))
                # plt.imshow(images_pure[0, :, :].squeeze())
                # plt.show()
                images_pure, labels_pure = images_pure.view(-1, image_size).to(device), labels_pure.to(device)
                images = torch.cat([images, images_pure], dim=0)
                labels = torch.cat([labels, labels_pure], dim=0)
                labels_bin = binarize_labels(labels, idx_class_train[j])
                probs = module(images).squeeze()
                # with torch.autograd.no_grad():
                #     preds = (probs > 0.5)
                #     print(preds == labels_bin)
                loss_bce = binary_cross_entropy(probs, labels_bin)
                loss_norm = torch.sum((module.score_weight1 - avg1)**2) + torch.sum((module.score_weight2 - avg2)**2)
                loss = loss_bce + (alpha/module.n_params) * loss_norm
                optims[j].zero_grad()
                loss.backward()
                # print(loss.item())
                writer.add_scalar(f'loss\{j}', loss.item(), i)
                optims[j].step()
            avg1, avg2 = get_avg_score(modules)
  
    # test_downstream(classifier_original)                  
    for i, m in enumerate(modules):
        test_downstream_binary(m)
    
    # test(modules[0], 3)
    
