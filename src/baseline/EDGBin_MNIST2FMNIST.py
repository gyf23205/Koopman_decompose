import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets.FMNIST import FMNIST
from datasets.FMNISTPerClass import FMNISTPerClass
# from datasets.FMNISTPerClass import FMNISTPerClass
from classifier import MLP, MLPPOP
from tqdm import tqdm
from GoBin import Binary

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
            model_acc[idx] = np.round(accuracy, 1)
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
                logits = model(images)
                outputs = bin(logits).squeeze()
                predicted = (outputs > 0.5).type(torch.float32)
                total += labels.size(0)
                correct += (predicted == labels_bin).sum().item()
            accuracy = 100 * correct / total
            model_acc[idx] = np.round(accuracy, 1)
        print(f'Model accuracy: {model_acc}')

def binarize_labels(labels, target_class):
    return (labels == target_class).type(torch.float32)

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(suppress=True)
    results = torch.load('results/result.pth', map_location=torch.device(device))
    state_dict = results['param_original']
    image_size = 784  # 28x28 images flattened
    hidden_size = 16
    num_mode = 4
    num_classes = 10
    classifier_original = MLP(image_size, hidden_size, num_classes).to(device)
    classifier_original.load_state_dict(state_dict)

    # Get the masked model
    k = 10
    alpha = 1
   

    # Set up training
    epoch = 100
    batch_size = 64
    idx_class_train_list = [i for i in range(10)] # Target class
    for idx_class_train in idx_class_train_list:
        mlp_masked = MLPPOP(image_size, hidden_size, num_classes, k).to(device)
        manual_load_state_dict(mlp_masked, state_dict)
        mlp_masked.train()
        params_trainable = [p for n, p in mlp_masked.named_parameters() if 'score' in n]
        bin = Binary().to(device)
        fmnist = FMNIST(batch_size=batch_size)
        trainloader = fmnist.train_loader
        fmnist_per_class = FMNISTPerClass(batch_size=batch_size)
        trainloader_pure = fmnist_per_class.sub_trainloaders[idx_class_train]
        optim = torch.optim.Adam(params_trainable)
        optim_bin = torch.optim.Adam(bin.parameters())
        bce = nn.BCELoss()
        for i in range(epoch):
            images, labels = next(iter(trainloader))
            images, labels = images.view(-1, image_size).to(device), labels.to(device)
            images_pure, labels_pure = next(iter(trainloader_pure))
            images_pure, labels_pure = images_pure.view(-1, image_size).to(device), labels_pure.to(device)
            images = torch.cat([images, images_pure], dim=0)
            labels = torch.cat([labels, labels_pure], dim=0)
            labels_bin = binarize_labels(labels, idx_class_train)
            logits = mlp_masked(images)
            probs = bin(logits).squeeze()
            loss_ce = bce(probs, labels_bin)

            optim.zero_grad()
            optim_bin.zero_grad()
            loss_ce.backward()
            # print(mlp_masked.I1.grad)
            # print(mlp_masked.I2.grad)
            # if i == 0: # No stemming
            grad_in1 = mlp_masked.I1.grad
            grad1 = grad_in1.unsqueeze(2) * mlp_masked.weight1.repeat(batch_size*2, 1, 1) * images.unsqueeze(1)

            grad_in2 = mlp_masked.I2.grad
            grad2 = grad_in2.unsqueeze(2) * mlp_masked.weight2.repeat(batch_size*2, 1, 1) * mlp_masked.x_inter.unsqueeze(1)
            with torch.autograd.no_grad():
                mlp_masked.score_weight1.grad = torch.mean(grad1, dim=0)
                mlp_masked.score_weight2.grad = torch.mean(grad2, dim=0)
            optim.step()
            optim_bin.step()

        test_downstream_binary(mlp_masked)
        # test_downstream(classifier_original)
