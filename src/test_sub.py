import torch
import pickle
import numpy as np
np.set_printoptions(precision=2, suppress=True)
import copy
import torch.nn as nn
from classifier import MLP
from datasets.MNISTPerClass import MNISTPerClass
from datasets.MNIST import MNIST

def test_sub_own_class(param_sub_all):
    with torch.autograd.no_grad():
        for idx_sub in range(num_classes):
            # param_sub = compute_theta_sub(kae, z, idx_sub)
            param_sub = param_sub_all[:, idx_sub]
            classifier_sub = copy.deepcopy(classifier)
            classifier_sub.eval()
            nn.utils.vector_to_parameters(param_sub, classifier_sub.parameters())
            testloader = mnist_per_class.sub_trainloaders[idx_sub]
            total = 0
            correct = 0
            for images, labels in testloader:
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                outputs = classifier_sub(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f'Test Accuracy of class {idx_sub:d}: {accuracy:.2f}%')


def test_recon_all(param_sub_all):
    with torch.autograd.no_grad():
        param_re = param_sub_all.sum(1)
        classifier_sub = copy.deepcopy(classifier)
        classifier_sub.eval()
        nn.utils.vector_to_parameters(param_re, classifier_sub.parameters())
        testloader = mnist.test_loader
        total = 0
        correct = 0
        for images, labels in testloader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = classifier_sub(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Test Accuracy : {accuracy:.2f}%')


def test_sub_all(param_sub_all):
    with torch.autograd.no_grad():
        for idx_model in range(num_mode):
            model_acc = np.zeros((num_classes,))
            for idx_class in range(num_classes):
                total = 0
                correct = 0
                param_sub = param_sub_all[:, idx_model]
                classifier_sub = copy.deepcopy(classifier)
                classifier_sub.eval()
                nn.utils.vector_to_parameters(param_sub, classifier_sub.parameters())
                testloader = mnist_per_class.sub_trainloaders[idx_class]
                total = 0
                correct = 0
                for images, labels in testloader:
                    images = images.reshape(-1, 28*28).to(device)
                    labels = labels.to(device)
                    outputs = classifier_sub(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                model_acc[idx_class] = accuracy
            print(f'Model {idx_model} accuracy: {model_acc}')


if __name__=='__main__':
    image_size = 784  # 28x28 images flattened
    hidden_sizes = 16
    num_mode = 4
    num_classes = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # with open('data/submodels/params.pkl', 'rb') as f:
    #     param_sub_all = pickle.load(f)
    result = torch.load('results/result.pth')
    param_sub_all = result['params_sub']
    classifier = MLP(image_size, hidden_sizes, num_classes).to(device)
    batch_size = 64
    mnist_per_class = MNISTPerClass(batch_size=batch_size)
    mnist = MNIST(batch_size=batch_size)

    test_sub_all(param_sub_all)
    test_recon_all(param_sub_all)