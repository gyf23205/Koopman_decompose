import torch
import pickle
import copy
import torch.nn as nn
from classifier import MLP
from datasets.MNISTPerClass import MNISTPerClass


image_size = 784  # 28x28 images flattened
hidden_sizes = 4
num_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('data/submodels/params.pkl', 'rb') as f:
    param_sub_all = pickle.load(f)
classifier = MLP(image_size, hidden_sizes, num_classes).to(device)
batch_size = 64
mnist_per_class = MNISTPerClass(batch_size=batch_size)

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