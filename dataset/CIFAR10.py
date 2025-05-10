import torchvision
from collections import defaultdict
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

class CIFAR10(object):
    def __init__(self, batch_size=64):
        super(CIFAR10, self).__init__()
        transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.Resize((28, 28)),
        # ToGray()
        ])

        # Train and test datasets
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                transform=transform, download=True)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        # Test loader
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                transform=transform, download=True)

        self.test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)