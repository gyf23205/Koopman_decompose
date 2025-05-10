import torchvision
from collections import defaultdict
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

class FMNIST(object):
    def __init__(self, batch_size=64):
        super(FMNIST, self).__init__()
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])

        # Train and test datasets
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                transform=transform, download=True)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        # Test loader
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                transform=transform, download=True)

        self.test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)