import torchvision
from collections import defaultdict
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

class MNISTPerClass(object):
    def __init__(self, batch_size=64):
        super(MNISTPerClass, self).__init__()
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Train and test datasets
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                                transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                                transform=transform, download=True)
        
        # Train loaders by class
        class_indices_train = defaultdict(list)
        class_indices_test = defaultdict(list)
        for idx, (x, y) in enumerate(train_dataset):
            class_indices_train[y].append(idx)

        for idx, (x, y) in enumerate(test_dataset):
            class_indices_test[y].append(idx)
        

        self.sub_trainloaders = {}
        self.sub_testloaders = {}
        for y in range(10):
            subset_train = Subset(train_dataset, class_indices_train[y])
            self.sub_trainloaders[y] = DataLoader(dataset=subset_train, batch_size=batch_size, shuffle=True)

            subset_test = Subset(test_dataset, class_indices_test[y])
            self.sub_testloaders[y] = DataLoader(dataset=subset_test, batch_size=batch_size, shuffle=False)

        