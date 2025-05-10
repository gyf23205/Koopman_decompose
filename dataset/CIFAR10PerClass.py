import torchvision
from collections import defaultdict
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Subset

class CIFARPerClass(object):
    def __init__(self, batch_size=64):
        super(CIFARPerClass, self).__init__()
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
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
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


class ToGray:
    def __init__(self):
        pass

    def __call__(self, image):
        return F.rgb_to_grayscale(image)