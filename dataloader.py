from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch


class FastMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).double().div(255)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data, torch.nn.functional.one_hot(self.targets).double()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target


def get_mnist_dataloader(args, path_to_data='mnist'):
    """MNIST dataloader with (28, 28) images."""

    all_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = FastMNIST(path_to_data, train=True, download=True, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle, pin_memory=False,
                              num_workers=0)
    _, c, x, y = next(iter(train_loader))[0].size()
    return train_loader, c * x * y, c
