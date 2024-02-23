from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def get_mnist(path_to_data='mnist'):
    """MNIST dataloader with (28, 28) images."""
    all_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(path_to_data, train=True, download=True, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=60000, pin_memory=True)
    # im output range [0, 1]
    return next(iter(train_loader))
