from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler


def generate_dataset(batch_size):
    ds = []

    dataset = datasets.MNIST("/mnt/data03/renge/public_dataset/pytorch/mnist-data", train=True, download=True,
                       transform=transforms.ToTensor())

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(dataset))
    ds.append(train_loader)

    dataset = datasets.MNIST("/mnt/data03/renge/public_dataset/pytorch/mnist-data", train=False, download=True,
                             transform=transforms.ToTensor())
    test_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        )
    ds.append(test_loader)

    return ds