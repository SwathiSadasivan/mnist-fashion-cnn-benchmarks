import os
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(
    dataset_name: str = "fashion_mnist",
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    dataset_clean = dataset_name.lower().strip()

    if dataset_clean == "mnist":
        mean, std = (0.1307,), (0.3081,)
        dataset_cls = datasets.MNIST
    elif dataset_clean in ["fashion_mnist", "fmnist"]:
        mean, std = (0.2860,), (0.3530,)
        dataset_cls = datasets.FashionMNIST
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Choose 'mnist' or 'fashion_mnist'.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    raw_data_dir = os.path.join(data_dir, "raw")

    train_dataset = dataset_cls(
        root=raw_data_dir, train=True, download=True, transform=transform
    )
    test_dataset = dataset_cls(
        root=raw_data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader