import torch
import numpy as np
import torchvision
from torchvision import transforms

from torchvision.datasets import ImageFolder
import os

def get_transform(dataset_name):
    if dataset_name == 'tinyimagenet':
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    elif dataset_name in ['fmnist', 'cifar10', 'cifar100']:
        return transforms.Compose([
        torchvision.transforms.ToTensor(),
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_loader(dataset_name, train=True, batch_size=128, shuffle=True):
    transform_train = get_transform(dataset_name)
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform_train)
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform_train)
    elif dataset_name == 'fmnist':
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=train, download=True,
                                                    transform=transform_train)
    # elif dataset_name == 'tinyimagenet':
    #     subdir = 'train' if train else 'val'
    #     data_samples = ImageFolder(root=os.path.join('./data/tiny-imagenet-200', subdir), transform=transform_train)
    #     num_samples = len(data_samples)
    #     dataset = torch.zeros((num_samples, 3, 64, 64), dtype=torch.float32)
    #     for i, (img, _) in enumerate(data_samples):
    #         dataset[i] = img
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def get_data_and_labels(dataset_name, train = True):
    transform_train = get_transform(dataset_name)
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        data = torch.tensor(np.transpose(dataset.data, [0, 3, 1, 2]) / 255, dtype=torch.float32)
        labels = torch.tensor(dataset.targets)
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        data = torch.tensor(np.transpose(dataset.data, [0, 3, 1, 2]) / 255, dtype=torch.float32)
        labels = torch.tensor(dataset.targets)
    elif dataset_name == 'fmnist':
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
        data = dataset.data.unsqueeze(1).float() / 255.0
        labels = torch.tensor(dataset.targets)
    elif dataset_name == 'tinyimagenet':
        subdir = 'train' if train else 'val'
        dataset = ImageFolder(root=os.path.join('./data/tiny-imagenet-200', subdir), transform=transform_train)
        num_samples = len(dataset)
        data = torch.zeros((num_samples, 3, 64, 64), dtype=torch.float32)
        labels = torch.zeros((num_samples,), dtype=torch.long)

        for i, (img, label) in enumerate(dataset):
            data[i] = img  # img 已经是 transform 后的 tensor
            labels[i] = label
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return data, labels


def get_n_classes(dataset_name):
    if dataset_name == 'cifar10':
        return 10
    elif dataset_name == 'cifar100':
        return 100
    elif dataset_name == 'fmnist':
        return 10
    elif dataset_name == 'tinyimagenet':
        return 200
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

