import numpy as np
import torch
from torchvision import datasets, transforms
from config import NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS


# ToTensor() converts PIL image to tensor and scales pixel values to [0, 1], reorders dimensions to (channel, height, width)
def get_dataloaders(BATCH_SIZE):
    
    train_transform =  transforms.Compose([
        transforms.RandomHorizontalFlip(),      # data augmentation
        transforms.RandomCrop(32, padding=4),    # data augmentation
        transforms.ToTensor(),                    
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    test_transform =  transforms.Compose([
        transforms.ToTensor(),                    
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])


    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )

    return train_loader, test_loader