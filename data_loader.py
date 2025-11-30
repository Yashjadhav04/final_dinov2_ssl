from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from augmentations import get_simclr_augmentations

def get_dataloaders(data_path, batch_size=128, num_workers=4):
    transform = get_simclr_augmentations()

    train_dataset = ImageFolder(
        root=data_path,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    return train_loader
