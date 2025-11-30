import torchvision.transforms as T

def get_simclr_augmentations(size=224):
    return T.Compose([
        T.RandomResizedCrop(size=size),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
    ])
