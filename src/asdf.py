from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop, Normalize, RandomResizedCrop, RandomHorizontalFlip


def train_transforms():
    return Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.500, 0.500, 0.500], [0.500, 0.500, 0.500])
    ])
