from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop, Normalize, RandomResizedCrop, RandomHorizontalFlip


train_transforms = Compose([
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize([0.500, 0.500, 0.500], [0.500, 0.500, 0.500])
])
val_transforms = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize([0.500, 0.500, 0.500], [0.500, 0.500, 0.500])
])
