from torchvision.transforms import Compose, Resize, Scale, ToTensor, CenterCrop, Normalize, RandomResizedCrop, RandomHorizontalFlip


def imagenet_normalize():
    return Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def imagenet_train_transforms():
    return Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        imagenet_normalize()
    ])


def imagenet_val_transforms():
    return Compose([
        Scale(256),
        CenterCrop(224),
        ToTensor(),
        imagenet_normalize,
    ])
