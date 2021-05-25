from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop, Normalize, RandomResizedCrop, RandomHorizontalFlip
from PIL.ImageFile import ImageFile
from typing import Callable
from torch import Tensor

_train_transforms = Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.500, 0.500, 0.500], [0.500, 0.500, 0.500])
    ])

def train_transforms(x: ImageFile) -> Tensor:
    return _train_transforms(x)

val_transforms = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize([0.500, 0.500, 0.500], [0.500, 0.500, 0.500])
])
