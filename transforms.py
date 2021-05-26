from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop, Normalize, RandomResizedCrop, RandomHorizontalFlip
# from PIL.ImageFile import ImageFile
# from typing import Callable
# from torch import Tensor

def train_transforms(**kwargs):
    return Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.500, 0.500, 0.500], [0.500, 0.500, 0.500])
    ])

# val_transforms = Compose([
#     Resize(224),
#     CenterCrop(224),
#     ToTensor(),
#     Normalize([0.500, 0.500, 0.500], [0.500, 0.500, 0.500])
# ])
