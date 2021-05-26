from src.transforms import imagenet_train_transforms, imagenet_val_transforms, imagenet_normalize
from src.main import run, wrapped_run
from src.data import example_image
from src.classifier import Classifier


def lightning_classifier(**kwargs):
    return Classifier(**kwargs)
