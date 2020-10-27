from torchvision import transforms
import torch
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import random
import numpy as np
import torchvision.transforms.functional as TF
import random

from albumentations import Compose, HorizontalFlip, VerticalFlip, Normalize, Resize, \
            RandomResizedCrop, MotionBlur, RandomBrightness, RandomContrast, OneOf, RandomGamma, \
            CLAHE, Blur, CoarseDropout, GaussNoise, HueSaturationValue, ShiftScaleRotate, Transpose, \
            ElasticTransform, IAAAdditiveGaussianNoise, IAASharpen, RandomBrightnessContrast, \
            GridDistortion, RandomResizedCrop, OpticalDistortion, NoOp, MedianBlur, IAAEmboss
import albumentations.augmentations.functional as AF
from albumentations.pytorch import ToTensor
from albumentations.core.transforms_interface import ImageOnlyTransform

class VerticalCut(ImageOnlyTransform):
    def apply(self, image, fill_value=0, holes=(), **params):
        return AF.cutout(image, holes, fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]

        holes = []
        col = random.randint(0, 4)
        col_width = int(width // 5)
        holes.append((col*col_width, 0, (col+1)*col_width, height))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("num_holes", "max_h_size", "max_w_size")

class AlbuAugment:
    def __init__(self):
        transformation = []
        transformation += [
            RandomResizedCrop(512, 512, scale=(0.8, 1.0)),
            ShiftScaleRotate(),
            OneOf([
                GridDistortion(),
                OpticalDistortion(),
                ElasticTransform(approximate=True)], p=0.8),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
                MedianBlur(blur_limit=3),
                Blur(blur_limit=3)], p=0.8),
            CoarseDropout(max_holes=2, max_height=32, max_width=32)
        ]

        self.transform = Compose(transformation)

    def __call__(self, x):
        return self.transform(image=x)['image']

class to_tensor_albu:
    def __init__(self):
        transformation = []
        transformation += [Normalize(),
                           ToTensor()]
        self.transform = Compose(transformation)

    def __call__(self, x):
        return self.transform(image=x)['image']