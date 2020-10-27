from collections import OrderedDict
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import gc
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations import pytorch
from .randaug import RandAugment, to_tensor_randaug
from .albu import AlbuAugment, to_tensor_albu
import albumentations
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
import glob
import random

class ImageLabelDataset(Dataset):
    def __init__(self, images, label, mode='train', cfg=None):
        self.cfg = cfg
        self.images = images
        self.mode = mode
        assert self.cfg.DATA.TYPE in ("multiclass", "multilabel")
        assert self.mode in ("train", "valid", "test", "embeddings")
        if mode == "train":
            self.dir = self.cfg.DIRS.TRAIN_IMAGES
        elif mode == "valid" or mode == "embeddings":
            self.dir = self.cfg.DIRS.VALIDATION_IMAGES
        else:
            self.dir = self.cfg.DIRS.TEST_IMAGES

        if self.mode in ("train", "valid"):
            self.label = label

        assert self.cfg.DATA.AUGMENT in ("randaug", "albumentations")
        if self.cfg.DATA.AUGMENT == "randaug":
            self.transform = RandAugment(n=self.cfg.DATA.RANDAUG.N,
                m=self.cfg.DATA.RANDAUG.M, random_magnitude=self.cfg.DATA.RANDAUG.RANDOM_MAGNITUDE)
            self.to_tensor = to_tensor_randaug()
        elif self.cfg.DATA.AUGMENT == "albumentations":
            self.transform = AlbuAugment()
            self.to_tensor = to_tensor_albu()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        lb = None
        if self.mode in ("train", "valid"):
            lb = self.label[idx]
            if self.cfg.DATA.TYPE == "multilabel":
                lb = lb.astype(np.float32)
                if not isinstance(lb, list):
                    lb = [lb]
                lb = torch.Tensor(lb)
            second_lb = (lb > 0).astype("int")
            lb = (lb, second_lb)
        sop_study = self.images[idx].split('-')[0]
        sop_series = self.images[idx].split('-')[1]
        sop_instance = self.images[idx].split('-')[2]
        # image_name = [f for f in glob.glob(f"{self.dir}/{sop_study}/{sop_series}/*") if sop_instance in f][0]
        image_name = f"{self.dir}/{sop_study}/{sop_series}/{sop_instance}.png"
        image = Image.open(image_name)
        if self.cfg.DATA.INP_CHANNEL == 3:
            image = image.convert("RGB")
        elif self.cfg.DATA.INP_CHANNEL == 1:
            image = image.convert("L")
        if self.mode == "train":
            if isinstance(self.transform, AlbuAugment):
                image = np.asarray(image)
            image = self.transform(image)
            image = self.to_tensor(image)
            return image, lb
        elif self.mode == "valid":
            if isinstance(self.transform, AlbuAugment):
                image = np.asarray(image)
            image = self.to_tensor(image)
            return image, lb
        elif self.mode == "embeddings":
            if isinstance(self.transform, AlbuAugment):
                image = np.asarray(image)
            image = self.to_tensor(image)
            return image, sop_instance
        else:
            if isinstance(self.transform, AlbuAugment):
                image = np.asarray(image)
            image = self.to_tensor(image)
            return image, sop_instance


class EmbeddingsLabelDataset(Dataset):
    def __init__(self, images, label, mode='train', cfg=None):
        self.cfg = cfg
        self.images = images
        self.mode = mode
        assert self.cfg.DATA.TYPE in ("multiclass", "multilabel")
        assert self.mode in ("train", "valid", "test")
        if mode == "train":
            self.dir = self.cfg.DIRS.TRAIN_IMAGES
        elif mode == "valid":
            self.dir = self.cfg.DIRS.VALIDATION_IMAGES
        else:
            self.dir = self.cfg.DIRS.TEST_IMAGES

        if self.mode in ("train", "valid"):
            self.label = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        lb = None
        if self.mode in ("train", "valid"):
            lb = self.label[idx]
            if self.cfg.DATA.TYPE == "multilabel":
                lb = lb.astype(np.float32)
                if not isinstance(lb, list):
                    lb = [lb]
                lb = torch.Tensor(lb)
            second_lb = (lb > 0).astype("int")
            lb = (lb, second_lb)
        sop_instance = self.images[idx].split('-')[2]
        image_name = os.path.join(self.dir, sop_instance + '.npy')
        if random.random() < 0.5:
            image = np.load(image_name)
        else:
            image = np.load(image_name)[::-1,:]
        image = torch.from_numpy(image.copy()).unsqueeze(0)
        if self.mode in ("train", "valid"):
            return image, lb
        elif self.mode == "test":
            return image, sop_instance

class SeriesEmbeddingsLabelDataset(Dataset):
    def __init__(self, images, label, mode='train', cfg=None):
        self.cfg = cfg
        self.images = images
        self.mode = mode
        assert self.cfg.DATA.TYPE in ("multiclass", "multilabel")
        assert self.mode in ("train", "valid", "test")
        if mode == "train":
            self.dir = self.cfg.DIRS.TRAIN_IMAGES
        elif mode == "valid":
            self.dir = self.cfg.DIRS.VALIDATION_IMAGES
        else:
            self.dir = self.cfg.DIRS.TEST_IMAGES

        if self.mode in ("train", "valid"):
            self.label = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        lb = None
        if self.mode in ("train", "valid"):
            lb = self.label[idx]
            if self.cfg.DATA.TYPE == "multilabel":
                lb = lb.astype(np.float32)
                if not isinstance(lb, list):
                    lb = [lb]
                lb = torch.Tensor(lb)
        sop_instance = self.images[idx]
        image_name = os.path.join(self.dir, sop_instance + '.npy')
        try:
            image = np.load(image_name)
        except:
            print(self.images[idx])
        image = torch.from_numpy(image.copy()).unsqueeze(0)
        if self.mode in ("train", "valid"):
            return image, lb
        elif self.mode == "test":
            return image, sop_instance


def make_image_label_dataloader(cfg, mode, images, labels):
    dataset = ImageLabelDataset(images, labels, mode=mode, cfg=cfg)
    if cfg.DATA.DEBUG:
        dataset = Subset(dataset,
                         np.random.choice(np.arange(len(dataset)), 500))
    shuffle = True if mode == "train" else False
    dataloader = DataLoader(dataset, cfg.TRAIN.BATCH_SIZE,
                            pin_memory=False, shuffle=shuffle,
                            drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
    return dataloader


def make_embeddings_label_dataloader(cfg, mode, images, labels):
    dataset = EmbeddingsLabelDataset(images, labels, mode=mode, cfg=cfg)
    if cfg.DATA.DEBUG:
        dataset = Subset(dataset,
                         np.random.choice(np.arange(len(dataset)), 500))
    shuffle = True if mode == "train" else False
    dataloader = DataLoader(dataset, cfg.TRAIN.BATCH_SIZE,
                            pin_memory=False, shuffle=shuffle,
                            drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
    return dataloader


def make_series_embeddings_label_dataloader(cfg, mode, images, labels):
    dataset = SeriesEmbeddingsLabelDataset(images, labels, mode=mode, cfg=cfg)
    if cfg.DATA.DEBUG:
        dataset = Subset(dataset,
                         np.random.choice(np.arange(len(dataset)), 50))
    shuffle = True if mode == "train" else False
    dataloader = DataLoader(dataset, cfg.TRAIN.BATCH_SIZE,
                            pin_memory=False, shuffle=shuffle,
                            drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
    return dataloader
