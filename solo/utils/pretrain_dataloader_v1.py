import os
import random
import glob
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Type, Union
import numpy as np

import torch
import torchvision
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder
import torchvision.transforms.functional as TF

from torchvision.utils import save_image
from torchvision.ops import masks_to_boxes

from pytorch_lightning import LightningDataModule


class SSL_DataModule(LightningDataModule):

    def __init__(self,
                 data_dir: str,
                 meta_dir: Optional[str] = None,
                 num_workers: int = 4,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 *args: Any,
                 **kwargs: Any,) -> None:
        super().__init__(*args, **kwargs)

        """MNCRL Dataloader Module.
        Args:
            dl_path: root directory where to download the data
            num_workers: number of CPU workers
            batch_size: number of sample in a batch
        """
        super().__init__()

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.meta_dir = meta_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    @property
    def num_classes(self) -> int:
        return 1000

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    def prepare_data(self):
        """
        prepare_data()
        This is where we can download the dataset. We point to our desired dataset.
        Note we do not make any state assignments in this function (i.e. self.something = ...)
        """

    def setup(self):
        """
        Loads in data from file and prepares PyTorch tensor datasets for each split (train, val, test).
        Setup expects a 'stage' arg which is used to separate logic for 'fit' and 'test'.(Optional for now)
        If you don't mind loading all your datasets at once, 
        you can set up a condition to allow for both 'fit' related setup and 'test' related setup to run whenever None is passed to stage (or ignore it altogether and exclude any conditionals).
        Note this runs across all GPUs and it is safe to make state assignments here
        """

    def create_dataset(self, root, transform):
        return ImageFolder(root=root, transform=transform)


    def train_dataloader(self,):
        """
        train_dataloader(), val_dataloader(), and test_dataloader() 
        all return PyTorch DataLoader instances that are created by wrapping their respective datasets that we prepared in setup()
        """
        pass

    def val_dataloader(self,):
        pass
