# software-engerneering pkgs
from attr import attrs, attrib
from typing import Any, Callable, List, Optional, Sequence, Type, Union

# other pkgs
import os
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps

## torch eco-system
from pytorch_lightning import LightningDataModule

#  dataset, trfs
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms

from torch.utils.data import DataLoader

# make use of attrs to support invarient-programming (args validator)
@attrs
class Imagenet_DataModule(LightningDataModule):
    # dataset path cfg
    data_dir = attrib(type=str, default='')
    tra_dir = attrib(type=str, default='train')
    val_dir = attrib(type=str, default='val')
    download = attrib(type=bool, default=False)
    
    # data processing cfg
    num_workers = attrib(type=int, default=4)
    pin_memory = attrib(type=bool, default=True)
    
    # data loading cfg
    batch_size = attrib(type=int, default=256)
    shuffle = attrib(type=bool, default=True)
    drop_last = attrib(type=bool, default=False)

    def __attrs_post_init__(self, *args: Any, **kwargs: Any) -> None:
        
        def chk_path(ds_path):
            if not isinstance(ds_path, Path):
                raise TypeError(f"The given { str(ds_path) } is not the Path data type")
            if not ds_path.exists():
                raise ValueError(f"The given data_path : { str(ds_path) } is not the exists")

        super().__init__(*args, **kwargs)
        self.tra_path = Path(self.data_dir).joinpath(self.tra_dir)
        self.val_path = Path(self.data_dir).joinpath(self.val_dir)
        chk_path(self.tra_path)
        chk_path(self.val_path)

        # setup default transform
        self._norm_trfs = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._train_trfs = self._valid_trfs = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self._norm_trfs,
            ]
        )
        self.setup()
    
    ## Declared properties
    @property
    def num_classes(self) -> int:
        return 1000

    @property
    def normalize_transform(self):
        return self._norm_trfs

    @property
    def train_transform(self):
        return self._train_trfs

    @train_transform.setter
    def train_transform(self, new_trfs):
        self._train_trfs = new_trfs

    @property
    def valid_transform(self):
        return self._valid_trfs

    @valid_transform.setter
    def valid_transform(self, new_trfs):
        self._valid_trfs = new_trfs

    ## Override built-in methods
    def prepare_data(self):
        """
        There is no official support to download the ImageNet dataset.
        So, we perform an experimental download by wget & unzip cmd.
        Please make sure this run on Linux-based system.
        """
        base_url = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_{}.tar"
        dwn_tar = [self.tra_dir, self.val_dir]
        if self.download:
            for tar in dwn_tar:
                target_url = base_url.format(tar)
                os.system(f"wget {target_url} -P { str(self.ds_path) } | tar -xvf")

    def setup(self):
        """
        Loads in data from file and prepares PyTorch tensor datasets for each split (train, val).
        Note this runs across all GPUs and it is safe to make state assignments here
        """
        class DatasetWithIndex(ImageFolder):
            def __getitem__(self, index):
                data = super().__getitem__(index)
                return (index, *data)
        # ImageNet already have split the train/valid subset (we treat valid as the test set in general)
        self.train_dataset = DatasetWithIndex(self.tra_path, self._train_trfs)
        self.valid_dataset = DatasetWithIndex(self.val_path, self._valid_trfs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def valid_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)
    

if __name__ == "__main__":
    params = {'data_dir':'/data', 'batch_size':64}
    imgnet_ds = Imagenet_DataModule(**params)

    print(f"train trfs : {imgnet_ds.train_transform}\n")
    print(f"valid trfs : {imgnet_ds.valid_transform}\n")

    for idx, *ims in imgnet_ds.train_dataloader():
        print(f"batch idx : {idx[-1]}\n")
        im_shape = ims[0].shape
        print(f"batch size : {im_shape[0]}\n")
        print(f"im shape : {im_shape[1:]}\n")
