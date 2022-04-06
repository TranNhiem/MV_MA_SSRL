# 2022 Tran, Harry, Josef (SSL Team)
# Dataloader Partially Inherence from 2021 solo-learn team development.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.



import os
import random
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Type, Union
from solo.utils.custom_augment_transforms import GaussianBlur, Solarization
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision import transforms as T
import numpy as np
import pickle
from torchvision.datasets import STL10, ImageFolder

# pluggin multiple DA support
from torchvision.transforms import autoaugment as auto_aug
from solo.utils.Fast_Auto_Augment.Fast_AutoAugment import Fast_AutoAugment
from solo.utils.Custom_RandAugment_v2 import Extended_RangAugment

def dataset_with_index(DatasetClass: Type[Dataset]) -> Type[Dataset]:
    """Factory for datasets that also returns the data index.

    Args:
        DatasetClass (Type[Dataset]): Dataset class to be wrapped.

    Returns:
        Type[Dataset]: dataset with index.
    """

    class DatasetWithIndex(DatasetClass):
        def __getitem__(self, index):
            data = super().__getitem__(index)
            return (index, *data)

    return DatasetWithIndex


class CustomDatasetWithoutLabels(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.images = os.listdir(root)

    def __getitem__(self, index):
        path = self.root / self.images[index]
        x = Image.open(path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x, -1

    def __len__(self):
        return len(self.images)

class NCropAugmentation:
    def __init__(self, transform: Callable, num_crops: int):
        """Creates a pipeline that apply a transformation pipeline multiple times.

        Args:
            transform (Callable): transformation pipeline.
            num_crops (int): number of crops to create from the transformation pipeline.
        """

        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """
        # this case self num_crop only
        # the same
        return [self.transform(x) for _ in range(self.num_crops)]

    def __repr__(self) -> str:
        return f"{self.num_crops} x [{self.transform}]"

class FullTransformPipeline:
    def __init__(self, transforms: Callable) -> None:
        self.transforms = transforms

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        out = []
        for idx, transform in enumerate(self.transforms):
            out.extend(transform(x))
        
        #random.shuffle(out)
        
        return out

    def __repr__(self) -> str:
        return "\n".join([str(transform) for transform in self.transforms])

class FullTransformPipeline_v1:
    def __init__(self, transforms: Callable) -> None:
        self.transforms = transforms

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.
        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """
        # Default is Inception Style Cropping 
        crop_inception=T.Compose([T.RandomResizedCrop(size=224,
             interpolation=T.InterpolationMode.BICUBIC)])
        x1 = crop_inception(x)
        #x1=PIL.Image.fromarray(np.uint8(x1))
        x2 = crop_inception(x)
        #x2=PIL.Image.fromarray(np.uint8(x2))
        out = []
        for idx, transform in enumerate(self.transforms):
            out.extend(transform(x1))
            out.extend(transform(x2))
        random.shuffle(out)
      
        return out

    def __repr__(self) -> str:
        return "\n".join([str(transform) for transform in self.transforms])

class FullTransformPipeline_ma_mv:
    def __init__(self, transforms: Callable, num_crops_glob: int, crop_size_glob: int,
                    num_crops_loc: int, crop_size_loc: int, crop_type: str, 
                    min_scale_loc=0.1, max_scale_loc=0.34,  min_scale_glob=0.3, max_scale_glob=1.0) -> None:
        
        self.transforms = transforms
        self.num_crop_glob = num_crops_glob
        self.num_crop_loc = num_crops_loc
        self.crop_size_glob= crop_size_glob
        self.crop_size_loc= crop_size_loc
        self.crop_type= crop_type
        self.min_scale_loc= min_scale_loc
        self.max_scale_loc= max_scale_loc
        self.min_scale_glob= min_scale_glob
        self.max_scale_glob= max_scale_glob

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """
        # Try to generate Crop for 2
        mean = (0.485, 0.456, 0.406)
        std = (0.228, 0.224, 0.225)

        x_glob_crops=[]
        for _ in range(self.num_crop_glob): 
            
            if self.crop_type == "inception_crop": 
                crop_strategy=T.Compose([T.RandomResizedCrop(size=self.crop_size_glob,
                    interpolation=T.InterpolationMode.BICUBIC),])# transforms.Normalize(mean=mean, std=std)
            
            elif self.crop_type == "random_uniform":
                    crop_strategy=T.Compose([T.RandomResizedCrop(size=self.crop_size_glob,
                    scale=(self.min_scale_glob, self.max_scale_glob),
                        interpolation=T.InterpolationMode.BICUBIC)])
            else: 
                raise ValueError("Croping_strategy_Invalid")
            crop_view = crop_strategy(x)
            x_glob_crops.append(crop_view)
            torch.save(x,"orginal_image")
            torch.save(x_glob_crops, "crops_tensor",  pickle_module=pickle)

        x_loc_crops=[]

        for _ in range(self.num_crop_loc): 
            
            if self.crop_type == "inception_crop": 
                crop_strategy=T.Compose([T.RandomResizedCrop(size=self.crop_size_loc,
                    interpolation=T.InterpolationMode.BICUBIC), ])
            
            elif self.crop_type == "random_uniform":
                    crop_strategy=T.Compose([T.RandomResizedCrop(size=self.crop_size_loc,
                    scale=(self.min_scale_loc, self.max_scale_loc),
                        interpolation=T.InterpolationMode.BICUBIC), ])
            else: 
                raise ValueError("Croping_strategy_Invalid")
            crop_view = crop_strategy(x)
            x_loc_crops.append(crop_view)

        out = []
        if len(x_glob_crops) & len( x_loc_crops) >= 1: 
            print("Gloabl ^&^ Local Crops Apply Transform")
            out_glob=[]
            for x_glob in x_glob_crops:
                for idx, transform in enumerate(self.transforms):
                    out_glob.extend(transform(x_glob))
            #random.shuffle(out_glob)
            out.extend(out_glob)
            
            out_loc=[]
            for x_loc in x_loc_crops:
                for idx, transform in enumerate(self.transforms):
                    out_loc.extend(transform(x_loc))
            random.shuffle(out_loc)
            out.extend(out_loc)
        
        elif len( x_loc_crops) ==0:  
            print("Croping with Only Global Crop")
            out_glob=[]
            for x_glob in x_glob_crops:
                for idx, transform in enumerate(self.transforms):
                    out_glob.extend(transform(x_glob))
            #random.shuffle(out_glob)
            out.extend(out_glob)

        else: 
            raise ValueError("Croping should have num_glob_crop & num_loc_crop")
        
        return out

    def __repr__(self) -> str:
        return "\n".join([str(transform) for transform in self.transforms])


def prepare_n_crop_transform(
    transforms: List[Callable], num_crops_per_aug: List[int]
) -> NCropAugmentation:
    """Turns a single crop transformation to an N crops transformation.

    Args:
        transforms (List[Callable]): list of transformations.
        num_crops_per_aug (List[int]): number of crops per pipeline.

    Returns:
        NCropAugmentation: an N crop transformation.
    """

    assert len(transforms) == len(num_crops_per_aug)

    T = []
    for transform, num_crops in zip(transforms, num_crops_per_aug):
        T.append(NCropAugmentation(transform, num_crops))
    return FullTransformPipeline(T)
    

def prepare_n_crop_transform_v1(
    transforms: List[Callable], num_crops_per_aug: List[int], 
) -> NCropAugmentation:
    """Turns a single crop transformation to an N crops transformation.

    Args:
        transforms (List[Callable]): list of transformations.
        num_crops_per_aug (List[int]): number of crops per pipeline.

    Returns:
        NCropAugmentation: an N crop transformation.
    """

    assert len(transforms) == len(num_crops_per_aug)

    T = []
    for transform, num_crops in zip(transforms, num_crops_per_aug):
        T.append(NCropAugmentation(transform, num_crops))
    return FullTransformPipeline_v1(T)


## stable version
def prepare_n_crop_transform_mv_ma(
    transforms: List[Callable], num_crops_per_aug: List[int],  num_crop_glob:int, crop_size_glob: int,num_crop_loc:int, crop_size_loc: int
                                ,crop_type: str, min_loc: float =0.1, max_loc: float=0.34,
                                 min_glob: float=0.3, max_glob: float=1.0
) -> NCropAugmentation:
    """Turns a single crop transformation to an N crops transformation.

    Args:
        transforms (List[Callable]): list of transformations.
        num_crops_per_aug (List[int]): number of crops per pipeline.

    Returns:
        NCropAugmentation: an N crop transformation.
    """
    print("len transform", len(transforms))
    print("len num_crops_per_aug", len(num_crops_per_aug))
    assert len(transforms) == len(num_crops_per_aug)
    

    T = []
    for transform, num_crops in zip(transforms, num_crops_per_aug):
        T.append(NCropAugmentation(transform, num_crops))


    return FullTransformPipeline_ma_mv(T, num_crop_glob, crop_size_glob,num_crop_loc, crop_size_loc
                                ,crop_type,min_scale_loc=min_loc, max_scale_loc=max_loc,
                                 min_scale_glob=min_glob, max_scale_glob=max_glob )

class BaseTransform:
    """Adds callable base class to implement different transformation pipelines."""

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(x)

    def __repr__(self) -> str:
        return str(self.transform)

class ImagenetTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 224,
    ):
        """Class that applies Imagenet transformations.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 224.
        """

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        )

class CustomTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 224,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.228, 0.224, 0.225),
    ):
        """
        Class that applies Custom transformations.
        If you want to do exoteric augmentations, you can just re-write this class.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 224.
            mean (Sequence[float], optional): mean values for normalization.
                Defaults to (0.485, 0.456, 0.406).
            std (Sequence[float], optional): std values for normalization.
                Defaults to (0.228, 0.224, 0.225).
        """

        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    crop_size,
                    scale=(min_scale, max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

class CustomTransform_no_crop(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        color_jitter_prob: float = 0.8,
        gray_scale_prob: float = 0.2,
        horizontal_flip_prob: float = 0.5,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        crop_size: int = 224,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.228, 0.224, 0.225),
    ):
        """Class that applies Custom transformations.
        If you want to do exoteric augmentations, you can just re-write this class.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            color_jitter_prob (float, optional): probability of applying color jitter.
                Defaults to 0.8.
            gray_scale_prob (float, optional): probability of converting to gray scale.
                Defaults to 0.2.
            horizontal_flip_prob (float, optional): probability of flipping horizontally.
                Defaults to 0.5.
            gaussian_prob (float, optional): probability of applying gaussian blur.
                Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization.
                Defaults to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
            crop_size (int, optional): size of the crop. Defaults to 224.
            mean (Sequence[float], optional): mean values for normalization.
                Defaults to (0.485, 0.456, 0.406).
            std (Sequence[float], optional): std values for normalization.
                Defaults to (0.228, 0.224, 0.225).
        """

        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=color_jitter_prob,
                ),
                transforms.RandomGrayscale(p=gray_scale_prob),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

def prepare_transform(dataset: str, trfs_kwargs, da_kwargs=None) -> Any:
    """Prepares transforms for a specific dataset. Optionally uses multi crop.

    Args:
        dataset (str): name of the dataset.

    Returns:
        Any: a transformation for a specific dataset.
    """

   
    if dataset in ["imagenet", "imagenet100"]:
        return ImagenetTransform(**trfs_kwargs)
    elif dataset == "custom":
        return CustomTransform(**trfs_kwargs)

    # pluggin proposed multiple-DA
    
    elif dataset == "mulda":
        policy_dict = {'imagenet':auto_aug.AutoAugmentPolicy.IMAGENET}
        ## DA args def :
        num_ops, magnitude = da_kwargs['rda_num_ops'], da_kwargs['rda_magnitude']
        ada_policy = policy_dict[ da_kwargs['ada_policy'] ]
        fda_policy = da_kwargs['fda_policy']
        # common crop settings : 
        mean = (0.485, 0.456, 0.406)
        std = (0.228, 0.224, 0.225)

        rnd_crp = transforms.RandomResizedCrop(
            trfs_kwargs['crop_size'],
            scale=(trfs_kwargs['min_scale'], trfs_kwargs['max_scale']),
            interpolation=transforms.InterpolationMode.BICUBIC
        )
    
        # prepare various da
        auto_da = transforms.Compose( [rnd_crp, auto_aug.AutoAugment(policy=ada_policy), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std) ] )
        
        rand_da = transforms.Compose( [rnd_crp, auto_aug.RandAugment(num_ops=num_ops, magnitude=magnitude), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)] )
        #rand_da = transforms.Compose( [rnd_crp, Extended_RangAugment(num_ops=num_ops, magnitude=magnitude), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)] )
        
        #fast_da = Fast_AutoAugment(policy_type=fda_policy).get_trfs(rnd_crp)
        fast_da = transforms.Compose( [rnd_crp, Fast_AutoAugment(policy_type=fda_policy).get_trfs(), transforms.Normalize(mean=mean, std=std)] )

        
        #  ret [simclr_da, rand_da, auto_da, fast_da]  4 views trfs
        return [CustomTransform(**trfs_kwargs), rand_da, auto_da, fast_da ]#fast_da
    
    elif dataset == "mulda_v1":
        """
        mulda_v1 --> is the version removing Random Crop. 
        The cropping is Performed in FullTransformPipeline_v1
        x--> X1, X2 --> as the result each Augmentations also Generate two version
        --> Current Implementation X--> X1, X2 --> Then Apply augmentation
        """

        policy_dict = {'imagenet':auto_aug.AutoAugmentPolicy.IMAGENET}
        ## DA args def :
        mean = (0.485, 0.456, 0.406)
        std = (0.228, 0.224, 0.225)
        num_ops, magnitude = da_kwargs['rda_num_ops'], da_kwargs['rda_magnitude']
        ada_policy = policy_dict[ da_kwargs['ada_policy'] ]
        fda_policy = da_kwargs['fda_policy']
        # common crop settings : 

        # prepare various da
        auto_da = transforms.Compose( [ auto_aug.AutoAugment(policy=ada_policy), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)] )
        
        rand_da = transforms.Compose( [auto_aug.RandAugment(num_ops=num_ops, magnitude=magnitude), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)] )
        #ssrand_da = transforms.Compose( [ Extended_RangAugment(num_ops=num_ops, magnitude=magnitude), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)] )
        
        #fast_da = Fast_AutoAugment(policy_type=fda_policy).get_trfs()
        fast_da = transforms.Compose( [Fast_AutoAugment(policy_type=fda_policy).get_trfs(), transforms.Normalize(mean=mean, std=std)] )
        
        #  ret [simclr_da, rand_da, auto_da, fast_da]  4 views trfs
        return [ CustomTransform_no_crop(**trfs_kwargs), rand_da, auto_da, fast_da ]#fast_da
     
    elif dataset =="mv_ma": 
        """
        mv_ma --> is the version removing Random Crop. 
        The cropping is Performed in FullTransformPipeline
        x--> X1, X2, Xn as NCropAugmentation_mv_ma function (args: num_crops, crop_size, crop_type, scale_of_crops)
        --> Current Implementation X--> X1, X2 --> Then Apply augmentation
        """
        mean = (0.485, 0.456, 0.406)
        std = (0.228, 0.224, 0.225)
        policy_dict = {'imagenet':auto_aug.AutoAugmentPolicy.IMAGENET}
        ## DA args def :
        num_ops, magnitude = da_kwargs['rda_num_ops'], da_kwargs['rda_magnitude']
        ada_policy = policy_dict[ da_kwargs['ada_policy'] ]
        fda_policy = da_kwargs['fda_policy']
        # common crop settings : 

        # prepare various da
        auto_da = transforms.Compose( [ auto_aug.AutoAugment(policy=ada_policy), transforms.ToTensor(),transforms.Normalize(mean=mean, std=std)])# transforms.ToTensor(),# transforms.Normalize(mean=mean, std=std)
        
        rand_da = transforms.Compose( [auto_aug.RandAugment(num_ops=num_ops, magnitude=magnitude), transforms.ToTensor(),transforms.Normalize(mean=mean, std=std)])# transforms.ToTensor()] )
        #rand_da = transforms.Compose( [Extended_RangAugment(num_ops=num_ops, magnitude=magnitude),transforms.Normalize(mean=mean, std=std)] )#
        
        fast_da = transforms.Compose( [Fast_AutoAugment(policy_type=fda_policy).get_trfs(), transforms.Normalize(mean=mean, std=std)] )
        
        #  ret [simclr_da, rand_da, auto_da, fast_da]  4 views trfs
        return [ CustomTransform_no_crop(**trfs_kwargs), rand_da, auto_da, fast_da,]#fast_da
        
    else:
        raise ValueError(f"{dataset} is not currently supported.")

def prepare_datasets(
    dataset: str,
    transform: Callable,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    subset_class_num: int =1000, 
    no_labels: Optional[Union[str, Path]] = False,
    download: bool = True,
) -> Dataset:
    """Prepares the desired dataset.

    Args:
        dataset (str): the name of the dataset.
        transform (Callable): a transformation.
        data_dir (Optional[Union[str, Path]], optional): the directory to load data from.
            Defaults to None.
        train_dir (Optional[Union[str, Path]], optional): training data directory
            to be appended to data_dir. Defaults to None.
        no_labels (Optional[bool], optional): if the custom dataset has no labels.

    Returns:
        Dataset: the desired dataset with transformations.
    """
    if data_dir is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        data_dir = sandbox_folder / "datasets"

    if train_dir is None:
        train_dir = Path(f"{dataset}/train")
    else:
        train_dir = Path(train_dir)


    if dataset in ["imagenet", "imagenet100"]:
        train_dir = data_dir / train_dir
        train_dataset = dataset_with_index(ImageFolder)(train_dir, transform)

    elif dataset == "custom":
        train_dir = data_dir / train_dir

        if no_labels:
            dataset_class = CustomDatasetWithoutLabels
        else:
            dataset_class = ImageFolder

        train_dataset = dataset_with_index(dataset_class)(train_dir, transform)
    
    ## pluggin support code snippet
    #  only support imagenet/imagenet100 ds
    elif dataset == "mulda" or "mulda_v1" or "mv_ma":
        train_dir = data_dir / train_dir
        train_dataset = dataset_with_index(ImageFolder)(train_dir, transform)

    return train_dataset

def prepare_dataloader(
    train_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> DataLoader:
    """Prepares the training dataloader for pretraining.
    Args:
        train_dataset (Dataset): the name of the dataset.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 4.
    Returns:
        DataLoader: the training dataloader with the desired dataset.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader



# Note that, the code snippet below can not be directly executed
if __name__ == "__main__":
    # Seems the multi-view work almost done by combining Transform & Dataset
    from pretrain_dataloader_v2 import Imagenet_DataModule

    params = {'data_dir':'/data', 'batch_size':64}
    imgnet_ds = Imagenet_DataModule(**params)

    # i keep valid_transform, but it should be the same way to assign the trfs.
    imgnet_ds.train_transform = prepare_transform(dataset="mv_ma")

    # confirm the multi-view transform
    print(f"train trfs : {imgnet_ds.train_transform}\n")
    print(f"valid trfs : {imgnet_ds.valid_transform}\n")

    ## After we confirm this part, we can put it into prepare_dataset function :
    '''
    # 
    ... 

    elif dataset == "mulda" or "mulda_v1" or "mv_ma":
        train_dir = data_dir / train_dir
        imgnet_ds = Imagenet_DataModule(data_dir=train_dir)
        imgnet_ds.train_transform = transform

        return imgnet_ds.train_dataloader()

    '''
