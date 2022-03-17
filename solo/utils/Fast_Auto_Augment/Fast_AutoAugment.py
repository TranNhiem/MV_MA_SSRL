# Fast Auto Augument
import torch
import torchvision
from torchvision.transforms import transforms
from .searched_policies import fa_reduced_cifar10, fa_resnet50_rimagenet, fa_reduced_svhn
from .transform_table import augment_list
import random  


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies
        self.augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = self.apply_augment(img, name, level)
        return img
    
    def apply_augment(self, img, name, level):
        augment_fn, low, high = self.augment_dict[name]
        return augment_fn(img.copy(), level * (high - low) + low) 


class Fast_AutoAugment(object):

    def __init__(self, policy_type="imagenet"):
        # preprocess..
        mean = (0.485, 0.456, 0.406)
        std = (0.228, 0.224, 0.225)
        self.trfs_cntr = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        if policy_type == "imagenet":
            ds_policies = Augmentation(fa_resnet50_rimagenet())
        elif policy_type == "redu_cifar10": 
            ds_policies = Augmentation(fa_reduced_cifar10())
        elif policy_type == "redu_svhn":
            ds_policies = Augmentation(fa_reduced_svhn())
        else:
            raise ValueError("The policies of indicated dataset have not been searched")
        
        self.policy_type = policy_type
        self.trfs_cntr.transforms.insert(0, ds_policies)

    def prnt_policies(self):
        if self.policy_type == "imagenet":
            ds_policies = fa_resnet50_rimagenet()
        elif self.policy_type == "redu_cifar10": 
            ds_policies = fa_reduced_cifar10()
        elif self.policy_type == "redu_svhn":
            ds_policies = fa_reduced_svhn()

        return ds_policies

    def get_trfs(self):
        return self.trfs_cntr


# sample code snippet..
if __name__ == '__main__':
    import numpy as np
    img = np.random.random((14, 14, 3))

    fa = Fast_AutoAugment()
    #print(fa.prnt_policies()[0])
    print(fa.distort(img))