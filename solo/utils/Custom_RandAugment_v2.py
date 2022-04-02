# 2022/04/01, 5:15 pm, tech support : Harry, Josef
import torch
from torchvision.transforms import RandAugment, InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms import ColorJitter, Grayscale, GaussianBlur

# typing check
from typing import List, Tuple, Optional, Dict
from torch import Tensor

## The original data augmentation of SimCLR repository 
#  plz referes : https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
class Extended_RangAugment(RandAugment):
    ## public interface
    def __init__(self, num_ops: int = 2, magnitude: int = 9, num_magnitude_bins: int = 31, 
                        interpolation: InterpolationMode = InterpolationMode.NEAREST, fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__(num_ops, magnitude, num_magnitude_bins, interpolation, fill)

    def _ext_apply_op(self, img, op_name, magnitude, interpolation, fill):
        # extended version of data aug and we allow the independent search :
        #   [Color_jitter (brightness, contrast, saturation, hue), 
        #                 Gaussian_blur, Gray_scale (rgb2gray)]
        if op_name == "rand_brightness":
            brightness = ColorJitter(brightness=magnitude, contrast=0, saturation=0, hue=0)
            img = brightness(img)

        elif op_name == "rand_contrast":
            contrast = ColorJitter(brightness=0, contrast=magnitude, saturation=0, hue=0)
            img = contrast(img)

        elif op_name == "rand_saturation":
            saturation = ColorJitter(brightness=0, contrast=0, saturation=magnitude, hue=0)
            img = saturation(img)

        elif op_name == "rand_hue":
            hue = ColorJitter(brightness=0, contrast=0, saturation=0, hue=magnitude)
            img = hue(img)

        elif op_name == "rand_gaussian_blur":
            blur = GaussianBlur(kernel_size=int(magnitude), sigma=(0.1, 2.0))
            img = blur(img)

        elif op_name == "rand_gray_scale": 
            gray_scale = Grayscale() # it doesn't have the params to search, only random apply
            img = gray_scale(img)    # you can consider to remove it..

        ## Original transformation space :
        elif op_name == "ShearX":
            # magnitude should be arctan(magnitude)
            # official autoaug: (1, level, 0, 0, 1, 0)
            # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
            # compared to
            # torchvision:      (1, tan(level), 0, 0, 1, 0)
            # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
            img = F.affine(
                img,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[math.degrees(math.atan(magnitude)), 0.0],
                interpolation=interpolation,
                fill=fill,
                center=[0, 0],
            )
        elif op_name == "ShearY":
            # magnitude should be arctan(magnitude)
            # See above
            img = F.affine(
                img,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[0.0, math.degrees(math.atan(magnitude))],
                interpolation=interpolation,
                fill=fill,
                center=[0, 0],
            )
        elif op_name == "TranslateX":
            img = F.affine(
                img,
                angle=0.0,
                translate=[int(magnitude), 0],
                scale=1.0,
                interpolation=interpolation,
                shear=[0.0, 0.0],
                fill=fill,
            )
        elif op_name == "TranslateY":
            img = F.affine(
                img,
                angle=0.0,
                translate=[0, int(magnitude)],
                scale=1.0,
                interpolation=interpolation,
                shear=[0.0, 0.0],
                fill=fill,
            )
        elif op_name == "Rotate":
            img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
        elif op_name == "Brightness":
            img = F.adjust_brightness(img, 1.0 + magnitude)
        elif op_name == "Color":
            img = F.adjust_saturation(img, 1.0 + magnitude)
        elif op_name == "Contrast":
            img = F.adjust_contrast(img, 1.0 + magnitude)
        elif op_name == "Sharpness":
            img = F.adjust_sharpness(img, 1.0 + magnitude)
        elif op_name == "Posterize":
            img = F.posterize(img, int(magnitude))
        elif op_name == "Solarize":
            img = F.solarize(img, magnitude)
        elif op_name == "AutoContrast":
            img = F.autocontrast(img)
        elif op_name == "Equalize":
            img = F.equalize(img)
        elif op_name == "Invert":
            img = F.invert(img)
        elif op_name == "Identity":
            pass
        else:
            raise ValueError(f"The provided operator {op_name} is not recognized.")
        return img


    def _ext_aug_space(num_bins, image_size):
        return {
            # make your new extension..
            "rand_brightness": (torch.linspace(0.0, 0.8, num_bins), False),
            "rand_contrast": (torch.linspace(0.0, 0.8, num_bins), False),
            "rand_saturation": (torch.linspace(0.0, 0.8, num_bins), False),
            "rand_hue": (torch.linspace(0.0, 0.5, num_bins), False),
            # I think 11x11 filter map is enough as maximum value,
            #   and 1x1 filter map is capable to keep the orignal info of img.
            "rand_gaussian_blur": (torch.linspace(1.0, 11.0, num_bins), False), 
            "rand_gray_scale": (torch.tensor(0.0), False),
            
            # original trfs..
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
    
    def forward(self, img):
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._ext_aug_space(self.num_magnitude_bins, (height, width))
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = self._ext_apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img


if __name__ == "__main__":
    rand = Extended_RangAugment()
    print(rand._augmentation_space(num_bins=31, image_size=[224, 224, 3]))

    
