## Rick & Joshep RandAugment V3 Stable Version Implementation 

import torch 
from typing import List, Tuple, Optional, Dict

import math
from enum import Enum
from torch import Tensor
from torchvision.transforms import  InterpolationMode
#from torch.nn import functional as F
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter, Grayscale, GaussianBlur, ToTensor, ToPILImage

## Special Argument intput for GaussianBlur
sigma: Tuple[int, int] = (0.1, 2.0)

def _apply_op(
    img: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):

    ## Editting Operation Transform
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
        # (option) since our kernel map range is safe, it could be +1 to enlarge the kernel effect 
        odd_mag = int(magnitude) if int(magnitude) % 2 == 1 else int(magnitude) - 1
        blur = GaussianBlur(kernel_size=odd_mag, sigma=sigma)
        img = blur(img)

    elif op_name == "rand_gray_scale": 
        gray_scale = Grayscale(num_output_channels=3) # it doesn't have the params to search, only random apply
        img = gray_scale(img)    # you can consider to remove it..

    
    
    
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
            #center=[0, 0],
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
           # center=[0, 0],
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
            #center=[0, 0],
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
           # center=[0, 0],
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
    
class RandAugment(torch.nn.Module):
    """RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int, image_size: List[int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),

            # make your new extension..
            "rand_brightness": (torch.linspace(0.0, 0.8, num_bins), False),
            "rand_contrast": (torch.linspace(0.0, 0.8, num_bins), False),
            "rand_saturation": (torch.linspace(0.0, 0.8, num_bins), False),
            "rand_hue": (torch.linspace(0.0, 0.5, num_bins), False),
            # I think 11x11 filter map is enough as maximum value,
            #   and 1x1 filter map is capable to keep the orignal info of img.
            "rand_gaussian_blur": (torch.linspace(1.0, 11.0, num_bins), False), 
            "rand_gray_scale": (torch.tensor(0.0), False),
            
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
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

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        for _ in range(self.num_ops):
            op_meta = self._augmentation_space(self.num_magnitude_bins, F.get_image_size(img))
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img


    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s


# Simple unittest 
if __name__ == "__main__":
    import numpy as np
    from torchvision import transforms
    img = np.random.random((3, 224, 224))
    img=torch.tensor(img)
    pill= transforms.ToPILImage()
    img=pill(img)
    img_trans =  transforms.Compose( [RandAugment(num_ops=2, magnitude=9), transforms.ToTensor()])
    img= img_trans(img)
    print(f"This is your Image Transform in Tensor Format{img.size()}")
    print("Ohhhh Yeahhh Awesome You complete testing RandAug Transform")
    #print(fa.prnt_policies()[0])
 
