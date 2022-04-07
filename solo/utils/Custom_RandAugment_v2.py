# 2022/04/01, 5:15 pm, tech support : Harry, Josef
import torch
from torchvision.transforms import RandAugment, InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms import ColorJitter, Grayscale, GaussianBlur, ToTensor, ToPILImage
from torchvision import transforms

# 3rd pkg
from typing import List, Tuple, Optional, Dict
from PIL import Image
import math

## The original data augmentation of SimCLR repository 
#  plz referes : https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
class Extended_RangAugment(RandAugment):
    ## public interface
    def __init__(self, num_ops: int = 2, magnitude: int = 9, num_magnitude_bins: int = 31, 
                        interpolation: InterpolationMode = InterpolationMode.NEAREST, 
                        fill: Optional[List[float]] = None, sigma: Tuple[int, int] = (0.1, 2.0),
                        debug_flag: bool = False
    ) -> None:
        super().__init__(num_ops, magnitude, num_magnitude_bins, interpolation, fill)
        self.sigma = sigma
        self.debug_flag = debug_flag

        ## debug info data-structure (you can remove it in release version)
        if self.debug_flag:
            self.im_idx = 0
            self.DEBUG_LIST = list() 

    def reset_im_idx(self):
        if self.debug_flag:
            self.im_idx = 0

    def _ext_apply_op(self, img, op_name, magnitude, interpolation, fill, sigma):
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
            # (option) since our kernel map range is safe, it could be +1 to enlarge the kernel effect 
            odd_mag = int(magnitude) if int(magnitude) % 2 == 1 else int(magnitude) - 1
            blur = GaussianBlur(kernel_size=odd_mag, sigma=sigma)
            img = blur(img)

        elif op_name == "rand_gray_scale": 
            gray_scale = Grayscale(num_output_channels=3) # it doesn't have the params to search, only random apply
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
                #center=[0, 0],
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
            img = F.posterize(img.to(dtype=torch.uint8), int(magnitude))
        elif op_name == "Solarize":
            img = F.solarize(img, magnitude)
        elif op_name == "AutoContrast":
            img = F.autocontrast(img)
        elif op_name == "Equalize":
            img = F.equalize( img.to(dtype=torch.uint8) )
        elif op_name == "Invert":
            img = F.invert(img)
        elif op_name == "Identity":
            pass
        else:
            raise ValueError(f"The provided operator {op_name} is not recognized.")
        return img


    def _ext_aug_space(self, num_bins, image_size):
        return {
            # make your new extension..
            #"rand_brightness": (torch.linspace(0.0, 0.8, num_bins), False),
            # "rand_contrast": (torch.linspace(0.0, 0.8, num_bins), False),
            # "rand_saturation": (torch.linspace(0.0, 0.8, num_bins), False),
            # "rand_hue": (torch.linspace(0.0, 0.5, num_bins), False),
            # # I think 11x11 filter map is enough as maximum value,
            # #   and 1x1 filter map is capable to keep the orignal info of img.
            # "rand_gaussian_blur": (torch.linspace(1.0, 11.0, num_bins), False), 
            # "rand_gray_scale": (torch.tensor(0.0), False),
            
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
        if isinstance(img, Image.Image):
            img = ToTensor()(img)

        if self.debug_flag:
            tmp_debug_lst = []

        channels, height, width = img.shape     

        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._ext_aug_space(self.num_magnitude_bins, [height, width])
        for trfs_idx in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0

            if self.debug_flag:
                tmp_debug_lst.append({'trfs_idx':trfs_idx, 'op_name':op_name, 'magnitude':magnitude})

            img = self._ext_apply_op(img, op_name, magnitude, 
                    # this part is for customized params, such as upsampling interpolation, blurring sigma range, ..., etc.
                        interpolation=self.interpolation, fill=fill, sigma=self.sigma)
        
        if self.debug_flag:
            self.DEBUG_LIST.append(tmp_debug_lst)
            self.im_idx+=1
        
        # image = transforms.ToTensor()
        # img=image(img)
        # print(img.shape)

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

# Simple unittest ; 2020/04/03 16:32pm Josef-Huang.www
if __name__ == "__main__":
    # testing function..
    def test_pil_format(tst_cfg, rand_aug):
        print('Testing data type..')
        print('-------------------')

        rnd_tnsrs = torch.rand(tst_cfg['image_size'])
        for idx, rnd_tnsr in enumerate(rnd_tnsrs):
            pil_tnsr = ToPILImage()(rnd_tnsr)

            # test input pil-format
            out_tnsr = rand_aug(pil_tnsr)
            if idx < 1:
                print(f'The output tensor type is {type(out_tnsr)}')

            if not isinstance(out_tnsr, torch.Tensor):
                _show_debug_info(rand_aug.DEBUG_LIST, idx)
                raise TypeError(f'The output tensor type is {type(out_tnsr)}, it should be {torch.Tensor}')
        print('\n')


    def test_output_shape(tst_cfg, rand_aug):
        print('Testing output_shape..')
        print('-------------------')

        rnd_tnsrs = torch.rand(tst_cfg['image_size'])
        GT_shp = tst_cfg['image_size'][1:]
        for idx, rnd_tnsr in enumerate(rnd_tnsrs):
            out_tnsr = rand_aug(rnd_tnsr)
            if idx < 1:
                print(f'The output tensor type is {out_tnsr.shape}')

            if list(out_tnsr.shape) != GT_shp:
                _show_debug_info(rand_aug.DEBUG_LIST, idx)
                raise ValueError(f"got output tensor shape : {list(out_tnsr.shape)}, but it should be {GT_shp}")
        print('\n')


    def test_subpolicy(tst_cfg, rand_aug, policy_lst):
        print('Testing subpolicy..')
        print('-------------------')

        GT_shp = tst_cfg['image_size'][1:]
        rnd_tnsr = torch.rand( [1,] + GT_shp) # only get single image to test
        op_meta = rand_aug._ext_aug_space(rand_aug.num_magnitude_bins, (GT_shp[1:]))
        
        for op_name in policy_lst:
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[rand_aug.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            print(f"Processing {op_name} trfs.. mag : {magnitude}  raw mag (before trfs) : {rand_aug.magnitude}\n")

            out_tnsr = rand_aug._ext_apply_op(rnd_tnsr, op_name, magnitude, 
                    # this part is for customized params, such as upsampling interpolation, blurring sigma range, ..., etc.
                        interpolation=rand_aug.interpolation, fill=rand_aug.fill, sigma=rand_aug.sigma)
            # 1. chk output tensor type
            if not isinstance(out_tnsr, torch.Tensor):
                _show_debug_info(rand_aug.DEBUG_LIST, idx)
                raise TypeError(f'The output tensor type is {type(out_tnsr)}, it should be {torch.Tensor}')
            # 2. chk output tensor shape
            if list(out_tnsr.shape) != [1,] + GT_shp:
                _show_debug_info(rand_aug.DEBUG_LIST, idx)
                raise ValueError(f"got output tensor shape : {list(out_tnsr.shape)}, but it should be {[1,] + GT_shp}")
        print('\n')

    def _show_debug_info(db_lst=None, idx=None):
        [op1, op2] = db_lst[idx]
        print(f'op1 : {op1} \n op2 : {op2} \n')

    tst_cfg = {
        'image_size':[64, 3, 224, 224]
    }
    if __debug__ != True : print('not debug mode in uniitest')   # active assertion error

    rand_aug_params = {
        'num_ops' : 2, 'magnitude' : 9, 'num_magnitude_bins' : 31, # so the degree range will be [0, 30]
        'interpolation' : InterpolationMode.NEAREST, 'fill' : None, 'sigma' : (0.1, 2.0),
        'debug_flag' : True
    }
    rand_aug = Extended_RangAugment( **rand_aug_params )
    
    # printout extended augmentation space
    aug_space = rand_aug._augmentation_space(num_bins=rand_aug_params['num_magnitude_bins'], image_size=tst_cfg['image_size'][1:])
    print("Testing begin..\n")
    print(f"Supported transformations : {aug_space.keys()} \n")
    
    n_err = 0
    try:
        test_pil_format(tst_cfg, rand_aug)
    except Exception as ex :
        print(ex)
        n_err += 1
    
    try:
        test_output_shape(tst_cfg, rand_aug)
    except Exception as ex :
        print(ex)
        n_err += 1

    try:
        # put the trfs name (str type) you want to test
        policy_lst = list( aug_space.keys() )  # test all sub-policy..
        test_subpolicy(tst_cfg, rand_aug, policy_lst)
    except Exception as ex :
        print(ex)
        n_err += 1

    print(f'Simple unittest complete, {n_err} error occurs\n')


    
            

    
