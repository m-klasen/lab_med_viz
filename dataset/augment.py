from PIL import Image
import albumentations as A
from monai import transforms as T
import cv2
from albumentations.pytorch.transforms import ToTensorV2

from dataset.rand_augment import rand_augment_transform
from dataset.rrc import RandomResizedCropAndInterpolation, ToNumpy
from torchvision import transforms

from skimage.restoration import denoise_tv_chambolle

from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform

mean = [0.5]
_pil_interpolation_to_str = {
    Image.NEAREST: 'nearest',
    Image.BILINEAR: 'bilinear',
    Image.BICUBIC: 'bicubic',
    Image.BOX: 'box',
    Image.HAMMING: 'hamming',
    Image.LANCZOS: 'lanczos',
}
#interpolation = 'random'
_str_to_pil_interpolation = {b: a for a, b in _pil_interpolation_to_str.items()}
def str_to_pil_interp(mode_str):
    return _str_to_pil_interpolation[mode_str]
aa_params = dict(
            translate_const=int(256 * 0.020),
            img_mean=tuple([0 for x in mean]),
            #interpolation=str_to_pil_interp(interpolation)
            )
auto_augment = "rand-m9-mstd0.5"

def auto_augment_fkt():
    return rand_augment_transform(auto_augment, aa_params)



class TV_aug(DualTransform):
    def apply(self, img, **params):
        return denoise_tv_chambolle(img, weight=0.1)


from dataset.rand_augment import AugmentOp, RandAugment, rand_augment_ops
import re

hparams = dict(
            translate_const=int(256 * 0.020),
            img_mean=tuple([0]),
            )
    
def augment_transform(transforms,config_str, hparams):
    num_layers = 2  # default to 2 ops per image
    weight_idx = None  # default to no probability weights for op choice
    config = config_str.split('-')
    assert config[0] == 'rand'
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            # noise param / randomization of magnitude values
            mstd = float(val)
            if mstd > 100:
                # use uniform sampling in 0 to magnitude if mstd is > 100
                mstd = float('inf')
            hparams.setdefault('magnitude_std', mstd)
        elif key == 'mmax':
            # clip magnitude between [0, mmax] instead of default [0, _LEVEL_DENOM]
            hparams.setdefault('magnitude_max', int(val))
        elif key == 'm':
            magnitude = int(val)
        elif key == 'n':
            num_layers = int(val)
        elif key == 'w':
            weight_idx = int(val)
        else:
            assert False, 'Unknown RandAugment config section'
        
        ra_ops = rand_augment_ops(magnitude=magnitude, hparams=hparams, transforms=transforms)
        choice_weights = None
        return RandAugment(ra_ops, num_layers, choice_weights=choice_weights)


def get_PIL_tfms(tfms):
    augment_transform(tfms,"rand-m9-mstd0.5",hparams)

def get_transforms(config):
    if config=="default":
        tfms = A.Compose([
             A.Normalize (mean=(0.122), std=(0.224), max_pixel_value=1.)
            ])

    elif config=="rrc":
        tfms = A.Compose([
             A.RandomResizedCrop(height=256, width=256, 
                                 scale=(0.08, 1.0), ratio=(3./4., 4./3.), 
                                 interpolation=cv2.INTER_CUBIC, p=0.5),
             A.Normalize(mean=(0.122), std=(0.224), max_pixel_value=1.)
            ])
    elif config=="weak_rrc":
        tfms = A.Compose([
             A.RandomResizedCrop(height=256, width=256, 
                                 scale=(0.5, 1.0), ratio=(3./4., 4./3.), 
                                 interpolation=cv2.INTER_CUBIC, p=0.5),
             A.Normalize(mean=(0.122), std=(0.224), max_pixel_value=1.)
            ])
    elif config=="totvar":
        tfms = A.Compose([
             TV_aug(p=0.5),
             A.Normalize(mean=(0.122), std=(0.224), max_pixel_value=1.)
            ])    
    elif config=="GridDistortion":
        tfms = A.Compose([
             A.GridDistortion (num_steps=5, distort_limit=0.3, p=0.5),
             A.Normalize(mean=(0.122), std=(0.224), max_pixel_value=1.)
            ])        
    elif config=="gaussian":
        tfms = A.Compose([
             A.GaussianBlur(blur_limit=(3, 7), p=0.5),
             A.Normalize(mean=(0.122), std=(0.224), max_pixel_value=1.)
            ]) 
    elif config=="gamma":
        tfms = A.Compose([
             A.RandomGamma(gamma_limit=(80, 120), p=0.5),
             A.Normalize(mean=(0.122), std=(0.224), max_pixel_value=1.)
            ]) 
    elif config=="gamma_fcm":
        tfms = A.Compose([
             A.RandomGamma(gamma_limit=(80, 120), p=0.5)
            ]) 
    elif config=="contrast_fcm":
        tfms = A.Compose([
            A.ColorJitter (brightness=0.2, contrast=0.0, saturation=0.0, hue=0.0, p=1.0)
        ])
    elif config=="intensity_fcm":
        tfms = T.Compose([
            T.ShiftIntensityd(keys=("image"), offset=0.05)
        ])
    elif config=="translation":
        tfms = A.Compose([
             A.Affine(translate_percent=(0.25,0.25)),
             A.Normalize(mean=(0.122), std=(0.224), max_pixel_value=1.)
            ]) 
    elif config=="rotation":
        tfms = A.Compose([
             A.Affine(rotate=(-90,90)),
             A.Normalize(mean=(0.122), std=(0.224), max_pixel_value=1.)
            ]) 
    elif config=="cutout":
        tfms = A.Compose([
             A.CoarseDropout(p=0.5, max_holes=8, max_height=20, max_width=20, min_holes=8, min_height=15, min_width=15),
             A.Normalize(mean=(0.122), std=(0.224), max_pixel_value=1.)
            ])

    elif config=="brightness":
        tfms = A.Compose([
             A.RandomBrightnessContrast(p=0.5, brightness_limit=(0.3, 0.5), contrast_limit=(0, 0), brightness_by_max=True),
             A.Normalize(mean=(0.122), std=(0.224), max_pixel_value=1.)
            ]) 
    elif config=="brightness0.2":
        tfms = A.Compose([
             A.RandomBrightnessContrast(p=0.2, brightness_limit=(0.1, 0.5), contrast_limit=(0, 0), brightness_by_max=True),
             A.Normalize(mean=(0.122), std=(0.224), max_pixel_value=1.)
            ]) 
    elif config=="brightness_contrast":
        tfms = A.Compose([
             A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True),
             A.Normalize(mean=(0.122), std=(0.224), max_pixel_value=1.)
            ]) 
    elif config=="gaussian_noise":
        tfms = A.Compose([
             A.GaussNoise(p=0.5, var_limit=(0., 0.01)),
             A.Normalize(mean=(0.122), std=(0.224), max_pixel_value=1.)
            ])
    elif config=="combo1":
        tfms = A.Compose([
             A.OneOf(
                 A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                 A.GridDistortion (num_steps=5, distort_limit=0.3, p=0.5),
                 ),
             A.Normalize(mean=(0.122), std=(0.224), max_pixel_value=1.)
            ])
    elif config=="rand_aug":
        tfms = A.Compose([
             #RandAug(p=1.0),
             A.Normalize(mean=(0.122), std=(0.224), max_pixel_value=1.)
            ])  
    elif config==None:
        return None
    return tfms




def get_test_transforms():
    tfms = A.Compose([
             A.Normalize (mean=(0.122), std=(0.224), max_pixel_value=1.)
            ])
    return tfms