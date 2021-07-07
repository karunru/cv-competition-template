import random

import albumentations as A
import numpy as np
import torch
import torchvision.transforms as T
from albumentations import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from espnet.transform.spec_augment import freq_mask, time_mask, time_warp

mean = (0.485, 0.456, 0.406)  # RGB
std = (0.229, 0.224, 0.225)  # RGB


class TimeMask(ImageOnlyTransform):
    def __init__(
        self,
        T=30,
        n_mask=2,
        replace_with_zero=True,
        inplace=False,
        always_apply=False,
        p=0.5,
    ):
        super(TimeMask, self).__init__(always_apply, p)
        self.T = T
        self.n_mask = n_mask
        self.replace_with_zero = replace_with_zero
        self.inplace = inplace

    def apply(self, image, **params):
        return time_mask(
            spec=image,
            T=self.T,
            n_mask=self.n_mask,
            replace_with_zero=self.replace_with_zero,
            inplace=self.inplace,
        )

    def get_transform_init_args_names(self):
        return (
            "T",
            "n_mask",
            "replace_with_zero",
            "inplace",
            "always_apply",
            "p",
        )


class FreqMask(ImageOnlyTransform):
    def __init__(
        self,
        F=30,
        n_mask=2,
        replace_with_zero=True,
        inplace=False,
        always_apply=False,
        p=0.5,
    ):
        super(FreqMask, self).__init__(always_apply, p)
        self.F = F
        self.n_mask = n_mask
        self.replace_with_zero = replace_with_zero
        self.inplace = inplace

    def apply(self, image, **params):
        return freq_mask(
            x=image.astype(float),
            F=self.F,
            n_mask=self.n_mask,
            replace_with_zero=self.replace_with_zero,
            inplace=self.inplace,
        )

    def get_transform_init_args_names(self):
        return (
            "F",
            "n_mask",
            "replace_with_zero",
            "inplace",
            "always_apply",
            "p",
        )


class TimeWarp(ImageOnlyTransform):
    def __init__(
        self,
        max_time_warp=80,
        inplace=False,
        mode="PIL",
        always_apply=False,
        p=0.5,
    ):
        super(TimeWarp, self).__init__(always_apply, p)
        self.max_time_warp = max_time_warp
        self.inplace = inplace
        self.mode = mode

    def apply(self, image, **params):
        return time_warp(
            x=image.astype(float),
            max_time_warp=self.max_time_warp,
            inplace=self.inplace,
            mode=self.mode,
        )

    def get_transform_init_args_names(self):
        return (
            "max_time_warp",
            "inplace",
            "mode",
            "always_apply",
            "p",
        )


class FreqWarp(ImageOnlyTransform):
    def __init__(
        self,
        max_time_warp=80,
        inplace=False,
        mode="PIL",
        always_apply=False,
        p=0.5,
    ):
        super(FreqWarp, self).__init__(always_apply, p)
        self.max_time_warp = max_time_warp
        self.inplace = inplace
        self.mode = mode

    def apply(self, image, **params):
        return time_warp(
            x=image.astype(float).T,
            max_time_warp=self.max_time_warp,
            inplace=self.inplace,
            mode=self.mode,
        ).T

    def get_transform_init_args_names(self):
        return (
            "max_time_warp",
            "inplace",
            "mode",
            "always_apply",
            "p",
        )


def seti_transform0(size):
    size = list(map(int, size.split(",")))
    transform = {
        "albu_train": A.Compose(
            [
                # A.RandomResizedCrop(height=size[0], width=size[1], p=0.5),
                TimeMask(p=0.5),
                FreqMask(p=0.5),
                TimeWarp(p=0.5),
                FreqWarp(p=0.5),
                # A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.MotionBlur(p=0.5),
                A.ShiftScaleRotate(rotate_limit=0, p=0.5),
                A.Resize(size[0], size[1]),
                ToTensorV2(),
            ]
        ),
        "torch_train": T.Compose(
            [
                T.Resize([size[0], size[1]]),
                T.ConvertImageDtype(torch.float),
            ]
        ),
        "albu_val": A.Compose(
            [
                A.Resize(size[0], size[1]),
                ToTensorV2(),
            ]
        ),
        "torch_val": T.Compose(
            [
                T.Resize([size[0], size[1]]),
                T.ConvertImageDtype(torch.float),
            ]
        ),
    }
    return transform
