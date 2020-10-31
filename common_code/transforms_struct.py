import sys
import collections
import random
import numbers
from PIL import Image
import torchvision.transforms.functional as trF
import torch
import numpy as np

#----------------------------------------------------------------------------
# Re-write transforms for our own use
# reference: official code https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
#----------------------------------------------------------------------------

if sys.version_info < (3, 3):
    Iterable = collections.Iterable
else:
    Iterable = collections.abc.Iterable

class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        image, skin, image_seg, label = sample['image'], sample['skin'], sample['image_seg'], sample['label']
        str_1, str_2, str_3, str_4 = sample['str_1'], sample['str_2'], sample['str_3'], sample['str_4']

        img = trF.resize(image, self.size, self.interpolation)
        skin = trF.resize(skin, self.size, Image.NEAREST)
        img_seg = trF.resize(image_seg, self.size, Image.NEAREST)
        str_1 = trF.resize(str_1,  self.size, Image.NEAREST)
        str_2 = trF.resize(str_2,  self.size, Image.NEAREST)
        str_3 = trF.resize(str_3,  self.size, Image.NEAREST)
        str_4 = trF.resize(str_4,  self.size, Image.NEAREST)

        return {'image': img, 'skin': skin, 'image_seg': img_seg, 'str_1': str_1, 'str_2': str_2, 'str_3': str_3, 'str_4': str_4, 'label': label}


class RatioCenterCrop(object):
    def __init__(self, ratio=1.):
        assert ratio <= 1. and ratio > 0
        # new_size = 0.8 * min(width, height)
        self.ratio = ratio

    def __call__(self, sample):
        image, skin, image_seg, label = sample['image'], sample['skin'], sample['image_seg'], sample['label']
        str_1, str_2, str_3, str_4 = sample['str_1'], sample['str_2'], sample['str_3'], sample['str_4']
        width, height = image.size
        new_size = self.ratio * min(width, height)
        img = trF.center_crop(image, new_size)
        skin = trF.center_crop(skin, new_size)
        img_seg = trF.center_crop(image_seg, new_size)
        str_1 = trF.center_crop(str_1, new_size)
        str_2 = trF.center_crop(str_2, new_size)
        str_3 = trF.center_crop(str_3, new_size)
        str_4 = trF.center_crop(str_4, new_size)
        return {'image': img, 'skin': skin, 'image_seg': img_seg, 'str_1': str_1, 'str_2': str_2, 'str_3': str_3, 'str_4': str_4, 'label': label}


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        image, skin, image_seg, label = sample['image'], sample['skin'], sample['image_seg'], sample['label']
        str_1, str_2, str_3, str_4 = sample['str_1'], sample['str_2'], sample['str_3'], sample['str_4']

        img = trF.center_crop(image, self.size)
        skin = trF.center_crop(skin, self.size)
        img_seg = trF.center_crop(image_seg, self.size)
        str_1 = trF.center_crop(str_1, self.size)
        str_2 = trF.center_crop(str_2, self.size)
        str_3 = trF.center_crop(str_3, self.size)
        str_4 = trF.center_crop(str_4, self.size)

        return {'image': img, 'skin': skin, 'image_seg': img_seg, 'str_1': str_1, 'str_2': str_2, 'str_3': str_3, 'str_4': str_4, 'label': label}


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):
        image, skin, image_seg, label = sample['image'], sample['skin'], sample['image_seg'], sample['label']
        str_1, str_2, str_3, str_4 = sample['str_1'], sample['str_2'], sample['str_3'], sample['str_4']

        i, j, h, w = self.get_params(image, self.size)
        img = trF.crop(image, i, j, h, w)
        skin = trF.crop(skin, i, j, h, w)
        img_seg = trF.crop(image_seg, i, j, h, w)
        str_1 = trF.crop(str_1, i, j, h, w)
        str_2 = trF.crop(str_2, i, j, h, w)
        str_3 = trF.crop(str_3, i, j, h, w)
        str_4 = trF.crop(str_4, i, j, h, w)

        return {'image': img, 'skin': skin, 'image_seg': img_seg, 'str_1': str_1, 'str_2': str_2, 'str_3': str_3, 'str_4': str_4, 'label': label}


class RandomRotate(object):
    def __init__(self, resample=False, expand=False, center=None):
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        idx = random.randint(0,3)
        angle = idx * 90
        return angle

    def __call__(self, sample):
        image, skin, image_seg, label = sample['image'], sample['skin'], sample['image_seg'], sample['label']
        str_1, str_2, str_3, str_4 = sample['str_1'], sample['str_2'], sample['str_3'], sample['str_4']

        angle = self.get_params()
        img = trF.rotate(image, angle, self.resample, self.expand, self.center)
        skin = trF.rotate(skin, angle, self.resample, self.expand, self.center, fill=(0,))
        #img_seg = trF.rotate(image_seg, angle, self.resample, self.expand, self.center)
        img_seg = trF.rotate(image_seg, angle, self.resample, self.expand, self.center, fill=(0,))
        str_1 = trF.rotate(str_1, angle, self.resample, self.expand, self.center)
        str_2 = trF.rotate(str_2, angle, self.resample, self.expand, self.center)
        str_3 = trF.rotate(str_3, angle, self.resample, self.expand, self.center)
        str_4 = trF.rotate(str_4, angle, self.resample, self.expand, self.center)

        return {'image': img, 'skin': skin, 'image_seg': img_seg, 'str_1': str_1, 'str_2': str_2, 'str_3': str_3, 'str_4': str_4, 'label': label}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image, skin, image_seg, label = sample['image'], sample['skin'], sample['image_seg'], sample['label']
            str_1, str_2, str_3, str_4 = sample['str_1'], sample['str_2'], sample['str_3'], sample['str_4']

            img = trF.hflip(image)
            skin = trF.hflip(skin)
            img_seg = trF.hflip(image_seg)
            str_1 = trF.hflip(str_1)
            str_2 = trF.hflip(str_2)
            str_3 = trF.hflip(str_3)
            str_4 = trF.hflip(str_4)

            return {'image': img, 'skin': skin, 'image_seg': img_seg, 'str_1': str_1, 'str_2': str_2, 'str_3': str_3, 'str_4': str_4, 'label': label}
        return sample


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image, skin, image_seg, label = sample['image'], sample['skin'], sample['image_seg'], sample['label']
            str_1, str_2, str_3, str_4 = sample['str_1'], sample['str_2'], sample['str_3'], sample['str_4']

            img = trF.vflip(image)
            skin = trF.vflip(skin)
            img_seg = trF.vflip(image_seg)
            str_1 = trF.vflip(str_1)
            str_2 = trF.vflip(str_2)
            str_3 = trF.vflip(str_3)
            str_4 = trF.vflip(str_4)

            return {'image': img, 'skin': skin, 'image_seg': img_seg, 'str_1': str_1, 'str_2': str_2, 'str_3': str_3, 'str_4': str_4, 'label': label}
        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, skin, image_seg, label = sample['image'], sample['skin'], sample['image_seg'], sample['label']
        str_1, str_2, str_3, str_4 = sample['str_1'], sample['str_2'], sample['str_3'], sample['str_4']

        img = trF.to_tensor(image)
        skin = trF.to_tensor(skin)
        img_seg = trF.to_tensor(image_seg)
        str_1 = trF.to_tensor(str_1)
        str_2 = trF.to_tensor(str_2)
        str_3 = trF.to_tensor(str_3)
        str_4 = trF.to_tensor(str_4)

        return {'image': img, 'skin': skin, 'image_seg': img_seg, 'str_1': str_1, 'str_2': str_2, 'str_3': str_3, 'str_4': str_4, 'label': label}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, skin, image_seg, label = sample['image'], sample['skin'], sample['image_seg'], sample['label']
        str_1, str_2, str_3, str_4 = sample['str_1'], sample['str_2'], sample['str_3'], sample['str_4']
        #print(np.unique(image))
        img = trF.normalize(image, self.mean, self.std)
        #print(np.unique(image))
        return {'image': img, 'skin': skin, 'image_seg': image_seg, 'str_1': str_1, 'str_2': str_2, 'str_3': str_3, 'str_4': str_4, 'label': label}