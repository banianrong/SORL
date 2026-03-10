import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
from PIL import Image
from taming.data.base import ConcatDatasetWithIndex
import cv2


class Ade20kBase(Dataset):
    def __init__(self, root, size=None, crop_size=None, random_crop=False, labels=None):
        with open(root, "r") as f:
            relpaths = f.read().splitlines()
        paths = relpaths

        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            # self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)

            self.im_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=cv2.INTER_CUBIC)
            self.seg_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                        interpolation=cv2.INTER_NEAREST)

        if crop_size is not None:
            if self.random_crop:
                self.cropper = albumentations.RandomCrop(height=crop_size, width=crop_size)
                self.flip = albumentations.HorizontalFlip(always_apply=False, p=0.5)
                self.transform = albumentations.Compose([self.cropper, self.flip, ],
                            additional_targets={"seg": "image"})                           
            else:
                self.cropper = albumentations.CenterCrop(height=crop_size, width=crop_size)
                self.transform = albumentations.Compose([self.cropper, ], additional_targets={"seg": "image"})

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict()
        example["image"], example['seg'] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        # image = np.transpose(image, (1,2,0))
        # image = Image.fromarray(image, mode="RGB")
        # image = np.array(image).astype(np.uint8)
        image = self.im_rescaler(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)

        # seg_path = image_path.replace('.jpg', '_seg.png')
        seg_path = os.path.join(r"/data1/lianjunrong/dataset/ADE20K/ADE20K_2021_17_01/annoations",image_path.split('/')[-1].replace('.jpg', '_seg.png') )
        seg = Image.open(seg_path)
        seg = np.array(seg)
        seg = self.seg_rescaler(image=seg)["image"]
        seg=(seg/255.).astype(np.float32)

        if hasattr(self, "cropper"):
            h, w, _ = image.shape
            out = self.transform(image=image, seg=seg)
            image = out["image"]
            seg = out["seg"]

        return image, seg

class Ade20kTrain(Dataset):
    def __init__(self, size, root=None, keys=None, crop_size=None, coord=False):
        d1 = Ade20kBase(root=root, size=size, crop_size=crop_size, random_crop=True)
        # d2 = FFHQTrain(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1])
        self.coord = coord
        self.keys = keys

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        ex["class"] = y
        return ex


class Ade20kValidation(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, root=None, keys=None, crop_size=None, coord=False):
        d1 = Ade20kBase(root=root, size=size, crop_size=crop_size, random_crop=False)
        # d2 = FFHQValidation(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1])
        self.coord = coord
        self.keys = keys

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        ex["class"] = y
        return ex
