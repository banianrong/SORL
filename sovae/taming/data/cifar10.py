import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


class ImagePaths(Dataset):
    def __init__(self, paths, train=True, size=None, random_crop=False, crop_size=None, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.cifar10 = CIFAR10(root=paths, train=train, download=True)
        self.labels["file_path_"] = [self.cifar10[i][1] for i in range(len(self.cifar10))]
        self._length = len(self.cifar10)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if crop_size is not None:
                self.cropper = albumentations.RandomCrop(height=crop_size, width=crop_size)
                self.flip = albumentations.HorizontalFlip(always_apply=False, p=0.5)
                self.cropper = albumentations.Compose([self.cropper, self.flip])
            else:
                if not self.random_crop:
                    self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
                else:
                    self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image):
        image = image[0]
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.cifar10[i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file, crop_size=None):
        super().__init__()
        self.data = ImagePaths(paths=training_images_list_file, train=True, size=size, random_crop=False, crop_size=crop_size)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        self.data = ImagePaths(paths=test_images_list_file, train=False, size=size, random_crop=False)


