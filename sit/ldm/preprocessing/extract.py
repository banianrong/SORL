import argparse
import os
import json
import random
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
import numpy as np
from tqdm import tqdm

from PIL import Image
import PIL.Image
try:
    import pyspng
except ImportError:
    pyspng = None


class CustomDataset(Dataset):
    def __init__(self, images_dir, features_dir):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.images_dir = images_dir # os.path.join(data_dir, 'images')
        self.features_dir = features_dir # os.path.join(data_dir, 'vae-sd')

        # images
        self._image_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.images_dir)
            for root, _dirs, files in os.walk(self.images_dir) for fname in files
            }
        self.image_fnames = sorted(
            fname for fname in self._image_fnames if self._file_ext(fname) in supported_ext
            )
        # features
        self._feature_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.features_dir)
            for root, _dirs, files in os.walk(self.features_dir) for fname in files
            }
        self.feature_fnames = sorted(
            fname for fname in self._feature_fnames if self._file_ext(fname) in supported_ext
            )
        # labels
        fname = 'dataset.json'
        with open(os.path.join(self.features_dir, fname), 'rb') as f:
            labels = json.load(f)['labels']
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.feature_fnames]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])


    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        assert len(self.image_fnames) == len(self.feature_fnames), \
            "Number of feature files and label files should be same"
        return len(self.feature_fnames)

    def __getitem__(self, idx):
        image_fname = self.image_fnames[idx]
        feature_fname = self.feature_fnames[idx]
        image_ext = self._file_ext(image_fname)
        with open(os.path.join(self.images_dir, image_fname), 'rb') as f:
            if image_ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif image_ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)

        features = np.load(os.path.join(self.features_dir, feature_fname))
        return torch.from_numpy(image), torch.from_numpy(features), torch.tensor(self.labels[idx])

def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions


def calculate_latents_stats(dataset, save_path, device="cuda"):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    all_latents = []
    
    for _, feature, _ in tqdm(dataloader):
        moments = feature.to(device)
        latent_mean, _ = torch.chunk(moments, 2, dim=1) 
        all_latents.append(latent_mean)

    all_latents = torch.cat(all_latents, dim=0)
    
    mean = all_latents.mean(dim=(0, 2, 3))
    std = all_latents.std(dim=(0, 2, 3))
    
    stats = {
        "latents_bias": (mean).cpu(),      
        "latents_scale": (1.0 / std).cpu() 
    }
    
    torch.save(stats, save_path)
    print(f"stats saved to: {save_path}")
    print(f"channel depth: {mean.shape[0]}")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_dir", default=None, type=str, required=True, help="Path to the images.")
    parser.add_argument("--features_dir", default=None, type=str, required=True, help="Path to the features.")
    parser.add_argument("--target_path", default=None, type=str, required=True, help='Path to the storage convert dataset.')
    args = parser.parse_args()

    data = CustomDataset(args.images_dir, args.features_dir)
    calculate_latents_stats(data, args.target_path)

    # python extract.py --images_dir /path/to/images --features_dir /path/to/latents --target_path ./latent_stats.pt
