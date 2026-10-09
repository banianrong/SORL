import os
import subprocess
from torch_fidelity import calculate_metrics
import sys
sys.path.append(".")

# also disable grad to save memory
import torch
torch.set_grad_enabled(False)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import yaml
import torch
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ
from torchvision.utils import save_image
from main import instantiate_from_config
import io
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np

import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# 参数设置
generated_images_path = "/path/to/generated/images/"
real_images_path = '/path/to/celebahq/test/'
preprocess_folder = "/path/to/celebahq/preprocessed/"
# 运行生成图像的脚本
# Generate images with scripts/sample_diffusion.py before running evaluation.

# def load_config(config_path, display=False):
#   config = OmegaConf.load(config_path)
#   if display:
#     print(yaml.dump(OmegaConf.to_container(config)))
#   return config

# def load_vqgan(config, ckpt_path=None, is_gumbel=False):
#   model = instantiate_from_config(config.model)
#   if ckpt_path is not None:
#     sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
#     missing, unexpected = model.load_state_dict(sd, strict=False)
#   return model.eval()

# def preprocess_vqgan(x):
#   x = 2.*x - 1.
#   return x

# def custom_to_pil(x):
#   x = x.detach().cpu()
#   x = torch.clamp(x, -1., 1.)
#   x = (x + 1.)/2.
#   x = x.permute(1,2,0).numpy()
#   x = (255*x).astype(np.uint8)
#   x = Image.fromarray(x)
#   if not x.mode == "RGB":
#     x = x.convert("RGB")
#   return x
size = 256
def preprocess(img, target_image_size=256, map_dalle=True):
    s = min(img.size)
    if s < target_image_size:
        r = target_image_size / s
        s = (round(r * img.size[1]), round(r * img.size[0]))
        img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
        s = target_image_size
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    if map_dalle: 
      img = map_pixels(img)
    return img.to(DEVICE)


def save_preprocessed_image(input_folder, preprocess_folder):
  for img in os.listdir(input_folder):
    image = Image.open(os.path.join(input_folder, img))
    preprocessed_img = preprocess(image, target_image_size=size, map_dalle=False)
    save_image(preprocessed_img, os.path.join(preprocess_folder, img))
    # print(f"saved preprocessed {img}")


# save_preprocessed_image(real_images_path, preprocess_folder)








# 计算FID
from pytorch_fid import fid_score
def eval_fid(preprocess_folder, recon_folder):
    fid_value = fid_score.calculate_fid_given_paths([str(preprocess_folder), str(recon_folder)], batch_size=64, device=DEVICE, dims=2048)
    return fid_value

# subprocess.run(['pytorch-fid', preprocess_folder, generated_images_path, '--batch-size', '64'])
# fid_score=eval_fid(preprocess_folder, generated_images_path)
# print(f'fid: {fid_score}\n')
# 计算IS
# metrics = calculate_metrics(input1=generated_images_path, input2=None, metrics=["isc"])
# print(f'Inception Score: {metrics["inception_score_mean"]}')
# # 计算Precision和Recall
# metrics = calculate_metrics(input1=generated_images_path, input2=real_images_path, metrics=['precision', 'recall'])
# print(f'Precision: {metrics["precision"]}')
# print(f'Recall: {metrics["recall"]}')

metrics_dict = calculate_metrics(
    input1=generated_images_path, 
    input2=preprocess_folder, 
    cuda=True, 
    isc=True, 
    fid=True, 
    kid=False, 
    prc=True, 
    verbose=False,
)
print(metrics_dict)
