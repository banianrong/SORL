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
import os, sys
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  model = instantiate_from_config(config.model)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  # z, _, [_, _, indices] = model.encode(x)
  # xrec = model.decode(z)
  xrec, diff = model(x)
  print(f" diff is : {diff}")
  return xrec

config = load_config("configs/new_example.yaml", display=False)
model = load_vqgan(config, ckpt_path="/path/to/first_stage.ckpt").to(DEVICE)

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

size=256

def reconstruction_pipeline(image):
  x_vqgan = preprocess(image, target_image_size=size, map_dalle=False)
  x_vqgan = x_vqgan.to(DEVICE)
  x0 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model)
  img = custom_to_pil(x0[0])
  return img

def save_recon_image(input_folder, recon_folder):
  for img in os.listdir(input_folder):
    image = Image.open(os.path.join(input_folder, img))
    recon_img = reconstruction_pipeline(image)
    recon_img.save(os.path.join(recon_folder, img))
    # print(f"saved recon {img}")

def save_preprocessed_image(input_folder, preprocess_folder):
  for img in os.listdir(input_folder):
    image = Image.open(os.path.join(input_folder, img))
    preprocessed_img = preprocess(image, target_image_size=size, map_dalle=False)
    save_image(preprocessed_img, os.path.join(preprocess_folder, img))
    # print(f"saved preprocessed {img}")

input_folder = "/path/to/validation/images"
recon_folder = "/path/to/reconstructions"
preprocess_folder = "/path/to/preprocessed"
# save_preprocessed_image(input_folder, preprocess_folder)
# save_recon_image(input_folder, recon_folder)
lpips_fn = lpips.LPIPS(net='vgg').cuda()

import cv2
def normalize(img, mode="-11"):
    img = np.array(img)
    # print(f"x.min={img.min()}, x.max={img.max()}")
    if mode == "01":
        img_norm = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    elif mode == "-11":
        img_norm = cv2.normalize(img.astype('float'), None, -1.0, 1.0, cv2.NORM_MINMAX)
    return img_norm
        
# calculate ssim, psnr and lpips
def eval_folder(preprocess_folder, recon_folder):
    ssim_score = []
    psnr_score = []
    lpips_score = []
    results = []
    preprocessed_images = sorted(os.listdir(preprocess_folder))
    reconstructed_images = sorted(os.listdir(recon_folder))
    for img in preprocessed_images:
      if img in reconstructed_images:
        # print(f"processing {img}")
        original_img = Image.open(os.path.join(preprocess_folder, img)).convert('RGB')
        recon_img = Image.open(os.path.join(recon_folder, img)).convert('RGB')
        # 归一化
        original_img_norm = normalize(original_img, mode="-11")
        recon_img_norm = normalize(recon_img, mode="-11")
        ssim_for_pic = ssim(np.array(original_img_norm), np.array(recon_img_norm), data_range=2.0, multichannel=True,channel_axis=-1)
        ssim_score.append(ssim_for_pic)
        psnr_for_pic = psnr(np.array(original_img_norm), np.array(recon_img_norm), data_range=2.0)
        psnr_score.append(psnr_for_pic)
        # lpips不需要归一化
        original_tensor = T.ToTensor()(original_img).float().unsqueeze(0).to(DEVICE)
        recon_tensor = T.ToTensor()(recon_img).float().unsqueeze(0).to(DEVICE)
        lpips_for_pic = lpips_fn(original_tensor, recon_tensor).item()
        lpips_score.append(lpips_for_pic)
    results.append(np.mean(ssim_score))
    results.append(np.mean(psnr_score))
    results.append(np.mean(lpips_score))
    return results, ssim_score, psnr_score, lpips_score
# calculate fid
from pytorch_fid import fid_score
def eval_fid(preprocess_folder, recon_folder):
    fid_value = fid_score.calculate_fid_given_paths([str(preprocess_folder), str(recon_folder)], batch_size=64, device=DEVICE, dims=2048)
    return fid_value

results, ssim_score, psnr_score, lpips_score = eval_folder(preprocess_folder, recon_folder)
results.append(eval_fid(preprocess_folder, recon_folder))
ssim_score_max = np.max(ssim_score)
ssim_score_min = np.min(ssim_score)
ssim_score_min_index = np.argmin(ssim_score)
psnr_score_max = np.max(psnr_score)
psnr_score_min = np.min(psnr_score)
psnr_score_min_index = np.argmin(psnr_score)
ssim_score.sort()
psnr_score.sort()
print(f"ssim: {results[0]}, psnr: {results[1]}, lpips: {results[2]}, fid: {results[3]}")
print(f"ssim_score_max: {ssim_score_max}, ssim_score_min: {ssim_score_min}, ssim_score_min_index: {ssim_score_min_index}")
print(f"psnr_score_max: {psnr_score_max}, psnr_score_min: {psnr_score_min}, psnr_score_min_index: {psnr_score_min_index}")
