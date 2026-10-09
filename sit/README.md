# SiT: stage-two latent generative modeling

This directory contains the second stage of SORL: latent preprocessing, SiT training, distributed sampling, and evaluation. The main SORL experiments live in `ldm/`; `jit/` contains the pixel-space/JIT comparison code used by related experiments.

## Directory map

```text
sit/
├── ldm/
│   ├── preprocessing/   image conversion, VAE conversion, latent extraction
│   ├── models/          SiT and the lightweight stage-one VAE definition
│   ├── configs/         SORL and baseline experiment configurations
│   ├── train.py         distributed SiT training
│   └── generate.py      distributed class-conditional sampling
├── jit/                 JIT/reference training path
├── metrics/             spatial metrics and plotting utilities
├── CLIP/                vendored OpenAI CLIP dependency
└── torch-fidelity/      vendored evaluation dependency
```

## Environment

```bash
conda create -n sorl-sit python=3.10 -y
conda activate sorl-sit
pip install -r environment/requirements.txt
```

Commands below are run from `sit/ldm` unless noted otherwise.

## 1. Prepare ImageNet images

The input ImageNet directory must retain its class folders:

```text
imagenet/train/
├── n01440764/
│   ├── image_1.JPEG
│   └── image_2.JPEG
└── n01443537/
    └── image_3.JPEG
```

Convert images to the deterministic 256×256 archive used by the latent pipeline:

```bash
python preprocessing/dataset_tools.py convert \
  --source /datasets/imagenet/train \
  --dest /datasets/sorl/imagenet-256 \
  --resolution 256x256 \
  --transform center-crop-dhariwal
```

## 2. Connect the stage-one model

Convert the SOVAE Lightning checkpoint into a local Diffusers directory:

```bash
python preprocessing/convert_vae_pt_to_diffusers.py \
  --vae_pt_path /checkpoints/stage1.ckpt \
  --config_path preprocessing/v1-inference.yaml \
  --dump_path /checkpoints/sorl-diffusers-vae
```

The converter accepts a Lightning checkpoint or a raw state dict and removes the stage-one loss and quantizer parameters. This conversion is only for offline encoding; retain `stage1.ckpt` for training-time reconstruction and sampling.

## 3. Extract and package latents

```bash
python preprocessing/dataset_tools.py encode \
  --source /datasets/sorl/imagenet-256 \
  --dest /datasets/sorl/imagenet-latents \
  --model-url /checkpoints/sorl-diffusers-vae

python preprocessing/convert_npy_to_arrow.py \
  --images_dir /datasets/sorl/imagenet-256 \
  --features_dir /datasets/sorl/imagenet-latents \
  --target_path /datasets/sorl/arrow

python preprocessing/extract.py \
  --images_dir /datasets/sorl/imagenet-256 \
  --features_dir /datasets/sorl/imagenet-latents \
  --target_path /datasets/sorl/latent_stats.pt
```

The final two artifacts consumed by training are the Arrow dataset directory and `latent_stats.pt`. The statistics file must contain `latents_scale` and `latents_bias` tensors.

## 4. Train SORL-SiT

```bash
accelerate launch --num_processes 8 train.py \
  --config configs/sorl.yaml \
  --model SiT-XL/2 \
  --encoder-depth 4 \
  --data-dir /datasets/sorl/arrow \
  --exp-name sorl-sit-xl2 \
  --vae-pt /checkpoints/stage1.ckpt \
  --latents-pt /datasets/sorl/latent_stats.pt
```

`--num_processes` is the number of GPUs. `--batch-size` is the global batch size expected by the training script. Adjust both together with gradient accumulation for your hardware. Logging defaults to Weights & Biases; use `--report-to tensorboard` if desired.

## 5. Generate samples

Sampling uses the same stage-one checkpoint and latent statistics as training:

```bash
torchrun --standalone --nproc_per_node=8 generate.py \
  --ckpt /checkpoints/sit/checkpoints/0400000.pt \
  --model SiT-XL/2 \
  --encoder-depth 4 \
  --vae-pt /checkpoints/stage1.ckpt \
  --latents-pt /datasets/sorl/latent_stats.pt \
  --sample-dir samples \
  --num-fid-samples 50000
```

The script writes rank-safe PNG files and then creates an NPZ archive for FID-style evaluation. Model architecture arguments must match the training run.

## Common failure modes

- **VAE missing/unexpected keys:** the stage-one architecture does not match `VAE_F8D4`, or the wrong checkpoint was passed.
- **Latent channel/shape mismatch:** preprocessing and training used different VAE checkpoints or resolutions.
- **Dataset not found:** `--data-dir` must point to the directory produced by `convert_npy_to_arrow.py`, not the source ImageNet folder.
- **Out of memory:** lower the batch size or per-process sampling batch size; do not change latent dimensions to solve memory pressure.

## Attribution

The stage-two code builds on SiT and related representation-alignment research code. The preprocessing path also contains NVIDIA-derived utilities, and the repository vendors CLIP and torch-fidelity. Their original notices remain in the corresponding files/directories; see [`THIRD_PARTY.md`](../THIRD_PARTY.md).
