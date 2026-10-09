# SOVAE: stage-one representation learning

This directory contains the first stage of SORL: a self-organized latent autoencoder trained with local codebook constraints. The implementation is a research modification of [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers), retaining its configuration-driven training structure and adversarial/perceptual reconstruction objective.

## What was changed

The main SORL-specific pieces are:

- `taming/models/vq_test.py`: autoencoder variants and the stage-one training module;
- `taming/modules/vqvae/quantize_test.py`: constrained/local-rule quantization;
- `taming/modules/distributions/distributions.py`: the latent distribution used by SORL;
- `configs/sorl_celebahq.yaml`: the three-channel, 4× setup paired with the paper's CelebA-HQ LDM configuration;
- `configs/imagenet_sorl.yaml`: the four-channel, 8× ImageNet/SiT setup;
- the remaining imported configs: VAE, WAE, RV-VAE, VQGAN, classification, and conditional-transformer development baselines.

The remaining `taming/` code largely follows the original Taming Transformers project and is included so the stage can run as a self-contained package.

## Environment

```bash
conda env create -f environment.yaml
conda activate sorl
pip install -e .
```

The pinned environment reflects the environment used for this release. If your CUDA driver does not support the pinned PyTorch build, install a compatible PyTorch/torchvision pair first and then install the remaining dependencies.

## Prepare ImageNet

Download ImageNet separately. Create one text file for training images and one for validation images; each line must contain one image path. Absolute paths are recommended.

```text
/datasets/imagenet/train/n01440764/example_1.JPEG
/datasets/imagenet/train/n01440764/example_2.JPEG
...
```

The helper can build each list from an image directory:

```bash
mkdir -p filelists
python build_filelist.py /datasets/imagenet/train filelists/train.txt
python build_filelist.py /datasets/imagenet/val filelists/val.txt
```

Then edit `configs/imagenet_sorl.yaml`:

- `data.params.train.params.training_images_list_file`: training list;
- `data.params.validation.params.test_images_list_file`: validation list;
- `data.params.batch_size`: per-device batch size;
- `lightning.trainer.devices`: number of GPUs.

The paths committed in the YAML are examples and must be replaced.

## Train

From this directory, run:

```bash
python main.py --base configs/imagenet_sorl.yaml --train true
```

Useful launcher options include:

- `--name EXPERIMENT_NAME` to name the run;
- `--resume /path/to/run_or_checkpoint` to resume;
- `--project /path/to/logs` to change the output root.

Checkpoints are written below the configured log directory. A healthy run reports reconstruction and discriminator losses and periodically writes input/reconstruction grids similar to [`images/start.png`](images/start.png).

## Output used by stage two

Keep the original Lightning `.ckpt` file. Stage two uses it in two ways:

1. The paper's classic LDM path loads it through `ldm.models.autoencoder.SORLModelInterface`; see [`../ldm`](../ldm).
2. The optional SiT path converts it to Diffusers format for offline latent extraction; see [`../sit`](../sit).

The CelebA-HQ LDM path assumes `embed_dim: 3`, `ch_mult: [1, 2, 4]`, and 4× spatial downsampling. The ImageNet/SiT path assumes `embed_dim: 4`, `ch_mult: [1, 2, 4, 4]`, and 8× spatial downsampling. Changing these values requires matching the selected stage-two configuration.

## Notes on provenance

This directory is based substantially on Taming Transformers rather than being an independent implementation. Preserve the upstream copyright and license notice when redistributing it. See the repository-level [third-party notice](../THIRD_PARTY.md).
