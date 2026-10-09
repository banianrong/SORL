# When Local Rules Create Global Order

## Self-Organized Representation Learning for Latent Diffusion Models

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue)](https://openaccess.thecvf.com/content/CVPR2026/html/Lian_When_Local_Rules_Create_Global_Order_Self-Organized_Representation_Learning_for_CVPR_2026_paper.html)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://openaccess.thecvf.com/content/CVPR2026/papers/Lian_When_Local_Rules_Create_Global_Order_Self-Organized_Representation_Learning_for_CVPR_2026_paper.pdf)
[![Project Page](https://img.shields.io/badge/Project-Page-bdf86b)](https://banianrong.github.io/SORL/)

Official PyTorch implementation of **SORL**, accepted at **CVPR 2026**.

Junrong Lian · Weijian Deng · Pengxu Wei · Yaqin Chen · Qixiang Ye · Liang Lin

### Project links

🌐 **[Project Page](https://banianrong.github.io/SORL/)** · 📄 **[Paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Lian_When_Local_Rules_Create_Global_Order_Self-Organized_Representation_Learning_for_CVPR_2026_paper.pdf)** · 💻 **[Code](https://github.com/banianrong/SORL)**

The project page provides a visual overview of the method, the two-stage SORL-to-LDM pipeline, key experimental results, and citation information. Its source is available in [`docs/`](docs/).

## Abstract

Latent diffusion models depend strongly on the structure learned by their first-stage autoencoder. An effective latent space should be locally smooth, so nearby codes decode consistently, while remaining globally dispersive enough to support diverse sampling. Existing representation models often favor smoothness at the cost of concentrating samples in a narrow region.

SORL is a bottom-up representation-learning framework in which global latent organization emerges from two local mechanisms: **local attraction** encourages coherent representations among nearby codes, while **local repulsion** prevents the representation space from collapsing into dense clusters. Their interaction produces a latent manifold that combines local smoothness with global dispersity, improving both reconstruction and downstream generation.

## Code structure

```text
SORL/
├── sovae/   Stage 1: self-organized autoencoder and baseline models
├── ldm/     Stage 2: CompVis latent-diffusion connected to SORL
└── sit/     Additional ImageNet/SiT experiments
```

The paper reproduction path is:

```text
training images
      │
      ▼
sovae/configs/OM_learn_sigma.yaml
      │  local attraction + local repulsion
      ▼
SORL first-stage checkpoint
      │
      ▼
ldm/configs/latent-diffusion/sorl-celebahq-ldm.yaml
      │  frozen SORL encoder/decoder
      ▼
unconditional latent diffusion model
```

The first stage is based on [Taming Transformers](https://github.com/CompVis/taming-transformers). The second stage is based on the official [CompVis latent-diffusion](https://github.com/CompVis/latent-diffusion) implementation. The repository retains the upstream licenses and describes code provenance in [THIRD_PARTY.md](THIRD_PARTY.md).

## Results

### CelebA-HQ unconditional generation

| Method | FID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
| --- | ---: | ---: | ---: | ---: |
| VAE | 13.02 | 3.30 | 0.33 | 0.63 |
| RV-VAE | 16.21 | 3.19 | 0.22 | 0.62 |
| EQ-VAE | 15.80 | 3.20 | 0.21 | 0.64 |
| VA-VAE | 53.15 | 3.18 | 0.22 | 0.38 |
| **SORL** | **9.74** | **3.34** | **0.47** | **0.65** |

SORL also improves reconstruction quality across ADE20K, LSUN-Churches, CelebA-HQ, and FFHQ, and shows stronger cross-domain and cross-resolution reconstruction behavior. Refer to the [paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Lian_When_Local_Rules_Create_Global_Order_Self-Organized_Representation_Learning_for_CVPR_2026_paper.pdf) for complete comparisons and ablations.

## Installation

The two stages preserve their original dependency stacks and are installed separately.

Stage one:

```bash
cd sovae
conda env create -f environment.yaml
conda activate sorl
pip install -e .
```

Stage two:

```bash
cd ../ldm
conda env create -f environment.yaml
conda activate ldm
pip install -e .
pip install -e ../sovae
```

## Stage 1: train SORL

Prepare training and validation file lists as described in the [stage-one guide](sovae/README.md), update the dataset paths in `sovae/configs/OM_learn_sigma.yaml`, and run:

```bash
cd sovae
python main.py \
  --base configs/OM_learn_sigma.yaml \
  --train true \
  --project logs
```

The imported `Extension_VQGAN` configurations for VAE, WAE, RV-VAE, VQGAN, and other development baselines are also available under `sovae/configs/`.

## Stage 2: train latent diffusion

Update the first-stage checkpoint and CelebA-HQ file-list paths in `ldm/configs/latent-diffusion/sorl-celebahq-ldm.yaml`, then run:

```bash
cd ldm
python main.py \
  --base configs/latent-diffusion/sorl-celebahq-ldm.yaml \
  --train true \
  --gpus 0,
```

The integration class `ldm.models.autoencoder.SORLModelInterface` loads the SORL encoder, decoder, latent projection, and post-projection while excluding the stage-one discriminator and local-rule training modules. Detailed training and sampling commands are in the [LDM guide](ldm/README.md).

## Additional SiT experiments

The [`sit`](sit) directory contains a separate ImageNet pipeline using SiT and offline latent preprocessing. It is retained as an additional experiment and is not the default second-stage implementation described above.

## Citation

```bibtex
@InProceedings{Lian_2026_CVPR,
  author    = {Lian, Junrong and Deng, Weijian and Wei, Pengxu and Chen, Yaqin and Ye, Qixiang and Lin, Liang},
  title     = {When Local Rules Create Global Order: Self-Organized Representation Learning for Latent Diffusion Models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026},
  pages     = {9445--9454}
}
```

## Acknowledgements

This project builds on [Taming Transformers](https://github.com/CompVis/taming-transformers) and [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion). We also thank the authors of the baseline autoencoders and evaluation tools used in the paper. Please preserve the corresponding license and attribution notices when redistributing this repository.
