# Third-party software and attribution

SORL is a research code release assembled from and extending several open-source projects. This file is a provenance guide, not a replacement for the license text distributed with each upstream project.

## Stage one

The `sovae/` implementation is based substantially on **Taming Transformers for High-Resolution Image Synthesis** by Patrick Esser, Robin Rombach, and Björn Ommer:

- Project: <https://github.com/CompVis/taming-transformers>
- License: MIT; a copy is included at `sovae/LICENSE`.
- SORL changes include the local-rule quantization/distribution modules, model variants, configuration, and integration with the stage-two latent pipeline.

## Stage two

The `sit/` implementation includes or adapts code from the following projects:

- **Latent Diffusion Models** — vendored and adapted under `ldm/`; <https://github.com/CompVis/latent-diffusion>. Its MIT license is retained at `ldm/LICENSE`.
- **SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers** — optional experiments under `sit/`; <https://github.com/willisma/SiT>
- **OpenAI CLIP** — vendored under `sit/CLIP/`; its MIT license is at `sit/CLIP/LICENSE`.
- **torch-fidelity** — vendored under `sit/torch-fidelity/`; its license is at `sit/torch-fidelity/LICENSE.md`.
- NVIDIA preprocessing utilities derived from EDM-style dataset tooling; source files retain their Creative Commons Attribution-NonCommercial-ShareAlike 4.0 notices.

Additional source-level references and copyright notices are preserved in the relevant files. Users are responsible for complying with all applicable upstream licenses, especially the non-commercial terms attached to NVIDIA-derived preprocessing code.

## Datasets and checkpoints

ImageNet is not distributed with this repository and is subject to its own terms. Third-party pretrained encoders and evaluation weights are likewise governed by their respective licenses and usage conditions.
