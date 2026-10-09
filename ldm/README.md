# SORL latent diffusion: stage two

This directory contains the paper's second-stage unconditional Latent Diffusion Model (LDM). It is adapted from [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) and connects directly to a checkpoint trained in [`../sovae`](../sovae).

## Integration with stage one

The key integration class is `ldm.models.autoencoder.SORLModelInterface`. It reconstructs the frozen SORL encoder, decoder, latent projection, and post-projection from the stage-one checkpoint while ignoring the discriminator, reconstruction loss, and local-rule quantizer used only during first-stage training.

Its latent sampling follows stage-one SORL:

```text
image → encoder → (mean, log-variance)
                    ↓
          z = mean + std × Uniform(-1, 1)
                    ↓
             diffusion model
```

The default paper-oriented configuration is [`configs/latent-diffusion/sorl-celebahq-ldm.yaml`](configs/latent-diffusion/sorl-celebahq-ldm.yaml). The other configurations are retained from the development repository for baselines and additional LDM experiments.

## Environment

```bash
conda env create -f environment.yaml
conda activate ldm
pip install -e .
pip install -e ../sovae
```

Installing `../sovae` makes its `taming.data.custom` dataset classes available to the LDM configuration.

## Prepare the configuration

Edit these fields in `configs/latent-diffusion/sorl-celebahq-ldm.yaml`:

- `model.params.first_stage_config.params.ckpt_path`: stage-one SORL checkpoint;
- `data.params.train.params.training_images_list_file`: CelebA-HQ training list;
- `data.params.validation.params.test_images_list_file`: CelebA-HQ validation list;
- `data.params.batch_size`: per-device batch size.

The released CelebA-HQ setup uses 256×256 images, a spatial downsampling factor of four, and three latent channels. The stage-one checkpoint and LDM configuration must agree on `embed_dim`, `z_channels`, `ch_mult`, and `double_z`.

## Train

From this directory:

```bash
python main.py \
  --base configs/latent-diffusion/sorl-celebahq-ldm.yaml \
  --train true \
  --gpus 0,
```

For distributed training, expose the desired devices and use the launcher supported by the pinned PyTorch Lightning version. Logs and checkpoints are written below `logs/` by default.

Configuration values can also be overridden using OmegaConf dot-list arguments, for example:

```bash
python main.py \
  --base configs/latent-diffusion/sorl-celebahq-ldm.yaml \
  --train true \
  --gpus 0, \
  model.params.first_stage_config.params.ckpt_path=/checkpoints/sorl.ckpt
```

## Sample and evaluate

Use the upstream-compatible unconditional sampling script:

```bash
python scripts/sample_diffusion.py \
  --resume /path/to/ldm/logdir-or-checkpoint \
  --config configs/latent-diffusion/sorl-celebahq-ldm.yaml \
  --n_samples 50000 \
  --batch_size 16 \
  --custom_steps 200 \
  --eta 0.0
```

Run `python scripts/sample_diffusion.py --help` because older checkpoints may use the short-form argument names inherited from latent-diffusion.

The paper evaluates unconditional CelebA-HQ generation with FID, Inception Score, Precision, and Recall. Reported SORL results are FID **9.74**, IS **3.34**, Precision **0.47**, and Recall **0.65**.

## Baseline configurations

The imported development configurations include VAE/WAE/VQ first-stage variants and standard LDM tasks. Several of these files preserve historical absolute paths and are provided for provenance rather than as ready-to-run SORL configurations. Use `sorl-celebahq-ldm.yaml` for the integrated paper pipeline.

## License

This directory retains the original latent-diffusion MIT license in [`LICENSE`](LICENSE). SORL-specific modifications are described in the repository-level [`THIRD_PARTY.md`](../THIRD_PARTY.md).
