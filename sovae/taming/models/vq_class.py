import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config
from taming.modules.diffusionmodules.model import Encoder, Decoder
# from taming.modules.diffusionmodules.interval_vq import Encoder
from taming.modules.vqvae.quantize_test import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer
from taming.modules.distributions.distributions import DiagonalGaussianDistribution, WAEDistribution, CodebookDistribution
from taming.modules.diffusionmodules import rv_model as rv


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 quantizeconfig,
                 lossconfig,
                 scale,
                 partitions,
                 n_embed,
                 embed_dim,
                 class_num=10,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 label_key="file_path_",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.label_key = label_key
        self.encoder = Encoder(**ddconfig)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(ddconfig["z_channels"]*((ddconfig["resolution"] // (2**(len(tuple(ddconfig["ch_mult"]))-1)))**2), class_num)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        # self.quantize = VectorQuantizer(scale, partitions, n_embed, embed_dim, beta=0.25,
        #                                 remap=remap, sane_index_shape=sane_index_shape)
        self.quantize = instantiate_from_config(quantizeconfig)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        z = self.flatten(quant)
        dec = self.linear(z)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        # print(f"finished encoding")
        dec = self.decode(quant)
        # print(f"finished decoding")
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x = self.get_input(batch, self.image_key)
        labels = batch[self.label_key]
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(0, labels, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/cross_loss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        labels = batch[self.label_key]
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(0, labels, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        rec_loss = log_dict_ae["val/cross_loss"]
        self.log("val/cross_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
                                #   list(self.encoder.parameters())+
                                #   list(self.decoder.parameters())+
                                #   list(self.quantize.parameters())+
                                #   list(self.quant_conv.parameters())+
                                #   list(self.post_quant_conv.parameters()),
                                  list(self.linear.parameters()), 
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    # def log_images(self, batch, **kwargs):
    #     log = dict()
    #     x = self.get_input(batch, self.image_key)
    #     x = x.to(self.device)
    #     xrec, _ = self(x)
    #     if x.shape[1] > 3:
    #         # colorize with random projection
    #         assert xrec.shape[1] > 3
    #         x = self.to_rgb(x)
    #         xrec = self.to_rgb(xrec)
    #     log["inputs"] = x
    #     log["reconstructions"] = xrec
    #     return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


class AEModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 quantizeconfig,
                 lossconfig,
                 scale,
                 partitions,
                 n_embed,
                 embed_dim,
                 class_num=10,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 label_key="file_path_",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.label_key = label_key
        self.encoder = Encoder(**ddconfig)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(ddconfig["z_channels"]*((ddconfig["resolution"] // (2**(len(tuple(ddconfig["ch_mult"]))-1)))**2), class_num)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        # self.quantize = VectorQuantizer(scale, partitions, n_embed, embed_dim, beta=0.25,
        #                                 remap=remap, sane_index_shape=sane_index_shape)
        # self.quantize = instantiate_from_config(quantizeconfig)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        # quant, emb_loss, info = self.quantize(h)
        # return quant, emb_loss, info
        return h, None, None

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        z = self.flatten(quant)
        dec = self.linear(z)
        return dec

    # def decode_code(self, code_b):
    #     quant_b = self.quantize.embed_code(code_b)
    #     dec = self.decode(quant_b)
    #     return dec

    def forward(self, input):
        quant, _, _ = self.encode(input)
        # print(f"finished encoding")
        dec = self.decode(quant)
        # print(f"finished decoding")
        return dec, _

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x = self.get_input(batch, self.image_key)
        labels = batch[self.label_key]
        xrec, _ = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(0, labels, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/cross_loss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        labels = batch[self.label_key]
        xrec, _ = self(x)
        aeloss, log_dict_ae = self.loss(0, labels, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        rec_loss = log_dict_ae["val/cross_loss"]
        self.log("val/cross_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
                                #   list(self.encoder.parameters())+
                                #   list(self.decoder.parameters())+
                                #   # list(self.quantize.parameters())+
                                #   list(self.quant_conv.parameters())+
                                #   list(self.post_quant_conv.parameters()),
                                  list(self.linear.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return [opt_ae], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    # def log_images(self, batch, **kwargs):
    #     log = dict()
    #     x = self.get_input(batch, self.image_key)
    #     x = x.to(self.device)
    #     xrec, _ = self(x)
    #     if x.shape[1] > 3:
    #         # colorize with random projection
    #         assert xrec.shape[1] > 3
    #         x = self.to_rgb(x)
    #         xrec = self.to_rgb(xrec)
    #     log["inputs"] = x
    #     log["reconstructions"] = xrec
    #     return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 class_num=10,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 label_key="file_path_",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.label_key = label_key
        self.encoder = Encoder(**ddconfig)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(ddconfig["z_channels"]*((ddconfig["resolution"] // (2**(len(tuple(ddconfig["ch_mult"]))-1)))**2), class_num)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        z = self.flatten(z)
        dec = self.linear(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        inputs = self.get_input(batch, self.image_key)
        labels = batch[self.label_key]
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(0, labels, reconstructions, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("cross_loss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        labels = batch[self.label_key]
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(0, labels, reconstructions, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/cross_loss", log_dict_ae["val/cross_loss"])
        self.log_dict(log_dict_ae)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
                                #   list(self.encoder.parameters())+
                                #   list(self.decoder.parameters())+
                                #   list(self.quant_conv.parameters())+
                                #   list(self.post_quant_conv.parameters()),
                                  list(self.linear.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return [opt_ae], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    # @torch.no_grad()
    # def log_images(self, batch, only_inputs=False, **kwargs):
    #     log = dict()
    #     x = self.get_input(batch, self.image_key)
    #     x = x.to(self.device)
    #     if not only_inputs:
    #         xrec, posterior = self(x)
    #         if x.shape[1] > 3:
    #             # colorize with random projection
    #             assert xrec.shape[1] > 3
    #             x = self.to_rgb(x)
    #             xrec = self.to_rgb(xrec)
    #         log["samples"] = self.decode(torch.randn_like(posterior.sample()))
    #         log["reconstructions"] = xrec
    #     log["inputs"] = x
    #     return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class WAEModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 quantizeconfig,
                 lossconfig,
                 scale,
                 partitions,
                 n_embed,
                 embed_dim,
                 class_num=10,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 label_key="file_path_",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.label_key = label_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        # self.quantize = VectorQuantizer(scale, partitions, n_embed, embed_dim, beta=0.25,
        #                                 remap=remap, sane_index_shape=sane_index_shape)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(ddconfig["z_channels"]*((ddconfig["resolution"] // (2**(len(tuple(ddconfig["ch_mult"]))-1)))**2), class_num)
        self.quantize = instantiate_from_config(quantizeconfig)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        quant = WAEDistribution(quant)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        z = self.flatten(quant)
        dec = self.linear(z)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, sample_posterior=True):
        quant, diff, _ = self.encode(input)
        # print(f"finished encoding")
        if sample_posterior:
            quant = quant.sample()
        else:
            quant = quant.mode()
        dec = self.decode(quant)
        # print(f"finished decoding")
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x = self.get_input(batch, self.image_key)
        labels = batch[self.label_key]
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(0, labels, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")

            self.log("train/cross_loss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        labels = batch[self.label_key]
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(0, labels, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        rec_loss = log_dict_ae["val/cross_loss"]
        self.log("val/cross_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
    
        self.log_dict(log_dict_ae)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
                                #   list(self.encoder.parameters())+
                                #   list(self.decoder.parameters())+
                                #   list(self.quantize.parameters())+
                                #   list(self.quant_conv.parameters())+
                                #   list(self.post_quant_conv.parameters())
                                  list(self.linear.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return [opt_ae], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    # def log_images(self, batch, **kwargs):
    #     log = dict()
    #     x = self.get_input(batch, self.image_key)
    #     x = x.to(self.device)
    #     xrec, _ = self(x)
    #     if x.shape[1] > 3:
    #         # colorize with random projection
    #         assert xrec.shape[1] > 3
    #         x = self.to_rgb(x)
    #         xrec = self.to_rgb(xrec)
    #     log["inputs"] = x
    #     log["reconstructions"] = xrec
    #     return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class AutoencoderCodebook(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 quantizeconfig,
                 lossconfig,
                 embed_dim,
                 class_num=10,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 label_key="file_path_",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.e_dim = embed_dim
        self.quantize = instantiate_from_config(quantizeconfig)
        self.image_key = image_key
        self.label_key = label_key
        self.encoder = Encoder(**ddconfig)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(ddconfig["z_channels"]*((ddconfig["resolution"] // (2**(len(tuple(ddconfig["ch_mult"]))-1)))**2), class_num)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = CodebookDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        z = self.flatten(z)
        dec = self.linear(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        inputs = self.get_input(batch, self.image_key)
        labels = batch[self.label_key]
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(0, labels, reconstructions, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("cross_loss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        labels = batch[self.label_key]
        reconstructions, posterior = self(inputs)

        _, kl_loss, _ = self.quantize(posterior.mode()) 

        aeloss, log_dict_ae = self.loss(kl_loss, labels, reconstructions, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        self.log("val/cross_loss", log_dict_ae["val/cross_loss"])
        self.log_dict(log_dict_ae)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
                                #   list(self.encoder.parameters())+
                                #   list(self.decoder.parameters())+
                                #   list(self.quantize.parameters())+
                                #   list(self.quant_conv.parameters())+
                                #   list(self.post_quant_conv.parameters())
                                  list(self.linear.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        
        return [opt_ae], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    # @torch.no_grad()
    # def log_images(self, batch, only_inputs=False, **kwargs):
    #     log = dict()
    #     x = self.get_input(batch, self.image_key)
    #     x = x.to(self.device)
    #     if not only_inputs:
    #         xrec, posterior = self(x)
    #         if x.shape[1] > 3:
    #             # colorize with random projection
    #             assert xrec.shape[1] > 3
    #             x = self.to_rgb(x)
    #             xrec = self.to_rgb(xrec)
    #         log["samples"] = self.decode(torch.randn_like(posterior.sample()))
    #         log["reconstructions"] = xrec
    #     log["inputs"] = x
    #     return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

# RV-VAE
class RVAutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 class_num=10,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 label_key="file_path_",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.label_key = label_key
        self.encoder = rv.Encoder(**ddconfig)
        self.decoder = rv.Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(ddconfig["z_channels"]*((ddconfig["resolution"] // (2**(len(tuple(ddconfig["ch_mult"]))-1)))**2), class_num)
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        z = self.flatten(z)
        dec = self.linear(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        inputs = self.get_input(batch, self.image_key)
        labels = batch[self.label_key]
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(0, labels, reconstructions, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("cross_loss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        labels = batch[self.label_key]
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(0, labels, reconstructions, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        self.log("val/cross_loss", log_dict_ae["val/cross_loss"])
        self.log_dict(log_dict_ae)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
                                #   list(self.encoder.parameters())+
                                #   list(self.decoder.parameters())+
                                #   list(self.quant_conv.parameters())+
                                #   list(self.post_quant_conv.parameters())
                                  list(self.linear.parameters()),
                                  lr=lr, betas=(0.5, 0.9))

        return [opt_ae], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    # @torch.no_grad()
    # def log_images(self, batch, only_inputs=False, **kwargs):
    #     log = dict()
    #     x = self.get_input(batch, self.image_key)
    #     x = x.to(self.device)
    #     if not only_inputs:
    #         xrec, posterior = self(x)
    #         if x.shape[1] > 3:
    #             # colorize with random projection
    #             assert xrec.shape[1] > 3
    #             x = self.to_rgb(x)
    #             xrec = self.to_rgb(xrec)
    #         log["samples"] = self.decode(torch.randn_like(posterior.sample()))
    #         log["reconstructions"] = xrec
    #     log["inputs"] = x
    #     return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x