import lightning as L  # type: ignore[import-not-found]
import pyro.distributions as dists
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from zuko.distributions import DiagNormal
from zuko.lazy import UnconditionalDistribution


class LightningVAE(L.LightningModule):
    def __init__(self, vae, lr, batch_size, beta, vae_ckpt_path=None, ckpt_path=None):
        # ckpt_path is for create_lightning_loader
        # it not as clean as it could be. Focus on single or double ckpt_path arg.
        super().__init__()
        self.vae = vae  # torch.compile(vae)
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta
        self.mse = F.mse_loss
        self.ckpt_path = ckpt_path

        self.alpha = torch.tensor(100.0)  # recon loss weight

        if vae_ckpt_path:
            state_dict = torch.load(
                vae_ckpt_path  # ,
                # map_location="cpu"
            )["state_dict"]
            state_dict = {k.replace("vae.", "", 1): v for k, v in state_dict.items()}
            self.vae.load_state_dict(state_dict, strict=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.vae.parameters(), lr=self.lr)

    def base_step(self, batch, partition):
        output = self.vae(batch["X"])
        # z = self.vae.reparametrize(output["mu"], output["log_var"])
        recon_loss = self.alpha * self.mse(
            output["recon"], batch["X"], reduction="mean"
        )

        kl_loss = torch.sum(
            -0.5
            * (1 + output["log_var"] - output["log_var"].exp() - output["mu"].pow(2)),
            axis=1,
        ).mean()

        loss = recon_loss + self.beta * kl_loss

        self.log(f"{partition}_loss", loss.mean(), sync_dist=True)
        self.log(f"{partition}_kl_loss", kl_loss, sync_dist=True)
        self.log(f"{partition}_recon_loss", recon_loss, sync_dist=True)
        return loss

    def training_step(self, batch, _batch_idx):
        return self.base_step(batch, "train")

    def validation_step(self, batch, _batch_idx):
        return self.base_step(batch, "val")

    def test_step(self, batch):
        return self.base_step(batch, "test")

    def predict_step(self, batch):
        output = self.vae(batch["X"])

        return {
            "X": batch["X"],
            "recon": output["recon"],
            "z": output["z"].flatten(start_dim=1),
            "catalog": batch["catalog"],
            "label": batch["label"],
        }