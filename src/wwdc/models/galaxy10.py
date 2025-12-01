import lightning as L
import torch
import torch.distributions as D
from diffusers import AutoencoderKL
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper, gradient
from timm.models.layers import trunc_normal_
from torch import Tensor, nn
from torchcfm.models.unet import UNetModel


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        n_layers = 4
        block_out_channels = [32, 64, 128, 256]
        self.latent_dim = 4

        self.vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=n_layers * ("DownEncoderBlock2D",),
            up_block_types=n_layers * ("UpDecoderBlock2D",),
            layers_per_block=4,
            latent_channels=self.latent_dim,
            block_out_channels=block_out_channels,
            norm_num_groups=4,
        )

        self.vae.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d | nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor):
        z_dist = self.vae.encode(x).latent_dist
        z = z_dist.sample()
        x = self.vae.decode(z)

        return {
            "recon": x["sample"],
            "z": z,
            "mu": z_dist.mean,
            "log_var": z_dist.logvar,
        }

class VF(nn.Module):
    def __init__(self):
        super().__init__()

        self.vf = UNetModel(
            channel_mult=[1, 2, 2, 2],
            dim=(4, 32, 32),
            num_channels=64,
            num_res_blocks=1,
            num_classes=11,
            class_cond=True,
            # use_scale_shift_norm=True
        )

    def forward(self, x_t, t, y, p_mask):
        bs, _, _, _ = x_t.shape
        if t.ndim == 0:
            t = t.expand(x_t.shape[0])

        mask = torch.rand(bs, device=x_t.device)
        y = torch.where(
            mask < p_mask, torch.tensor(10, device=x_t.device), y
        )  # replacement as dropout.
        return self.vf(t, x_t, y)


class WrappedModel(ModelWrapper):
    def forward(self, t: torch.Tensor, x: torch.Tensor, **model_extras):
        return self.model(x_t=x, t=t, **model_extras)


class LightningFlowMatching(L.LightningModule):
    def __init__(
        self,
        vae,
        lr,
        batch_size,
        size,
        hidden_dim,
        vae_ckpt_path,
        catalog,
        ckpt_path=None,
    ):
        super().__init__()
        self.vae = vae
        self.p_mask = 0.1

        self.vf = VF()  # (size, hidden_dim, catalog)
        # self.vf.apply(self._init_weights)

        self.hidden_dim = hidden_dim

        self.batch_size = batch_size
        self.size = size

        if ckpt_path:
            self.vf_state_dict = torch.load(ckpt_path)[
                "state_dict"
            ]  # map_location="cpu"
            self.load_state_dict(self.vf_state_dict, strict=False)
            print("✅ Loaded state dict from checkpoint.")
        else:
            self.vf.apply(self._init_weights)

        # ODE solver hparams
        self.step_size = 0.001
        self.n_steps = 100
        self.T = torch.linspace(1, 0, self.n_steps, device=self.device)
        self.solver = ODESolver(velocity_model=WrappedModel(self.vf))
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        try:
            state_dict = torch.load(vae_ckpt_path)["state_dict"]
            state_dict = {k.replace("vae.", "", 1): v for k, v in state_dict.items()}
            self.vae.load_state_dict(state_dict, strict=True)
            print(f"✅ Loaded VAE state dict from checkpoint: {vae_ckpt_path}")
        except Exception as e:
            print(f"Error loading VAE state dict: {e}")

        self.vae.eval()

        self.lr = lr
        self.catalog = catalog

    def _init_weights(self, m):
        if isinstance(m, nn.Linear | nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        params = list(self.vf.parameters())

        return torch.optim.AdamW(
            params,
            lr=self.lr,
        )

    def base_step(self, batch, partition):
        output = self.vae(batch["X"])
        z = output["z"]  # * 0.18215

        x_0 = torch.randn_like(z)
        t = torch.rand(z.shape[0], device=z.device)
        # sample probability path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=z)
        # flow matching l2 loss

        loss = torch.pow(
            self.vf(
                x_t=path_sample.x_t,
                y=batch["label"],
                t=path_sample.t,
                p_mask=self.p_mask,
            )
            - path_sample.dx_t,
            2,
        ).mean()

        print(f"loss: {loss}")
        self.log(f"{partition}_loss", loss)

        return loss

    def training_step(self, batch):
        return self.base_step(batch, "train")

    def validation_step(self, batch):
        return self.base_step(batch, "val")

    def test_step(self, batch):
        return self.base_step(batch, "test")

    def predict_step(self, batch):
        self.eval()
        with torch.no_grad():
            vae_output = self.vae(batch["X"])
            z = vae_output["z"]  # * 0.18215

            gaussian_log_density = D.Independent(
                D.Normal(
                    torch.zeros_like(z),
                    torch.ones_like(z),
                ),
                3,
            ).log_prob

            output_cond = self.solver.compute_likelihood(
                x_1=z,
                y=batch["label"],  
                p_mask=0.0,
                time_grid=self.T,
                method="midpoint",
                step_size=None,
                exact_divergence=False,
                log_p0=gaussian_log_density,
                return_intermediates=True,
            )

            output_uncond = self.solver.compute_likelihood(
                x_1=z,
                y=batch["label"], 
                p_mask=1.0,
                time_grid=self.T,
                method="midpoint",
                step_size=None,
                exact_divergence=False,
                log_p0=gaussian_log_density,
                return_intermediates=True,
            )

        return {
            "X": batch["X"],
            "z": z,
            "output_cond": output_cond,
            "output_uncond": output_uncond,
            "y": batch["y"],
            "catalog": batch["catalog"],
            "label": batch["label"],
        }

if __name__ == "__main__":
    import torch
    from diffusers import AutoencoderKL

    """n_layers = 9
    block_out_channels = [2*2**i for i in range(n_layers)]

    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=n_layers*(
            "DownEncoderBlock2D",
        ),
        up_block_types=n_layers*(
            "UpDecoderBlock2D",
        ),
        latent_channels=512,
        block_out_channels=block_out_channels,
        norm_num_groups=2
    )"""

    x = torch.randn(32, 3, 256, 256)
    """print(dir(vae.encode(x).latent_dist))
    z = vae.encode(x).latent_dist.sample()
    x_recon = vae.decode(z)"""

    # print(x_recon["sample"].shape)

    vae = VAE()
    out = vae(x)
    print(out["z"].shape)
    print(out["recon"].shape)
    print(out["mu"].shape)
    print(out["log_var"].shape)
