import lightning as L
import torch
import torch.distributions as D
import torch.nn as nn
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from timm.models.layers import trunc_normal_
from torch import Tensor

from wwdc.models.modules import get_conditional_len

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
        )

        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
        )

        self.encoder_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
        )
        self.encoder_block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
        )

        self.layers = nn.Sequential(
            self.encoder_block1,
            self.encoder_block2,
            self.encoder_block3,
            self.encoder_block4,
        )

    def forward(self, x: Tensor):
        for block in self.layers:
            for layer in block:
                if isinstance(layer, nn.MaxPool2d):
                    x, _ = layer(x)
                else:
                    x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_output_dim = 512

        self.block1 = nn.Sequential(
            nn.Upsample(size=3, mode="nearest"),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Upsample(size=7, mode="nearest"),
            nn.ConvTranspose2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            nn.Upsample(size=14, mode="nearest"),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.block4 = nn.Sequential(
            nn.Upsample(size=28, mode="nearest"),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
            ),
            nn.Sigmoid(),
        )

        self.layers = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
        )

    def forward(self, z: Tensor):
        x = z.view(z.size(0), 512, 1, 1)
        for block in self.layers:
            for layer in block:
                x = layer(x)
        return x


class VAE(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.encoder_output_dim = 512

        self.project_to_z_dist = nn.Sequential(
            nn.Linear(self.encoder_output_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim),
        )

        # before decoding.
        self.projection_up = nn.Sequential(
            nn.Linear(self.hidden_dim, self.encoder_output_dim)
        )

    def reparametrize(self, mu, logvar):
        return mu + torch.randn_like(logvar) * torch.exp(logvar * 0.5)

    def forward(self, x):
        for block in self.encoder.layers:
            for layer in block:
                if isinstance(layer, nn.MaxPool2d):
                    x, _ = layer(x)
                else:
                    x = layer(x)

        x = x.view(x.size(0), -1)

        x = self.project_to_z_dist(x)

        # sample from the latent network
        mu, log_var = x.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, -30.0, 20.0)

        z = self.reparametrize(mu, log_var).view(mu.size(0), -1)

        z_ = self.projection_up(z)  # 64 -> 512
        x = z_.view(z_.size(0), self.encoder_output_dim, 1, 1)
        # decode
        for block in self.decoder.layers:
            x = block(x)
        x = x.view(x.size(0), 3, 28, 28)
        return {"z": z, "recon": x, "mu": mu, "log_var": log_var}

class MLP(nn.Module):
    def __init__(self, size, hidden_dim, catalog):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.catalog = catalog

        self.vf = nn.Sequential(  # vector field
            nn.Linear(size + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, size),
        )

        self.conditional_size = get_conditional_len(catalog)

        self.digit_embedding = nn.Embedding(
            num_embeddings=10,
            embedding_dim=10,
        )

        self.FiLM_params = nn.Linear(self.conditional_size, 2 * hidden_dim)

        self.null_y = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.conditional_size,
        )

    def forward(self, t: Tensor, x_t: Tensor, y: Tensor, p_mask: Tensor):
        if t.ndim == 0:
            t = t.expand(x_t.shape[0])

        conditioning_vars = []

        for k, v in y.items():
            if k not in self.catalog["drop_variables"]:
                if k == "digit":
                    conditioning_vars.append(self.digit_embedding(v))
                else:
                    conditioning_vars.append(v)

        conditioning_vars = torch.cat(conditioning_vars, dim=1)

        mask = torch.rand(conditioning_vars.size(0), 1, device=x_t.device)
        conditioning_vars = torch.where(
            mask < p_mask,
            self.null_y(
                torch.zeros(
                    conditioning_vars.size(0), dtype=torch.long, device=x_t.device
                )
            ),
            conditioning_vars,
        )

        gamma, beta = self.FiLM_params(conditioning_vars).chunk(2, dim=1)

        x_t = torch.cat([x_t, t.unsqueeze(-1)], dim=-1)

        for idx, layer in enumerate(self.vf):
            x_t = layer(x_t)
            if idx != len(self.vf) - 1 and isinstance(layer, nn.Linear):
                x_t = gamma * x_t + beta  # add conditioning and time information
        return x_t


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

        self.vf = MLP(size, hidden_dim, catalog)
        self.vf.apply(self._init_weights)

        self.hidden_dim = hidden_dim

        self.batch_size = batch_size
        self.size = size

        state_dict = torch.load(vae_ckpt_path)["state_dict"]  # map_location="cpu"

        state_dict = {k: v for k, v in state_dict.items() if k.startswith("vae.")}

        state_dict = {k.replace("vae.", ""): v for k, v in state_dict.items()}

        self.vae.load_state_dict(state_dict, strict=False)
        self.vae.eval()

        if ckpt_path:
            self.vf_state_dict = torch.load(ckpt_path)[
                "state_dict"
            ]  # map_location="cpu"
            self.load_state_dict(self.vf_state_dict, strict=False)
            print("✅ Loaded state dict from checkpoint.")
            self.wrapped_vf = WrappedModel(self.vf)
            # ODE solver hparams
            self.step_size = 0.001
            self.n_steps = 100
            self.T = torch.linspace(1, 0, self.n_steps, device=self.device)
            self.solver = ODESolver(velocity_model=self.wrapped_vf)
            self.wrapped_vf = WrappedModel(self.vf)

        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.lr = lr
        self.catalog = catalog

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
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
        z = output["z"]

        x_0 = torch.randn_like(z)
        t = torch.rand(z.shape[0], device=z.device)
        # sample probability path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=z)
        # flow matching l2 loss
        loss = torch.pow(
            self.vf(
                x_t=path_sample.x_t, y=batch["catalog"], t=path_sample.t, p_mask=0.1
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

            gaussian_log_density = D.Independent(
                D.Normal(
                    torch.zeros(self.size, device=self.device),
                    torch.ones(self.size, device=self.device),
                ),
                1,
            ).log_prob

            output_cond = self.solver.compute_likelihood(
                x_1=vae_output["z"],
                y=batch["catalog"],
                p_mask=0.0,
                time_grid=self.T,
                method="midpoint",
                step_size=None,
                exact_divergence=False,
                log_p0=gaussian_log_density,
                return_intermediates=True,
            )

            output_uncond = self.solver.compute_likelihood(
                x_1=vae_output["z"],
                y=batch["catalog"],  
                p_mask=1.0,
                time_grid=self.T,
                method="midpoint",
                step_size=None,
                exact_divergence=False,
                log_p0=gaussian_log_density,
                return_intermediates=True,
            )

        return {
            "z": vae_output["z"],
            "output_cond": output_cond,
            "output_uncond": output_uncond,
            "y": batch["y"],
            "catalog": batch["catalog"],
        }


if __name__ == "__main__":
    model = VAE(hidden_dim=64)
    print(model)
