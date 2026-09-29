"""Use the upstream CSDI denoiser with a fixed forecasting mask."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from main_model import CSDI_base


class DLinear(nn.Module):
    """Channel-independent trend/seasonal linear forecast, shared or individual."""

    def __init__(self, history, horizon, variables, kernel=25, individual=True):
        super().__init__()
        if kernel < 1 or kernel % 2 != 1:
            raise ValueError("DLinear kernel must be a positive odd integer")
        self.kernel = kernel
        self.individual = individual
        count = variables if individual else 1
        self.trend = nn.ModuleList([nn.Linear(history, horizon) for _ in range(count)])
        self.seasonal = nn.ModuleList([nn.Linear(history, horizon) for _ in range(count)])
        for layer in list(self.trend) + list(self.seasonal):
            nn.init.constant_(layer.weight, 1 / history)
            nn.init.zeros_(layer.bias)

    def forward(self, history):
        # Public shape convention throughout this package: [batch, time, variable].
        x = history.transpose(1, 2)
        pad = (self.kernel - 1) // 2
        trend = F.avg_pool1d(F.pad(x, (pad, pad), mode="replicate"), self.kernel, stride=1)
        seasonal = x - trend
        if self.individual:
            result = torch.stack([
                self.trend[i](trend[:, i]) + self.seasonal[i](seasonal[:, i])
                for i in range(x.shape[1])
            ], dim=1)
        else:
            result = self.trend[0](trend) + self.seasonal[0](seasonal)
        return result.transpose(1, 2)


class ForecastCSDI(CSDI_base):
    """Matched denoiser for Y or Y-mu; both condition on the same raw history.

    The frozen baseline is applied outside this class. There is no additional
    baseline conditioning channel, no feature subsampling, and no residual scaler.
    """

    def __init__(self, variables, config, device):
        diff = config["diffusion"]
        if config["model"]["is_unconditional"]:
            raise ValueError("The step study requires history-conditioned diffusion")
        if diff["schedule"] != "vp_continuous" and (diff["schedule"] not in ("linear", "quad") or not 0 < diff["beta_start"] <= diff["beta_end"] < 1):
            raise ValueError("Use a linear/quad schedule with 0 < beta_start <= beta_end < 1")
        if diff["num_steps"] < 2:
            raise ValueError("At least two training diffusion steps are required")
        super().__init__(variables, copy.deepcopy(config), device)
        self.register_buffer("alpha_bar", torch.tensor(self.alpha, dtype=torch.float32))
        self.register_buffer("betas", torch.tensor(self.beta, dtype=torch.float32))
        if diff["schedule"] == "vp_continuous":
            # Encode the same normalized time with the same features across T.
            embedding = self.diffmodel.diffusion_embedding
            dim = diff["diffusion_embedding_dim"] // 2
            u = torch.arange(1, self.num_steps + 1, dtype=torch.float32) / self.num_steps
            frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0)
            phase = (1000 * u[:, None]) * frequencies[None, :]
            embedding.embedding = torch.cat([phase.sin(), phase.cos()], dim=1)

    def context(self, history, horizon):
        b, length, variables = history.shape
        mask = history.new_zeros(b, variables, length + horizon)
        mask[:, :, :length] = 1
        positions = torch.arange(length + horizon, device=history.device).expand(b, -1)
        side = self.get_side_info(positions, mask)
        observed = F.pad(history.transpose(1, 2), (0, horizon))
        return observed, mask, side

    def predict_noise(self, noisy_future, history, context, t):
        observed, mask, side = context
        noisy = F.pad(noisy_future.transpose(1, 2), (history.shape[1], 0))
        inputs = self.set_input_to_diffmodel(noisy, observed, mask)
        return self.diffmodel(inputs, side, t)[:, :, history.shape[1]:].transpose(1, 2)

    def loss(self, history, target):
        t = torch.randint(self.num_steps, (len(history),), device=history.device)
        noise = torch.randn_like(target)
        alpha = self.alpha_bar[t, None, None]
        noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
        predicted = self.predict_noise(noisy, history, self.context(history, target.shape[1]), t)
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def sample_ddim(self, history, initial_noise, steps):
        """DDIM eta=0, exactly `steps` NFE, always traversing to clean time.

        initial_noise: [batch, samples, horizon, variables]; caller reuses it for
        every step count and both models. Does not receive future observations.
        """
        if not 1 <= steps <= self.num_steps:
            raise ValueError("sampling steps must be between 1 and training steps")
        if self.training:
            raise RuntimeError("Call eval() before sampling (CSDI contains dropout)")
        b, samples, horizon, variables = initial_noise.shape
        history = history.repeat_interleave(samples, dim=0)
        current = initial_noise.reshape(b * samples, horizon, variables).clone()
        context = self.context(history, horizon)
        times = torch.linspace(self.num_steps - 1, 0, steps).round().long().tolist()
        for index, time in enumerate(times):
            previous = times[index + 1] if index + 1 < len(times) else -1
            alpha = self.alpha_bar[time]
            next_alpha = self.alpha_bar[previous] if previous >= 0 else current.new_tensor(1.)
            t = torch.full((len(history),), time, device=history.device, dtype=torch.long)
            noise = self.predict_noise(current, history, context, t)
            clean = (current - (1 - alpha).sqrt() * noise) / alpha.sqrt()
            current = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * noise
        return current.reshape(b, samples, horizon, variables)

    @torch.no_grad()
    def sample_ddpm(self, history, initial_noise, generator):
        """Full ancestral DDPM chain: exactly T evaluations, posterior variance.

        Uses a supplied CPU generator for reproducible, paired reverse noise.
        The initial Gaussian is supplied separately and shared across T and arms.
        """
        if self.training:
            raise RuntimeError("Call eval() before sampling (CSDI contains dropout)")
        b, samples, horizon, variables = initial_noise.shape
        history = history.repeat_interleave(samples, dim=0)
        current = initial_noise.reshape(b * samples, horizon, variables).clone()
        context = self.context(history, horizon)
        for time in range(self.num_steps - 1, -1, -1):
            alpha = self.alpha_bar[time]
            beta = self.betas[time]
            t = torch.full((len(history),), time, device=history.device, dtype=torch.long)
            predicted = self.predict_noise(current, history, context, t)
            current = (current - beta / (1 - alpha).sqrt() * predicted) / (1 - beta).sqrt()
            if time > 0:
                variance = beta * (1 - self.alpha_bar[time - 1]) / (1 - alpha)
                noise = torch.randn(current.shape, generator=generator, dtype=current.dtype).to(current.device)
                current = current + variance.sqrt() * noise
        return current.reshape(b, samples, horizon, variables)
