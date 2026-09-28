from __future__ import annotations

from copy import deepcopy

import torch

from main_model import CSDI_base


class ConditionalCSDI(CSDI_base):
    """Use CSDI's observed/missing channel interface, not a replacement backbone.

    Layout: [four energy targets, weather/calendar conditions, continuous year].
    All energy cells are missing conditions in both training and evaluation.
    Only their noise residual contributes to the original CSDI training loss.
    """

    def __init__(self, config, condition_channels):
        if config["data"]["seq_len"] != 24:
            raise ValueError("This protocol requires complete 24-hour days")
        diffusion = config["diffusion"]
        if diffusion["schedule"] not in ("quad", "linear"):
            raise ValueError("Use upstream quad or linear beta schedule")
        if not 0 < diffusion["beta_start"] <= diffusion["beta_end"] < 1:
            raise ValueError("Require 0 < beta_start <= beta_end < 1")
        if diffusion["num_steps"] < 2 or diffusion["layers"] < 1:
            raise ValueError("Require at least two diffusion steps and one residual layer")
        if diffusion["channels"] % diffusion["nheads"]:
            raise ValueError("channels must be divisible by nheads")
        if config["model"]["timeemb"] % 2 or diffusion["diffusion_embedding_dim"] < 4 or diffusion["diffusion_embedding_dim"] % 2:
            raise ValueError("Time embeddings must be even; diffusion embedding >= 4")
        if config["model"]["is_unconditional"]:
            raise ValueError("Baseline 5 requires conditional CSDI")
        super().__init__(condition_channels + 5, deepcopy(config), torch.device("cpu"))
        self.condition_channels = condition_channels
        # Upstream holds this as a plain tensor; register it for .to()/checkpoints.
        alpha_torch = self.alpha_torch
        del self.alpha_torch
        self.register_buffer("alpha_torch", alpha_torch)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.device = self.embed_layer.weight.device
        return result

    def pack_conditions(self, conditions, year, target=None):
        if conditions.ndim != 3 or conditions.shape[1:] != (self.condition_channels, 24):
            raise ValueError("conditions must be [batch, condition_channels, 24]")
        if year.shape != (len(conditions), 1, 24):
            raise ValueError("year must be [batch, 1, 24]")
        if target is None:
            target = conditions.new_zeros((len(conditions), 4, 24))
        if target.shape != (len(conditions), 4, 24):
            raise ValueError("target must be [batch, 4, 24]")
        observed = torch.cat([target, conditions, year], dim=1)
        cond_mask = torch.ones_like(observed)
        cond_mask[:, :4] = 0
        timepoints = torch.arange(24, device=conditions.device, dtype=conditions.dtype)
        side_info = self.get_side_info(timepoints.expand(len(conditions), -1), cond_mask)
        return observed, cond_mask, side_info

    def forward(self, target, conditions, year):
        observed, cond_mask, side_info = self.pack_conditions(conditions, year, target)
        return self.calc_loss(observed, cond_mask, torch.ones_like(observed), side_info, is_train=1)

    @torch.no_grad()
    def sample(self, conditions, year, generator):
        """Original full-step DDPM update, with explicit isolated sampling RNG.

        Known channels are masked out of the noisy input as in upstream impute.
        They are never generated outputs and never enter the scoring target.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before sampling")
        observed, cond_mask, side_info = self.pack_conditions(conditions, year)
        current = torch.randn(observed.shape, device=observed.device,
                              dtype=observed.dtype, generator=generator)
        for t in range(self.num_steps - 1, -1, -1):
            total_input = self.set_input_to_diffmodel(current, observed, cond_mask)
            predicted = self.diffmodel(total_input, side_info, torch.tensor([t], device=self.device))
            coeff1 = 1 / self.alpha_hat[t] ** 0.5
            coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
            current = coeff1 * (current - coeff2 * predicted)
            if t > 0:
                noise = torch.randn(current.shape, device=current.device,
                                    dtype=current.dtype, generator=generator)
                sigma = ((1 - self.alpha[t - 1]) / (1 - self.alpha[t]) * self.beta[t]) ** 0.5
                current += sigma * noise
        return current[:, :4]


@torch.no_grad()
def generate_scenarios(model, conditions, year, scenarios, generator, chunk_size=32):
    if scenarios < 1 or chunk_size < 1:
        raise ValueError("scenarios and chunk_size must be positive")
    # Bound peak attention memory. Flat order is day-major, scenario-minor.
    batches = []
    for start in range(0, len(conditions) * scenarios, chunk_size):
        indices = torch.arange(start, min(start + chunk_size, len(conditions) * scenarios),
                               device=conditions.device) // scenarios
        batches.append(model.sample(conditions[indices], year[indices], generator))
    return torch.cat(batches).reshape(len(conditions), scenarios, 4, 24)
