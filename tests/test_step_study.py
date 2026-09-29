import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from step_study.data import prepare, synthetic_series
from step_study.metrics import crps, energy_score, step_summary
from step_study.models import DLinear, ForecastCSDI


class StepStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with open("config/step_study.yaml") as file:
            cls.config = yaml.safe_load(file)
        cls.config["diffusion"].update(layers=1, channels=8, nheads=2, diffusion_embedding_dim=16)
        cls.config["model"].update(timeemb=8, featureemb=4)

    def test_split_and_scaler_use_training_only(self):
        values = synthetic_series(200)
        splits, metadata = prepare(values, 12, 4, 4)
        changed = values.copy()
        changed[120:] += 1000
        _, other = prepare(changed, 12, 4, 4)
        self.assertEqual(metadata["mean"], other["mean"])
        self.assertEqual(metadata["scale"], other["scale"])
        for key, lower, upper in (("train", 12, 120), ("val", 120, 160), ("test", 160, 200)):
            self.assertTrue(all(lower <= s and s + 4 <= upper for s in splits[key].starts))

    def test_dlinear_shapes_constant_and_gradients(self):
        for individual in (True, False):
            model = DLinear(12, 4, 3, 3, individual)
            x = torch.ones(2, 12, 3)
            out = model(x)
            self.assertEqual(out.shape, (2, 4, 3))
            torch.testing.assert_close(out, torch.ones_like(out))
            out.square().mean().backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_empirical_crps_matches_pairwise_definition(self):
        samples, truth = torch.randn(2, 7, 4, 3), torch.randn(2, 4, 3)
        reference = (samples-truth[:, None]).abs().mean(1) - .5 * (
            samples[:, :, None]-samples[:, None, :]).abs().mean((1, 2))
        torch.testing.assert_close(crps(samples, truth), reference)
        torch.testing.assert_close(crps(samples[:, :1], truth), (samples[:, 0]-truth).abs())
        torch.testing.assert_close(energy_score(truth[:, None], truth), torch.zeros(2))

    def test_same_denoiser_initialization_and_joint_gradients(self):
        original = copy.deepcopy(self.config)
        torch.manual_seed(2)
        direct = ForecastCSDI(3, self.config, "cpu")
        torch.manual_seed(2)
        residual = ForecastCSDI(3, self.config, "cpu")
        self.assertEqual(original, self.config)
        for a, b in zip(direct.parameters(), residual.parameters()):
            torch.testing.assert_close(a, b)
        history, truth = torch.randn(2, 12, 3), torch.randn(2, 4, 3)
        loss = direct.loss(history, truth)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in direct.parameters()))

    def test_sampler_reproducibility_nfe_and_terminal_formula(self):
        model = ForecastCSDI(3, self.config, "cpu").eval()
        history, noise = torch.randn(2, 12, 3), torch.randn(2, 2, 4, 3)
        for steps in (1, 2, 7):
            with patch.object(model, "predict_noise", side_effect=lambda x, *args: torch.zeros_like(x)) as predict:
                result = model.sample(history, noise, steps)
                self.assertEqual(predict.call_count, steps)
                torch.testing.assert_close(result, noise/model.alpha_bar[-1].sqrt(), rtol=1e-4, atol=1e-4)
        a, b = model.sample(history, noise, 3), model.sample(history, noise, 3)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model.sample(history, noise, 0)
        model.train()
        with self.assertRaises(RuntimeError):
            model.sample(history, noise, 2)

    def test_forecast_mask_never_exposes_future(self):
        model = ForecastCSDI(3, self.config, "cpu")
        history = torch.randn(2, 12, 3)
        observed, mask, _ = model.context(history, 4)
        self.assertEqual(mask[:, :, 12:].sum().item(), 0)
        self.assertEqual(observed[:, :, 12:].sum().item(), 0)
        torch.testing.assert_close(observed[:, :, :12], history.transpose(1, 2))

    def test_validation_selects_steps_even_when_test_disagrees(self):
        rows = []
        for split, matrix in (("val", [[1., 2.], [2., 1.]]), ("test", [[9., 1.], [1., 9.]])):
            for k, scores in zip((5, 10), matrix):
                for variable, score in zip(("a", "b"), scores):
                    rows.append(dict(seed=42, method="direct", split=split, steps=k,
                                     variable=variable, crps_scaled=score))
        summary = step_summary(rows)[0]
        self.assertEqual(summary["validation_optimal_steps"], [5, 10])
        self.assertEqual(summary["validation_shared_steps"], 5)
        self.assertEqual(summary["validation_common_near_optimal_steps"], [])
        self.assertEqual(summary["test_shared_minus_variable_selected_scaled_crps"], -4.)


if __name__ == "__main__":
    unittest.main()
