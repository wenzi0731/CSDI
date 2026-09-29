import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from step_study.data import prepare, synthetic_series
from step_study.metrics import crps, energy_score, ncrps_from_sums, relative_loss_rows, step_summary
from step_study.models import DLinear, ForecastCSDI
from step_study.plotting import curve_values, select_plot_variables
from step_study.run import evaluate


class StepStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with open("config/step_study.yaml") as file:
            cls.config = yaml.safe_load(file)
        cls.config["diffusion"].update(layers=1, channels=8, nheads=2, diffusion_embedding_dim=16, num_steps=8)
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
                result = model.sample_ddim(history, noise, steps)
                self.assertEqual(predict.call_count, steps)
                torch.testing.assert_close(result, noise/model.alpha_bar[-1].sqrt(), rtol=1e-4, atol=1e-4)
        a, b = model.sample_ddim(history, noise, 3), model.sample_ddim(history, noise, 3)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model.sample_ddim(history, noise, 0)
        model.train()
        with self.assertRaises(RuntimeError):
            model.sample_ddim(history, noise, 2)

    def test_full_ddpm_uses_every_timestep_and_paired_randomness(self):
        model = ForecastCSDI(3, self.config, "cpu").eval()
        history, initial = torch.randn(2, 12, 3), torch.randn(2, 2, 4, 3)
        with patch.object(model, "predict_noise", side_effect=lambda x, *args: torch.zeros_like(x)) as predict:
            with patch("step_study.models.torch.randn", side_effect=lambda shape, **kwargs: torch.zeros(shape, dtype=kwargs.get("dtype"))) as draw:
                actual = model.sample_ddpm(history, initial, torch.Generator().manual_seed(123))
            self.assertEqual(predict.call_count, model.num_steps)
            self.assertEqual(draw.call_count, model.num_steps - 1)
            self.assertEqual([call.args[-1][0].item() for call in predict.call_args_list], list(range(7, -1, -1)))
            torch.testing.assert_close(actual, initial / model.alpha_bar[-1].sqrt(), rtol=1e-4, atol=1e-4)
        a = model.sample_ddpm(history, initial, torch.Generator().manual_seed(3))
        b = model.sample_ddpm(history, initial, torch.Generator().manual_seed(3))
        c = model.sample_ddpm(history, initial, torch.Generator().manual_seed(4))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertFalse(torch.equal(a, c))

    def test_noise_schedule_and_time_embedding_match_across_total_steps(self):
        models = []
        for steps in (10, 20, 100):
            config = copy.deepcopy(self.config)
            config["diffusion"]["num_steps"] = steps
            models.append(ForecastCSDI(3, config, "cpu"))
        expected_terminal = np.exp(-10.05)
        for model in models:
            self.assertAlmostEqual(model.alpha_bar[-1].item(), expected_terminal, places=8)
            torch.testing.assert_close((1-model.betas).cumprod(0), model.alpha_bar, rtol=2e-5, atol=1e-7)
        torch.testing.assert_close(models[0].alpha_bar[4], models[1].alpha_bar[9])
        torch.testing.assert_close(models[0].diffmodel.diffusion_embedding.embedding[4],
                                   models[1].diffmodel.diffusion_embedding.embedding[9])
        self.assertEqual(len({sum(p.numel() for p in model.parameters()) for model in models}), 1)

    def test_ncrps_normalization_original_units_and_zero_guard(self):
        score = torch.tensor([4., 9.])
        magnitude = torch.tensor([20., 30.])
        torch.testing.assert_close(ncrps_from_sums(score, magnitude), torch.tensor([.2, .3]))
        scale = torch.tensor([2., 7.])
        torch.testing.assert_close(ncrps_from_sums(score*scale, magnitude*scale), torch.tensor([.2, .3]))
        with self.assertRaises(ValueError):
            ncrps_from_sums(score, torch.tensor([0., 1.]))

    def test_relative_losses_use_each_method_own_minimum(self):
        rows = []
        for method, scores in (("direct", [1., 2., 3.]), ("residual", [.5, .75, .5])):
            for steps, score in zip((10, 20, 40), scores):
                rows.append(dict(seed=42, method=method, split="test", variable="a",
                                 diffusion_steps=steps, ncrps=score))
        relative = relative_loss_rows(rows)
        np.testing.assert_allclose([r["relative_loss"] for r in relative], [0, 1, 2, 0, .5, 0], atol=1e-7)
        self.assertTrue(all(r["reference"].endswith("descriptive") for r in relative))
        self.assertEqual(relative[-1]["optimal_diffusion_steps"], 10)

    def test_evaluation_restores_target_mean_before_normalization(self):
        class ZeroForecast:
            num_steps = 2

            def pack_conditions(self, conditions, skeleton=None):
                return conditions

            def sample_ddpm(self, history, initial_noise, generator):
                return torch.zeros_like(initial_noise)

        dataset = torch.utils.data.TensorDataset(torch.zeros(3, 12, 4), torch.full((3, 4, 4), 2.))
        models = {"direct": ZeroForecast(), "residual": ZeroForecast()}
        baseline = lambda history: history.new_zeros(len(history), 4, 4)
        with patch("builtins.print"):
            rows, _ = evaluate(models, baseline, {"val": dataset, "test": dataset},
                {"samples": 3, "batch_size": 2}, {"scale": [3.]*4, "mean": [10.]*4},
                ["a", "b", "c", "d"], "cpu", 42)
        self.assertEqual(len(rows), 16)
        for row in rows:
            self.assertEqual(row["crps_original"], 6.)
            self.assertEqual(row["ncrps_denominator_mean_abs_target"], 16.)
            self.assertEqual(row["ncrps"], .375)
            self.assertEqual(row["diffusion_steps"], row["sampling_steps"])

    def test_four_variable_selection_and_plot_grid_validation(self):
        self.assertEqual(select_plot_variables(["a", "b", "c", "d"]), ["a", "b", "c", "d"])
        self.assertEqual(select_plot_variables(["a", "b", "c", "d", "e"], "e,c,b,a"), ["e", "c", "b", "a"])
        with self.assertRaises(ValueError):
            select_plot_variables(["a", "b", "c"])
        rows = [dict(seed=42, method="direct", split="val", variable="a", diffusion_steps=10, ncrps=.3)]
        steps, values = curve_values(rows, "direct", "val", "a", "ncrps")
        self.assertEqual(steps, [10])
        self.assertEqual(values.shape, (1, 1))
        with self.assertRaises(ValueError):
            curve_values(rows + rows, "direct", "val", "a", "ncrps")

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
                    rows.append(dict(seed=42, method="direct", split=split, diffusion_steps=k,
                                     variable=variable, ncrps=score))
        summary = step_summary(rows)[0]
        self.assertEqual(summary["validation_optimal_steps"], [5, 10])
        self.assertEqual(summary["validation_shared_steps"], 5)
        self.assertEqual(summary["validation_common_near_optimal_steps"], [])
        self.assertEqual(summary["test_shared_minus_variable_selected_ncrps"], -4.)
        self.assertAlmostEqual(summary["validation_minimum_shared_relative_loss_G"], .5)


if __name__ == "__main__":
    unittest.main()
