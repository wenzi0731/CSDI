import copy
import csv
from datetime import datetime
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from step_study.condition_data import (TARGETS, WEATHER_SETS, calendar_features,
    load_daily_csv, prepare_daily, synthetic_daily)
from step_study.models import ConditionalDLinear, ExogenousCSDI


class ConditionTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = yaml.safe_load(Path("config/step_study.yaml").read_text())
        cls.config["diffusion"].update(layers=1, channels=8, nheads=2, diffusion_embedding_dim=16, num_steps=10)
        cls.config["model"].update(timeemb=8, featureemb=4)

    def test_day_splits_and_no_energy_in_conditions(self):
        raw = synthetic_daily(days=20)
        splits, info = prepare_daily(raw)
        self.assertEqual(info["split_day_counts"], {"train": 16, "val": 2, "test": 2})
        self.assertEqual(info["condition_dim"], 19)
        conditions, targets = splits["val"][0]
        self.assertEqual(conditions.shape, (24, 19))
        self.assertEqual(targets.shape, (24, 4))
        changed_targets = raw[0].copy()
        changed_targets[16:] += 10000
        changed_splits, changed_info = prepare_daily((changed_targets, *raw[1:]))
        self.assertEqual(info["mean"], changed_info["mean"])
        self.assertEqual(info["scale"], changed_info["scale"])
        torch.testing.assert_close(conditions, changed_splits["val"][0][0], rtol=0, atol=0)
        self.assertFalse(torch.equal(targets, changed_splits["val"][0][1]))

    def test_weather_scaler_train_only_and_calendar_periodicity(self):
        raw = synthetic_daily(days=20)
        _, before = prepare_daily(raw)
        changed = raw[1].copy()
        changed[16:] += 123
        _, after = prepare_daily((raw[0], changed, *raw[2:]))
        self.assertEqual(before["weather_mean"], after["weather_mean"])
        self.assertEqual(before["weather_scale"], after["weather_scale"])
        features = calendar_features([datetime(2020, 2, 29, 0), datetime(2020, 3, 1, 0)])
        np.testing.assert_allclose(features[:, -2:], [[0, 1], [0, 1]], atol=1e-7)
        self.assertAlmostEqual(float(features[0, 2]), np.sin(2*np.pi*59/366), places=6)

    def test_timestamp_join_and_incomplete_day_reporting(self):
        with tempfile.TemporaryDirectory() as folder:
            energy_path, weather_path = Path(folder)/"energy.csv", Path(folder)/"weather.csv"
            prefix = ["Year", "Month", "Day", "Hour"]
            def write(path, columns, rows):
                with path.open("w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(prefix + columns)
                    writer.writerows(rows)
            energy_rows = [[2020, 1, 1, hour]+[float(hour)]*4 for hour in range(24)]
            weather_rows = [[2020, 1, 1, hour]+[float(hour)]*11 for hour in reversed(range(24))]
            energy_rows.append([2020, 1, 2, 0]+[0.]*4)
            weather_rows.append([2020, 1, 2, 0]+[0.]*11)
            write(energy_path, TARGETS, energy_rows)
            write(weather_path, WEATHER_SETS["pv11"], weather_rows)
            energy, meteo, _, dates, dropped = load_daily_csv(energy_path, weather_path)
            self.assertEqual(dates, ["2020-01-01"])
            self.assertEqual(dropped, 1)
            np.testing.assert_array_equal(energy[0, :, 0], meteo[0, :, 0])
            write(weather_path, WEATHER_SETS["pv11"], weather_rows[:-1])
            with self.assertRaisesRegex(ValueError, "timestamps differ"):
                load_daily_csv(energy_path, weather_path)
            write(weather_path, WEATHER_SETS["pv11"], weather_rows + weather_rows[:1])
            with self.assertRaisesRegex(ValueError, "duplicate timestamp"):
                load_daily_csv(energy_path, weather_path)

    def test_conditional_dlinear_maps_19_conditions_to_four_targets(self):
        model = ConditionalDLinear(24, 19, 4)
        conditions = torch.randn(2, 24, 19)
        result = model(conditions)
        self.assertEqual(result.shape, (2, 24, 4))
        result.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_external_conditions_and_skeleton_reach_denoiser(self):
        model = ExogenousCSDI(4, self.config, "cpu", 19).eval()
        conditions = torch.randn(2, 24, 19)
        skeleton = torch.randn(2, 24, 4)
        direct = model.pack_conditions(conditions)
        residual = model.pack_conditions(conditions, skeleton)
        torch.testing.assert_close(direct[..., :19], residual[..., :19])
        self.assertEqual(direct[..., 19:].count_nonzero(), 0)
        observed, mask, side = model.context(residual, 24)
        torch.testing.assert_close(observed, skeleton.transpose(1, 2))
        self.assertEqual(mask.count_nonzero(), 0)  # no observed energy anywhere
        torch.testing.assert_close(side[:, -19:, 0], conditions.transpose(1, 2))
        noisy = torch.randn(2, 24, 4)
        t = torch.tensor([3, 4])
        # Upstream output weights initialize to zero; activate them to test dependence.
        torch.nn.init.normal_(model.diffmodel.output_projection2.weight, std=.1)
        a = model.predict_noise(noisy, direct, model.context(direct, 24), t)
        changed = model.pack_conditions(conditions + 1)
        b = model.predict_noise(noisy, changed, model.context(changed, 24), t)
        c = model.predict_noise(noisy, residual, model.context(residual, 24), t)
        self.assertGreater((a-b).abs().max().item(), 1e-6)
        self.assertGreater((a-c).abs().max().item(), 1e-6)

    def test_joint_sampling_and_skeleton_ablation(self):
        config = copy.deepcopy(self.config)
        config["model"]["condition_on_skeleton"] = False
        model = ExogenousCSDI(4, config, "cpu", 19).eval()
        conditions = torch.randn(2, 24, 19)
        torch.testing.assert_close(model.pack_conditions(conditions),
                                   model.pack_conditions(conditions, torch.randn(2, 24, 4)))
        packed = model.pack_conditions(conditions)
        samples = model.sample_ddpm(packed, torch.randn(2, 3, 24, 4), torch.Generator().manual_seed(1))
        self.assertEqual(samples.shape, (2, 3, 24, 4))
        self.assertTrue(torch.isfinite(samples).all())
        model.train()
        model.loss(packed, torch.randn(2, 24, 4)).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
