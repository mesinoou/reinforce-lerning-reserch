import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import ocatari_source_ppo as ppo
import run_all_objects as runner
from run_input_hypotheses import save_json
from summarize_input_hypotheses import normalized_auc, summarize


def obj(category, xywh, **kwargs):
    return SimpleNamespace(category=category, xywh=xywh, **kwargs)


class ObjectChannelTests(unittest.TestCase):
    def test_binary_rasterization_clips_and_keeps_thin_objects(self):
        objects = [obj("Player", (80, 180, 4, 5)), obj("Shield", (-4, -5, 8, 10)),
                   obj("Bullet", (159, 209, 1, 1)), obj("Alien", (160, 0, 1, 2)),
                   obj("Satellite", (5, 5, 0, 2)), None,
                   obj("Bullet", (159, 209, 1, 1)), obj("Lives", (0, 0, 5, 5), hud=True)]
        maps = ppo.object_channel_frame(objects, ppo.PixelConfig())
        self.assertEqual(maps.shape, (5, 84, 84))
        self.assertEqual(maps.dtype, np.float32)
        expected = np.zeros_like(maps)
        expected[0, 72:74, 42:45] = 1
        expected[1, :2, :3] = 1
        expected[2, 83, 83] = 1
        np.testing.assert_array_equal(maps, expected)

    def test_stack_order_and_single_environment_step(self):
        base = SimpleNamespace(action_space=object())
        base.reset = lambda seed=None: (None, {})
        first = obj("Player", (10, 180, 4, 5))
        base.objects = [first]

        def step(action):
            base.objects = [obj("Alien", (100, 50, 4, 5))]
            return None, 7.0, False, {}

        with patch.object(ppo, "ObjectEnv", return_value=base):
            env = ppo.make_observation_env(input_mode="object_channels", env_id="ALE/SpaceInvaders-v5",
                                           encoder_config=ppo.EncoderConfig(), object_mode="ram",
                                           frameskip=4, repeat_action_probability=0.0,
                                           pixel_config=ppo.PixelConfig())
        initial, _ = env.reset(seed=0)
        self.assertEqual(initial.shape, (20, 84, 84))
        np.testing.assert_array_equal(initial[:5], initial[15:])
        with patch.object(base, "step", side_effect=step, create=True) as transition:
            after, reward, done, _ = env.step(1)
            transition.assert_called_once_with(1)
        self.assertEqual((reward, done), (7.0, False))
        np.testing.assert_array_equal(after[:15], initial[5:])
        self.assertGreater(after[-1].sum(), 0)
        base.objects = [first]
        np.testing.assert_array_equal(env.reset()[0], initial)
        names = ppo.observation_feature_names("object_channels", ppo.EncoderConfig(), ppo.PixelConfig())
        stats = ppo.ObservationStatistics(len(names))
        stats.update(after, 4, summarize_pixels=True, channels_per_frame=5)
        self.assertEqual(len(names), 100)
        self.assertGreater(stats.mean[names.index("frame[3].Alien.nonzero_rate")], 0)

    def test_cnn_forward_and_parameter_count(self):
        model = ppo.make_actor_critic(input_mode="object_channels", observation_shape=(20, 84, 84),
                                      n_actions=6, hidden_size=256)
        with torch.no_grad():
            logits, values = model(torch.zeros((2, 20, 84, 84)))
        self.assertEqual(tuple(logits.shape), (2, 6))
        self.assertEqual(tuple(values.shape), (2,))
        self.assertEqual(ppo.model_parameter_count(model), 915623)

    def test_resume_rejects_different_input_mode_before_creating_environment(self):
        args = ppo.parse_args(["--input-mode", "object_channels", "--resume", "fake.pt", "--device", "cpu"])
        with patch.object(ppo.torch, "load", return_value={"input_mode": "pixels"}), \
                patch.object(ppo, "make_observation_env") as factory:
            with self.assertRaisesRegex(ValueError, "input-mode mismatch"):
                ppo.train(args)
        factory.assert_not_called()

    def test_plan_preserves_default_and_hashes_the_new_schema(self):
        _, _, default = runner.build_plan([])
        self.assertEqual(list(default["conditions"]), runner.DEFAULT_CONDITIONS)
        argv = ["--conditions", "pixels_256", "object_channels_256", "--gpus", "0,1",
                "--seeds", "0", "--total-steps", "14000000"]
        args, _, plan = runner.build_plan(argv)
        self.assertEqual(args.gpus, ["0", "1"])
        self.assertEqual([j["arguments"]["total_steps"] for j in plan["jobs"]], [14000000] * 2)
        self.assertEqual(plan["object_channel_spec"]["channels"], list(ppo.OBJECT_CHANNELS))
        _, _, faster_monitor = runner.build_plan(argv + ["--status-interval", "5"])
        self.assertEqual(plan["plan_sha256"], faster_monitor["plan_sha256"])


class ComparisonTests(unittest.TestCase):
    def test_interval_integration(self):
        curve = {0: 0, 10: 10, 20: 20}
        self.assertEqual(normalized_auc(curve, (2, 15)), 8.5)
        self.assertIsNone(normalized_auc(curve, (5, 25)))

    def test_partial_intersection_and_paired_difference(self):
        _, _, plan = runner.build_plan(["--conditions", "pixels_256", "object_channels_256",
                                        "--seeds", "0", "--total-steps", "14000000"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_json(root/"experiment_plan.json", plan)
            for index, job in enumerate(plan["jobs"]):
                directory = root/job["id"]
                directory.mkdir(parents=True)
                save_json(directory/"job.json", {"plan_sha256": plan["plan_sha256"]})
                save_json(directory/"config.json", {"arguments": job["arguments"],
                          "model_parameter_count": 1, "observation_dimension": 1,
                          "object_channel_spec": plan["object_channel_spec"]})
                with (directory/"periodic_evaluation.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["stage", "policy", "env_steps", "raw_return_mean", "episodes"])
                    writer.writerow(["initial", "stochastic", 0, 10, 10])
                    writer.writerow(["periodic", "stochastic", 100000, 20 + 10 * index, 10])
                    writer.writerow(["periodic", "stochastic", 200000 + 100000 * index, 50, 10])
            result = summarize(root, plot=False, partial=True)
            self.assertEqual(result["auc_interval"], [0, 100000])
            pair = result["paired_comparisons"][0]
            self.assertEqual(pair["paired_auc_difference"]["mean"], 5)
            self.assertEqual(pair["paired_last_common_return_difference"]["mean"], 10)
            self.assertIsNone(pair["paired_early_auc_difference"])
            self.assertTrue((root/"comparison_partial"/"per_seed.csv").exists())
            self.assertFalse((root/"comparison").exists())


if __name__ == "__main__":
    unittest.main()
