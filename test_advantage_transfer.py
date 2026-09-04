import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from advantage_transfer import (
    advantage_metrics, file_sha256, mix_advantages, model_sha256,
    protect_output_directory, read_source_checkpoint, source_argument_defaults,
    teacher_td_advantage, validate_source_settings,
)
from compare_transfer_runs import curve_auc, validate_pair
from ocatari_source_ppo import (
    EncoderConfig, PixelConfig, PPOConfig, RewardConfig, make_actor_critic,
    parse_args, ppo_update, train,
)
from ocatari_target_transfer import parse_args as parse_target_args
from ocatari_transfer_full_experiment import source_td_advantage


class ScalarCritic(torch.nn.Module):
    def forward(self, obs):
        return torch.zeros(len(obs), 3), obs[:, 0]


class AdvantageTests(unittest.TestCase):
    def test_td_matches_previous_rule_and_masks_resets(self):
        observations = torch.tensor([[2.], [4.], [7.], [9.]])
        final = torch.tensor([11.])
        rewards = torch.ones(4)
        dones = torch.tensor([0., 1., 0., 1.])
        source, values = teacher_td_advantage(
            ScalarCritic(), observations, final, rewards, dones, 0.9, 2,
        )
        expected = source_td_advantage(
            ScalarCritic(), observations,
            torch.cat((observations[1:], final.unsqueeze(0))),
            rewards, dones, 0.9,
        )
        torch.testing.assert_close(source, expected)
        torch.testing.assert_close(source, torch.tensor([2.6, -3., 2.1, -8.]))
        torch.testing.assert_close(values, observations[:, 0])
        self.assertFalse(source.requires_grad)

    def test_single_step_nonterminal_bootstrap(self):
        source, _ = teacher_td_advantage(
            ScalarCritic(), torch.tensor([[2.]]), torch.tensor([5.]),
            torch.tensor([1.]), torch.tensor([0.]), 0.9,
        )
        torch.testing.assert_close(source, torch.tensor([3.5]))

    def test_raw_mix_without_individual_normalization(self):
        target = torch.tensor([2., 4.])
        source = torch.tensor([100., 200.], requires_grad=True)
        torch.testing.assert_close(mix_advantages(target, source, 0.5), torch.tensor([51., 102.]))
        self.assertFalse(mix_advantages(target, source, 0.5).requires_grad)
        self.assertIs(mix_advantages(target, None, 0.0), target)
        torch.testing.assert_close(mix_advantages(target, source, 1.0), source.detach())

    def test_invalid_mix_and_finite_checks(self):
        for alpha in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                mix_advantages(torch.ones(2), torch.ones(2), alpha)
        with self.assertRaises(ValueError):
            mix_advantages(torch.ones(2), torch.ones(3), 0.5)
        with self.assertRaises(FloatingPointError):
            mix_advantages(torch.ones(2), torch.tensor([float("nan"), 0.]), 0.5)

    def test_metrics_constant_and_opposite_signals(self):
        target = torch.tensor([1., -1.])
        metrics = advantage_metrics(target, -target, torch.zeros(2), 0.5)
        self.assertEqual(metrics["source_target_sign_agreement"], 0.0)
        self.assertAlmostEqual(metrics["source_target_correlation"], -1.0)
        self.assertIsNone(advantage_metrics(target, torch.ones(2), target, 0.5)["source_target_correlation"])

    def test_teacher_frozen_while_target_updates(self):
        torch.manual_seed(4)
        kwargs = dict(input_mode="objects", observation_shape=(4,), n_actions=3, hidden_size=16)
        source = make_actor_critic(**kwargs).eval().requires_grad_(False)
        target = make_actor_critic(**kwargs)
        before_source, before_target = model_sha256(source), model_sha256(target)
        obs = torch.rand(16, 4)
        with torch.no_grad():
            logits, old_values = target(obs)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = distribution.sample()
            old_log_probs = distribution.log_prob(actions)
        source_adv, _ = teacher_td_advantage(source, obs, torch.rand(4), torch.ones(16), torch.zeros(16), 0.99)
        target_adv = torch.linspace(-2, 2, 16)
        target_returns = old_values + target_adv
        before_returns = target_returns.clone()
        ppo_update(target, torch.optim.Adam(target.parameters()), obs, actions,
                   old_log_probs, old_values, mix_advantages(target_adv, source_adv, 0.5),
                   target_returns, PPOConfig(ppo_epochs=1, minibatch_size=8))
        self.assertEqual(before_source, model_sha256(source))
        self.assertNotEqual(before_target, model_sha256(target))
        self.assertTrue(all(parameter.grad is None for parameter in source.parameters()))
        torch.testing.assert_close(target_returns, before_returns)


class SourceLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_dir = self.root / "source"
        self.source_dir.mkdir()
        self.path = self.source_dir / "model.pt"
        self.args = parse_args(["--hidden-size", "16"])
        self.encoder = asdict(EncoderConfig())
        self.pixels = asdict(PixelConfig())
        self.reward = asdict(RewardConfig())
        model = make_actor_critic(input_mode="objects", observation_shape=(364,), n_actions=6, hidden_size=16)
        self.payload = {
            "model_state_dict": model.state_dict(), "input_mode": "objects",
            "env_id": "ALE/SpaceInvaders-v5", "n_actions": 6,
            "encoder_config": self.encoder, "pixel_config": self.pixels,
            "reward_config": self.reward, "ppo_config": asdict(PPOConfig(hidden_size=16)),
            "steps": 14000000,
        }
        torch.save(self.payload, self.path)
        (self.source_dir / "config.json").write_text(json.dumps({"arguments": vars(self.args)}), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_final_model_with_sidecar_and_defaults(self):
        source = read_source_checkpoint(self.path)
        self.assertEqual(source["sha256"], file_sha256(self.path))
        defaults = source_argument_defaults(source)
        self.assertEqual(defaults["hidden_size"], 16)
        self.assertEqual(defaults["reward_mode"], "scaled_raw")
        self.assertNotIn("output_dir", defaults)
        self.assertNotIn("seed", defaults)
        validate_source_settings(source, self.args, self.encoder, self.pixels, self.reward)

    def test_standalone_checkpoint(self):
        self.payload["configuration"] = vars(self.args)
        torch.save(self.payload, self.path)
        (self.source_dir / "config.json").unlink()
        self.assertEqual(read_source_checkpoint(self.path)["input_mode"], "objects")

    def test_missing_runtime_rejected(self):
        (self.source_dir / "config.json").unlink()
        with self.assertRaisesRegex(ValueError, "runtime settings missing"):
            read_source_checkpoint(self.path)

    def test_same_shape_different_slot_strategy_rejected(self):
        encoder = dict(self.encoder, slot_strategy="distance")
        with self.assertRaisesRegex(ValueError, "encoder_config"):
            validate_source_settings(read_source_checkpoint(self.path), self.args, encoder, self.pixels, self.reward)

    def test_reward_scale_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "reward_config"):
            validate_source_settings(read_source_checkpoint(self.path), self.args, self.encoder, self.pixels, dict(self.reward, reward_scale=1.0))

    def test_output_protection(self):
        for output in (self.source_dir, self.root):
            with self.assertRaises(ValueError):
                protect_output_directory(output, self.path, "")
        target = self.root / "target"
        protect_output_directory(target, self.path, "")
        target.mkdir()
        (target / "model.pt").touch()
        with self.assertRaisesRegex(ValueError, "already contains"):
            protect_output_directory(target, self.path, "")
        protect_output_directory(target, self.path, str(target / "checkpoint_latest.pt"))

    def test_target_entrypoint_preserves_defaults_and_overrides(self):
        args = parse_target_args(["--source-checkpoint", str(self.path), "--seed", "2", "--total-steps", "400"])
        self.assertEqual(args.env, "ALE/Galaxian-v5")
        self.assertEqual(args.hidden_size, 16)
        self.assertEqual(args.transfer_alpha, 0.5)
        self.assertEqual(args.total_steps, 400)
        self.assertEqual(args.seed, 2)
        self.assertEqual(args.reward_scale, 0.1)

    def test_alpha_zero_keeps_teacher_configuration(self):
        args = parse_target_args(["--source-checkpoint", str(self.path), "--transfer-alpha", "0"])
        self.assertEqual(args.transfer_alpha, 0.0)
        self.assertEqual(args.hidden_size, 16)

    def test_resume_teacher_change_rejected_before_environment_creation(self):
        target_dir = self.root / "target"
        target_dir.mkdir()
        checkpoint = target_dir / "checkpoint_latest.pt"
        configuration = vars(self.args).copy()
        configuration.update({
            "env": "ALE/Galaxian-v5", "source_checkpoint": str(self.path),
            "source_checkpoint_sha256": "different-teacher", "transfer_alpha": 0.5,
        })
        torch.save({"configuration": configuration}, checkpoint)
        args = parse_target_args(["--resume", str(checkpoint), "--device", "cpu"])
        with patch("ocatari_source_ppo.make_observation_env") as create_env:
            with self.assertRaisesRegex(ValueError, "hash changed"):
                train(args)
            create_env.assert_not_called()


class ComparisonTests(unittest.TestCase):
    def test_paired_configuration_and_initialization(self):
        config = {"arguments": {"seed": 0, "source_checkpoint_sha256": "teacher"}, "initial_target_sha256": "target"}
        validate_pair(config, config)
        with self.assertRaisesRegex(ValueError, "initial parameters"):
            validate_pair(config, dict(config, initial_target_sha256="other"))

    def test_auc_normalizes_step_span(self):
        self.assertEqual(curve_auc({0: 10., 20: 30., 40: 50.}), 30.)


if __name__ == "__main__":
    unittest.main()
