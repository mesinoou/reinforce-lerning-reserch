import unittest
import time

import numpy as np
import torch
import torch.optim as optim
from PIL import Image

from ocatari_source_ppo import (
    ObservationStatistics,
    PPOConfig,
    RewardConfig,
    RolloutBuffer,
    count_ale_frames_advanced,
    format_duration,
    make_progress_row,
    ppo_update,
    reward_components,
    transform_reward,
)
from benchmark_source_training import projection_rows
from render_trained_agent import (
    CapturedFrame,
    RepresentativeFrameReservoir,
    make_contact_sheet,
)
from ocatari_transfer_full_experiment import (
    ActorCriticMLP,
    CommonObjectEncoder,
    EncoderConfig,
)


class FakeObject:
    def __init__(self, category, xywh):
        self.category = category
        self.xywh = xywh

    def __bool__(self):
        return True


class EncoderTests(unittest.TestCase):
    def test_default_dimension_is_364(self):
        encoder = CommonObjectEncoder(EncoderConfig())
        observation = encoder.reset(
            [
                FakeObject("Player", (75, 185, 10, 10)),
                FakeObject("Alien", (70, 100, 8, 8)),
                FakeObject("Bullet", (80, 150, 2, 4)),
            ]
        )
        self.assertEqual(observation.shape, (364,))
        self.assertTrue(np.isfinite(observation).all())

    def test_temporal_slots_do_not_reorder_when_player_moves(self):
        temporal = CommonObjectEncoder(
            EncoderConfig(
                max_enemies=2,
                max_projectiles=1,
                stack_size=1,
                slot_strategy="temporal",
            )
        )
        distance = CommonObjectEncoder(
            EncoderConfig(
                max_enemies=2,
                max_projectiles=1,
                stack_size=1,
                slot_strategy="distance",
            )
        )
        first = [
            FakeObject("Player", (70, 180, 10, 10)),
            FakeObject("Alien", (65, 100, 8, 8)),
            FakeObject("Alien", (105, 100, 8, 8)),
        ]
        second = [
            FakeObject("Player", (110, 180, 10, 10)),
            FakeObject("Alien", (65, 100, 8, 8)),
            FakeObject("Alien", (105, 100, 8, 8)),
        ]
        temporal.reset(first)
        distance.reset(first)
        temporal_frame = temporal.step(second)
        distance_frame = distance.step(second)
        # Enemy slot 0 relative-x is frame index 6.
        self.assertLess(temporal_frame[6], 0.0)
        self.assertGreater(distance_frame[6], -0.2)

    def test_game_specific_categories_enter_common_groups(self):
        encoder = CommonObjectEncoder(
            EncoderConfig(
                max_enemies=2,
                max_projectiles=2,
                stack_size=1,
            )
        )
        observation = encoder.reset(
            [
                FakeObject("Player", (75, 185, 10, 10)),
                FakeObject("Satellite", (20, 40, 8, 8)),
                FakeObject("Bullet", (80, 150, 2, 4)),
            ]
        )
        self.assertEqual(observation[5], 1.0)
        projectile_start = 5 + 2 * 5
        self.assertEqual(observation[projectile_start], 1.0)


class RewardAndBufferTests(unittest.TestCase):
    def test_reward_modes(self):
        self.assertEqual(
            transform_reward(5.0, False, RewardConfig(mode="raw")),
            5.0,
        )
        self.assertEqual(
            transform_reward(5.0, False, RewardConfig(mode="clipped")),
            1.0,
        )
        self.assertAlmostEqual(
            transform_reward(
                5.0,
                False,
                RewardConfig(mode="scaled_raw", reward_scale=0.1),
            ),
            0.5,
        )
        self.assertAlmostEqual(
            transform_reward(
                5.0,
                True,
                RewardConfig(
                    mode="shaped",
                    reward_scale=0.1,
                    survival_bonus=0.5,
                    death_penalty=-5.0,
                ),
            ),
            0.05,
        )

    def test_scaled_survival_uses_actual_ale_frames(self):
        components = reward_components(
            5.0,
            True,
            RewardConfig(
                mode="scaled_survival",
                reward_scale=0.1,
                survival_reward_per_frame=0.001,
                life_loss_penalty=-1.0,
            ),
            ale_frames_advanced=4,
        )
        self.assertAlmostEqual(components.score, 0.5)
        self.assertAlmostEqual(components.survival, 0.004)
        self.assertAlmostEqual(components.life_loss, -1.0)
        self.assertAlmostEqual(components.total, -0.496)

    def test_ale_frame_counter_prefers_measured_difference(self):
        self.assertEqual(
            count_ale_frames_advanced(
                {"frame_number": 100},
                {"frame_number": 104},
                fallback_frameskip=8,
            ),
            4,
        )
        self.assertEqual(
            count_ale_frames_advanced({}, {}, fallback_frameskip=4),
            4,
        )

    def test_progress_row_contains_eta_and_remaining_steps(self):
        now = time.time()
        row = make_progress_row(
            status="running",
            env_steps=500,
            total_env_steps=1_000,
            update_index=5,
            episode_index=2,
            starting_env_steps=0,
            start_time=now - 10.0,
            previous_progress_step=400,
            previous_progress_time=now - 2.0,
            episode_rows=[{"raw_return": 100.0}, {"raw_return": 200.0}],
        )
        self.assertAlmostEqual(row["progress_percent"], 50.0)
        self.assertEqual(row["remaining_env_steps"], 500)
        self.assertGreater(row["average_steps_per_second"], 0.0)
        self.assertIsNotNone(row["eta_seconds"])
        self.assertEqual(row["recent_raw_return_mean"], 150.0)

    def test_duration_format_and_benchmark_projection(self):
        self.assertEqual(format_duration(3_661), "01:01:01")
        by_time, by_steps = projection_rows(
            steps_per_second=100.0,
            reserve_fraction=0.10,
            project_hours=[1.0],
            target_steps=[360_000],
        )
        self.assertEqual(
            by_time[0]["measured_projection_env_steps"],
            360_000,
        )
        self.assertEqual(
            by_time[0]["conservative_projection_env_steps"],
            324_000,
        )
        self.assertEqual(
            by_steps[0]["measured_estimated_time"],
            "01:00:00",
        )

    def test_terminal_breaks_gae(self):
        buffer = RolloutBuffer(2, 1)
        buffer.add(np.array([0.0], dtype=np.float32), 0, 1.0, True, 0.0, 0.0)
        buffer.add(np.array([0.0], dtype=np.float32), 0, 2.0, False, 0.0, 0.0)
        advantages, returns = buffer.compute_gae(3.0, gamma=1.0, gae_lambda=1.0)
        np.testing.assert_allclose(advantages, [1.0, 5.0])
        np.testing.assert_allclose(returns, advantages)

    def test_observation_statistics(self):
        statistics = ObservationStatistics(2)
        statistics.update(np.array([0.0, 1.0]), stack_size=1)
        statistics.update(np.array([2.0, 3.0]), stack_size=1)
        rows = statistics.rows(["a", "b"])
        self.assertEqual(rows[0]["mean"], 1.0)
        self.assertEqual(rows[0]["nonzero_rate"], 0.5)
        self.assertEqual(rows[1]["min"], 1.0)
        self.assertEqual(rows[1]["max"], 3.0)


class PPOTests(unittest.TestCase):
    def test_update_is_finite_and_changes_parameters(self):
        torch.manual_seed(0)
        model = ActorCriticMLP(obs_dim=4, n_actions=2, hidden_size=16)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        observations = torch.randn(32, 4)
        with torch.no_grad():
            logits, old_values = model(observations)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = distribution.sample()
            old_log_probs = distribution.log_prob(actions)
        advantages = torch.randn(32)
        returns = old_values + advantages
        before = [parameter.detach().clone() for parameter in model.parameters()]
        metrics = ppo_update(
            model,
            optimizer,
            observations,
            actions,
            old_log_probs,
            old_values,
            advantages,
            returns,
            PPOConfig(
                ppo_epochs=2,
                minibatch_size=8,
                hidden_size=16,
            ),
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters())
            )
        )


class PlaybackImageTests(unittest.TestCase):
    def test_representative_frames_keep_start_and_end(self):
        reservoir = RepresentativeFrameReservoir(capacity=5, seed=0)
        for step in range(10):
            reservoir.add(
                CapturedFrame(
                    step=step,
                    cumulative_raw_return=float(step),
                    action_meaning="TEST",
                    image=Image.new("RGB", (16, 16), color=(step, 0, 0)),
                )
            )
        selected = reservoir.selected()
        selected_steps = [capture.step for capture in selected]
        self.assertEqual(selected_steps[0], 0)
        self.assertEqual(selected_steps[-1], 9)
        self.assertLessEqual(len(selected), 5)
        self.assertEqual(selected_steps, sorted(selected_steps))

    def test_contact_sheet_has_expected_grid_size(self):
        captures = [
            CapturedFrame(
                step=step,
                cumulative_raw_return=0.0,
                action_meaning="NOOP",
                image=Image.new("RGB", (20, 30)),
            )
            for step in range(5)
        ]
        sheet = make_contact_sheet(captures, columns=2, title="test")
        self.assertEqual(sheet.size, (40, 56 + 3 * (30 + 28)))


if __name__ == "__main__":
    unittest.main()
