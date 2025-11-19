import argparse
import random
from collections import deque

import gymnasium as gym
import numpy as np
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
import matplotlib.pyplot as plt


# ==========
# FrameStack + 生存報酬 Wrapper
# ==========
class FrameStackEnv(gym.Wrapper):
    def __init__(self, env, k: int, living_reward: float = 0.01):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        self.living_reward = living_reward

        assert isinstance(env.observation_space, spaces.Box)
        h, w = env.observation_space.shape

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(k, h, w),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 生存報酬
        reward = float(reward) + self.living_reward

        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self.frames, axis=0)


# ==========
# 深い Actor-Critic ネットワーク
# ==========
class DeepActorCriticCNN(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        c, h, w = obs_shape

        # ---- CNN 層数を増やした（全5層）----
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),   # Conv1
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),  # Conv2
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),  # Conv3
        )

        # Conv 出力サイズを自動検出
        with torch.no_grad():
            tmp = torch.zeros(1, c, h, w)
            n_flatten = self.conv(tmp).view(1, -1).shape[1]

        # ---- FC 層も増やす（1024→512 の2層）----
        self.fc1 = nn.Linear(n_flatten, 1024)
        self.fc2 = nn.Linear(1024, 512)

        # ---- 出力ヘッド（Actor / Critic）----
        self.policy_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

        self._init_weights()

    def _init_weights(self):
        # Orthogonal 初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                gain = np.sqrt(2)
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, x):
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    def act(self, x):
        logits, value = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(self, x, actions):
        logits, values = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(actions), dist.entropy(), values


# ==========
# A2C 学習ループ（前回と同じ）
# ==========
def train(
    env_id="ALE/Breakout-v5",
    total_timesteps=200_000,
    gamma=0.99,
    lr=1e-4,
    rollout_length=5,
    value_coef=0.5,
    entropy_coef=0.01,
    seed=0,
    living_reward=0.01,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    base_env = gym.make(
        env_id,
        obs_type="grayscale",
        frameskip=1,
        render_mode=None,
    )
    env = FrameStackEnv(base_env, k=4, living_reward=living_reward)

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    # ★ ここだけ変更：深いネットを使用
    model = DeepActorCriticCNN(obs_shape, n_actions).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    obs, info = env.reset(seed=seed)
    obs = torch.tensor(obs).unsqueeze(0).to(device)

    global_step = 0
    episode_return = 0
    episode_len = 0

    # ------------------
    #     ログ保存用
    # ------------------
    episode_list = []
    return_list = []

    while global_step < total_timesteps:

        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []

        for _ in range(rollout_length):

            with torch.no_grad():
                action, logp, value = model.act(obs)

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(torch.tensor([reward], dtype=torch.float32, device=device))
            done_buf.append(torch.tensor([done], dtype=torch.float32, device=device))
            val_buf.append(value.unsqueeze(0))

            episode_return += reward
            episode_len += 1
            global_step += 1

            if done:
                episode_list.append(len(episode_list) + 1)
                return_list.append(episode_return)
                print(f"Episode {len(episode_list)}, Return={episode_return:.2f}, Steps={episode_len}")

                episode_return = 0
                episode_len = 0
                next_obs, info = env.reset()

            obs = torch.tensor(next_obs).unsqueeze(0).to(device)

            if global_step >= total_timesteps:
                break

        with torch.no_grad():
            _, next_value = model.forward(obs)

        acts = torch.cat(act_buf)
        logps = torch.cat(logp_buf)
        rewards = torch.cat(rew_buf)
        dones = torch.cat(done_buf)
        values = torch.cat(val_buf).squeeze(-1)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        for t in reversed(range(len(rewards))):
            next_v = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * (1 - dones[t]) * next_v - values[t]
            gae = delta + gamma * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = values[t] + advantages[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_batch = torch.cat(obs_buf, dim=0)
        new_logps, entropy, values_pred = model.evaluate_actions(obs_batch, acts)

        policy_loss = -(advantages.detach() * new_logps).mean()
        value_loss = F.mse_loss(values_pred, returns)
        entropy_loss = entropy.mean()

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    env.close()

    # ------------------------------------
    # エピソード報酬のグラフを保存
    # ------------------------------------
    if len(episode_list) > 0:
        plt.figure()
        plt.plot(episode_list, return_list)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Episode Return (Deep Model)")
        plt.grid()
        plt.savefig("deep_model_returns.png")
        plt.close()

    print("Training complete. Graph saved: deep_model_returns.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--living-reward", type=float, default=0.01)
    args = parser.parse_args()

    train(
        env_id=args.env_id,
        total_timesteps=args.total_timesteps,
        living_reward=args.living_reward,
    )


if __name__ == "__main__":
    main()
