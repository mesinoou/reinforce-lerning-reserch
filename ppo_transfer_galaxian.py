# ppo_transfer_galaxian.py
# Transfer PPO for Gymnasium Atari (ALE) Galaxian using Adaptive Policy Gradient Transfer
#
# Based on:
#   "Reinforcement Learning With Adaptive Policy Gradient Transfer Across Heterogeneous Problems"
#   - Transfer knowledge via advantage value (Eq. 12, 13, 14)
#   - Optional online source selection using local V-value sequence distance (Eq. 10, Alg. 2)
#
# What this script does:
# - Train a TARGET PPO agent on ALE/Galaxian-v5
# - Load one or more SOURCE pretrained checkpoints (ActorCritic same arch)
# - Each rollout:
#     * Compute target self-learning advantage A_T (GAE or TD-like)
#     * Compute source transferred advantage A_S from selected source critic (Eq.12)
#     * Mix them: A_total = (1-alpha)*A_T + alpha*A_S (Eq.13)
#     * Update target policy with PPO using A_total (Eq.14 flavor)
# - If multiple sources are provided:
#     * Select source online each rollout using distance between normalized V sequences
#       D = || Norm(seq_V^S) - Norm(seq_V^T) || (Eq.10)
#
# Requirements:
#   pip install gymnasium ale-py torch numpy matplotlib
#
# Example:
#   # single source
#   python ppo_transfer_galaxian.py --episodes 500 --source_ckpts ppo_galaxian.pt --alpha 0.5
#
#   # multiple sources (online select each rollout)
#   python ppo_transfer_galaxian.py --episodes 500 --source_ckpts src1.pt src2.pt src3.pt --alpha 0.5 --select_source 1
#
import argparse
import os
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
import ale_py  # noqa: F401
from gymnasium.wrappers import FrameStackObservation


# -----------------------------
# Obs preprocessing
# -----------------------------
def resize_gray_to_84x84(gray210x160: np.ndarray) -> np.ndarray:
    x = gray210x160.astype(np.float32)
    x = x[18:210, :]  # crop HUD a bit -> (192,160)
    h_in, w_in = x.shape
    h_out, w_out = 84, 84
    h_idx = (np.linspace(0, h_in - 1, h_out)).astype(np.int32)
    w_idx = (np.linspace(0, w_in - 1, w_out)).astype(np.int32)
    x_resized = x[h_idx][:, w_idx]
    x_resized /= 255.0
    return x_resized.astype(np.float32)


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    if obs.ndim == 2:
        return resize_gray_to_84x84(obs)[None, ...]
    if obs.ndim == 3:
        k = obs.shape[0]
        out = np.empty((k, 84, 84), dtype=np.float32)
        for i in range(k):
            out[i] = resize_gray_to_84x84(obs[i])
        return out
    raise ValueError(f"Unexpected obs shape: {obs.shape}")


# -----------------------------
# Model
# -----------------------------
class ActorCritic(nn.Module):
    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            n_flat = self.features(dummy).view(1, -1).shape[1]

        self.policy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )
        self.value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.features(x)
        logits = self.policy(z)
        v = self.value(z).squeeze(-1)
        return logits, v


# -----------------------------
# Rollout buffer
# -----------------------------
@dataclass
class RolloutBatch:
    obs: torch.Tensor      # (T,C,H,W)
    next_obs: torch.Tensor # (T,C,H,W)
    act: torch.Tensor      # (T,)
    logp: torch.Tensor     # (T,)
    ret: torch.Tensor      # (T,)
    adv: torch.Tensor      # (T,)
    val: torch.Tensor      # (T,)
    rew: torch.Tensor      # (T,)
    done: torch.Tensor     # (T,)


class RolloutBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int], device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.act = np.zeros((capacity,), dtype=np.int64)
        self.rew = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.logp = np.zeros((capacity,), dtype=np.float32)
        self.val = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def add(self, obs: np.ndarray, next_obs: np.ndarray, act: int, rew: float, done: bool, logp: float, val: float):
        if self.ptr >= self.capacity:
            raise RuntimeError("RolloutBuffer overflow")
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.act[i] = act
        self.rew[i] = rew
        self.done[i] = 1.0 if done else 0.0
        self.logp[i] = logp
        self.val[i] = val
        self.ptr += 1

    def compute_gae(self, last_val: float, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
        T = self.ptr
        adv = np.zeros((T,), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - self.done[t]
            next_val = last_val if t == T - 1 else self.val[t + 1]
            delta = self.rew[t] + gamma * next_val * nonterminal - self.val[t]
            last_gae = delta + gamma * lam * nonterminal * last_gae
            adv[t] = last_gae
        ret = adv + self.val[:T]
        return adv, ret

    def to_batch(self, adv: np.ndarray, ret: np.ndarray) -> RolloutBatch:
        T = self.ptr
        return RolloutBatch(
            obs=torch.from_numpy(self.obs[:T]).to(self.device),
            next_obs=torch.from_numpy(self.next_obs[:T]).to(self.device),
            act=torch.from_numpy(self.act[:T]).to(self.device),
            logp=torch.from_numpy(self.logp[:T]).to(self.device),
            val=torch.from_numpy(self.val[:T]).to(self.device),
            adv=torch.from_numpy(adv).to(self.device),
            ret=torch.from_numpy(ret).to(self.device),
            rew=torch.from_numpy(self.rew[:T]).to(self.device),
            done=torch.from_numpy(self.done[:T]).to(self.device),
        )


# -----------------------------
# PPO update (uses provided advantages)
# -----------------------------
def ppo_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    obs: torch.Tensor,
    act: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    n_epochs: int,
    minibatch_size: int,
):
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    T = obs.shape[0]
    idx = np.arange(T)

    for _ in range(n_epochs):
        np.random.shuffle(idx)
        for start in range(0, T, minibatch_size):
            mb = idx[start:start + minibatch_size]
            obs_mb = obs[mb]
            act_mb = act[mb]
            old_logp_mb = old_logp[mb]
            adv_mb = adv[mb]
            ret_mb = returns[mb]

            logits, v = model(obs_mb)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act_mb)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - old_logp_mb)
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
            policy_loss = -(torch.min(surr1, surr2)).mean()

            value_loss = 0.5 * (ret_mb - v).pow(2).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()


# -----------------------------
# Env
# -----------------------------
def make_env(env_id: str, seed: int, render_mode: Optional[str]) -> gym.Env:
    env = gym.make(env_id, obs_type="grayscale", frameskip=4, render_mode=render_mode)
    env = FrameStackObservation(env, stack_size=4)
    env.reset(seed=seed)
    return env


# -----------------------------
# Transfer helpers (paper-inspired)
# -----------------------------
@torch.no_grad()
def compute_v_sequence(model: ActorCritic, obs_seq: torch.Tensor) -> np.ndarray:
    """
    obs_seq: (T,C,H,W) torch
    returns: (T,) numpy float32
    """
    _logits, v = model(obs_seq)
    return v.detach().float().cpu().numpy().astype(np.float32)


def minmax_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


@torch.no_grad()
def select_source_by_vdist(
    target_model: ActorCritic,
    source_models: List[ActorCritic],
    obs_archive: torch.Tensor,
) -> int:
    """
    Implements Eq.(10) style distance on normalized local V sequences.
    Returns argmin source index.
    """
    vt = compute_v_sequence(target_model, obs_archive)
    vt_n = minmax_norm(vt)

    best_i, best_d = 0, float("inf")
    for i, sm in enumerate(source_models):
        vs = compute_v_sequence(sm, obs_archive)
        vs_n = minmax_norm(vs)
        d = float(np.linalg.norm(vs_n - vt_n))
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


@torch.no_grad()
def compute_source_advantage_tdstyle(
    source_model: ActorCritic,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    rew: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Paper Eq.(12): A_S(s_t,a_t) = r_t + gamma*V_S(s_{t+1}) - V_S(s_t)
    (We mask terminal transitions with (1-done).)
    """
    _l1, v_s = source_model(obs)
    _l2, v_sn = source_model(next_obs)
    nonterminal = 1.0 - done
    a_s = rew + gamma * v_sn * nonterminal - v_s
    return a_s


def moving_average(x: List[float], window: int = 10) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    if x.size < window:
        return np.full_like(x, x.mean(), dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="valid")

def load_checkpoint_into(model: ActorCritic, ckpt_path: str, device: torch.device) -> None:
    """
    Loads a checkpoint saved as:
      {"model_state_dict": ..., "env_id": ..., "n_actions": ...}
    PyTorch 2.6+ compatibility:
      - If the checkpoint contains non-tensor objects (e.g., numpy scalars),
        torch.load(weights_only=True) may fail.
      - If you TRUST the checkpoint, load with weights_only=False.
    """
    # 1) First try safe path (weights_only=True)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        # 2) If it fails, fall back to full unpickling (ONLY if you trust this file)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        # assume state_dict directly
        sd = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")

    model.load_state_dict(sd, strict=True)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="ALE/Galaxian-v5")

    # Target training
    parser.add_argument("--episodes", type=int, default=8000)
    parser.add_argument("--steps_per_rollout", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--clip_eps", type=float, default=0.1)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=256)

    # Reward shaping (same as your updated PPO)
    parser.add_argument("--survival_bonus", type=float, default=0.5)
    parser.add_argument("--death_penalty", type=float, default=-5.0)

    # Transfer learning knobs
    parser.add_argument("--source_ckpts", type=str, nargs="+", required=True,
                        help="One or more pretrained source checkpoints (.pt).")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Transfer weight alpha in A_total = (1-alpha)A_T + alpha A_S (paper Eq.13).")
    parser.add_argument("--select_source", type=int, default=1,
                        help="1: online source selection via V-sequence distance (Eq.10). 0: always use first source.")
    parser.add_argument("--archive_size", type=int, default=512,
                        help="How many recent states to use for source selection (local V distribution).")

    parser.add_argument("--alpha_decay_coef", type=float, default=0.99,
                    help="After each episode ends: alpha <- alpha * alpha_decay_coef. "
                         "1.0 means no decay. Typical: 0.995, 0.99, 0.97, 0.95")
    parser.add_argument("--alpha_min", type=float, default=0.0,
                    help="Lower bound for alpha (clamp) after decay.")

    # I/O
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="ppo_galaxian_transfer.pt")
    parser.add_argument("--log_csv", type=str, default="")
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0,1]")

    device = torch.device(args.device)
    render_mode = "human" if args.render == 1 else None
    env = make_env(args.env_id, seed=args.seed, render_mode=render_mode)
    n_actions = env.action_space.n

    # Target model
    target = ActorCritic(n_actions=n_actions, in_channels=4).to(device)
    optimizer = optim.Adam(target.parameters(), lr=args.lr, eps=1e-5)

    # Source models (critics used; we keep full ActorCritic for simplicity)
    source_models: List[ActorCritic] = []
    for p in args.source_ckpts:
        sm = ActorCritic(n_actions=n_actions, in_channels=4).to(device)
        sm.eval()
        load_checkpoint_into(sm, p, device=device)
        for param in sm.parameters():
            param.requires_grad_(False)
        source_models.append(sm)

    # Logging
    ep_returns: List[float] = []
    ep_steps: List[int] = []
    if args.log_csv:
        os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
        with open(args.log_csv, "w", encoding="utf-8") as f:
            f.write("episode,return,steps,source_idx\n")

    obs, info = env.reset(seed=args.seed)
    obs_p = preprocess_obs(np.array(obs))
    prev_lives = info.get("lives", None)

    rollout = RolloutBuffer(args.steps_per_rollout, obs_shape=(4, 84, 84), device=device)

    # archive of recent states for source selection (stores obs tensors)
    obs_archive: List[np.ndarray] = []
    selected_source_idx = 0

    episode = 0
    ep_ret = 0.0
    ep_len = 0
    start_time = time.time()

    alpha_now = float(args.alpha)  # current alpha (decayed over episodes)

    while episode < args.episodes:
        rollout.reset()

        # Collect rollout
        for _t in range(args.steps_per_rollout):
            obs_t = torch.from_numpy(obs_p).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, v = target(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)

            action = int(act.item())
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # reward shaping
            reward = float(reward) + float(args.survival_bonus)
            lives = info.get("lives", None)
            if (prev_lives is not None) and (lives is not None) and (lives < prev_lives):
                reward += float(args.death_penalty)
            if lives is not None:
                prev_lives = lives

            reward*=0.1

            next_obs_p = preprocess_obs(np.array(next_obs))

            # save for source selection archive
            obs_archive.append(obs_p.copy())
            if len(obs_archive) > args.archive_size:
                obs_archive.pop(0)

            rollout.add(
                obs=obs_p,
                next_obs=next_obs_p,
                act=action,
                rew=reward,
                done=done,
                logp=float(logp.item()),
                val=float(v.item()),
            )

            ep_ret += reward
            ep_len += 1
            obs_p = next_obs_p

            if done:
                ep_returns.append(ep_ret)
                ep_steps.append(ep_len)
                episode += 1

                alpha_now *= float(args.alpha_decay_coef)
                if alpha_now < float(args.alpha_min):
                    alpha_now = float(args.alpha_min)
                if alpha_now > 1.0:
                    alpha_now = 1.0

                if len(ep_returns) >= 10:
                    avg10 = float(np.mean(ep_returns[-10:]))
                    avglen10 = float(np.mean(ep_steps[-10:]))
                    elapsed = time.time() - start_time
                    print(f"Ep {episode:4d}/{args.episodes} | "
                          f"R={ep_ret:8.2f} (avg10={avg10:8.2f}) | "
                          f"Len={ep_len:4d} (avg10={avglen10:7.1f}) | "
                          f"src={selected_source_idx} | elapsed={elapsed:7.1f}s")
                else:
                    print(f"Ep {episode:4d}/{args.episodes} | R={ep_ret:8.2f} | Len={ep_len:4d} | src={selected_source_idx}| alpha={alpha_now:.4f}")

                if args.log_csv:
                    with open(args.log_csv, "a", encoding="utf-8") as f:
                        f.write(f"{episode},{ep_ret},{ep_len},{selected_source_idx}\n")

                obs, info = env.reset()
                obs_p = preprocess_obs(np.array(obs))
                prev_lives = info.get("lives", None)
                ep_ret = 0.0
                ep_len = 0

                if episode >= args.episodes:
                    break

        # Bootstrap (target GAE)
        with torch.no_grad():
            obs_last = torch.from_numpy(obs_p).unsqueeze(0).to(device)
            _l, last_val = target(obs_last)
            last_val = float(last_val.item())

        adv_t, ret_t = rollout.compute_gae(last_val=last_val, gamma=args.gamma, lam=args.gae_lambda)
        batch = rollout.to_batch(adv=adv_t, ret=ret_t)

        # Online source selection (paper Eq.10 + Alg.2 spirit)
        if args.select_source == 1 and len(source_models) > 1 and len(obs_archive) >= 8:
            # build tensor of archive states
            arch_np = np.stack(obs_archive, axis=0)  # (K,C,H,W)
            arch_t = torch.from_numpy(arch_np).to(device)
            selected_source_idx = select_source_by_vdist(target, source_models, arch_t)
        else:
            selected_source_idx = 0

        source = source_models[selected_source_idx]

        # Transferred advantage (paper Eq.12)
        a_s = compute_source_advantage_tdstyle(
            source_model=source,
            obs=batch.obs,
            next_obs=batch.next_obs,
            rew=batch.rew,
            done=batch.done,
            gamma=args.gamma,
        )

        # Total advantage (paper Eq.13)
        a_total = (1.0 - alpha_now) * batch.adv + alpha_now * a_s

        # PPO update using A_total (paper Eq.14 flavor: grad weighted by A_total)
        ppo_update(
            model=target,
            optimizer=optimizer,
            obs=batch.obs,
            act=batch.act,
            old_logp=batch.logp,
            advantages=a_total,
            returns=batch.ret,  # critic still trained on target returns
            clip_eps=args.clip_eps,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
            n_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
        )

    env.close()

    # Save target model after transfer training
    torch.save(
        {
            "model_state_dict": target.state_dict(),
            "env_id": args.env_id,
            "n_actions": n_actions,
            "alpha": args.alpha,
            "source_ckpts": args.source_ckpts,
        },
        args.save_path,
    )
    print(f"Saved: {args.save_path}")

    # Plots: 10-episode moving average
    try:
        import matplotlib.pyplot as plt
        window = 100

        suffix = f"alpha{args.alpha}"

        avg_ret = moving_average(ep_returns, window=window)
        plt.figure()
        plt.plot(avg_ret)
        plt.title(f"Episode Return (Moving Avg {window}) - Galaxian PPO Transfer")
        plt.xlabel(f"Episode (starting at {window})")
        plt.ylabel("Return (avg)")
        plt.tight_layout()
        plt.savefig(f"learning_curve_return_transfer_{suffix}.png", dpi=150)
        plt.close()

        avg_steps = moving_average(ep_steps, window=window)
        plt.figure()
        plt.plot(avg_steps)
        plt.title(f"Episode Stepsreward*=0.1 (Moving Avg {window}) - Galaxian PPO Transfer")
        plt.xlabel(f"Episode (starting at {window})")
        plt.ylabel("Steps (avg)")
        plt.tight_layout()
        plt.savefig(f"learning_curve_steps_transfer_{suffix}.png", dpi=150)
        plt.close()

        print("Saved plots: episode_return_avg10_transfer.png, episode_steps_avg10_transfer.png")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
