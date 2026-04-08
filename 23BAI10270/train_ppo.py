"""
train_ppo.py
============
Trains a PPO agent on the LunarLander-v3 Gymnasium environment using
Stable-Baselines3. Saves:
  - ppo_lunarlander.zip        : trained model checkpoint
  - training_rewards.npy       : per-episode rewards recorded during training

Requirements
------------
    pip install gymnasium gymnasium[box2d] stable-baselines3 numpy matplotlib
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

os.makedirs("plots", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Custom Callback – logs episode rewards during training
# ─────────────────────────────────────────────────────────────────────────────

class RewardLoggerCallback(BaseCallback):
    """
    Records every episode's total reward while training.
    Rewards are stored in self.episode_rewards (list of floats).
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self._current_ep_reward: float = 0.0

    def _on_step(self) -> bool:
        # infos from done environments contain "episode" key
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  LunarLander-v3 — PPO Training")
print("=" * 60)

env = gym.make("LunarLander-v3")

# ─────────────────────────────────────────────────────────────────────────────
# PPO Hyper-parameters  (as specified in the report)
# ─────────────────────────────────────────────────────────────────────────────

TOTAL_TIMESTEPS = 200_000

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    clip_range=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,          # small entropy bonus encourages exploration
    tensorboard_log=None,
)

callback = RewardLoggerCallback()

print(f"\nTraining for {TOTAL_TIMESTEPS:,} timesteps …\n")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=False)

# ─────────────────────────────────────────────────────────────────────────────
# Save model & reward trace
# ─────────────────────────────────────────────────────────────────────────────

model.save("ppo_lunarlander")
print("\nModel saved → ppo_lunarlander.zip")

rewards_arr = np.array(callback.episode_rewards)
np.save("training_rewards.npy", rewards_arr)
print(f"Training rewards saved → training_rewards.npy  ({len(rewards_arr)} episodes)")

# ─────────────────────────────────────────────────────────────────────────────
# Quick post-training evaluation
# ─────────────────────────────────────────────────────────────────────────────

mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=100, deterministic=True
)
print(f"\n{'─'*45}")
print(f"  Trained PPO Agent — Evaluation (100 eps)")
print(f"  Mean Reward : {mean_reward:>8.2f}")
print(f"  Std  Dev    : {std_reward:>8.2f}")
print(f"{'─'*45}")

env.close()
print("\nDone! Run  python evaluate_baseline.py  to generate comparison plots.")
