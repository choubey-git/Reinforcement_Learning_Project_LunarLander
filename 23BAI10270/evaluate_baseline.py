"""
evaluate_baseline.py
====================
Loads the pre-trained PPO model and runs a full comparison against a
random-action baseline. Produces three publication-quality plots saved
inside the  plots/  directory.

Expected files before running
------------------------------
    ppo_lunarlander.zip     (created by train_ppo.py)
    training_rewards.npy    (created by train_ppo.py)

Output plots
------------
    plots/learning_curve.png      — smoothed reward vs training episode
    plots/comparison_bar.png      — mean reward ± std for PPO vs Random
    plots/reward_distribution.png — side-by-side violin / box plots
"""

import os
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")           # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

os.makedirs("plots", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Style / theme
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.facecolor": "#0f0f1a",
    "figure.facecolor": "#0f0f1a",
    "axes.edgecolor": "#3a3a5c",
    "axes.labelcolor": "#e0e0ff",
    "xtick.color": "#b0b0d0",
    "ytick.color": "#b0b0d0",
    "text.color": "#e0e0ff",
    "grid.color": "#2a2a4a",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "legend.framealpha": 0.25,
    "legend.edgecolor": "#5555aa",
})

PPO_COLOR    = "#7c6fff"   # violet
RANDOM_COLOR = "#ff6b6b"   # coral
SMOOTH_COLOR = "#00d4aa"   # teal accent

N_EVAL_EPISODES = 100

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load model
# ─────────────────────────────────────────────────────────────────────────────

print("Loading PPO model …")
model = PPO.load("ppo_lunarlander")

env = gym.make("LunarLander-v3")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Evaluate PPO – collect per-episode rewards
# ─────────────────────────────────────────────────────────────────────────────

print(f"Evaluating trained PPO agent ({N_EVAL_EPISODES} episodes) …")
ppo_rewards: list[float] = []

obs, _ = env.reset()
ep_r = 0.0
episodes_done = 0

while episodes_done < N_EVAL_EPISODES:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    ep_r += reward
    if done or truncated:
        ppo_rewards.append(ep_r)
        ep_r = 0.0
        episodes_done += 1
        obs, _ = env.reset()

ppo_rewards = np.array(ppo_rewards)
print(f"  PPO  → Mean: {ppo_rewards.mean():.2f}  Std: {ppo_rewards.std():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Random baseline – collect per-episode rewards
# ─────────────────────────────────────────────────────────────────────────────

print(f"Evaluating random baseline ({N_EVAL_EPISODES} episodes) …")
random_rewards: list[float] = []

obs, _ = env.reset()
ep_r = 0.0
episodes_done = 0

while episodes_done < N_EVAL_EPISODES:
    action = env.action_space.sample()
    obs, reward, done, truncated, _ = env.step(action)
    ep_r += reward
    if done or truncated:
        random_rewards.append(ep_r)
        ep_r = 0.0
        episodes_done += 1
        obs, _ = env.reset()

random_rewards = np.array(random_rewards)
print(f"  Random → Mean: {random_rewards.mean():.2f}  Std: {random_rewards.std():.2f}")

env.close()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Load training rewards
# ─────────────────────────────────────────────────────────────────────────────

training_rewards = np.load("training_rewards.npy")
print(f"\nTraining trace: {len(training_rewards)} episodes")


def smooth(data: np.ndarray, window: int = 30) -> np.ndarray:
    """Simple moving-average smoother (returns same-length array)."""
    kernel = np.ones(window) / window
    padded = np.pad(data, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(data)]


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 – Learning Curve
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor("#0f0f1a")

episodes = np.arange(1, len(training_rewards) + 1)
ax.plot(episodes, training_rewards, color=PPO_COLOR, alpha=0.22, linewidth=0.8, label="Episode reward")
smoothed = smooth(training_rewards, window=max(1, len(training_rewards) // 30))
ax.plot(episodes, smoothed, color=SMOOTH_COLOR, linewidth=2.2, label="Smoothed (30-ep MA)")
ax.axhline(200, color="#f9ca24", linestyle="--", linewidth=1.4, label="Target ≥ 200", alpha=0.8)
ax.axhline(0,   color="#636e72", linestyle=":",  linewidth=0.8, alpha=0.5)

ax.set_xlabel("Training Episode")
ax.set_ylabel("Episode Reward")
ax.set_title("LunarLander-v2 — PPO Learning Curve (200k timesteps)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.35)

fig.tight_layout()
fig.savefig("plots/learning_curve.png", bbox_inches="tight")
plt.close(fig)
print("  → plots/learning_curve.png")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 – Bar Chart Comparison
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_facecolor("#0f0f1a")

labels  = ["Random Policy", "PPO (200k steps)"]
means   = [random_rewards.mean(), ppo_rewards.mean()]
stds    = [random_rewards.std(),  ppo_rewards.std()]
colors  = [RANDOM_COLOR, PPO_COLOR]
x       = np.arange(len(labels))
width   = 0.45

bars = ax.bar(x, means, width, yerr=stds, color=colors, alpha=0.85,
              capsize=8, error_kw={"elinewidth": 2, "ecolor": "#ffffff99"})

for bar, val, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 6,
            f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.axhline(200, color="#f9ca24", linestyle="--", linewidth=1.4,
           label="Target ≥ 200", alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Mean Episode Reward (100 eps)")
ax.set_title("PPO vs Random Policy — Mean Reward Comparison")
ax.legend()
ax.grid(True, axis="y", alpha=0.35)

fig.tight_layout()
fig.savefig("plots/comparison_bar.png", bbox_inches="tight")
plt.close(fig)
print("  → plots/comparison_bar.png")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 – Reward Distribution (Violin + Box)
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_facecolor("#0f0f1a")

data = [random_rewards, ppo_rewards]
positions = [1, 2]

vp = ax.violinplot(data, positions=positions, showmedians=False,
                   showextrema=False, widths=0.55)
for body, clr in zip(vp["bodies"], [RANDOM_COLOR, PPO_COLOR]):
    body.set_facecolor(clr)
    body.set_alpha(0.45)
    body.set_edgecolor(clr)
    body.set_linewidth(1.5)

bp = ax.boxplot(data, positions=positions, widths=0.18,
                patch_artist=True, notch=False,
                whiskerprops=dict(color="#cccccc", linewidth=1.2),
                capprops=dict(color="#cccccc", linewidth=1.5),
                medianprops=dict(color="#f9ca24", linewidth=2.2),
                flierprops=dict(marker="o", color="#ffffff44", markersize=3))
for patch, clr in zip(bp["boxes"], [RANDOM_COLOR, PPO_COLOR]):
    patch.set_facecolor(clr)
    patch.set_alpha(0.70)

ax.axhline(200, color="#f9ca24", linestyle="--", linewidth=1.4,
           label="Target ≥ 200", alpha=0.9)
ax.set_xticks(positions)
ax.set_xticklabels(["Random Policy", "PPO (200k steps)"], fontsize=11)
ax.set_ylabel("Episode Reward")
ax.set_title("Reward Distribution — PPO vs Random Policy (100 episodes)")
ax.legend()
ax.grid(True, axis="y", alpha=0.35)

patch_r = mpatches.Patch(facecolor=RANDOM_COLOR, alpha=0.6, label="Random Policy")
patch_p = mpatches.Patch(facecolor=PPO_COLOR,    alpha=0.6, label="PPO Trained")
ax.legend(handles=[patch_r, patch_p] + [
    mpatches.Patch(facecolor="#f9ca24", alpha=0.9, label="Target ≥ 200")], loc="upper left")

fig.tight_layout()
fig.savefig("plots/reward_distribution.png", bbox_inches="tight")
plt.close(fig)
print("  → plots/reward_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

print()
print("╔══════════════════════════════════════════════════════╗")
print("║           RESULTS SUMMARY  (100 eval episodes)       ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  Random Policy  │ Mean: {random_rewards.mean():>8.2f}  │ Std: {random_rewards.std():>7.2f}  ║")
print(f"║  PPO (200k)     │ Mean: {ppo_rewards.mean():>8.2f}  │ Std: {ppo_rewards.std():>7.2f}  ║")
print("╚══════════════════════════════════════════════════════╝")
print("\nAll plots saved to  plots/")
