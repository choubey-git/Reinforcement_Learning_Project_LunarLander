"""
watch_agent.py
==============
Opens a LIVE pygame window and lets you WATCH your trained PPO agent
land the spacecraft in real time. Runs multiple episodes so you can
see it succeed (and sometimes fail!).

Requirements
------------
    ppo_lunarlander.zip must exist (run train_ppo.py first)

Usage
-----
    python watch_agent.py

Controls
--------
    Close the window to stop.
"""

import time
import gymnasium as gym
from stable_baselines3 import PPO

# ── How many episodes to watch ──────────────────────────────────────────────
N_EPISODES = 5          # watch 5 landings
RENDER_FPS  = 60        # frames per second (smooth)
SLOW_MOTION = False     # set True to slow it down (0.5x speed)

# ── Load the trained model ───────────────────────────────────────────────────
print("=" * 50)
print("  🚀 LunarLander — WATCH MODE")
print("=" * 50)
print("\nLoading trained PPO model…")

try:
    model = PPO.load("ppo_lunarlander")
except FileNotFoundError:
    print("\n❌  ppo_lunarlander.zip not found!")
    print("   Please run  python train_ppo.py  first.\n")
    exit(1)

# ── Create environment WITH rendering ────────────────────────────────────────
env = gym.make(
    "LunarLander-v3",
    render_mode="human",          # ← this opens the pygame window
)

print(f"\n✅  Model loaded. Watching {N_EPISODES} episodes…")
print("   Close the window to quit.\n")

# ── Episode loop ──────────────────────────────────────────────────────────────
total_rewards = []

for ep in range(1, N_EPISODES + 1):
    obs, _ = env.reset()
    ep_reward = 0.0
    step = 0
    done = False

    print(f"Episode {ep}/{N_EPISODES}  ", end="", flush=True)

    while not done:
        # Agent picks the best action (deterministic = True → no randomness)
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        step += 1
        done = terminated or truncated

        if SLOW_MOTION:
            time.sleep(1 / RENDER_FPS)  # slow-mo pause

    total_rewards.append(ep_reward)
    status = "✅ LANDED!" if ep_reward > 100 else ("💥 CRASHED" if ep_reward < -50 else "🟡 OK")
    print(f"Reward: {ep_reward:>8.2f}  Steps: {step:>4}  {status}")

env.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*40}")
print(f"  Watched {N_EPISODES} episodes")
print(f"  Mean Reward : {sum(total_rewards)/len(total_rewards):.2f}")
print(f"  Best Episode: {max(total_rewards):.2f}")
print(f"  Worst Episode: {min(total_rewards):.2f}")
print(f"{'─'*40}\n")
