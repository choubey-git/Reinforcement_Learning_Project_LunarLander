"""
record_video.py
===============
Records a MP4 video of your trained PPO agent landing the spacecraft.
Saves the video to:  videos/  folder.

Requirements
------------
    ppo_lunarlander.zip must exist (run train_ppo.py first)
    pip install moviepy  (if not already installed)

Usage
-----
    python record_video.py
"""

import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

VIDEO_FOLDER  = "videos"
N_EPISODES    = 3          # record 3 episodes
os.makedirs(VIDEO_FOLDER, exist_ok=True)

print("=" * 50)
print("  🎬 LunarLander — RECORD VIDEO MODE")
print("=" * 50)

# ── Load model ────────────────────────────────────────────────────────────────
print("\nLoading trained PPO model…")
try:
    model = PPO.load("ppo_lunarlander")
except FileNotFoundError:
    print("\n❌  ppo_lunarlander.zip not found!")
    print("   Please run  python train_ppo.py  first.\n")
    exit(1)

# ── Create environment with video recorder ────────────────────────────────────
env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder=VIDEO_FOLDER,
    name_prefix="lunarlander_ppo",
    episode_trigger=lambda ep: True,   # record EVERY episode
)

print(f"Recording {N_EPISODES} episodes → {VIDEO_FOLDER}/\n")

# ── Run episodes ──────────────────────────────────────────────────────────────
for ep in range(N_EPISODES):
    obs, _ = env.reset()
    ep_reward = 0.0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward
        done = terminated or truncated

    status = "✅ LANDED!" if ep_reward > 100 else ("💥 CRASHED" if ep_reward < -50 else "🟡 OK")
    print(f"  Episode {ep+1}: Reward {ep_reward:>8.2f}  {status}")

env.close()

# ── List saved files ──────────────────────────────────────────────────────────
print(f"\nVideos saved in  {VIDEO_FOLDER}/:")
for f in sorted(os.listdir(VIDEO_FOLDER)):
    if f.endswith(".mp4"):
        size_kb = os.path.getsize(os.path.join(VIDEO_FOLDER, f)) // 1024
        print(f"  📹 {f}  ({size_kb} KB)")

print("\nDone! Open the .mp4 files to watch the recordings.")
