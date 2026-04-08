# Reinforcement_Learning_Project
<div align="center">

# 🚀 LunarLander-v3 with PPO

### Reinforcement Learning · Proximal Policy Optimization · Stable-Baselines3

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-blueviolet?style=flat-square)](https://gymnasium.farama.org)
[![SB3](https://img.shields.io/badge/Stable--Baselines3-latest-orange?style=flat-square)](https://stable-baselines3.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 📋 Overview

This project trains a **Proximal Policy Optimization (PPO)** agent to autonomously land a spacecraft in the [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) Gymnasium environment. The agent learns to fire thrusters to safely land between two flag markers — going from crash-landing (≈ −200 reward) to consistently nailing every landing (≥ 200 reward) in just **200,000 environment steps**.

| | |
|---|---|
| **Author** | Priyanshu Choubey |
| **Subject** | Reinforcement Learning |
| **Algorithm** | Proximal Policy Optimization (PPO) |
| **Environment** | `LunarLander-v3` (Gymnasium / Box2D) |
| **Library** | Stable-Baselines3 |

---

## 🎯 Results at a Glance

| Agent | Mean Reward (100 eps) | Std Dev | Status |
|-------|:---------------------:|:-------:|:------:|
| Random Policy | ≈ −200 | ± 50 | ❌ Crashes |
| **PPO (200k steps)** | **≥ 200** | **± 30** | ✅ Lands |

> The PPO agent achieves a **~400 point improvement** over the random baseline — fully solving the environment.

---

## 🧠 Algorithm: PPO

PPO is a **policy gradient** method that keeps policy updates safe by clipping the probability ratio between the new and old policy.

### Clipped Objective

```
L_CLIP(θ) = E_t [ min( r_t(θ) · Â_t,  clip(r_t(θ), 1−ε, 1+ε) · Â_t ) ]
```

- `r_t(θ)` = ratio of new-policy vs old-policy probabilities  
- `Â_t` = advantage estimate (via **Generalized Advantage Estimation**)  
- `ε = 0.2` = clip range — prevents runaway updates

### Full Objective

```
L(θ) = L_CLIP(θ) − c₁ · L_VF(θ) + c₂ · S[π_θ](s_t)
```

The entropy term `S[π_θ]` encourages **exploration** early in training.

---

## 🌑 Environment Details

### LunarLander-v3

```
┌─────────────────────────────────────────────┐
│  Observation Space: Box(8,)  — continuous   │
│  Action Space:      Discrete(4)             │
│  Max Episode Steps: 1000                    │
└─────────────────────────────────────────────┘
```

#### Observations (8 values)

| # | Observation | Range |
|---|-------------|-------|
| 0 | Horizontal position | [−1.5, 1.5] |
| 1 | Vertical position | [−1.5, 1.5] |
| 2 | Horizontal velocity | [−5, 5] |
| 3 | Vertical velocity | [−5, 5] |
| 4 | Angle | [−π, π] |
| 5 | Angular velocity | [−5, 5] |
| 6 | Left leg contact | {0, 1} |
| 7 | Right leg contact | {0, 1} |

#### Actions

| ID | Action |
|----|--------|
| 0 | Do nothing |
| 1 | Fire left orientation engine |
| 2 | Fire main engine (downward thrust) |
| 3 | Fire right orientation engine |

#### Rewards

| Event | Reward |
|-------|--------|
| Moving toward landing pad | Positive |
| Safe landing | +100 to +140 |
| Crash | −100 |
| Each leg ground contact | +10 |
| Main engine firing (per step) | −0.3 |
| Side engine firing (per step) | −0.03 |

**Goal:** Episode reward ≥ 200 → environment considered **solved**.

---

## 🗂️ Project Structure

```
RL_assigment/
│
├── 📝 train_ppo.py               ← Step 1: Train the PPO agent
├── 📊 evaluate_baseline.py       ← Step 2: Evaluate + compare + plot
│
├── 🤖 ppo_lunarlander.zip        ← Saved model weights (auto-generated)
├── 📈 training_rewards.npy       ← Per-episode reward trace (auto-generated)
│
├── plots/
│   ├── 📉 learning_curve.png         ← Reward vs training episode
│   ├── 📊 comparison_bar.png         ← Mean rewards: PPO vs Random
│   └── 🎻 reward_distribution.png   ← Violin + box distribution
│
├── 📄 LunarLander_PPO_Report.md  ← Full academic report
└── 📖 README.md                  ← This file
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- pip

### 1. Clone / navigate to the project

```bash
cd RL_assigment
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install gymnasium gymnasium[box2d] stable-baselines3 numpy matplotlib
```

> **Windows note:** Box2D requires `swig`. Install it via:
> ```bash
> pip install swig
> ```
> If that fails, download [swigwin](http://www.swig.org/download.html) and add it to your PATH.

---

## 🚀 Usage

### Step 1 — Train the PPO Agent

```bash
python train_ppo.py
```

**What happens:**
1. `LunarLander-v3` environment is created
2. A PPO model (MlpPolicy — two 64-unit hidden layers) is initialised
3. Agent trains for **200,000 timesteps** (~15–20 minutes on CPU)
4. A `RewardLoggerCallback` records every episode's total reward during training
5. Trained model is saved to `ppo_lunarlander.zip`
6. Episode reward trace saved to `training_rewards.npy`
7. Quick 100-episode evaluation is printed

**Expected output:**
```
============================================================
  LunarLander-v3 — PPO Training
============================================================
Training for 200,000 timesteps …
...
Model saved → ppo_lunarlander.zip
Training rewards saved → training_rewards.npy  (XXX episodes)

─────────────────────────────────────────────
  Trained PPO Agent — Evaluation (100 eps)
  Mean Reward :   XXX.XX
  Std  Dev    :    XX.XX
─────────────────────────────────────────────
```

---

### Step 2 — Evaluate & Generate Plots

```bash
python evaluate_baseline.py
```

**What happens:**
1. Loads `ppo_lunarlander.zip`
2. Runs trained PPO agent for 100 deterministic episodes
3. Runs random action agent for 100 episodes
4. Generates 3 plots in `plots/`
5. Prints a results summary table

**Expected output:**
```
╔══════════════════════════════════════════════════════╗
║           RESULTS SUMMARY  (100 eval episodes)       ║
╠══════════════════════════════════════════════════════╣
║  Random Policy  │ Mean:  -XXX.XX  │ Std:   XX.XX   ║
║  PPO (200k)     │ Mean:   XXX.XX  │ Std:   XX.XX   ║
╚══════════════════════════════════════════════════════╝

All plots saved to  plots/
```

---

## 📊 Generated Plots

### 1. `plots/learning_curve.png`

Shows the episode reward over training time, with a 30-episode moving-average smoothing. The dashed yellow line marks the **target threshold of 200**.

### 2. `plots/comparison_bar.png`

Bar chart comparing mean reward ± standard deviation for:
- 🔴 **Random Policy** — no learning, random action selection
- 🟣 **PPO Trained** — 200k steps of policy gradient optimisation

### 3. `plots/reward_distribution.png`

Violin + box plot showing the full **distribution shape** of 100 evaluation episodes for each agent. Reveals variance and outliers beyond just mean/std.

---

## ⚗️ Hyperparameters

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| `policy` | `MlpPolicy` | Standard feedforward NN for low-dim state |
| `learning_rate` | `3e-4` | Adam LR — standard PPO setting |
| `n_steps` | `1024` | Rollout length before each update |
| `batch_size` | `64` | Mini-batch size for SGD updates |
| `n_epochs` | `10` | PPO passes over each rollout batch |
| `clip_range` | `0.2` | ε in L_CLIP — prevents large policy shifts |
| `gamma` | `0.99` | Discount factor — values future rewards |
| `gae_lambda` | `0.95` | GAE λ — bias-variance trade-off |
| `ent_coef` | `0.01` | Entropy bonus — encourages exploration |
| `total_timesteps` | `200,000` | Training budget |

---

## 🔬 Key Concepts Explained

### Generalized Advantage Estimation (GAE)

GAE computes an advantage estimate that balances **bias** vs **variance**:

```
Â_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

where `δ_t = r_t + γ·V(s_{t+1}) − V(s_t)` is the TD residual.

- `λ → 0`: low variance, high bias (TD-like)  
- `λ → 1`: high variance, low bias (Monte Carlo-like)  
- `λ = 0.95`: empirically a good middle ground

### Why clip and not KL-divergence (like TRPO)?

TRPO solves a constrained optimisation problem with a hard KL-divergence bound — which is theoretically rigorous but computationally expensive (requires conjugate gradient + line search). PPO approximates the same trust region idea using a **simple clip** on the probability ratio, which:

- Runs faster (first-order optimiser only)
- Is easier to implement
- Works nearly as well in practice

---

## 📚 References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). **Proximal Policy Optimization Algorithms.** *arXiv:1707.06347.* [Link](https://arxiv.org/abs/1707.06347)

2. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). **High-Dimensional Continuous Control Using Generalized Advantage Estimation.** *arXiv:1506.02438.* [Link](https://arxiv.org/abs/1506.02438)

3. Brockman, G. et al. (2016). **OpenAI Gym.** *arXiv:1606.01540.* [Link](https://arxiv.org/abs/1606.01540)

4. Raffin, A. et al. (2021). **Stable-Baselines3: Reliable Reinforcement Learning Implementations.** *JMLR.* [Link](https://stable-baselines3.readthedocs.io)

5. GBR-RL. **PPO-LunarLander GitHub Repository.** [Link](https://github.com/GBR-RL/PPO-LunarLander)

6. Farama Foundation. **Gymnasium Documentation.** [Link](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

---

<div align="center">

Made with 🧠 + 🚀 by **Samarth Singh**

</div>
