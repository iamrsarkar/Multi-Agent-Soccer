# 🎮 Multi-Agent-Soccer(Competitive Reinforcement Learning)

## 📘 Overview
This project implements **competitive multi-agent reinforcement learning (MARL)** for game-like environments such as **soccer**.  
Each agent independently learns to **cooperate with teammates** and **compete against opponents** using **self-play** and **policy gradient** methods.

The project demonstrates emergent teamwork, strategy formation, and adaptive play dynamics in simulated multi-agent environments.

---

## 🎯 Objective
Develop an **autonomous game AI system** in which multiple agents learn to:
- Compete against each other using reinforcement learning.  
- Develop cooperative team strategies in a shared environment.  
- Improve through **self-play** and **league training** (similar to DeepMind’s AlphaStar).  

---

## 🧩 Concept
Each player or unit is modeled as an **independent agent** that:
- Observes the game state (e.g., position, velocity, ball location).
- Chooses an action (move, shoot, pass, defend).
- Receives a reward based on performance (goals, captures, wins).  

Agents train via **multi-agent policy gradients**, using **centralized training with decentralized execution** (CTDE).  

---

## 🏗️ Environment Setup

### 🔹 Example Environments
- **⚽ Soccer (2v2)** – agents learn to score and defend.
- **🚩 Capture-the-Flag** – two teams try to capture the opponent’s flag.
- **🏓 Pong-Team** – cooperative paddle control to keep the ball in play.
- **🐾 PettingZoo Envs:** `simple_spread`, `multiwalker`, `pistonball`.

---

### 🔹 Observations
Each agent observes:
- Its own position, velocity, orientation.
- Relative positions of teammates, opponents, and objectives (e.g., ball or flag).
- Global game features (time left, score).

### 🔹 Actions
Continuous or discrete action space:
- Move Up / Down / Left / Right
- Pass / Shoot / Defend / Idle  

### 🔹 Rewards
Example reward shaping (Soccer):
\[
R_t = R_\text{goal} + R_\text{teamwork} - R_\text{foul} - R_\text{distance}
\]
Where:
- \( R_\text{goal} = +1 \) per goal scored  
- \( R_\text{teamwork} = +0.1 \) for successful passes  
- \( R_\text{foul} = -0.5 \) for collisions or going out of bounds  
- \( R_\text{distance} = -\text{dist(ball, goal)} \) for shaping movement  

---

## ⚙️ Algorithms Implemented

| Algorithm | Description | Application |
|------------|--------------|--------------|
| **Self-Play PPO** | Agents train by competing with versions of themselves | Core training loop |
| **League Training** | Multiple policy pools compete and evolve (AlphaStar-style) | Advanced training |
| **Centralized Critic, Decentralized Actors** | Shared value estimation for cooperative–competitive balance | Stability in multi-agent updates |
| **Curriculum Learning** | Gradually increases difficulty (1v1 → 2v2) | Robust policy formation |

---

## 🧠 Architecture

### Training Flow


┌───────────────────────────────────────────────┐
│ Initialize environment (PettingZoo/Unity) │
├───────────────────────────────────────────────┤
│ For each episode: │
│ • Agents observe environment │
│ • Take actions using current policy │
│ • Environment updates game state │
│ • Compute rewards for all agents │
│ • Store experiences (state, action, reward) │
│ • Update policies via PPO or League strategy │
└───────────────────────────────────────────────┘


### Directory Structure


├── envs/ # Game environments (PettingZoo or Unity)
│ ├── soccer_env.py
│ ├── capture_flag_env.py
│ └── pong_team_env.py
├── agents/ # RL agent implementations
│ ├── ppo_agent.py
│ ├── selfplay_manager.py
│ └── centralized_critic.py
├── training/ # Training & evaluation loops
│ ├── train_selfplay.py
│ └── league_training.py
├── results/ # Logs, graphs, and replay files
├── models/ # Trained checkpoints
└── main.py # Entry point


---

## 🧩 Frameworks & Libraries

- 🧠 **Reinforcement Learning:** PyTorch, Stable-Baselines3, RLlib  
- 🕹️ **Simulation Environments:** PettingZoo, Gymnasium, Unity ML-Agents  
- 📊 **Visualization:** Matplotlib, TensorBoard  
- ⚙️ **Physics (optional):** PyBullet or Mujoco  

---

## 📈 Evaluation Metrics

| Metric | Description |
|---------|--------------|
| **Win Rate** | % of matches won by agent/team |
| **Goal Difference** | Average goals scored − conceded |
| **Average Reward** | Mean episode reward |
| **Policy Entropy** | Diversity in learned strategies |
| **Training Stability** | Reward variance across episodes |

---

## 🎮 Experiments

| Experiment | Goal | Setup |
|-------------|------|-------|
| 1 | Train 1v1 Self-Play PPO | Baseline |
| 2 | Add Team Coordination (2v2 Soccer) | Shared rewards |
| 3 | League Training with Evolving Opponents | AlphaStar-style |
| 4 | Curriculum Difficulty (Easy → Hard Maps) | Progressive learning |

---

## 🚀 How to Run

### 1️⃣ Install Dependencies
```bash
conda create -n marl_game python=3.10
conda activate marl_game
pip install torch gymnasium pettingzoo stable-baselines3 matplotlib

2️⃣ Train Agent
python main.py --env soccer --algo selfplay_ppo --episodes 10000

3️⃣ Evaluate Policy
python evaluate.py --model models/soccer_ppo_final.pth

4️⃣ Visualize Results
python visualize.py --env soccer

📊 Visualization

Training Curves (Average Reward, Win Rate)

Agent Trajectories

Replay Videos (if using Unity ML-Agents)

🧩 Research Extensions

Add Graph Neural Networks (GNN) for agent communication.

Explore Opponent Modeling (explicit opponent policy prediction).

Combine Self-Play + Imitation Learning (for human-like strategies).

Integrate League ELO rating for opponent matchmaking.

📚 References

Silver et al., “Mastering the Game of Go with Deep Neural Networks and Tree Search,” Nature, 2016.

Vinyals et al., “Grandmaster Level in StarCraft II using Multi-Agent Reinforcement Learning,” Nature, 2019 (AlphaStar).

PettingZoo: Multi-Agent Reinforcement Learning Environment Library.

Schulman et al., “Proximal Policy Optimization (PPO),” 2017.

👨‍💻 Contributors

Ronak Sarkar – Project Lead, Multi-Agent RL Researcher

Group RR – Team Members (Radheshyam Routh, Ronak Sarkar)

MSc Big Data Analytics, RKMVERI (2024–2026)

🪙 License

MIT License © 2025 Ronak Sarkar
You are free to use, modify, and distribute this code with proper attribution.

🖼️ Example Simulation Snapshot


---

Would you like me to **generate this README.md file (downloadable)** or also create the **project folder structure with stub `.py` files** so you can directly initialize it as a GitHub repo (with working placeholders for PettingZoo + PPO integration)?

