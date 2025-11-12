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
