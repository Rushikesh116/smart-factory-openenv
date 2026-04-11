---
title: Smart Factory OpenEnv
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - reinforcement-learning
  - smart-factory
  - openenv
  - ppo
  - gymnasium
---

# 🏭 Smart Factory Optimization — OpenEnv

An advanced **Reinforcement Learning environment** for optimizing machine scheduling, energy management, and predictive maintenance in a simulated smart factory. Built on the [OpenEnv](https://github.com/openenv) framework with the **Gymnasium** API.

🚀 **Live Space:** [pheonix-overwatch/smart-factory-openenv](https://huggingface.co/spaces/pheonix-overwatch/smart-factory-openenv)

---

## 📋 Problem Statement

Manage a factory floor with multiple machines, a stream of incoming jobs, and a limited energy budget. The agent must learn to:

- **Schedule jobs** to idle machines based on priority and processing time
- **Manage energy** — each machine consumes energy proportional to job priority and machine wear
- **Perform proactive maintenance** — machines degrade over time and can break down stochastically
- **Minimize delay** — jobs waiting in queue accumulate waiting time penalties

The challenge scales across three difficulty levels with increasing machines, jobs, failure rates, and tighter energy constraints.

---

## 🔍 Observation Space

The environment exposes a **40-dimensional** continuous observation vector (`Box(0, 1, shape=(40,))`):

| Index | Feature | Description | Range |
|-------|---------|-------------|-------|
| 0 | `time_step` | Normalized simulation time | [0, 1] |
| 1 | `current_energy` | Current energy usage / max | [0, 1] |
| 2 | `energy_budget` | Budget headroom | [0, 1] |
| 3 | `throughput` | Jobs completed per step | [0, 1] |
| 4 | `avg_delay` | Average queue waiting time | [0, 1] |
| 5 | `queue_length` | Normalized queue size | [0, 1] |
| 6 | `breakdown_risk` | System-wide breakdown risk | [0, 1] |
| 7 | `jobs_completed` | Normalized completion count | [0, 1] |
| 8–23 | `machine_status[i]`, `machine_health[i]` | Per-machine status (idle=0, busy=0.33, broken=0.67, maint=1.0) and health | [0, 1] × 8 machines |
| 24–39 | `job_priority[i]`, `job_wait[i]` | Per-job priority (1–3 normalized) and waiting time | [0, 1] × 8 queue slots |

---

## 🎮 Action Space

**Discrete(99)** with the following mapping:

| Action Range | Type | Description |
|-------------|------|-------------|
| `0` | `do_nothing` | Skip this time step |
| `1–8` | `maintenance` | Trigger maintenance on machine `i-1` |
| `9–18` | `delay` | Delay job `i-9` (push to back of queue) |
| `19–98` | `assign` | Assign job `(a-19)//8` to machine `(a-19)%8` |

**Action masking** is supported — invalid actions (e.g., assigning to a busy machine) are masked out using `sb3-contrib`'s `MaskablePPO`.

---

## 🎯 Tasks & Difficulty Levels

| Task | ID | Objective | Machines | Jobs | Energy Budget | Max Steps | Failure Rate |
|------|-----|-----------|----------|------|---------------|-----------|-------------|
| 🟢 **Easy** | `easy` | Energy Efficiency | 3 | 12 | 28.0 | 60 | 1.5% |
| 🟡 **Medium** | `medium` | Throughput Optimization | 5 | 20 | 44.0 | 90 | 2.5% |
| 🔴 **Hard** | `hard` | Low Latency Scheduling | 6 | 24 | 52.0 | 110 | 3.5% |

Each task has a **dedicated grader** that evaluates the metric most relevant to the objective:
- **Easy:** `energy_efficiency_grader` — rewards lower energy usage relative to budget
- **Medium:** `throughput_grader` — rewards higher job completion rate per step
- **Hard:** `delay_grader` — rewards lower average waiting time

---

## 🧮 Reward Function

The reward is a **weighted multi-objective** score combining three components minus a breakdown penalty:

```
score = 0.4 × energy_score + 0.4 × throughput_score + 0.2 × delay_score - breakdown_penalty
```

| Component | Formula | Weight |
|-----------|---------|--------|
| Energy Score | `1 - (current_energy / energy_bound)` | 40% |
| Throughput Score | `throughput / throughput_bound` | 40% |
| Delay Score | `1 - (delay / delay_bound)` | 20% |
| Breakdown Penalty | `risk × 0.15 + broken_ratio × 0.10` | Subtracted |

All scores are **clamped to (0.01, 0.99)** to ensure valid grading.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   inference.py                       │
│   Decision Cascade:                                  │
│   Trained MaskablePPO → LLM Reasoning → Heuristic   │
└─────────────────┬───────────────────────────────────┘
                  │ action (int 0-98)
                  ▼
┌─────────────────────────────────────────────────────┐
│              SmartFactoryEnv (Gymnasium)              │
│   • Observation: 40-dim float vector                 │
│   • Action masking for invalid actions               │
│   • Reward shaping with graded + shaped blend        │
└─────────────────┬───────────────────────────────────┘
                  │ decoded action dict
                  ▼
┌─────────────────────────────────────────────────────┐
│           SmartFactorySimulator                      │
│   • Machine health degradation                       │
│   • Stochastic breakdowns & intermittent failures    │
│   • Energy consumption modeling                      │
│   • Job queue management                             │
└─────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│            FastAPI Server (server/app.py)             │
│   POST /reset • POST /step • GET /state              │
│   GET /tasks  • GET /grader • GET /health            │
└─────────────────────────────────────────────────────┘
```

---

## 🏆 Results

Performance comparison across policies on the **medium** task (5 episodes):

| Metric | Random | Heuristic | **PPO Agent** |
|--------|--------|-----------|---------------|
| Avg Reward | 29.36 | 12.06 | **54.34** |
| Avg Throughput | 0.272 | 0.757 | **0.211** |
| Avg Energy | 9.79 | 23.22 | **8.15** |
| Avg Delay | 32.67 | 11.45 | **3.12** |
| Completion Rate | 100% | 100% | **100%** |
| Action Validity | 32.5% | 100% | **100%** |

> The PPO agent achieves **4.5× the reward** of the heuristic baseline with significantly lower energy usage and delay.

---

## ⚙️ Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/Rushikesh116/smart-factory-openenv
cd smart-factory-openenv
pip install -r requirements.txt
```

### Docker

```bash
docker build -t smart-factory .
docker run -p 7860:7860 smart-factory
```

---

## ▶️ Usage

### Train an RL Agent

```bash
# Single task training
python -m rl.train --task-id medium --timesteps 50000

# Curriculum training (easy → medium → hard)
python -m rl.train --mode curriculum
```

### Evaluate

```bash
python -m rl.evaluate --num-episodes 10
```

### Run Inference

```bash
python inference.py
```

### Start Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## 📂 Project Structure

```
smart-factory-openenv/
├── app/
│   ├── __init__.py          # App initialization
│   ├── simulator.py          # Core factory simulation engine
│   ├── models.py             # Pydantic data models (State, Action, Machine, Job)
│   ├── reward.py             # Multi-objective reward function
│   ├── grader.py             # Episode grading logic
│   └── tasks.py              # Task definitions & grader registry
├── rl/
│   ├── gym_env.py            # Gymnasium environment wrapper
│   ├── train.py              # MaskablePPO training with curriculum learning
│   ├── evaluate.py           # Policy evaluation & comparison
│   ├── baselines.py          # Task-aware heuristic baseline policy
│   ├── config.py             # Hyperparameter configuration
│   └── utils.py              # File utilities
├── server/
│   └── app.py                # FastAPI server (OpenEnv API)
├── models/                   # Trained model weights (.zip + .pkl)
├── inference.py              # Inference script for evaluation
├── openenv.yaml              # OpenEnv specification
├── Dockerfile                # Container configuration
└── requirements.txt          # Python dependencies
```

---

## 🔗 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/tasks` | GET | List available tasks |
| `/reset` | POST | Reset environment with `{"task_id": "easy"}` |
| `/step` | POST | Execute action with `{"action": {...}}` |
| `/state` | GET | Get current state |
| `/grader` | GET | Get current episode score |

---

## 🛠️ Tech Stack

- **RL Framework:** Stable-Baselines3 + sb3-contrib (MaskablePPO)
- **Environment:** Gymnasium
- **Server:** FastAPI + Uvicorn
- **Standard:** OpenEnv
- **ML:** PyTorch
- **Deployment:** Docker on HuggingFace Spaces

---

## 👥 Team Phoenix

- **Rushikesh Yadav** — Environment design, RL training
- **Nayan** — Agent optimization, deployment
