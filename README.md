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
---

# 🏭 AI-Powered Smart Factory Optimization (OpenEnv)

🚀 **Live Demo:** https://huggingface.co/spaces/rushikesh116/smart-factory-openenv
💻 **Code:** https://github.com/Rushikesh116/smart-factory-openenv

---

## 🧠 Overview

This project builds a **Reinforcement Learning environment** for optimizing smart factory operations using OpenEnv.

The goal is to balance:

* 🔋 Energy consumption
* ⚡ Throughput
* ⏱️ Delay
* ⚠️ Breakdown risk

The agent learns to make efficient decisions in a dynamic industrial setting.

---

## 🏭 Environment Overview

The environment simulates a factory with:

* Multiple machines
* Task queues
* Energy tracking
* Delay accumulation
* Stochastic breakdown risk

Each episode represents a production cycle.

---

## 🔍 Observation Space

The agent observes:

```text
[energy, throughput, delay, queue_length, breakdown_risk]
```

---

## 🎮 Action Space

* Discrete actions
* Each action = machine control decision
* Maskable (invalid actions are filtered)

---

## 🎯 Tasks & Difficulty Levels

| Task   | Description                      | Difficulty |
| ------ | -------------------------------- | ---------- |
| Easy   | Stable demand, fewer constraints | 🟢 Easy    |
| Medium | Moderate workload                | 🟡 Medium  |
| Hard   | High demand + failures           | 🔴 Hard    |

---

## 🧮 Reward Design

* Multi-objective reward
* Encourages:

  * Higher throughput
  * Lower energy
  * Lower delay
* Penalizes unsafe states

👉 All scores are normalized in **(0, 1)**

---

## 🏆 Results

| Metric          | Random | RL Agent    |
| --------------- | ------ | ----------- |
| Total Reward    | 10.12  | **10.67**   |
| Avg Energy      | 29.60  | **28.81 ↓** |
| Avg Throughput  | 0.485  | **0.528 ↑** |
| Avg Delay       | 13.05  | 13.08       |
| Completion Rate | 0.85   | 0.85        |

### 📈 Improvements

* Reward: **+5.4%**
* Throughput: **+8.9%**
* Energy: **+2.7%**
* Delay: ~ unchanged

---

## 📊 Baseline

Random policy:

* No optimization
* Lower reward (~0.40–0.45 per step)
* Inefficient decisions

---

## ⚙️ Setup

```bash
git clone https://github.com/Rushikesh116/smart-factory-openenv
cd smart-factory-openenv
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train

```bash
python -m rl.train
```

### Evaluate
```bash
python -m rl.evaluate
```


### Inference

```bash
python inference.py
```

### Run Demo(Optional)

```bash
python app.py
```

---

## 🎮 Demo

👉 https://huggingface.co/spaces/rushikesh116/smart-factory-openenv

* Interactive simulation
* Real-time metrics
* RL vs Random comparison

---

## 💡 Why RL Works

* Learns trade-offs automatically
* Handles conflicting objectives
* Adapts to dynamic conditions

---

## 🛠️ Tech Stack

* Python
* Stable-Baselines3
* OpenEnv
* Gradio

---

## 👤 Author

Rushikesh Yadav
