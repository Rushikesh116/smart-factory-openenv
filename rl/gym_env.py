import time
import random
from typing import Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.reward import calculate_component_scores
from app.simulator import SmartFactorySimulator
from app.tasks import TASKS, TASK_LIST, grade_task

DEBUG = False


class SmartFactoryEnv(gym.Env):
    """
    Offline Gymnasium environment around SmartFactorySimulator.
    """

    metadata = {"render_modes": [None, "human"], "render_fps": 8}

    def __init__(self, task_id: str = "easy", render_mode: str | None = None):
        super().__init__()
        self.task_id = task_id
        self.render_mode = render_mode
        self.config = self._get_task_config(task_id)
        self.sim = SmartFactorySimulator(self.config)
        self.simulator = self.sim

        # 0: do nothing
        # 1-8: maintenance on machine i
        # 9-18: delay job i
        # 19-98: assign job i to machine j
        self.action_space = spaces.Discrete(99)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(40,), dtype=np.float32)

        self.max_machines = 8
        self.max_queue = 8
        self.max_action_queue = 10
        self.max_steps = float(max(self.config["max_steps"], 1))
        self.max_energy = float(max(self.config["energy_budget"] * 1.25, 1.0))
        self.max_jobs = float(max(self.config["initial_jobs"], 1))
        self.status_map = {
            "idle": 0,
            "busy": 1,
            "working": 1,
            "broken": 2,
            "maintenance": 3,
            "intermittent_failure": 2,
        }

        self._last_state: Dict[str, Any] | None = None
        self._last_render_ts = 0.0
        self._action_count = 0
        self._valid_action_count = 0

    def _get_task_config(self, task_id: str) -> Dict[str, Any]:
        if isinstance(TASKS, dict) and task_id in TASKS:
            return dict(TASKS[task_id]["config"])
        for task in TASK_LIST:
            if task["id"] == task_id:
                return dict(task["config"])
        return dict(TASK_LIST[0]["config"])

    def _normalize(self, value: float, upper: float) -> float:
        return float(np.clip(value / max(upper, 1e-6), 0.0, 1.0))

    def _queue_delay(self, state: Dict[str, Any]) -> tuple[float, float]:
        queue = state.get("job_queue", [])
        if not queue:
            return 0.0, 0.0
        waiting_times = [float(job.get("waiting_time", 0.0)) for job in queue]
        return float(np.mean(waiting_times)), float(np.max(waiting_times))

    def _completion_rate(self, state: Dict[str, Any]) -> float:
        jobs_completed = float(state.get("jobs_completed", 0.0))
        total_jobs = jobs_completed + float(state.get("queue_length", len(state.get("job_queue", []))))
        return float(np.clip(jobs_completed / max(total_jobs, 1.0), 0.0, 1.0))

    def _get_obs(self, state: Dict[str, Any]) -> np.ndarray:
        avg_delay, _ = self._queue_delay(state)
        queue_length = float(state.get("queue_length", len(state.get("job_queue", []))))
        throughput = float(state.get("throughput", state.get("jobs_completed", 0.0) / max(state.get("time_step", 1), 1)))
        breakdown_risk = float(np.clip(state.get("breakdown_risk", 0.0), 0.0, 1.0))

        obs = [
            self._normalize(float(state.get("time_step", 0.0)), self.max_steps),
            self._normalize(float(state.get("current_energy_usage", 0.0)), self.max_energy),
            self._normalize(float(state.get("energy_budget", 0.0)), self.max_energy),
            float(np.clip(throughput, 0.0, 1.0)),
            self._normalize(avg_delay, self.max_steps),
            self._normalize(queue_length, self.max_jobs),
            breakdown_risk,
            self._normalize(float(state.get("jobs_completed", 0.0)), self.max_jobs),
        ]

        machines = state.get("machines", [])
        for i in range(self.max_machines):
            if i < len(machines):
                machine = machines[i]
                obs.append(self.status_map.get(machine.get("status", "idle"), 0) / 3.0)
                obs.append(float(np.clip(machine.get("health", 0.0), 0.0, 1.0)))
            else:
                obs.extend([0.0, 0.0])

        queue = state.get("job_queue", [])
        for i in range(self.max_queue):
            if i < len(queue):
                job = queue[i]
                obs.append(float(np.clip(job.get("priority", 1) / 3.0, 0.0, 1.0)))
                obs.append(self._normalize(float(job.get("waiting_time", 0.0)), self.max_steps))
            else:
                obs.extend([0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def _decode_action(self, action_idx: int, state: Dict[str, Any]) -> tuple[Dict[str, Any], float, bool]:
        penalty = 0.0
        machines = state.get("machines", [])
        queue = state.get("job_queue", [])
        energy_budget = float(max(state.get("energy_budget", 1.0), 1.0))
        current_energy = float(state.get("current_energy_usage", 0.0))
        allow_delay = len(queue) > 0 and (current_energy / energy_budget > 0.85 or not any(
            machine.get("status") == "idle" and machine.get("health", 0.0) > 0.35
            for machine in machines
        ))

        if action_idx == 0:
            return {"type": "do_nothing"}, penalty, True

        if 1 <= action_idx <= 8:
            machine_idx = action_idx - 1
            if machine_idx < len(machines):
                machine = machines[machine_idx]
                valid = machine.get("status") in {"idle", "broken", "intermittent_failure"} and (
                    machine.get("health", 1.0) < 0.9 or machine.get("status") in {"broken", "intermittent_failure"}
                )
                if valid:
                    return {"type": "maintenance", "machine_id": machine["id"]}, penalty, True
            return {"type": "do_nothing"}, penalty - 0.05, False

        if 9 <= action_idx <= 18:
            job_idx = action_idx - 9
            if job_idx < len(queue) and allow_delay:
                return {"type": "delay", "job_id": queue[job_idx]["id"]}, penalty, True
            return {"type": "do_nothing"}, penalty - 0.05, False

        job_idx = (action_idx - 19) // 8
        machine_idx = (action_idx - 19) % 8
        if job_idx < len(queue) and machine_idx < len(machines):
            machine = machines[machine_idx]
            if machine.get("status") == "idle" and machine.get("health", 0.0) > 0.35:
                return {
                    "type": "assign",
                    "job_id": queue[job_idx]["id"],
                    "machine_id": machine["id"],
                }, penalty, True
        return {"type": "do_nothing"}, penalty - 0.05, False

    def action_masks(self) -> np.ndarray:
        state = self.sim.get_state() if self._last_state is None else self._last_state
        masks = np.zeros(self.action_space.n, dtype=bool)
        machines = state.get("machines", [])
        queue = state.get("job_queue", [])
        energy_budget = float(max(state.get("energy_budget", 1.0), 1.0))
        current_energy = float(state.get("current_energy_usage", 0.0))
        allow_delay = len(queue) > 0 and (current_energy / energy_budget > 0.85 or not any(
            machine.get("status") == "idle" and machine.get("health", 0.0) > 0.35
            for machine in machines
        ))

        for index, machine in enumerate(machines[:8]):
            valid = machine.get("status") in {"idle", "broken", "intermittent_failure"} and (
                machine.get("health", 1.0) < 0.9 or machine.get("status") in {"broken", "intermittent_failure"}
            )
            if valid:
                masks[1 + index] = True

        if allow_delay:
            for index in range(min(self.max_action_queue, len(queue))):
                masks[9 + index] = True

        for job_index in range(min(self.max_action_queue, len(queue))):
            for machine_index, machine in enumerate(machines[:8]):
                if machine.get("status") == "idle" and machine.get("health", 0.0) > 0.35:
                    masks[19 + job_index * 8 + machine_index] = True

        if not np.any(masks[1:]):
            masks[0] = True
        return masks

    def _shape_reward(
        self,
        state: Dict[str, Any],
        simulator_reward: float,
        invalid_penalty: float,
    ) -> tuple[float, Dict[str, float]]:
        components = calculate_component_scores(state)
        reward = float(np.clip(simulator_reward + invalid_penalty, -1.0, 1.0))

        reward_breakdown = {
            "energy_score": components["energy_score"],
            "throughput_score": components["throughput_score"],
            "delay_score": components["delay_score"],
            "breakdown_penalty": components["breakdown_penalty"],
            "simulator_reward": simulator_reward,
            "invalid_action_penalty": invalid_penalty,
            "total_reward": reward,
        }
        return reward, reward_breakdown

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.sim.reset()
        self._last_state = self.sim.get_state()
        self._action_count = 0
        self._valid_action_count = 0
        obs = self._get_obs(self._last_state)
        info = {"task_id": self.task_id, "initial_jobs": self.config["initial_jobs"]}
        return obs, info

    def step(self, action: int):
        current_state = self.sim.get_state() if self._last_state is None else self._last_state
        decoded_action, invalid_penalty, is_valid_action = self._decode_action(int(action), current_state)
        if DEBUG:
            print(f"[DEBUG] action={action} decoded={decoded_action} valid={is_valid_action}")

        next_state, simulator_reward, done, sim_info = self.sim.step(decoded_action)
        self.simulator.state = next_state
        terminated = bool(done and len(next_state.get("job_queue", [])) == 0)
        truncated = bool(done and not terminated)

        self._action_count += 1
        if is_valid_action:
            self._valid_action_count += 1

        avg_delay, max_delay = self._queue_delay(next_state)
        jobs_completed = int(next_state.get("jobs_completed", 0))
        machine_states = [machine.get("status", "idle") for machine in next_state.get("machines", [])]
        idle_machines = int(sum(1 for status in machine_states if status == "idle"))
        busy_machines = int(sum(1 for status in machine_states if status in {"busy", "working"}))
        broken_machines = int(sum(1 for status in machine_states if status in {"broken", "intermittent_failure"}))
        queue_length = int(next_state.get("queue_length", len(next_state.get("job_queue", []))))
        score = grade_task(self.task_id, next_state)
        try:
            score = float(score)
            if not np.isfinite(score):
                score = 0.5
        except:
            score = 0.5

        score = max(0.01, min(0.99, score))
        reward = score
        components = calculate_component_scores(next_state)
        reward_breakdown = {
            "energy_score": components.get("energy_score", 0.5),
            "throughput_score": components.get("throughput_score", 0.5),
            "delay_score": components.get("delay_score", 0.5),
            "graded_score": score,
            "simulator_reward": float(simulator_reward),
            "invalid_action_penalty": invalid_penalty,
            "total_reward": reward,
        }

        info = {
            "jobs_completed": jobs_completed,
            "queue_length": queue_length,
            "energy_budget": float(next_state.get("energy_budget", 0.0)),
            "idle_machines": idle_machines,
            "busy_machines": busy_machines,
            "broken_machines": broken_machines,
            "completion_rate": self._completion_rate(next_state),
            "current_energy_usage": float(next_state.get("current_energy_usage", 0.0)),
            "throughput": float(next_state.get("throughput", jobs_completed / max(next_state.get("time_step", 1), 1))),
            "avg_waiting_time": avg_delay,
            "max_waiting_time": max_delay,
            "delay": float(next_state.get("delay", avg_delay)),
            "breakdown_risk": float(np.clip(next_state.get("breakdown_risk", 0.0), 0.0, 1.0)),
            "action_valid": bool(is_valid_action),
            "action_validity_rate": float(self._valid_action_count / max(self._action_count, 1)),
            "reward_breakdown": reward_breakdown,
        }
        if isinstance(sim_info, dict):
            info.update(sim_info)

        self._last_state = next_state
        if self.render_mode == "human":
            self.render()
        return self._get_obs(next_state), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None

        state = self.sim.get_state()
        now = time.time()
        min_period = 1.0 / float(self.metadata["render_fps"])
        if now - self._last_render_ts < min_period:
            return None
        self._last_render_ts = now

        avg_delay, _ = self._queue_delay(state)
        print(
            "Step: {step} | Jobs: {jobs} | Queue: {queue} | Energy: {energy:.1f}/{budget:.1f} | Delay: {delay:.2f}".format(
                step=state.get("time_step", 0),
                jobs=state.get("jobs_completed", 0),
                queue=state.get("queue_length", len(state.get("job_queue", []))),
                energy=state.get("current_energy_usage", 0.0),
                budget=state.get("energy_budget", 0.0),
                delay=avg_delay,
            )
        )
        return None

    def close(self):
        return None
