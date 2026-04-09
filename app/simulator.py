from __future__ import annotations

import random
from typing import Any, Dict

from app.reward import SmartFactoryReward


DEFAULT_CONFIG: Dict[str, Any] = {
    "num_machines": 4,
    "initial_jobs": 12,
    "energy_budget": 40.0,
    "failure_rate": 0.02,
    "intermittent_failure_rate": 0.015,
    "degradation_rate": 0.007,
    "high_intensity_multiplier": 1.35,
    "failure_threshold": 0.35,
    "idle_energy": 0.4,
    "maintenance_energy": 1.8,
    "intermittent_failure_energy": 0.8,
    "broken_energy": 0.2,
    "busy_energy_base": 6.0,
    "busy_priority_energy": 1.1,
    "aging_energy_penalty": 2.2,
    "maintenance_duration": 2,
    "maintenance_recovery": 0.24,
    "max_steps": 80,
    "seed": 7,
}


class SmartFactorySimulator:
    def __init__(self, config: Dict[str, Any]):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._base_seed = int(self.config.get("seed", 7))
        self._reset_index = -1
        self.reset()

    def reset(self):
        self._reset_index += 1
        self.rng = random.Random(self._base_seed + self._reset_index)
        self.machines = [self._build_machine(i) for i in range(self.config["num_machines"])]
        self.job_queue = [self.generate_job(0, i) for i in range(self.config["initial_jobs"])]
        self.next_job_id = self.config["initial_jobs"]
        self.time_step = 0
        self.total_reward = 0.0
        self.jobs_completed = 0
        self.done = False
        self.current_energy_usage = 0.0
        self.cumulative_energy_usage = 0.0
        self.queue_length = len(self.job_queue)
        self.throughput = 0.0
        self.delay = 0.0
        self.avg_waiting_time = 0.0
        self.max_waiting_time = 0.0
        self.breakdown_risk = 0.0
        self.breakdown_events = 0
        self.broken_machines = 0
        self.maintenance_machines = 0
        self.idle_machines = len(self.machines)
        self.busy_machines = 0
        self._refresh_metrics(accumulate=False)
        return self.get_state()

    def _build_machine(self, machine_index: int) -> Dict[str, Any]:
        return {
            "id": f"machine_{machine_index}",
            "status": "idle",
            "health": 1.0,
            "energy_usage": float(self.config["idle_energy"]),
            "current_job_id": None,
            "current_job_priority": None,
            "remaining_time": 0,
            "breakdown_risk": 0.0,
            "cycles_since_maintenance": 0,
        }

    def generate_job(self, time_step: int, job_id: int):
        priority = self.rng.randint(1, 3)
        processing_time = self.rng.randint(2, 5) + (1 if priority == 3 else 0)
        return {
            "id": f"job_{job_id}",
            "processing_time": processing_time,
            "priority": priority,
            "arrival_time": time_step,
            "waiting_time": 0,
        }

    def _safe_index(self, value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _machine_by_id(self, machine_id: Any) -> Dict[str, Any] | None:
        if machine_id is None:
            return None
        for machine in self.machines:
            if machine["id"] == machine_id:
                return machine
        return None

    def _job_index_by_id(self, job_id: Any) -> int:
        if job_id is None:
            return -1
        for index, job in enumerate(self.job_queue):
            if job["id"] == job_id:
                return index
        return -1

    def _normalize_action(self, action: Dict[str, Any] | None) -> Dict[str, Any]:
        payload = action if isinstance(action, dict) else {}
        action_type = payload.get("type")

        if action_type in {"assign", "maintenance", "delay", "do_nothing"}:
            return {
                "type": action_type,
                "job_id": payload.get("job_id"),
                "machine_id": payload.get("machine_id"),
            }

        machine_index = self._safe_index(payload.get("machine"))
        task_index = self._safe_index(payload.get("task"))

        if machine_index is not None and task_index is not None:
            if 0 <= machine_index < len(self.machines) and 0 <= task_index < len(self.job_queue):
                return {
                    "type": "assign",
                    "machine_id": self.machines[machine_index]["id"],
                    "job_id": self.job_queue[task_index]["id"],
                }
            if 0 <= machine_index < len(self.machines) and task_index == -1:
                return {"type": "maintenance", "machine_id": self.machines[machine_index]["id"]}
            if machine_index == -1 and 0 <= task_index < len(self.job_queue):
                return {"type": "delay", "job_id": self.job_queue[task_index]["id"]}

        return {"type": "do_nothing", "job_id": None, "machine_id": None}

    def _machine_breakdown_risk(self, machine: Dict[str, Any]) -> float:
        status = machine.get("status", "idle")
        if status == "broken":
            return 0.99
        if status == "intermittent_failure":
            return 0.8
        if status == "maintenance":
            return 0.08

        health_pressure = 1.0 - float(machine.get("health", 1.0))
        load_pressure = 0.03 if status == "idle" else 0.22
        priority_pressure = 0.0
        if status == "busy":
            priority_pressure = 0.08 * max(int(machine.get("current_job_priority") or 1) - 1, 0)
        maintenance_pressure = min(float(machine.get("cycles_since_maintenance", 0)) / 8.0, 1.0) * 0.12
        risk = (
            float(self.config["failure_rate"])
            + (health_pressure * 0.55)
            + load_pressure
            + priority_pressure
            + maintenance_pressure
        )
        return max(0.01, min(0.99, risk))

    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, float]:
        context = {
            "jobs_completed": 0.0,
            "maintenance_completed": 0.0,
            "invalid_action": 0.0,
            "breakdowns": 0.0,
            "delayed_jobs": 0.0,
        }
        action_type = action.get("type", "do_nothing")

        if action_type == "assign":
            machine = self._machine_by_id(action.get("machine_id"))
            job_index = self._job_index_by_id(action.get("job_id"))
            valid_assignment = (
                machine is not None
                and job_index >= 0
                and machine["status"] == "idle"
                and float(machine["health"]) >= float(self.config["failure_threshold"])
            )
            if valid_assignment:
                job = self.job_queue.pop(job_index)
                machine["status"] = "busy"
                machine["current_job_id"] = job["id"]
                machine["current_job_priority"] = job["priority"]
                machine["remaining_time"] = max(1, int(job["processing_time"]))
                machine["cycles_since_maintenance"] += 1
            else:
                context["invalid_action"] = 1.0
            return context

        if action_type == "maintenance":
            machine = self._machine_by_id(action.get("machine_id"))
            valid_maintenance = machine is not None and machine["status"] in {"idle", "broken", "intermittent_failure"}
            if valid_maintenance:
                machine["status"] = "maintenance"
                machine["current_job_id"] = None
                machine["current_job_priority"] = None
                machine["remaining_time"] = max(int(self.config["maintenance_duration"]), 1)
                machine["energy_usage"] = float(self.config["maintenance_energy"])
            else:
                context["invalid_action"] = 1.0
            return context

        if action_type == "delay":
            job_index = self._job_index_by_id(action.get("job_id"))
            if job_index >= 0:
                delayed_job = self.job_queue.pop(job_index)
                delayed_job["waiting_time"] += 1
                self.job_queue.append(delayed_job)
                context["delayed_jobs"] = 1.0
            else:
                context["invalid_action"] = 1.0
            return context

        return context

    def _advance_time(self, context: Dict[str, float]) -> None:
        self.time_step += 1
        degradation_rate = float(self.config["degradation_rate"])
        intermittent_failure_rate = float(self.config["intermittent_failure_rate"])

        for machine in self.machines:
            status = machine["status"]

            if status == "busy":
                priority = int(machine.get("current_job_priority") or 1)
                intensity = 1.0 + (priority - 1) * 0.2
                if priority == 3:
                    intensity *= float(self.config["high_intensity_multiplier"])
                machine["health"] = max(0.1, float(machine["health"]) - (degradation_rate * intensity))
                machine["energy_usage"] = (
                    float(self.config["busy_energy_base"])
                    + (priority * float(self.config["busy_priority_energy"]))
                    + ((1.0 - float(machine["health"])) * float(self.config["aging_energy_penalty"]))
                )

                failure_probability = self._machine_breakdown_risk(machine) * float(self.config["failure_rate"])
                if self.rng.random() < failure_probability:
                    machine["status"] = "broken"
                    machine["current_job_id"] = None
                    machine["current_job_priority"] = None
                    machine["remaining_time"] = 0
                    machine["energy_usage"] = float(self.config["broken_energy"])
                    self.breakdown_events += 1
                    context["breakdowns"] += 1.0
                    continue

                machine["remaining_time"] -= 1
                if machine["remaining_time"] <= 0:
                    machine["status"] = "idle"
                    machine["current_job_id"] = None
                    machine["current_job_priority"] = None
                    machine["remaining_time"] = 0
                    machine["energy_usage"] = float(self.config["idle_energy"])
                    self.jobs_completed += 1
                    context["jobs_completed"] += 1.0
                continue

            if status == "maintenance":
                machine["energy_usage"] = float(self.config["maintenance_energy"])
                machine["remaining_time"] -= 1
                machine["health"] = min(
                    1.0,
                    float(machine["health"])
                    + (float(self.config["maintenance_recovery"]) / max(int(self.config["maintenance_duration"]), 1)),
                )
                if machine["remaining_time"] <= 0:
                    machine["status"] = "idle"
                    machine["remaining_time"] = 0
                    machine["health"] = min(1.0, float(machine["health"]) + 0.08)
                    machine["energy_usage"] = float(self.config["idle_energy"])
                    machine["cycles_since_maintenance"] = 0
                    context["maintenance_completed"] += 1.0
                continue

            if status == "intermittent_failure":
                machine["energy_usage"] = float(self.config["intermittent_failure_energy"])
                machine["remaining_time"] -= 1
                if machine["remaining_time"] <= 0:
                    machine["status"] = "idle"
                    machine["remaining_time"] = 0
                    machine["energy_usage"] = float(self.config["idle_energy"])
                continue

            if status == "broken":
                machine["energy_usage"] = float(self.config["broken_energy"])
                continue

            machine["status"] = "idle"
            machine["energy_usage"] = float(self.config["idle_energy"])
            machine["health"] = max(0.15, float(machine["health"]) - (degradation_rate * 0.05))
            intermittent_probability = intermittent_failure_rate * (1.0 + max(0.0, 0.55 - float(machine["health"])))
            if self.rng.random() < intermittent_probability:
                machine["status"] = "intermittent_failure"
                machine["remaining_time"] = self.rng.randint(1, 2)
                machine["energy_usage"] = float(self.config["intermittent_failure_energy"])
                self.breakdown_events += 1
                context["breakdowns"] += 0.5

        for job in self.job_queue:
            job["waiting_time"] += 1

    def _refresh_metrics(self, accumulate: bool = True) -> None:
        self.current_energy_usage = sum(float(machine.get("energy_usage", 0.0)) for machine in self.machines)
        if accumulate:
            self.cumulative_energy_usage += self.current_energy_usage

        waiting_times = [int(job.get("waiting_time", 0)) for job in self.job_queue]
        self.queue_length = len(self.job_queue)
        self.avg_waiting_time = sum(waiting_times) / max(len(waiting_times), 1)
        self.max_waiting_time = max(waiting_times) if waiting_times else 0.0
        self.delay = self.avg_waiting_time
        self.throughput = self.jobs_completed / max(self.time_step, 1)

        self.idle_machines = sum(1 for machine in self.machines if machine.get("status") == "idle")
        self.busy_machines = sum(1 for machine in self.machines if machine.get("status") == "busy")
        self.broken_machines = sum(
            1 for machine in self.machines if machine.get("status") in {"broken", "intermittent_failure"}
        )
        self.maintenance_machines = sum(1 for machine in self.machines if machine.get("status") == "maintenance")

        machine_risks = []
        for machine in self.machines:
            machine["breakdown_risk"] = self._machine_breakdown_risk(machine)
            machine_risks.append(machine["breakdown_risk"])

        average_machine_risk = sum(machine_risks) / max(len(machine_risks), 1)
        queue_pressure = self.queue_length / max(int(self.config["initial_jobs"]), 1)
        energy_pressure = max(0.0, self.current_energy_usage - float(self.config["energy_budget"])) / max(
            float(self.config["energy_budget"]),
            1.0,
        )
        self.breakdown_risk = max(
            0.0,
            min(1.0, (average_machine_risk * 0.75) + (queue_pressure * 0.15) + (energy_pressure * 0.10)),
        )

    def step(self, action: Dict[str, Any]):
        if self.done:
            return self.get_state(), 0.0, True, {"jobs_completed": self.jobs_completed}

        normalized_action = self._normalize_action(action)
        reward_context = self._execute_action(normalized_action)
        self._advance_time(reward_context)
        self._refresh_metrics(accumulate=True)

        all_machines_clear = all(
            machine.get("status") in {"idle", "broken", "intermittent_failure"} for machine in self.machines
        )
        self.done = bool(
            self.time_step >= int(self.config["max_steps"])
            or (self.queue_length == 0 and all_machines_clear and self.jobs_completed >= int(self.config["initial_jobs"]))
        )

        step_reward = SmartFactoryReward.calculate(self.get_state(), reward_context)
        self.total_reward += step_reward

        info = {
            "jobs_completed": self.jobs_completed,
            "queue_length": self.queue_length,
            "current_energy_usage": self.current_energy_usage,
            "throughput": self.throughput,
            "delay": self.delay,
            "breakdown_risk": self.breakdown_risk,
            "avg_waiting_time": self.avg_waiting_time,
            "max_waiting_time": self.max_waiting_time,
            "broken_machines": self.broken_machines,
            "maintenance_machines": self.maintenance_machines,
            "breakdown_events": self.breakdown_events,
            "reward_context": reward_context,
        }
        return self.get_state(), step_reward, self.done, info

    def get_state(self):
        return {
            "machines": self.machines,
            "job_queue": self.job_queue,
            "energy_budget": float(self.config["energy_budget"]),
            "current_energy_usage": self.current_energy_usage,
            "cumulative_energy_usage": self.cumulative_energy_usage,
            "throughput": self.throughput,
            "delay": self.delay,
            "queue_length": self.queue_length,
            "breakdown_risk": self.breakdown_risk,
            "avg_waiting_time": self.avg_waiting_time,
            "max_waiting_time": self.max_waiting_time,
            "broken_machines": self.broken_machines,
            "maintenance_machines": self.maintenance_machines,
            "initial_jobs": int(self.config["initial_jobs"]),
            "num_machines": int(self.config["num_machines"]),
            "time_step": self.time_step,
            "total_reward": self.total_reward,
            "jobs_completed": self.jobs_completed,
            "breakdown_events": self.breakdown_events,
            "max_steps": int(self.config["max_steps"]),
            "done": self.done,
        }
