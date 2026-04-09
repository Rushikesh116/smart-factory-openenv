from __future__ import annotations

from typing import Any, Dict, Mapping

OBJECTIVE_WEIGHTS: Dict[str, float] = {
    "energy": 0.4,
    "throughput": 0.4,
    "delay": 0.2,
}
BREAKDOWN_RISK_PENALTY = 0.15


def state_to_dict(state: Any) -> Dict[str, Any]:
    if state is None:
        return {}
    if isinstance(state, dict):
        return state
    if hasattr(state, "model_dump"):
        return state.model_dump()
    if hasattr(state, "dict"):
        return state.dict()
    return {}


def clamp_score(score: float) -> float:
    return max(0.01, min(0.99, float(score)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_operational_metrics(state: Any) -> Dict[str, Any]:
    state_data = state_to_dict(state)
    machines = list(state_data.get("machines", []))
    job_queue = list(state_data.get("job_queue", []))
    waiting_times = [_safe_float(job.get("waiting_time", 0.0)) for job in job_queue]
    num_machines = max(_safe_int(state_data.get("num_machines", len(machines))), len(machines), 1)
    queue_length = max(_safe_int(state_data.get("queue_length", len(job_queue))), 0)
    time_step = max(_safe_int(state_data.get("time_step", 0)), 0)
    jobs_completed = max(_safe_int(state_data.get("jobs_completed", 0)), 0)
    initial_jobs = max(
        _safe_int(state_data.get("initial_jobs", jobs_completed + queue_length)),
        jobs_completed + queue_length,
        1,
    )
    current_energy_usage = max(_safe_float(state_data.get("current_energy_usage", 0.0)), 0.0)
    energy_budget = max(_safe_float(state_data.get("energy_budget", 1.0)), 1.0)
    throughput = max(
        _safe_float(state_data.get("throughput", jobs_completed / max(time_step, 1))),
        0.0,
    )
    avg_waiting_time = max(
        _safe_float(
            state_data.get(
                "avg_waiting_time",
                state_data.get("delay", sum(waiting_times) / max(len(waiting_times), 1)),
            ),
        ),
        0.0,
    )
    delay = max(_safe_float(state_data.get("delay", avg_waiting_time)), 0.0)
    max_waiting_time = max(
        _safe_float(state_data.get("max_waiting_time", max(waiting_times) if waiting_times else delay)),
        0.0,
    )
    broken_machines = max(
        _safe_int(
            state_data.get(
                "broken_machines",
                sum(1 for machine in machines if machine.get("status") in {"broken", "intermittent_failure"}),
            )
        ),
        0,
    )
    maintenance_machines = max(
        _safe_int(
            state_data.get(
                "maintenance_machines",
                sum(1 for machine in machines if machine.get("status") == "maintenance"),
            )
        ),
        0,
    )
    breakdown_risk = _safe_float(
        state_data.get("breakdown_risk", broken_machines / max(num_machines, 1)),
        broken_machines / max(num_machines, 1),
    )
    return {
        "machines": machines,
        "job_queue": job_queue,
        "num_machines": num_machines,
        "queue_length": queue_length,
        "time_step": time_step,
        "jobs_completed": jobs_completed,
        "initial_jobs": initial_jobs,
        "current_energy_usage": current_energy_usage,
        "energy_budget": energy_budget,
        "throughput": throughput,
        "delay": delay,
        "avg_waiting_time": avg_waiting_time,
        "max_waiting_time": max_waiting_time,
        "breakdown_risk": max(0.0, min(1.0, breakdown_risk)),
        "broken_machines": broken_machines,
        "maintenance_machines": maintenance_machines,
        "max_steps": max(_safe_int(state_data.get("max_steps", 1)), 1),
        "completion_rate": min(jobs_completed / max(initial_jobs, 1), 1.0),
        "cumulative_energy_usage": max(_safe_float(state_data.get("cumulative_energy_usage", current_energy_usage)), 0.0),
    }


def calculate_component_scores(state: Any) -> Dict[str, float]:
    metrics = extract_operational_metrics(state)
    energy_bound = max(metrics["energy_budget"] * 1.15, metrics["num_machines"] * 6.0, 1.0)
    throughput_bound = max(metrics["num_machines"] * 0.35, 0.75)
    delay_bound = max(metrics["max_steps"] * 0.3, metrics["num_machines"] * 2.5, 5.0)

    energy_score = 1.0 - (metrics["current_energy_usage"] / energy_bound)
    throughput_score = metrics["throughput"] / throughput_bound
    delay_score = 1.0 - (metrics["delay"] / delay_bound)
    broken_machine_ratio = metrics["broken_machines"] / max(metrics["num_machines"], 1)
    breakdown_penalty = min(
        0.45,
        (metrics["breakdown_risk"] * BREAKDOWN_RISK_PENALTY) + (broken_machine_ratio * 0.1),
    )

    return {
        "energy_score": clamp_score(energy_score),
        "throughput_score": clamp_score(throughput_score),
        "delay_score": clamp_score(delay_score),
        "breakdown_penalty": max(0.0, breakdown_penalty),
    }


def calculate_weighted_grade(state: Any) -> float:
    components = calculate_component_scores(state)
    score = (
        (components["energy_score"] * OBJECTIVE_WEIGHTS["energy"])
        + (components["throughput_score"] * OBJECTIVE_WEIGHTS["throughput"])
        + (components["delay_score"] * OBJECTIVE_WEIGHTS["delay"])
        - components["breakdown_penalty"]
    )
    final_score = max(0.01, min(0.99, score))
    return final_score


class SmartFactoryReward:
    @staticmethod
    def calculate(state: Dict[str, Any], step_reward: Any = None) -> float:
        components = calculate_component_scores(state)
        reward = (
            (components["energy_score"] * OBJECTIVE_WEIGHTS["energy"])
            + (components["throughput_score"] * OBJECTIVE_WEIGHTS["throughput"])
            + (components["delay_score"] * OBJECTIVE_WEIGHTS["delay"])
            - components["breakdown_penalty"]
        )

        if isinstance(step_reward, Mapping):
            reward += 0.08 * _safe_float(step_reward.get("jobs_completed", 0.0))
            reward += 0.03 * _safe_float(step_reward.get("maintenance_completed", 0.0))
            reward -= 0.06 * _safe_float(step_reward.get("invalid_action", 0.0))
            reward -= 0.04 * _safe_float(step_reward.get("breakdowns", 0.0))
            reward -= 0.02 * _safe_float(step_reward.get("delayed_jobs", 0.0))
        elif isinstance(step_reward, (int, float)):
            reward += 0.1 * float(step_reward)

        return max(-1.0, min(1.0, float(reward)))

    @staticmethod
    def grade(state: Dict[str, Any]) -> float:
        return calculate_weighted_grade(state)
