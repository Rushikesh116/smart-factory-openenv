from __future__ import annotations

from typing import Any, Dict, List

from app.reward import calculate_component_scores


def energy_efficiency_grader(state: Any) -> float:
    score = calculate_component_scores(state)["energy_score"]
    final_score = max(0.01, min(0.99, score))
    return final_score


def throughput_grader(state: Any) -> float:
    score = calculate_component_scores(state)["throughput_score"]
    final_score = max(0.01, min(0.99, score))
    return final_score


def delay_grader(state: Any) -> float:
    score = calculate_component_scores(state)["delay_score"]
    final_score = max(0.01, min(0.99, score))
    return final_score


TASK_LIST: List[Dict[str, Any]] = [
    {
        "id": "easy",
        "canonical_id": "energy_efficiency",
        "name": "Energy Efficiency",
        "objective": "energy_efficiency",
        "config": {
            "num_machines": 3,
            "initial_jobs": 12,
            "energy_budget": 28.0,
            "failure_rate": 0.015,
            "intermittent_failure_rate": 0.01,
            "degradation_rate": 0.006,
            "maintenance_duration": 2,
            "maintenance_recovery": 0.24,
            "idle_energy": 0.35,
            "busy_energy_base": 5.6,
            "busy_priority_energy": 1.0,
            "aging_energy_penalty": 2.0,
            "max_steps": 60,
            "seed": 11,
        },
        "grader": energy_efficiency_grader,
    },
    {
        "id": "medium",
        "canonical_id": "throughput",
        "name": "Throughput Optimization",
        "objective": "throughput",
        "config": {
            "num_machines": 5,
            "initial_jobs": 20,
            "energy_budget": 44.0,
            "failure_rate": 0.025,
            "intermittent_failure_rate": 0.02,
            "degradation_rate": 0.008,
            "maintenance_duration": 2,
            "maintenance_recovery": 0.22,
            "idle_energy": 0.45,
            "busy_energy_base": 6.8,
            "busy_priority_energy": 1.1,
            "aging_energy_penalty": 2.4,
            "max_steps": 90,
            "seed": 23,
        },
        "grader": throughput_grader,
    },
    {
        "id": "hard",
        "canonical_id": "low_latency",
        "name": "Low Latency Scheduling",
        "objective": "low_latency",
        "config": {
            "num_machines": 6,
            "initial_jobs": 24,
            "energy_budget": 52.0,
            "failure_rate": 0.035,
            "intermittent_failure_rate": 0.03,
            "degradation_rate": 0.01,
            "maintenance_duration": 3,
            "maintenance_recovery": 0.2,
            "idle_energy": 0.5,
            "busy_energy_base": 7.4,
            "busy_priority_energy": 1.3,
            "aging_energy_penalty": 2.8,
            "high_intensity_multiplier": 1.4,
            "max_steps": 110,
            "seed": 37,
        },
        "grader": delay_grader,
    },
]

TASKS: Dict[str, Dict[str, Any]] = {task["id"]: task for task in TASK_LIST}
TASK_ALIASES: Dict[str, str] = {task["canonical_id"]: task["id"] for task in TASK_LIST}


def get_task(task_id: str) -> Dict[str, Any]:
    resolved_task_id = TASK_ALIASES.get(task_id, task_id)
    return TASKS.get(resolved_task_id, TASK_LIST[0])


def grade_task(task_id: str, state: Any) -> float:
    task = get_task(task_id)
    return task["grader"](state)


def serialize_tasks() -> List[Dict[str, Any]]:
    return [
        {
            "id": task["id"],
            "canonical_id": task["canonical_id"],
            "name": task["name"],
            "objective": task["objective"],
            "config": dict(task["config"]),
            "grader": task["grader"].__name__,
        }
        for task in TASK_LIST
    ]
