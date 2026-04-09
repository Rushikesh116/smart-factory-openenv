from __future__ import annotations

from typing import Any, Dict

from app.reward import calculate_component_scores, calculate_weighted_grade, extract_operational_metrics


def grade_episode(total_jobs_completed: int, total_jobs: int) -> float:
    score = total_jobs_completed / max(total_jobs, 1)
    final_score = max(0.01, min(0.99, score))
    return final_score


class SmartFactoryGrader:
    @staticmethod
    def grade(state: Dict[str, Any]) -> float:
        return calculate_weighted_grade(state)

    @staticmethod
    def get_detailed_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
        if not state:
            return {}

        metrics = extract_operational_metrics(state)
        components = calculate_component_scores(state)
        machines = metrics["machines"]

        return {
            "energy": metrics["current_energy_usage"],
            "throughput": metrics["throughput"],
            "delay": metrics["delay"],
            "queue_length": metrics["queue_length"],
            "breakdown_risk": metrics["breakdown_risk"],
            "completion_rate": metrics["completion_rate"],
            "avg_machine_health": sum(machine.get("health", 1.0) for machine in machines) / max(len(machines), 1),
            "energy_score": components["energy_score"],
            "throughput_score": components["throughput_score"],
            "delay_score": components["delay_score"],
            "breakdown_penalty": components["breakdown_penalty"],
            "broken_machines": metrics["broken_machines"],
            "maintenance_machines": metrics["maintenance_machines"],
            "total_reward": state.get("total_reward", 0.0),
        }
