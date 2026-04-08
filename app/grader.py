from typing import Dict, Any


def grade_episode(total_jobs_completed, total_jobs):
    if total_jobs == 0:
        return 0.0
    score = total_jobs_completed / total_jobs
    return max(0.0, min(1.0, float(score)))


class SmartFactoryGrader:
    @staticmethod
    def grade(state: Dict[str, Any]) -> float:
        if not state or state.get("time_step", 0) == 0:
            return 0.0
        
        # 1. Throughput Score (0.0 to 1.0)
        jobs_completed = state.get("jobs_completed", 0)
        job_queue_len = len(state.get("job_queue", []))
        total_jobs = jobs_completed + job_queue_len
        throughput_score = jobs_completed / total_jobs if total_jobs > 0 else 1.0
        
        # 2. Machine Health Score (0.0 to 1.0)
        machines = state.get("machines", [])
        avg_health = sum(m["health"] for m in machines) / len(machines) if machines else 1.0
        
        # 3. Energy Efficiency Score (0.0 to 1.0)
        energy_budget = state.get("energy_budget", 100.0)
        current_energy = state.get("current_energy_usage", 0.0)
        energy_score = 1.0
        if current_energy > energy_budget:
            overage_ratio = (current_energy - energy_budget) / energy_budget
            energy_score = max(0.0, 1.0 - overage_ratio)
            
        # 4. Uptime Score (0.0 to 1.0)
        broken_count = sum(1 for m in machines if m["status"] == "broken")
        uptime_score = 1.0 - (broken_count / len(machines)) if machines else 1.0
        
        # Weighted final score
        final_score = (throughput_score * 0.4) + (avg_health * 0.2) + (energy_score * 0.2) + (uptime_score * 0.2)
        return min(1.0, max(0.0, final_score))

    @staticmethod
    def get_detailed_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
        if not state:
            return {}
        
        jobs_completed = state.get("jobs_completed", 0)
        time_step = state.get("time_step", 1)
        machines = state.get("machines", [])
        energy_budget = state.get("energy_budget", 100.0)
        current_energy = state.get("current_energy_usage", 0.0)
        
        return {
            "throughput": jobs_completed / (time_step or 1),
            "completion_rate": jobs_completed / (jobs_completed + len(state.get("job_queue", [])) or 1),
            "avg_machine_health": sum(m["health"] for m in machines) / (len(machines) or 1),
            "energy_efficiency": 1.0 if current_energy <= energy_budget else 0.0,
            "broken_machines": sum(1 for m in machines if m["status"] == "broken"),
            "total_reward": state.get("total_reward", 0.0)
        }
