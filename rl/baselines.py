from __future__ import annotations

from typing import Any, Dict


class HeuristicBaselinePolicy:
    """Simple local heuristic for offline benchmarking."""

    def act(self, env: Any, _obs: Any = None) -> int:
        state: Dict[str, Any] = env.sim.get_state()
        machines = state.get("machines", [])
        queue = state.get("job_queue", [])
        energy_budget = float(max(state.get("energy_budget", 1.0), 1.0))
        current_energy = float(state.get("current_energy_usage", 0.0))

        for machine_index, machine in enumerate(machines[:8]):
            if machine.get("status") in {"broken", "intermittent_failure"}:
                return 1 + machine_index

        for machine_index, machine in enumerate(machines[:8]):
            if machine.get("status") == "idle" and machine.get("health", 1.0) < 0.55:
                return 1 + machine_index

        idle_candidates = [
            (machine_index, machine)
            for machine_index, machine in enumerate(machines[:8])
            if machine.get("status") == "idle" and machine.get("health", 0.0) > 0.35
        ]
        if idle_candidates and queue:
            ranked_jobs = sorted(
                enumerate(queue[:10]),
                key=lambda item: (
                    -item[1].get("priority", 1),
                    -item[1].get("waiting_time", 0),
                    -item[1].get("processing_time", 1),
                ),
            )
            for job_index, job in ranked_jobs:
                estimated_energy = current_energy + 6.0 + (job.get("priority", 1) * 1.2)
                if estimated_energy <= energy_budget * 1.1:
                    machine_index, _ = idle_candidates[0]
                    return 19 + (job_index * 8) + machine_index

        if queue and current_energy > energy_budget * 0.9:
            return 9

        return 0
