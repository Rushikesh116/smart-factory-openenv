from __future__ import annotations

from typing import Any, Dict, List, Tuple


class HeuristicBaselinePolicy:
    """Task-aware heuristic policy for smart factory optimization.

    Adapts strategy based on the task objective:
    - energy_efficiency (easy): Conservative energy usage, avoid high-priority jobs when budget is tight
    - throughput (medium): Maximize concurrent machine usage, prefer fast jobs
    - low_latency (hard): Minimize waiting time, assign longest-waiting jobs first
    """

    def __init__(self, task_id: str = "easy"):
        self.task_id = task_id

    def act(self, env: Any, _obs: Any = None) -> int:
        state: Dict[str, Any] = env.sim.get_state()
        machines = state.get("machines", [])
        queue = state.get("job_queue", [])
        energy_budget = float(max(state.get("energy_budget", 1.0), 1.0))
        current_energy = float(state.get("current_energy_usage", 0.0))
        energy_ratio = current_energy / energy_budget

        # 1. Always fix broken machines first (all tasks benefit)
        for idx, machine in enumerate(machines[:8]):
            if machine.get("status") in {"broken", "intermittent_failure"}:
                return 1 + idx

        # 2. Proactive maintenance — task-specific thresholds
        maint_health_threshold = self._maintenance_threshold()
        for idx, machine in enumerate(machines[:8]):
            if (
                machine.get("status") == "idle"
                and machine.get("health", 1.0) < maint_health_threshold
            ):
                return 1 + idx

        # 3. Also maintain if cycles_since_maintenance is high (prevents future breakdowns)
        for idx, machine in enumerate(machines[:8]):
            if (
                machine.get("status") == "idle"
                and machine.get("cycles_since_maintenance", 0) >= 6
                and machine.get("health", 1.0) < 0.75
            ):
                return 1 + idx

        # 4. Assign jobs to idle machines
        idle_candidates = self._get_idle_candidates(machines)
        if idle_candidates and queue:
            ranked_jobs = self._rank_jobs(queue, energy_ratio)

            for job_rank, (job_idx, job) in enumerate(ranked_jobs):
                # Energy check: estimate if we can afford this job
                estimated_extra_energy = 6.0 + (job.get("priority", 1) * 1.2)
                if self.task_id == "easy":
                    # Energy task: strict budget adherence
                    if current_energy + estimated_extra_energy > energy_budget * 0.95:
                        continue
                else:
                    # Other tasks: more lenient
                    if current_energy + estimated_extra_energy > energy_budget * 1.15:
                        continue

                # Pick the best machine for this job
                best_machine_idx = self._pick_best_machine(idle_candidates, job)
                if best_machine_idx is not None:
                    return 19 + (job_idx * 8) + best_machine_idx

        # 5. Delay if energy is critically high and we have jobs
        if queue and energy_ratio > 0.85:
            return 9  # delay first job

        # 6. Do nothing
        return 0

    def _maintenance_threshold(self) -> float:
        """Health threshold below which we trigger proactive maintenance."""
        if self.task_id == "easy":
            return 0.60  # Energy task: maintain often to avoid breakdowns (broken = waste energy)
        elif self.task_id == "medium":
            return 0.50  # Throughput: maintain less aggressively to keep machines busy
        else:
            return 0.55  # Delay: balanced approach

    def _get_idle_candidates(self, machines: List[Dict]) -> List[Tuple[int, Dict]]:
        """Get idle machines with sufficient health, sorted by health (best first)."""
        candidates = [
            (idx, m)
            for idx, m in enumerate(machines[:8])
            if m.get("status") == "idle" and m.get("health", 0.0) > 0.35
        ]
        # Sort by health descending — prefer healthier machines
        candidates.sort(key=lambda x: -x[1].get("health", 0.0))
        return candidates

    def _rank_jobs(self, queue: List[Dict], energy_ratio: float) -> List[Tuple[int, Dict]]:
        """Rank jobs based on task objective."""
        indexed = list(enumerate(queue[:10]))

        if self.task_id == "easy":
            # Energy efficiency: prefer LOW priority jobs (less energy), then shortest processing
            indexed.sort(key=lambda item: (
                item[1].get("priority", 1),        # low priority first (less energy)
                item[1].get("processing_time", 1),  # short jobs first
                -item[1].get("waiting_time", 0),    # then longest waiting
            ))
        elif self.task_id == "medium":
            # Throughput: prefer SHORT processing time (more completions), then high priority
            indexed.sort(key=lambda item: (
                item[1].get("processing_time", 1),  # shortest first
                -item[1].get("priority", 1),        # then high priority
                -item[1].get("waiting_time", 0),
            ))
        else:
            # Low latency: prefer LONGEST WAITING jobs (reduce delay), then high priority
            indexed.sort(key=lambda item: (
                -item[1].get("waiting_time", 0),    # longest waiting first
                -item[1].get("priority", 1),        # then high priority
                item[1].get("processing_time", 1),  # then shortest
            ))

        return indexed

    def _pick_best_machine(
        self, idle_candidates: List[Tuple[int, Dict]], job: Dict
    ) -> int | None:
        """Pick the best machine for a given job based on task objective."""
        if not idle_candidates:
            return None

        if self.task_id == "easy":
            # Energy task: pick the healthiest machine (less aging energy penalty)
            return idle_candidates[0][0]  # already sorted by health desc
        elif self.task_id == "medium":
            # Throughput: pick machine with fewest cycles (less breakdown risk)
            best = min(
                idle_candidates,
                key=lambda x: x[1].get("cycles_since_maintenance", 0),
            )
            return best[0]
        else:
            # Low latency: just pick the first available (speed matters)
            return idle_candidates[0][0]
