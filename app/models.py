from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Machine(BaseModel):
    id: str
    status: str
    health: float
    energy_usage: float
    current_job_id: Optional[str] = None
    current_job_priority: Optional[int] = None
    remaining_time: int
    breakdown_risk: float = 0.0


class Job(BaseModel):
    id: str
    processing_time: int
    priority: int
    arrival_time: int
    waiting_time: int


class State(BaseModel):
    machines: List[Machine]
    job_queue: List[Job]
    energy_budget: float
    current_energy_usage: float
    cumulative_energy_usage: float = 0.0
    throughput: float = 0.0
    delay: float = 0.0
    queue_length: int = 0
    breakdown_risk: float = 0.0
    avg_waiting_time: float = 0.0
    max_waiting_time: float = 0.0
    broken_machines: int = 0
    maintenance_machines: int = 0
    initial_jobs: int = 0
    num_machines: int = 0
    time_step: int
    total_reward: float
    jobs_completed: int
    breakdown_events: int = 0
    max_steps: int
    done: bool


class Action(BaseModel):
    type: str = Field(default="do_nothing")
    job_id: Optional[str] = None
    machine_id: Optional[str] = None
    machine: Optional[int] = None
    task: Optional[int] = None
