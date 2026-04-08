from pydantic import BaseModel
from typing import List, Optional

class Machine(BaseModel):
    id: str
    status: str
    health: float
    energy_usage: float
    current_job_id: Optional[str] = None
    remaining_time: int

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
    time_step: int
    total_reward: float
    jobs_completed: int
    max_steps: int
    done: bool

class Action(BaseModel):
    type: str
    job_id: Optional[str] = None
    machine_id: Optional[str] = None
