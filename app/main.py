from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any
import random

app = FastAPI(title="Smart Factory RL Environment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
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

# In-memory environment state
env_state = None
current_config = None

TASKS = [
    {
        "id": "easy",
        "name": "Stable System",
        "config": {"num_machines": 2, "initial_jobs": 10, "energy_budget": 50.0, "failure_rate": 0.0, "max_steps": 50}
    },
    {
        "id": "medium",
        "name": "Energy Constraints",
        "config": {"num_machines": 4, "initial_jobs": 20, "energy_budget": 80.0, "failure_rate": 0.05, "max_steps": 100}
    },
    {
        "id": "hard",
        "name": "Dynamic Factory",
        "config": {"num_machines": 8, "initial_jobs": 30, "energy_budget": 150.0, "failure_rate": 0.15, "max_steps": 200}
    }
]

@app.post("/reset")
async def reset(request: Request):
    global env_state, current_config
    data = await request.json()
    task_id = data.get("task_id", "easy")
    task = next((t for t in TASKS if t["id"] == task_id), TASKS[0])
    current_config = task["config"]
    
    machines = []
    for i in range(current_config["num_machines"]):
        machines.append(Machine(
            id=f"machine_{i}",
            status="idle",
            health=1.0,
            energy_usage=0.0,
            remaining_time=0
        ))
    
    job_queue = []
    for i in range(current_config["initial_jobs"]):
        job_queue.append(Job(
            id=f"job_{i}",
            processing_time=random.randint(2, 6),
            priority=random.randint(1, 3),
            arrival_time=0,
            waiting_time=0
        ))
    
    env_state = State(
        machines=machines,
        job_queue=job_queue,
        energy_budget=current_config["energy_budget"],
        current_energy_usage=0.0,
        time_step=0,
        total_reward=0.0,
        jobs_completed=0,
        max_steps=current_config["max_steps"],
        done=False
    )
    return env_state

@app.post("/step")
async def step(request: Request):
    global env_state
    if not env_state:
        return {"error": "Environment not initialized"}
    
    data = await request.json()
    action = data.get("action", {})
    step_reward = 0.0
    
    # Process action
    if action.get("type") == "assign":
        job_id = action.get("job_id")
        machine_id = action.get("machine_id")
        machine = next((m for m in env_state.machines if m.id == machine_id), None)
        job_idx = next((i for i, j in enumerate(env_state.job_queue) if j.id == job_id), -1)
        
        if machine and machine.status == "idle" and job_idx != -1:
            job = env_state.job_queue.pop(job_idx)
            machine.status = "busy"
            machine.current_job_id = job.id
            machine.remaining_time = job.processing_time
            machine.energy_usage = 10.0 + job.priority * 2.0
            step_reward += 1.0
        else:
            step_reward -= 2.0
            
    # Advance time
    env_state.time_step += 1
    env_state.current_energy_usage = 0.0
    
    for machine in env_state.machines:
        env_state.current_energy_usage += machine.energy_usage
        if machine.status == "busy":
            machine.remaining_time -= 1
            machine.health -= 0.01
            if machine.remaining_time <= 0:
                machine.status = "idle"
                machine.current_job_id = None
                machine.energy_usage = 0.0
                env_state.jobs_completed += 1
                step_reward += 10.0
        elif machine.status == "maintenance":
            machine.remaining_time -= 1
            if machine.remaining_time <= 0:
                machine.status = "idle"
                machine.health = 1.0
                machine.energy_usage = 0.0
                step_reward += 5.0
                
    for job in env_state.job_queue:
        job.waiting_time += 1
        step_reward -= 0.1 * job.priority
        
    if env_state.current_energy_usage > env_state.energy_budget:
        step_reward -= (env_state.current_energy_usage - env_state.energy_budget) * 2.0
        
    if env_state.time_step >= env_state.max_steps:
        env_state.done = True
        
    env_state.total_reward += step_reward
    return {"state": env_state, "reward": step_reward, "done": env_state.done}

@app.get("/state")
async def get_state():
    return env_state

@app.get("/tasks")
async def get_tasks():
    return TASKS

@app.get("/grader")
async def get_grader():
    if not env_state:
        return {"score": 0.0}
    completion_rate = env_state.jobs_completed / (env_state.jobs_completed + len(env_state.job_queue) or 1)
    avg_health = sum(m.health for m in env_state.machines) / len(env_state.machines)
    score = (completion_rate * 0.4) + (avg_health * 0.3) + (0.3 if env_state.current_energy_usage <= env_state.energy_budget else 0.1)
    return {"score": min(1.0, max(0.0, score))}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
