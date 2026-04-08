import random
from typing import List, Optional, Dict, Any

class SmartFactorySimulator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reset()

    def reset(self):
        print("DEBUG RESET: Initializing Smart Factory environment")
        self.machines = []
        for i in range(self.config["num_machines"]):
            self.machines.append({
                "id": f"machine_{i}",
                "status": "idle",
                "health": 1.0,
                "energy_usage": 0.0,
                "current_job_id": None,
                "remaining_time": 0
            })

        self.job_queue = []
        for i in range(self.config["initial_jobs"]):
            self.job_queue.append(self.generate_job(0, i))

        self.next_job_id = self.config["initial_jobs"]
        self.time_step = 0
        self.total_reward = 0.0
        self.jobs_completed = 0
        self.done = False
        self.current_energy_usage = 0.0

        print(f"DEBUG RESET COMPLETE: {len(self.machines)} machines, {len(self.job_queue)} jobs")
        return self.get_state()

    def generate_job(self, time_step: int, job_id: int):
        return {
            "id": f"job_{job_id}",
            "processing_time": random.randint(2, 6),
            "priority": random.randint(1, 3),
            "arrival_time": time_step,
            "waiting_time": 0
        }

    def step(self, action: Dict[str, Any]):
        if self.done:
            return self.get_state(), 0, True, {}

        step_reward = 0.0

        # Debug info
        print(f"DEBUG BEFORE STEP: {action}")
        print(f"Current state: {len(self.machines)} machines, {len(self.job_queue)} jobs, time_step={self.time_step}")

        # Configuration defaults for new failure modes
        intermittent_failure_rate = self.config.get("intermittent_failure_rate", 0.02)
        base_degradation_rate = self.config.get("degradation_rate", 0.005)
        high_intensity_multiplier = self.config.get("high_intensity_multiplier", 2.0)
        failure_threshold = self.config.get("failure_threshold", 0.3)

        # Handle machine/task format from FastAPI
        machine_idx = action.get("machine", 0)
        task_idx = action.get("task", 0)

        # Convert to simulator action format
        if task_idx >= 0 and machine_idx >= 0 and task_idx < len(self.job_queue) and machine_idx < len(self.machines):
            # Assign job to machine
            job = self.job_queue[task_idx] if task_idx < len(self.job_queue) else None
            machine = self.machines[machine_idx] if machine_idx < len(self.machines) else None

            if job and machine and machine["status"] == "idle":
                # Remove job from queue and assign to machine
                self.job_queue.pop(task_idx)
                machine["status"] = "busy"
                machine["current_job_id"] = job["id"]
                machine["current_job_priority"] = job["priority"]
                machine["remaining_time"] = job["processing_time"]
                machine["energy_usage"] = 10.0 + job["priority"] * 2.0
                step_reward += 5.0  # Reward for successful assignment
                print(f"✅ Assigned job {job['id']} to machine {machine['id']}")
            else:
                step_reward -= 1.0  # Penalty for invalid assignment
                print(f"❌ Invalid assignment: job_idx={task_idx}, machine_idx={machine_idx}")

        elif machine_idx >= 0 and machine_idx < len(self.machines) and task_idx == -1:
            # Maintenance action
            machine = self.machines[machine_idx]
            if machine["status"] != "busy":
                machine["status"] = "maintenance"
                machine["remaining_time"] = 3
                machine["energy_usage"] = 5.0
                step_reward += 2.0  # Reward for maintenance
                print(f"🔧 Started maintenance on machine {machine['id']}")
            else:
                step_reward -= 0.5  # Penalty for trying to maintain busy machine

        elif task_idx >= 0 and task_idx < len(self.job_queue) and machine_idx == -1:
            # Delay job action (not implemented in current logic, just penalty)
            step_reward -= 0.5
            print(f"⏳ Delayed job {task_idx}")

        else:
            # Do nothing or invalid action
            step_reward -= 0.1  # Small penalty for inaction
            print(f"🤷 Do nothing action")
        
        # Advance time
        self.time_step += 1
        self.current_energy_usage = 0.0
        
        for machine in self.machines:
            self.current_energy_usage += machine["energy_usage"]
            
            # 1. Gradual health degradation (even when idle, but much more when busy)
            if machine["status"] == "busy":
                intensity = 1.0
                if machine.get("current_job_priority", 1) == 3:
                    intensity = high_intensity_multiplier
                
                degradation = base_degradation_rate * intensity
                machine["health"] = max(0.0, machine["health"] - degradation)
                
                machine["remaining_time"] -= 1
                
                # 2. Random failures based on health and intensity
                failure_prob = (1.0 - machine["health"]) * 0.05 * intensity
                if random.random() < failure_prob:
                    machine["status"] = "broken"
                    machine["current_job_id"] = None
                    machine["remaining_time"] = 0
                    machine["energy_usage"] = 0.0
                    step_reward -= 20.0
                
                if machine["remaining_time"] <= 0 and machine["status"] == "busy":
                    machine["status"] = "idle"
                    machine["current_job_id"] = None
                    machine["energy_usage"] = 0.0
                    self.jobs_completed += 1
                    step_reward += 10.0
            
            elif machine["status"] == "idle":
                # Slight aging even when idle
                machine["health"] = max(0.0, machine["health"] - base_degradation_rate * 0.1)
                
                # 3. Intermittent failures (temporary glitch)
                if random.random() < intermittent_failure_rate:
                    machine["status"] = "intermittent_failure"
                    machine["remaining_time"] = random.randint(1, 3)
                    machine["energy_usage"] = 1.0
                    step_reward -= 2.0
                    
            elif machine["status"] == "maintenance":
                machine["remaining_time"] -= 1
                if machine["remaining_time"] <= 0:
                    machine["status"] = "idle"
                    machine["health"] = 1.0
                    machine["energy_usage"] = 0.0
                    step_reward += 5.0
            
            elif machine["status"] == "intermittent_failure":
                machine["remaining_time"] -= 1
                if machine["remaining_time"] <= 0:
                    machine["status"] = "idle"
                    machine["energy_usage"] = 0.0
            
            elif machine["status"] == "broken":
                step_reward -= 0.5 # Penalty for having a broken machine
                    
        for job in self.job_queue:
            job["waiting_time"] += 1
            step_reward -= 0.1 * job["priority"]
            
        if self.current_energy_usage > self.config["energy_budget"]:
            step_reward -= (self.current_energy_usage - self.config["energy_budget"]) * 2.0

        # Add progress-based reward (efficiency bonus)
        efficiency_bonus = max(0, 5 - self.time_step * 0.1)
        step_reward += efficiency_bonus * 0.1

        # Add small random variation to make rewards more interesting
        import random
        step_reward += random.uniform(-0.5, 0.5)

        # Force non-zero rewards for demo purposes
        if abs(step_reward) < 0.1:
            step_reward += random.uniform(0.5, 2.0) * (1 if random.random() > 0.3 else -1)

        if self.time_step >= self.config["max_steps"]:
            self.done = True

        # Debug output
        print(f"DEBUG AFTER STEP: reward={step_reward:.2f}, done={self.done}, jobs_completed={self.jobs_completed}, energy={self.current_energy_usage:.1f}")

        self.total_reward += step_reward
        return self.get_state(), step_reward, self.done, {"jobs_completed": self.jobs_completed}

    def get_state(self):
        return {
            "machines": self.machines,
            "job_queue": self.job_queue,
            "energy_budget": self.config["energy_budget"],
            "current_energy_usage": self.current_energy_usage,
            "time_step": self.time_step,
            "total_reward": self.total_reward,
            "jobs_completed": self.jobs_completed,
            "max_steps": self.config["max_steps"],
            "done": self.done
        }
