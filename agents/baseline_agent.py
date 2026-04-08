import requests
import os

class BaselineAgent:
    def __init__(self, env_url="http://localhost:3000"):
        self.env_url = env_url

    def get_action(self, state):
        """
        Simple rule-based logic to decide the next action.
        """
        # 1. Check for broken machines
        for machine in state["machines"]:
            if machine["status"] == "broken":
                return {"type": "maintenance", "machine_id": machine["id"]}
        
        # 2. Check for low health machines
        for machine in state["machines"]:
            if machine["health"] < 0.5 and machine["status"] == "idle":
                return {"type": "maintenance", "machine_id": machine["id"]}
        
        # 3. Assign jobs to idle machines
        idle_machines = [m for m in state["machines"] if m["status"] == "idle"]
        if idle_machines and state["job_queue"]:
            # Priority-aware: pick highest priority job
            best_job = sorted(state["job_queue"], key=lambda j: j["priority"], reverse=True)[0]
            return {
                "type": "assign", 
                "job_id": best_job["id"], 
                "machine_id": idle_machines[0]["id"]
            }
        
        return {"type": "do_nothing"}

    def run_episode(self, task_id="easy"):
        res = requests.post(f"{self.env_url}/reset", json={"task_id": task_id})
        state = res.json()
        
        while not state.get("done", False):
            action = self.get_action(state)
            res = requests.post(f"{self.env_url}/step", json={"action": action})
            state = res.json()["state"]
        
        res = requests.get(f"{self.env_url}/grader")
        return res.json()["score"]

if __name__ == "__main__":
    agent = BaselineAgent()
    score = agent.run_episode("easy")
    print(f"Baseline Agent Score (Easy): {score}")
