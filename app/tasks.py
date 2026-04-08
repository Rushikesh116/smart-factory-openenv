TASKS = {
    "easy": {
        "id": "easy",
        "name": "Stable System",
        "config": {"num_machines": 2, "initial_jobs": 10, "energy_budget": 50.0, "failure_rate": 0.0, "max_steps": 50}
    },
    "medium": {
        "id": "medium",
        "name": "Energy Constraints",
        "config": {"num_machines": 4, "initial_jobs": 20, "energy_budget": 80.0, "failure_rate": 0.05, "max_steps": 100}
    },
    "hard": {
        "id": "hard",
        "name": "Dynamic Factory",
        "config": {
            "num_machines": 8,
            "initial_jobs": 30,
            "energy_budget": 150.0,
            "failure_rate": 0.15,
            "max_steps": 200,
            "intermittent_failure_rate": 0.05,
            "degradation_rate": 0.01,
            "high_intensity_multiplier": 3.0
        }
    },
}

# Backward compatibility for existing code that expects a list.
TASK_LIST = [
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
        "config": {
            "num_machines": 8, 
            "initial_jobs": 30, 
            "energy_budget": 150.0, 
            "failure_rate": 0.15, 
            "max_steps": 200,
            "intermittent_failure_rate": 0.05,
            "degradation_rate": 0.01,
            "high_intensity_multiplier": 3.0
        }
    }
]
