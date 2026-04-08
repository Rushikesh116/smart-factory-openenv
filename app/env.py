from .simulator import SmartFactorySimulator
from typing import Dict, Any

class SmartFactoryEnv:
    def __init__(self, config: Dict[str, Any]):
        self.simulator = SmartFactorySimulator(config)

    def reset(self):
        self.simulator.reset()
        return self.simulator.get_state()

    def step(self, action: Dict[str, Any]):
        return self.simulator.step(action)

    def state(self):
        return self.simulator.get_state()
