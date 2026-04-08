from typing import Dict, Any

class SmartFactoryReward:
    @staticmethod
    def calculate(state: Dict[str, Any], step_reward: float) -> float:
        # The simulator already calculates the step reward, 
        # but we can add additional logic here if needed.
        return step_reward
