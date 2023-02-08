from stable_baselines3 import DQN
from bsuite_utils.model_config import ModelConfig
from bsuite_utils.runner import main

experiment_tag = "3_2"
# Default is 3e-4. Already done as part of 1_1
learning_rates = [10**(-x) for x in range(8)]
model_configs = [
    ModelConfig(name=f"DQN_lr{x}", cls=DQN, kwargs=dict(learning_starts=1000, learning_rate=x, buffer_size=10_000)) for x in learning_rates
]

if __name__ == "__main__":
    main(model_configs, experiment_id=experiment_tag)
