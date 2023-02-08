from stable_baselines3 import DQN
from bsuite_utils.model_config import ModelConfig
from bsuite_utils.runner import main

experiment_tag = "1_3"
model_configs = [
    ModelConfig(name="DQN", cls=DQN, kwargs=dict(learning_starts=1000, learning_rate=7e-4, buffer_size=1_000)),
    # Done as part of 1.1 -- buffer_size=10_000
    ModelConfig(name="DQN", cls=DQN, kwargs=dict(learning_starts=1000, learning_rate=7e-4, buffer_size=100_000)),
]

if __name__ == "__main__":
    main(model_configs, experiment_id=experiment_tag)
