from stable_baselines3 import DQN, A2C, PPO

from bsuite_utils.model_config import ModelConfig
from bsuite_utils.runner import main

model_configs = [
    ModelConfig(name="DQN", cls=DQN, kwargs=dict(learning_starts=1000, learning_rate=7e-4, buffer_size=10_000)),
    ModelConfig(name="A2C", cls=A2C, kwargs=dict()),
    ModelConfig(name="PPO", cls=PPO, kwargs=dict()),
]

if __name__ == "__main__":
    main(model_configs, experiment_id="1_1")
