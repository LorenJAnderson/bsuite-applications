from stable_baselines3 import PPO
from bsuite_utils.model_config import ModelConfig
from bsuite_utils.runner import main

experiment_tag = "3_2"
# Default is 3e-4. Already done as part of 1_1
learning_rates = [3e-3, 1e-4, 9e-4, 3e-5]
model_configs = [
    ModelConfig(name=f"PPO_lr{x}", cls=PPO, kwargs=dict(learning_rates=x)) for x in learning_rates
]

if __name__ == "__main__":
    main(model_configs, experiment_id=experiment_tag)
