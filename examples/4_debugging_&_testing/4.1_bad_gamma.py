from stable_baselines3 import DQN
from bsuite_utils.model_config import ModelConfig
from bsuite_utils.runner import main

experiment_tag = "4_1"
# Default gamma is 0.99, already done as part of 1_1
gammas = [1.0]
model_configs = [
    ModelConfig(name=f"DQN_gamma{x}", cls=DQN,
                kwargs=dict(learning_starts=1000, learning_rate=7e-4, buffer_size=10_000, gamma=x)) for x
    in gammas
]

if __name__ == "__main__":
    main(model_configs, experiment_id=experiment_tag)
