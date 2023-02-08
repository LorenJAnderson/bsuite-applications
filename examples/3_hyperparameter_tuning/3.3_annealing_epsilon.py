from stable_baselines3 import DQN
from bsuite_utils.model_config import ModelConfig
from bsuite_utils.runner import main

experiment_tag = "3_3"
# Fraction (over entire training period) in which exploration is annealed from initial_eps<=1 to final_eps<=1
# default is 0.1, covered by 1_1
exploration_fraction = [0.2, 0.4, 0.6]
model_configs = [
    ModelConfig(name=f"DQN_eps{x}", cls=DQN,
                kwargs=dict(learning_starts=1000, learning_rate=7e-4, buffer_size=10_000, exploration_fraction=x)) for x
    in exploration_fraction
]

if __name__ == "__main__":
    main(model_configs, experiment_id=experiment_tag)
