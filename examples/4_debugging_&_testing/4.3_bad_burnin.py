from stable_baselines3 import DQN
from bsuite_utils.model_config import ModelConfig
from bsuite_utils.runner import main

experiment_tag = "4_3"
# Good burn-in of 1_000 covered by 1_1
burn_in_steps = [10_000, 100_000]
model_configs = [
    ModelConfig(name=f"DQN_burn{x}", cls=DQN,
                kwargs=dict(learning_starts=x, learning_rate=7e-4, buffer_size=10_000)) for x
    in burn_in_steps
]

if __name__ == "__main__":
    main(model_configs, experiment_id=experiment_tag)
