from stable_baselines3 import PPO
from bsuite_utils.model_config import ModelConfig
from bsuite_utils.runner import main

experiment_tag = "3_1"
entropy_bonuses = [
    # 0.0 # default, covered by exp 1_1
    0.001, 0.01, 0.1
]
model_configs = [
    ModelConfig(name=f"PPO_ent{x}", cls=PPO, kwargs=dict(ent_coef=x)) for x in entropy_bonuses
]

if __name__ == "__main__":
    main(model_configs, experiment_id=experiment_tag)
