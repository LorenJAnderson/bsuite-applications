from stable_baselines3 import PPO
from sb3_contrib.ppo_recurrent import RecurrentPPO

from bsuite_utils.model_config import ModelConfig
from bsuite_utils.runner import main

experiment_tag = "5_1"
model_configs = [
    # Note: already run as part of 1.1
    # ModelConfig(name="PPO", cls=PPO),
    ModelConfig(name="PPO_RNN", cls=RecurrentPPO, policy="MlpLstmPolicy"),
]

if __name__ == "__main__":
    main(model_configs, experiment_id=experiment_tag)
