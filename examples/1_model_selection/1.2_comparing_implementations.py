"""
Example 1.2 - comparing implementations. Similar to 1.1, but comparing implementations of the same algorithm from
different libraries.

For this example we compare DQN implemented in baselines vs DQN implemented in RLLib
Since we already ran DQN for example 1_1 we can skip it here.
"""
from bsuite.baselines import experiment
from bsuite.baselines.tf import dqn
from bsuite.utils import gym_wrapper
from bsuite_utils.model_config import ModelConfig
from bsuite_utils.runner import main

experiment_tag = "1_2"


class BSuiteDQNShim:
    def __init__(self, policy, env: gym_wrapper.GymFromDMEnv):
        self.env = gym_wrapper.DMEnvFromGym(env)
        self.agent = dqn.default_agent(
            obs_spec=self.env.observation_spec(),
            action_spec=self.env.action_spec(),
        )

    def learn(self, total_timesteps: int, reset_num_timesteps: bool = False):
        experiment.run(self.agent, self.env, num_episodes=total_timesteps)


model_configs = [
    ModelConfig(name="BSuite_DQN", cls=BSuiteDQNShim),
]

if __name__ == "__main__":
    main(model_configs, experiment_id=experiment_tag)
