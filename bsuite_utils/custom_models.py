from bsuite.baselines import experiment
from bsuite.baselines.tf import dqn
from bsuite.utils import gym_wrapper

class BSuiteDQNShim:
    def __init__(self, policy, env: gym_wrapper.GymFromDMEnv):
        self.env = gym_wrapper.DMEnvFromGym(env)
        self.agent = dqn.default_agent(
            obs_spec=self.env.observation_spec(),
            action_spec=self.env.action_spec(),
        )

    def learn(self, total_timesteps: int, reset_num_timesteps: bool = False):
        # TODO: this is wrong
        experiment.run(self.agent, self.env, num_episodes=total_timesteps)
