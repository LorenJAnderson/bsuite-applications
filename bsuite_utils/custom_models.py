from bsuite.baselines import experiment
from bsuite.baselines.tf import dqn
from bsuite.utils import gym_wrapper
import gym

class BSuiteDQNShim:
    def __init__(self, policy, env: gym_wrapper.GymFromDMEnv):
        self.env = gym_wrapper.DMEnvFromGym(env)
        self.agent = dqn.default_agent(
            obs_spec=self.env.observation_spec(),
            action_spec=self.env.action_spec(),
        )

    def learn(self, total_timesteps: int, reset_num_timesteps: bool = False):
        # TODO: pretty sure this is wrong
        experiment.run(self.agent, self.env, num_episodes=total_timesteps)


class LifeWrapper:
    def __init__(self, env):
        self.count_resets = 0
        self.base_env = env
        self.obs = None
        self.rew = None
        self.done = None
        self.info = None

    def reset(self):
        if (self.count_resets % 5) == 0:
            return self.hard_reset()
        else:
            return self.soft_reset()

    def soft_reset(self):
        obs = self.base_env.reset()
        return obs, self.rew, self.done, self.info

    def hard_reset(self):
        return self.base_env.reset()

    def step(self, action):
        self.obs, self.rew, self.done, self.info = self.base_env.step(action)
        if self.done:
            self.count_resets += 1
            return self.reset()
        else:
            return self.obs, self.rew, self.done, self.info

    def __getattr__(self, name):
        return getattr(self.base_env, name)


class SkipWrapper:
    def __init__(self, env):
        self.base_env = env

    def reset(self):
        return self.base_env.reset()

    def step(self, action):
        done = False
        rew_total = 0
        for _ in range(4):
            obs, rew, done, info = self.base_env.step(action)
            rew_total += rew
            if done:
                break
        return obs, rew_total, done, info

    def __getattr__(self, name):
        return getattr(self.base_env, name)
