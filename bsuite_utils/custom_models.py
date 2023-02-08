from typing import Tuple, Dict, Any

import numpy as np
from bsuite.baselines.tf import dqn
from bsuite.utils import gym_wrapper

# Type aliases
_Observation = np.ndarray
_GymTimestep = Tuple[_Observation, float, bool, Dict[str, Any]]


class BSuiteDQNShim:
    def __init__(self, policy, env: gym_wrapper.GymFromDMEnv):
        self.env = gym_wrapper.DMEnvFromGym(env)
        self.agent = dqn.default_agent(
            obs_spec=self.env.observation_spec(),
            action_spec=self.env.action_spec(),
        )

    def learn(self, total_timesteps: int, reset_num_timesteps: bool = False):
        steps = 0
        while steps < total_timesteps:
            # Start a new episode.
            timestep = self.env.reset()
            while not timestep.last():
                # Generate an action from the agent's policy.
                action = self.agent.select_action(timestep)
                # Step the environment.
                new_timestep = self.env.step(action)
                # Tell the agent about what just happened.
                self.agent.update(timestep, action, new_timestep)

                timestep = new_timestep
                steps += 1


class LifeWrapper:
    def __init__(self, env: gym_wrapper.GymFromDMEnv, n_lives: int = 5):
        self.count_resets = 0
        self.n_lives = n_lives
        self.base_env = env

    def reset(self) -> _Observation:  # return an observation
        self.count_resets = 0
        return self.base_env.reset()

    def step(self, action) -> _GymTimestep:
        obs, rew, done, info = self.base_env.step(action)
        if done:
            self.count_resets += 1
        done = self.count_resets == self.n_lives
        return obs, rew, done, info

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
