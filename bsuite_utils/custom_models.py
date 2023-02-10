from typing import Tuple, Dict, Any

import gym
import numpy as np
from bsuite.baselines.tf import dqn
from bsuite.utils import gym_wrapper
from collections import deque

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

class NormalizeWrapper:
    """ Normalize rewards based the last n rewards"""
    def __init__(self, env: gym_wrapper.GymFromDMEnv, maxlen=1000, method='zscore'):
        self.method = method
        self.base_env = env
        self.rewards = deque(maxlen=maxlen)

    def reset(self):
        # We don't necessarily need to clear rewards between episodes
        # self.rewards.clear()
        return self.base_env.reset()

    def step(self, action):
        obs, rew, done, info = self.base_env.step(action)
        self.rewards.append(rew)

        if self.method == 'avg':
            rew = rew / np.mean(self.rewards) + 1e-8
        elif self.method == 'zscore':
            rew = (rew - np.mean(self.rewards)) / (np.std(self.rewards) + 1e-8)
        else:
            raise ValueError("Unknown method")

        return obs, rew, done, info

    def __getattr__(self, name):
        return getattr(self.base_env, name)


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


class FrameSkipWrapper:
    def __init__(self, env, n_skip: int = 3):
        self.base_env = env
        self.n_skip = n_skip
        print(f"FrameSkipWrapper: Skipping {self.n_skip} frames")

    def reset(self):
        return self.base_env.reset()

    def step(self, action):
        obs, rew_total, done, info = self.base_env.step(action)
        for _ in range(self.n_skip):
            if done:
                break
            obs, rew, done, info = self.base_env.step(action)
            rew_total += rew
        return obs, rew_total, done, info

    def __getattr__(self, name):
        return getattr(self.base_env, name)


class FrameStackWrapper(gym.Env):
    """
    Stack Observations from the past n steps
    """
    def __init__(self, env: gym_wrapper.GymFromDMEnv, n: int = 3):
        self.base_env = env
        self.action_space = env.action_space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=env.observation_space.shape + (n,),
            dtype=env.observation_space.dtype
        )
        self.observation_history = deque(maxlen=n)

    def reset(self):
        self.observation_history.clear()
        obs = self.base_env.reset()
        for _ in range(self.observation_history.maxlen):
            self.observation_history.append(obs)
        return np.stack(self.observation_history, axis=-1)

    def step(self, action):
        obs, rew, done, info = self.base_env.step(action)
        self.observation_history.append(obs)
        return np.stack(self.observation_history, axis=-1), rew, done, info

    def render(self, mode="human"):
        pass

