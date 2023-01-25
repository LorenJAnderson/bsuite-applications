import numpy as np

import bsuite
from bsuite.utils import gym_wrapper


from custom_envs.deep_sea_test.loader import load_deep_sea


def sweep_list():
    return {0: {'env_name': 'bandit',
                'env_type': 'bsuite',
                'ids': [0, 1, 2, 3],
                'time_steps': 10_000
                },
            1: {'env_name': 'deep_sea',
                'env_type': 'own',
                'ids': [0, 1, 2, 3],
                'time_steps': 10_000
                }
            }


def load_bsuite_env(env_name, id):
    env_name = env_name + str(id)
    env = gym_wrapper.GymFromDMEnv(bsuite.load_from_id(env_name))
    return env, env_name


def load_own_env(env_name, id):
    if env_name == 'deep_sea':
        return load_deep_sea(id), env_name + '/' + str(id)
    else:
        pass
