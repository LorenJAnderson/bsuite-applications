"""
Example 1.1 - model-domain alignment. The purpose of this example is to show that different models do excel in different
problem domains. This is shown also in the original paper.

Models included in this analysis, all from stable baselines v3: DQN, A2C, PPO
"""
import argparse
import multiprocessing
import os

from stable_baselines3 import DQN, A2C, PPO
import bsuite
from bsuite.utils import gym_wrapper
from termcolor import termcolor

import bsuite_utils.config
from bsuite_utils.mini_sweep import SWEEP, SWEEP_SETTINGS

SAVE_PATH = os.path.join(bsuite_utils.config.RESULTS_PATH, "1_1")

MODEL_LOOKUP_BY_ID = dict(
    DQN=DQN,
    A2C=A2C,
    PPO=PPO,
)


def run_single(model_name: str, bsuite_id: str, skip_existing: bool):
    save_path = os.path.join(SAVE_PATH, f"{model_name}_{bsuite_id.split('/')[0]}")
    final_path = os.path.join(save_path, )
    if skip_existing and os.path.exists(final_path):
        termcolor.cprint(f"Skipping {model_name} {bsuite_id}", "yellow")
        return
    base_env = bsuite.load_and_record(bsuite_id=bsuite_id, save_path=save_path, overwrite=True)
    env = gym_wrapper.GymFromDMEnv(base_env)
    model = MODEL_LOOKUP_BY_ID[model_name]("MlpPolicy", env)
    exp_conf = SWEEP_SETTINGS[bsuite_id]
    model.learn(total_timesteps=exp_conf.time_steps, reset_num_timesteps=exp_conf.reset_timestep)


def run_parallel(n_jobs, skip_existing):
    tasks = []
    for model_name in MODEL_LOOKUP_BY_ID.keys():
        for bsuite_id in SWEEP:
            tasks.append((model_name, bsuite_id, skip_existing))

    with multiprocessing.Pool(n_jobs) as pool:
        pool.starmap(run_single, tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobs', help="Num processes to run in parallel", type=int,
                        default=multiprocessing.cpu_count())
    parser.add_argument('-f', '--force', help="Overwrite previous outputs", action='store_false', default=True)
    args = parser.parse_args()

    run_parallel(args.jobs, not args.force)
