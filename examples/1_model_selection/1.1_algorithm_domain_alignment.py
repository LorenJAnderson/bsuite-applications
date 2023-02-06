"""
Example 1.1 - model-domain alignment. The purpose of this example is to show that different models do excel in different
problem domains. This is shown also in the original paper.

Models included in this analysis, all from stable baselines v3: DQN, A2C, PPO
"""
import argparse
import multiprocessing
import os
import time

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


def run_single(model_name: str, bsuite_id: str, overwrite: bool):
    save_path = os.path.join(SAVE_PATH, f"{model_name}_{bsuite_id.split('/')[0]}")
    final_path = os.path.join(save_path, f"bsuite_id_-_{bsuite_id.replace('/', '-')}.csv")
    if (not overwrite) and os.path.exists(final_path):
        termcolor.cprint(f"Skipping {model_name} {bsuite_id}", "yellow")
        return
    termcolor.cprint(f"Starting {model_name} {bsuite_id}", "green")
    tick = time.time()
    base_env = bsuite.load_and_record(bsuite_id=bsuite_id, save_path=save_path, overwrite=overwrite)
    env = gym_wrapper.GymFromDMEnv(base_env)
    model = MODEL_LOOKUP_BY_ID[model_name]("MlpPolicy", env)
    exp_conf = SWEEP_SETTINGS[bsuite_id]
    model.learn(total_timesteps=exp_conf.time_steps, reset_num_timesteps=exp_conf.reset_timestep)
    tock = time.time()
    termcolor.cprint(f"Finished {model_name}-{bsuite_id} in {tock - tick:.2f} seconds", "white", "on_green")


def run_parallel(n_jobs, overwrite):
    tasks = []
    for model_name in MODEL_LOOKUP_BY_ID.keys():
        for bsuite_id in SWEEP:
            tasks.append((model_name, bsuite_id, overwrite))

    with multiprocessing.Pool(n_jobs) as pool:
        pool.starmap(run_single, tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobs', help="Num processes to run in parallel", type=int,
                        default=-1)
    parser.add_argument('-f', '--force', help="Overwrite previous outputs", action='store_false', default=False)
    args = parser.parse_args()

    if args.jobs == -1:
        args.jobs = multiprocessing.cpu_count()

    termcolor.cprint(f"Running with n_proc={args.jobs}, force={args.force}", "green")

    tick = time.time()
    run_parallel(args.jobs, args.force)
    tock = time.time()
    termcolor.cprint(f"Finished {len(SWEEP)} minisweep experiments in {tock - tick:.2f} seconds", "green")
