"""
Example 1.2 - comparing implementations. Similar to 1.1, but comparing implementations of the same algorithm from
different libraries.

For this example we compare DQN implemented in baselines vs DQN implemented in RLLib
Since we already ran DQN for example 1_1 we can skip it here.
"""

import argparse
import multiprocessing
import os
import time
from typing import List

import bsuite
from bsuite.utils import gym_wrapper
from termcolor import termcolor

from bsuite_utils.mini_sweep import SWEEP, SWEEP_SETTINGS
from bsuite_utils.model_config import ModelConfig
import bsuite_utils.config


def run_single(model_conf: ModelConfig, bsuite_id: str, save_path: str, overwrite: bool):
    # Note: Assumes model is a stable baselines model
    save_path = os.path.join(save_path, model_conf.name)
    final_path = os.path.join(save_path, f"bsuite_id_-_{bsuite_id.replace('/', '-')}.csv")
    if (not overwrite) and os.path.exists(final_path):
        termcolor.cprint(f"Skipping {model_conf.name} {bsuite_id}", "yellow")
        return
    termcolor.cprint(f"Starting {model_conf.name} {bsuite_id}", "green")
    tick = time.time()
    base_env = bsuite.load_and_record(bsuite_id=bsuite_id, save_path=save_path, overwrite=overwrite)
    env = gym_wrapper.GymFromDMEnv(base_env)
    model = model_conf.cls(policy=model_conf.policy, env=env, **model_conf.kwargs)
    exp_conf = SWEEP_SETTINGS[bsuite_id]
    model.learn(total_timesteps=exp_conf.time_steps, reset_num_timesteps=exp_conf.reset_timestep)
    tock = time.time()
    termcolor.cprint(f"Finished {model_conf.name}-{bsuite_id} in {tock - tick:.2f} seconds", "white", "on_green")


def run_parallel(model_configs: List[ModelConfig], save_path: str, n_jobs: int, overwrite: bool):
    tasks = []
    for model_conf in model_configs:
        for bsuite_id in SWEEP:
            tasks.append((model_conf, bsuite_id, save_path, overwrite))

    with multiprocessing.Pool(n_jobs) as pool:
        pool.starmap(run_single, tasks)


def parse_stdin():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobs', help="Num processes to run in parallel", type=int,
                        default=-1)
    parser.add_argument('-f', '--overwrite', help="Overwrite previous outputs", action='store_false', default=False)
    return parser.parse_args()


def main(model_configs: List[ModelConfig], experiment_id: str):
    args = parse_stdin()
    save_path = os.path.join(bsuite_utils.config.RESULTS_PATH, experiment_id)

    if args.jobs == -1:
        args.jobs = multiprocessing.cpu_count()

    termcolor.cprint(f"Running with n_proc={args.jobs}, overwrite={args.overwrite}", "green")

    tick = time.time()
    run_parallel(model_configs, save_path, args.jobs, args.overwrite)
    tock = time.time()
    termcolor.cprint(f"Finished {len(SWEEP)} minisweep experiments in {tock - tick:.2f} seconds", "green")
