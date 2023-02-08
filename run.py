#!/usr/bin/env python3
import argparse
import multiprocessing
import os
import time
from typing import List

import bsuite
from bsuite.utils import gym_wrapper
from termcolor import termcolor

from bsuite_utils.experiment_definitions import EXPERIMENTS, ExperimentConfig
from bsuite_utils.mini_sweep import SWEEP, SWEEP_SETTINGS
from bsuite_utils.experiment_definitions import ID_EXPERIMENT_MAP
from bsuite_utils.model_configs import ModelConfig

DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
_HEADER_COLOR = "blue"
_INFO_COLOR = "yellow"  # Mirror bsuite logging
_FOOTER_COLOR = "green"
_WARN_COLOR = "yellow"


def run_single(model_conf: ModelConfig, bsuite_id: str, save_path: str, overwrite: bool):
    # Execute a single run of a model against a bsuite environment
    # Note: Assumes model conforms to stable baselines interface. See bsuite_utils/custom_models.py for an example of
    # how to shim other models to this interface
    save_path = os.path.join(save_path, model_conf.name)
    final_path = os.path.join(save_path, f"bsuite_id_-_{bsuite_id.replace('/', '-')}.csv")
    if (not overwrite) and os.path.exists(final_path):
        termcolor.cprint(f"Skipping {model_conf.name} {bsuite_id}", _WARN_COLOR)
        return
    termcolor.cprint(f"Starting {model_conf.name} {bsuite_id}", "white", attrs=["bold"])

    tick = time.time()
    base_env = bsuite.load_and_record(bsuite_id=bsuite_id, save_path=save_path, overwrite=overwrite)
    env = gym_wrapper.GymFromDMEnv(base_env)
    if model_conf.env_wrapper:
        env = model_conf.env_wrapper(env)
    model = model_conf.cls(policy=model_conf.policy, env=env, **model_conf.kwargs)
    exp_conf = SWEEP_SETTINGS[bsuite_id]
    # TODO: don't need both
    model.learn(total_timesteps=exp_conf.time_steps, reset_num_timesteps=exp_conf.reset_timestep)
    tock = time.time()
    termcolor.cprint(f"Finished {model_conf.name}-{bsuite_id} in {tock - tick:.2f} seconds", _FOOTER_COLOR,
                     attrs=["bold"])


def run_parallel(experiment_ids: List[str], results_root: str, n_jobs: int, overwrite: bool):
    encountered = set()
    tasks = []  # Some experiments share the same model
    for eid in experiment_ids:
        exp_conf: ExperimentConfig = ID_EXPERIMENT_MAP[eid]
        for model_conf in exp_conf.model_configs:
            if model_conf.name in encountered:
                termcolor.cprint(f"Will ruse results of {model_conf.name} from another experiment", _WARN_COLOR)
                continue
            for bsuite_id in SWEEP:
                tasks.append((model_conf, bsuite_id, results_root, overwrite))
            encountered.add(model_conf.name)

    termcolor.cprint(f"Total: {len(tasks)} tasks", )
    with multiprocessing.Pool(n_jobs) as pool:
        pool.starmap(run_single, tasks)


def main():
    args = parse_stdin()
    termcolor.cprint(f"Running the following experiments "
                     f"with n_proc={args.jobs}, overwrite={args.overwrite}, results_dir={args.results_dir}:",
                     _HEADER_COLOR, attrs=['bold'])
    for eid in args.experiments:
        termcolor.cprint(f"\t- {eid}", _HEADER_COLOR, attrs=['bold'])
    tick = time.time()
    run_parallel(args.experiments, args.results_dir, args.jobs, args.overwrite)
    tock = time.time()
    termcolor.cprint(f"Finished {len(SWEEP)} minisweep experiments in {tock - tick:.2f} seconds", _FOOTER_COLOR,
                     attrs=['bold'])


def parse_stdin():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e',
        "--experiments",
        type=str,
        nargs="+",
        choices=ID_EXPERIMENT_MAP.keys(),
        help="Which experiment(s) to run",
        required=True,
    )
    parser.add_argument('-j', '--jobs', help="Num processes to run in parallel. -1 is maximum)", type=int,
                        default=-1)
    parser.add_argument('-f', '--overwrite', help="Overwrite previous bsuite .csv files", action='store_false',
                        default=False)
    parser.add_argument("-o", "--results_dir", help="Root path into which results from experiment runs will be written",
                        default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    if args.jobs == -1:
        args.jobs = multiprocessing.cpu_count()

    return args


if __name__ == "__main__":
    main()
