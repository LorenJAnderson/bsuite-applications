import argparse
import multiprocessing
import os
from typing import List

import pandas as pd
from bsuite.experiments import summary_analysis
from bsuite.experiments.bandit import analysis as bandit_analysis
from bsuite.experiments.bandit_noise import analysis as bandit_noise_analysis
from bsuite.experiments.bandit_scale import analysis as bandit_scale_analysis
from bsuite.experiments.cartpole import analysis as cartpole_analysis
from bsuite.experiments.cartpole_noise import analysis as cartpole_noise_analysis
from bsuite.experiments.cartpole_scale import analysis as cartpole_scale_analysis
from bsuite.experiments.cartpole_swingup import analysis as cartpole_swingup_analysis
from bsuite.experiments.catch import analysis as catch_analysis
from bsuite.experiments.catch_noise import analysis as catch_noise_analysis
from bsuite.experiments.catch_scale import analysis as catch_scale_analysis
from bsuite.experiments.deep_sea import analysis as deep_sea_analysis
from bsuite.experiments.deep_sea_stochastic import analysis as deep_sea_stochastic_analysis
from bsuite.experiments.discounting_chain import analysis as discounting_chain_analysis
from bsuite.experiments.memory_len import analysis as memory_len_analysis
from bsuite.experiments.memory_size import analysis as memory_size_analysis
from bsuite.experiments.mnist import analysis as mnist_analysis
from bsuite.experiments.mnist_noise import analysis as mnist_noise_analysis
from bsuite.experiments.mnist_scale import analysis as mnist_scale_analysis
from bsuite.experiments.mountain_car import analysis as mountain_car_analysis
from bsuite.experiments.mountain_car_noise import analysis as mountain_car_noise_analysis
from bsuite.experiments.mountain_car_scale import analysis as mountain_car_scale_analysis
from bsuite.experiments.umbrella_distract import analysis as umbrella_distract_analysis
from bsuite.experiments.umbrella_length import analysis as umbrella_length_analysis
from bsuite.logging import csv_load
from termcolor import termcolor

from bsuite_utils.experiment_definitions import ID_EXPERIMENT_MAP
from bsuite_utils.model_configs import ModelConfig

ENV_ANALYSIS_MAP = {
    "bandit": bandit_analysis,
    "bandit_noise": bandit_noise_analysis,
    "bandit_scale": bandit_scale_analysis,
    "cartpole": cartpole_analysis,
    "cartpole_noise": cartpole_noise_analysis,
    "cartpole_scale": cartpole_scale_analysis,
    "cartpole_swingup": cartpole_swingup_analysis,
    "catch": catch_analysis,
    "catch_noise": catch_noise_analysis,
    "catch_scale": catch_scale_analysis,
    "deep_sea": deep_sea_analysis,
    "deep_sea_stochastic": deep_sea_stochastic_analysis,
    "discounting_chain": discounting_chain_analysis,
    "memory_len": memory_len_analysis,
    "memory_size": memory_size_analysis,
    "mnist": mnist_analysis,
    "mnist_noise": mnist_noise_analysis,
    "mnist_scale": mnist_scale_analysis,
    "mountain_car": mountain_car_analysis,
    "mountain_car_noise": mountain_car_noise_analysis,
    "mountain_car_scale": mountain_car_scale_analysis,
    "umbrella_distract": umbrella_distract_analysis,
    "umbrella_length": umbrella_length_analysis,
}

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")
REPORTS_ROOT = os.path.join(os.path.dirname(__file__), "reports")
if not os.path.exists(REPORTS_ROOT):
    os.mkdir(REPORTS_ROOT)


def get_results_path(model_config: ModelConfig):
    return os.path.join(RESULTS_ROOT, model_config.name)


def prettify_agent_name(agent_name: str):
    agent_name = os.path.basename(agent_name)
    agent_name = agent_name.replace("_", "-")
    words = agent_name.split("-")
    words = [word[0].upper() + word[1:] for word in words]
    agent_name = " ".join(words)
    return agent_name


def single(experiment_id, output_dir):
    # Union dfs from the different model directories
    experiment = ID_EXPERIMENT_MAP[experiment_id]
    save_path = os.path.join(output_dir, experiment_id)
    os.makedirs(save_path, exist_ok=True)

    dfs = []
    for model_config in experiment.model_configs:
        model_path = get_results_path(model_config)
        print(model_path)
        if not os.path.exists(model_path):
            termcolor.cprint(f"Missing results for {model_config.name}", "red")
            return
        df, _ = csv_load.load_bsuite(model_path)
        dfs.append(df)

    # Concatenate all the dfs
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df['agent_name'] = df['agent_name'].apply(lambda x: prettify_agent_name(x))
    sweep_vars = ['agent_name']
    bsuite_score = summary_analysis.bsuite_score(df, sweep_vars)
    bsuite_summary_by_tag = summary_analysis.ave_score_by_tag(bsuite_score, sweep_vars)
    radar_plot = summary_analysis.bsuite_radar_plot(bsuite_summary_by_tag, sweep_vars)
    radar_plot.savefig(os.path.join(output_dir, experiment_id, "radar.png"), bbox_inches='tight')

    bar_chart = summary_analysis.bsuite_bar_plot(bsuite_score, sweep_vars)
    bar_chart.save(filename=os.path.join(output_dir, experiment_id, "bar_chart.png"))

    bar_chart = summary_analysis.bsuite_bar_plot_compare(bsuite_score, sweep_vars)
    bar_chart.save(filename=os.path.join(output_dir, experiment_id, "bar_chart_compare.png"))

    # Environment-specific analysis. Get specific env names from df.bsuite_env column and then
    # call the appropriate analysis function.
    for env_name in df.bsuite_env.unique():
        print(f"Processing {env_name}...")
        env_img_dir = os.path.join(output_dir, experiment_id, env_name)
        if not os.path.exists(env_img_dir):
            os.makedirs(env_img_dir, exist_ok=True)
        single_problem_score = summary_analysis.plot_single_experiment(bsuite_score, env_name, sweep_vars)
        single_problem_score.save(filename=os.path.join(env_img_dir, "single.png"))

        ## obtain correct module for analysis
        # env_analysis = ENV_ANALYSIS_MAP[env_name]
        # sub_df = df[df.bsuite_env == env_name].copy()
        # try:
        #     x = env_analysis.plot_learning(sub_df, sweep_vars)
        #     x.save(filename=os.path.join(env_img_dir, "learning.png"))
        # except Exception as e:
        #     print(e)
        #     print("Failed to plot learning curve")
        # env_analysis.plot_seeds(sub_df, sweep_vars)
        # x.save(filename=os.path.join(env_img_dir, "seeds.png"))
        # env_analysis.plot_average(sub_df, sweep_vars)
        # x.save(filename=os.path.join(env_img_dir, "average.png"))
        # env_analysis.plot_scale(sub_df, sweep_vars)
        # x.save(filename=os.path.join(env_img_dir, "scale.png"))


def analyze_experiments(experiment_ids: List[str], reports_dir: str):
    tasks = []
    for experiment_id in experiment_ids:
        tasks.append((experiment_id, reports_dir))
    with multiprocessing.Pool(min(len(experiment_ids), multiprocessing.cpu_count())) as pool:
        pool.starmap(single, tasks)


def main():
    # Assumes that each folder inside the experiment corresponds to one run of the experiment. These should be
    # composited into a single radar chart. Additionally, we may generate problem-specific analysis as pertinent
    # to our mini-suite. Save all images in a folder named after the experiment. Sub files corresponding to a specific
    # run within the experiment will be stored in this folder as well and prefixed with the tag for that run, e.g.
    args = parse_stdin()
    termcolor.cprint(f"Generating report files for experiments={args.experiments}")
    analyze_experiments(args.experiments, args.reports_dir)


def parse_stdin():
    # Example usage of this script:
    # (do all) python analysis.py
    # (do specific) python analysis.py --experiments 1.1 1.2 1.3
    parser = argparse.ArgumentParser()
    # default is to run all
    parser.add_argument(
        "-e",
        "--experiments",
        type=str,
        nargs="+",
        choices=ID_EXPERIMENT_MAP.keys(),
        help="Which experiment(s) to generate reports for",
        default=ID_EXPERIMENT_MAP.keys(),
    )
    parser.add_argument(
        "-o", "--reports_dir", help="Path to write experiment reports to", default=REPORTS_ROOT
    )
    parser.add_argument("-i", "--results_dir", help="Path to read experiment results from", default=RESULTS_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
