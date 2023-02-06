"""
Defines a simplified subset of bsuite for use in the blogpost experiments.
The goal of the subset is to reduce runtime while maintaining the core testing axes of the original suite.
"""

import dataclasses
from typing import List, Dict

sweep_config = dict(
    bandit={
        'ids': list(range(20)),
        'time_steps': [10_000] * 20,
        'fixed': False
    },
    bandit_noise={
        'ids': list(range(20)),
        'time_steps': [10_000] * 20,
        'fixed': False
    },
    bandit_scale={
        'ids': list(range(20)),
        'time_steps': [10_000] * 20,
        'fixed': False
    },
    mountain_car={
        'ids': [0, 1, 2],
        'time_steps': [1_000_000] * 3,
        'fixed': True
    },
    mountain_car_scale={
        'ids': [0, 4, 8, 12, 16],
        'time_steps': [1_000_000] * 5,
        'fixed': True
    },
    deep_sea={
        'ids': [0, 5, 10, 15, 20],
        'time_steps': [100_000, 200_000, 300_000, 400_000, 500_000],
        'fixed': False
    },
    deep_sea_stochastic={
        'ids': [0, 5, 10, 15, 20],
        'time_steps': [100_000, 200_000, 300_000, 400_000, 500_000],
        'fixed': False
    },
    umbrella_length={
        'ids': [0, 5, 11, 16, 22],
        'time_steps': [10_000, 60_000, 150_000, 410_000, 1_010_000],
        'fixed': False
    },
    umbrella_distract={
        'ids': [0, 5, 11, 16, 22],
        'time_steps': [200_000] * 5,
        'fixed': False
    },
    discounting_chain={
        'ids': [0, 1, 2, 3, 4],
        'time_steps': [100_000] * 5,
        'fixed': False
    },
    memory_len={
        'ids': [0, 5, 11, 16, 22],
        'time_steps': [20_000, 70_000, 160_000, 420_000, 1_020_000],
        'fixed': False
    },
    memory_size={
        'ids': [0, 4, 8, 12, 16],
        'time_steps': [30_000] * 5,
        'fixed': False
    },
)


@dataclasses.dataclass
class ExperimentSettings:
    time_steps: int  # number of time steps per episode
    reset_timestep: bool  # whether to use fixed or variable length episodes

SWEEP_SETTINGS: Dict[str, ExperimentSettings] = dict()
SWEEP: List[str] = []  # list of bsuite_ids

for key in sweep_config:
    experiment = sweep_config[key]
    for idx, id_number in enumerate(experiment['ids']):
        SWEEP.append(f"{key}/{id_number}")
        SWEEP_SETTINGS[f"{key}/{id_number}"] = ExperimentSettings(
            time_steps=experiment['time_steps'][idx],
            reset_timestep=experiment['fixed']
        )
